# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import DType, APInt, i32


def make_packed_weight_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Kt: int, Nt: int, P=2, G=18, depth=2, ii=1
):
    """Build + schedule a DSP-packed weight-stationary GEMM; return ``(top, top_s)``.

    Low-bitwidth integer only. A packed PE processes ``P`` adjacent output columns
    that share the streamed activation, computing all ``P`` products in ONE wide
    multiply (DSP packing). ``G`` is the bit gap between packed products (must
    exceed the product width); the caller picks ``P``/``G`` from ``Tin``. Requires
    ``Nt % P == 0``. ``depth`` is the inter-process FIFO depth."""
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert Tin.is_int() and Tacc.is_int(), "DSP packing is integer-only"
    assert N % Nt == 0 and K % Kt == 0 and Nt % P == 0, "tiling must be even"
    NT, KT, NG = N // Nt, K // Kt, Nt // P
    PW = (P - 1) * G + Tin.primitive_width  # packed-operand width
    Wt = APInt(PW + 1, signed=True)  # packed weight (signed)
    Pt = APInt(PW + Tin.primitive_width + 2, signed=True)  # product width

    @kernel
    def load_W(B: Tin[K, N], fifo_W: Stream[Wt, depth][Kt, NG]):
        # pack P weights per PE: packed = w0 + w1<<G (+ w2<<2G + w3<<3G). Each wj is
        # widened to Wt FIRST, else `wj<<(j*G)` truncates in the narrow operand type.
        for nt in range(NT):
            for kt in range(KT, name="kt"):
                for kk in range(Kt, name="kk"):
                    for ng in range(NG, name="ng"):
                        base: i32 = nt * Nt + ng * P
                        w0: Wt = B[kt * Kt + kk, base]
                        w1: Wt = B[kt * Kt + kk, base + 1]
                        packed: Wt = w0 + (w1 << G)
                        if P >= 4:
                            w2: Wt = B[kt * Kt + kk, base + 2]
                            w3: Wt = B[kt * Kt + kk, base + 3]
                            packed = packed + (w2 << (2 * G)) + (w3 << (3 * G))
                        fifo_W[kk, ng].put(packed)

    @kernel
    def load_A(A: Tin[M, K], fifo_A: Stream[Tin, depth][Kt, NG]):
        for nt in range(NT):
            for kt in range(KT):
                for m in range(M, name="m"):
                    for kk in range(Kt, name="kk"):
                        fifo_A[kk, 0].put(A[m, kt * Kt + kk])

    @kernel(mapping=[Kt, NG])
    def pe(
        fifo_W: Stream[Wt, depth][Kt, NG],
        fifo_A: Stream[Tin, depth][Kt, NG],
        fifo_P: Stream[Tacc, depth][Kt, Nt],
        fifo_O: Stream[Tacc, depth][Kt, Nt],
    ):
        kk = allo.get_wid(0)  # array row = contraction
        ng = allo.get_wid(1)  # array col-group (P columns each)
        c0 = ng * P
        for nt in range(NT):
            for kt in range(KT):
                wpk: Wt = fifo_W[kk, ng].get()  # resident packed weight
                for m in range(M, name="m"):
                    a: Tin = fifo_A[kk, ng].get()
                    prod: Pt = a * wpk  # one multiply -> P products in bit fields
                    if ng < NG - 1:
                        fifo_A[kk, ng + 1].put(a)  # forward activation east
                    # signed borrow-chain unpack: p_j = signext_G(window_j) + bit_{j-1}
                    b0: Tacc = prod[G - 1 : G]
                    p0: Tacc = prod[0:G] - (b0 << G)
                    b1: Tacc = prod[2 * G - 1 : 2 * G]
                    p1: Tacc = prod[G : 2 * G] - (b1 << G) + b0
                    if kk > 0:  # add partial sums from the north neighbor
                        p0 = p0 + fifo_P[kk, c0].get()
                        p1 = p1 + fifo_P[kk, c0 + 1].get()
                    if kk < Kt - 1:  # push partial sums south
                        fifo_P[kk + 1, c0].put(p0)
                        fifo_P[kk + 1, c0 + 1].put(p1)
                    else:  # bottom row emits
                        fifo_O[kk, c0].put(p0)
                        fifo_O[kk, c0 + 1].put(p1)
                    if P >= 4:
                        b2: Tacc = prod[3 * G - 1 : 3 * G]
                        p2: Tacc = prod[2 * G : 3 * G] - (b2 << G) + b1
                        b3: Tacc = prod[4 * G - 1 : 4 * G]
                        p3: Tacc = prod[3 * G : 4 * G] - (b3 << G) + b2
                        if kk > 0:
                            p2 = p2 + fifo_P[kk, c0 + 2].get()
                            p3 = p3 + fifo_P[kk, c0 + 3].get()
                        if kk < Kt - 1:
                            fifo_P[kk + 1, c0 + 2].put(p2)
                            fifo_P[kk + 1, c0 + 3].put(p3)
                        else:
                            fifo_O[kk, c0 + 2].put(p2)
                            fifo_O[kk, c0 + 3].put(p3)

    @kernel
    def reduce_C(fifo_O: Stream[Tacc, depth][Kt, Nt], fifo_Ct: Stream[Tacc, depth][Nt]):
        accC: Tacc[M, Nt]
        for nt in range(NT):
            for kt in range(KT):
                for m in range(M, name="m"):
                    for nn in range(Nt, name="nn"):
                        p: Tacc = fifo_O[Kt - 1, nn].get()
                        acc_val: Tacc = p
                        if kt > 0:
                            acc_val = accC[m, nn] + p
                        if kt == KT - 1:
                            fifo_Ct[nn].put(acc_val)
                        else:
                            accC[m, nn] = acc_val

    @kernel
    def write_C(C: Tout[M, N], fifo_Ct: Stream[Tacc, depth][Nt]):
        for nt in range(NT):
            for m in range(M, name="m"):
                for nn in range(Nt, name="nn"):
                    C[m, nt * Nt + nn] = fifo_Ct[nn].get()

    @kernel
    def top(A: Tin[M, K], B: Tin[K, N], C: Tout[M, N]):
        """Weight-stationary systolic GEMM with **low-bitwidth DSP packing**.

        Computes ``C[M,N] = A[M,K] @ B[K,N]`` (A standard layout). Same WS dataflow
        as :mod:`ws_direct` (weights resident, activations stream west->east,
        partial sums reduce spatially down columns, overlapped reduce/write), but a
        PE handles ``P`` adjacent output columns that share the streamed activation:
        ``load_W`` packs their ``P`` weights into one operand, the PE does ONE wide
        multiply, and a signed borrow-chain unpacks the ``P`` products into ``P``
        separate psum chains. The array is ``Kt x (Nt/P)``.

        WHY (DSP packing): one DSP48 multiply yields ``P`` int products instead of
        one, so the multiply DSP count drops ~``P x``. Measured on u280/2023.2,
        128^3/16x16 vs the unpacked WS: **int8 P=2 -> 240->128 DSP (2x)**, latency
        unchanged, LUT lower. For **int4**, Vitis maps the narrow packed multiply to
        LUTs -> **0 DSP** (so int4 GEMM frees all DSPs); the pack factor then only
        nudges LUT/FF, hence ``P=2`` is the sensible default for int4 too.

        Integer only. ``G`` (product bit gap) must exceed the product width
        (``2*Tin_bits + 2`` works: i8->18, i4->10). ``Nt % P == 0`` required.
        Float / int16: packing gives no DSP saving (products too wide to share a
        27-bit DSP port) -- use :mod:`ws_direct` instead.
        """
        fifo_W: Stream[Wt, depth][Kt, NG]
        fifo_A: Stream[Tin, depth][Kt, NG]
        fifo_P: Stream[Tacc, depth][Kt, Nt]
        fifo_O: Stream[Tacc, depth][Kt, Nt]
        fifo_Ct: Stream[Tacc, depth][Nt]
        load_W(B, fifo_W)
        load_A(A, fifo_A)
        pe(fifo_W, fifo_A, fifo_P, fifo_O)
        reduce_C(fifo_O, fifo_Ct)
        write_C(C, fifo_Ct)

    pe_s = pe.schedule()
    pe_s.pipeline("m", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("ng")
    lw_s.unroll("kk")
    lw_s.pipeline("kt", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("kk")
    la_s.pipeline("m", ii=ii)

    rc_s = reduce_C.schedule()
    rc_s.partition(rc_s.buffer("accC"), dim=2, kind=rc_s.Complete)
    rc_s.unroll("nn")
    rc_s.pipeline("m", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("nn")
    wc_s.pipeline("m", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, rc_s, wc_s)

    return top, top_s
