# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import APInt, i32


def make_packed_output_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Mt: int, Nt: int, P=2, G=18, depth=2, ii=1
):
    MT, NT, NG = M // Mt, N // Nt, Nt // P
    PW = (P - 1) * G + Tin.primitive_width
    Bt = APInt(PW + 1, signed=True)  # packed B operand (signed)
    Pt = APInt(PW + Tin.primitive_width + 2, signed=True)  # product width

    @kernel
    def load_A(At: Tin[K, M], fifo_A: Stream[Tin, depth][Mt, NG]):
        for mo in range(MT):
            for no in range(NT, name="no"):
                for k in range(K, name="k"):
                    for r in range(Mt, name="r"):  # lane (west edge)
                        fifo_A[r, 0].put(At[k, mo * Mt + r])

    @kernel
    def load_B(B: Tin[K, N], fifo_B: Stream[Bt, depth][Mt, NG]):
        # pack the P columns of each group into one operand (each widened to Bt).
        for mo in range(MT, name="mo"):
            for no in range(NT):
                for k in range(K, name="k"):
                    for cg in range(NG, name="cg"):  # lane (north edge)
                        base: i32 = no * Nt + cg * P
                        b0: Bt = B[k, base]
                        b1: Bt = B[k, base + 1]
                        packed: Bt = b0 + (b1 << G)
                        if P >= 4:
                            b2: Bt = B[k, base + 2]
                            b3: Bt = B[k, base + 3]
                            packed = packed + (b2 << (2 * G)) + (b3 << (3 * G))
                        fifo_B[0, cg].put(packed)

    @kernel(mapping=[Mt, NG])
    def pe(
        fifo_A: Stream[Tin, depth][Mt, NG],
        fifo_B: Stream[Bt, depth][Mt, NG],
        fifo_C: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        cg = allo.get_wid(1)  # col-group (P columns)
        c0 = cg * P
        for mo in range(MT):
            for no in range(NT):
                acc0: Tacc = 0
                acc1: Tacc = 0
                acc2: Tacc = 0
                acc3: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_A[r, cg].get()
                    bpk: Bt = fifo_B[r, cg].get()
                    prod: Pt = a * bpk  # one multiply -> P products
                    if cg < NG - 1:
                        fifo_A[r, cg + 1].put(a)  # activation east
                    if r < Mt - 1:
                        fifo_B[r + 1, cg].put(bpk)  # packed weights south
                    # signed borrow-chain unpack
                    b0: Tacc = prod[G - 1 : G]
                    acc0 = acc0 + prod[0:G] - (b0 << G)
                    b1: Tacc = prod[2 * G - 1 : 2 * G]
                    acc1 = acc1 + prod[G : 2 * G] - (b1 << G) + b0
                    if P >= 4:
                        b2: Tacc = prod[3 * G - 1 : 3 * G]
                        acc2 = acc2 + prod[2 * G : 3 * G] - (b2 << G) + b1
                        b3: Tacc = prod[4 * G - 1 : 4 * G]
                        acc3 = acc3 + prod[3 * G : 4 * G] - (b3 << G) + b2
                fifo_C[r, c0].put(acc0)
                fifo_C[r, c0 + 1].put(acc1)
                if P >= 4:
                    fifo_C[r, c0 + 2].put(acc2)
                    fifo_C[r, c0 + 3].put(acc3)

    @kernel
    def store_C(C: Tout[M, N], fifo_C: Stream[Tacc, depth][Mt, Nt]):
        for mo in range(MT):
            for no in range(NT, name="no"):
                for r in range(Mt, name="r"):
                    for c in range(Nt, name="c"):
                        C[mo * Mt + r, no * Nt + c] = fifo_C[r, c].get()

    @kernel
    def top(At: Tin[K, M], B: Tin[K, N], C: Tout[M, N]):
        """Output-stationary systolic GEMM with **low-bitwidth DSP packing**.

        Computes ``C[M,N] = AT^T @ B`` (``AT`` pre-transposed ``[K,M]``). Same OS
        dataflow as :mod:`os_direct` (C accumulated in the PE over K, A east, B
        south, no drains), but a PE owns ``P`` adjacent output columns sharing the
        eastward activation: ``load_B`` packs their ``P`` ``B`` values, the PE does
        ONE wide multiply, and a signed borrow-chain unpacks the ``P`` products into
        ``P`` per-column accumulators. The array is ``Mt x (Nt/P)``.

        WHY (DSP packing): one DSP48 multiply produces ``P`` int products instead of
        one -> ~``P x`` fewer multiply DSPs. Measured (u280/2023.2, 128^3): **int8
        P=2 halves DSP**, latency-neutral; **int4 -> 0 DSP** (Vitis uses LUTs for the
        narrow packed multiply), so ``P=2`` suffices for int4 too. Integer only;
        ``G`` must exceed the product width (i8->18, i4->10), ``Nt % P == 0``. For
        float / int16 use :mod:`os_direct` (products too wide to share a DSP port).
        """
        fifo_A: Stream[Tin, depth][Mt, NG]
        fifo_B: Stream[Bt, depth][Mt, NG]
        fifo_C: Stream[Tacc, depth][Mt, Nt]
        load_A(At, fifo_A)
        load_B(B, fifo_B)
        pe(fifo_A, fifo_B, fifo_C)
        store_C(C, fifo_C)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("r")
    la_s.pipeline("k", ii=ii)

    lb_s = load_B.schedule()
    lb_s.unroll("cg")
    lb_s.pipeline("k", ii=ii)

    st_s = store_C.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, la_s, lb_s, st_s)

    return top, top_s
