# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import APInt, i32


def make_packed_output_stationary_conv2d(
    Tin,
    Tacc,
    Tout,
    Co: int,
    Ci: int,
    IH: int,
    IW: int,
    KH: int,
    KW: int,
    Mt: int,
    Nt: int,
    P=2,
    G=18,
    stride: int = 1,
    pad: int = 0,
    depth=2,
    ii=1,
):
    """Factory for an output-stationary systolic **conv2d** (NHWC) with
    **low-bitwidth DSP packing**: the on-chip-buffered input loader packs ``P``
    adjacent output pixels' activations into one operand, the PE does one wide
    multiply, and a signed borrow-chain unpacks the ``P`` products. See
    :func:`conv` for the full docstring.
    """
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    M, N, K = Co, OH * OW, KH * KW * Ci
    MT, NT, NG = M // Mt, N // Nt, Nt // P
    NTW = OW // Nt
    IHP, IWP = IH + 2 * pad, IW + 2 * pad  # zero-padded on-chip extent
    BANK = Nt * stride
    FILLU = math.gcd(Ci, 8)  # fill ci-lanes/cycle: hide the one-time bufX prologue
    PW = (P - 1) * G + Tin.primitive_width
    Bt = APInt(PW + 1, signed=True)  # packed activation operand (signed)
    Pt = APInt(PW + Tin.primitive_width + 2, signed=True)  # product width

    @kernel
    def load_W(Wt: Tin[K, Co], fifo_A: Stream[Tin, depth][Mt, NG]):
        for mo in range(MT):
            for no in range(NT, name="no"):
                for k in range(K, name="k"):
                    for r in range(Mt, name="r"):  # lane -> output channel
                        fifo_A[r, 0].put(Wt[k, mo * Mt + r])

    @kernel
    def load_X(X: Tin[IH, IW, Ci], fifo_B: Stream[Bt, depth][Mt, NG]):
        # Stage X into a zero-padded on-chip buffer (padding materialized here at
        # the Tin level), then pack the P pixels of each group with no per-read
        # bounds logic -- the gather reads only the wide Bt operand, so there is
        # no ambiguous ap_int ternary.
        bufX: Tin[IHP, IWP, Ci]
        for ih in range(IHP, name="fih"):
            for iw in range(IWP, name="fiw"):
                for cb in range(Ci // FILLU, name="fcb"):
                    for cl in range(FILLU, name="fcl"):
                        ci2: i32 = cb * FILLU + cl
                        xi: i32 = ih - pad
                        xj: i32 = iw - pad
                        inb = xi >= 0 and xi < IH and xj >= 0 and xj < IW
                        sxi: i32 = xi if xi >= 0 and xi < IH else 0
                        sxj: i32 = xj if xj >= 0 and xj < IW else 0
                        v: Tin = X[sxi, sxj, ci2]
                        bufX[ih, iw, ci2] = v if inb else 0
        for mo in range(MT, name="mo"):
            for no in range(NT):
                oh: i32 = no // NTW
                ow0: i32 = (no % NTW) * Nt
                for k in range(K, name="k"):
                    ci: i32 = k % Ci
                    kw: i32 = (k // Ci) % KW
                    kh: i32 = k // (Ci * KW)
                    ih: i32 = oh * stride + kh  # indices into the padded buffer
                    for cg in range(NG, name="cg"):  # lane -> pixel-group (P pixels)
                        c0: i32 = cg * P
                        b0: Bt = bufX[ih, (ow0 + c0) * stride + kw, ci]
                        b1: Bt = bufX[ih, (ow0 + c0 + 1) * stride + kw, ci]
                        fifo_B[0, cg].put(b0 + (b1 << G))

    @kernel(mapping=[Mt, NG])
    def pe(
        fifo_A: Stream[Tin, depth][Mt, NG],
        fifo_B: Stream[Bt, depth][Mt, NG],
        fifo_C: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        cg = allo.get_wid(1)
        c0 = cg * P
        for mo in range(MT):
            for no in range(NT):
                acc0: Tacc = 0
                acc1: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_A[r, cg].get()
                    bpk: Bt = fifo_B[r, cg].get()
                    prod: Pt = a * bpk  # one multiply -> P products
                    if cg < NG - 1:
                        fifo_A[r, cg + 1].put(a)  # weight east
                    if r < Mt - 1:
                        fifo_B[r + 1, cg].put(bpk)  # packed activation south
                    b0: Tacc = prod[G - 1 : G]
                    acc0 = acc0 + prod[0:G] - (b0 << G)
                    b1: Tacc = prod[2 * G - 1 : 2 * G]
                    acc1 = acc1 + prod[G : 2 * G] - (b1 << G) + b0
                fifo_C[r, c0].put(acc0)
                fifo_C[r, c0 + 1].put(acc1)

    @kernel
    def store_Y(Y: Tout[OH, OW, Co], fifo_C: Stream[Tacc, depth][Mt, Nt]):
        for mo in range(MT):
            for no in range(NT, name="no"):
                oh: i32 = no // NTW
                ow0: i32 = (no % NTW) * Nt
                for r in range(Mt, name="r"):
                    for c in range(Nt, name="c"):
                        Y[oh, ow0 + c, mo * Mt + r] = fifo_C[r, c].get()

    @kernel
    def conv(Wt: Tin[K, Co], X: Tin[IH, IW, Ci], Y: Tout[OH, OW, Co]):
        """Output-stationary systolic **conv2d** (NHWC) with **low-bitwidth DSP
        packing**, on-chip-buffered input.

        Same OS dataflow / on-chip ``bufX`` staging as :mod:`buffered`, but a PE
        owns ``P`` adjacent output **pixels** (columns) that share the eastward
        weight: ``load_X`` packs their ``P`` activations into one ``Bt`` operand,
        the PE issues ONE wide multiply, and a signed borrow-chain splits the
        ``P`` products into ``P`` per-pixel accumulators. The array is
        ``Mt x (Nt/P)``.

        WHY: one DSP48 multiply yields ``P`` int products instead of one ->
        ~``P x`` fewer multiply DSPs (measured on GEMM: int8 ``P=2`` halves DSP,
        int4 -> 0 DSP). Integer only; ``G`` must exceed the product width
        (i8->18, i4->10), ``Nt % P == 0`` and ``Nt | OW``. For float / int16 use
        :mod:`buffered` (products too wide to share a DSP port).
        """
        fifo_A: Stream[Tin, depth][Mt, NG]
        fifo_B: Stream[Bt, depth][Mt, NG]
        fifo_C: Stream[Tacc, depth][Mt, Nt]
        load_W(Wt, fifo_A)
        load_X(X, fifo_B)
        pe(fifo_A, fifo_B, fifo_C)
        store_Y(Y, fifo_C)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("r")
    lw_s.pipeline("k", ii=ii)

    lx_s = load_X.schedule()
    lx_s.partition(
        lx_s.buffer("bufX"), dim=2, kind=lx_s.Cyclic, factor=BANK
    )  # iw (gather)
    if FILLU > 1:
        lx_s.partition(
            lx_s.buffer("bufX"), dim=3, kind=lx_s.Cyclic, factor=FILLU
        )  # ci (fill)
    lx_s.unroll("fcl")
    lx_s.unroll("cg")
    lx_s.pipeline("k", ii=ii)
    lx_s.pipeline(lx_s.flatten(("fih", "fiw", "fcb")), ii=ii)

    st_s = store_Y.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = conv.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, lx_s, st_s)

    return conv, top_s
