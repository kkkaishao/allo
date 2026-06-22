# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import i32


def make_direct_output_stationary_depthwise(
    Tin,
    Tacc,
    Tout,
    C: int,
    IH: int,
    IW: int,
    KH: int,
    KW: int,
    Ct: int,
    Pt: int,
    stride: int = 1,
    pad: int = 0,
    depth=2,
    ii=1,
):
    """Factory for an output-stationary systolic **depthwise** conv2d (NHWC).

    See :func:`dwconv` for the algorithm/dataflow docstring.
    """
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    N, K = OH * OW, KH * KW  # pixels; per-channel reduction window (no Ci!)
    CT, NT = C // Ct, N // Pt  # channel-tile / pixel-tile counts

    @kernel
    def load_W(W: Tin[KH, KW, C], fifo_W: Stream[Tin, depth][Ct, Pt]):
        # per-channel filter -> west edge, flows east (shared by all pixels in row).
        for ct in range(CT):
            for no in range(NT, name="no"):  # re-read W for each pixel-tile
                for k in range(K, name="k"):
                    kh: i32 = k // KW
                    kw: i32 = k % KW
                    for r in range(Ct, name="r"):  # lane -> channel
                        fifo_W[r, 0].put(W[kh, kw, ct * Ct + r])

    @kernel
    def load_X(X: Tin[IH, IW, C], fifo_X: Stream[Tin, depth][Ct, Pt]):
        # depthwise input has no cross-channel sharing, so every PE is fed its own
        # window directly (both lanes unrolled); im2col on the fly per channel.
        for ct in range(CT, name="ct"):
            for no in range(NT):
                for k in range(K, name="k"):
                    kh: i32 = k // KW
                    kw: i32 = k % KW
                    for r in range(Ct, name="r"):  # channel lane
                        ch: i32 = ct * Ct + r
                        for c in range(Pt, name="c"):  # pixel lane
                            n: i32 = no * Pt + c
                            oh: i32 = n // OW
                            ow: i32 = n % OW
                            ih: i32 = oh * stride + kh - pad
                            iw: i32 = ow * stride + kw - pad
                            ok = ih >= 0 and ih < IH and iw >= 0 and iw < IW
                            sih: i32 = ih if ih >= 0 and ih < IH else 0
                            siw: i32 = iw if iw >= 0 and iw < IW else 0
                            v: Tin = X[sih, siw, ch]
                            fifo_X[r, c].put(v if ok else 0)

    @kernel(mapping=[Ct, Pt])
    def pe(
        fifo_W: Stream[Tin, depth][Ct, Pt],
        fifo_X: Stream[Tin, depth][Ct, Pt],
        fifo_C: Stream[Tacc, depth][Ct, Pt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for ct in range(CT):
            for no in range(NT):
                acc: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_W[r, c].get()
                    b: Tin = fifo_X[r, c].get()
                    acc += a * b
                    if c < Pt - 1:
                        fifo_W[r, c + 1].put(a)  # weight east (per-channel reuse)
                fifo_C[r, c].put(acc)

    @kernel
    def store_Y(Y: Tout[OH, OW, C], fifo_C: Stream[Tacc, depth][Ct, Pt]):
        for ct in range(CT):
            for no in range(NT, name="no"):
                for r in range(Ct, name="r"):
                    for c in range(Pt, name="c"):
                        n: i32 = no * Pt + c
                        Y[n // OW, n % OW, ct * Ct + r] = fifo_C[r, c].get()

    @kernel
    def dwconv(W: Tin[KH, KW, C], X: Tin[IH, IW, C], Y: Tout[OH, OW, C]):
        """Output-stationary systolic **depthwise** conv2d (NHWC), im2col-streamed.

        Computes ``Y[OH,OW,C] = depthwise_conv2d(X[IH,IW,C], W[KH,KW,C])`` -- each
        channel convolved with its own ``KH x KW`` filter, **no cross-channel
        reduction** (``K = KH*KW``). The ``Ct x Pt`` PE grid maps **channels to
        rows** and **output pixels to columns**, sweeping ``CT x NT`` tiles and
        accumulating each output over its window inside the PE.

        Unlike standard conv, depthwise shares no operand *down* a column (each
        row is a different channel), so the structure differs from
        :mod:`mm.os.direct` in two ways:

        * **Weights flow east only.** A channel's filter is identical for every
          pixel in its row, so ``load_W`` feeds the west edge and each PE forwards
          it east -- genuine systolic weight reuse across the ``Pt`` pixels.
        * **Inputs are fed per-PE.** Activations are channel-specific, so
          ``load_X`` pushes every PE its own im2col window (both lanes unrolled);
          there is no south forwarding.

        ``store_Y`` writes ``Y[oh, ow, channel]`` exactly as the GEMM store.
        Depthwise is low arithmetic intensity (``KH*KW`` MACs/output) and bound by
        input bandwidth / parallelism, not a 2D reduction; this direct variant
        re-reads the overlapping input halo (no line buffer), trading DRAM traffic
        for simplicity.
        """
        fifo_W: Stream[Tin, depth][Ct, Pt]
        fifo_X: Stream[Tin, depth][Ct, Pt]
        fifo_C: Stream[Tacc, depth][Ct, Pt]
        load_W(W, fifo_W)
        load_X(X, fifo_X)
        pe(fifo_W, fifo_X, fifo_C)
        store_Y(Y, fifo_C)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("r")
    lw_s.pipeline("k", ii=ii)

    lx_s = load_X.schedule()
    lx_s.unroll("r")
    lx_s.unroll("c")
    lx_s.pipeline("k", ii=ii)

    st_s = store_Y.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = dwconv.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, lx_s, st_s)

    return dwconv, top_s


def make_buffered_output_stationary_depthwise(
    Tin,
    Tacc,
    Tout,
    C: int,
    IH: int,
    IW: int,
    KH: int,
    KW: int,
    Ct: int,
    Pt: int,
    stride: int = 1,
    pad: int = 0,
    depth=2,
    ii=1,
):
    """Factory for an output-stationary systolic **depthwise** conv2d (NHWC) with
    the feature map staged in **on-chip BRAM** so the per-PE gather pipelines at
    II=1. See :func:`dwconv` for the algorithm/dataflow docstring.

    Requires ``Pt | OW`` (pixel-tile = output-row segment). ``bufX`` is banked on
    both the channel axis (factor ``Ct``, so the ``Ct`` row lanes hit distinct
    banks) and the ``iw`` axis (factor ``Pt*stride``, for the ``Pt`` column
    lanes). Depthwise has trivial arithmetic intensity (``K = KH*KW``), so even
    with an II=1 gather the design stays **input-load bound** -- buffering only
    removes the DRAM-gather penalty of :func:`make_direct_output_stationary_depthwise`.
    """
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    N, K = OH * OW, KH * KW
    CT, NT = C // Ct, N // Pt
    IHP, IWP = IH + 2 * pad, IW + 2 * pad
    NTW = OW // Pt
    BANKW = Pt * stride  # iw banking (pixel lanes)
    FILLU = math.gcd(C, Ct)  # fill ci-lanes/cycle; <= Ct -> distinct channel banks

    @kernel
    def load_W(W: Tin[KH, KW, C], fifo_W: Stream[Tin, depth][Ct, Pt]):
        for ct in range(CT):
            for no in range(NT, name="no"):
                for k in range(K, name="k"):
                    kh: i32 = k // KW
                    kw: i32 = k % KW
                    for r in range(Ct, name="r"):  # lane -> channel
                        fifo_W[r, 0].put(W[kh, kw, ct * Ct + r])

    @kernel
    def load_X(X: Tin[IH, IW, C], fifo_X: Stream[Tin, depth][Ct, Pt]):
        bufX: Tin[IHP, IWP, C]
        for ih in range(IHP, name="fih"):
            for iw in range(IWP, name="fiw"):
                for cb in range(C // FILLU, name="fcb"):
                    for cl in range(FILLU, name="fcl"):
                        ci2: i32 = cb * FILLU + cl
                        xi: i32 = ih - pad
                        xj: i32 = iw - pad
                        inb = xi >= 0 and xi < IH and xj >= 0 and xj < IW
                        sxi: i32 = xi if xi >= 0 and xi < IH else 0
                        sxj: i32 = xj if xj >= 0 and xj < IW else 0
                        v: Tin = X[sxi, sxj, ci2]
                        bufX[ih, iw, ci2] = v if inb else 0
        for ct in range(CT, name="ct"):
            for no in range(NT):
                oh: i32 = no // NTW
                ow0: i32 = (no % NTW) * Pt
                for k in range(K, name="k"):
                    kh: i32 = k // KW
                    kw: i32 = k % KW
                    ih: i32 = oh * stride + kh  # into padded buffer
                    for r in range(Ct, name="r"):  # channel lane
                        ch: i32 = ct * Ct + r
                        for c in range(Pt, name="c"):  # pixel lane
                            iw: i32 = (ow0 + c) * stride + kw
                            fifo_X[r, c].put(bufX[ih, iw, ch])

    @kernel(mapping=[Ct, Pt])
    def pe(
        fifo_W: Stream[Tin, depth][Ct, Pt],
        fifo_X: Stream[Tin, depth][Ct, Pt],
        fifo_C: Stream[Tacc, depth][Ct, Pt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for ct in range(CT):
            for no in range(NT):
                acc: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_W[r, c].get()
                    b: Tin = fifo_X[r, c].get()
                    acc += a * b
                    if c < Pt - 1:
                        fifo_W[r, c + 1].put(a)  # weight east
                fifo_C[r, c].put(acc)

    @kernel
    def store_Y(Y: Tout[OH, OW, C], fifo_C: Stream[Tacc, depth][Ct, Pt]):
        for ct in range(CT):
            for no in range(NT, name="no"):
                oh: i32 = no // NTW
                ow0: i32 = (no % NTW) * Pt
                for r in range(Ct, name="r"):
                    for c in range(Pt, name="c"):
                        Y[oh, ow0 + c, ct * Ct + r] = fifo_C[r, c].get()

    @kernel
    def dwconv(W: Tin[KH, KW, C], X: Tin[IH, IW, C], Y: Tout[OH, OW, C]):
        """On-chip-buffered output-stationary systolic depthwise conv2d (NHWC).

        Same channel-row / pixel-col mapping and east-flowing per-channel weights
        as :func:`make_direct_output_stationary_depthwise`, but the feature map is
        staged into a zero-padded on-chip ``bufX`` first, so the per-PE gather is
        an II=1 BRAM read instead of a DRAM scatter. ``bufX`` is banked on the
        channel axis (``Ct``) and the ``iw`` axis (``Pt*stride``) for conflict-free
        ``Ct x Pt`` parallel reads. Depthwise stays load-bound (``K = KH*KW``).
        """
        fifo_W: Stream[Tin, depth][Ct, Pt]
        fifo_X: Stream[Tin, depth][Ct, Pt]
        fifo_C: Stream[Tacc, depth][Ct, Pt]
        load_W(W, fifo_W)
        load_X(X, fifo_X)
        pe(fifo_W, fifo_X, fifo_C)
        store_Y(Y, fifo_C)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("r")
    lw_s.pipeline("k", ii=ii)

    lx_s = load_X.schedule()
    lx_s.partition(lx_s.buffer("bufX"), dim=2, kind=lx_s.Cyclic, factor=BANKW)  # iw
    lx_s.partition(lx_s.buffer("bufX"), dim=3, kind=lx_s.Cyclic, factor=Ct)  # channel
    lx_s.unroll("fcl")
    lx_s.unroll("r")
    lx_s.unroll("c")
    lx_s.pipeline("k", ii=ii)
    lx_s.pipeline(lx_s.flatten(("fih", "fiw", "fcb")), ii=ii)

    st_s = store_Y.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = dwconv.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, lx_s, st_s)

    return dwconv, top_s
