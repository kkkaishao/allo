# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import i32


def make_buffered_output_stationary_conv2d(
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
    stride: int = 1,
    pad: int = 0,
    depth=2,
    ii=1,
):
    """Factory for an output-stationary systolic **conv2d** (NHWC) with the input
    feature map staged in **on-chip BRAM** so the im2col gather pipelines at II=1.

    Requires ``Nt | OW`` (a column-tile is a contiguous segment of one output
    row), so all ``Nt`` lanes share the same input row and differ only in the
    column ``iw`` -- ``bufX`` is then banked on ``iw`` for conflict-free parallel
    reads. See :func:`conv` for the full docstring.
    """
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    M, N, K = Co, OH * OW, KH * KW * Ci
    MT, NT = M // Mt, N // Nt
    IHP, IWP = IH + 2 * pad, IW + 2 * pad  # zero-padded on-chip extent
    NTW = OW // Nt  # column-tiles per output row
    BANK = Nt * stride  # iw banking factor: lanes read iw0 + c*stride, c<Nt
    FILLU = math.gcd(Ci, 8)  # fill ci-lanes/cycle: hide the one-time bufX prologue

    @kernel
    def load_W(Wt: Tin[K, Co], fifo_A: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MT):
            for no in range(NT, name="no"):
                for k in range(K, name="k"):
                    for r in range(Mt, name="r"):
                        fifo_A[r, 0].put(Wt[k, mo * Mt + r])

    @kernel
    def load_X(X: Tin[IH, IW, Ci], fifo_B: Stream[Tin, depth][Mt, Nt]):
        # Stage X into a zero-padded on-chip buffer (one widened pass: the padding
        # halo is materialized here at the Tin level), then im2col-gather from
        # BRAM with no per-read bounds logic.  ci is blocked into FILLU lanes so
        # the prologue copies FILLU channels per cycle (hidden behind compute).
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
        for mo in range(MT, name="mo"):  # replay buffer across channel-tiles
            for no in range(NT):
                oh: i32 = no // NTW  # tile -> output row (Nt | OW)
                ow0: i32 = (no % NTW) * Nt  # tile -> first output col
                for k in range(K, name="k"):
                    ci: i32 = k % Ci
                    kw: i32 = (k // Ci) % KW
                    kh: i32 = k // (Ci * KW)
                    ih: i32 = oh * stride + kh  # indices into the padded buffer
                    for c in range(Nt, name="c"):
                        iw: i32 = (ow0 + c) * stride + kw
                        fifo_B[0, c].put(bufX[ih, iw, ci])

    @kernel(mapping=[Mt, Nt])
    def pe(
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_B: Stream[Tin, depth][Mt, Nt],
        fifo_C: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for mo in range(MT):
            for no in range(NT):
                acc: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_A[r, c].get()
                    b: Tin = fifo_B[r, c].get()
                    acc += a * b
                    if c < Nt - 1:
                        fifo_A[r, c + 1].put(a)
                    if r < Mt - 1:
                        fifo_B[r + 1, c].put(b)
                fifo_C[r, c].put(acc)

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
        """Output-stationary systolic **conv2d** (NHWC), **on-chip-buffered** input.

        Same OS GEMM lowering as :mod:`direct` (``K = KH*KW*Ci``; weights flow
        east, activations south; ``store_Y`` writes ``Y[oh, ow, co]``), but the
        input loader first **bursts the whole feature map into a zero-padded
        on-chip ``bufX[IH+2P, IW+2P, Ci]``**, then performs im2col **from BRAM** --
        so the data-dependent gather that capped :mod:`direct` at II~16 (each tap
        an individual AXI transaction) now pipelines at II=1, leaving the design
        **PE-bound** at the compute floor.

        The padding halo is materialized **at fill time** (a Tin-level select), so
        the gather reads ``bufX[oh*S+kh, (ow0+c)*S+kw, ci]`` with **no per-read
        bounds logic**. A column-tile is a contiguous **segment of one output
        row** (``Nt | OW``), so all ``Nt`` lanes share the same input row ``ih``
        and differ only in ``iw`` -- ``bufX`` is banked cyclically on ``iw``
        (factor ``Nt*stride``) for conflict-free parallel reads. The one-time fill
        copies ``FILLU = gcd(Ci, 8)`` channels per cycle (``bufX`` banked on
        ``ci``) to hide the prologue. ``bufX`` is filled once and **replayed
        across the ``MT`` channel-tiles**, so ``X`` is read from DRAM exactly once
        (vs ``MT`` times in :mod:`direct`).

        WHEN TO USE: the realistic conv dataflow -- prefer it whenever the feature
        map fits on chip (``IH*IW*Ci`` words). For very large feature maps where
        ``bufX`` would exhaust BRAM, fall back to :mod:`direct` (which needs no
        input buffer) or tile the feature map outside the kernel.
        """
        fifo_A: Stream[Tin, depth][Mt, Nt]
        fifo_B: Stream[Tin, depth][Mt, Nt]
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
    )  # iw (gather lanes)
    if FILLU > 1:
        lx_s.partition(
            lx_s.buffer("bufX"), dim=3, kind=lx_s.Cyclic, factor=FILLU
        )  # ci (fill lanes)
    lx_s.unroll("fcl")  # widen the prologue copy to FILLU writes/cycle
    lx_s.unroll("c")
    lx_s.pipeline("k", ii=ii)
    lx_s.pipeline(lx_s.flatten(("fih", "fiw", "fcb")), ii=ii)  # burst-fill

    st_s = store_Y.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = conv.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, lx_s, st_s)

    return conv, top_s
