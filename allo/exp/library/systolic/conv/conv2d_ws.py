# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.exp as allo
from allo.exp import kernel
from allo.exp.lang import Stream, range
from allo.exp.lang.core import DType, i32


def make_direct_weight_stationary_conv2d(
    Tin,
    Tacc,
    Tout,
    IC,
    OC,
    IH,
    IW,
    KH,
    KW,
    Ct,
    Mt,
    stride=1,
    pad=0,
    dil=1,
    depth=2,
    ii=1,
):
    """Build + schedule a direct weight-stationary 2D convolution; return
    ``(top, top_s)``.

    The ``Ct x Mt`` PE array maps ``(IC-tile, OC-tile)``: a tile of input
    channels (the contraction) along the rows and a tile of output channels
    along the columns. Output positions ``OH*OW`` stream through time. ``Ct``
    tiles ``IC`` and ``Mt`` tiles ``OC`` (so ``IC % Ct == 0`` and ``OC % Mt ==
    0``). ``stride``/``pad``/``dil`` give strided / zero-padded / dilated conv.
    ``depth`` is the inter-process FIFO depth; bump it if co-simulation reports
    a stall.

    TENSOR LAYOUTS (see the ``top`` docstring for why):
    * input  ``inp`` : ``[IH, IW, IC]``      (NHWC -- channels innermost)
    * weight ``w``   : ``[KH, KW, IC, OC]``   (taps outer, channels innermost)
    * output ``out`` : ``[OH, OW, OC]``       (NHWC)
    """
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert IC % Ct == 0 and OC % Mt == 0, "array must tile IC and OC evenly"
    assert stride >= 1 and dil >= 1 and pad >= 0
    OH = (IH + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    assert OH >= 1 and OW >= 1, "kernel/stride/pad yield an empty output"
    ICT, OCT = IC // Ct, OC // Mt  # channel-tile counts
    OP = OH * OW  # output positions (the stream / time dimension)
    RT = ICT * KH * KW  # reduction tiles per oc-tile (channel-tiles x taps)

    @kernel
    def load_W(w: Tin[KH, KW, IC, OC], fifo_W: Stream[Tin, depth][Ct, Mt]):
        # PE(cc,oo) latches w[kh, kw, ict*Ct+cc, oct*Mt+oo] once per reduction
        # tile. The [.,.,IC,OC] layout keeps the unrolled (ic,oc) lanes
        # contiguous (short address path), mirroring GEMM's B[k,n].
        for oct in range(OCT):
            for ict in range(ICT, name="ict"):
                for kh in range(KH, name="kh"):
                    for kw in range(KW, name="kw"):
                        for cc in range(Ct, name="cc"):  # lane
                            for oo in range(Mt, name="oo"):  # lane
                                fifo_W[cc, oo].put(
                                    w[kh, kw, ict * Ct + cc, oct * Mt + oo]
                                )

    @kernel
    def load_A(inp: Tin[IH, IW, IC], fifo_A: Stream[Tin, depth][Ct, Mt]):
        # On-the-fly im2col: for output position (oh,ow), tap (kh,kw) and channel
        # ic feed inp[oh*stride+kh*dil-pad, ow*stride+kw*dil-pad, ic] (0 if the
        # padded location is out of bounds). NHWC keeps the Ct channel-lanes
        # contiguous so the unrolled west-edge reads coalesce to II=1.
        for oct in range(OCT):
            for ict in range(ICT):
                for kh in range(KH):
                    for kw in range(KW):
                        for oh in range(OH, name="oh"):
                            for ow in range(OW, name="ow"):
                                for cc in range(Ct, name="cc"):  # lane (west edge)
                                    ih: i32 = oh * stride + kh * dil - pad
                                    iw: i32 = ow * stride + kw * dil - pad
                                    a: Tin = 0
                                    if ih >= 0:
                                        if ih < IH:
                                            if iw >= 0:
                                                if iw < IW:
                                                    a = inp[ih, iw, ict * Ct + cc]
                                    fifo_A[cc, 0].put(a)

    @kernel(mapping=[Ct, Mt])
    def pe(
        fifo_W: Stream[Tin, depth][Ct, Mt],
        fifo_A: Stream[Tin, depth][Ct, Mt],
        fifo_P: Stream[Tacc, depth][Ct, Mt],
        fifo_O: Stream[Tacc, depth][Ct, Mt],
    ):
        cc = allo.get_wid(0)  # array row = input channel within tile (contraction)
        oo = allo.get_wid(1)  # array col = output channel within tile
        for oct in range(OCT):
            for ict in range(ICT):
                for kh in range(KH):
                    for kw in range(KW):
                        w: Tin = fifo_W[cc, oo].get()  # latch resident weight
                        for p in range(OP, name="p"):
                            a: Tin = fifo_A[cc, oo].get()
                            acc: Tacc = a * w
                            if cc > 0:  # add partial sum from the north neighbor
                                acc = acc + fifo_P[cc, oo].get()
                            if oo < Mt - 1:  # forward activation east
                                fifo_A[cc, oo + 1].put(a)
                            if cc < Ct - 1:  # push partial sum south
                                fifo_P[cc + 1, oo].put(acc)
                            else:  # bottom row emits the channel-tile partial
                                fifo_O[cc, oo].put(acc)

    @kernel
    def reduce_C(fifo_O: Stream[Tacc, depth][Ct, Mt], fifo_Ct: Stream[Tacc, depth][Mt]):
        # Sum the bottom-row partials across all RT reduction tiles (channel-tiles
        # x kernel taps); emit each finalized output position so the DRAM write
        # can overlap.
        accC: Tacc[OP, Mt]  # per-(output-position, oc-lane) accumulator across tiles
        for oct in range(OCT):
            for rt in range(RT):
                for p in range(OP, name="p"):
                    for oo in range(Mt, name="oo"):
                        pv: Tacc = fifo_O[Ct - 1, oo].get()
                        acc_val: Tacc = pv
                        if rt > 0:
                            acc_val = accC[p, oo] + pv
                        if rt == RT - 1:  # last reduction tile -> emit
                            fifo_Ct[oo].put(acc_val)
                        else:
                            accC[p, oo] = acc_val

    @kernel
    def write_C(out: Tout[OH, OW, OC], fifo_Ct: Stream[Tacc, depth][Mt]):
        # NHWC output -> the Mt oc-lanes are contiguous (coalesced writes).
        for oct in range(OCT):
            for oh in range(OH, name="oh"):
                for ow in range(OW, name="ow"):
                    for oo in range(Mt, name="oo"):
                        out[oh, ow, oct * Mt + oo] = fifo_Ct[oo].get()

    @kernel
    def top(inp: Tin[IH, IW, IC], w: Tin[KH, KW, IC, OC], out: Tout[OH, OW, OC]):
        """Weight-stationary systolic **2D convolution** via on-the-fly im2col.

        Computes ``out = conv2d(inp, w)`` with ``stride``/``pad``/``dil``. A conv
        is run as a GEMM over ``K = IC*KH*KW`` without materializing im2col: the
        ``Ct x Mt`` PE array maps ``(IC-tile, OC-tile)``, output positions stream
        through time, weights stay resident, and ``reduce_C`` sums the spatial
        column partials. ``direct`` re-reads the input from DRAM (no buffer).

        Tensors are **NHWC** (channels innermost) so the array's channel/oc lanes
        read contiguously; the weight is reordered to put those lanes innermost:

        * input  ``inp`` : ``[IH, IW, IC]``
        * weight ``w``   : ``[KH, KW, IC, OC]``
        * output ``out`` : ``[OH, OW, OC]``
        """
        fifo_W: Stream[Tin, depth][Ct, Mt]
        fifo_A: Stream[Tin, depth][Ct, Mt]
        fifo_P: Stream[Tacc, depth][Ct, Mt]
        fifo_O: Stream[Tacc, depth][Ct, Mt]
        fifo_Ct: Stream[Tacc, depth][Mt]
        load_W(w, fifo_W)
        load_A(inp, fifo_A)
        pe(fifo_W, fifo_A, fifo_P, fifo_O)
        reduce_C(fifo_O, fifo_Ct)
        write_C(out, fifo_Ct)

    pe_s = pe.schedule()
    pe_s.pipeline("p", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("oo")  # lane
    lw_s.unroll("cc")  # lane
    lw_s.pipeline("kw", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("cc")  # lane -> static stream index
    la_s.pipeline("ow", ii=ii)

    rc_s = reduce_C.schedule()
    rc_s.partition(rc_s.buffer("accC"), dim=2, kind=rc_s.Complete)  # oc lanes
    rc_s.unroll("oo")
    rc_s.pipeline("p", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("oo")
    wc_s.pipeline("ow", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, rc_s, wc_s)

    return top, top_s
