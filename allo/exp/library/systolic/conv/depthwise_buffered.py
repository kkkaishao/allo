# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.exp as allo
from allo.exp import kernel
from allo.exp.lang import Stream, range
from allo.exp.lang.core import DType, i32


def make_buffered_weight_stationary_depthwise(
    Tin,
    Tacc,
    Tout,
    C,
    IH,
    IW,
    KH,
    KW,
    Ct,
    Mt,
    stride=1,
    pad=0,
    dil=1,
    OHb=None,
    depth=2,
    ii=1,
):
    """Build + schedule a **line-buffered** weight-stationary depthwise conv2d;
    return ``(top, top_s)``.

    Same channel-parallel mapping as :mod:`depthwise` (``Ct*Mt`` PEs = independent
    per-channel ``KH*KW`` FIR lanes), but output rows are processed in **blocks of
    ``OHb``**: ``load_A`` stages the ``IHb`` input rows a block touches into an
    on-chip buffer (partitioned by channel) and replays the sliding window from
    there, so each input pixel is read from DRAM **once** instead of ``KH*KW``
    times. This removes the DRAM-bandwidth bottleneck of the direct variant (whose
    window re-reads make it memory-bound). ``OHb`` (defaults to ``OH`` -- buffer
    the whole plane) trades buffer size against DRAM halo / weight re-reads; must
    satisfy ``OH % OHb == 0``. ``P = Ct*Mt`` must divide ``C``.

    Tensor layouts (NHWC, channels innermost): ``inp [IH,IW,C]``, ``w [KH,KW,C]``,
    ``out [OH,OW,C]``.
    """
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert stride >= 1 and dil >= 1 and pad >= 0
    P = Ct * Mt
    assert C % P == 0, "Ct*Mt channel lanes must tile C"
    OH = (IH + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    assert OH >= 1 and OW >= 1, "kernel/stride/pad yield an empty output"
    OHb = OH if OHb is None else OHb
    assert OH % OHb == 0, "OHb must divide OH"
    CT = C // P  # channel-tile count
    KK = KH * KW  # taps per output
    OHB = OH // OHb  # output-row blocks
    IHb = (OHb - 1) * stride + dil * (KH - 1) + 1  # input rows a block touches
    PB = OHb * OW  # outputs per block

    @kernel
    def load_W(w: Tin[KH, KW, C], fifo_W: Stream[Tin, depth][Ct, Mt]):
        # per-channel KH*KW filter; re-streamed once per output-row block.
        for ob in range(OHB):
            for ct in range(CT):
                for kh in range(KH, name="kh"):
                    for kw in range(KW, name="kw"):
                        for cc in range(Ct, name="cc"):  # lane
                            for mm in range(Mt, name="mm"):  # lane
                                fifo_W[cc, mm].put(w[kh, kw, ct * P + cc * Mt + mm])

    @kernel
    def load_A(inp: Tin[IH, IW, C], fifo_A: Stream[Tin, depth][Ct, Mt]):
        bufI: Tin[IHb, IW, C]  # input rows the current block touches (all channels)
        for ob in range(OHB):
            # fill: read this block's input rows once (rows out of range -> 0).
            for r in range(IHb, name="fr"):
                for iw in range(IW, name="fw"):
                    for ct in range(CT, name="fct"):
                        for cc in range(Ct, name="fcc"):  # lane (channel)
                            for mm in range(Mt, name="fmm"):  # lane (channel)
                                gih: i32 = ob * OHb * stride - pad + r
                                v: Tin = 0
                                if gih >= 0:
                                    if gih < IH:
                                        v = inp[gih, iw, ct * P + cc * Mt + mm]
                                bufI[r, iw, ct * P + cc * Mt + mm] = v
            # stream the sliding window from the buffer, flattened (output x tap)
            # to match the PE -> one continuous II=1 pipeline (no per-output drain).
            for ct in range(CT):
                for f in range(PB * KK, name="sf"):
                    o: i32 = f // KK
                    t: i32 = f % KK
                    ohl: i32 = o // OW
                    ow: i32 = o % OW
                    kh: i32 = t // KW
                    kw: i32 = t % KW
                    brow: i32 = ohl * stride + kh * dil
                    iw: i32 = ow * stride + kw * dil - pad
                    for cc in range(Ct, name="cc"):  # lane
                        for mm in range(Mt, name="mm"):  # lane
                            a: Tin = 0
                            if iw >= 0:
                                if iw < IW:
                                    a = bufI[brow, iw, ct * P + cc * Mt + mm]
                            fifo_A[cc, mm].put(a)

    @kernel(mapping=[Ct, Mt])
    def pe(
        fifo_W: Stream[Tin, depth][Ct, Mt],
        fifo_A: Stream[Tin, depth][Ct, Mt],
        fifo_O: Stream[Tacc, depth][Ct, Mt],
    ):
        cc = allo.get_wid(0)
        mm = allo.get_wid(1)
        for ob in range(OHB):
            for ct in range(CT):
                wreg: Tin[KH, KW]  # resident per-channel filter
                for kh in range(KH, name="lwh"):
                    for kw in range(KW, name="lww"):
                        wreg[kh, kw] = fifo_W[cc, mm].get()
                # one continuous II=1 pipeline over (output x tap); tap counter `t`
                # resets the accumulator (t==0) and emits the output (t==KK-1).
                acc: Tacc = 0
                for f in range(PB * KK, name="f"):
                    t: i32 = f % KK
                    kh: i32 = t // KW
                    kw: i32 = t % KW
                    a: Tin = fifo_A[cc, mm].get()
                    prod: Tacc = a * wreg[kh, kw]
                    if t == 0:
                        acc = prod
                    else:
                        acc = acc + prod
                    if t == KK - 1:
                        fifo_O[cc, mm].put(acc)

    @kernel
    def write_C(out: Tout[OH, OW, C], fifo_O: Stream[Tacc, depth][Ct, Mt]):
        for ob in range(OHB):
            for ct in range(CT):
                for ohl in range(OHb):
                    for ow in range(OW, name="ow"):
                        for cc in range(Ct, name="cc"):  # lane
                            for mm in range(Mt, name="mm"):  # lane
                                out[ob * OHb + ohl, ow, ct * P + cc * Mt + mm] = fifo_O[
                                    cc, mm
                                ].get()

    @kernel
    def top(inp: Tin[IH, IW, C], w: Tin[KH, KW, C], out: Tout[OH, OW, C]):
        """Line-buffered weight-stationary systolic **depthwise** 2D convolution.

        Computes ``out[oh,ow,c] = sum_{kh,kw} inp[oh*S+kh*D-pad, ow*S+kw*D-pad,
        c] * w[kh,kw,c]``. Same channel-parallel FIR mapping as :mod:`depthwise`,
        but processes output rows in blocks of ``OHb`` and stages each block's
        input rows on-chip, so the input is read from DRAM ~once (a line buffer)
        -- removing the DRAM-bandwidth bottleneck of the direct variant. ``OHb``
        trades buffer size for DRAM halo/weight re-reads.

        Tensors are **NHWC** (channels innermost) so the lanes read contiguously:

        * input  ``inp`` : ``[IH, IW, C]``
        * weight ``w``   : ``[KH, KW, C]``
        * output ``out`` : ``[OH, OW, C]``
        """
        fifo_W: Stream[Tin, depth][Ct, Mt]
        fifo_A: Stream[Tin, depth][Ct, Mt]
        fifo_O: Stream[Tacc, depth][Ct, Mt]
        load_W(w, fifo_W)
        load_A(inp, fifo_A)
        pe(fifo_W, fifo_A, fifo_O)
        write_C(out, fifo_O)

    pe_s = pe.schedule()
    pe_s.pipeline("f", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("cc")
    lw_s.unroll("mm")
    lw_s.pipeline("kw", ii=ii)

    la_s = load_A.schedule()
    la_s.partition(la_s.buffer("bufI"), dim=3, kind=la_s.Cyclic, factor=P)  # channel
    la_s.unroll("fcc")
    la_s.unroll("fmm")
    la_s.pipeline("fct", ii=ii)
    la_s.unroll("cc")
    la_s.unroll("mm")
    la_s.pipeline("sf", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("cc")
    wc_s.unroll("mm")
    wc_s.pipeline("ow", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, wc_s)

    return top, top_s
