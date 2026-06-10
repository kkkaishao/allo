# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.exp as allo
from allo.exp import kernel
from allo.exp.lang import Stream, range
from allo.exp.lang.core import DType, i32


def make_buffered_weight_stationary_conv2d(
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
    OHb=None,
    depth=2,
    ii=1,
):
    """Build + schedule a **line-buffered** weight-stationary conv2d; return
    ``(top, top_s)``.

    Same systolic mapping as :mod:`conv2d_ws` (``Ct`` tiles ``IC``, ``Mt`` tiles
    ``OC``, output positions stream), but the input is consumed in **blocks of
    ``OHb`` output rows**: ``load_A`` stages the ``IHb`` input rows a block
    touches into an on-chip buffer and replays them across all kernel taps /
    oc-tiles, so each input pixel is read from DRAM ~once instead of
    ``OCT*KH*KW`` times.

    ``OHb`` (defaults to ``OH`` -- buffer the whole plane) is the buffer-size vs
    DRAM-traffic knob: small ``OHb`` is the classic ``KH``-row line buffer
    (smallest buffer) at the cost of re-reading the ``(KH-1)``-row halo between
    blocks and re-streaming the (tiny) weights per block. Must satisfy
    ``OH % OHb == 0``. ``depth`` is the inter-process FIFO depth.

    Unlike the GEMM buffered variant, conv buffering is **not** csynth-neutral:
    the per-block fill and weight re-stream are serial, so latency grows modestly
    as ``OHb`` shrinks (measured 128^3-class 32x32/16^2/k3: direct ~9.4k, OHb=4
    ~10.3k (+9%), OHb=1 ~12.1k (+28%)). The payoff is ~``OCT*KH*KW`` less input
    DRAM traffic and a small on-chip footprint -- a power/bandwidth win on-board.

    Tensor layouts (NHWC, weight reordered): ``inp [IH,IW,IC]``,
    ``w [KH,KW,IC,OC]``, ``out [OH,OW,OC]``.
    """
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert IC % Ct == 0 and OC % Mt == 0, "array must tile IC and OC evenly"
    assert stride >= 1 and dil >= 1 and pad >= 0
    OH = (IH + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    assert OH >= 1 and OW >= 1, "kernel/stride/pad yield an empty output"
    OHb = OH if OHb is None else OHb
    assert OH % OHb == 0, "OHb must divide OH"
    ICT, OCT = IC // Ct, OC // Mt
    OHB = OH // OHb  # output-row blocks
    IHb = (OHb - 1) * stride + dil * (KH - 1) + 1  # input rows a block touches
    PB = OHb * OW  # output positions per block
    RT = ICT * KH * KW  # reduction tiles per oc-tile

    @kernel
    def load_W(w: Tin[KH, KW, IC, OC], fifo_W: Stream[Tin, depth][Ct, Mt]):
        # re-streamed once per output-row block (weights are tiny vs activations).
        for ob in range(OHB):
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
        bufI: Tin[IHb, IW, IC]  # the input rows the current block touches
        for ob in range(OHB):
            # fill: read this block's input rows once (rows out of range -> 0).
            for r in range(IHb, name="fr"):
                for iw in range(IW, name="fw"):
                    for ict in range(ICT, name="fct"):
                        for cc in range(Ct, name="fcc"):  # lane (channel)
                            gih: i32 = ob * OHb * stride - pad + r
                            v: Tin = 0
                            if gih >= 0:
                                if gih < IH:
                                    v = inp[gih, iw, ict * Ct + cc]
                            bufI[r, iw, ict * Ct + cc] = v
            # stream im2col from the on-chip buffer (no DRAM; free replay). The
            # buffer row a tap reads is brow = ohl*stride + kh*dil (column padding
            # is still handled here; row padding was baked in at fill time).
            for oct in range(OCT):
                for ict in range(ICT):
                    for kh in range(KH):
                        for kw in range(KW):
                            for ohl in range(OHb, name="ohl"):
                                for ow in range(OW, name="ow"):
                                    for cc in range(Ct, name="cc"):  # lane
                                        brow: i32 = ohl * stride + kh * dil
                                        iw: i32 = ow * stride + kw * dil - pad
                                        a: Tin = 0
                                        if iw >= 0:
                                            if iw < IW:
                                                a = bufI[brow, iw, ict * Ct + cc]
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
        for ob in range(OHB):
            for oct in range(OCT):
                for ict in range(ICT):
                    for kh in range(KH):
                        for kw in range(KW):
                            w: Tin = fifo_W[cc, oo].get()  # latch resident weight
                            for pb in range(PB, name="pb"):
                                a: Tin = fifo_A[cc, oo].get()
                                acc: Tacc = a * w
                                if cc > 0:
                                    acc = acc + fifo_P[cc, oo].get()
                                if oo < Mt - 1:
                                    fifo_A[cc, oo + 1].put(a)
                                if cc < Ct - 1:
                                    fifo_P[cc + 1, oo].put(acc)
                                else:
                                    fifo_O[cc, oo].put(acc)

    @kernel
    def reduce_C(fifo_O: Stream[Tacc, depth][Ct, Mt], fifo_Ct: Stream[Tacc, depth][Mt]):
        accC: Tacc[PB, Mt]  # per-(block-position, oc-lane) accumulator across tiles
        for ob in range(OHB):
            for oct in range(OCT):
                for rt in range(RT):
                    for pb in range(PB, name="pb"):
                        for oo in range(Mt, name="oo"):
                            pv: Tacc = fifo_O[Ct - 1, oo].get()
                            acc_val: Tacc = pv
                            if rt > 0:
                                acc_val = accC[pb, oo] + pv
                            if rt == RT - 1:
                                fifo_Ct[oo].put(acc_val)
                            else:
                                accC[pb, oo] = acc_val

    @kernel
    def write_C(out: Tout[OH, OW, OC], fifo_Ct: Stream[Tacc, depth][Mt]):
        for ob in range(OHB):
            for oct in range(OCT):
                for ohl in range(OHb, name="ohl"):
                    for ow in range(OW, name="ow"):
                        for oo in range(Mt, name="oo"):
                            out[ob * OHb + ohl, ow, oct * Mt + oo] = fifo_Ct[oo].get()

    @kernel
    def top(inp: Tin[IH, IW, IC], w: Tin[KH, KW, IC, OC], out: Tout[OH, OW, OC]):
        """Line-buffered weight-stationary systolic **2D convolution**.

        Computes ``out = conv2d(inp, w)`` with ``stride``/``pad``/``dil`` using
        the same conv-as-GEMM mapping as :mod:`conv2d_ws`, but processes output
        rows in blocks of ``OHb`` and stages each block's input rows on-chip so
        the input is read from DRAM ~once (a line buffer). ``OHb`` trades buffer
        size against DRAM halo/weight re-reads; smaller ``OHb`` slightly raises
        latency (serial per-block fill + weight re-stream) for less DRAM traffic.

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
    pe_s.pipeline("pb", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("oo")  # lane
    lw_s.unroll("cc")  # lane
    lw_s.pipeline("kw", ii=ii)

    la_s = load_A.schedule()
    la_s.partition(la_s.buffer("bufI"), dim=3, kind=la_s.Cyclic, factor=Ct)  # channel
    la_s.unroll("fcc")
    la_s.pipeline("fct", ii=ii)
    la_s.unroll("cc")  # lane -> static stream index
    la_s.pipeline("ow", ii=ii)

    rc_s = reduce_C.schedule()
    rc_s.partition(rc_s.buffer("accC"), dim=2, kind=rc_s.Complete)  # oc lanes
    rc_s.unroll("oo")
    rc_s.pipeline("pb", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("oo")
    wc_s.pipeline("ow", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, rc_s, wc_s)

    return top, top_s
