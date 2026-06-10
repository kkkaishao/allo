# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.exp as allo
from allo.exp import kernel
from allo.exp.lang import Stream, range
from allo.exp.lang.core import DType, APInt, i32


def make_packed_weight_stationary_conv2d(
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
    P=2,
    G=18,
    depth=2,
    ii=1,
):
    """Build + schedule a **DSP-packed** weight-stationary conv2d; return
    ``(top, top_s)``.

    Low-bitwidth integer only. Same conv-as-GEMM mapping as :mod:`conv2d_ws`, but
    a packed PE owns ``P`` adjacent **output channels** that share the eastward
    activation: their ``P`` weights are packed into one operand, one wide multiply
    yields all ``P`` products, and a signed borrow-chain unpacks them into ``P``
    per-channel psum chains. The array is ``Ct x (Mt/P)``. ``G`` is the product
    bit gap (must exceed the product width: i8->18, i4->10); requires
    ``Mt % P == 0``.

    Tensor layouts (NHWC, weight reordered): ``inp [IH,IW,IC]``,
    ``w [KH,KW,IC,OC]``, ``out [OH,OW,OC]``.
    """
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert Tin.is_int() and Tacc.is_int(), "DSP packing is integer-only"
    assert IC % Ct == 0 and OC % Mt == 0 and Mt % P == 0, "tiling must be even"
    assert stride >= 1 and dil >= 1 and pad >= 0
    OH = (IH + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    assert OH >= 1 and OW >= 1, "kernel/stride/pad yield an empty output"
    ICT, OCT, MG = IC // Ct, OC // Mt, Mt // P
    OP = OH * OW  # output positions (stream dim)
    RT = ICT * KH * KW  # reduction tiles per oc-tile
    PW = (P - 1) * G + Tin.primitive_width
    Wt = APInt(PW + 1, signed=True)  # packed weight (signed)
    Pt = APInt(PW + Tin.primitive_width + 2, signed=True)  # product width

    @kernel
    def load_W(w: Tin[KH, KW, IC, OC], fifo_W: Stream[Wt, depth][Ct, MG]):
        # pack P adjacent output channels per group (each widened to Wt FIRST,
        # else `wj<<(j*G)` truncates in the narrow operand type).
        for oct in range(OCT):
            for ict in range(ICT, name="ict"):
                for kh in range(KH, name="kh"):
                    for kw in range(KW, name="kw"):
                        for cc in range(Ct, name="cc"):  # lane
                            for og in range(MG, name="og"):  # lane (col-group)
                                base: i32 = oct * Mt + og * P
                                w0: Wt = w[kh, kw, ict * Ct + cc, base]
                                w1: Wt = w[kh, kw, ict * Ct + cc, base + 1]
                                packed: Wt = w0 + (w1 << G)
                                if P >= 4:
                                    w2: Wt = w[kh, kw, ict * Ct + cc, base + 2]
                                    w3: Wt = w[kh, kw, ict * Ct + cc, base + 3]
                                    packed = packed + (w2 << (2 * G)) + (w3 << (3 * G))
                                fifo_W[cc, og].put(packed)

    @kernel
    def load_A(inp: Tin[IH, IW, IC], fifo_A: Stream[Tin, depth][Ct, MG]):
        # on-the-fly im2col (unchanged from conv2d_ws; packing is on the oc dim).
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

    @kernel(mapping=[Ct, MG])
    def pe(
        fifo_W: Stream[Wt, depth][Ct, MG],
        fifo_A: Stream[Tin, depth][Ct, MG],
        fifo_P: Stream[Tacc, depth][Ct, Mt],
        fifo_O: Stream[Tacc, depth][Ct, Mt],
    ):
        cc = allo.get_wid(0)  # array row = input channel within tile (contraction)
        og = allo.get_wid(1)  # array col-group (P output channels each)
        c0 = og * P
        for oct in range(OCT):
            for ict in range(ICT):
                for kh in range(KH):
                    for kw in range(KW):
                        wpk: Wt = fifo_W[cc, og].get()  # resident packed weight
                        for p in range(OP, name="p"):
                            a: Tin = fifo_A[cc, og].get()
                            prod: Pt = a * wpk  # one multiply -> P products
                            if og < MG - 1:
                                fifo_A[cc, og + 1].put(a)  # activation east
                            # signed borrow-chain unpack
                            b0: Tacc = prod[G - 1 : G]
                            p0: Tacc = prod[0:G] - (b0 << G)
                            b1: Tacc = prod[2 * G - 1 : 2 * G]
                            p1: Tacc = prod[G : 2 * G] - (b1 << G) + b0
                            if cc > 0:  # partial sums from the north neighbor
                                p0 = p0 + fifo_P[cc, c0].get()
                                p1 = p1 + fifo_P[cc, c0 + 1].get()
                            if cc < Ct - 1:  # push south
                                fifo_P[cc + 1, c0].put(p0)
                                fifo_P[cc + 1, c0 + 1].put(p1)
                            else:  # bottom row emits
                                fifo_O[cc, c0].put(p0)
                                fifo_O[cc, c0 + 1].put(p1)
                            if P >= 4:
                                b2: Tacc = prod[3 * G - 1 : 3 * G]
                                p2: Tacc = prod[2 * G : 3 * G] - (b2 << G) + b1
                                b3: Tacc = prod[4 * G - 1 : 4 * G]
                                p3: Tacc = prod[3 * G : 4 * G] - (b3 << G) + b2
                                if cc > 0:
                                    p2 = p2 + fifo_P[cc, c0 + 2].get()
                                    p3 = p3 + fifo_P[cc, c0 + 3].get()
                                if cc < Ct - 1:
                                    fifo_P[cc + 1, c0 + 2].put(p2)
                                    fifo_P[cc + 1, c0 + 3].put(p3)
                                else:
                                    fifo_O[cc, c0 + 2].put(p2)
                                    fifo_O[cc, c0 + 3].put(p3)

    @kernel
    def reduce_C(fifo_O: Stream[Tacc, depth][Ct, Mt], fifo_Ct: Stream[Tacc, depth][Mt]):
        accC: Tacc[OP, Mt]
        for oct in range(OCT):
            for rt in range(RT):
                for p in range(OP, name="p"):
                    for oo in range(Mt, name="oo"):
                        pv: Tacc = fifo_O[Ct - 1, oo].get()
                        acc_val: Tacc = pv
                        if rt > 0:
                            acc_val = accC[p, oo] + pv
                        if rt == RT - 1:
                            fifo_Ct[oo].put(acc_val)
                        else:
                            accC[p, oo] = acc_val

    @kernel
    def write_C(out: Tout[OH, OW, OC], fifo_Ct: Stream[Tacc, depth][Mt]):
        for oct in range(OCT):
            for oh in range(OH, name="oh"):
                for ow in range(OW, name="ow"):
                    for oo in range(Mt, name="oo"):
                        out[oh, ow, oct * Mt + oo] = fifo_Ct[oo].get()

    @kernel
    def top(inp: Tin[IH, IW, IC], w: Tin[KH, KW, IC, OC], out: Tout[OH, OW, OC]):
        """Weight-stationary systolic **2D convolution** with low-bitwidth DSP
        packing.

        Computes ``out = conv2d(inp, w)`` with ``stride``/``pad``/``dil`` like
        :mod:`conv2d_ws`, but a PE handles ``P`` adjacent output channels sharing
        the streamed activation via one packed multiply -> the multiply DSP count
        drops ~``Px`` (measured u280/2023.2: int8 P=2 halves DSP, latency-neutral;
        int4 maps the narrow packed multiply to LUTs -> 0 DSP, so P=2 is the
        sensible default there too). Integer only; for int16/float use
        :mod:`conv2d_ws` (products too wide to share a 27-bit DSP port).

        Tensors are **NHWC** (channels innermost) so the array's channel/oc lanes
        read contiguously; the weight is reordered to put those lanes innermost:

        * input  ``inp`` : ``[IH, IW, IC]``
        * weight ``w``   : ``[KH, KW, IC, OC]``
        * output ``out`` : ``[OH, OW, OC]``
        """
        fifo_W: Stream[Wt, depth][Ct, MG]
        fifo_A: Stream[Tin, depth][Ct, MG]
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
    lw_s.unroll("og")  # lane
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
