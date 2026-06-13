# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import DType, i32


def make_weight_stationary_depthwise(
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
    depth=2,
    ii=1,
):
    """Build + schedule a weight-stationary **depthwise** conv2d; return
    ``(top, top_s)``.

    Depthwise has no cross-channel reduction, so each output channel is an
    independent ``KH*KW`` FIR. The ``Ct x Mt`` PE grid is used as
    ``P = Ct*Mt`` **channel lanes**: each PE runs the full FIR for one channel
    (its ``KH*KW`` filter resident, no psum/forwarding). ``P`` must divide ``C``;
    ``stride``/``pad``/``dil`` give strided / zero-padded / dilated depthwise.
    ``depth`` is the inter-process FIFO depth.

    Tensor layouts (NHWC, channels innermost so the ``P`` lanes read
    contiguously): ``inp [IH,IW,C]``, ``w [KH,KW,C]``, ``out [OH,OW,C]``.
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
    CT = C // P  # channel-tile count
    KK = KH * KW  # taps per output

    @kernel
    def load_W(w: Tin[KH, KW, C], fifo_W: Stream[Tin, depth][Ct, Mt]):
        # per-channel KH*KW filter; lane (cc,mm) -> channel ct*P + cc*Mt + mm.
        for ct in range(CT):
            for kh in range(KH, name="kh"):
                for kw in range(KW, name="kw"):
                    for cc in range(Ct, name="cc"):  # lane
                        for mm in range(Mt, name="mm"):  # lane
                            fifo_W[cc, mm].put(w[kh, kw, ct * P + cc * Mt + mm])

    @kernel
    def load_A(inp: Tin[IH, IW, C], fifo_A: Stream[Tin, depth][Ct, Mt]):
        # feed each PE its KH*KW window taps per output (direct re-read). The P
        # channel lanes are contiguous in C (NHWC) -> coalesced reads.
        for ct in range(CT):
            for oh in range(OH):
                for ow in range(OW, name="ow"):
                    for kh in range(KH, name="kh"):
                        for kw in range(KW, name="kw"):
                            for cc in range(Ct, name="cc"):  # lane
                                for mm in range(Mt, name="mm"):  # lane
                                    ih: i32 = oh * stride + kh * dil - pad
                                    iw: i32 = ow * stride + kw * dil - pad
                                    a: Tin = 0
                                    if ih >= 0:
                                        if ih < IH:
                                            if iw >= 0:
                                                if iw < IW:
                                                    a = inp[
                                                        ih, iw, ct * P + cc * Mt + mm
                                                    ]
                                    fifo_A[cc, mm].put(a)

    @kernel(mapping=[Ct, Mt])
    def pe(
        fifo_W: Stream[Tin, depth][Ct, Mt],
        fifo_A: Stream[Tin, depth][Ct, Mt],
        fifo_O: Stream[Tacc, depth][Ct, Mt],
    ):
        cc = allo.get_wid(0)
        mm = allo.get_wid(1)
        for ct in range(CT):
            wreg: Tin[KH, KW]  # resident per-channel filter
            for kh in range(KH, name="lwh"):
                for kw in range(KW, name="lww"):
                    wreg[kh, kw] = fifo_W[cc, mm].get()
            # One continuous II=1 pipeline over all (output x tap) steps: a tap
            # counter `t` resets the accumulator and emits the output, so there is
            # no per-output pipeline drain (-> input-bound floor at 1 DSP/PE).
            acc: Tacc = 0
            for f in range(OH * OW * KK, name="f"):
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
        for ct in range(CT):
            for oh in range(OH):
                for ow in range(OW, name="ow"):
                    for cc in range(Ct, name="cc"):  # lane
                        for mm in range(Mt, name="mm"):  # lane
                            out[oh, ow, ct * P + cc * Mt + mm] = fifo_O[cc, mm].get()

    @kernel
    def top(inp: Tin[IH, IW, C], w: Tin[KH, KW, C], out: Tout[OH, OW, C]):
        """Weight-stationary systolic **depthwise** 2D convolution.

        Computes ``out[oh,ow,c] = sum_{kh,kw} inp[oh*S+kh*D-pad, ow*S+kw*D-pad,
        c] * w[kh,kw,c]`` with ``stride``/``pad``/``dil``. With no channel
        reduction the ``Ct x Mt`` grid is used as ``Ct*Mt`` independent channel
        lanes (one ``KH*KW`` FIR per PE, filter resident).

        This **direct** variant re-reads each output's ``KH*KW`` window from DRAM,
        so it is **DRAM-bandwidth-bound** (the PE itself runs at the compute floor,
        but ``load_A`` streams ``KH*KW`` * ``C`` inputs per output position). That
        is the usual story for depthwise -- it is memory-bound; a line-buffered
        variant (read each pixel once, reuse the window on-chip) would reach the
        compute floor. Use this when input bandwidth is ample or memory is tight.

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
    # 1 DSP/PE: pipeline the single flattened (output x tap) loop at II=1. (Do
    # NOT unroll the taps -- the per-tap fifo_A.get() would become 9 reads of one
    # FIFO/cycle, which can't parallelize and just wastes DSP.)
    pe_s.pipeline("f", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("cc")
    lw_s.unroll("mm")
    lw_s.pipeline("kw", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("cc")
    la_s.unroll("mm")
    la_s.pipeline("ow", ii=ii)  # taps unroll inside -> ~floor, no DSP cost

    wc_s = write_C.schedule()
    wc_s.unroll("cc")
    wc_s.unroll("mm")
    wc_s.pipeline("ow", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, wc_s)

    return top, top_s
