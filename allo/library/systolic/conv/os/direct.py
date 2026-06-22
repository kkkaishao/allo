# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import i32


def make_direct_output_stationary_conv2d(
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
    """Factory for an output-stationary systolic **conv2d** (NHWC), im2col-streamed.

    See :func:`conv` for the full algorithm/dataflow docstring.
    """
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    M, N, K = Co, OH * OW, KH * KW * Ci  # GEMM view: C[M,N] = Wt^T @ im2col(X)
    MT, NT = M // Mt, N // Nt  # output-channel / output-pixel tile counts

    @kernel
    def load_W(Wt: Tin[K, Co], fifo_A: Stream[Tin, depth][Mt, Nt]):
        # weights are pre-flattened [K,Co] = [KH*KW*Ci, Co]; identical to GEMM load_A.
        for mo in range(MT):
            for no in range(NT, name="no"):  # re-read W for each pixel-tile
                for k in range(K, name="k"):
                    for r in range(Mt, name="r"):  # lane -> output channel
                        fifo_A[r, 0].put(Wt[k, mo * Mt + r])

    @kernel
    def load_X(X: Tin[IH, IW, Ci], fifo_B: Stream[Tin, depth][Mt, Nt]):
        # im2col on the fly: for each reduction index k = (kh, kw, ci) and output
        # pixel n = (oh, ow), gather X[oh*S+kh-pad, ow*S+kw-pad, ci] (0 if padded).
        for mo in range(MT, name="mo"):  # re-read X for each channel-tile
            for no in range(NT):
                for k in range(K, name="k"):
                    ci: i32 = k % Ci
                    kw: i32 = (k // Ci) % KW
                    kh: i32 = k // (Ci * KW)
                    for c in range(Nt, name="c"):  # lane -> output pixel
                        n: i32 = no * Nt + c
                        oh: i32 = n // OW
                        ow: i32 = n % OW
                        ih: i32 = oh * stride + kh - pad
                        iw: i32 = ow * stride + kw - pad
                        ok = ih >= 0 and ih < IH and iw >= 0 and iw < IW
                        sih: i32 = ih if ih >= 0 and ih < IH else 0
                        siw: i32 = iw if iw >= 0 and iw < IW else 0
                        v: Tin = X[sih, siw, ci]
                        fifo_B[0, c].put(v if ok else 0)

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
                for r in range(Mt, name="r"):
                    for c in range(Nt, name="c"):
                        n: i32 = no * Nt + c
                        Y[n // OW, n % OW, mo * Mt + r] = fifo_C[r, c].get()

    @kernel
    def conv(Wt: Tin[K, Co], X: Tin[IH, IW, Ci], Y: Tout[OH, OW, Co]):
        """Output-stationary systolic **conv2d** (NHWC), **direct / im2col-streamed**.

        Computes ``Y[OH,OW,Co] = conv2d(X[IH,IW,Ci], W)`` with the standard
        GEMM lowering: the reduction window ``K = KH*KW*Ci`` is the contraction
        axis, output channels ``Co`` index one PE-array axis, output pixels
        ``OH*OW`` the other. Weights arrive **pre-flattened and pre-transposed**
        as ``Wt[K, Co]`` (a plain ``W[KH,KW,Ci,Co].reshape(K, Co)`` on the host),
        so the weight loader is byte-for-byte the GEMM ``load_A``.

        Only the **input loader** is conv-specific: it performs **im2col on the
        fly**, decoding each flat ``k`` into ``(kh, kw, ci)`` and each output
        pixel into ``(oh, ow)``, then gathering ``X[oh*S+kh-pad, ow*S+kw-pad,
        ci]`` (zero on the padding halo). Because the feature map is **NHWC**,
        ``ci`` is the contiguous fast axis, so each window tap reads a contiguous
        ``Ci`` run -- friendly to a wide ``m_axi`` burst despite the gather.

        The ``Mt x Nt`` PE grid (channels x pixels) sweeps ``MT x NT`` output
        tiles, accumulating each output element across the full ``K`` inside its
        PE (output-stationary); weights flow east, activations flow south, and
        boundary PEs drop the forwarded value so no drain process is needed --
        all identical to :mod:`mm.os.direct`.

        Loaders stream straight from DRAM with **no on-chip buffer**, re-reading
        each operand once per reuse step (``W`` x``NT``, ``X`` x``MT``). Prefer
        this variant when compute-bound or on-chip memory is tight; use
        :mod:`buffered` to recover the conv halo reuse when DRAM bandwidth bounds.
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
    lx_s.unroll("c")
    lx_s.pipeline("k", ii=ii)

    st_s = store_Y.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = conv.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, lx_s, st_s)

    return conv, top_s
