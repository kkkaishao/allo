# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import allo
from allo import kernel
from allo.lang import Stream, range, Module
from typing import Literal
from allo.lang.core import DType, i32
from allo.operators import math as m

NEG = -1e30  # causal mask additive "-inf"


def _make(
    Tin,
    Tacc,
    Tout,
    S,
    H,
    Hkv,
    dh,
    *,
    variant: Literal["dense", "flash", "flash_dataflow", "systolic_dataflow"] = "flash",
    SB=8,
    Br=8,
    Mt=16,
    Nt=16,
    L=16,
    depth=2,
    ii=1,
):
    if variant == "dense":
        return _dense(Tin, Tacc, Tout, S, H, Hkv, dh, SB, depth, ii)
    if variant == "flash":
        return _flash(Tin, Tacc, Tout, S, H, Hkv, dh, Br, depth, ii)
    if variant == "flash_dataflow":
        return _flash_dataflow(Tin, Tacc, Tout, S, H, Hkv, dh, Br, depth, ii)
    if variant == "systolic_dataflow":
        return _systolic(Tin, Tacc, Tout, S, dh, Mt, Nt, L, depth, ii)
    raise ValueError(
        f"unknown variant {variant!r}; choose dense/flash/flash_dataflow/systolic_dataflow"
    )


def _dense(Tin, Tacc, Tout, S, H, Hkv, dh, SB=8, depth=2, ii=1):
    """Internal: ``variant='dense'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def gqa(
        Q: Tin[S, H, dh], K: Tin[S, Hkv, dh], V: Tin[S, Hkv, dh], O: Tout[S, H, dh]
    ):
        """**prefill multi-head attention** (causal, GQA).

        Per query head ``h`` (sharing KV head ``hk = h // (H/Hkv)`` -- grouped-query
        attention; ``Hkv == H`` is standard MHA)::

            scores[i,j] = (Q[i,h,:] . K[j,hk,:]) / sqrt(dh),  causal: j>i -> -inf
            P[i,:]      = softmax(scores[i,:])
            out[i,h,:]  = sum_j P[i,j] * V[j,hk,:]

        Notes
        -----
        Each head's Q/K/V are staged on-chip (read once; the dh dot-product is then a
        fully-unrolled tree -> QK^T at II=1) and the ``[S,S]`` scores stay on-chip
        (fits for prefill S up to a few hundred). The softmax and the PV reduction are
        loop-carried float ops (~II=4, fadd-bound, the usual limit) interleaved over
        ``SB`` query rows. Heads run sequentially. For large S, a blocked / online
        (flash) formulation with a spatial-reduction PV would cut the softmax + PV
        cost further.

        Layout: ``Q[S, H, dh]``, ``K/V[S, Hkv, dh]`` (apply RoPE to Q/K first),
        ``O[S, H, dh]``; ``H % Hkv == 0``, ``S % SB == 0``."""
        for h in range(H):
            hk: i32 = h // G
            sc: Tacc[S, S]  # on-chip scores / probabilities
            Qh: Tacc[S, dh]  # this (kv-)head's Q/K/V staged on-chip
            Kh: Tacc[S, dh]
            Vh: Tacc[S, dh]
            for i in range(S, name="li"):
                for d in range(dh, name="ld"):  # unrolled
                    Qh[i, d] = Q[i, h, d]
                    Kh[i, d] = K[i, hk, d]
                    Vh[i, d] = V[i, hk, d]
            # 1. QK^T + scale + causal mask (on-chip operands -> dot tree II=1)
            for i in range(S, name="qi"):
                for j in range(S, name="qj"):
                    acc: Tacc = 0.0
                    for d in range(dh, name="qd"):  # unrolled -> dot tree
                        acc = acc + Qh[i, d] * Kh[j, d]
                    val: Tacc = acc * scale
                    if j > i:
                        val = NEG
                    sc[i, j] = val
            # 2. row softmax in place, SB-interleaved over query rows
            mx: Tacc[SB]
            ssm: Tacc[SB]
            for ib in range(S // SB, name="sb"):
                for s0 in range(SB, name="im"):
                    mx[s0] = NEG
                    ssm[s0] = 0.0
                for j in range(S, name="mj"):
                    for s in range(SB, name="ms"):
                        mx[s] = allo.max(mx[s], sc[ib * SB + s, j])
                for j in range(S, name="ej"):
                    for s in range(SB, name="es"):
                        e: Tacc = m.exp(sc[ib * SB + s, j] - mx[s])
                        sc[ib * SB + s, j] = e
                        ssm[s] = ssm[s] + e
                inv: Tacc[SB]
                for s1 in range(SB, name="iv"):
                    inv[s1] = 1.0 / ssm[s1]
                for j in range(S, name="dj"):
                    for s in range(SB, name="ds"):
                        sc[ib * SB + s, j] = sc[ib * SB + s, j] * inv[s]
            # 3. PV: out[i,d] = sum_j P[i,j] V[j,d], SB-interleave over query rows
            for ib in range(S // SB, name="pb"):
                outr: Tacc[SB, dh]
                for s0 in range(SB, name="po"):
                    for d0 in range(dh, name="pod"):
                        outr[s0, d0] = 0.0
                for j in range(S, name="pj"):
                    for s in range(SB, name="ps"):
                        for d in range(dh, name="pd"):  # unrolled
                            outr[s, d] = outr[s, d] + sc[ib * SB + s, j] * Vh[j, d]
                for s in range(SB, name="pws"):
                    for d in range(dh, name="pw"):
                        O[ib * SB + s, h, d] = outr[s, d]

    s = gqa.schedule()
    s.partition(s.buffer("mx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("ssm"), dim=1, kind=s.Complete)
    s.partition(s.buffer("inv"), dim=1, kind=s.Complete)
    s.partition(s.buffer("outr"), dim=1, kind=s.Complete)
    s.partition(s.buffer("outr"), dim=2, kind=s.Complete)
    s.partition(s.buffer("Qh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("Kh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("Vh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("sc"), dim=1, kind=s.Cyclic, factor=SB)

    s.unroll("ld")
    s.pipeline("li", ii=ii)

    s.unroll("qd")
    s.pipeline(s.flatten(("qi", "qj")), ii=ii)

    s.unroll("im")
    s.pipeline(s.flatten(("mj", "ms")), ii=ii)
    s.pipeline(s.flatten(("ej", "es")), ii=ii)

    s.unroll("iv")
    s.pipeline(s.flatten(("dj", "ds")), ii=ii)

    s.unroll("po")
    s.unroll("pod")
    s.unroll("pd")
    s.pipeline(s.flatten(("pj", "ps")), ii=ii)

    s.unroll("pws")
    s.unroll("pw")

    return gqa, s


def _flash(Tin, Tacc, Tout, S, H, Hkv, dh, Br=8, depth=2, ii=1):
    """Internal: ``variant='flash'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def gqa(
        Q: Tin[S, H, dh], K: Tin[S, Hkv, dh], V: Tin[S, Hkv, dh], O: Tout[S, H, dh]
    ):
        """**Flash** prefill attention (causal, GQA): tiled online-softmax; no ``[S,S]``
        score matrix materialized.

        Notes
        -----
        Per query-row tile (``Br`` rows interleaved, sharing one K/V stream) it walks
        all keys ``j`` in ONE fused pass, keeping running max ``rmx[Br]`` / sum
        ``rsm[Br]`` / output ``acc[Br,dh]`` and rescaling on a new max (the flash
        online softmax). vs :func:`_dense` this drops the ``sc[S,S]`` on-chip
        buffer (O(Br*dh) state instead of O(S^2)) and fuses QK^T / softmax / PV into
        one loop -- the basis for the streaming dual-systolic-array form. ``Br`` rows
        interleave the loop-carried float ops (as ``SB`` does in the dense version).

        Each head's K/V is staged on-chip once (read from BRAM in the fused loop,
        reused across all ``S/Br`` query tiles) -- avoids per-``j`` DRAM reads and the
        wide-AXI-port adapter blow-up. Measured S512/H32/Hkv8/dh64: ~10.6x faster than
        :func:`_dense` (29 vs 309 ms) at 248 BRAM (vs the dense ``sc[S,S]``).
        The fused loop is II~7 (online-softmax mul-add rescale recurrence) -- breaking
        that needs the dual-systolic-array form (separate QK^T / PV arrays at II=1
        with the softmax as a bypass pipeline).

        Layout ``Q[S,H,dh]``, ``K/V[S,Hkv,dh]``, ``O[S,H,dh]``; ``H % Hkv == 0``,
        ``S % Br == 0``. Keys are streamed fully (causal mask per row); causal-tile
        pruning is a later optimization."""
        for h in range(H):
            hk: i32 = h // G
            Kb: Tacc[S, dh]  # this head's K/V staged on-chip (read once, reused
            Vb: Tacc[S, dh]  # across all S/Br query tiles -- avoids per-j DRAM reads)
            for j in range(S, name="kl"):
                for d in range(dh, name="kd"):  # unrolled
                    Kb[j, d] = K[j, hk, d]
                    Vb[j, d] = V[j, hk, d]
            for i0 in range(S // Br):
                Qi: Tacc[Br, dh]
                for r in range(Br, name="qr"):
                    for d in range(dh, name="qd"):  # unrolled
                        Qi[r, d] = Q[i0 * Br + r, h, d]
                rmx: Tacc[Br]
                rsm: Tacc[Br]
                acc: Tacc[Br, dh]
                for r in range(Br, name="vi"):  # unrolled init
                    rmx[r] = NEG
                    rsm[r] = 0.0
                    for d in range(dh, name="ci"):
                        acc[r, d] = 0.0
                for j in range(S, name="fj"):  # one fused online-softmax pass
                    kj: Tacc[dh]
                    vj: Tacc[dh]
                    for d in range(dh, name="rd"):  # unrolled stage one key (BRAM)
                        kj[d] = Kb[j, d]
                        vj[d] = Vb[j, d]
                    for r in range(Br, name="fr"):  # interleaved over Br rows
                        sc: Tacc = 0.0
                        for d in range(dh, name="fd"):  # unrolled -> dot tree
                            sc = sc + Qi[r, d] * kj[d]
                        val: Tacc = sc * scale
                        if j > i0 * Br + r:  # causal mask
                            val = NEG
                        mo: Tacc = rmx[r]
                        mn: Tacc = allo.max(mo, val)
                        corr: Tacc = m.exp(mo - mn)
                        p: Tacc = m.exp(val - mn)
                        rsm[r] = rsm[r] * corr + p
                        for d in range(dh, name="fad"):  # unrolled
                            acc[r, d] = acc[r, d] * corr + p * vj[d]
                        rmx[r] = mn
                for r in range(Br, name="wr"):
                    inv: Tacc = 1.0 / rsm[r]
                    for d in range(dh, name="wd"):  # unrolled
                        O[i0 * Br + r, h, d] = acc[r, d] * inv

    s = gqa.schedule()
    s.partition(s.buffer("Qi"), dim=1, kind=s.Complete)
    s.partition(s.buffer("Qi"), dim=2, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("rmx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("rsm"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=2, kind=s.Complete)
    s.partition(s.buffer("Kb"), dim=2, kind=s.Complete)
    s.partition(s.buffer("Vb"), dim=2, kind=s.Complete)

    s.unroll("kd")
    s.pipeline("kl", ii=ii)
    s.unroll("qr")
    s.unroll("qd")
    s.unroll("vi")
    s.unroll("ci")
    s.unroll("rd")
    s.unroll("fr")
    s.unroll("fd")
    s.unroll("fad")
    s.pipeline("fj", ii=ii)
    s.unroll("wr")
    s.unroll("wd")

    return gqa, s


def _flash_dataflow(Tin, Tacc, Tout, S, H, Hkv, dh, Br=8, depth=2, ii=1):
    """Internal: ``variant='flash_dataflow'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    scale = 1.0 / math.sqrt(dh)
    NT = S // Br

    @kernel
    def qk(
        Q: Tin[S, H, dh],
        K: Tin[S, Hkv, dh],
        s_S: Stream[Tacc, depth][Br],
        s_M: Stream[Tacc, depth][Br],
    ):
        for h in range(H):
            hk: i32 = h // G
            Kb: Tacc[S, dh]
            for j in range(S, name="kl"):
                for d in range(dh, name="kd"):  # unrolled
                    Kb[j, d] = K[j, hk, d]
            for it in range(NT):
                Qi: Tacc[Br, dh]
                for r in range(Br, name="qr"):
                    for d in range(dh, name="qd"):  # unrolled
                        Qi[r, d] = Q[it * Br + r, h, d]
                Sq: Tacc[Br, S]  # buffer scores so the row max can be emitted first
                mxb: Tacc[Br]
                for r0 in range(Br, name="qmi"):  # unrolled init
                    mxb[r0] = NEG
                for j in range(S, name="qj"):  # QK^T + causal + running row max
                    for r in range(Br, name="qsr"):  # unrolled over query rows
                        a: Tacc = 0.0
                        for d in range(dh, name="qdd"):  # unrolled -> dot tree
                            a = a + Qi[r, d] * Kb[j, d]
                        val: Tacc = a * scale
                        if j > it * Br + r:  # causal mask (here, once)
                            val = NEG
                        Sq[r, j] = val
                        mxb[r] = allo.max(mxb[r], val)
                for r in range(Br, name="qmr"):  # emit row max, then the scores
                    s_M[r].put(mxb[r])
                for j in range(S, name="qel"):
                    for r in range(Br, name="qer"):  # unrolled
                        s_S[r].put(Sq[r, j])

    @kernel
    def softmax(
        s_S: Stream[Tacc, depth][Br],
        s_M: Stream[Tacc, depth][Br],
        s_P: Stream[Tacc, depth][Br],
        s_L: Stream[Tacc, depth][Br],
    ):
        # one fused exp+sum pass: max comes from qk, normalize is folded into pv.
        for h in range(H):
            for it in range(NT):
                mxb: Tacc[Br]
                for r in range(Br, name="smr"):  # unrolled
                    mxb[r] = s_M[r].get()
                smb: Tacc[Br]
                for r1 in range(Br, name="sii"):  # unrolled init
                    smb[r1] = 0.0
                for j in range(S, name="sej"):  # exp(score-max) + row sum, emit P
                    for r in range(Br, name="ser"):  # unrolled
                        e: Tacc = m.exp(s_S[r].get() - mxb[r])
                        smb[r] = smb[r] + e
                        s_P[r].put(e)
                for r in range(Br, name="slr"):  # unrolled
                    s_L[r].put(smb[r])

    @kernel
    def pv(
        s_P: Stream[Tacc, depth][Br],
        s_L: Stream[Tacc, depth][Br],
        V: Tin[S, Hkv, dh],
        O: Tout[S, H, dh],
    ):
        for h in range(H):
            hk: i32 = h // G
            Vb: Tacc[S, dh]
            for j in range(S, name="vl"):
                for d in range(dh, name="vd"):  # unrolled
                    Vb[j, d] = V[j, hk, d]
            for it in range(NT):
                acc: Tacc[Br, dh]
                for r in range(Br, name="pir"):  # unrolled init
                    for d in range(dh, name="pid"):
                        acc[r, d] = 0.0
                for j in range(S, name="pj"):  # accumulate unnormalized P*V
                    pjb: Tacc[Br]
                    for r in range(Br, name="prr"):  # unrolled
                        pjb[r] = s_P[r].get()
                    for r in range(Br, name="par"):  # unrolled
                        for d in range(dh, name="pad"):  # unrolled
                            acc[r, d] = acc[r, d] + pjb[r] * Vb[j, d]
                invb: Tacc[Br]
                for r in range(Br, name="pli"):  # normalize folded in here
                    invb[r] = 1.0 / s_L[r].get()
                for r in range(Br, name="pwr"):
                    for d in range(dh, name="pwd"):  # unrolled
                        O[it * Br + r, h, d] = acc[r, d] * invb[r]

    @kernel
    def gqa(
        Q: Tin[S, H, dh], K: Tin[S, Hkv, dh], V: Tin[S, Hkv, dh], O: Tout[S, H, dh]
    ):
        """**Dual-array flash** attention as a 3-stage dataflow pipeline (the
        ``QK^T -> softmax -> PV`` architecture).

        Three streaming stages, composed under ``top.dataflow()`` so consecutive
        query tiles overlap across them:

        * ``qk``      -- ``S_i = Q_i K^T`` per ``Br``-query tile (dh dot-tree), applies
          the causal mask and computes the **row max** in-stage, emits max then scores.
        * ``softmax`` -- a SINGLE fused ``exp(score-max) + row-sum`` pass (the bypass
          scalar pipeline), emits unnormalized ``P_i`` and the row sums.
        * ``pv``      -- ``O_i = P_i V`` accumulated over keys, **normalize folded in**
          (divide by the row sum at write-out).

        Notes
        -----
        Folding the max into ``qk`` and the normalize into ``pv`` leaves softmax as one
        pass, so it stops being the pipeline bottleneck (the naive form -- separate
        max / exp-sum / normalize passes -- is ~3x slower). Each stage stages its
        head's K / V on-chip. The GEMMs are still dot-tree / reduction (the systolic-
        array swap is the next step). ``H % Hkv == 0``, ``S % Br == 0``."""
        s_S: Stream[Tacc, depth][Br]
        s_M: Stream[Tacc, depth][Br]
        s_P: Stream[Tacc, depth][Br]
        s_L: Stream[Tacc, depth][Br]
        qk(Q, K, s_S, s_M)
        softmax(s_S, s_M, s_P, s_L)
        pv(s_P, s_L, V, O)

    qk_s = qk.schedule()
    qk_s.partition(qk_s.buffer("Kb"), dim=2, kind=qk_s.Complete)
    qk_s.partition(qk_s.buffer("Qi"), dim=1, kind=qk_s.Complete)
    qk_s.partition(qk_s.buffer("Qi"), dim=2, kind=qk_s.Complete)
    qk_s.partition(qk_s.buffer("Sq"), dim=1, kind=qk_s.Complete)
    qk_s.partition(qk_s.buffer("mxb"), dim=1, kind=qk_s.Complete)
    qk_s.unroll("kd")
    qk_s.pipeline("kl", ii=ii)
    qk_s.unroll("qd")
    qk_s.unroll("qmi")
    qk_s.unroll("qsr")
    qk_s.unroll("qdd")
    qk_s.pipeline("qj", ii=ii)
    qk_s.unroll("qmr")
    qk_s.unroll("qer")
    qk_s.pipeline("qel", ii=ii)

    sm_s = softmax.schedule()
    sm_s.partition(sm_s.buffer("mxb"), dim=1, kind=sm_s.Complete)
    sm_s.partition(sm_s.buffer("smb"), dim=1, kind=sm_s.Complete)
    sm_s.unroll("smr")
    sm_s.unroll("sii")
    sm_s.unroll("ser")
    sm_s.pipeline("sej", ii=ii)
    sm_s.unroll("slr")

    pv_s = pv.schedule()
    pv_s.partition(pv_s.buffer("Vb"), dim=2, kind=pv_s.Complete)
    pv_s.partition(pv_s.buffer("acc"), dim=1, kind=pv_s.Complete)
    pv_s.partition(pv_s.buffer("acc"), dim=2, kind=pv_s.Complete)
    pv_s.partition(pv_s.buffer("pjb"), dim=1, kind=pv_s.Complete)
    pv_s.partition(pv_s.buffer("invb"), dim=1, kind=pv_s.Complete)
    pv_s.unroll("vd")
    pv_s.pipeline("vl", ii=ii)
    pv_s.unroll("pir")
    pv_s.unroll("pid")
    pv_s.unroll("prr")
    pv_s.unroll("par")
    pv_s.unroll("pad")
    pv_s.pipeline("pj", ii=ii)
    pv_s.unroll("pli")
    pv_s.unroll("pwr")
    pv_s.unroll("pwd")

    ts = gqa.schedule()
    ts.dataflow()
    ts.compose(qk_s, sm_s, pv_s)

    return gqa, ts


def _systolic(Tin, Tacc, Tout, S, dh, Mt=16, Nt=16, L=16, depth=2, ii=1):
    """Internal: ``variant='systolic_dataflow'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``gqa``."""
    scale = 1.0 / math.sqrt(dh)
    DL = dh // L
    SL = S // L
    # QK: C[S,S] = Q[S,dh] @ Kt[dh,S]   (M=S, N=S, K=dh)
    MTq, NTq = S // Mt, S // Nt
    # PV: O[S,dh] = P[S,S] @ V[S,dh]     (M=S, N=dh, K=S)
    MTp, NTp = S // Mt, dh // Nt

    # ---- QK array (output-stationary): A=Q (west), B=Kt (north) ----------------
    @kernel
    def load_QA(Q: Tin[S, dh], fifo_A: Stream[Tin, depth][Mt, Nt]):
        Qbuf: Tin[S, dh]  # stage Q once (contiguous), then re-read per N-tile
        for m in range(S, name="qa_m"):
            for k in range(dh, name="qa_k"):
                Qbuf[m, k] = Q[m, k]
        for mo in range(MTq, name="qa_mo"):
            for no in range(NTq, name="qa_no"):  # re-read A per col-tile
                for k in range(dh, name="qa_fk"):
                    for r in range(Mt, name="qa_r"):  # unrolled lane -> west edge
                        fifo_A[r, 0].put(Qbuf[mo * Mt + r, k])

    @kernel
    def load_K(Kt_in: Tin[dh, S], fifo_W: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MTq, name="k_mo"):  # re-read B per row-tile
            for no in range(NTq, name="k_no"):
                for k in range(dh, name="k_k"):
                    for c in range(Nt, name="k_c"):  # unrolled lane -> north edge
                        fifo_W[0, c].put(Kt_in[k, no * Nt + c])

    @kernel(mapping=[Mt, Nt])
    def qk_pe(
        fifo_W: Stream[Tin, depth][Mt, Nt],
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_O: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for mo in range(MTq):
            for no in range(NTq):
                acc: Tacc = 0
                for k in range(dh, name="k"):
                    a: Tin = fifo_A[r, c].get()
                    b: Tin = fifo_W[r, c].get()
                    acc += a * b
                    if c < Nt - 1:
                        fifo_A[r, c + 1].put(a)
                    if r < Mt - 1:
                        fifo_W[r + 1, c].put(b)
                fifo_O[r, c].put(acc)

    # QK collector: buffer scores, apply scale + causal mask, emit the row max
    # (s_M) then the row (s_S) -- folds the softmax max-reduction into this stage.
    @kernel
    def qk_collect(
        fifo_O: Stream[Tacc, depth][Mt, Nt],
        s_M: Stream[Tacc, depth][1],
        s_S: Stream[Tacc, depth][L],
    ):
        Cbuf: Tacc[S, S]
        for mo in range(MTq, name="cmo"):
            for no in range(NTq, name="cno"):
                for r in range(Mt, name="cr"):  # unrolled
                    for c in range(Nt, name="cc"):  # unrolled
                        Cbuf[mo * Mt + r, no * Nt + c] = fifo_O[r, c].get()
        for i in range(S, name="qci"):
            row: Tacc[S]
            mxv: Tacc[L]  # per-lane running max -> L independent recurrences
            for l2 in range(L, name="qmi"):  # unrolled init
                mxv[l2] = NEG
            for cm in range(SL, name="qmc"):
                for l in range(L, name="qml"):  # unrolled
                    v: Tacc = Cbuf[i, cm * L + l] * scale
                    if cm * L + l > i:  # causal mask, once
                        v = NEG
                    row[cm * L + l] = v
                    mxv[l] = allo.max(mxv[l], v)
            mx: Tacc = NEG
            for l3 in range(L, name="qmr"):  # tree reduce (outside the II=1 loop)
                mx = allo.max(mx, mxv[l3])
            s_M[0].put(mx)
            for ce in range(SL, name="qec"):
                for l in range(L, name="qel"):  # unrolled
                    s_S[l].put(row[ce * L + l])

    # softmax: ONE streaming pass -- exp(score - max), partial row-sum, emit
    # unnormalized P. Folding max into qk_collect and 1/sum into pv_collect leaves
    # this single pass (the 3-pass row-serial softmax was the single-head bottleneck).
    @kernel
    def softmax_fold(
        s_M: Stream[Tacc, depth][1],
        s_S: Stream[Tacc, depth][L],
        s_P: Stream[Tin, depth][L],
        s_L: Stream[Tacc, depth][1],
    ):
        for i in range(S, name="si"):
            mx: Tacc = s_M[0].get()
            smv: Tacc[L]
            for l0 in range(L, name="sz"):  # unrolled init
                smv[l0] = 0.0
            for cs in range(SL, name="sc"):
                for l in range(L, name="sl"):  # unrolled
                    e: Tacc = m.exp(s_S[l].get() - mx)
                    smv[l] = smv[l] + e
                    s_P[l].put(e)
            tot: Tacc = 0.0
            for l1 in range(L, name="sr"):  # unrolled tree reduce
                tot = tot + smv[l1]
            s_L[0].put(tot)

    # ---- PV array (output-stationary): A=P (west), B=V (north) -----------------
    @kernel
    def feed_P(s_P: Stream[Tin, depth][L], fifo_A: Stream[Tin, depth][Mt, Nt]):
        Pbuf: Tin[S, S]  # buffer the streamed P[S,S], then re-read per N-tile
        for m in range(S, name="p_m"):
            for kl in range(SL, name="p_kl"):
                for l in range(L, name="p_ll"):  # unrolled
                    Pbuf[m, kl * L + l] = s_P[l].get()
        for mo in range(MTp, name="p_mo"):
            for no in range(NTp, name="p_no"):  # re-read A per col-tile
                for k in range(S, name="p_fk"):
                    for r in range(Mt, name="p_r"):  # unrolled lane -> west edge
                        fifo_A[r, 0].put(Pbuf[mo * Mt + r, k])

    @kernel
    def load_V(V: Tin[S, dh], fifo_W: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MTp, name="v_mo"):  # re-read B per row-tile
            for no in range(NTp, name="v_no"):
                for k in range(S, name="v_k"):
                    for c in range(Nt, name="v_c"):  # unrolled lane -> north edge
                        fifo_W[0, c].put(V[k, no * Nt + c])

    @kernel(mapping=[Mt, Nt])
    def pv_pe(
        fifo_W: Stream[Tin, depth][Mt, Nt],
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_O: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for mo in range(MTp):
            for no in range(NTp):
                acc: Tacc = 0
                for k in range(S, name="k"):
                    a: Tin = fifo_A[r, c].get()
                    b: Tin = fifo_W[r, c].get()
                    acc += a * b
                    if c < Nt - 1:
                        fifo_A[r, c + 1].put(a)
                    if r < Mt - 1:
                        fifo_W[r + 1, c].put(b)
                fifo_O[r, c].put(acc)

    # PV collector: buffer unnormalized O = P @ V, then divide each row by its
    # softmax sum (drained first -- feed_P has already buffered all of P).
    @kernel
    def pv_collect(
        fifo_O: Stream[Tacc, depth][Mt, Nt],
        s_L: Stream[Tacc, depth][1],
        O: Tout[S, dh],
    ):
        Lsum: Tacc[S]
        for i in range(S, name="pli"):
            Lsum[i] = s_L[0].get()
        Cbuf: Tacc[S, dh]
        for mo in range(MTp, name="pmo"):
            for no in range(NTp, name="pno"):
                for r in range(Mt, name="pr"):  # unrolled
                    for c in range(Nt, name="pc"):  # unrolled
                        Cbuf[mo * Mt + r, no * Nt + c] = fifo_O[r, c].get()
        for i in range(S, name="pwi"):
            inv: Tacc = 1.0 / Lsum[i]
            for cw in range(DL, name="pwc"):
                for l in range(L, name="pwl"):  # unrolled
                    O[i, cw * L + l] = Cbuf[i, cw * L + l] * inv

    @kernel
    def gqa(Q: Tin[S, dh], Kt_in: Tin[dh, S], V: Tin[S, dh], O: Tout[S, dh]):
        """**Systolic** dual-array attention (single head), 3-stage dataflow with a
        *folded* streaming softmax: QK^T and PV are both **output-stationary** (``os``)
        systolic GEMM arrays and the softmax is split across the array collectors so it
        stays a single streaming pass.

        QK runs the systolic core as ``Q[S,dh] @ Kt[dh,S] -> scores[S,S]`` (pass K
        pre-transposed to ``[dh,S]``); PV as ``P[S,S] @ V[S,dh] -> O[S,dh]``. The
        softmax is folded: the QK collector emits each row's **max** alongside the
        scaled, causally-masked scores, the softmax stage does one ``exp(score-max) +
        row-sum`` pass emitting unnormalized ``P``, and the PV collector divides by the
        row sum at write-out.

        Notes
        -----
        The PE array is ``Mt x Nt`` -- the DSP knob: ``~Mt*Nt*5`` DSP per array for f32
        (so ~``2*Mt*Nt*5`` for the two arrays), far less than the fully-unrolled
        dot-tree dataflow. Each output-stationary PE accumulates its dot product across
        the contraction ``K`` in a single register, so the f32 accumulate is a
        loop-carried ``fadd`` (~``II=4`` recurrence -- the hardware floor); this is the
        deliberate trade for the **weight-stationary** form, whose spatial reduction
        ran at ``II=1`` but deadlocked in cosim. Both collectors materialize the full
        ``[S,S]`` score / prob matrix (a per-head barrier), so for MHA/GQA heads run
        serially; the dot-tree ``flash_dataflow`` variant streams without an ``[S,S]``
        buffer and is the latency-best form at its DSP point.

        ``dh % L == 0``, ``S % L == 0``, ``S % Mt == 0``, ``S % Nt == 0``,
        ``dh % Nt == 0``, ``Nt <= L``. Single head; wrap a head loop for MHA/GQA."""
        qA: Stream[Tin, depth][Mt, Nt]
        qW: Stream[Tin, depth][Mt, Nt]
        qO: Stream[Tacc, depth][Mt, Nt]
        s_M: Stream[Tacc, depth][1]
        s_S: Stream[Tacc, depth][L]
        s_P: Stream[Tin, depth][L]
        s_L: Stream[Tacc, depth][1]
        pA: Stream[Tin, depth][Mt, Nt]
        pW: Stream[Tin, depth][Mt, Nt]
        pO: Stream[Tacc, depth][Mt, Nt]
        load_QA(Q, qA)
        load_K(Kt_in, qW)
        qk_pe(qW, qA, qO)
        qk_collect(qO, s_M, s_S)
        softmax_fold(s_M, s_S, s_P, s_L)
        feed_P(s_P, pA)
        load_V(V, pW)
        pv_pe(pW, pA, pO)
        pv_collect(pO, s_L, O)

    qa_s = load_QA.schedule()
    qa_s.partition(qa_s.buffer("Qbuf"), dim=1, kind=qa_s.Cyclic, factor=Mt)
    qa_s.pipeline(qa_s.flatten(("qa_m", "qa_k")), ii=ii)
    qa_s.unroll("qa_r")
    qa_s.pipeline("qa_fk", ii=ii)

    lk_s = load_K.schedule()
    lk_s.unroll("k_c")
    lk_s.pipeline("k_k", ii=ii)

    qpe_s = qk_pe.schedule()
    qpe_s.pipeline("k", ii=ii)

    qc_s = qk_collect.schedule()
    qc_s.partition(qc_s.buffer("Cbuf"), dim=2, kind=qc_s.Cyclic, factor=L)
    qc_s.partition(qc_s.buffer("Cbuf"), dim=1, kind=qc_s.Cyclic, factor=Mt)
    qc_s.bind_storage(qc_s.buffer("Cbuf"), impl=qc_s.BRAM, mem_type=qc_s.RAM_T2P)
    qc_s.partition(qc_s.buffer("row"), dim=1, kind=qc_s.Cyclic, factor=L)
    qc_s.partition(qc_s.buffer("mxv"), dim=1, kind=qc_s.Complete)
    qc_s.unroll("cr")
    qc_s.unroll("cc")
    qc_s.pipeline("cno", ii=ii)
    qc_s.unroll("qmi")
    qc_s.unroll("qml")
    qc_s.pipeline("qmc", ii=ii)
    qc_s.unroll("qmr")  # tree reduce, multicycle -> off the critical path
    qc_s.unroll("qel")
    qc_s.pipeline("qec", ii=ii)

    sf_s = softmax_fold.schedule()
    sf_s.partition(sf_s.buffer("smv"), dim=1, kind=sf_s.Complete)
    sf_s.unroll("sz")
    sf_s.unroll("sl")
    sf_s.unroll("sr")
    sf_s.pipeline("sc", ii=ii)

    fp_s = feed_P.schedule()
    fp_s.partition(fp_s.buffer("Pbuf"), dim=2, kind=fp_s.Cyclic, factor=L)
    fp_s.partition(fp_s.buffer("Pbuf"), dim=1, kind=fp_s.Cyclic, factor=Mt)
    fp_s.bind_storage(fp_s.buffer("Pbuf"), impl=fp_s.BRAM, mem_type=fp_s.RAM_T2P)
    fp_s.unroll("p_ll")
    fp_s.pipeline(fp_s.flatten(("p_m", "p_kl")), ii=ii)
    fp_s.unroll("p_r")
    fp_s.pipeline("p_fk", ii=ii)

    lv_s = load_V.schedule()
    lv_s.unroll("v_c")
    lv_s.pipeline("v_k", ii=ii)

    ppe_s = pv_pe.schedule()
    ppe_s.pipeline("k", ii=ii)

    pc_s = pv_collect.schedule()
    pc_s.partition(pc_s.buffer("Cbuf"), dim=2, kind=pc_s.Cyclic, factor=L)
    pc_s.partition(pc_s.buffer("Cbuf"), dim=1, kind=pc_s.Cyclic, factor=Mt)
    pc_s.pipeline("pli", ii=ii)
    pc_s.unroll("pr")
    pc_s.unroll("pc")
    pc_s.pipeline("pno", ii=ii)
    pc_s.unroll("pwl")
    pc_s.pipeline("pwc", ii=ii)

    ts = gqa.schedule()
    ts.dataflow()
    ts.compose(qa_s, lk_s, qpe_s, qc_s, sf_s, fp_s, lv_s, ppe_s, pc_s)

    return gqa, ts


class GQA(Module):
    """**Prefill multi-head attention** (causal, GQA).

    Signature: ``gqa(Q: Tin[S, H, dh], K: Tin[S, Hkv, dh], V: Tin[S, Hkv, dh], O: Tout[S, H, dh])``.

    Per query head ``h`` (sharing KV head ``hk = h // (H/Hkv)`` -- grouped-query
    attention; ``Hkv == H`` is standard MHA)::

        scores[i,j] = (Q[i,h,:] . K[j,hk,:]) / sqrt(dh),  causal: j>i -> -inf
        P[i,:]      = softmax(scores[i,:])
        out[i,h,:]  = sum_j P[i,j] * V[j,hk,:]

    Parameters
    ----------
    variant : {"dense", "flash", "flash_dataflow", "systolic_dataflow"}, default "flash"
        All causal + GQA; measured S512/H32/Hkv8/dh64:

        * ``dense``            -- materializes ``[S,S]`` scores on-chip, heads serial;
          simple, but slow at large S (309 ms) -- the reference.
        * ``flash``            -- tiled online-softmax, no ``[S,S]`` buffer, K/V on-chip
          (29 ms, 10.6x). ``Br`` = query-row tile. **The default.**
        * ``flash_dataflow``   -- the same flash math split into a ``qk -> softmax ->
          pv`` dataflow pipeline (folded stats); fastest dot-tree form (15 ms).
        * ``systolic_dataflow``-- ``qk``/``pv`` as **os systolic arrays** (``Mt``/``Nt``/
          ``L`` size them; DSP ~= 2*Mt*Nt*5) with a *folded* streaming softmax. **Single
          head** (``H``/``Hkv`` ignored), signature ``gqa(Q[S,dh], Kt[dh,S], V[S,dh],
          O[S,dh])`` (pass K pre-transposed). DSP-light + tunable; the f32 PE accumulate
          is a loop-carried fadd (~II=4 -- the os trade for the ws form that deadlocked).
          Still materializes ``[S,S]``, so slower than ``flash_dataflow``.

    Notes
    -----
    Each head's Q/K/V are staged on-chip (read once; the dh dot-product is then a
    fully-unrolled tree -> QK^T at II=1) and the ``[S,S]`` scores stay on-chip
    (fits for prefill S up to a few hundred). The softmax and the PV reduction are
    loop-carried float ops (~II=4, fadd-bound, the usual limit) interleaved over
    ``SB`` query rows. Heads run sequentially. For large S, a blocked / online
    (flash) formulation with a spatial-reduction PV would cut the softmax + PV
    cost further.

    Layout: ``Q[S, H, dh]``, ``K/V[S, Hkv, dh]`` (apply RoPE to Q/K first),
    ``O[S, H, dh]``
    """

    def __init__(
        self,
        Tin,
        Tacc,
        Tout,
        S,
        H,
        Hkv,
        dh,
        *,
        variant: Literal[
            "dense", "flash", "flash_dataflow", "systolic_dataflow"
        ] = "flash",
        SB=8,
        Br=8,
        Mt=16,
        Nt=16,
        L=16,
        depth=2,
        ii=1,
    ):
        if variant not in ("dense", "flash", "flash_dataflow", "systolic_dataflow"):
            raise ValueError(
                f"unsupported variant: {variant}, expected one of 'dense', 'flash', 'flash_dataflow', 'systolic_dataflow'"
            )
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise TypeError("Tin/Tacc/Tout must be Allo DType")
        # verify the config is valid for the chosen variant
        if variant == "dense":
            if H % Hkv != 0:
                raise ValueError(
                    f"H must be divisible by Hkv for dense variant, got H={H}, Hkv={Hkv}"
                )
            if S % SB != 0:
                raise ValueError(
                    f"S must be divisible by SB for dense variant, got S={S}, SB={SB}"
                )
        elif variant in ("flash", "flash_dataflow"):
            if H % Hkv != 0:
                raise ValueError(
                    f"H must be divisible by Hkv for flash variants, got H={H}, Hkv={Hkv}"
                )
            if S % Br != 0:
                raise ValueError(
                    f"S must be divisible by Br for flash variants, got S={S}, Br={Br}"
                )
        else:  # systolic_dataflow: single head; os arrays tile S (M) and S/dh (N)
            for name, val in (("S", S), ("dh", dh)):
                if val % L != 0:
                    raise ValueError(f"{name}={val} must be divisible by L={L}")
            if S % Mt != 0:
                raise ValueError(f"S={S} must be divisible by Mt={Mt}")
            if S % Nt != 0 or dh % Nt != 0:
                raise ValueError(f"S={S} and dh={dh} must be divisible by Nt={Nt}")
            if Nt > L:
                raise ValueError(f"Nt={Nt} must be <= L={L} for the score collectors")

        top, s = _make(
            Tin,
            Tacc,
            Tout,
            S,
            H,
            Hkv,
            dh,
            variant=variant,
            SB=SB,
            Br=Br,
            Mt=Mt,
            Nt=Nt,
            L=L,
            depth=depth,
            ii=ii,
        )
        name = f"GQA_{variant}_S{S}_H{H}_Hkv{Hkv}_dh{dh}"
        super().__init__(name, top, s)


__all__ = ["GQA"]
