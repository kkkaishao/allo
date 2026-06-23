# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA-3.2 decoder-layer FPGA accelerator: hardware kernels and schedules."""

import math
from collections import defaultdict
from typing import Literal
import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import f32, i32, i4, i8
from allo.operators import math as m

NEG = -1e30

# Pipeline initiation-interval target
ii = 1

_QUANT_BITS = {"w4a16": 4, "w8a16": 8}


def compose_stages(s, stages):
    """Compose each ``(schedule, name)`` pair; the k-th occurrence of a name gets
    its repeat-copy ``id`` (1st no id, then ``id=str(k)``) so several call-sites of
    one factory's kernel (feed_A/load_W/pe/collect_C) compose correctly."""
    seen = defaultdict(int)
    for sc, nm in stages:
        s.compose(sc) if seen[nm] == 0 else s.compose(sc, id=str(seen[nm]))
        seen[nm] += 1


def compose_schedules(k, parts, unrolls, pipes):
    """A helper to compose the various loop transformations"""
    sc = k.schedule()
    for p in parts:
        b, d = p[0], p[1]
        if len(p) > 2 and p[2] == "cyclic":
            sc.partition(sc.buffer(b), dim=d, kind=sc.Cyclic, factor=p[3])
        else:
            sc.partition(sc.buffer(b), dim=d, kind=sc.Complete)
    for u in unrolls:
        sc.unroll(u)
    for p in pipes:
        sc.pipeline(sc.flatten(p) if isinstance(p, tuple) else p, ii=ii)
    return sc


def os_gemm_components(Tin, Tacc, M, N, K, Mt, Nt, L, depth=2):
    """Streaming **output-stationary** systolic GEMM stage kernels -- a self-contained
    copy of the ``systolic.mm.os`` array adapted to stream ``A`` in / ``C`` out."""
    assert N % Nt == 0 and M % Mt == 0 and K % L == 0 and N % L == 0
    MT, NT = M // Mt, N // Nt
    KL, NL = K // L, N // L

    @kernel
    def feed_A(s_A: Stream[Tin, depth][L], fifo_A: Stream[Tin, depth][Mt, Nt]):
        Abuf: Tin[M, K]
        for mm in range(M, name="ra"):
            for kl in range(KL, name="rk"):
                for l in range(L, name="rl"):  # unrolled
                    Abuf[mm, kl * L + l] = s_A[l].get()
        for mo in range(MT, name="fmo"):
            for no in range(NT, name="fno"):  # re-read A per col-tile
                for k in range(K, name="fk"):
                    for r in range(Mt, name="fr"):  # unrolled lane -> west edge
                        fifo_A[r, 0].put(Abuf[mo * Mt + r, k])

    @kernel
    def load_W(B: Tin[K, N], fifo_W: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MT, name="wmo"):  # re-read B per row-tile
            for no in range(NT, name="wno"):
                for k in range(K, name="wk"):
                    for c in range(Nt, name="wc"):  # unrolled lane -> north edge
                        fifo_W[0, c].put(B[k, no * Nt + c])

    @kernel(mapping=[Mt, Nt])
    def pe(
        fifo_W: Stream[Tin, depth][Mt, Nt],
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_O: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for mo in range(MT):
            for no in range(NT):
                acc: Tacc = 0
                for k in range(K, name="k"):
                    a: Tin = fifo_A[r, c].get()
                    b: Tin = fifo_W[r, c].get()
                    acc += a * b
                    if c < Nt - 1:
                        fifo_A[r, c + 1].put(a)
                    if r < Mt - 1:
                        fifo_W[r + 1, c].put(b)
                fifo_O[r, c].put(acc)

    @kernel
    def collect_C(fifo_O: Stream[Tacc, depth][Mt, Nt], s_C: Stream[Tacc, depth][L]):
        Cbuf: Tacc[M, N]
        for mo in range(MT, name="cmo"):
            for no in range(NT, name="cno"):
                for r in range(Mt, name="cr"):  # unrolled
                    for c in range(Nt, name="cc"):  # unrolled
                        Cbuf[mo * Mt + r, no * Nt + c] = fifo_O[r, c].get()
        for mm in range(M, name="em"):  # re-emit row-major, L lanes
            for nl in range(NL, name="en"):
                for l in range(L, name="el"):  # unrolled
                    s_C[l].put(Cbuf[mm, nl * L + l])

    fa_s = feed_A.schedule()
    fa_s.partition(fa_s.buffer("Abuf"), dim=2, kind=fa_s.Cyclic, factor=L)
    fa_s.partition(fa_s.buffer("Abuf"), dim=1, kind=fa_s.Cyclic, factor=Mt)
    # Full-M partition -> 1-deep banks default to distributed RAM w/ huge addr fanout
    fa_s.bind_storage(fa_s.buffer("Abuf"), impl=fa_s.BRAM, mem_type=fa_s.RAM_T2P)
    fa_s.unroll("rl")
    fa_s.pipeline(fa_s.flatten(("ra", "rk")), ii=ii)
    fa_s.unroll("fr")
    fa_s.pipeline("fk", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("wc")
    lw_s.pipeline("wk", ii=ii)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    cc_s = collect_C.schedule()
    cc_s.partition(cc_s.buffer("Cbuf"), dim=2, kind=cc_s.Cyclic, factor=L)
    cc_s.partition(cc_s.buffer("Cbuf"), dim=1, kind=cc_s.Cyclic, factor=Mt)
    cc_s.bind_storage(
        cc_s.buffer("Cbuf"), impl=cc_s.BRAM, mem_type=cc_s.RAM_T2P
    )  # see feed_A
    cc_s.unroll("cr")
    cc_s.unroll("cc")
    cc_s.pipeline("cno", ii=ii)
    cc_s.unroll("el")
    cc_s.pipeline(cc_s.flatten(("em", "en")), ii=ii)

    return (feed_A, fa_s), (load_W, lw_s), (pe, pe_s), (collect_C, cc_s)


def dequant_os_gemm_components(Tin, Tacc, M, N, K, Mt, Nt, L, Tw, gs, depth=2):
    """Weight-only group-quant version of :func:`_os_gemm_components`."""
    assert Tw.is_int(), "weights must be integer (i4/i8)"
    assert K % gs == 0, "group size must divide K"
    (feed_A, fa_s), _f32_lw, (pe, pe_s), (collect_C, cc_s) = os_gemm_components(
        Tin, Tacc, M, N, K, Mt, Nt, L, depth
    )
    MT, NT, NG = M // Mt, N // Nt, K // gs

    @kernel
    def load_W(
        Wq: Tw[K, N],
        Sc: Tin[NG, N],
        Z: Tw[NG, N],
        fifo_W: Stream[Tin, depth][Mt, Nt],
    ):
        # Dequant each weight as it streams into the array's north edge. Read each
        # group's Nt-wide scale/zero ONCE into a register buffer and reuse it for the
        # gs rows of the group.
        for mo in range(MT, name="wmo"):
            for no in range(NT, name="wno"):
                for grp in range(NG, name="wg"):
                    scb: Tin[Nt]
                    zb: i32[Nt]
                    for c in range(Nt, name="wsc"):  # group scale/zero, read once
                        scb[c] = Sc[grp, no * Nt + c]
                        zb[c] = Z[grp, no * Nt + c]
                    for ki in range(gs, name="wk"):  # gs rows share the group params
                        for c in range(Nt, name="wc"):  # unrolled lane
                            wi: i32 = Wq[grp * gs + ki, no * Nt + c]
                            d: i32 = wi - zb[c]
                            df: Tin = d  # int -> float (dequant)
                            fifo_W[0, c].put(df * scb[c])

    lw_s = load_W.schedule()
    lw_s.partition(lw_s.buffer("scb"), dim=1, kind=lw_s.Complete)
    lw_s.partition(lw_s.buffer("zb"), dim=1, kind=lw_s.Complete)
    lw_s.unroll("wsc")
    lw_s.unroll("wc")
    lw_s.pipeline("wk", ii=ii)
    return (feed_A, fa_s), (load_W, lw_s), (pe, pe_s), (collect_C, cc_s)


def make(
    S,
    H,
    Hkv,
    dh,
    Dff,
    Mt=8,
    Nt=16,
    L=16,
    Br=8,
    eps=1e-5,
    depth=2,
    *,
    variant: Literal["f32", "w4a16", "w8a16"] = "f32",
    group_size=None,
):
    D = H * dh
    Dkv = Hkv * dh
    Nqkv = D + 2 * Dkv  # concatenated QKV width
    N2 = 2 * Dff  # concatenated gate|up width
    dh2 = dh // 2
    G = H // Hkv
    scale = 1.0 / math.sqrt(dh)
    for w in (D, Dkv, Nqkv, N2, Dff, dh2):
        assert w % L == 0, f"{w} % {L}"
    assert Nqkv % Nt == 0 and D % Nt == 0 and N2 % Nt == 0
    assert S % Mt == 0, f"S={S} % Mt={Mt}"
    assert S % Br == 0, f"S={S} % Br={Br}"
    DT, KT2, FT, NT = D // L, Dkv // L, Dff // L, S // Br

    quant = variant in _QUANT_BITS
    if quant:
        assert group_size is not None, f"variant={variant!r} requires group_size"
        gs = group_size
        assert D % gs == 0 and Dff % gs == 0, "group_size must divide D and Dff"
        NGd, NGf = D // gs, Dff // gs  # group counts for K=D / K=Dff projections

    # ---- 4 systolic GEMM cores (each: 4 (kernel,sched)). quant: dequant load_W ----
    if quant:
        Tw = i4 if variant == "w4a16" else i8  # int weight bitwidth from the variant
        mk = dequant_os_gemm_components
        qc = mk(f32, f32, S, Nqkv, D, Mt, Nt, L, Tw, gs, depth)  # h @ [Wq|Wk|Wv]
        oc = mk(f32, f32, S, D, D, Mt, Nt, L, Tw, gs, depth)  # a @ Wo
        gc = mk(f32, f32, S, N2, D, Mt, Nt, L, Tw, gs, depth)  # h2 @ [Wg|Wu]
        dc = mk(f32, f32, S, D, Dff, Mt, Nt, L, Tw, gs, depth)  # hf @ Wd
    else:
        mk = os_gemm_components
        qc = mk(f32, f32, S, Nqkv, D, Mt, Nt, L, depth)  # h @ [Wq|Wk|Wv]
        oc = mk(f32, f32, S, D, D, Mt, Nt, L, depth)  # a @ Wo
        gc = mk(f32, f32, S, N2, D, Mt, Nt, L, depth)  # h2 @ [Wg|Wu]
        dc = mk(f32, f32, S, D, Dff, Mt, Nt, L, depth)  # hf @ Wd
    (qfa, qfa_s), (qlw, qlw_s), (qpe, qpe_s), (qcc, qcc_s) = qc
    (ofa, ofa_s), (olw, olw_s), (ope, ope_s), (occ, occ_s) = oc
    (gfa, gfa_s), (glw, glw_s), (gpe, gpe_s), (gcc, gcc_s) = gc
    (dfa, dfa_s), (dlw, dlw_s), (dpe, dpe_s), (dcc, dcc_s) = dc

    # ---- vector / glue stages ----
    @kernel
    def load_x(x: f32[S, D], s_xn: Stream[f32, depth][L], s_xr1: Stream[f32, depth][L]):
        for s in range(S, name="a0"):
            for dt in range(DT, name="a1"):
                for l in range(L, name="a2"):
                    v: f32 = x[s, dt * L + l]
                    s_xn[l].put(v)
                    s_xr1[l].put(v)

    @kernel
    def norm1(s_xn: Stream[f32, depth][L], g1: f32[D], s_h: Stream[f32, depth][L]):
        for s in range(S, name="b0"):
            row: f32[D]
            # Per-lane partial sum-of-squares -> L independent fadd recurrences, no
            # shared-accumulator feedback mux. A single `ss += v*v` over the unrolled
            # lanes serializes onto one fabric fadd (II=64, 4.4ns path, misses 300MHz).
            psum: f32[L]
            for li in range(L, name="b1i"):
                psum[li] = 0.0
            for dt in range(DT, name="b1"):
                for l in range(L, name="b2"):
                    v: f32 = s_xn[l].get()
                    row[dt * L + l] = v
                    psum[l] = psum[l] + v * v
            ss: f32 = 0.0
            for lr in range(L, name="b1r"):  # combine partials (off the hot path)
                ss = ss + psum[lr]
            inv: f32 = m.rsqrt(ss * (1.0 / D) + eps)
            for dt in range(DT, name="b3"):
                for l in range(L, name="b4"):
                    s_h[l].put(row[dt * L + l] * inv * g1[dt * L + l])

    @kernel
    def qkv_split(
        s_qkv: Stream[f32, depth][L],
        s_Q: Stream[f32, depth][L],
        s_K: Stream[f32, depth][L],
        s_V: Stream[f32, depth][L],
    ):
        # row-major [Q(D) | K(Dkv) | V(Dkv)] -> 3 streams
        for s in range(S, name="q0"):
            for dt in range(DT, name="q1"):
                for l in range(L, name="q2"):
                    s_Q[l].put(s_qkv[l].get())
            for kt in range(KT2, name="q3"):
                for l in range(L, name="q4"):
                    s_K[l].put(s_qkv[l].get())
            for kt in range(KT2, name="q5"):
                for l in range(L, name="q6"):
                    s_V[l].put(s_qkv[l].get())

    @kernel
    def rope_q(
        s_in: Stream[f32, depth][L],
        cos: f32[S, dh2],
        sin: f32[S, dh2],
        s_out: Stream[f32, depth][L],
    ):
        for s in range(S, name="rq0"):
            rb: f32[D]
            for it in range(DT, name="rq1"):
                for l in range(L, name="rq2"):
                    rb[it * L + l] = s_in[l].get()
            ob: f32[D]
            for hh in range(H, name="rq3"):
                for it in range(dh2 // L, name="rq4"):
                    for l in range(L, name="rq5"):
                        i: i32 = it * L + l
                        b: i32 = hh * dh
                        x0: f32 = rb[b + i]
                        x1: f32 = rb[b + i + dh2]
                        c: f32 = cos[s, i]
                        sn: f32 = sin[s, i]
                        ob[b + i] = x0 * c - x1 * sn
                        ob[b + i + dh2] = x1 * c + x0 * sn
            for it in range(DT, name="rq6"):
                for l in range(L, name="rq7"):
                    s_out[l].put(ob[it * L + l])

    @kernel
    def rope_k(
        s_in: Stream[f32, depth][L],
        cos: f32[S, dh2],
        sin: f32[S, dh2],
        s_out: Stream[f32, depth][L],
    ):
        for s in range(S, name="rk0"):
            rb: f32[Dkv]
            for it in range(KT2, name="rk1"):
                for l in range(L, name="rk2"):
                    rb[it * L + l] = s_in[l].get()
            ob: f32[Dkv]
            for hh in range(Hkv, name="rk3"):
                for it in range(dh2 // L, name="rk4"):
                    for l in range(L, name="rk5"):
                        i: i32 = it * L + l
                        b: i32 = hh * dh
                        x0: f32 = rb[b + i]
                        x1: f32 = rb[b + i + dh2]
                        c: f32 = cos[s, i]
                        sn: f32 = sin[s, i]
                        ob[b + i] = x0 * c - x1 * sn
                        ob[b + i + dh2] = x1 * c + x0 * sn
            for it in range(KT2, name="rk6"):
                for l in range(L, name="rk7"):
                    s_out[l].put(ob[it * L + l])

    # Attention as the dual-array FLASH dataflow (qk -> softmax -> pv): row-max
    # folded into qk, normalize folded into pv, so softmax is one streaming pass.
    @kernel
    def attn_qk(
        s_Qr: Stream[f32, depth][L],
        s_Kr: Stream[f32, depth][L],
        s_S: Stream[f32, depth][Br],
        s_M: Stream[f32, depth][Br],
    ):
        Qfull: f32[S, D]
        Kfull: f32[S, Dkv]
        # Drain Q-row THEN K-row per s, matching qkv_split's per-row [Q|K|V]
        # production order.
        for s in range(S, name="qdr0"):
            for dt in range(DT, name="qdr1"):
                for l in range(L, name="qdr2"):
                    Qfull[s, dt * L + l] = s_Qr[l].get()
            for kt in range(KT2, name="kdr1"):
                for l in range(L, name="kdr2"):
                    Kfull[s, kt * L + l] = s_Kr[l].get()
        for h in range(H, name="qh"):
            hk: i32 = h // G
            Kh: f32[S, dh]
            for j in range(S, name="kl"):
                for d in range(dh, name="kd"):
                    Kh[j, d] = Kfull[j, hk * dh + d]
            for it in range(NT, name="qit"):
                Qi: f32[Br, dh]
                for r in range(Br, name="qr"):
                    for d in range(dh, name="qd"):
                        Qi[r, d] = Qfull[it * Br + r, h * dh + d]
                Sq: f32[Br, S]
                mxb: f32[Br]
                for r0 in range(Br, name="qmi"):
                    mxb[r0] = NEG
                for j in range(S, name="qj"):
                    for r in range(Br, name="qsr"):
                        a: f32 = 0.0
                        for d in range(dh, name="qdd"):
                            a = a + Qi[r, d] * Kh[j, d]
                        val: f32 = a * scale
                        if j > it * Br + r:
                            val = NEG
                        Sq[r, j] = val
                        mxb[r] = allo.max(mxb[r], val)
                for r in range(Br, name="qmr"):
                    s_M[r].put(mxb[r])
                for j in range(S, name="qel"):
                    for r in range(Br, name="qer"):
                        s_S[r].put(Sq[r, j])

    @kernel
    def attn_sm(
        s_S: Stream[f32, depth][Br],
        s_M: Stream[f32, depth][Br],
        s_P: Stream[f32, depth][Br],
        s_L: Stream[f32, depth][Br],
    ):
        for h in range(H, name="sh"):
            for it in range(NT, name="sit"):
                mxb: f32[Br]
                for r in range(Br, name="smr"):
                    mxb[r] = s_M[r].get()
                smb: f32[Br]
                for r1 in range(Br, name="sii"):
                    smb[r1] = 0.0
                for j in range(S, name="sej"):
                    for r in range(Br, name="ser"):
                        e: f32 = m.exp(s_S[r].get() - mxb[r])
                        smb[r] = smb[r] + e
                        s_P[r].put(e)
                for r in range(Br, name="slr"):
                    s_L[r].put(smb[r])

    @kernel
    def attn_pv(
        s_P: Stream[f32, depth][Br],
        s_L: Stream[f32, depth][Br],
        s_V: Stream[f32, depth][L],
        s_a: Stream[f32, depth][L],
    ):
        Vfull: f32[S, Dkv]
        for s in range(S, name="vdr0"):
            for kt in range(KT2, name="vdr1"):
                for l in range(L, name="vdr2"):
                    Vfull[s, kt * L + l] = s_V[l].get()
        Ob: f32[S, D]
        for h in range(H, name="ph"):
            hk: i32 = h // G
            Vbh: f32[S, dh]
            for j in range(S, name="vl"):
                for d in range(dh, name="vd"):
                    Vbh[j, d] = Vfull[j, hk * dh + d]
            for it in range(NT, name="pit"):
                acc: f32[Br, dh]
                for r in range(Br, name="pir"):
                    for d in range(dh, name="pid"):
                        acc[r, d] = 0.0
                for j in range(S, name="pj"):
                    pjb: f32[Br]
                    for r in range(Br, name="prr"):
                        pjb[r] = s_P[r].get()
                    for r in range(Br, name="par"):
                        for d in range(dh, name="pad"):
                            acc[r, d] = acc[r, d] + pjb[r] * Vbh[j, d]
                invb: f32[Br]
                for r in range(Br, name="pli"):
                    invb[r] = 1.0 / s_L[r].get()
                for r in range(Br, name="pwr"):
                    for d in range(dh, name="pwd"):
                        Ob[it * Br + r, h * dh + d] = acc[r, d] * invb[r]
        for s in range(S, name="por0"):
            for dt in range(DT, name="por1"):
                for l in range(L, name="por2"):
                    s_a[l].put(Ob[s, dt * L + l])

    @kernel
    def res1(
        s_xr1: Stream[f32, depth][L],
        s_ao: Stream[f32, depth][L],
        s_x1n: Stream[f32, depth][L],
        s_x1r2: Stream[f32, depth][L],
    ):
        Xr: f32[S, D]
        for s in range(S, name="g0"):
            for dt in range(DT, name="g1"):
                for l in range(L, name="g2"):
                    Xr[s, dt * L + l] = s_xr1[l].get()
        for s in range(S, name="g3"):
            for dt in range(DT, name="g4"):
                for l in range(L, name="g5"):
                    v: f32 = Xr[s, dt * L + l] + s_ao[l].get()
                    s_x1n[l].put(v)
                    s_x1r2[l].put(v)

    @kernel
    def norm2(s_x1n: Stream[f32, depth][L], g2: f32[D], s_h2: Stream[f32, depth][L]):
        for s in range(S, name="h0"):
            row: f32[D]
            psum: f32[L]  # per-lane partials -> clean fadd path @300MHz (see norm1)
            for li in range(L, name="h1i"):
                psum[li] = 0.0
            for dt in range(DT, name="h1"):
                for l in range(L, name="h2"):
                    v: f32 = s_x1n[l].get()
                    row[dt * L + l] = v
                    psum[l] = psum[l] + v * v
            ss: f32 = 0.0
            for lr in range(L, name="h1r"):
                ss = ss + psum[lr]
            inv: f32 = m.rsqrt(ss * (1.0 / D) + eps)
            for dt in range(DT, name="h3"):
                for l in range(L, name="h4"):
                    s_h2[l].put(row[dt * L + l] * inv * g2[dt * L + l])

    @kernel
    def swiglu(s_gu: Stream[f32, depth][L], s_hf: Stream[f32, depth][L]):
        for s in range(S, name="w0"):
            gbuf: f32[Dff]
            for fl in range(FT, name="w1"):
                for l in range(L, name="w2"):
                    gbuf[fl * L + l] = s_gu[l].get()
            for fl in range(FT, name="w3"):
                for l in range(L, name="w4"):
                    g: f32 = gbuf[fl * L + l]
                    u: f32 = s_gu[l].get()
                    s_hf[l].put((g * (1.0 / (1.0 + m.exp(-g)))) * u)

    @kernel
    def res2(
        s_x1r2: Stream[f32, depth][L], s_fo: Stream[f32, depth][L], out: f32[S, D]
    ):
        Xr: f32[S, D]
        for s in range(S, name="l0"):
            for dt in range(DT, name="l1"):
                for l in range(L, name="l2"):
                    Xr[s, dt * L + l] = s_x1r2[l].get()
        for s in range(S, name="l3"):
            for dt in range(DT, name="l4"):
                for l in range(L, name="l5"):
                    out[s, dt * L + l] = Xr[s, dt * L + l] + s_fo[l].get()

    # ===== MULTI-CU SPLIT (f32) =====================================================
    # The monolithic `top` (a single 28-process dataflow) is cut into THREE top-level
    # kernels, one per SLR, to break the level-7 routing congestion. The cut-boundary
    # ACTIVATION streams are promoted to top-level args bound to AXI4-Stream (axis), so
    # they cross CUs on-chip via `stream_connect` (NO HBM round-trip). Weights/IO stay
    # m_axi. CU1 isolates attention (the attn_pv BRAM-100% hotspot) in its own SLR.
    #   CU0  load_x->norm1->qkv-GEMM->qkv_split->rope_q/rope_k   out: Qr,Kr,V,xr1
    #   CU1  attn(qk/sm/pv)->o-GEMM->res1->norm2                 in:  Qr,Kr,V,xr1  out: h2,x1r2
    #   CU2  gate/up-GEMM->swiglu->down-GEMM->res2               in:  h2,x1r2
    @kernel
    def cu0(
        x: f32[S, D],
        g1: f32[D],
        Wqkv: f32[D, Nqkv],
        cos_q: f32[S, dh2],
        sin_q: f32[S, dh2],
        cos_k: f32[S, dh2],
        sin_k: f32[S, dh2],
        s_Qr: Stream[f32, depth][L],
        s_Kr: Stream[f32, depth][L],
        s_V: Stream[f32, depth][L],
        s_xr1: Stream[f32, depth][L],
    ):
        """CU0 (SLR0, at HBM): RMSNorm1 + QKV projection + split + RoPE. Emits the
        rotated Q/K, the V, and the residual branch as AXIS streams to CU1."""
        s_xn: Stream[f32, depth][L]
        s_h: Stream[f32, depth][L]
        qW: Stream[f32, depth][Mt, Nt]
        qA: Stream[f32, depth][Mt, Nt]
        qO: Stream[f32, depth][Mt, Nt]
        s_qkv: Stream[f32, depth][L]
        s_Q: Stream[f32, depth][L]
        s_K: Stream[f32, depth][L]
        load_x(x, s_xn, s_xr1)
        norm1(s_xn, g1, s_h)
        qfa(s_h, qA)
        qlw(Wqkv, qW)
        qpe(qW, qA, qO)
        qcc(qO, s_qkv)
        qkv_split(s_qkv, s_Q, s_K, s_V)
        rope_q(s_Q, cos_q, sin_q, s_Qr)
        rope_k(s_K, cos_k, sin_k, s_Kr)

    @kernel
    def cu1(
        Wo: f32[D, D],
        g2: f32[D],
        s_Qr: Stream[f32, depth][L],
        s_Kr: Stream[f32, depth][L],
        s_V: Stream[f32, depth][L],
        s_xr1: Stream[f32, depth][L],
        s_h2: Stream[f32, depth][L],
        s_x1r2: Stream[f32, depth][L],
    ):
        """CU1 (SLR1): flash attention (qk/softmax/pv) + output projection + residual1
        + RMSNorm2. Consumes Q/K/V/residual from CU0; emits normed h2 + residual to CU2.
        """
        s_S: Stream[f32, depth][Br]
        s_M: Stream[f32, depth][Br]
        s_P: Stream[f32, depth][Br]
        s_L: Stream[f32, depth][Br]
        s_a: Stream[f32, depth][L]
        oW: Stream[f32, depth][Mt, Nt]
        oA: Stream[f32, depth][Mt, Nt]
        oO: Stream[f32, depth][Mt, Nt]
        s_ao: Stream[f32, depth][L]
        s_x1n: Stream[f32, depth][L]
        attn_qk(s_Qr, s_Kr, s_S, s_M)
        attn_sm(s_S, s_M, s_P, s_L)
        attn_pv(s_P, s_L, s_V, s_a)
        ofa(s_a, oA)
        olw(Wo, oW)
        ope(oW, oA, oO)
        occ(oO, s_ao)
        res1(s_xr1, s_ao, s_x1n, s_x1r2)
        norm2(s_x1n, g2, s_h2)

    @kernel
    def cu2(
        Wgu: f32[D, N2],
        Wd: f32[Dff, D],
        out: f32[S, D],
        s_h2: Stream[f32, depth][L],
        s_x1r2: Stream[f32, depth][L],
    ):
        """CU2 (SLR2): gate/up projection + SwiGLU + down projection + residual2.
        Consumes normed h2 + residual from CU1; writes the layer output to HBM."""
        gW: Stream[f32, depth][Mt, Nt]
        gA: Stream[f32, depth][Mt, Nt]
        gO: Stream[f32, depth][Mt, Nt]
        s_gu: Stream[f32, depth][L]
        s_hf: Stream[f32, depth][L]
        dW: Stream[f32, depth][Mt, Nt]
        dA: Stream[f32, depth][Mt, Nt]
        dO: Stream[f32, depth][Mt, Nt]
        s_fo: Stream[f32, depth][L]
        gfa(s_h2, gA)
        glw(Wgu, gW)
        gpe(gW, gA, gO)
        gcc(gO, s_gu)
        swiglu(s_gu, s_hf)
        dfa(s_hf, dA)
        dlw(Wd, dW)
        dpe(dW, dA, dO)
        dcc(dO, s_fo)
        res2(s_x1r2, s_fo, out)

    if quant:
        # W4A16 3-CU split: SAME cut as f32, but each projection weight is the triple
        # (Wq int, Sc group-scale, Z zero) and the dequant load_W (qlw/olw/glw/dlw)
        # consumes it. feed_A/pe/collect_C and every vector stage are
        # byte-identical to f32, so the schedules + compose lists are unchanged.
        @kernel
        def cu0(
            x: f32[S, D],
            g1: f32[D],
            Wqkv_q: Tw[D, Nqkv],
            Wqkv_s: f32[NGd, Nqkv],
            Wqkv_z: Tw[NGd, Nqkv],
            cos_q: f32[S, dh2],
            sin_q: f32[S, dh2],
            cos_k: f32[S, dh2],
            sin_k: f32[S, dh2],
            s_Qr: Stream[f32, depth][L],
            s_Kr: Stream[f32, depth][L],
            s_V: Stream[f32, depth][L],
            s_xr1: Stream[f32, depth][L],
        ):
            """CU0 W4A16: RMSNorm1 + dequant QKV projection + split + RoPE."""
            s_xn: Stream[f32, depth][L]
            s_h: Stream[f32, depth][L]
            qW: Stream[f32, depth][Mt, Nt]
            qA: Stream[f32, depth][Mt, Nt]
            qO: Stream[f32, depth][Mt, Nt]
            s_qkv: Stream[f32, depth][L]
            s_Q: Stream[f32, depth][L]
            s_K: Stream[f32, depth][L]
            load_x(x, s_xn, s_xr1)
            norm1(s_xn, g1, s_h)
            qfa(s_h, qA)
            qlw(Wqkv_q, Wqkv_s, Wqkv_z, qW)
            qpe(qW, qA, qO)
            qcc(qO, s_qkv)
            qkv_split(s_qkv, s_Q, s_K, s_V)
            rope_q(s_Q, cos_q, sin_q, s_Qr)
            rope_k(s_K, cos_k, sin_k, s_Kr)

        @kernel
        def cu1(
            Wo_q: Tw[D, D],
            Wo_s: f32[NGd, D],
            Wo_z: Tw[NGd, D],
            g2: f32[D],
            s_Qr: Stream[f32, depth][L],
            s_Kr: Stream[f32, depth][L],
            s_V: Stream[f32, depth][L],
            s_xr1: Stream[f32, depth][L],
            s_h2: Stream[f32, depth][L],
            s_x1r2: Stream[f32, depth][L],
        ):
            """CU1 W4A16: flash attention + dequant O projection + residual1 + RMSNorm2."""
            s_S: Stream[f32, depth][Br]
            s_M: Stream[f32, depth][Br]
            s_P: Stream[f32, depth][Br]
            s_L: Stream[f32, depth][Br]
            s_a: Stream[f32, depth][L]
            oW: Stream[f32, depth][Mt, Nt]
            oA: Stream[f32, depth][Mt, Nt]
            oO: Stream[f32, depth][Mt, Nt]
            s_ao: Stream[f32, depth][L]
            s_x1n: Stream[f32, depth][L]
            attn_qk(s_Qr, s_Kr, s_S, s_M)
            attn_sm(s_S, s_M, s_P, s_L)
            attn_pv(s_P, s_L, s_V, s_a)
            ofa(s_a, oA)
            olw(Wo_q, Wo_s, Wo_z, oW)
            ope(oW, oA, oO)
            occ(oO, s_ao)
            res1(s_xr1, s_ao, s_x1n, s_x1r2)
            norm2(s_x1n, g2, s_h2)

        @kernel
        def cu2(
            Wgu_q: Tw[D, N2],
            Wgu_s: f32[NGd, N2],
            Wgu_z: Tw[NGd, N2],
            Wd_q: Tw[Dff, D],
            Wd_s: f32[NGf, D],
            Wd_z: Tw[NGf, D],
            out: f32[S, D],
            s_h2: Stream[f32, depth][L],
            s_x1r2: Stream[f32, depth][L],
        ):
            """CU2 W4A16: dequant gate/up projection + SwiGLU + dequant down projection
            + residual2; writes the layer output to HBM."""
            gW: Stream[f32, depth][Mt, Nt]
            gA: Stream[f32, depth][Mt, Nt]
            gO: Stream[f32, depth][Mt, Nt]
            s_gu: Stream[f32, depth][L]
            s_hf: Stream[f32, depth][L]
            dW: Stream[f32, depth][Mt, Nt]
            dA: Stream[f32, depth][Mt, Nt]
            dO: Stream[f32, depth][Mt, Nt]
            s_fo: Stream[f32, depth][L]
            gfa(s_h2, gA)
            glw(Wgu_q, Wgu_s, Wgu_z, gW)
            gpe(gW, gA, gO)
            gcc(gO, s_gu)
            swiglu(s_gu, s_hf)
            dfa(s_hf, dA)
            dlw(Wd_q, Wd_s, Wd_z, dW)
            dpe(dW, dA, dO)
            dcc(dO, s_fo)
            res2(s_x1r2, s_fo, out)

    # ---- schedules ----
    lx = compose_schedules(load_x, [], ["a2"], [("a0", "a1")])
    n1 = compose_schedules(
        norm1, [("row", 1, "cyclic", L), ("psum", 1)], ["b1i", "b2", "b4"], ["b1", "b3"]
    )
    qs = compose_schedules(qkv_split, [], ["q2", "q4", "q6"], ["q1", "q3", "q5"])
    # Dimension-aware partitioning: the row/full buffers below (rb/ob, Qfull/Kfull/
    # Vfull/Ob) span D or Dkv. Partition only by the parallelism each access actually needs
    rq = compose_schedules(
        rope_q,
        [("rb", 1, "cyclic", L), ("ob", 1, "cyclic", L)],
        ["rq2", "rq5", "rq7"],
        ["rq1", ("rq3", "rq4"), "rq6"],
    )
    rk = compose_schedules(
        rope_k,
        [("rb", 1, "cyclic", L), ("ob", 1, "cyclic", L)],
        ["rk2", "rk5", "rk7"],
        ["rk1", ("rk3", "rk4"), "rk6"],
    )
    aqk = compose_schedules(
        attn_qk,
        [
            ("Qfull", 2, "cyclic", dh),
            ("Kfull", 2, "cyclic", dh),
            ("Kh", 2),
            ("Qi", 1),
            ("Qi", 2),
            ("Sq", 1),
            ("mxb", 1),
        ],
        # Flatten (qj,qsr)+pipeline -> 1 row * dh-deep dot/cyc instead of a Br*dh-wide
        # MAC array (2584 DSP / 159K LUT); ~free (attn_qk is ~0.2% of layer latency).
        ["qdr2", "kdr2", "kd", "qd", "qmi", "qdd", "qmr", "qer"],
        ["qdr1", "kdr1", "kl", ("qj", "qsr"), "qel"],  # Q/K drained per-row inside qdr0
    )
    asm = compose_schedules(
        attn_sm,
        [("mxb", 1), ("smb", 1)],
        ["smr", "sii", "ser", "slr"],
        ["sej"],
    )
    apv = compose_schedules(
        attn_pv,
        [
            ("Vfull", 2, "cyclic", dh),
            ("Ob", 2, "cyclic", dh),
            ("Vbh", 2),
            ("acc", 1),
            ("acc", 2),
            ("pjb", 1),
            ("invb", 1),
        ],
        # Pipeline the inner `par` (Br) loop, unroll only `pad` (dh).
        ["vdr2", "vd", "pir", "pid", "prr", "pad", "pli", "pwr", "pwd", "por2"],
        [("vdr0", "vdr1"), "vl", "par", ("por0", "por1")],
    )
    r1 = compose_schedules(res1, [], ["g2", "g5"], [("g0", "g1"), ("g3", "g4")])
    n2 = compose_schedules(
        norm2, [("row", 1, "cyclic", L), ("psum", 1)], ["h1i", "h2", "h4"], ["h1", "h3"]
    )  # see norm1
    # gbuf holds one row of gate values while the matching up values stream in.
    sw = compose_schedules(
        swiglu, [("gbuf", 1, "cyclic", L)], ["w2", "w4"], ["w1", "w3"]
    )
    r2 = compose_schedules(res2, [], ["l2", "l5"], [("l0", "l1"), ("l3", "l4")])

    GN = ["feed_A", "load_W", "pe", "collect_C"]
    qsc = [qfa_s, qlw_s, qpe_s, qcc_s]
    osc = [ofa_s, olw_s, ope_s, occ_s]
    gsc = [gfa_s, glw_s, gpe_s, gcc_s]
    dsc = [dfa_s, dlw_s, dpe_s, dcc_s]

    # Each CU is its own dataflow region.
    ts0 = cu0.schedule()
    ts0.dataflow()
    compose_stages(
        ts0,
        [(lx, "load_x"), (n1, "norm1")]
        + list(zip(qsc, GN))
        + [(qs, "qkv_split"), (rq, "rope_q"), (rk, "rope_k")],
    )

    ts1 = cu1.schedule()
    ts1.dataflow()
    compose_stages(
        ts1,
        [(aqk, "attn_qk"), (asm, "attn_sm"), (apv, "attn_pv")]
        + list(zip(osc, GN))
        + [(r1, "res1"), (n2, "norm2")],
    )

    ts2 = cu2.schedule()
    ts2.dataflow()
    compose_stages(
        ts2,
        list(zip(gsc, GN)) + [(sw, "swiglu")] + list(zip(dsc, GN)) + [(r2, "res2")],
    )

    return [(cu0, ts0), (cu1, ts1), (cu2, ts2)]


# =======================================
# Multi-CU build interface
# Allo emits only the per-CU kernel .cpp.
# =======================================

import os

# ARCHITECTURE knobs (not model dims)
DEFAULT_S = 8
MT, NT, L_LANES, BR = 8, 16, 16, 8

# CU -> CU on-chip activation handoffs: (producer, consumer, [stream names]).
LINKS = [("cu0", "cu1", ["qr", "kr", "v", "xr1"]), ("cu1", "cu2", ["h2", "x1r2"])]


def dims(config, S):
    """Derived width parameters from a LlamaConfig + sequence length S."""
    D, Dkv, Dff = config.D, config.Dkv, config.Dff
    return dict(
        S=S, D=D, Dkv=Dkv, Dff=Dff, dh2=config.dh // 2, Nqkv=D + 2 * Dkv, N2=2 * Dff
    )


def _proj_triple(prefix, K, N, NG, dt):
    """The three m_axi ports for one W4A16 projection ``W[K,N]``: the int weight
    ``Wq``, the per-group f32 scale ``Sc[NG,N]`` and the int zero ``Z[NG,N]`` --
    matching the ``(Wq, Sc, Z)`` triple the dequant ``load_W`` consumes."""
    return [
        (f"{prefix}_q", (K, N), "in", dt),
        (f"{prefix}_s", (NG, N), "in", "f32"),
        (f"{prefix}_z", (NG, N), "in", dt),
    ]


def cu_meta(config, S, variant="f32", group_size=None):
    """Per-CU interface: m_axi args ``(name, shape, dir, hbm, dtype)`` come FIRST so
    arg ``i == v{i}`` (Allo emits m_axi args before stream args); then the AXIS stream
    args ``(name, dir)``. The arg order mirrors the ``cu0``/``cu1``/``cu2`` signatures
    in :func:`make`. ``dtype`` is the element type (``"f32"``/``"i8"``/``"i4"``) so the
    harness can size host buffers / inputs."""
    d = dims(config, S)
    S_, D, Dff, dh2 = d["S"], d["D"], d["Dff"], d["dh2"]
    Nqkv, N2 = d["Nqkv"], d["N2"]
    quant = variant in _QUANT_BITS

    if quant:
        assert group_size, f"variant={variant!r} needs group_size"
        dt = f"i{_QUANT_BITS[variant]}"
        NGd, NGf = D // group_size, Dff // group_size
        cu0_w = _proj_triple("Wqkv", D, Nqkv, NGd, dt)
        cu1_w = _proj_triple("Wo", D, D, NGd, dt)
        cu2_w = _proj_triple("Wgu", D, N2, NGd, dt) + _proj_triple(
            "Wd", Dff, D, NGf, dt
        )
    else:
        cu0_w = [("Wqkv", (D, Nqkv), "in", "f32")]
        cu1_w = [("Wo", (D, D), "in", "f32")]
        cu2_w = [("Wgu", (D, N2), "in", "f32"), ("Wd", (Dff, D), "in", "f32")]

    cu0_maxi = (
        [("x", (S_, D), "in", "f32"), ("g1", (D,), "in", "f32")]
        + cu0_w
        + [
            ("cos_q", (S_, dh2), "in", "f32"),
            ("sin_q", (S_, dh2), "in", "f32"),
            ("cos_k", (S_, dh2), "in", "f32"),
            ("sin_k", (S_, dh2), "in", "f32"),
        ]
    )
    cu1_maxi = cu1_w + [("g2", (D,), "in", "f32")]
    cu2_maxi = cu2_w + [("out", (S_, D), "out", "f32")]
    meta = {
        "cu0": {
            "kernel": "cu0",
            "slr": "SLR0",
            "maxi": cu0_maxi,
            "axis": [("qr", "out"), ("kr", "out"), ("v", "out"), ("xr1", "out")],
        },
        "cu1": {
            "kernel": "cu1",
            "slr": "SLR1",
            "maxi": cu1_maxi,
            "axis": [
                ("qr", "in"),
                ("kr", "in"),
                ("v", "in"),
                ("xr1", "in"),
                ("h2", "out"),
                ("x1r2", "out"),
            ],
        },
        "cu2": {
            "kernel": "cu2",
            "slr": "SLR2",
            "maxi": cu2_maxi,
            "axis": [("h2", "in"), ("x1r2", "in")],
        },
    }

    # HBM map: the 4 big int weights on the 4 quadrant heads (0/8/16/24)
    f32_hbm = {"cu0": [0, 1, 2, 3, 4, 5, 6], "cu1": [8, 9], "cu2": [16, 24, 17]}
    quant_hbm = {
        "Wqkv_q": 0,
        "Wqkv_s": 1,
        "Wqkv_z": 2,
        "x": 3,
        "g1": 4,  # Q0
        "cos_q": 5,
        "sin_q": 6,
        "cos_k": 7,
        "Wo_q": 8,
        "Wo_s": 9,
        "Wo_z": 10,
        "g2": 11,
        "sin_k": 12,  # Q1
        "Wgu_q": 16,
        "Wgu_s": 17,
        "Wgu_z": 18,  # Q2
        "Wd_q": 24,
        "Wd_s": 25,
        "Wd_z": 26,
        "out": 27,  # Q3
    }
    for key in ("cu0", "cu1", "cu2"):
        ports = meta[key]["maxi"]
        meta[key]["maxi"] = [
            (name, shape, dr, (quant_hbm[name] if quant else f32_hbm[key][i]), dtp)
            for i, (name, shape, dr, dtp) in enumerate(ports)
        ]
    return meta


def generate_kernel_code(
    config,
    out_dir,
    *,
    S=DEFAULT_S,
    Mt=MT,
    Nt=NT,
    L=L_LANES,
    freq_mhz=300.0,
    variant="f32",
    group_size=None,
):
    """Generate the three CU kernels (``cu0.cpp``/``cu1.cpp``/``cu2.cpp``) + the
    combined ``kernel.h`` into ``out_dir`` and return the :func:`cu_meta` used."""
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    meta = cu_meta(config, S, variant, group_size)
    cus = make(
        S,
        config.H,
        config.Hkv,
        config.dh,
        config.Dff,
        Mt=Mt,
        Nt=Nt,
        L=L,
        Br=BR,
        eps=config.eps,
        variant=variant,
        group_size=group_size,
    )
    assert len(cus) == 3, "make() must return exactly 3 CUs"
    headers = []
    for (_, ts), key in zip(cus, ("cu0", "cu1", "cu2")):
        m = meta[key]
        mod = ts.export("vitis", device="u55c", freq_mhz=freq_mhz)
        for i, (name, _, direction, _, _) in enumerate(m["maxi"]):
            tune = (
                dict(num_read_outstanding=1, max_read_burst_length=2)
                if direction == "out"
                else dict(num_write_outstanding=1, max_write_burst_length=2)
            )
            mod.set_axi(
                i, bundle=name, offset="slave", **tune
            )  # one m_axi port / tensor
        base = len(m["maxi"])
        for j, (name, _) in enumerate(m["axis"]):
            mod.set_axis(base + j, name=name)  # named AXIS port (same on both CUs)
        code = mod.hls_code
        with open(os.path.join(out_dir, f"{key}.cpp"), "w") as f:
            f.write(code)
        sig = next(
            l for l in code.splitlines() if l.startswith('extern "C"') and "{" not in l
        )
        headers.append(sig if sig.endswith(";") else sig + ";")
    with open(os.path.join(out_dir, "kernel.h"), "w") as f:
        f.write("#ifndef KERNEL_H\n#define KERNEL_H\n#include <hls_stream.h>\n\n")
        f.write("\n".join(headers) + "\n\n#endif\n")
    return meta
