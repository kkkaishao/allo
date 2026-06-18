# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal
import allo
from allo import kernel
from allo.lang import range, Module
from allo.lang.core import DType, i8, i32
from allo.operators import math as m

NEG = -1e30


def _make(
    Tin,
    Tacc,
    Tout,
    H,
    Hkv,
    dh,
    Lmax,
    *,
    variant: Literal["vanilla", "grouped", "flash", "flash_int8kv"] = "flash",
    HB=8,
    ii=1,
):
    if variant == "vanilla":
        return _vanilla(Tin, Tacc, Tout, H, Hkv, dh, Lmax, ii)
    if variant == "grouped":
        return _grouped(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB, ii)
    if variant == "flash":
        return _flash(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB, ii)
    if variant == "flash_int8kv":
        return _flash_int8kv(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB, ii)
    raise ValueError(
        f"unknown variant {variant!r}; choose vanilla/grouped/flash/flash_int8kv"
    )


def _vanilla(Tin, Tacc, Tout, H, Hkv, dh, Lmax, ii=1):
    G = H // Hkv
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def top(
        q: Tin[H, dh],
        k_new: Tin[Hkv, dh],
        v_new: Tin[Hkv, dh],
        Kc: Tin[Hkv, Lmax, dh],
        Vc: Tin[Hkv, Lmax, dh],
        O: Tout[H, dh],
        L: i32,
    ):
        pos: i32 = L - 1
        # 1. append new token's k/v at the tail of the cache
        for hk in range(Hkv, name="ak"):
            for d in range(dh, name="ad"):  # unrolled
                Kc[hk, pos, d] = k_new[hk, d]
                Vc[hk, pos, d] = v_new[hk, d]
        # 2. single-query attention, per head
        for h in range(H):
            hk: i32 = h // G
            qh: Tacc[dh]
            for d in range(dh, name="qd"):  # unrolled stage q on-chip
                qh[d] = q[h, d]
            sc: Tacc[Lmax]
            mx: Tacc = NEG
            for j in range(L, name="sj"):
                acc: Tacc = 0.0
                for d in range(dh, name="sd"):  # unrolled -> dot tree, II=1
                    acc = acc + qh[d] * Kc[hk, j, d]
                val: Tacc = acc * scale
                sc[j] = val
                mx = allo.max(mx, val)
            sm: Tacc = 0.0
            for j in range(L, name="ej"):
                e: Tacc = m.exp(sc[j] - mx)
                sc[j] = e
                sm = sm + e
            inv: Tacc = 1.0 / sm
            outr: Tacc[dh]
            for d0 in range(dh, name="o0"):  # unrolled
                outr[d0] = 0.0
            for j in range(L, name="pj"):
                p: Tacc = sc[j] * inv
                for d in range(dh, name="pd"):  # unrolled
                    outr[d] = outr[d] + p * Vc[hk, j, d]
            for d in range(dh, name="ow"):  # unrolled
                O[h, d] = outr[d]

    s = top.schedule()
    s.partition(s.buffer("qh"), dim=1, kind=s.Complete)
    s.partition(s.buffer("outr"), dim=1, kind=s.Complete)

    s.unroll("ad")
    s.pipeline("ak", ii=ii)
    s.unroll("qd")
    s.unroll("sd")
    s.pipeline("sj", ii=ii)
    s.pipeline("ej", ii=ii)
    s.unroll("o0")
    s.unroll("pd")
    s.pipeline("pj", ii=ii)
    s.unroll("ow")

    return top, s


def _grouped(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB=8, ii=1):
    """Internal: ``variant='grouped'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    KB = HB // G  # KV heads covered by one head-block
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def top(
        q: Tin[H, dh],
        k_new: Tin[Hkv, dh],
        v_new: Tin[Hkv, dh],
        Kc: Tin[Hkv, Lmax, dh],
        Vc: Tin[Hkv, Lmax, dh],
        O: Tout[H, dh],
        L: i32,
    ):
        """v2: bandwidth-reusing, head-interleaved decode attention.

        Two structural wins over the vanilla (measured ~7x at H32/Hkv8/dh64/L2048):

        * **Head-block parallelism** -- ``HB`` query heads are computed *inside* each
          pipelined ``j`` loop (unrolled), so the score / softmax / PV trip counts
          stay ``L`` instead of the vanilla's ``H * L`` (heads run serially there).
        * **K/V reuse** -- a head block covers ``KB = HB/G`` consecutive KV heads, so
          each ``Kc[hk,j]`` / ``Vc[hk,j]`` is read **once** and shared by the ``G``
          heads of its group (the vanilla re-reads K/V per query head -> ``G x`` more
          DRAM traffic; decode is bandwidth-bound, so this matters).

        Notes
        -----
        The score loop reaches II~1-2 (dot-tree); the softmax-sum and PV loops stay
        **II~4** -- the loop-carried fadd recurrence over ``j``. This II=4 is a
        *hardware-fundamental* floor (fadd latency), confirmed by isolated synth: a
        single-accumulator, a partial-sum ``acc[j%P]``, AND an HB-way head-interleave
        with ``loop_flatten`` all measure II=4 even reducing from on-chip BRAM. So
        head-interleave buys **parallelism across heads** (the ~7x), not II=1; pushing
        past the floor needs a different structure (e.g. flash-style on-chip tiling +
        a fully-unrolled tree over a constant tile).

        Requires ``H % HB == 0`` and ``HB % G == 0``."""
        pos: i32 = L - 1
        for hk in range(Hkv, name="ak"):
            for d in range(dh, name="ad"):  # unrolled
                Kc[hk, pos, d] = k_new[hk, d]
                Vc[hk, pos, d] = v_new[hk, d]
        for b in range(H // HB):
            qh: Tacc[HB, dh]
            for t in range(HB, name="qt"):
                for d in range(dh, name="qd"):  # unrolled
                    qh[t, d] = q[b * HB + t, d]
            sc: Tacc[HB, Lmax]
            mx: Tacc[HB]
            for t0 in range(HB, name="mi"):  # unrolled
                mx[t0] = NEG
            # scores: read the block's KB KV heads once per j, feed all HB heads
            for j in range(L, name="sj"):
                kj: Tacc[KB, dh]
                for kb in range(KB, name="kb"):  # unrolled
                    for d in range(dh, name="kd"):  # unrolled
                        kj[kb, d] = Kc[b * KB + kb, j, d]
                for t in range(HB, name="st"):  # unrolled over heads
                    acc: Tacc = 0.0
                    for d in range(dh, name="sd"):  # unrolled -> dot tree
                        acc = acc + qh[t, d] * kj[t // G, d]
                    val: Tacc = acc * scale
                    sc[t, j] = val
                    mx[t] = allo.max(mx[t], val)
            sm: Tacc[HB]
            for t1 in range(HB, name="ei"):  # unrolled
                sm[t1] = 0.0
            for j in range(L, name="ej"):
                for t in range(HB, name="et"):  # interleaved over heads -> II=1
                    e: Tacc = m.exp(sc[t, j] - mx[t])
                    sc[t, j] = e
                    sm[t] = sm[t] + e
            inv: Tacc[HB]
            for t2 in range(HB, name="ii_"):  # unrolled
                inv[t2] = 1.0 / sm[t2]
            outr: Tacc[HB, dh]
            for t3 in range(HB, name="oi"):  # unrolled
                for d0 in range(dh, name="oid"):
                    outr[t3, d0] = 0.0
            for j in range(L, name="pj"):
                vj: Tacc[KB, dh]
                for kb in range(KB, name="vb"):  # unrolled
                    for d in range(dh, name="vd"):  # unrolled
                        vj[kb, d] = Vc[b * KB + kb, j, d]
                for t in range(HB, name="pt"):  # interleaved over heads
                    p: Tacc = sc[t, j] * inv[t]
                    for d in range(dh, name="pd"):  # unrolled
                        outr[t, d] = outr[t, d] + p * vj[t // G, d]
            for t in range(HB, name="wt"):
                for d in range(dh, name="wd"):  # unrolled
                    O[b * HB + t, d] = outr[t, d]

    s = top.schedule()
    s.partition(s.buffer("qh"), dim=1, kind=s.Complete)
    s.partition(s.buffer("qh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("sc"), dim=1, kind=s.Complete)
    s.partition(s.buffer("mx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("sm"), dim=1, kind=s.Complete)
    s.partition(s.buffer("inv"), dim=1, kind=s.Complete)
    s.partition(s.buffer("outr"), dim=1, kind=s.Complete)
    s.partition(s.buffer("outr"), dim=2, kind=s.Complete)

    s.unroll("ad")
    s.pipeline("ak", ii=ii)
    s.unroll("qt")
    s.unroll("qd")
    s.unroll("mi")
    s.unroll("kb")
    s.unroll("kd")
    s.unroll("st")
    s.unroll("sd")
    s.pipeline("sj", ii=ii)
    s.unroll("ei")
    s.unroll("et")
    s.pipeline("ej", ii=ii)
    s.unroll("ii_")
    s.unroll("oi")
    s.unroll("oid")
    s.unroll("vb")
    s.unroll("vd")
    s.unroll("pt")
    s.unroll("pd")
    s.pipeline("pj", ii=ii)
    s.unroll("wt")
    s.unroll("wd")

    return top, s


def _flash(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB=8, ii=1):
    """Internal: ``variant='flash'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    KB = HB // G
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def top(
        q: Tin[H, dh],
        k_new: Tin[Hkv, dh],
        v_new: Tin[Hkv, dh],
        Kc: Tin[Hkv, Lmax, dh],
        Vc: Tin[Hkv, Lmax, dh],
        O: Tout[H, dh],
        L: i32,
    ):
        """v3: **flash** decode -- online-softmax single fused pass.

        Notes
        -----
        Keeps v2's head-block parallelism + K/V reuse but FUSES the score / softmax /
        PV passes into ONE streaming pass over ``j``: per cache entry it does the
        ``dh`` dot product, updates the running max ``rmx[t]`` / sum ``rsm[t]`` and
        rescales the running output ``acc[t,d]`` (the flash online-softmax). Vs v2
        this drops the ``sc[HB,Lmax]`` score buffer and one cache traversal; the
        per-``j`` recurrence is longer (mul-add rescale) so the fused loop II is a few,
        but there is only ONE such loop instead of v2's three. Same ``H % HB == 0``,
        ``HB % G == 0``."""
        pos: i32 = L - 1
        for hk in range(Hkv, name="ak"):
            for d in range(dh, name="ad"):  # unrolled
                Kc[hk, pos, d] = k_new[hk, d]
                Vc[hk, pos, d] = v_new[hk, d]
        for b in range(H // HB):
            qh: Tacc[HB, dh]
            for t in range(HB, name="qt"):
                for d in range(dh, name="qd"):  # unrolled
                    qh[t, d] = q[b * HB + t, d]
            rmx: Tacc[HB]
            rsm: Tacc[HB]
            acc: Tacc[HB, dh]
            for t0 in range(HB, name="mi"):  # unrolled init
                rmx[t0] = NEG
                rsm[t0] = 0.0
                for d0 in range(dh, name="ai"):
                    acc[t0, d0] = 0.0
            for j in range(L, name="fj"):  # ONE fused online-softmax pass
                kj: Tacc[KB, dh]
                vj: Tacc[KB, dh]
                for kb in range(KB, name="rb"):  # unrolled
                    for d in range(dh, name="rd"):  # unrolled
                        kj[kb, d] = Kc[b * KB + kb, j, d]
                        vj[kb, d] = Vc[b * KB + kb, j, d]
                for t in range(HB, name="ft"):  # unrolled over heads
                    s: Tacc = 0.0
                    for d in range(dh, name="fd"):  # unrolled -> dot tree
                        s = s + qh[t, d] * kj[t // G, d]
                    s = s * scale
                    mo: Tacc = rmx[t]
                    mn: Tacc = allo.max(mo, s)
                    corr: Tacc = m.exp(mo - mn)  # rescale old stats to new max
                    p: Tacc = m.exp(s - mn)
                    rsm[t] = rsm[t] * corr + p
                    for d in range(dh, name="fad"):  # unrolled
                        acc[t, d] = acc[t, d] * corr + p * vj[t // G, d]
                    rmx[t] = mn
            for t in range(HB, name="wt"):  # normalize + write
                inv: Tacc = 1.0 / rsm[t]
                for d in range(dh, name="wd"):  # unrolled
                    O[b * HB + t, d] = acc[t, d] * inv

    s = top.schedule()
    s.partition(s.buffer("qh"), dim=1, kind=s.Complete)
    s.partition(s.buffer("qh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("rmx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("rsm"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=2, kind=s.Complete)

    s.unroll("ad")
    s.pipeline("ak", ii=ii)
    s.unroll("qt")
    s.unroll("qd")
    s.unroll("mi")
    s.unroll("ai")
    s.unroll("rb")
    s.unroll("rd")
    s.unroll("ft")
    s.unroll("fd")
    s.unroll("fad")
    s.pipeline("fj", ii=ii)
    s.unroll("wt")
    s.unroll("wd")

    return top, s


def _flash_int8kv(Tin, Tacc, Tout, H, Hkv, dh, Lmax, HB=8, ii=1):
    """Internal: ``variant='flash_int8kv'`` impl (routed by :func:`make`). The
    architecture / trade-offs are documented on the returned ``top``."""
    G = H // Hkv
    KB = HB // G
    scale = 1.0 / math.sqrt(dh)

    @kernel
    def top(
        q: Tin[H, dh],
        k_new: Tin[Hkv, dh],
        v_new: Tin[Hkv, dh],
        Kc: i8[Hkv, Lmax, dh],
        Ks: Tin[Hkv, Lmax],
        Vc: i8[Hkv, Lmax, dh],
        Vs: Tin[Hkv, Lmax],
        O: Tout[H, dh],
        L: i32,
    ):
        """**int8-KV** flash decode -- the ``flash`` variant with a quantized cache.

        Notes
        -----
        The K/V cache lives in DRAM as **int8** (``Kc/Vc[Hkv,Lmax,dh]``) with a
        per-(head,token) float scale (``Ks/Vs[Hkv,Lmax]``); this halves KV DRAM vs
        f16 (4x vs f32), and decode is bandwidth-bound, so it is the KV-side
        bandwidth lever (complements weight quant). The new token is quantized on
        append (per-vector ``amax/127`` symmetric int8); the online-softmax pass is
        identical to :func:`_flash` except K/V are read as int8, cast to float, and
        the **per-token scale is folded**: the K scale multiplies the raw score and
        the V scale folds into the softmax weight ``p`` -- so the ``dh`` dot products
        stay float (``q``/``p`` float) with no per-element dequant multiply.
        Measured int8-KV vs full precision: <1% relative output error.

        Same head-block parallelism + K/V reuse as :func:`_flash` (``H % HB == 0``,
        ``HB % G == 0``). Per-token (not per-tensor) scales keep accuracy high at
        negligible cost (one extra DRAM scalar per cached vector)."""
        pos: i32 = L - 1
        # append: quantize the new token's k/v per (head) vector to int8 + scale
        for hk in range(Hkv, name="aq"):
            kam: Tacc = 0.0
            vam: Tacc = 0.0
            for d in range(dh, name="am"):  # unrolled max-abs reduction
                kv: Tacc = k_new[hk, d]
                ka: Tacc = kv
                if kv < 0.0:
                    ka = -kv
                kam = allo.max(kam, ka)
                vv: Tacc = v_new[hk, d]
                va: Tacc = vv
                if vv < 0.0:
                    va = -vv
                vam = allo.max(vam, va)
            Ks[hk, pos] = kam / 127.0
            Vs[hk, pos] = vam / 127.0
            kinv: Tacc = 127.0 / (kam + 1e-9)
            vinv: Tacc = 127.0 / (vam + 1e-9)
            for d in range(dh, name="aw"):  # unrolled quantize + store (round, clip)
                kq: Tacc = k_new[hk, d] * kinv
                if kq >= 0.0:
                    kq = kq + 0.5
                else:
                    kq = kq - 0.5
                kqi: i32 = kq
                if kqi > 127:
                    kqi = 127
                if kqi < -127:
                    kqi = -127
                Kc[hk, pos, d] = kqi
                vq: Tacc = v_new[hk, d] * vinv
                if vq >= 0.0:
                    vq = vq + 0.5
                else:
                    vq = vq - 0.5
                vqi: i32 = vq
                if vqi > 127:
                    vqi = 127
                if vqi < -127:
                    vqi = -127
                Vc[hk, pos, d] = vqi
        # flash decode over the int8 cache (scales folded on load)
        for b in range(H // HB):
            qh: Tacc[HB, dh]
            for t in range(HB, name="qt"):
                for d in range(dh, name="qd"):  # unrolled
                    qh[t, d] = q[b * HB + t, d]
            rmx: Tacc[HB]
            rsm: Tacc[HB]
            acc: Tacc[HB, dh]
            for t0 in range(HB, name="mi"):  # unrolled init
                rmx[t0] = NEG
                rsm[t0] = 0.0
                for d0 in range(dh, name="ai"):
                    acc[t0, d0] = 0.0
            for j in range(L, name="fj"):  # one fused online-softmax pass
                kj: Tacc[KB, dh]
                vj: Tacc[KB, dh]
                ksb: Tacc[KB]
                vsb: Tacc[KB]
                for kb in range(KB, name="rb"):  # unrolled load + cast int8->float
                    ksb[kb] = Ks[b * KB + kb, j]
                    vsb[kb] = Vs[b * KB + kb, j]
                    for d in range(dh, name="rd"):  # unrolled
                        kj[kb, d] = Kc[b * KB + kb, j, d]  # i8 -> Tacc
                        vj[kb, d] = Vc[b * KB + kb, j, d]
                for t in range(HB, name="ft"):  # unrolled over heads
                    sdot: Tacc = 0.0
                    for d in range(dh, name="fd"):  # unrolled -> dot tree
                        sdot = sdot + qh[t, d] * kj[t // G, d]
                    s: Tacc = sdot * scale * ksb[t // G]  # qk scale * K per-token scale
                    mo: Tacc = rmx[t]
                    mn: Tacc = allo.max(mo, s)
                    corr: Tacc = m.exp(mo - mn)
                    p: Tacc = m.exp(s - mn)
                    rsm[t] = rsm[t] * corr + p
                    pv: Tacc = p * vsb[t // G]  # fold V per-token scale into p
                    for d in range(dh, name="fad"):  # unrolled
                        acc[t, d] = acc[t, d] * corr + pv * vj[t // G, d]
                    rmx[t] = mn
            for t in range(HB, name="wt"):  # normalize + write
                inv: Tacc = 1.0 / rsm[t]
                for d in range(dh, name="wd"):  # unrolled
                    O[b * HB + t, d] = acc[t, d] * inv

    s = top.schedule()
    s.partition(s.buffer("qh"), dim=1, kind=s.Complete)
    s.partition(s.buffer("qh"), dim=2, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("kj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=1, kind=s.Complete)
    s.partition(s.buffer("vj"), dim=2, kind=s.Complete)
    s.partition(s.buffer("ksb"), dim=1, kind=s.Complete)
    s.partition(s.buffer("vsb"), dim=1, kind=s.Complete)
    s.partition(s.buffer("rmx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("rsm"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=1, kind=s.Complete)
    s.partition(s.buffer("acc"), dim=2, kind=s.Complete)

    s.unroll("am")
    s.unroll("aw")
    s.pipeline("aq", ii=ii)
    s.unroll("qt")
    s.unroll("qd")
    s.unroll("mi")
    s.unroll("ai")
    s.unroll("rb")
    s.unroll("rd")
    s.unroll("ft")
    s.unroll("fd")
    s.unroll("fad")
    s.pipeline("fj", ii=ii)
    s.unroll("wt")
    s.unroll("wd")

    return top, s


class GQAKVCache(Module):
    """**Single-query KV-cache decode attention (causal GQA over a cached K/V history)**

    Signature -- for quantitized version::

        kvcache(q: Tin[H, dh], k_new: Tin[Hkv, dh], v_new: Tin[Hkv, dh], Kc: i8[Hkv, Lmax, dh],
        Ks: Tin[Hkv, Lmax], Vc: i8[Hkv, Lmax, dh], Vs: Tin[Hkv, Lmax], O: Tout[H, dh], L: i32)

    For full-precision version::

        kvcache(q: Tin[H, dh], k_new: Tin[Hkv, dh], v_new: Tin[Hkv, dh], Kc: Tin[Hkv, Lmax, dh],
        Vc: Tin[Hkv, Lmax, dh], O: Tout[H, dh], L: i32)

    Returns ``(top, top_s)``. ``Lmax`` is the cache capacity; the current length
    ``L`` is a runtime kernel argument.

    Per step the module (1) appends the new token's ``k_new/v_new`` to the cache at
    position ``L-1`` and (2) for each query head ``h`` (sharing KV head
    ``hk = h // (H/Hkv)``)::

        scores[j] = (q[h,:] . Kc[hk,j,:]) / sqrt(dh),   j = 0 .. L-1
        p[:]      = softmax(scores[:])
        out[h,:]  = sum_j p[j] * Vc[hk,j,:]

    Parameters
    ----------
    variant : {"vanilla", "grouped", "flash", "flash_int8kv"}, default "flash"
        Latency, best last:

        * ``vanilla`` -- one head at a time, naive 3-pass; reference only.
        * ``grouped``  -- ``HB`` query heads computed together + K/V reuse across the
          group (~7x over vanilla); reductions still II=4.
        * ``flash``    -- online-softmax single fused pass (no score buffer); ~1.4x
          over ``grouped`` at less DSP/BRAM. **The default.**
        * ``flash_int8kv`` -- ``flash`` with the K/V cache stored **int8** + per-token
          scales (quantized on append, dequant folded on load); halves KV DRAM vs f16
          (4x vs f32) -- the decode bandwidth lever. Extra cache args ``Ks/Vs[Hkv,Lmax]``.

    Notes
    -----
    ``L`` (current cache length, including the new token) is a runtime argument so
    one synthesized kernel serves every decode step. Cache capacity ``Lmax`` is
    compile-time (sizes the DRAM arrays).

    Layout: ``q[H,dh]``, ``k_new/v_new[Hkv,dh]``, caches ``Kc/Vc[Hkv,Lmax,dh]``,
    ``O[H,dh]``; ``H % Hkv == 0``.
    """

    def __init__(
        self,
        Tin,
        Tacc,
        Tout,
        H,
        Hkv,
        dh,
        Lmax,
        *,
        variant: Literal["vanilla", "grouped", "flash", "flash_int8kv"] = "flash",
        HB=8,
        ii=1,
    ):
        if not (
            isinstance(Tin, DType)
            and isinstance(Tacc, DType)
            and isinstance(Tout, DType)
        ):
            raise TypeError("Tin/Tacc/Tout must be DType instances")
        if variant not in {"vanilla", "grouped", "flash", "flash_int8kv"}:
            raise ValueError(
                f"unknown variant {variant!r}; choose vanilla/grouped/flash/flash_int8kv"
            )
        # verify the supported configurations / constraints for each variant
        if H % Hkv != 0:
            raise ValueError("H must be divisible by Hkv")
        G = H // Hkv
        if variant in {"grouped", "flash", "flash_int8kv"}:
            if H % HB != 0 or HB % G != 0:
                raise ValueError(
                    "HB must tile H and be a multiple of G for grouped/flash variants"
                )
        top, s = _make(
            Tin,
            Tacc,
            Tout,
            H,
            Hkv,
            dh,
            Lmax,
            variant=variant,
            HB=HB,
            ii=ii,
        )
        name = f"GQAKVCache_{variant}_H{H}_Hkv{Hkv}_dh{dh}_Lmax{Lmax}"
        super().__init__(name, top, s)


__all__ = ["GQAKVCache"]
