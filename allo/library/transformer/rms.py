# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo import kernel
from allo.lang import range
from allo.lang.core import DType
from allo.operators import math as m
from allo.lang.module import Module


def _make(Tin, Tacc, Tout, S, D, L=16, SB=8, eps=1e-5, ii=1):
    DT = D // L
    SBN = S // SB
    invD = 1.0 / D

    @kernel
    def rmsnorm(x: Tin[S, D], g: Tin[D], y: Tout[S, D]):
        """RMSNorm ``y = x * rsqrt(mean(x^2)+eps) * g`` row-wise.

        Two passes over an on-chip row buffer (sum-of-squares, then normalize),
        ``SB`` rows interleaved to hide the sum-of-squares fadd recurrence."""
        for sb in range(SBN):
            buf: Tacc[SB, D]  # SB rows staged on-chip, read once
            ss: Tacc[SB]  # per-row sum of squares
            for s0 in range(SB, name="is"):
                ss[s0] = 0.0
            # pass 1: read SB rows + reduce. The (ct,s) nest is flattened into one
            # continuous pipeline with s fast-varying, so ss[s] is revisited every
            # SB cycles -- interleaving SB rows to hide the fadd recurrence.
            for ct in range(DT, name="rct"):
                for s in range(SB, name="rs"):
                    part: Tacc = 0.0
                    for l in range(L, name="rl"):  # unrolled -> adder tree
                        xv: Tacc = x[sb * SB + s, ct * L + l]
                        buf[s, ct * L + l] = xv
                        part = part + xv * xv
                    ss[s] = ss[s] + part
            inv: Tacc[SB]
            for s2 in range(SB, name="iv"):
                inv[s2] = m.rsqrt(ss[s2] * invD + eps)
            # pass 2: normalize (L lanes/cycle), flattened pipeline
            for s3 in range(SB, name="ns"):
                for dt in range(DT, name="ndt"):
                    for l in range(L, name="nl"):  # unrolled
                        o: Tacc = buf[s3, dt * L + l] * inv[s3] * g[dt * L + l]
                        y[sb * SB + s3, dt * L + l] = o

    s = rmsnorm.schedule()
    s.partition(s.buffer("buf"), dim=2, kind=s.Cyclic, factor=L)
    s.partition(s.buffer("ss"), dim=1, kind=s.Complete)
    s.partition(s.buffer("inv"), dim=1, kind=s.Complete)

    s.unroll("is")
    s.unroll("rl")
    s.pipeline(s.flatten(("rct", "rs")), ii=ii)

    s.unroll("iv")
    s.unroll("nl")
    s.pipeline(s.flatten(("ns", "ndt")), ii=ii)

    return rmsnorm, s


class RMSNorm(Module):
    """**RMSNorm over the hidden dimension (LLaMA pre-norm)**

    Signature: ``rmsnorm(x: Tin[S, D], g: Tin[D], y: Tout[S, D])``.

    Computes ``y[s,d] = x[s,d] * rsqrt(mean_d(x[s,d]^2) + eps) * g[d]`` (LLaMA's
    pre-norm; no mean-subtraction, unlike LayerNorm). Each row needs two passes
    (sum-of-squares, then normalize), so ``SB`` rows are buffered on-chip and read
    once. ``L`` lanes are processed per cycle (``D % L == 0``).

    Notes
    -----
    The sum-of-squares is a loop-carried float add whose recurrence (~fadd
    latency) cannot be pipelined to II=1 by Vitis; processing ``SB`` rows
    interleaved (``S % SB == 0``, ``SB >=`` ~8) overlaps that latency across rows.
    RMSNorm is O(S*D) -- negligible beside the O(S*D^2) projections -- so this
    residual ~II=4 reduction cost is not on the critical path.

    Layout: row-major ``x[S, D]`` (sequence x hidden), weight ``g[D]``, ``y[S, D]``.
    """

    def __init__(self, Tin, Tacc, Tout, S, D, L=16, SB=8, eps=1e-5, ii=1):
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise TypeError("Tin/Tacc/Tout must be Allo DType")
        if D % L != 0:
            raise ValueError("L lanes must tile D")
        if S % SB != 0:
            raise ValueError("SB row-block must tile S")
        top, s = _make(Tin, Tacc, Tout, S, D, L, SB, eps, ii)
        name = f"RMSNorm_S{S}_D{D}_L{L}_SB{SB}"
        super().__init__(name, top, s)


__all__ = ["RMSNorm"]
