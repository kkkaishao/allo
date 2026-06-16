# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import range
from allo.lang.core import DType
from allo.operators import math as m
from allo.lang import Module

NEG = -1e30  # additive mask "-inf" for stable softmax


def _make(Tin, Tacc, Tout, R, Cn, L, SB, ii):
    CT = Cn // L
    RBN = R // SB

    @kernel
    def top(x: Tin[R, Cn], y: Tout[R, Cn]):
        """Row softmax ``y[r,:] = softmax(x[r,:])``.

        Three passes over an on-chip row buffer (max, exp-sum, normalize);
        ``SB`` rows interleaved to hide the max/sum recurrences. Causal masking
        is applied upstream."""
        for rb in range(RBN):
            buf: Tacc[SB, Cn]
            mx: Tacc[SB]
            for s0 in range(SB, name="im"):
                mx[s0] = NEG
            # pass 1: read SB rows -> buf, max-reduce (s fast-varying -> II hides
            # the fmax latency across the SB interleaved rows).
            for ct in range(CT, name="mct"):
                for s in range(SB, name="ms"):
                    tmax: Tacc = NEG
                    for l in range(L, name="ml"):  # unrolled -> max tree
                        xv: Tacc = x[rb * SB + s, ct * L + l]
                        buf[s, ct * L + l] = xv
                        tmax = allo.max(tmax, xv)
                    mx[s] = allo.max(mx[s], tmax)
            # pass 2: exp(x-mx) -> buf, sum-reduce (same interleave)
            ss: Tacc[SB]
            for s1 in range(SB, name="iss"):
                ss[s1] = 0.0
            for ct in range(CT, name="ect"):
                for s in range(SB, name="es"):
                    part: Tacc = 0.0
                    for l in range(L, name="el"):  # unrolled -> adder tree
                        e: Tacc = m.exp(buf[s, ct * L + l] - mx[s])
                        buf[s, ct * L + l] = e
                        part = part + e
                    ss[s] = ss[s] + part
            inv: Tacc[SB]
            for s2 in range(SB, name="iv"):
                inv[s2] = 1.0 / ss[s2]
            # pass 3: normalize
            for s3 in range(SB, name="ns"):
                for ct in range(CT, name="dct"):
                    for l in range(L, name="dl"):  # unrolled
                        o: Tacc = buf[s3, ct * L + l] * inv[s3]
                        y[rb * SB + s3, ct * L + l] = o

    s = top.schedule()
    s.partition(s.buffer("buf"), dim=2, kind=s.Cyclic, factor=L)
    s.partition(s.buffer("mx"), dim=1, kind=s.Complete)
    s.partition(s.buffer("ss"), dim=1, kind=s.Complete)
    s.partition(s.buffer("inv"), dim=1, kind=s.Complete)

    s.unroll("im")
    s.unroll("ml")
    s.pipeline(s.flatten(("mct", "ms")), ii=ii)

    s.unroll("iss")
    s.unroll("el")
    s.pipeline(s.flatten(("ect", "es")), ii=ii)

    s.unroll("iv")
    s.unroll("dl")
    s.pipeline(s.flatten(("ns", "dct")), ii=ii)

    return top, s


class Softmax(Module):
    """Row-wise softmax.

    Signature: ``softmax(Tin[R, Cn], Tout[R, Cn])``.

    Computes ``y[r,:] = softmax(x[r,:])`` as three passes over an on-chip row
    buffer: (1) read ``x`` -> buf, max-reduce; (2) buf <- ``exp(x-max)``,
    sum-reduce; (3) buf * (1/sum). ``L`` columns are processed per cycle
    (``Cn % L == 0``).

    Notes
    -----
    Both reductions (max, sum) are loop-carried; ``SB`` rows are interleaved
    through the pipeline (``R % SB == 0``, ``SB >=`` ~8) to hide the op latency.
    **Causal masking is applied upstream** -- masked score columns arrive as
    ``-inf`` (a large negative) -- so this is a dense row softmax.

    Layout: row-major ``x[R, Cn]`` (e.g. attention scores: ``R`` queries x ``Cn``
    keys), ``y[R, Cn]``.
    """

    def __init__(self, Tin, Tacc, Tout, R, Cn, L=16, SB=8, ii=1):
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise TypeError("Tin/Tacc/Tout must be Allo DType")
        if Cn % L != 0:
            raise ValueError("Lanes L must tile Cn")
        if R % SB != 0:
            raise ValueError("Row-block SB must tile R")
        top, s = _make(Tin, Tacc, Tout, R, Cn, L, SB, ii)
        name = f"Softmax_R{R}_Cn{Cn}_L{L}_SB{SB}"
        super().__init__(name, top, s)


__all__ = ["Softmax"]
