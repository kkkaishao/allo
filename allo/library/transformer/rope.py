# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo import kernel
from allo.lang import range
from allo.lang.core import DType, i32
from allo.lang import Module


def _make(Tin, Tout, S, H, dh, L=16, ii=1):
    dh2 = dh // 2
    HD = H * dh
    IT = dh2 // L
    HT = HD // L

    @kernel
    def rope(x: Tin[S, HD], cos: Tin[S, dh2], sin: Tin[S, dh2], y: Tout[S, HD]):
        """RoPE rotate-half.

        Each row staged on-chip, head-dim pairs ``(i,i+dh/2)`` rotated by the
        precomputed ``cos/sin`` tables into a separate write buffer (split
        read/write buffers + flattened ``[S,H*dh]`` layout -> II=1)."""
        for s in range(S, name="s"):
            ib: Tin[HD]  # input s-row (read-only in compute -> 2 reads/bank)
            ob: Tout[HD]  # output s-row (write-only in compute -> 2 writes/bank)
            for it in range(HT, name="li"):
                for l in range(L, name="ll"):  # unrolled -> contiguous burst
                    ib[it * L + l] = x[s, it * L + l]
            for h in range(H, name="h"):
                for it in range(IT, name="ci"):
                    for l in range(L, name="cl"):  # unrolled
                        i: i32 = it * L + l
                        b: i32 = h * dh
                        x0: Tin = ib[b + i]
                        x1: Tin = ib[b + i + dh2]
                        c: Tin = cos[s, i]
                        sn: Tin = sin[s, i]
                        ob[b + i] = x0 * c - x1 * sn
                        ob[b + i + dh2] = x1 * c + x0 * sn
            for it in range(HT, name="si"):
                for l in range(L, name="sl"):  # unrolled -> contiguous burst
                    y[s, it * L + l] = ob[it * L + l]

    sch = rope.schedule()
    sch.partition(sch.buffer("ib"), dim=1, kind=sch.Cyclic, factor=L)
    sch.partition(sch.buffer("ob"), dim=1, kind=sch.Cyclic, factor=L)

    sch.unroll("ll")
    sch.pipeline("li", ii=ii)

    sch.unroll("cl")
    sch.pipeline(sch.flatten(("h", "ci")), ii=ii)

    sch.unroll("sl")
    sch.pipeline("si", ii=ii)

    return rope, sch


class RoPE(Module):
    """**Rotary position embedding (rotate-half)**

    Signature: ``rope(Tin[S, H*dh], cos[S, dh/2], sin[S, dh/2], Tout[S, H*dh])``.

    Rotates head-dim pairs ``(i, i+dh/2)`` by a position angle whose cos/sin are
    precomputed tables ``cos/sin[S, dh/2]`` (shared across heads)::

        out[i]      = x[i]      * cos - x[i+dh2] * sin
        out[i+dh2]  = x[i+dh2]  * cos + x[i]     * sin

    Notes
    -----
    The pair read ``(i, i+dh2)`` is two disjoint spans, which would stop the AXI
    port from widening; so ``x``/``y`` are flattened to ``[S, H*dh]`` (one
    contiguous DRAM span per row -> 512-bit burst) and each ``S``-row is staged
    on-chip. The rotation uses a **separate read-only input buffer and write-only
    output buffer** (not in-place): with cyclic-``L`` partitioning the pair offset
    ``dh2`` is a multiple of ``L``, so ``i`` and ``i+dh2`` share a bank -- an
    in-place RMW would need 4 accesses/bank/cycle, but split read(2)/write(2)
    buffers fit a dual-port BRAM -> II=1. ``L`` lanes/cycle; ``dh`` even,
    ``dh/2 % L == 0``, ``H*dh % L == 0``.

    Layout: ``x[S, H*dh]`` (a no-op view of ``[S, H, dh]``), ``cos/sin[S, dh/2]``,
    ``y[S, H*dh]``. Apply to Q and K after the projection, before attention.
    """

    def __init__(self, Tin, Tout, S, H, dh, L=16, ii=1):
        if not isinstance(Tin, DType) or not isinstance(Tout, DType):
            raise ValueError("RoPE data types must be DType instances")
        if dh % 2 != 0 or (dh // 2) % L != 0 or (H * dh) % L != 0:
            raise ValueError(
                "Invalid RoPE config: dh even, dh/2 % L == 0, H*dh % L == 0"
            )
        top, s = _make(Tin, Tout, S, H, dh, L, ii)
        name = f"RoPE_S{S}_H{H}_dh{dh}_L{L}"
        super().__init__(name, top, s)


__all__ = ["RoPE"]
