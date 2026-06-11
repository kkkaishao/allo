# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming weight-stationary systolic GEMM **components** for fused dataflow
accelerators.

Unlike :mod:`.ws_direct` (which loads ``A`` and stores ``C`` to DRAM), these
components take activations from an input ``Stream`` (row-major ``[M,K]``, ``L``
lanes) and emit ``C`` to an output ``Stream`` (row-major ``[M,N]``, ``L`` lanes);
only the weight ``B[K,N]`` is DRAM. They are the area-lean drop-in for
fully-unrolled dot-tree matmuls inside a fused layer -- the ``Kt x Nt`` PE array
does a *spatial* reduction (float II=1, no per-output fadd recurrence) and is
**fixed in size regardless of K** (it tiles the contraction), so it scales to
large hidden dims.

``A`` is buffered on-chip (the WS array re-reads it once per N-tile); ``C`` is
collected into an on-chip buffer and re-emitted row-major so every inter-stage
stream shares one simple layout.

:func:`make_weight_stationary_gemm_components` returns the four stage
``(kernel, schedule)`` pairs ``((feed_A, fa_s), (load_W, lw_s), (pe, pe_s),
(collect_C, cc_s))`` to call inside a fused ``top`` and compose.
"""

import allo.exp as allo
from allo.exp import kernel
from allo.exp.lang import Stream, range
from allo.exp.lang.core import DType


def make_weight_stationary_gemm_components(
    Tin, Tacc, M, N, K, Kt, Nt, L, depth=2, ii=1
):
    """Build + schedule the stream-in / stream-out WS-GEMM stage kernels. ``s_A``:
    row-major ``[M,K]`` ``L``-lane input; ``B[K,N]`` DRAM weight; ``s_C``: row-major
    ``[M,N]`` ``L``-lane output. Requires ``N % Nt == 0``, ``K % Kt == 0``,
    ``K % L == 0``, ``N % L == 0``.

    Returns the four ``(kernel, schedule)`` pairs
    ``((feed_A, fa_s), (load_W, lw_s), (pe, pe_s), (collect_C, cc_s))``. Reuse the
    SAME factory for several matmuls -- call each kernel in the fused ``top`` and
    compose its schedule (with the per-call-site repeat-copy id)."""
    assert isinstance(Tin, DType) and isinstance(Tacc, DType)
    assert N % Nt == 0 and K % Kt == 0 and K % L == 0 and N % L == 0
    NT, KT = N // Nt, K // Kt
    KL, NL = K // L, N // L

    @kernel
    def feed_A(s_A: Stream[Tin, depth][L], fifo_A: Stream[Tin, depth][Kt, Nt]):
        Abuf: Tin[M, K]
        for m in range(M, name="ra"):
            for kl in range(KL, name="rk"):
                for l in range(L, name="rl"):  # unrolled
                    Abuf[m, kl * L + l] = s_A[l].get()
        for nt in range(NT):  # re-read A once per N-tile, feed the west edge
            for kt in range(KT):
                for m in range(M, name="fm"):
                    for kk in range(Kt, name="fk"):  # unrolled lane
                        fifo_A[kk, 0].put(Abuf[m, kt * Kt + kk])

    @kernel
    def load_W(B: Tin[K, N], fifo_W: Stream[Tin, depth][Kt, Nt]):
        for nt in range(NT):
            for kt in range(KT, name="wt"):
                for kk in range(Kt, name="wk"):
                    for nn in range(Nt, name="wn"):
                        fifo_W[kk, nn].put(B[kt * Kt + kk, nt * Nt + nn])

    @kernel(mapping=[Kt, Nt])
    def pe(
        fifo_W: Stream[Tin, depth][Kt, Nt],
        fifo_A: Stream[Tin, depth][Kt, Nt],
        fifo_P: Stream[Tacc, depth][Kt, Nt],
        fifo_O: Stream[Tacc, depth][Kt, Nt],
    ):
        kk = allo.get_wid(0)
        nn = allo.get_wid(1)
        for nt in range(NT):
            for kt in range(KT):
                w: Tin = fifo_W[kk, nn].get()
                for m in range(M, name="m"):
                    a: Tin = fifo_A[kk, nn].get()
                    acc: Tacc = a * w
                    if kk > 0:
                        acc = acc + fifo_P[kk, nn].get()
                    if nn < Nt - 1:
                        fifo_A[kk, nn + 1].put(a)
                    if kk < Kt - 1:
                        fifo_P[kk + 1, nn].put(acc)
                    else:
                        fifo_O[kk, nn].put(acc)

    @kernel
    def collect_C(fifo_O: Stream[Tacc, depth][Kt, Nt], s_C: Stream[Tacc, depth][L]):
        Cbuf: Tacc[M, N]
        for nt in range(NT):
            for kt in range(KT):
                for m in range(M, name="cm"):
                    for nn in range(Nt, name="cn"):
                        p: Tacc = fifo_O[Kt - 1, nn].get()
                        acc_val: Tacc = p
                        if kt > 0:
                            acc_val = Cbuf[m, nt * Nt + nn] + p
                        Cbuf[m, nt * Nt + nn] = acc_val
        for m in range(M, name="em"):  # re-emit row-major, L lanes
            for nl in range(NL, name="en"):
                for l in range(L, name="el"):  # unrolled
                    s_C[l].put(Cbuf[m, nl * L + l])

    fa_s = feed_A.schedule()
    fa_s.partition(fa_s.buffer("Abuf"), dim=2, kind=fa_s.Cyclic, factor=L)
    fa_s.unroll("rl")
    fa_s.pipeline(fa_s.flatten(("ra", "rk")), ii=ii)
    fa_s.unroll("fk")
    fa_s.pipeline("fm", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("wk")
    lw_s.unroll("wn")
    lw_s.pipeline("wt", ii=ii)

    pe_s = pe.schedule()
    pe_s.pipeline("m", ii=ii)

    cc_s = collect_C.schedule()
    cc_s.partition(cc_s.buffer("Cbuf"), dim=2, kind=cc_s.Cyclic, factor=L)
    cc_s.unroll("cn")
    cc_s.pipeline("cm", ii=ii)
    cc_s.unroll("el")
    cc_s.pipeline(cc_s.flatten(("em", "en")), ii=ii)

    return (feed_A, fa_s), (load_W, lw_s), (pe, pe_s), (collect_C, cc_s)
