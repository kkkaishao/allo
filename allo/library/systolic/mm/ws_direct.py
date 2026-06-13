# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import DType


def make_direct_weight_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Kt: int, Nt: int, depth=2, ii=1
):
    """Build + schedule the direct weight-stationary GEMM; return ``(top, top_s)``.

    The ``Kt x Nt`` PE array holds a tile of the weight matrix ``B`` resident in
    its PEs; activations ``A`` stream straight from DRAM (no activation buffer) and
    are re-read once per N-tile. ``depth`` is the inter-process FIFO depth; bump it
    if co-simulation reports a stall."""
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert N % Nt == 0 and K % Kt == 0, "array must tile N and K evenly"
    NT, KT = N // Nt, K // Kt  # tile counts

    @kernel
    def load_W(B: Tin[K, N], fifo_W: Stream[Tin, depth][Kt, Nt]):
        # Each B element loaded once: PE(kk,nn) latches B[kt*Kt+kk, nt*Nt+nn].
        for nt in range(NT):
            for kt in range(KT, name="kt"):
                for kk in range(Kt, name="kk"):
                    for nn in range(Nt, name="nn"):
                        fifo_W[kk, nn].put(B[kt * Kt + kk, nt * Nt + nn])

    @kernel
    def load_A(A: Tin[M, K], fifo_A: Stream[Tin, depth][Kt, Nt]):
        # Feed the west edge straight from DRAM; A is re-read once per N-tile.
        for nt in range(NT):
            for kt in range(KT):
                for m in range(M, name="m"):
                    for kk in range(Kt, name="kk"):
                        fifo_A[kk, 0].put(A[m, kt * Kt + kk])

    @kernel(mapping=[Kt, Nt])
    def pe(
        fifo_W: Stream[Tin, depth][Kt, Nt],
        fifo_A: Stream[Tin, depth][Kt, Nt],
        fifo_P: Stream[Tacc, depth][Kt, Nt],
        fifo_O: Stream[Tacc, depth][Kt, Nt],
    ):
        kk = allo.get_wid(0)  # array row = contraction
        nn = allo.get_wid(1)  # array col = output column
        for nt in range(NT):
            for kt in range(KT):
                w: Tin = fifo_W[kk, nn].get()  # latch resident weight
                for m in range(M, name="m"):
                    a: Tin = fifo_A[kk, nn].get()
                    acc: Tacc = a * w
                    if kk > 0:  # add partial sum from the north neighbor
                        acc = acc + fifo_P[kk, nn].get()
                    if nn < Nt - 1:  # forward activation east
                        fifo_A[kk, nn + 1].put(a)
                    if kk < Kt - 1:  # push partial sum south
                        fifo_P[kk + 1, nn].put(acc)
                    else:  # bottom row emits the column's partial sum
                        fifo_O[kk, nn].put(acc)

    @kernel
    def reduce_C(fifo_O: Stream[Tacc, depth][Kt, Nt], fifo_Ct: Stream[Tacc, depth][Nt]):
        # Accumulate the bottom-row partial sums across K-tiles; emit each
        # finalized C tile to a stream so the DRAM write can overlap.
        accC: Tacc[M, Nt]  # per-N-tile partial-C accumulator across K-tiles
        for nt in range(NT):
            for kt in range(KT):
                for m in range(M, name="m"):
                    for nn in range(Nt, name="nn"):
                        p: Tacc = fifo_O[Kt - 1, nn].get()
                        acc_val: Tacc = p
                        if kt > 0:
                            acc_val = accC[m, nn] + p
                        if kt == KT - 1:  # final K-tile -> emit
                            fifo_Ct[nn].put(acc_val)
                        else:
                            accC[m, nn] = acc_val

    @kernel
    def write_C(C: Tout[M, N], fifo_Ct: Stream[Tacc, depth][Nt]):
        for nt in range(NT):
            for m in range(M, name="m"):
                for nn in range(Nt, name="nn"):
                    C[m, nt * Nt + nn] = fifo_Ct[nn].get()

    @kernel
    def top(A: Tin[M, K], B: Tin[K, N], C: Tout[M, N]):
        """Weight-stationary systolic GEMM, **direct** (no activation buffer).

        Computes ``C[M,N] = A[M,K] @ B[K,N]`` (``A`` standard layout, NOT
        transposed). The ``Kt x Nt`` PE array maps to ``(K, N)``: PE(kk,nn) holds
        the resident weight ``B[kt*Kt+kk, nt*Nt+nn]`` (loaded once via ``load_W``).
        Activations ``A`` stream west->east; partial sums reduce **spatially** down
        each column (north->south) via ``fifo_P``; the bottom row emits the column's
        partial sum. Tiles are swept over ``(K/Kt) x (N/Nt)`` and the per-K-tile
        partials are summed across ``K`` in ``reduce_C``, whose finalized tiles
        ``write_C`` drains to DRAM in an **overlapped** dataflow stage (so the
        write-back hides behind the next tile's reduction).

        WS vs OS -- WHY THIS DATAFLOW, ESPECIALLY FOR FLOAT
        ---------------------------------------------------
        The K reduction here is *spatial* (psum chains down the column), so a PE's
        inner ``m`` loop has **no loop-carried accumulator**. For floating point
        this is decisive: OS accumulates ``acc += a*b`` over ``K`` inside the PE --
        a loop-carried ``fadd`` recurrence that forces **II=4** (the adder latency),
        so OS float runs ~4x slower. WS keeps **II=1 for float**, i.e. ~4x lower
        latency than OS float (measured ~3.8x on 128^3/16x16), at the cost of more
        DSP (separate multiply + spatial add instead of a fused FMA). For integer
        both dataflows are at parity (II=1). So prefer WS for latency-critical or
        floating-point GEMM; either works for integer.

        DATA REUSE / DRAM TRAFFIC
        -------------------------
        * weights ``B``: read **once** (resident in PEs, reused across all ``M``) --
          the defining weight-stationary property;
        * activations ``A``: re-read **NT = N/Nt** times (streamed afresh per
          N-tile), straight from DRAM with no on-chip buffer;
        * ``C``: written once.

        WHEN TO USE THIS VARIANT
        ------------------------
        Prefer ``ws_direct`` when the design is compute-bound (wide HBM port hides
        the A re-reads) or when on-chip memory is tight -- it needs only the small
        per-N-tile accumulator, no activation buffer. If A bandwidth dominates
        (narrow port, large N, bandwidth-starved), buffer the activations with
        :mod:`ws_buffered` instead (csynth latency is identical; the trade only
        shows on-board).
        """
        fifo_W: Stream[Tin, depth][Kt, Nt]
        fifo_A: Stream[Tin, depth][Kt, Nt]
        fifo_P: Stream[Tacc, depth][Kt, Nt]
        fifo_O: Stream[Tacc, depth][Kt, Nt]
        fifo_Ct: Stream[Tacc, depth][Nt]
        load_W(B, fifo_W)
        load_A(A, fifo_A)
        pe(fifo_W, fifo_A, fifo_P, fifo_O)
        reduce_C(fifo_O, fifo_Ct)
        write_C(C, fifo_Ct)

    pe_s = pe.schedule()
    pe_s.pipeline("m", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("nn")  # lane
    lw_s.unroll("kk")  # lane
    lw_s.pipeline("kt", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("kk")  # lane -> static stream index
    la_s.pipeline("m", ii=ii)

    rc_s = reduce_C.schedule()
    rc_s.partition(rc_s.buffer("accC"), dim=2, kind=rc_s.Complete)  # column lanes
    rc_s.unroll("nn")
    rc_s.pipeline("m", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("nn")
    wc_s.pipeline("m", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, rc_s, wc_s)

    return top, top_s
