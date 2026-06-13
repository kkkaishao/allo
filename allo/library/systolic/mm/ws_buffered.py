# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import DType


def make_buffered_weight_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Kt: int, Nt: int, Mc=None, depth=2, ii=1
):
    """Build + schedule the activation-buffered weight-stationary GEMM; return
    ``(top, top_s)``.

    ``Mc`` is the activation block height (number of ``A``/``C`` rows processed per
    outer block); it bounds the on-chip activation buffer to ``[Mc,K]`` and is the
    buffer-size vs DRAM-traffic knob (see the ``top`` docstring). Must satisfy
    ``M % Mc == 0``; defaults to ``M`` (buffer all activations -- the TPU
    Unified-Buffer style, with both A and weights read once). ``depth`` is the
    inter-process FIFO depth; bump it if co-simulation reports a stall."""
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    ), "Tin/Tacc/Tout must be Allo DType"
    assert N % Nt == 0 and K % Kt == 0, "array must tile N and K evenly"
    Mc = M if Mc is None else Mc
    assert M % Mc == 0, "Mc must divide M"
    NT, KT = N // Nt, K // Kt  # tile counts
    MBLK = M // Mc  # number of activation blocks

    @kernel
    def load_W(B: Tin[K, N], fifo_W: Stream[Tin, depth][Kt, Nt]):
        # PE(kk,nn) latches B[kt*Kt+kk, nt*Nt+nn]; re-streamed once per M-block.
        for mb in range(MBLK):
            for nt in range(NT):
                for kt in range(KT, name="kt"):
                    for kk in range(Kt, name="kk"):
                        for nn in range(Nt, name="nn"):
                            fifo_W[kk, nn].put(B[kt * Kt + kk, nt * Nt + nn])

    @kernel
    def load_A(A: Tin[M, K], fifo_A: Stream[Tin, depth][Kt, Nt]):
        bufA: Tin[Mc, K]  # activation block, read once and replayed across N-tiles
        for mb in range(MBLK):
            for nt in range(NT):  # replay block across N-tiles
                for kt in range(KT):
                    for m in range(Mc, name="m"):
                        for kk in range(Kt, name="kk"):  # lane (west edge)
                            if nt == 0:  # fill once per M-block, contiguous read
                                a: Tin = A[mb * Mc + m, kt * Kt + kk]
                                bufA[m, kt * Kt + kk] = a
                                fifo_A[kk, 0].put(a)
                            else:
                                fifo_A[kk, 0].put(bufA[m, kt * Kt + kk])

    @kernel(mapping=[Kt, Nt])
    def pe(
        fifo_W: Stream[Tin, depth][Kt, Nt],
        fifo_A: Stream[Tin, depth][Kt, Nt],
        fifo_P: Stream[Tacc, depth][Kt, Nt],
        fifo_O: Stream[Tacc, depth][Kt, Nt],
    ):
        kk = allo.get_wid(0)  # array row = contraction
        nn = allo.get_wid(1)  # array col = output column
        for mb in range(MBLK):
            for nt in range(NT):
                for kt in range(KT):
                    w: Tin = fifo_W[kk, nn].get()  # latch resident weight
                    for m in range(Mc, name="m"):
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
        # Accumulate the bottom-row partial sums across K-tiles; emit finalized C
        # tiles to a stream so the DRAM write can overlap.
        accC: Tacc[Mc, Nt]  # per-(M-block, N-tile) accumulator across K-tiles
        for mb in range(MBLK):
            for nt in range(NT):
                for kt in range(KT):
                    for m in range(Mc, name="m"):
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
        for mb in range(MBLK):
            for nt in range(NT):
                for m in range(Mc, name="m"):
                    for nn in range(Nt, name="nn"):
                        C[mb * Mc + m, nt * Nt + nn] = fifo_Ct[nn].get()

    @kernel
    def top(A: Tin[M, K], B: Tin[K, N], C: Tout[M, N]):
        """Weight-stationary systolic GEMM, **activation-buffered** (Unified-Buffer).

        Computes ``C[M,N] = A[M,K] @ B[K,N]`` (``A`` standard layout, NOT
        transposed). The ``Kt x Nt`` PE array maps to ``(K, N)``: PE(kk,nn) holds
        the resident weight ``B[kt*Kt+kk, nt*Nt+nn]``. Activations stream
        west->east; partial sums reduce **spatially** down each column
        (north->south); ``reduce_C`` sums per-K-tile partials and ``write_C`` drains
        them in an **overlapped** stage. Activations are processed in **row-blocks of
        height ``Mc``**: ``load_A`` buffers one ``[Mc,K]`` block on-chip and replays
        it across all ``NT`` N-tiles, so each activation element is read from DRAM
        **once** (the on-chip activation buffer = the TPU Unified Buffer).

        WS vs OS -- WHY THIS DATAFLOW, ESPECIALLY FOR FLOAT
        ---------------------------------------------------
        The K reduction is *spatial* (psum chains down the column), so a PE's inner
        ``m`` loop has **no loop-carried accumulator**. For floating point this is
        decisive: OS accumulates ``acc += a*b`` over ``K`` inside the PE -- a
        loop-carried ``fadd`` recurrence that forces **II=4**, ~4x slower. WS keeps
        **II=1 for float** (~3.8x lower latency than OS float on 128^3/16x16), at the
        cost of more DSP (separate multiply + spatial add vs a fused FMA). Integer
        is at parity (II=1) on both. Prefer WS for latency-critical / float GEMM.

        DATA REUSE / DRAM TRAFFIC
        -------------------------
        * weights ``B``: read once per M-block (resident in PEs);
        * activations ``A``: buffered ``[Mc,K]`` and **read once** (replayed across
          all N-tiles) -- the WS analog of buffering an OS operand;
        * ``C``: written once.

        ``Mc`` -- THE BUFFER-SIZE vs DRAM-TRAFFIC KNOB
        ----------------------------------------------
        * ``Mc = M`` (default): buffer the whole activation matrix ``[M,K]``; BOTH A
          and weights read exactly once (the TPU Unified-Buffer extreme). Best when
          on-chip memory is ample.
        * small ``Mc``: tiny activation buffer ``[Mc,K]`` (and a smaller ``[Mc,Nt]``
          accumulator), but the weights are re-streamed ``M/Mc`` times. Best when
          BRAM/URAM is tight or ``M`` is large.
        This is the same buffer-size vs DRAM-traffic trade as the OS ``Nc`` knob,
        with the roles swapped (WS re-reads activations; OS re-reads an operand
        matrix). csynth latency is ~independent of ``Mc`` (only the buffer /
        DRAM-traffic change); the trade only shows on-board. With no spare on-chip
        memory, use :mod:`ws_direct`.
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
    la_s.partition(la_s.buffer("bufA"), dim=2, kind=la_s.Cyclic, factor=Kt)  # lane (kk)
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
