# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range


def make_buffered_output_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Mt: int, Nt: int, Nc=None, depth=2, ii=1
):
    Nc = N if Nc is None else Nc
    MT = M // Mt  # row-tile count
    NB = Nc // Nt  # column-tiles per block
    NBLK = N // Nc  # number of column-blocks

    @kernel
    def load_A(At: Tin[K, M], fifo_A: Stream[Tin, depth][Mt, Nt]):
        bufA: Tin[Mt, K]  # one row-panel, [lane, k]; refilled per (block, row-tile)
        for nblk in range(NBLK):
            for mo in range(MT):
                for noi in range(NB, name="noi"):  # replay across block's columns
                    for k in range(K, name="k"):
                        for r in range(Mt, name="r"):  # lane
                            if noi == 0:  # fill once per (block, row-tile)
                                a: Tin = At[k, mo * Mt + r]
                                bufA[r, k] = a
                                fifo_A[r, 0].put(a)
                            else:
                                fifo_A[r, 0].put(bufA[r, k])

    @kernel
    def load_B(B: Tin[K, N], fifo_B: Stream[Tin, depth][Mt, Nt]):
        bufB: Tin[K, Nc]  # one B column-block [K,Nc], reused across the mo loop
        for nblk in range(NBLK):
            for mo in range(MT, name="mo"):
                for noi in range(NB, name="noi"):
                    for k in range(K, name="k"):
                        for c in range(Nt, name="c"):  # lane
                            if mo == 0:  # fill once per block
                                b: Tin = B[k, nblk * Nc + noi * Nt + c]
                                bufB[k, noi * Nt + c] = b
                                fifo_B[0, c].put(b)
                            else:
                                fifo_B[0, c].put(bufB[k, noi * Nt + c])

    @kernel(mapping=[Mt, Nt])
    def pe(
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_B: Stream[Tin, depth][Mt, Nt],
        fifo_C: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for nblk in range(NBLK):
            for mo in range(MT):
                for noi in range(NB):
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
    def store_C(C: Tout[M, N], fifo_C: Stream[Tacc, depth][Mt, Nt]):
        for nblk in range(NBLK):
            for mo in range(MT, name="mo"):
                for noi in range(NB, name="noi"):
                    for r in range(Mt, name="r"):
                        for c in range(Nt, name="c"):
                            C[mo * Mt + r, nblk * Nc + noi * Nt + c] = fifo_C[
                                r, c
                            ].get()

    @kernel
    def top(At: Tin[K, M], B: Tin[K, N], C: Tout[M, N]):
        """Output-stationary systolic GEMM, **N-blocked on-chip-buffered** loaders.

        Computes ``C[M,N] = AT^T @ B`` where ``AT`` is the **pre-transposed** ``A``
        (shape ``[K,M]``), so each loader reads its operand contiguously along the
        fast axis. The output is swept in **column-blocks of width ``Nc``**; within
        a block an ``Mt x Nt`` PE grid sweeps ``(M/Mt) x (Nc/Nt)`` output tiles,
        accumulating each output element across the full ``K`` inside its PE
        (output-stationary). ``A`` flows east, ``B`` flows south; boundary PEs drop
        the forwarded value, so no drain process is needed.

        This is the realistic *blocked* dataflow used by real accelerators
        (DRAM -> tile-sized on-chip buffer -> PE registers): on-chip buffers are
        bounded by the **block**, not the full matrix.

        * ``load_B`` buffers one B column-block ``[K,Nc]`` and replays it across the
          ``MT`` row-tiles -> ``B`` read from DRAM exactly once; buffer bounded by
          ``Nc`` (NOT the full ``N``).
        * ``load_A`` buffers the current ``A`` row-panel ``[Mt,K]`` (bounded by the
          array, independent of ``M``/``N``) and replays it across the block's
          ``NB = Nc/Nt`` columns. ``A`` is re-read once per column-block, i.e.
          ``N/Nc`` times total.

        Both use *inline fill* (fill on the first replay pass via ``if noi == 0`` /
        ``if mo == 0``), so loader latency stays at the streaming floor -- no
        separate fill phase.

        ``Nc`` -- THE BUFFER-SIZE vs DRAM-TRAFFIC KNOB
        ----------------------------------------------
        * large ``Nc`` (=> ``N``): bigger B buffer, ``A`` read fewer times (once at
          ``Nc==N``). Best when on-chip memory is ample.
        * small ``Nc`` (=> ``Nt``): tiny B buffer ``[K,Nt]``, but ``A`` re-read
          ``N/Nc`` times. Best when BRAM/URAM is tight or ``N`` is large.
        Pick the largest ``Nc`` whose ``[K,Nc]`` B-block fits the on-chip budget --
        exactly how real designs size their tiles. csynth latency is ~independent
        of ``Nc`` (only the B buffer / DRAM-traffic change); the trade only shows
        on-board. For very large ``K``, the next blocking level is ``K`` (partial
        sums) -- not done here. With no spare BRAM at all, use :mod:`os_direct`.
        """
        fifo_A: Stream[Tin, depth][Mt, Nt]
        fifo_B: Stream[Tin, depth][Mt, Nt]
        fifo_C: Stream[Tacc, depth][Mt, Nt]
        load_A(At, fifo_A)
        load_B(B, fifo_B)
        pe(fifo_A, fifo_B, fifo_C)
        store_C(C, fifo_C)

    pe_s = pe.schedule()
    pe_s.pipeline("k", ii=ii)

    la_s = load_A.schedule()
    la_s.partition(la_s.buffer("bufA"), dim=1, kind=la_s.Complete)  # lane (r)
    la_s.unroll("r")  # lane -> static stream index
    la_s.pipeline("k", ii=ii)

    lb_s = load_B.schedule()
    lb_s.partition(lb_s.buffer("bufB"), dim=2, kind=lb_s.Cyclic, factor=Nt)  # lane (c)
    lb_s.unroll("c")
    lb_s.pipeline("k", ii=ii)

    st_s = store_C.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("noi", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, la_s, lb_s, st_s)

    return top, top_s
