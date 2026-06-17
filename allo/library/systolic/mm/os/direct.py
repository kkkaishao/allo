# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo import kernel
from allo.lang import Stream, range


def make_direct_output_stationary_gemm(
    Tin, Tacc, Tout, M: int, N: int, K: int, Mt: int, Nt: int, depth=2, ii=1
):
    MT, NT = M // Mt, N // Nt  # tile counts

    @kernel
    def load_A(At: Tin[K, M], fifo_A: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MT):
            for no in range(NT, name="no"):  # re-read A for each column-tile
                for k in range(K, name="k"):
                    for r in range(Mt, name="r"):  # lane
                        fifo_A[r, 0].put(At[k, mo * Mt + r])

    @kernel
    def load_B(B: Tin[K, N], fifo_B: Stream[Tin, depth][Mt, Nt]):
        for mo in range(MT, name="mo"):  # re-read B for each row-tile
            for no in range(NT):
                for k in range(K, name="k"):
                    for c in range(Nt, name="c"):  # lane
                        fifo_B[0, c].put(B[k, no * Nt + c])

    @kernel(mapping=[Mt, Nt])
    def pe(
        fifo_A: Stream[Tin, depth][Mt, Nt],
        fifo_B: Stream[Tin, depth][Mt, Nt],
        fifo_C: Stream[Tacc, depth][Mt, Nt],
    ):
        r = allo.get_wid(0)
        c = allo.get_wid(1)
        for mo in range(MT):
            for no in range(NT):
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
        for mo in range(MT):
            for no in range(NT, name="no"):
                for r in range(Mt, name="r"):
                    for c in range(Nt, name="c"):
                        C[mo * Mt + r, no * Nt + c] = fifo_C[r, c].get()

    @kernel
    def top(At: Tin[K, M], B: Tin[K, N], C: Tout[M, N]):
        """Output-stationary systolic GEMM, **direct / re-reading** loaders.

        Computes ``C[M,N] = AT^T @ B`` where ``AT`` is the **pre-transposed** ``A`` (shape
        ``[K,M]``), so each loader reads its operand contiguously along the fast axis.
        An ``Mt x Nt`` PE grid sweeps ``(M/Mt) x (N/Nt)`` output tiles, accumulating each
        output element across ``K`` inside its PE (output-stationary); ``A`` flows east,
        ``B`` flows south, boundary PEs drop the forwarded value so no drain process is
        needed.

        Loaders stream straight from DRAM with **no on-chip operand buffer**, re-reading
        each operand once per reuse step:

        * ``load_A`` re-reads the ``A`` row-panel for each of the ``NT`` column-tiles
          (``A`` read ``NT`` times total);
        * ``load_B`` re-reads ``B`` for each of the ``MT`` row-tiles (``B`` read ``MT``
          times total).

        Reads stay contiguous thanks to the pre-transposed ``A`` layout (``At[k, .]``)
        and the natural ``B[k, .]`` layout, so a wide ``m_axi`` port bursts efficiently
        despite the redundant traffic. On-chip footprint is just the inter-process FIFOs.

        WHEN TO USE THIS VARIANT
        ------------------------
        Prefer ``os_direct`` when the design is **compute-bound** or on-chip memory is
        tight:

        * a well-provisioned, wide ``m_axi`` / HBM port where the re-read traffic is
          fully hidden behind compute (typical for square-ish GEMMs on a 16x16 array);
        * large ``N``, where buffering the whole ``B`` (as :mod:`os_buffered` does) would
          exhaust BRAM/URAM -- this variant needs no operand buffer at all;
        * simplicity: no buffers, no inline-fill conditional.

        COST / LIMITS
        -------------
        Operand DRAM traffic is the reuse factor higher than :mod:`os_buffered` (``A``
        x``NT``, ``B`` x``MT``), costing extra power and bandwidth. If the port is narrow
        / bandwidth-starved (DDR or a shared channel) the re-reads can dominate and the
        array stalls -- use :mod:`os_buffered` there. (csynth latency is identical for
        both; the difference only appears on-board under a real memory model.)
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
    la_s.unroll("r")  # lane -> static stream index
    la_s.pipeline("k", ii=ii)

    lb_s = load_B.schedule()
    lb_s.unroll("c")  # lane -> static stream index
    lb_s.pipeline("k", ii=ii)

    st_s = store_C.schedule()
    st_s.unroll("r")
    st_s.unroll("c")
    st_s.pipeline("no", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, la_s, lb_s, st_s)

    return top, top_s
