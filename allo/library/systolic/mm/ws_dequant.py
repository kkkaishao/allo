# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-only **group-quantized** weight-stationary GEMM (W4A16 / W8A16).

The drop-in quantized counterpart of :mod:`ws_direct` for LLM linear projections:
weights are stored low-bitwidth integer (``i4``/``i8``), grouped along the
contraction dim ``K`` with a per-group, per-output-column scale (and zero-point);
activations and the accumulation stay floating point. ``load_W`` dequantizes each
resident weight ``(w - z) * s`` to float once, then the **identical** float ws PE
array runs -- so this is the area-cheap way to cut weight DRAM 4x (int4) / 2x
(int8), which is the dominant lever for memory-bound LLM *decode*.
"""

import allo
from allo import kernel
from allo.lang import Stream, range
from allo.lang.core import DType, i32


def make_dequant_weight_stationary_gemm(
    Tin,
    Tacc,
    Tout,
    Tw,
    M: int,
    N: int,
    K: int,
    Kt: int,
    Nt: int,
    gs: int,
    depth=2,
    ii=1,
):
    """Build + schedule a weight-only group-quant ws GEMM; return ``(top, top_s)``.

    ``Tw`` (``i4``/``i8``) is the stored weight dtype; ``Tin``/``Tacc``/``Tout`` are
    float (activation / accumulate / output). Weights ``Wq[K,N]`` are grouped every
    ``gs`` rows of ``K`` with scale ``Sc[K/gs, N]`` (dtype ``Tin``) and zero-point
    ``Z[K/gs, N]`` (dtype ``Tw``; pass all-zero for symmetric quant). Requires
    ``N % Nt == 0``, ``K % Kt == 0``, ``K % gs == 0``."""
    assert (
        isinstance(Tin, DType) and isinstance(Tacc, DType) and isinstance(Tout, DType)
    )
    assert Tw.is_int(), "weights must be integer (i4/i8)"
    assert N % Nt == 0 and K % Kt == 0 and K % gs == 0, "tiling/grouping must divide"
    NT, KT, NG = N // Nt, K // Kt, K // gs

    @kernel
    def load_W(
        Wq: Tw[K, N],
        Sc: Tin[NG, N],
        Z: Tw[NG, N],
        fifo_W: Stream[Tin, depth][Kt, Nt],
    ):
        # Dequant each resident weight once: widen (w - z) to i32, cast to float,
        # multiply by the group scale; the product is reused across all M rows.
        for nt in range(NT):
            for kt in range(KT, name="kt"):
                for kk in range(Kt, name="kk"):  # unrolled lane
                    for nn in range(Nt, name="nn"):  # unrolled lane
                        k: i32 = kt * Kt + kk
                        g: i32 = k // gs
                        wi: i32 = Wq[k, nt * Nt + nn]  # i4/i8 -> i32 (sign-extend)
                        zi: i32 = Z[g, nt * Nt + nn]
                        d: i32 = wi - zi
                        df: Tin = d  # int -> float cast
                        fifo_W[kk, nn].put(df * Sc[g, nt * Nt + nn])

    @kernel
    def load_A(A: Tin[M, K], fifo_A: Stream[Tin, depth][Kt, Nt]):
        for nt in range(NT):
            for kt in range(KT):
                for mm in range(M, name="m"):
                    for kk in range(Kt, name="kk"):  # unrolled
                        fifo_A[kk, 0].put(A[mm, kt * Kt + kk])

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
                w: Tin = fifo_W[kk, nn].get()  # latch resident dequantized weight
                for mm in range(M, name="m"):
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
    def reduce_C(fifo_O: Stream[Tacc, depth][Kt, Nt], fifo_Ct: Stream[Tacc, depth][Nt]):
        accC: Tacc[M, Nt]
        for nt in range(NT):
            for kt in range(KT):
                for mm in range(M, name="m"):
                    for nn in range(Nt, name="nn"):
                        p: Tacc = fifo_O[Kt - 1, nn].get()
                        acc_val: Tacc = p
                        if kt > 0:
                            acc_val = accC[mm, nn] + p
                        if kt == KT - 1:
                            fifo_Ct[nn].put(acc_val)
                        else:
                            accC[mm, nn] = acc_val

    @kernel
    def write_C(C: Tout[M, N], fifo_Ct: Stream[Tacc, depth][Nt]):
        for nt in range(NT):
            for mm in range(M, name="m"):
                for nn in range(Nt, name="nn"):
                    C[mm, nt * Nt + nn] = fifo_Ct[nn].get()

    @kernel
    def top(A: Tin[M, K], Wq: Tw[K, N], Sc: Tin[NG, N], Z: Tw[NG, N], C: Tout[M, N]):
        """Weight-only **group-quantized** weight-stationary GEMM (W4A16 / W8A16).

        Computes ``C[M,N] = A[M,K] @ dequant(Wq[K,N])`` where the int weight is
        dequantized group-wise along ``K``::

            dequant(Wq)[k,n] = (Wq[k,n] - Z[k//gs, n]) * Sc[k//gs, n]

        i.e. ``Wq`` is ``i4``/``i8``, every ``gs`` contraction rows share a float
        scale ``Sc`` and integer zero-point ``Z`` per output column ``n`` (symmetric
        quant -> pass ``Z = 0``). The standard GPTQ/AWQ weight-only scheme.

        Architecture: this is :mod:`ws_direct` with a **dequantizing ``load_W``** --
        the int weight is widened, cast to float and scaled *once* as it is latched
        into the resident PE (reused across all ``M`` rows, so the dequant cost is
        fully amortized), and the **identical float ws array** then runs (spatial K
        reduction -> float II=1, no fadd recurrence). So the dequant is ~free: vs the
        plain f32 ws GEMM (128^3/16x16) it adds only a handful of int ops in the
        weight load (~+4% DSP) at the same latency, while the int4 weights cut
        weight-DRAM traffic 4x. That traffic is the bottleneck for memory-bound LLM
        **decode**, so W4A16 is the highest-value quantization for serving.

        Layout: ``A[M,K]`` float (standard), ``Wq[K,N]`` int, ``Sc[K/gs,N]`` /
        ``Z[K/gs,N]`` group params, ``C[M,N]`` float. ``N % Nt == 0``, ``K % Kt == 0``,
        ``K % gs == 0``. (Activations stay float -- to also quantize activations for
        an int-MAC datapath you need W8A8 with per-token activation quant, a
        different, lossier scheme.)"""
        fifo_W: Stream[Tin, depth][Kt, Nt]
        fifo_A: Stream[Tin, depth][Kt, Nt]
        fifo_P: Stream[Tacc, depth][Kt, Nt]
        fifo_O: Stream[Tacc, depth][Kt, Nt]
        fifo_Ct: Stream[Tacc, depth][Nt]
        load_W(Wq, Sc, Z, fifo_W)
        load_A(A, fifo_A)
        pe(fifo_W, fifo_A, fifo_P, fifo_O)
        reduce_C(fifo_O, fifo_Ct)
        write_C(C, fifo_Ct)

    pe_s = pe.schedule()
    pe_s.pipeline("m", ii=ii)

    lw_s = load_W.schedule()
    lw_s.unroll("nn")
    lw_s.unroll("kk")
    lw_s.pipeline("kt", ii=ii)

    la_s = load_A.schedule()
    la_s.unroll("kk")
    la_s.pipeline("m", ii=ii)

    rc_s = reduce_C.schedule()
    rc_s.partition(rc_s.buffer("accC"), dim=2, kind=rc_s.Complete)
    rc_s.unroll("nn")
    rc_s.pipeline("m", ii=ii)

    wc_s = write_C.schedule()
    wc_s.unroll("nn")
    wc_s.pipeline("m", ii=ii)

    top_s = top.schedule()
    top_s.dataflow()
    top_s.compose(pe_s, lw_s, la_s, rc_s, wc_s)

    return top, top_s
