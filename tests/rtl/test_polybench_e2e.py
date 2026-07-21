# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for kernels transcribed from examples/polybench.

The scheduling tests run the kernels at their reference problem sizes; the
``_cosim`` tests re-state them shrunk to a size verilator can simulate and drive
the emitted RTL against a NumPy golden.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from allo import kernel  # noqa: E402
from allo.lang import f32, index  # noqa: E402
from allo.operators import math as amath  # noqa: E402
from allo.lang.ip import ip  # noqa: E402
from allo.backend.rtl.device import builtin_device  # noqa: E402
from _common import (  # noqa: E402
    _sched,
    _to_rtl,
    _iis,
    FADD,
    FDIV,
    MEM_REDUCE_II,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

# f32 accumulation reassociates in hardware (the reduction is balanced into a
# tree), so a cosim result matches a sequential NumPy golden only to f32 epsilon
# grown by the reduction depth -- compare with a tolerance, never exactly.
FTOL = {"rtol": 2e-3, "atol": 2e-3}


def _f32(seed, *shape):
    """Deterministic f32 test data in [-0.5, 0.5)."""
    rng = np.random.default_rng(seed)
    return (rng.random(shape, dtype=np.float32) - np.float32(0.5)).astype(np.float32)


def test_matmul_reductions():
    """Matmul stages accumulate into memory (II = read + add + write); the
    elementwise and writeback stages carry no recurrence and pipeline at II=1."""
    P, R, Q, S, alpha, beta = 40, 50, 70, 80, 0.1, 0.5

    # gemm = matmul then a scaled elementwise add.
    @kernel
    def gemm_mm(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def gemm_add(out_AB: f32[P, R], C: f32[P, R], output: f32[P, R]):
        for i2 in range(P):
            for j2 in range(R):
                output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]

    @kernel
    def gemm(A: f32[P, Q], B: f32[Q, R], C: f32[P, R], output: f32[P, R]):
        out_AB: f32[P, R] = 0.0
        gemm_mm(A, B, out_AB)
        gemm_add(out_AB, C, output)

    res = _sched(gemm)
    assert res.func("gemm_mm").cyclic()[0].ii == MEM_REDUCE_II
    assert res.func("gemm_add").cyclic()[0].ii == 1

    # two_mm = (A*B)*C: two chained matmul reductions feeding an elementwise stage.
    @kernel
    def tmm_ab(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def tmm_abc(out_AB: f32[P, R], C: f32[R, S], out_ABC: f32[P, S]):
        for i1 in range(P):
            for j1 in range(S):
                for k1 in range(R):
                    out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]

    @kernel
    def tmm_add(out_ABC: f32[P, S], D: f32[P, S], output: f32[P, S]):
        for i2 in range(P):
            for j2 in range(S):
                output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha

    @kernel
    def two_mm(A: f32[P, Q], B: f32[Q, R], C: f32[R, S], D: f32[P, S]) -> f32[P, S]:
        out_AB: f32[P, R] = 0.0
        out_ABC: f32[P, S] = 0.0
        output: f32[P, S]
        tmm_ab(A, B, out_AB)
        tmm_abc(out_AB, C, out_ABC)
        tmm_add(out_ABC, D, output)
        return output

    res = _sched(two_mm)
    assert res.func("tmm_ab").cyclic()[0].ii == MEM_REDUCE_II
    assert res.func("tmm_abc").cyclic()[0].ii == MEM_REDUCE_II
    assert res.func("tmm_add").cyclic()[0].ii == 1

    # doitgen: a four-deep nest decomposing into a reduction and a copy region.
    DQ, DR, DP = 20, 25, 30

    @kernel
    def doitgen(A: f32[DR, DQ, DP], x: f32[DP, DP], sum_: f32[DP]):
        for r in range(DR):
            for q in range(DQ):
                for p in range(DP):
                    sum_[p] = 0.0
                    for s in range(DP):
                        sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
                for p1 in range(DP):
                    A[r, q, p1] = sum_[p1]

    iis = _iis(_sched(doitgen).cyclic())
    assert MEM_REDUCE_II in iis  # the inner accumulation
    assert 1 in iis  # the writeback copy


def test_reduction_ii_follows_accumulator_location():
    """The accumulator's location sets the II: a memory cell indexed by the inner
    IV carries no recurrence (II=1), one indexed by the outer IV closes a
    read->add->write recurrence, and a scalar keeps the partial in a register."""
    M, N = 116, 124

    # bicg: stageS accumulates into s[j0], the INNER index -- every iteration
    # touches a different cell, so there is no carried recurrence. stageQ
    # accumulates into q[i1], the outer index, across the inner loop.
    @kernel
    def stageS(A: f32[N, M], r: f32[N], s: f32[M]):
        for i0 in range(N):
            local_r: f32 = r[i0]
            for j0 in range(M):
                s[j0] += local_r * A[i0, j0]

    @kernel
    def stageQ(A: f32[N, M], p: f32[M], q: f32[N]):
        for i1 in range(N):
            for j1 in range(M):
                q[i1] += A[i1, j1] * p[j1]

    @kernel
    def bicg(
        A: f32[N, M], A_copy: f32[N, M], p: f32[M], r: f32[N], q: f32[N], s: f32[M]
    ):
        stageS(A, r, s)
        stageQ(A_copy, p, q)

    res = _sched(bicg)
    assert res.func("stageS").cyclic()[0].ii == 1
    assert res.func("stageQ").cyclic()[0].ii == MEM_REDUCE_II

    # atax = A^T (A x): both stages accumulate into a memory cell.
    @kernel
    def atax_m(A: f32[M, N], x: f32[N], out_Ax: f32[M]):
        for m in range(M):
            for r in range(N):
                out_Ax[m] += A[m, r] * x[r]

    @kernel
    def atax_n(A: f32[M, N], out_Ax: f32[M], y: f32[N]):
        for n in range(N):
            for k in range(M):
                y[n] += A[k, n] * out_Ax[k]

    @kernel
    def atax(A: f32[M, N], x: f32[N], y: f32[N]):
        out_Ax: f32[M] = 0.0
        atax_m(A, x, out_Ax)
        atax_n(A, out_Ax, y)

    res = _sched(atax)
    assert res.func("atax_m").cyclic()[0].ii == MEM_REDUCE_II
    assert res.func("atax_n").cyclic()[0].ii == MEM_REDUCE_II

    # mvt accumulates into a scalar local, so the recurrence stays in a register
    # and the II is just the add latency. The load-init before and store after the
    # inner loop make it an imperfect nest: a sequential outer wrapper around the
    # pipelined inner region plus acyclic prologue/epilogue regions.
    V = 120

    @kernel
    def stageA(x1_in: f32[V], x1_out: f32[V], A: f32[V, V], y1: f32[V]):
        for i0 in range(V):
            x: f32 = x1_in[i0]
            for j0 in range(V):
                x += A[i0, j0] * y1[j0]
            x1_out[i0] = x

    @kernel
    def stageB(x2_in: f32[V], x2_out: f32[V], A: f32[V, V], y2: f32[V]):
        for i1 in range(V):
            x: f32 = x2_in[i1]
            for j1 in range(V):
                x += A[j1, i1] * y2[j1]
            x2_out[i1] = x

    @kernel
    def mvt(
        A: f32[V, V],
        A_copy: f32[V, V],
        y1: f32[V],
        y2: f32[V],
        x1: f32[V],
        x2: f32[V],
        x1_out: f32[V],
        x2_out: f32[V],
    ):
        stageA(x1, x1_out, A, y1)
        stageB(x2, x2_out, A_copy, y2)

    sa = _sched(mvt).func("stageA")
    assert sa.cyclic()[0].ii == FADD  # scalar recurrence, not MEM_REDUCE_II
    assert len([r for r in sa.regions if r.kind == "acyclic"]) >= 2  # prologue+epilogue
    wrapper = next(r for r in sa.regions if r.is_wrapper)
    assert wrapper.depth == 0 and wrapper.trip == V


def test_stencil_ii_port_vs_recurrence_bound():
    """A dependence-free stencil is bound by memory-port pressure; an in-place one
    is bound by its carried recurrence."""
    TSTEPS, N = 40, 120

    # jacobi_1d: three reads over two ports -> II = ceil(3/2) = 2.
    @kernel
    def jacobi_1d(A: f32[N], B: f32[N]):
        for m in range(TSTEPS):
            for i0 in range(1, N - 1):
                B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])
            for i1 in range(1, N - 1):
                A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])

    cyclic = _sched(jacobi_1d).cyclic()
    assert len(cyclic) == 2 and all(r.ii == 2 for r in cyclic)

    # fdtd_2d: four dependence-free update stages, each at II=1.
    Tmax, Nx, Ny = 40, 60, 80

    @kernel
    def fdtd_2d(ex: f32[Nx, Ny], ey: f32[Nx, Ny], hz: f32[Nx, Ny], fict: f32[Tmax]):
        for m in range(Tmax):
            for j in range(Ny):
                ey[0, j] = fict[m]
            for i in range(1, Nx):
                for j in range(Ny):
                    ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
            for i in range(Nx):
                for j in range(1, Ny):
                    ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
            for i in range(Nx - 1):
                for j in range(Ny - 1):
                    hz[i, j] = hz[i, j] - 0.7 * (
                        ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                    )

    cyclic = _sched(fdtd_2d).cyclic()
    assert len(cyclic) == 4 and all(r.ii == 1 for r in cyclic)

    # heat_3d: a 7-point stencil issues many loads per iteration, so with no
    # recurrence the II is dominated by port pressure.
    H = 20

    @kernel
    def heat_3d(A: f32[H, H, H], B: f32[H, H, H]):
        const0: f32 = 0.125
        const1: f32 = 2.0
        for m in range(TSTEPS):
            for i in range(1, H - 1):
                for j in range(1, H - 1):
                    for k in range(1, H - 1):
                        B[i, j, k] = (
                            const0
                            * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                            + const0
                            * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                            + const0
                            * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                            + A[i, j, k]
                        )
                        A[i, j, k] = (
                            const0
                            * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                            + const0
                            * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                            + const0
                            * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                            + B[i, j, k]
                        )

    assert _sched(heat_3d).cyclic()[0].ii > FADD

    # seidel_2d: a 9-point Gauss-Seidel sweep updates A in place, so A[i,j-1] and
    # A[i-1,*] read values written earlier in the same sweep -- the II is set by
    # that carried recurrence (the divide is on its critical path), not by ports.
    @kernel
    def seidel_2d(A: f32[N, N]):
        for t in range(TSTEPS):
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    A[i, j] = (
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9.0

    cyclic = _sched(seidel_2d).cyclic()
    assert len(cyclic) == 1 and cyclic[0].ii > FDIV


def test_multi_region_single_func():
    """Several sweeps in one function schedule to one cyclic region each, mixing
    dependence-free (II=1) and memory-carried reduction (II>1) loops."""
    N, alpha, beta = 120, 0.1, 0.1

    @kernel
    def gemver(
        A: f32[N, N],
        u1: f32[N],
        u2: f32[N],
        v1: f32[N],
        v2: f32[N],
        x: f32[N],
        y: f32[N],
        w: f32[N],
        z: f32[N],
    ):
        for i in range(N):
            for j in range(N):
                A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
        for i in range(N):
            for j in range(N):
                x[i] = x[i] + beta * A[j, i] * y[j]
        for i in range(N):
            x[i] = x[i] + z[i]
        for i in range(N):
            for j in range(N):
                w[i] = w[i] + alpha * A[i, j] * x[j]

    iis = set(_iis(_sched(gemver).cyclic()))
    assert 1 in iis and any(v > 1 for v in iis)

    # gesummv: tmp = A*x (reduction) then y = alpha*tmp + beta*x (elementwise).
    G = 90

    @kernel
    def compute_tmp(
        y_in: f32[G], y_out: f32[G], A: f32[G, G], B: f32[G, G], x: f32[G], tmp: f32[G]
    ):
        tt: f32[G] = 0.0
        yy: f32[G]
        for i0 in range(G):
            yy[i0] = y_in[i0]
        for i in range(G):
            for j in range(G):
                tt[i] += A[i, j] * x[j]
                yy[i] += B[i, j] * x[j]
        for i1 in range(G):
            tmp[i1] = tt[i1]
            y_out[i1] = yy[i1]

    @kernel
    def compute_y(y_in: f32[G], y_out: f32[G], tmp: f32[G]):
        for i0 in range(G):
            y_out[i0] = alpha * tmp[i0] + beta * y_in[i0]

    @kernel
    def gesummv(A: f32[G, G], B: f32[G, G], x: f32[G], y: f32[G]):
        y_init: f32[G] = 0.0
        y_fifo: f32[G]
        tmp: f32[G]
        compute_tmp(y_init, y_fifo, A, B, x, tmp)
        compute_y(y_fifo, y, tmp)

    res = _sched(gesummv)
    assert MEM_REDUCE_II in _iis(res.func("compute_tmp").cyclic())
    assert res.func("compute_y").cyclic()[0].ii == 1


def test_if_conversion_in_loops():
    """A guard inside a loop body if-converts to a select so the loop still
    pipelines; a guard that is affine in the IV folds into the loop bound."""
    M, N, alpha, beta = 60, 80, 1.5, 1.2

    # trmm: a triangular guard (k > i) over a memory-carried accumulate.
    @kernel
    def S0(A: f32[M, M], B: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        B[i1, j1] += A[k1, i1] * B[k1, j1]

    @kernel
    def S1(B: f32[M, N]):
        for i0 in range(M):
            for j0 in range(N):
                B[i0, j0] = B[i0, j0] * alpha

    @kernel
    def trmm(A: f32[M, M], B: f32[M, N]):
        S0(A, B)
        S1(B)

    res = _sched(trmm)
    assert res.func("S0").cyclic()[0].ii > 1
    assert res.func("S1").cyclic()[0].ii == 1

    # floyd_warshall: a conditional store becomes a predicated read-modify-write.
    F = 180

    @kernel
    def floyd_warshall(path: f32[F, F]):
        for k in range(F):
            for i in range(F):
                for j in range(F):
                    path_: f32 = path[i, k] + path[k, j]
                    if path[i, j] >= path_:
                        path[i, j] = path_

    loop = _sched(floyd_warshall).cyclic()[0]
    assert loop.has("select") and not loop.has("if")

    # syrk: an if/else over a triangular region if-converts to a select over the
    # two speculated values.
    @kernel
    def update_C(Cin: f32[N, N], Cout: f32[N, N]):
        for i0 in range(N):
            for j0 in range(N):
                if j0 <= i0:
                    Cout[i0, j0] = beta * Cin[i0, j0]
                else:
                    Cout[i0, j0] = Cin[i0, j0]

    @kernel
    def compute_sum(A: f32[N, M], A_copy: f32[N, M], Cin: f32[N, N], Cout: f32[N, N]):
        buffer: f32[N, N] = 0.0
        for i0 in range(N):
            for j0 in range(N):
                buffer[i0, j0] = Cin[i0, j0]
        for i1 in range(N):
            for k1 in range(M):
                for j1 in range(N):
                    if j1 <= i1:
                        buffer[i1, j1] += alpha * A[i1, k1] * A_copy[j1, k1]
        for i2 in range(N):
            for j2 in range(N):
                Cout[i2, j2] = buffer[i2, j2]

    @kernel
    def syrk(A: f32[N, M], A_copy: f32[N, M], Cin: f32[N, N], Cout: f32[N, N]):
        C: f32[N, N] = 0.0
        update_C(Cin, C)
        compute_sum(A, A_copy, C, Cout)

    update = _sched(syrk).func("update_C").cyclic()[0]
    assert update.has("select") and not update.has("if")

    # correlation: `if j > i` is affine in the IV, so it folds into the loop's
    # lower bound -- the dead iterations are skipped, not masked.
    CN, CM = 100, 40

    @kernel
    def compute_corr(data: f32[CN, CM], corr: f32[CM, CM]):
        for i in range(CM - 1):
            corr[i, i] = 1.0
            for j in range(CM):
                if j > i:
                    corr_v: f32 = 0.0
                    for k in range(CN):
                        corr_v += data[k, i] * data[k, j]
                    corr[j, i] = corr_v
                    corr[i, j] = corr_v
        corr[CM - 1, CM - 1] = 1.0

    mod = _to_rtl(compute_corr)
    assert "affine.if" not in mod.dcp  # folded into the bound, not predicated
    assert _iis(mod.schedule().func("compute_corr").cyclic()) == [FADD]


def test_data_dependent_bounds_leave_latency_unknown():
    """Loops with data-dependent (triangular) trip counts still pipeline, but the
    whole-function latency is left undetermined rather than fabricated."""
    N = 120

    # sqrt is non-combinational with no built-in characterization (design doc
    # SS5.4): declare it as an operator IP so the kernel is fully characterized.
    @ip(optype="sqrt", latency=7, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def cholesky(A: f32[N, N]):
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
                A[i, j] = A[i, j] / A[j, j]
            for k in range(i):
                A[i, i] = A[i, i] - A[i, k] * A[i, k]
            A[i, i] = amath.sqrt(A[i, i] * 1.0)

    dev = builtin_device.copy()
    dev.add_operator(fsqrt)
    res = _sched(cholesky, device=dev)
    assert res.func("cholesky").latency is None
    assert any(r.ii > 1 for r in res.cyclic())

    # nussinov: a triangular DP with an inner max-reduction and boundary guards.
    D = 180

    @kernel
    def nussinov(seq: f32[D], table: f32[D, D]):
        for i_inv in range(D):
            i: index = D - 1 - i_inv
            for j in range(i + 1, D):
                if j - 1 >= 0:
                    if table[i, j] < table[i, j - 1]:
                        table[i, j] = table[i, j - 1]
                if i + 1 < D:
                    if table[i, j] < table[i + 1, j]:
                        table[i, j] = table[i + 1, j]
                if j - 1 >= 0 and i + 1 < D:
                    if i < j - 1:
                        w: f32 = seq[i] + seq[j]
                        match: f32 = 0.0
                        if w == 3.0:
                            match = 1.0
                        s2: f32 = table[i + 1, j - 1] + match
                        if table[i, j] < s2:
                            table[i, j] = s2
                    else:
                        if table[i, j] < table[i + 1, j - 1]:
                            table[i, j] = table[i + 1, j - 1]
                for k in range(i + 1, j):
                    s3: f32 = table[i, k] + table[k + 1, j]
                    if table[i, j] < s3:
                        table[i, j] = s3

    res = _sched(nussinov)
    assert res.func("nussinov").latency is None
    loop = res.cyclic()[0]
    assert loop.ii > 1  # memory-carried max recurrence into table[i, j]
    assert loop.has("select")  # boundary/compare guards if-converted


def test_matmul_reductions_cosim():
    """The matmul chain drives correctly end to end: a memory-carried reduction
    feeding an elementwise stage (gemm), two chained reductions (two_mm), and a
    reduction whose partials are republished through a copy region (doitgen)."""
    P, R, Q, S, alpha, beta = 4, 5, 6, 3, 0.1, 0.5

    @kernel
    def gemm_mm(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def gemm_add(out_AB: f32[P, R], C: f32[P, R], output: f32[P, R]):
        for i2 in range(P):
            for j2 in range(R):
                output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]

    @kernel
    def gemm(A: f32[P, Q], B: f32[Q, R], C: f32[P, R], output: f32[P, R]):
        out_AB: f32[P, R] = 0.0
        gemm_mm(A, B, out_AB)
        gemm_add(out_AB, C, output)

    A, B, C = _f32(0, P, Q), _f32(1, Q, R), _f32(2, P, R)
    output = np.zeros((P, R), np.float32)
    _to_rtl(gemm).cosim(A, B, C, output)
    assert np.allclose(output, beta * C + A @ B, **FTOL)

    # two_mm = (A*B)*C scaled and added to D: the second reduction consumes the
    # first through an internal buffer.
    @kernel
    def tmm_ab(A: f32[P, Q], B: f32[Q, R], out_AB: f32[P, R]):
        for i0 in range(P):
            for j0 in range(R):
                for k0 in range(Q):
                    out_AB[i0, j0] += A[i0, k0] * B[k0, j0]

    @kernel
    def tmm_abc(out_AB: f32[P, R], C: f32[R, S], out_ABC: f32[P, S]):
        for i1 in range(P):
            for j1 in range(S):
                for k1 in range(R):
                    out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]

    @kernel
    def tmm_add(out_ABC: f32[P, S], D: f32[P, S], output: f32[P, S]):
        for i2 in range(P):
            for j2 in range(S):
                output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha

    @kernel
    def two_mm(
        A: f32[P, Q], B: f32[Q, R], C: f32[R, S], D: f32[P, S], output: f32[P, S]
    ):
        out_AB: f32[P, R] = 0.0
        out_ABC: f32[P, S] = 0.0
        tmm_ab(A, B, out_AB)
        tmm_abc(out_AB, C, out_ABC)
        tmm_add(out_ABC, D, output)

    A, B, C, D = _f32(0, P, Q), _f32(1, Q, R), _f32(2, R, S), _f32(3, P, S)
    output = np.zeros((P, S), np.float32)
    _to_rtl(two_mm).cosim(A, B, C, D, output)
    assert np.allclose(output, (A @ B) @ C * beta + D * alpha, **FTOL)

    # doitgen: the reduction writes sum_[p], then a second region copies it back
    # over A[r, q, :] -- the copy must not start until the reduction has drained.
    DQ, DR, DP = 3, 4, 5

    @kernel
    def doitgen(A: f32[DR, DQ, DP], x: f32[DP, DP], sum_: f32[DP]):
        for r in range(DR):
            for q in range(DQ):
                for p in range(DP):
                    sum_[p] = 0.0
                    for s in range(DP):
                        sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
                for p1 in range(DP):
                    A[r, q, p1] = sum_[p1]

    A, x = _f32(0, DR, DQ, DP), _f32(1, DP, DP)
    exp = A.copy()
    for r in range(DR):
        for q in range(DQ):
            exp[r, q, :] = exp[r, q, :] @ x
    _to_rtl(doitgen).cosim(A, x, np.zeros(DP, np.float32))
    assert np.allclose(A, exp, **FTOL)


def test_reduction_accumulator_location_cosim():
    """Both accumulator placements produce the right values: bicg's stageS writes
    a different cell each iteration (II=1) while stageQ closes a read-add-write
    recurrence, and atax chains two such reductions through a buffer."""
    M, N = 6, 5

    @kernel
    def stageS(A: f32[N, M], r: f32[N], s: f32[M]):
        for i0 in range(N):
            local_r: f32 = r[i0]
            for j0 in range(M):
                s[j0] += local_r * A[i0, j0]

    @kernel
    def stageQ(A: f32[N, M], p: f32[M], q: f32[N]):
        for i1 in range(N):
            for j1 in range(M):
                q[i1] += A[i1, j1] * p[j1]

    @kernel
    def bicg(
        A: f32[N, M], A_copy: f32[N, M], p: f32[M], r: f32[N], q: f32[N], s: f32[M]
    ):
        stageS(A, r, s)
        stageQ(A_copy, p, q)

    A, p, r = _f32(0, N, M), _f32(1, M), _f32(2, N)
    q, s = np.zeros(N, np.float32), np.zeros(M, np.float32)
    _to_rtl(bicg).cosim(A, A.copy(), p, r, q, s)
    assert np.allclose(s, r @ A, **FTOL)
    assert np.allclose(q, A @ p, **FTOL)

    # atax = A^T (A x): the second stage may not read out_Ax before the first has
    # finished writing it.
    AM, AN = 5, 6

    @kernel
    def atax_m(A: f32[AM, AN], x: f32[AN], out_Ax: f32[AM]):
        for m in range(AM):
            for rr in range(AN):
                out_Ax[m] += A[m, rr] * x[rr]

    @kernel
    def atax_n(A: f32[AM, AN], out_Ax: f32[AM], y: f32[AN]):
        for n in range(AN):
            for k in range(AM):
                y[n] += A[k, n] * out_Ax[k]

    @kernel
    def atax(A: f32[AM, AN], x: f32[AN], y: f32[AN]):
        out_Ax: f32[AM] = 0.0
        atax_m(A, x, out_Ax)
        atax_n(A, out_Ax, y)

    A, x = _f32(0, AM, AN), _f32(1, AN)
    y = np.zeros(AN, np.float32)
    _to_rtl(atax).cosim(A, x, y)
    assert np.allclose(y, A.T @ (A @ x), **FTOL)

    # mvt seeds each accumulator from a LOAD rather than from the reduction
    # identity, so the init is a prologue region's survivor rather than a
    # constant. The inner pipeline must re-inject it on every outer iteration:
    # left unmodelled, the accumulator keeps its reset value on the first row and
    # free-runs across the rest, absorbing the previous row's partials -- x1_in
    # would have no effect on the result at all.
    V = 4

    @kernel
    def stageA(x1_in: f32[V], x1_out: f32[V], A: f32[V, V], y1: f32[V]):
        for i0 in range(V):
            x: f32 = x1_in[i0]
            for j0 in range(V):
                x += A[i0, j0] * y1[j0]
            x1_out[i0] = x

    @kernel
    def stageB(x2_in: f32[V], x2_out: f32[V], A: f32[V, V], y2: f32[V]):
        for i1 in range(V):
            x: f32 = x2_in[i1]
            for j1 in range(V):
                x += A[j1, i1] * y2[j1]
            x2_out[i1] = x

    @kernel
    def mvt(
        A: f32[V, V],
        A_copy: f32[V, V],
        y1: f32[V],
        y2: f32[V],
        x1: f32[V],
        x2: f32[V],
        x1_out: f32[V],
        x2_out: f32[V],
    ):
        stageA(x1, x1_out, A, y1)
        stageB(x2, x2_out, A_copy, y2)

    A, y1, y2 = _f32(0, V, V), _f32(1, V), _f32(2, V)
    x1, x2 = _f32(3, V), _f32(4, V)
    x1_out, x2_out = np.zeros(V, np.float32), np.zeros(V, np.float32)
    _to_rtl(mvt).cosim(A, A.copy(), y1, y2, x1, x2, x1_out, x2_out)
    assert np.allclose(x1_out, x1 + A @ y1, **FTOL)
    assert np.allclose(x2_out, x2 + A.T @ y2, **FTOL)


def test_stencil_cosim():
    """The stencil family end to end: a two-sweep 1-D jacobi, fdtd_2d's four
    dependence-free stages, heat_3d updating both buffers in one body, and
    seidel_2d's in-place carried recurrence -- the last two only reproduce the
    sequential result if the recurrence actually serializes."""
    TSTEPS, N = 3, 8
    c = np.float32(0.33333)

    @kernel
    def jacobi_1d(A: f32[N], B: f32[N]):
        for m in range(TSTEPS):
            for i0 in range(1, N - 1):
                B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])
            for i1 in range(1, N - 1):
                A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])

    A, B = _f32(0, N), _f32(1, N)
    Ag, Bg = A.copy(), B.copy()
    for _ in range(TSTEPS):
        for i in range(1, N - 1):
            Bg[i] = c * (Ag[i - 1] + Ag[i] + Ag[i + 1])
        for i in range(1, N - 1):
            Ag[i] = c * (Bg[i - 1] + Bg[i] + Bg[i + 1])
    _to_rtl(jacobi_1d).cosim(A, B)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(B, Bg, **FTOL)

    # fdtd_2d: four stages per timestep over three shared buffers; each stage
    # reads what the previous one wrote, so they must not overlap.
    Tmax, Nx, Ny = 2, 4, 5
    h, s = np.float32(0.5), np.float32(0.7)

    @kernel
    def fdtd_2d(ex: f32[Nx, Ny], ey: f32[Nx, Ny], hz: f32[Nx, Ny], fict: f32[Tmax]):
        for m in range(Tmax):
            for j in range(Ny):
                ey[0, j] = fict[m]
            for i in range(1, Nx):
                for j in range(Ny):
                    ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
            for i in range(Nx):
                for j in range(1, Ny):
                    ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
            for i in range(Nx - 1):
                for j in range(Ny - 1):
                    hz[i, j] = hz[i, j] - 0.7 * (
                        ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                    )

    ex, ey, hz, fict = _f32(0, Nx, Ny), _f32(1, Nx, Ny), _f32(2, Nx, Ny), _f32(3, Tmax)
    exg, eyg, hzg = ex.copy(), ey.copy(), hz.copy()
    for m in range(Tmax):
        for j in range(Ny):
            eyg[0, j] = fict[m]
        for i in range(1, Nx):
            for j in range(Ny):
                eyg[i, j] = eyg[i, j] - h * (hzg[i, j] - hzg[i - 1, j])
        for i in range(Nx):
            for j in range(1, Ny):
                exg[i, j] = exg[i, j] - h * (hzg[i, j] - hzg[i, j - 1])
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                hzg[i, j] = hzg[i, j] - s * (
                    exg[i, j + 1] - exg[i, j] + eyg[i + 1, j] - eyg[i, j]
                )
    _to_rtl(fdtd_2d).cosim(ex, ey, hz, fict)
    assert np.allclose(ex, exg, **FTOL)
    assert np.allclose(ey, eyg, **FTOL)
    assert np.allclose(hz, hzg, **FTOL)

    # heat_3d: B[i,j,k] is written and then immediately re-read by the A update in
    # the *same* body, an intra-iteration dependence on top of the 7-point window.
    HT, H = 2, 5
    c0, c1 = np.float32(0.125), np.float32(2.0)

    @kernel
    def heat_3d(A: f32[H, H, H], B: f32[H, H, H]):
        const0: f32 = 0.125
        const1: f32 = 2.0
        for m in range(HT):
            for i in range(1, H - 1):
                for j in range(1, H - 1):
                    for k in range(1, H - 1):
                        B[i, j, k] = (
                            const0
                            * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                            + const0
                            * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                            + const0
                            * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                            + A[i, j, k]
                        )
                        A[i, j, k] = (
                            const0
                            * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                            + const0
                            * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                            + const0
                            * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                            + B[i, j, k]
                        )

    A, B = _f32(0, H, H, H), _f32(1, H, H, H)
    Ag, Bg = A.copy(), B.copy()
    for _ in range(HT):
        for i in range(1, H - 1):
            for j in range(1, H - 1):
                for k in range(1, H - 1):
                    Bg[i, j, k] = (
                        c0 * (Ag[i + 1, j, k] - c1 * Ag[i, j, k] + Ag[i - 1, j, k])
                        + c0 * (Ag[i, j + 1, k] - c1 * Ag[i, j, k] + Ag[i, j - 1, k])
                        + c0 * (Ag[i, j, k + 1] - c1 * Ag[i, j, k] + Ag[i, j, k - 1])
                        + Ag[i, j, k]
                    )
                    Ag[i, j, k] = (
                        c0 * (Bg[i + 1, j, k] - c1 * Bg[i, j, k] + Bg[i - 1, j, k])
                        + c0 * (Bg[i, j + 1, k] - c1 * Bg[i, j, k] + Bg[i, j - 1, k])
                        + c0 * (Bg[i, j, k + 1] - c1 * Bg[i, j, k] + Bg[i, j, k - 1])
                        + Bg[i, j, k]
                    )
    _to_rtl(heat_3d).cosim(A, B)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(B, Bg, **FTOL)

    # seidel_2d: every read of A[i-1,*] / A[i,j-1] must see the value written
    # earlier in this same sweep.
    SN = 6

    @kernel
    def seidel_2d(A: f32[SN, SN]):
        for t in range(TSTEPS):
            for i in range(1, SN - 1):
                for j in range(1, SN - 1):
                    A[i, j] = (
                        A[i - 1, j - 1]
                        + A[i - 1, j]
                        + A[i - 1, j + 1]
                        + A[i, j - 1]
                        + A[i, j]
                        + A[i, j + 1]
                        + A[i + 1, j - 1]
                        + A[i + 1, j]
                        + A[i + 1, j + 1]
                    ) / 9.0

    A = _f32(0, SN, SN)
    Ag = A.copy()
    for _ in range(TSTEPS):
        for i in range(1, SN - 1):
            for j in range(1, SN - 1):
                Ag[i, j] = (
                    Ag[i - 1, j - 1]
                    + Ag[i - 1, j]
                    + Ag[i - 1, j + 1]
                    + Ag[i, j - 1]
                    + Ag[i, j]
                    + Ag[i, j + 1]
                    + Ag[i + 1, j - 1]
                    + Ag[i + 1, j]
                    + Ag[i + 1, j + 1]
                ) / np.float32(9.0)
    _to_rtl(seidel_2d).cosim(A)
    assert np.allclose(A, Ag, **FTOL)


def test_multi_region_single_func_cosim():
    """Sweeps chained through shared arrays inside one function: gemver's four
    regions each consume the previous one's writes, and gesummv fuses two
    reductions in one body before an elementwise stage."""
    N, alpha, beta = 5, 0.1, 0.1

    @kernel
    def gemver(
        A: f32[N, N],
        u1: f32[N],
        u2: f32[N],
        v1: f32[N],
        v2: f32[N],
        x: f32[N],
        y: f32[N],
        w: f32[N],
        z: f32[N],
    ):
        for i in range(N):
            for j in range(N):
                A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
        for i in range(N):
            for j in range(N):
                x[i] = x[i] + beta * A[j, i] * y[j]
        for i in range(N):
            x[i] = x[i] + z[i]
        for i in range(N):
            for j in range(N):
                w[i] = w[i] + alpha * A[i, j] * x[j]

    A = _f32(0, N, N)
    u1, u2, v1, v2 = _f32(1, N), _f32(2, N), _f32(3, N), _f32(4, N)
    x, y, z = _f32(5, N), _f32(6, N), _f32(7, N)
    w = np.zeros(N, np.float32)
    Ag, xg, wg = A.copy(), x.copy(), w.copy()
    Ag = Ag + np.outer(u1, v1) + np.outer(u2, v2)
    xg = xg + np.float32(beta) * (Ag.T @ y) + z
    wg = wg + np.float32(alpha) * (Ag @ xg)
    _to_rtl(gemver).cosim(A, u1, u2, v1, v2, x, y, w, z)
    assert np.allclose(A, Ag, **FTOL)
    assert np.allclose(x, xg, **FTOL)
    assert np.allclose(w, wg, **FTOL)

    # gesummv: tt and yy accumulate in the same inner body, then a second kernel
    # combines them through a handoff buffer.
    G = 5

    @kernel
    def compute_tmp(
        y_in: f32[G], y_out: f32[G], A: f32[G, G], B: f32[G, G], x: f32[G], tmp: f32[G]
    ):
        tt: f32[G] = 0.0
        yy: f32[G]
        for i0 in range(G):
            yy[i0] = y_in[i0]
        for i in range(G):
            for j in range(G):
                tt[i] += A[i, j] * x[j]
                yy[i] += B[i, j] * x[j]
        for i1 in range(G):
            tmp[i1] = tt[i1]
            y_out[i1] = yy[i1]

    @kernel
    def compute_y(y_in: f32[G], y_out: f32[G], tmp: f32[G]):
        for i0 in range(G):
            y_out[i0] = alpha * tmp[i0] + beta * y_in[i0]

    @kernel
    def gesummv(A: f32[G, G], B: f32[G, G], x: f32[G], y: f32[G]):
        y_init: f32[G] = 0.0
        y_fifo: f32[G]
        tmp: f32[G]
        compute_tmp(y_init, y_fifo, A, B, x, tmp)
        compute_y(y_fifo, y, tmp)

    A, B, x = _f32(0, G, G), _f32(1, G, G), _f32(2, G)
    y = np.zeros(G, np.float32)
    _to_rtl(gesummv).cosim(A, B, x, y)
    assert np.allclose(
        y, np.float32(alpha) * (A @ x) + np.float32(beta) * (B @ x), **FTOL
    )


def test_if_conversion_in_loops_cosim():
    """trmm's triangular guard survives the nest coalescing that makes it
    quasi-affine in the one surviving IV; floyd_warshall's conditional store
    if-converts to a predicated read-modify-write; the relaxation is a real
    carried dependence, so the pipelined sweep must still agree with the
    sequential one."""
    # trmm: `flatten-perfect-loops` coalesces the nest and rewrites k1/i1 as
    # floordiv/mod of the surviving IV, so the guard is no longer a bound on any
    # loop -- only the if-conversion can honour it. Folding it into a bound
    # instead silently drops iterations, which a count-only body reads off
    # directly, so accumulate 1.0 rather than a product: any wrong trip count
    # shows up as an exact integer.
    M, N = 4, 5

    @kernel
    def tri_count(Cout: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        Cout[i1, j1] += 1.0

    Cout = np.zeros((M, N), np.float32)
    _to_rtl(tri_count).cosim(Cout)
    # Row i1 runs the guard for k1 in (i1, M) -> M - 1 - i1 accumulates.
    assert np.array_equal(
        Cout, np.tile((M - 1 - np.arange(M, dtype=np.float32))[:, None], (1, N))
    )

    # The same guard over a real triangular update: B[i1, j1] accumulates
    # A[k1, i1] * B[k1, j1] over the strict upper triangle, then scales.
    alpha = 1.5

    @kernel
    def trmm_S0(A: f32[M, M], B: f32[M, N]):
        for i1 in range(M):
            for j1 in range(N):
                for k1 in range(M):
                    if k1 > i1:
                        B[i1, j1] += A[k1, i1] * B[k1, j1]

    @kernel
    def trmm_S1(B: f32[M, N]):
        for i0 in range(M):
            for j0 in range(N):
                B[i0, j0] = B[i0, j0] * alpha

    @kernel
    def trmm(A: f32[M, M], B: f32[M, N]):
        trmm_S0(A, B)
        trmm_S1(B)

    A, B = _f32(0, M, M), _f32(1, M, N)
    g = B.copy()
    for i1 in range(M):
        for j1 in range(N):
            for k1 in range(M):
                if k1 > i1:
                    g[i1, j1] += A[k1, i1] * g[k1, j1]
    g *= np.float32(alpha)
    _to_rtl(trmm).cosim(A, B)
    assert np.allclose(B, g, **FTOL)

    # syrk's update_C: an if/ELSE over the same triangular region, so the guard
    # if-converts to a select between two speculated values rather than gating an
    # accumulate. Its nest coalesces too, so the predicate reads the surviving IV
    # through a delinearizing affine.apply.
    @kernel
    def update_C(Cin: f32[M, M], Cout: f32[M, M]):
        for i0 in range(M):
            for j0 in range(M):
                if j0 <= i0:
                    Cout[i0, j0] = alpha * Cin[i0, j0]
                else:
                    Cout[i0, j0] = Cin[i0, j0]

    Cin = _f32(2, M, M)
    Cout = np.zeros((M, M), np.float32)
    gc = np.where(
        np.arange(M)[None, :] <= np.arange(M)[:, None], np.float32(alpha) * Cin, Cin
    )
    _to_rtl(update_C).cosim(Cin, Cout)
    assert np.allclose(Cout, gc, **FTOL)

    F = 6

    @kernel
    def floyd_warshall(path: f32[F, F]):
        for k in range(F):
            for i in range(F):
                for j in range(F):
                    path_: f32 = path[i, k] + path[k, j]
                    if path[i, j] >= path_:
                        path[i, j] = path_

    # Positive edge weights, so the relaxation converges the way a distance
    # matrix should rather than running away negative.
    path = (np.abs(_f32(0, F, F)) + np.float32(0.5)).astype(np.float32)
    g = path.copy()
    for k in range(F):
        for i in range(F):
            for j in range(F):
                p = g[i, k] + g[k, j]
                if g[i, j] >= p:
                    g[i, j] = p
    _to_rtl(floyd_warshall).cosim(path)
    assert np.allclose(path, g, **FTOL)
