# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for kernels transcribed from examples/machsuite.

The scheduling tests run the kernels at their reference problem sizes; the
``_cosim`` tests re-state them shrunk to a size verilator can simulate and drive
the emitted RTL against a NumPy golden. An array return has no hardware meaning,
so a kernel that returns one is re-stated with an explicit out-parameter.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import allo  # noqa: E402
from allo import kernel  # noqa: E402
from allo.lang import i32, f32, f64, u8, index  # noqa: E402
from tests.rtl._common import (  # noqa: E402
    _sched,
    _to_rtl,
    _iis,
    FADD,
    MEM_REDUCE_II,
)

# The scheduling tests are pure compiler queries, so only the cosim ones need a
# simulator.
pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def test_runtime_vs_static_bounds():
    """A runtime-bounded loop leaves the whole-kernel latency unknown; a
    statically-bounded one resolves it. Both still pipeline."""
    M, N, K, S = 64, 64, 64, 8

    # bbgemm: tile bounds (`i_max = min(i+S, M)`) make the inner loops
    # runtime-bounded scf.for with non-affine accesses.
    @kernel
    def bbgemm(A: i32[M, K], B: i32[K, N]) -> i32[M, N]:
        C: i32[M, N] = 0
        i_max: i32 = 0
        j_max: i32 = 0
        k_max: i32 = 0
        sum_value: i32 = 0
        for i in range(0, M, S):
            i_max = i + S if i + S < M else M
            for j in range(0, N, S):
                j_max = j + S if j + S < N else N
                for k in range(0, K, S):
                    k_max = k + S if k + S < K else K
                    for ii in range(i, i_max):
                        for jj in range(j, j_max):
                            sum_value = 0
                            for kk in range(k, k_max):
                                sum_value += A[ii, kk] * B[kk, jj]
                            C[ii, jj] += sum_value
        return C

    res = _sched(bbgemm)
    assert res.func("bbgemm").latency is None  # runtime tile bounds
    assert any(r.has("muli") for r in res.cyclic())  # the matmul pipelines

    # CRS sparse mat-vec: the inner loop runs between row-pointer bounds loaded
    # from memory (a runtime-trip scf.for) and gathers through vec[cols[j]].
    SN, NNZ = 64, 192

    @kernel
    def crs(val: f64[NNZ], cols: i32[NNZ], row: i32[SN + 1], vec: f64[SN]) -> f64[SN]:
        out: f64[SN] = 0.0
        for i in range(SN):
            tmp_begin: i32 = row[i]
            tmp_end: i32 = row[i + 1]
            for j in range(tmp_begin, tmp_end):
                out[i] += val[j] * vec[cols[j]]
        return out

    res = _sched(crs)
    assert res.func("crs").latency is None  # dynamic trip -> latency omitted
    assert len(res.func("crs").cyclic()) >= 1

    # ELLPACK has a static inner bound but a per-element validity guard, which
    # if-converts to a select-gated accumulate; the static bounds give a latency.
    L = 4

    @kernel
    def ellpack(NZ: f64[SN * L], cols: i32[SN * L], vec: f64[SN]) -> f64[SN]:
        out: f64[SN] = 0.0
        for i in range(SN):
            for j in range(L):
                idx: i32 = j + i * L
                if cols[idx] != -1:
                    out[i] += NZ[idx] * vec[cols[idx]]
        return out

    res = _sched(ellpack)
    assert res.func("ellpack").latency is not None  # static trip
    assert any(r.has("select") for r in res.func("ellpack").cyclic())


def test_dynamic_programming():
    """DP kernels: many loop nests with long recurrences, all statically bounded
    so the latency resolves."""
    ALEN = BLEN = 32
    RESULT_LEN = ALEN + BLEN
    MATRIX_SIZE = (ALEN + 1) * (BLEN + 1)
    MATCH_SCORE, MISMATCH_SCORE, GAP_SCORE = 1, -1, -1
    ALIGN_VAL, SKIPA_VAL, SKIPB_VAL = 1, 2, 3

    # Needleman-Wunsch: fill a scoring matrix, then trace back.
    @kernel
    def needwun(SEQA: i32[ALEN], SEQB: i32[BLEN]) -> i32[2, RESULT_LEN]:
        M: i32[MATRIX_SIZE] = 0
        ptr: i32[MATRIX_SIZE] = 0
        result: i32[2, RESULT_LEN] = 0

        score: i32 = 0
        row_up: i32 = 0
        row: i32 = 0
        up_left: i32 = 0
        up: i32 = 0
        left: i32 = 0
        max_val: i32 = 0

        for i in range(ALEN + 1):
            M[i] = i * GAP_SCORE
        for j in range(BLEN + 1):
            M[j * (ALEN + 1)] = j * GAP_SCORE

        for bi in range(1, BLEN + 1):
            for ai in range(1, ALEN + 1):
                if SEQA[ai - 1] == SEQB[bi - 1]:
                    score = MATCH_SCORE
                else:
                    score = MISMATCH_SCORE

                row_up = (bi - 1) * (ALEN + 1)
                row = bi * (ALEN + 1)

                up_left = M[row_up + (ai - 1)] + score
                up = M[row_up + ai] + GAP_SCORE
                left = M[row + (ai - 1)] + GAP_SCORE

                max_val = up_left
                if up > max_val:
                    max_val = up
                if left > max_val:
                    max_val = left

                M[row + ai] = max_val
                if max_val == left:
                    ptr[row + ai] = SKIPB_VAL
                elif max_val == up:
                    ptr[row + ai] = SKIPA_VAL
                else:
                    ptr[row + ai] = ALIGN_VAL

        a_idx: i32 = ALEN
        b_idx: i32 = BLEN
        a_str_idx: i32 = 0
        b_str_idx: i32 = 0
        r: i32 = 0

        for step in range(ALEN + BLEN):
            if a_idx > 0 or b_idx > 0:
                if a_idx == 0:
                    result[0, a_str_idx] = 45
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    b_idx = b_idx - 1
                elif b_idx == 0:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = 45
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                else:
                    r = b_idx * (ALEN + 1)
                    if ptr[r + a_idx] == ALIGN_VAL:
                        result[0, a_str_idx] = SEQA[a_idx - 1]
                        result[1, b_str_idx] = SEQB[b_idx - 1]
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        a_idx = a_idx - 1
                        b_idx = b_idx - 1
                    elif ptr[r + a_idx] == SKIPB_VAL:
                        result[0, a_str_idx] = SEQA[a_idx - 1]
                        result[1, b_str_idx] = 45
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        a_idx = a_idx - 1
                    else:
                        result[0, a_str_idx] = 45
                        result[1, b_str_idx] = SEQB[b_idx - 1]
                        a_str_idx = a_str_idx + 1
                        b_str_idx = b_str_idx + 1
                        b_idx = b_idx - 1

        for idx in range(RESULT_LEN):
            if result[0, idx] == 0:
                result[0, idx] = 95
            if result[1, idx] == 0:
                result[1, idx] = 95

        return result

    f = _sched(needwun).func("needwun")
    assert f.latency is not None
    cyclic = [r for r in f.regions if r.kind == "cyclic"]
    assert len(cyclic) >= 4
    assert max(r.ii for r in cyclic) > FADD

    # Viterbi: dependence-free per-state updates across several regions.
    N_OBS, N_STATES, N_TOKENS = 32, 16, 16

    @kernel
    def viterbi(
        obs: i32[N_OBS],
        init: f32[N_STATES],
        transition: f32[N_STATES, N_STATES],
        emission: f32[N_STATES, N_TOKENS],
    ) -> i32[N_OBS]:
        llike: f32[N_OBS, N_STATES]

        for s in range(N_STATES):
            llike[0, s] = init[s] + emission[s, obs[0]]

        for t in range(1, N_OBS):
            for curr in range(N_STATES):
                min_p: f32 = (
                    llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
                )
                for prev in range(1, N_STATES):
                    p: f32 = (
                        llike[t - 1, prev]
                        + transition[prev, curr]
                        + emission[curr, obs[t]]
                    )
                    if p < min_p:
                        min_p = p
                llike[t, curr] = min_p

        min_s: i32 = 0
        min_p: f32 = llike[N_OBS - 1, 0]
        for s in range(1, N_STATES):
            p: f32 = llike[N_OBS - 1, s]
            if p < min_p:
                min_p = p
                min_s = s

        path: i32[N_OBS]
        path[N_OBS - 1] = min_s

        for t in range(N_OBS - 1):
            actual_t: i32 = N_OBS - 2 - t
            min_s = 0
            min_p = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
            for s in range(1, N_STATES):
                p: f32 = llike[actual_t, s] + transition[s, path[actual_t + 1]]
                if p < min_p:
                    min_p = p
                    min_s = s
            path[actual_t] = min_s

        return path

    f = _sched(viterbi).func("viterbi")
    assert f.latency is not None
    assert len([r for r in f.regions if r.kind == "cyclic"]) >= 1


def test_while_loops():
    """A data-dependent `while` schedules as a conditional (flushing) pipeline and
    leaves the latency unknown; no raw scf.while survives into the DCP IR."""
    S, P = 32, 16

    # KMP: two data-dependent while loops (failure-function backtracking) nested
    # in counted for loops.
    @kernel
    def kmp(pattern: u8[P], input_str: u8[S], kmp_next: u8[P], matches: u8[1]):
        k: index = 0
        x: index = 1
        for i in range(P - 1):
            while k > 0 and pattern[k] != pattern[x]:
                k = kmp_next[k - 1]
            if pattern[k] == pattern[x]:
                k += 1
            kmp_next[x] = k
            x += 1
        q: index = 0
        for i in range(S):
            while q > 0 and pattern[q] != input_str[i]:
                q = kmp_next[q - 1]
            if pattern[q] == input_str[i]:
                q += 1
            if q >= P:
                matches[0] += 1
                q = kmp_next[q - 1]

    res = _sched(kmp)
    assert res.func("kmp").latency is None  # data-dependent while trips
    assert len([r for r in res.cyclic() if r.conditional]) == 2  # both whiles

    # bfs_queue: an uncounted `while front != rear` whose body holds a nested
    # data-dependent `for e` carrying the queue tail as an iter-arg. The scatter
    # loop schedules as its own pipeline; the while closes into a sequential
    # (data-dependent length) dcp.pipeline, so it carries no static II.
    N_NODES = 32
    N_NODES_2 = N_NODES * 2
    N_EDGES = 128
    N_LEVELS = 6
    MAX_LEVEL = 999999

    @kernel
    def bfs_queue(
        nodes: i32[N_NODES_2], edges: i32[N_EDGES], starting_node: i32
    ) -> (i32[N_NODES], i32[N_LEVELS]):
        level: i32[N_NODES] = MAX_LEVEL
        level_counts: i32[N_LEVELS] = 0
        queue: i32[N_NODES] = 0
        front: i32 = 0
        rear: i32 = 0
        level[starting_node] = 0
        level_counts[0] = 1
        queue[rear] = starting_node
        rear = (rear + 1) % N_NODES
        while front != rear:
            n: i32 = queue[front]
            front = (front + 1) % N_NODES
            tmp_begin: i32 = nodes[2 * n]
            tmp_end: i32 = nodes[2 * n + 1]
            for e in range(tmp_begin, tmp_end):
                tmp_dst: i32 = edges[e]
                tmp_level: i32 = level[tmp_dst]
                if tmp_level == MAX_LEVEL:
                    tmp_level = level[n] + 1
                    level[tmp_dst] = tmp_level
                    level_counts[tmp_level] += 1
                    queue[rear] = tmp_dst
                    rear = (rear + 1) % N_NODES
        return level, level_counts

    mod = _to_rtl(bfs_queue)
    res = mod.schedule()
    assert len(res.cyclic()) >= 1  # the nested scatter loop got its own pipeline
    # A region nested in an scf.while reports an unknown execution count, so the
    # whole-kernel latency stays unknown.
    assert res.func("bfs_queue").latency is None
    assert "scf.while" not in mod.dcp and "allo.dcp.condition" in mod.dcp
    # The while wraps the scatter, so it is a container region.
    guard = next(r for r in res.cyclic(wrappers=True) if r.conditional)
    assert guard.ii is None  # sequential (data-dependent length), no static II


def test_grid_parallel():
    """`allo.grid` lowers to a nested affine.for band that the whole scheduling
    pipeline handles: constant trips give a static latency, and a real
    memory-carried recurrence still closes despite the grid's nodep hint."""
    P = 64

    # The canonical grid() matmul: C[i, j] is affine, so the grid's assume.nodep
    # does not touch the real k-reduction recurrence.
    @kernel
    def gemm(A: f32[P, P], B: f32[P, P]) -> f32[P, P]:
        C: f32[P, P] = 0.0
        for i, j in allo.grid(P, P):
            for k in range(P):
                C[i, j] += A[i, k] * B[k, j]
        return C

    res = _sched(gemm)
    assert res.func("gemm").latency is not None
    assert res.func("gemm").cyclic()[-1].ii == MEM_REDUCE_II

    # A 2-D grid stencil: the 3x3 window accumulation pipelines at II=1 and the
    # write to a distinct sol[i, j] per iteration carries no dependence.
    ROW, COL, F = 32, 32, 9

    @kernel
    def stencil2d(orig: i32[ROW, COL], filt: i32[F]) -> i32[ROW, COL]:
        sol: i32[ROW, COL] = 0
        for i, j in allo.grid(ROW - 2, COL - 2):
            temp: i32 = 0
            for m in range(3):
                for n in range(3):
                    mul: i32 = filt[m * 3 + n] * orig[i + m, j + n]
                    temp += mul
            sol[i, j] = temp
        return sol

    res = _sched(stencil2d)
    assert res.func("stencil2d").latency is not None
    assert res.cyclic() and all(r.ii == 1 for r in res.cyclic())

    # A 3-D grid stencil: three boundary-copy grids plus one interior 6-neighbor
    # accumulation grid; the boundary copies pipeline at II=1.
    R, C, H = 8, 16, 16

    @kernel
    def stencil3d(coeff: i32[2], orig: i32[R, C, H]) -> i32[R, C, H]:
        sol: i32[R, C, H] = 0
        for j, k in allo.grid(C, R):
            sol[k, j, 0] = orig[k, j, 0]
            sol[k, j, H - 1] = orig[k, j, H - 1]
        for i, k in allo.grid(H - 1, R):
            sol[k, 0, i + 1] = orig[k, 0, i + 1]
            sol[k, C - 1, i + 1] = orig[k, C - 1, i + 1]
        for j, i in allo.grid(C - 2, H - 2):
            sol[0, j + 1, i + 1] = orig[0, j + 1, i + 1]
            sol[R - 1, j + 1, i + 1] = orig[R - 1, j + 1, i + 1]
        for i, j, k in allo.grid(H - 2, C - 2, R - 2):
            sum0: i32 = orig[k + 1, j + 1, i + 1]
            sum1: i32 = (
                orig[k + 1, j + 1, i + 2]
                + orig[k + 1, j + 1, i]
                + orig[k + 1, j + 2, i + 1]
                + orig[k + 1, j, i + 1]
                + orig[k + 2, j + 1, i + 1]
                + orig[k, j + 1, i + 1]
            )
            sol[k + 1, j + 1, i + 1] = sum0 * coeff[0] + sum1 * coeff[1]
        return sol

    res = _sched(stencil3d)
    assert res.func("stencil3d").latency is not None
    assert min(_iis(res.cyclic())) == 1  # boundary copies fully pipelined


def test_double_precision_divide():
    """A double-precision force computation with division exercises the f64
    datapath and the multi-cycle divide."""
    nAtoms, maxNeighbors = 64, 8
    lj1, lj2, domainEdge = 1.5, 2.0, 20.0

    @kernel
    def md_x(
        position_x: f64[nAtoms],
        position_y: f64[nAtoms],
        position_z: f64[nAtoms],
        NL: i32[nAtoms * maxNeighbors],
    ) -> f64[nAtoms]:
        i_x: f64 = 0.0
        i_y: f64 = 0.0
        i_z: f64 = 0.0
        jidx: i32 = 0
        j_x: f64 = 0.0
        j_y: f64 = 0.0
        j_z: f64 = 0.0
        delx: f64 = 0.0
        dely: f64 = 0.0
        delz: f64 = 0.0
        r2inv: f64 = 0.0
        r6inv: f64 = 0.0
        potential: f64 = 0.0
        force: f64 = 0.0
        fx: f64 = 0.0
        force_x: f64[nAtoms] = 0.0

        for i in range(nAtoms):
            i_x = position_x[i]
            i_y = position_y[i]
            i_z = position_z[i]
            fx = 0.0

            for j in range(maxNeighbors):
                jidx = NL[i * maxNeighbors + j]
                j_x = position_x[jidx]
                j_y = position_y[jidx]
                j_z = position_z[jidx]
                delx = i_x - j_x
                dely = i_y - j_y
                delz = i_z - j_z
                if (delx * delx + dely * dely + delz * delz) == 0:
                    r2inv = (domainEdge * domainEdge * 3.0) * 1000
                else:
                    r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
                r6inv = r2inv * r2inv * r2inv
                potential = r6inv * (lj1 * r6inv - lj2)
                force = r2inv * potential
                fx = fx + delx * force
            force_x[i] = fx
        return force_x

    res = _sched(md_x)
    assert res.cyclic()  # pipelines
    assert any(r.has("divf") for r in res.cyclic())  # f64 reciprocal


def test_port_bound_ii_read_write_same_array():
    """A load and a store contending for one array's ports bound the II by
    resource, not by operator type: `weights` sees 3 reads + 1 write per
    iteration over 2 ports, so II = ceil(4/2) = 2."""
    IN, NPL, LR = 13, 16, 2

    # Integer accumulate, so the norm recurrence is combinational and the port
    # oversubscription -- the actual subject -- is the binding constraint.
    @kernel
    def wnorm(weights: i32[IN * NPL], dweights: i32[IN * NPL]) -> i32:
        norm: i32 = 0
        for i in range(IN):
            for j in range(NPL):
                weights[i * NPL + j] -= dweights[i * NPL + j] * LR
                norm += weights[i * NPL + j] * weights[i * NPL + j]
        return norm

    assert _iis(_sched(wnorm).cyclic()) == [2]


def test_runtime_vs_static_bounds_cosim():
    """CRS drives correctly end to end. Its inner loop's trip comes from a pair of
    row-pointer loads, so it is a runtime-bounded scf.for rather than an
    affine.for -- but the accumulate into out[i] is still a memory-carried
    recurrence that loop has to serialize. Left unserialized it pipelines at II=1
    and every iteration in one recurrence window reads the same stale out[i], so
    only one accumulate per window survives. An empty row (row[i] == row[i + 1])
    covers the zero-trip case."""
    SN, NNZ = 4, 6

    @kernel
    def crs(
        val: f64[NNZ], cols: i32[NNZ], row: i32[SN + 1], vec: f64[SN], out: f64[SN]
    ):
        for i in range(SN):
            tmp_begin: i32 = row[i]
            tmp_end: i32 = row[i + 1]
            for j in range(tmp_begin, tmp_end):
                out[i] += val[j] * vec[cols[j]]

    # Row 1 is empty, and rows 0/2/3 hold two non-zeros each, so a dropped
    # accumulate cannot hide behind a single-element row.
    dense = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ],
        np.float64,
    )
    r, c = np.nonzero(dense)
    val = dense[r, c].copy()
    cols = c.astype(np.int32)
    row = np.concatenate(([0], np.cumsum((dense != 0).sum(axis=1)))).astype(np.int32)
    assert row[-1] == NNZ and row[1] == row[2]  # the CSR the kernel is given
    vec = np.array([1.5, -2.0, 0.25, 3.0], np.float64)
    out = np.zeros(SN, np.float64)
    _to_rtl(crs).cosim(val, cols, row, vec, out)
    assert np.allclose(out, dense @ vec)

    # ELLPACK is the static-bound counterpart: a per-element validity guard
    # rather than a runtime trip. Its nest coalesces, so the guard's operands
    # come back as an affine.apply delinearizing the surviving IV.
    L = 4

    @kernel
    def ellpack(NZ: f64[SN * L], cols_e: i32[SN * L], vec_e: f64[SN], out_e: f64[SN]):
        for i in range(SN):
            for j in range(L):
                idx: i32 = j + i * L
                if cols_e[idx] != -1:
                    out_e[i] += NZ[idx] * vec_e[cols_e[idx]]

    rng = np.random.default_rng(0)
    NZ = rng.random(SN * L)
    cols_e = rng.integers(0, SN, SN * L).astype(np.int32)
    cols_e[3] = cols_e[9] = -1  # invalid slots the guard must mask out
    vec_e = rng.random(SN)
    out_e = np.zeros(SN, np.float64)
    g = np.zeros(SN, np.float64)
    for i in range(SN):
        for j in range(L):
            idx = j + i * L
            if cols_e[idx] != -1:
                g[i] += NZ[idx] * vec_e[cols_e[idx]]
    _to_rtl(ellpack).cosim(NZ, cols_e, vec_e, out_e)
    assert np.allclose(out_e, g, rtol=2e-3, atol=2e-3)


def test_grid_parallel_cosim():
    """The grid() matmul drives correctly end to end: the band coalesces to one
    counted loop whose index delinearizes back to (i, j), and the k-reduction
    still serializes into C[i, j] despite the grid's nodep hint."""
    P = 8

    @kernel
    def gemm(A: f32[P, P], B: f32[P, P], C: f32[P, P]):
        for i, j in allo.grid(P, P):
            for k in range(P):
                C[i, j] += A[i, k] * B[k, j]

    rng = np.random.default_rng(0)
    A = (rng.random((P, P), dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random((P, P), dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    C = np.zeros((P, P), np.float32)  # a pure-output buffer is zero-inited by cosim
    _to_rtl(gemm).cosim(A, B, C)
    # f32 accumulation reassociates in hardware, so compare to a tolerance.
    assert np.allclose(C, A @ B, rtol=2e-3, atol=2e-3)


def test_double_precision_divide_cosim():
    """The f64 datapath end to end: the multi-cycle divide, the guard selecting
    between the reciprocal and its fallback, and a gather through NL[...] all have
    to land on the same cycle budget for `fx` to accumulate the right value."""
    nAtoms, maxNeighbors = 8, 4
    lj1, lj2, domainEdge = 1.5, 2.0, 20.0

    @kernel
    def md_x(
        position_x: f64[nAtoms],
        position_y: f64[nAtoms],
        position_z: f64[nAtoms],
        NL: i32[nAtoms * maxNeighbors],
        force_x: f64[nAtoms],
    ):
        i_x: f64 = 0.0
        i_y: f64 = 0.0
        i_z: f64 = 0.0
        jidx: i32 = 0
        j_x: f64 = 0.0
        j_y: f64 = 0.0
        j_z: f64 = 0.0
        delx: f64 = 0.0
        dely: f64 = 0.0
        delz: f64 = 0.0
        r2inv: f64 = 0.0
        r6inv: f64 = 0.0
        potential: f64 = 0.0
        force: f64 = 0.0
        fx: f64 = 0.0

        for i in range(nAtoms):
            i_x = position_x[i]
            i_y = position_y[i]
            i_z = position_z[i]
            fx = 0.0

            for j in range(maxNeighbors):
                jidx = NL[i * maxNeighbors + j]
                j_x = position_x[jidx]
                j_y = position_y[jidx]
                j_z = position_z[jidx]
                delx = i_x - j_x
                dely = i_y - j_y
                delz = i_z - j_z
                if (delx * delx + dely * dely + delz * delz) == 0:
                    r2inv = (domainEdge * domainEdge * 3.0) * 1000
                else:
                    r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
                r6inv = r2inv * r2inv * r2inv
                potential = r6inv * (lj1 * r6inv - lj2)
                force = r2inv * potential
                fx = fx + delx * force
            force_x[i] = fx

    rng = np.random.default_rng(0)
    px, py, pz = (rng.standard_normal(nAtoms) for _ in range(3))
    # A neighbour list that includes i itself for some i, so the ==0 guard (the
    # self-distance) is actually exercised rather than always taking the divide.
    NL = rng.integers(0, nAtoms, size=nAtoms * maxNeighbors).astype(np.int32)
    NL[0] = 0

    exp = np.zeros(nAtoms, np.float64)
    for i in range(nAtoms):
        fx = 0.0
        for j in range(maxNeighbors):
            jidx = int(NL[i * maxNeighbors + j])
            dx, dy, dz = px[i] - px[jidx], py[i] - py[jidx], pz[i] - pz[jidx]
            r2 = dx * dx + dy * dy + dz * dz
            r2inv = (domainEdge * domainEdge * 3.0) * 1000 if r2 == 0 else 1.0 / r2
            r6inv = r2inv * r2inv * r2inv
            fx += dx * (r2inv * (r6inv * (lj1 * r6inv - lj2)))
        exp[i] = fx

    force_x = np.zeros(nAtoms, np.float64)
    _to_rtl(md_x).cosim(px, py, pz, NL, force_x)
    assert np.allclose(force_x, exp, rtol=1e-2, atol=1e-2)


def test_port_bound_ii_read_write_same_array_cosim():
    """The port-bound loop drives correctly: `weights` is updated in place and
    read back for the norm in the same iteration, so time-sharing the two ports
    must not let the norm see a stale (pre-update) value. The norm comes back on
    the scalar result port, sampled at done."""
    IN, NPL, LR = 4, 4, 2

    @kernel
    def wnorm(weights: i32[IN * NPL], dweights: i32[IN * NPL]) -> i32:
        norm: i32 = 0
        for i in range(IN):
            for j in range(NPL):
                weights[i * NPL + j] -= dweights[i * NPL + j] * LR
                norm += weights[i * NPL + j] * weights[i * NPL + j]
        return norm

    rng = np.random.default_rng(0)
    weights = rng.integers(0, 8, size=IN * NPL).astype(np.int32)
    dweights = rng.integers(0, 8, size=IN * NPL).astype(np.int32)
    exp_w = (weights - dweights * LR).astype(np.int32)
    exp_norm = int(np.sum(exp_w.astype(np.int64) ** 2))

    r = _to_rtl(wnorm).cosim(weights, dweights)
    assert np.array_equal(weights, exp_w)
    assert r.result == exp_norm
