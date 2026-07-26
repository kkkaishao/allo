# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`for`-loop structure, scheduling, and the shared iteration-control controller."""

import os
import shutil
import sys

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, index

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    Mod,
    _sched,
    _to_rtl,
    _latency,
    _iis,
    FDIV,
    MEM,
    MEM_REDUCE_II,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
A44 = (np.arange(16, dtype=np.int32) * 3 + 1).reshape(4, 4) & 0x3F


def _f32(*shape):
    return np.random.default_rng(0).random(shape, dtype=np.float32)


# --- loop structures ---------------------------------------------------------


# The loop shapes around a datapath: a 2-D nest, sibling loops chained
# through an array, a reduction nest, and a loop-free straight line.
def test_loop_structures():
    @kernel
    def nest(A: i32[4, 4], out: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                out[i, j] = A[i, j] & 5

    A2 = ((np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 4)
    out = np.zeros((4, 4), np.int32)
    _to_rtl(nest).cosim(A2, out)
    assert np.array_equal(out, A2 & 5)

    @kernel
    def two(A: i32[8], B: i32[8], C: i32[8]):
        for i in range(8):
            B[i] = A[i] & 5
        for i in range(8):
            C[i] = B[i] & A[i]

    A8 = (np.arange(8, dtype=np.int32) * 7 + 13) & 0xFF
    B = np.zeros(8, np.int32)
    C = np.zeros(8, np.int32)
    _to_rtl(two).cosim(A8, B, C)
    assert np.array_equal(B, A8 & 5)
    assert np.array_equal(C, (A8 & 5) & A8)

    # A container `for i` with two children per outer iteration -- an inner
    # store-less reduction and a store of its result -- exercising multi-child
    # container sequencing, the cross-child survivor, and the retriggered
    # accumulator (its init re-injected each row).
    @kernel
    def rowor(A: i32[4, 4], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for j in range(4):
                acc = acc | A[i, j]
            out[i] = acc

    out = np.zeros(4, np.int32)
    _to_rtl(rowor).cosim(A2, out)
    assert np.array_equal(out, np.bitwise_or.reduce(A2, axis=1))

    # No loop -> one acyclic (dcp.sequential) region; each array touched once.
    @kernel
    def strl(A: i32[1], B: i32[1], C: i32[1], D: i32[1], o1: i32[1], o2: i32[1]):
        o1[0] = A[0] & B[0]
        o2[0] = C[0] & D[0]

    a, b, c, d = (np.array([v], np.int32) for v in (0xF0, 0xFF, 0x3C, 0x0F))
    o1, o2 = np.zeros(1, np.int32), np.zeros(1, np.int32)
    _to_rtl(strl).cosim(a, b, c, d, o1, o2)
    assert o1[0] == (0xF0 & 0xFF) and o2[0] == (0x3C & 0x0F)


# The scheduled IR is closed over the dcp dialect: every counted loop (an
# imperfect-nest wrapper or a non-flattenable dynamic band) materializes into
# a dcp.pipeline, so no raw affine.for / scf.for survives.
def test_residual_loops_closed_into_pipelines():
    N = 8

    @kernel
    def imperfect(A: f32[N, N], x: f32[N], y: f32[N]):
        for i in range(N):
            y[i] = 0.0
            for j in range(N):
                y[i] += A[i, j] * x[j]

    mod = _to_rtl(imperfect)
    # An outer sequential wrapper (ii = body length) around the inner pipeline.
    assert any(r.is_wrapper for r in mod.schedule().funcs[0].regions)

    @kernel
    def band(A: f32[N, N], y: f32[N], n: index):
        for i in range(n):  # dynamic trip -> band cannot be flattened
            for j in range(N):
                y[i] += A[i, j]

    mod = _to_rtl(band)
    # Dynamic outer trip: the wrapper's II is still concrete (inner-derived), but
    # its trip is unknown.
    wrapper = next(r for r in mod.schedule().funcs[0].regions if r.is_wrapper)
    assert wrapper.trip is None and wrapper.ii > 0

    @kernel
    def dyn_inner(A: f32[N, N], y: f32[N], n: index):
        for i in range(N):  # static outer
            for j in range(n):  # DYNAMIC inner trip -> body length data-dependent
                y[i] += A[i, j]

    mod = _to_rtl(dyn_inner)
    # The outer wrapper's body length is data-dependent, so it carries no static
    # II (done-based sequential controller), but the loop still closes into a
    # dcp.pipeline (its static trip is known).
    wrapper = next(r for r in mod.schedule().funcs[0].regions if r.is_wrapper)
    assert wrapper.ii is None and wrapper.trip == N


def test_imperfect_reduction_nest_cosim():
    # The `imperfect` matvec above -- an imperfect nest (init prologue `y[i]=0.0` +
    # scalar-carried inner reduction) closed into a sequential-wrapper pipeline --
    # run end-to-end: pins that the residual-loop closure computes y = A @ x.
    N = 8

    @kernel
    def imperfect(A: f32[N, N], x: f32[N], y: f32[N]):
        for i in range(N):
            y[i] = 0.0
            for j in range(N):
                y[i] += A[i, j] * x[j]

    A = (np.arange(N * N, dtype=np.float32) * 0.1).reshape(N, N)
    x = np.arange(N, dtype=np.float32) * 0.1 + 1.0
    y = np.zeros(N, np.float32)
    _to_rtl(imperfect).cosim(A, x, y)
    assert np.allclose(y, A @ x, rtol=1e-3, atol=1e-3)


# --- dynamic trip counts & lb/step induction ---------------------------------


# A memory-loaded bound is not affine, so the loop stays a runtime-trip band:
# it still pipelines, but the latency is deferred rather than faked. A carried
# memory recurrence under such a bound is closed conservatively.
def test_dynamic_trip_scheduling():
    @kernel
    def dyn(A: i32[128], out: i32[1]):
        n: index = A[0]
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    loop = _sched(dyn).cyclic()[0]
    assert loop.ii == 1  # scalar int accumulate, add is combinational
    assert loop.latency is None  # unknown trip -> latency deferred, not faked

    @kernel
    def recur(A: i32[128], nb: i32[1]):
        n: index = nb[0]
        for i in range(1, n):
            A[i] = A[i - 1] + A[i]

    # A[i] reads A[i-1]: a conservative distance-1 back edge forces II > 1;
    # without it the II would be an unsound, optimistic 1.
    assert _sched(recur).cyclic()[0].ii > 1


# A dynamic-trip loop stays a free-running / modulo pipeline terminating on
# counter == bound against a runtime value: the count is data, the
# per-iteration timing stays static. No stall, no flush.
def test_dynamic_trip_cosim():
    # Store-less reduction: the runtime bound free-runs the pipeline and its
    # result flows to the epilogue store (capture-based done).
    @kernel
    def dyn(A: i32[128], out: i32[1]):
        n: index = A[0]
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    for N in (5, 1, 12):
        A = np.zeros(128, np.int32)
        A[0] = N
        A[1:N] = np.arange(1, N, dtype=np.int32) * 3 + 2
        out = np.zeros(1, np.int32)
        _to_rtl(dyn).cosim(A, out)
        assert out[0] == int(A[:N].sum())

    # Store-ful: the store-counting done retires `bound` stores, gated by a
    # has-run latch so a runtime bound reading 0 at reset cannot fire done before
    # the loop issues.
    @kernel
    def dynstore(A: i32[64], nb: i32[1], out: i32[64]):
        n: index = nb[0]
        for i in range(n):
            out[i] = A[i] * 2

    for N in (7, 3):
        A = np.arange(64, dtype=np.int32) * 2 + 1
        out = np.zeros(64, np.int32)
        _to_rtl(dynstore).cosim(A, np.array([N], np.int32), out)
        assert np.array_equal(out[:N], A[:N] * 2)

    # Runtime bound on a modulo (II>1) pipeline: the float accumulate recurrence
    # forces II=FADD, and termination is `counter+1 == bound` on the issue.
    @kernel
    def dynfsum(A: f32[64], nb: i32[1], out: f32[1]):
        n: index = nb[0]
        s: f32 = 0.0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    for N in (7, 3):
        Af = np.arange(64, dtype=np.float32) * 0.5 + 1.0
        outf = np.zeros(1, np.float32)
        _to_rtl(dynfsum).cosim(Af, np.array([N], np.int32), outf)
        assert abs(float(outf[0]) - float(Af[:N].sum())) < 1e-3


def test_nonzero_lb_stencil_cosim():
    # An affine static stencil with lb=1 (no loop-carried dep). The induction
    # register runs the real IV (1..N-2), so it writes B[1..N-2] and reads
    # A[i-1..i+1] with no off-by-lb address shift.
    @kernel
    def jac(A: f32[8], B: f32[8]):
        for i in range(1, 7):
            B[i] = (A[i - 1] + A[i] + A[i + 1]) * 0.5

    A = _f32(8)
    B = np.zeros(8, np.float32)  # a pure-output buffer is zero-inited by cosim
    _to_rtl(jac).cosim(A, B)
    exp = np.zeros(8, np.float32)
    exp[1:7] = (A[:-2] + A[1:-1] + A[2:]) * 0.5  # B[0], B[7] stay 0 (untouched)
    assert np.allclose(B, exp, rtol=1e-4, atol=1e-5)


def test_runtime_lb_fixed_window_cosim():
    # The fixed-window idiom `for j in range(i, i+K)`: a RUNTIME lower bound (the
    # enclosing counter i) with a COMPILE-TIME trip K. The reifier keeps trip=K
    # and wires i as a runtime lbBound (no dynamicBound), so the loop's upper
    # bound must be computed as `lb + K*step` from the resolved runtime lb/step,
    # NOT the lb/step attributes (which default to 0/1 here). The i>=K iteration
    # is the telltale: an attribute-derived ub=K makes it spuriously empty.
    A = np.arange(16, dtype=np.int32)

    # (1) leaf window, step 1. i runs 0..3 with K=3, so i=3 is exactly the old
    #     spurious-empty edge (ub would have been konst(3), and lb=3 >= 3).
    @kernel
    def win(A: i32[16], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 3):
                s = s + A[j]
            out[i] = s

    out = np.zeros(4, np.int32)
    _to_rtl(win).cosim(A, out)
    assert np.array_equal(out, [A[i] + A[i + 1] + A[i + 2] for i in range(4)])

    # (2) non-unit step: the span is trip*step (=3*2), still anchored at runtime i.
    @kernel
    def stride(A: i32[16], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 6, 2):  # j = i, i+2, i+4
                s = s + A[j]
            out[i] = s

    out = np.zeros(4, np.int32)
    _to_rtl(stride).cosim(A, out)
    assert np.array_equal(out, [A[i] + A[i + 2] + A[i + 4] for i in range(4)])

    # (3) the window loop is a CONTAINER (its body nests a k loop), exercising the
    #     emitContainer terminatorOf path, not just the leaf pipeline.
    @kernel
    def wcont(A: i32[8, 4], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 3):
                for k in range(4):
                    s = s + A[j, k]
            out[i] = s

    A2 = np.arange(32, dtype=np.int32).reshape(8, 4)
    out = np.zeros(4, np.int32)
    _to_rtl(wcont).cosim(A2, out)
    assert np.array_equal(
        out,
        [
            sum(int(A2[j, k]) for j in range(i, i + 3) for k in range(4))
            for i in range(4)
        ],
    )


def test_negative_lb_signed_counter_cosim():
    # A compile-time NEGATIVE lower bound: the induction counter runs through
    # negative values (-4..3), so the bound tests (isLast/isEmpty) must be SIGNED.
    # An unsigned compare reads lb=-4 as ~4.3e9 >= ub, so `isEmpty` fires and the
    # whole loop is dropped; the all-8-outputs result proves it is not. `i` is
    # used both as a shifted address (A[i+4]) and a signed compute operand (+ i).
    @kernel
    def neglb(A: i32[8], out: i32[8]):
        for i in range(-4, 4):
            out[i + 4] = A[i + 4] + i

    A = np.arange(8, dtype=np.int32) * 10
    out = np.zeros(8, np.int32)
    _to_rtl(neglb).cosim(A, out)
    assert np.array_equal(out, [A[i + 4] + i for i in range(-4, 4)])

    # Negative lb on a reduction (II is the memory-carried recurrence): the
    # counter still seeds -4 and the signed bound test bounds the trip at 8.
    @kernel
    def neg_reduce(A: i32[8], out: i32[1]):
        acc: i32 = 0
        for i in range(-4, 4):
            acc = acc + A[i + 4] * i
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(neg_reduce).cosim(A, out)
    assert out[0] == sum(int(A[i + 4]) * i for i in range(-4, 4))


def test_heat_3d_stencil_cosim():
    # A 3-D 7-point stencil (polybench heat_3d, shrunk): a perfect nest whose i/j
    # loops are non-zero-lb *containers* and whose innermost k loop reads/writes at
    # the real IV. Cross-buffer (B from A, then A from B), so the two sweeps
    # sequence with no in-place recurrence; correctness turns on every nested
    # container counting from lb=1, not 0.
    N = 5

    @kernel
    def heat(A: f32[N, N, N], B: f32[N, N, N]):
        c0: f32 = 0.125
        c1: f32 = 2.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        c0 * (A[i + 1, j, k] - c1 * A[i, j, k] + A[i - 1, j, k])
                        + c0 * (A[i, j + 1, k] - c1 * A[i, j, k] + A[i, j - 1, k])
                        + c0 * (A[i, j, k + 1] - c1 * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    A[i, j, k] = (
                        c0 * (B[i + 1, j, k] - c1 * B[i, j, k] + B[i - 1, j, k])
                        + c0 * (B[i, j + 1, k] - c1 * B[i, j, k] + B[i, j - 1, k])
                        + c0 * (B[i, j, k + 1] - c1 * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )

    A = _f32(N, N, N)
    B = (_f32(N, N, N) + np.float32(0.5)).astype(np.float32)  # decorrelate B from A
    Ag, Bg = A.copy(), B.copy()
    c0, c1 = np.float32(0.125), np.float32(2.0)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                Bg[i, j, k] = (
                    c0 * (Ag[i + 1, j, k] - c1 * Ag[i, j, k] + Ag[i - 1, j, k])
                    + c0 * (Ag[i, j + 1, k] - c1 * Ag[i, j, k] + Ag[i, j - 1, k])
                    + c0 * (Ag[i, j, k + 1] - c1 * Ag[i, j, k] + Ag[i, j, k - 1])
                    + Ag[i, j, k]
                )
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                Ag[i, j, k] = (
                    c0 * (Bg[i + 1, j, k] - c1 * Bg[i, j, k] + Bg[i - 1, j, k])
                    + c0 * (Bg[i, j + 1, k] - c1 * Bg[i, j, k] + Bg[i, j - 1, k])
                    + c0 * (Bg[i, j, k + 1] - c1 * Bg[i, j, k] + Bg[i, j, k - 1])
                    + Bg[i, j, k]
                )
    _to_rtl(heat).cosim(A, B)
    assert np.allclose(A, Ag, rtol=2e-3, atol=2e-3)
    assert np.allclose(B, Bg, rtol=2e-3, atol=2e-3)


def test_seidel_2d_inplace_recurrence_cosim():
    # A 2-D in-place stencil (polybench seidel_2d, shrunk): A[i,j] reads its
    # already-updated neighbours A[i-1,*] and A[i,j-1], a genuine loop-carried
    # memory recurrence over a non-zero-lb nest. The recurrence forces II>1, which
    # serializes each read after the prior write, so the in-place sweep reproduces
    # the sequential result exactly.
    N = 6

    @kernel
    def seidel(A: f32[N, N]):
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

    A = _f32(N, N)
    Ag = A.copy()
    for i in range(1, N - 1):
        for j in range(1, N - 1):
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
    _to_rtl(seidel).cosim(A)
    assert np.allclose(A, Ag, rtol=2e-3, atol=2e-3)


def test_nested_reduction_container_cosim():
    # A reduction whose accumulator crosses TWO loop levels: the outer `for m`
    # loop is a counted *container* that carries `acc` into the inner `for n`
    # reduction. The container latches its iter-arg into a survivor register
    # (init at start, advanced by the inner loop's result each outer iteration),
    # which both the inner accumulator (its init) and the outer store read. A
    # single-level reduction (`for j: acc += …`) uses a fused accumulator instead.
    @kernel
    def red2(A: i32[4, 4, 4], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for m in range(4):
                for n in range(4):
                    acc = acc + A[i, m, n]
            out[i] = acc

    A = (np.arange(64, dtype=np.int32) % 7 + 1).reshape(4, 4, 4)
    out = np.zeros(4, np.int32)
    _to_rtl(red2).cosim(A.copy(), out)
    assert np.array_equal(out, A.reshape(4, -1).sum(axis=1))


def test_stencil2d_grid_reduction_cosim():
    # MachSuite stencil2d (shrunk): a 2-D grid() whose body accumulates a 3x3
    # window into `temp`, then stores it. Exercises two gaps together: the window
    # `for m: for n: temp += …` is the nested-reduction container above, and the
    # grid (ROW-2 = COL-2 = 3, a non-power-of-two) coalesces to one loop whose
    # index delinearizes by `div/mod 3` -- a general unsigned divide/remainder,
    # not a shift/mask. `sol` is written through an explicit out-parameter (the
    # interior; the caller-zeroed border stays 0).
    ROW, COL, F = 5, 5, 9

    @kernel
    def stencil2d(orig: i32[ROW, COL], filt: i32[F], sol: i32[ROW, COL]):
        for i, j in allo.grid(ROW - 2, COL - 2):
            temp: i32 = 0
            for m in range(3):
                for n in range(3):
                    mul: i32 = filt[m * 3 + n] * orig[i + m, j + n]
                    temp += mul
            sol[i, j] = temp

    orig = (np.arange(ROW * COL, dtype=np.int32) % 5 + 1).reshape(ROW, COL)
    filt = np.arange(F, dtype=np.int32) % 3 + 1
    sol = np.zeros((ROW, COL), np.int32)
    _to_rtl(stencil2d).cosim(orig.copy(), filt.copy(), sol)
    exp = np.zeros((ROW, COL), np.int32)
    for i in range(ROW - 2):
        for j in range(COL - 2):
            exp[i, j] = sum(
                int(filt[m * 3 + n]) * int(orig[i + m, j + n])
                for m in range(3)
                for n in range(3)
            )
    assert np.array_equal(sol, exp)


def test_stencil3d_grid_boundary_cosim():
    # MachSuite stencil3d (shrunk): four grid() sweeps -- three boundary copies
    # plus an interior 6-neighbour sum -- over a shared out-param `sol`. No nested
    # reduction (the body is straight-line), but the boundary grids coalesce to
    # non-power-of-two extents (C, R = 6) whose index delinearizes by `div/mod 6`,
    # while the interior grid (4x4x4) delinearizes by a power-of-two shift -- so
    # both address-lowering paths run.
    R, C, H = 6, 6, 6

    @kernel
    def stencil3d(coeff: i32[2], orig: i32[R, C, H], sol: i32[R, C, H]):
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

    coeff = np.array([2, 3], np.int32)
    orig = (np.arange(R * C * H, dtype=np.int32) % 5 + 1).reshape(R, C, H)
    sol = np.zeros((R, C, H), np.int32)
    _to_rtl(stencil3d).cosim(coeff.copy(), orig.copy(), sol)
    exp = np.zeros((R, C, H), np.int32)
    for j in range(C):
        for k in range(R):
            exp[k, j, 0] = orig[k, j, 0]
            exp[k, j, H - 1] = orig[k, j, H - 1]
    for i in range(H - 1):
        for k in range(R):
            exp[k, 0, i + 1] = orig[k, 0, i + 1]
            exp[k, C - 1, i + 1] = orig[k, C - 1, i + 1]
    for j in range(C - 2):
        for i in range(H - 2):
            exp[0, j + 1, i + 1] = orig[0, j + 1, i + 1]
            exp[R - 1, j + 1, i + 1] = orig[R - 1, j + 1, i + 1]
    for i in range(H - 2):
        for j in range(C - 2):
            for k in range(R - 2):
                s1 = int(
                    orig[k + 1, j + 1, i + 2]
                    + orig[k + 1, j + 1, i]
                    + orig[k + 1, j + 2, i + 1]
                    + orig[k + 1, j, i + 1]
                    + orig[k + 2, j + 1, i + 1]
                    + orig[k, j + 1, i + 1]
                )
                exp[k + 1, j + 1, i + 1] = int(orig[k + 1, j + 1, i + 1]) * 2 + s1 * 3
    assert np.array_equal(sol, exp)


# The induction register holds the real IV, so a non-zero lower bound or a
# non-unit step addresses without an off-by-lb shift and touches only the
# indices the loop actually visits.
def test_static_lb_and_step_cosim():
    # lb=2: the IV runs 2..15, so out[0..1] is left alone.
    @kernel
    def shifted(A: i32[16], out: i32[16]):
        for i in range(2, 16):
            out[i] = A[i] * 3

    out = np.zeros(16, np.int32)
    _to_rtl(shifted).cosim(A16, out)
    assert np.array_equal(out[2:], A16[2:] * 3)
    assert np.all(out[:2] == 0)

    # step=2 (lb=0): the IV runs 0,2,4,...,14.
    @kernel
    def stride2(A: i32[16], out: i32[16]):
        for i in range(0, 16, 2):
            out[i] = A[i] + 5

    out = np.zeros(16, np.int32)
    _to_rtl(stride2).cosim(A16, out)
    exp = np.zeros(16, np.int32)
    exp[0:16:2] = A16[0:16:2] + 5  # odd indices stay 0: only the even IV writes
    assert np.array_equal(out, exp)

    # A static empty loop (trip=0): lb >= ub, so it issues nothing and completes
    # on `start` rather than deadlocking (a store-drain done would never fire).
    # The sibling store must still run.
    @kernel
    def zt(A: i32[8], out: i32[8]):
        for i in range(1, 1):
            out[i] = A[i] + 99
        for i in range(8):
            out[i] = A[i] * 2

    out = np.zeros(8, np.int32)
    _to_rtl(zt).cosim(A16[:8].copy(), out)
    assert np.array_equal(out, A16[:8] * 2)


# A runtime lower bound / upper bound / stride is wired as a bound operand,
# so the induction register still runs the real IV. Includes the runtime
# zero-trip cases (lb >= ub), which must complete rather than deadlock.
def test_runtime_lb_and_step_cosim():
    # Constant lb=1 with a RUNTIME ub: the in-place recurrence must index
    # correctly (no spurious A[0] = A[-1] + A[0]). n==1 is the dynamic zero-trip.
    @kernel
    def recur(A: i32[16], nb: i32[1]):
        n: index = nb[0]
        for i in range(1, n):
            A[i] = A[i - 1] + A[i]

    for N in (16, 5, 1):
        A = (np.arange(16, dtype=np.int32) % 7 + 1).copy()
        exp = A.copy()
        for i in range(1, N):
            exp[i] = exp[i - 1] + exp[i]
        _to_rtl(recur).cosim(A, np.array([N], np.int32))
        assert np.array_equal(A, exp), N

    # BOTH bounds SSA. Swept including m==0 (the operand carries a runtime 0) and
    # m >= n (empty on runtime operands).
    @kernel
    def rng(A: i32[16], mb: i32[1], nb: i32[1], out: i32[16]):
        m: index = mb[0]
        n: index = nb[0]
        for i in range(m, n):
            out[i] = A[i] * 2

    for m, n in [(0, 16), (3, 12), (5, 6), (7, 7), (10, 3)]:
        out = np.zeros(16, np.int32)
        exp = np.zeros(16, np.int32)
        for i in range(m, n):
            exp[i] = A16[i] * 2
        _to_rtl(rng).cosim(
            A16.copy(), np.array([m], np.int32), np.array([n], np.int32), out
        )
        assert np.array_equal(out, exp), (m, n)

    # An SSA stride: the induction register advances 0, s, 2s, ...
    @kernel
    def rstep(A: i32[16], sb: i32[1], out: i32[16]):
        s: index = sb[0]
        for i in range(0, 16, s):
            out[i] = A[i] * 2

    for st in (1, 2, 3, 4):
        out = np.zeros(16, np.int32)
        exp = np.zeros(16, np.int32)
        for i in range(0, 16, st):
            exp[i] = A16[i] * 2
        _to_rtl(rstep).cosim(A16.copy(), np.array([st], np.int32), out)
        assert np.array_equal(out, exp), st

    # A zero-trip run RE-RUN by an enclosing loop (the CSR empty-row shape).
    # `done` is a latched level the container completes on the rising edge of, so
    # an empty run must let that level fall to 0 before rising again: completing
    # on `start` itself would hold it high from the previous (non-empty)
    # iteration and the container would wait forever for an edge that never
    # comes. Only the FIRST run starts with done already 0, so sweep where the
    # empty row sits -- last, first and interior take different paths.
    @kernel
    def rows(ptr: i32[4], out: i32[3]):
        for r in range(3):
            b: index = ptr[r]
            e: index = ptr[r + 1]
            for j in range(b, e):
                out[r] += 1

    for ptr in ([0, 2, 4, 6], [0, 2, 2, 4], [0, 0, 2, 4], [0, 2, 4, 4], [0, 0, 0, 0]):
        out = np.zeros(3, np.int32)
        _to_rtl(rows).cosim(np.array(ptr, np.int32), out)
        assert np.array_equal(out, np.diff(np.array(ptr, np.int32))), ptr


# A loop-independent (distance-0) conflict between different subscripts is
# ordered within the iteration; provably disjoint subscripts are not, so the
# path does not degenerate into blanket same-array serialization.
def test_intra_iteration_dependence():
    @kernel
    def alias(A: f32[64], C: f32[32]):
        for i in range(32):
            A[2 * i] = 1.0  # write A[2i]
            C[i] = A[i]  # read A[i] -- aliases the write only at i == 0

    # The accesses land on the same element only at i == 0. The pair also carries
    # a loop-carried edge (i >= 1); keeping the tightest (dist 0) distance is what
    # preserves the same-iteration ordering.
    loop = _sched(alias).cyclic()[0]
    assert loop.ii == 1  # a dist-0 edge orders within the iteration, no recurrence
    assert loop.op("store").t < loop.op("load").t

    @kernel
    def disjoint(A: f32[64], C: f32[32]):
        for i in range(32):
            A[2 * i] = 1.0  # write even indices
            C[i] = A[2 * i + 1]  # read odd indices -- never the same element

    loop = _sched(disjoint).cyclic()[0]
    assert loop.op("store").t == 0
    assert loop.op("load").t == 0


# allo.assume feeds the scheduler facts the polyhedral test cannot prove:
# a bound on a dynamic trip, and the absence of an inter-iteration dependence.
# allo.grid carries the same independence guarantee implicitly.
def test_assume_hints():
    @kernel
    def k(A: i32[128], out: i32[1], n: index):
        allo.assume(n < 100)
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    # A bounded dynamic trip reports a worst-case latency flagged as a bound
    # rather than deferring it.
    loop = _sched(k).cyclic()[0]
    assert loop.latency > 0
    assert loop.latency_is_bound is True

    def hist(hint):
        @kernel
        def h(idx: i32[128], acc: i32[64]):
            for i in range(128):
                if hint:
                    allo.assume(acc, i, type="inter")
                acc[idx[i]] = acc[idx[i]] + 1

        return h

    # Without the hint the aliasing histogram update keeps a conservative
    # loop-carried edge; asserting no inter-iteration dependence prunes it.
    assert _sched(hist(False)).cyclic()[0].ii == 2
    assert _sched(hist(True)).cyclic()[0].ii == 1

    # A grid()'s independence guarantee lowers to `assume.nodep` on the written
    # array, dropping the conservative back edge on a non-affine aliasing write --
    # whereas the identical body in a sequential range() nest keeps it.
    N = 64

    @kernel
    def par_scatter(val: f32[N, N], out: f32[N]):
        for i, j in allo.grid(N, N):
            out[i * j] = out[i * j] + val[i, j]

    @kernel
    def seq_scatter(val: f32[N, N], out: f32[N]):
        for i in range(N):
            for j in range(N):
                out[i * j] = out[i * j] + val[i, j]

    assert _iis(_sched(par_scatter).cyclic()) == [1]
    assert _iis(_sched(seq_scatter).cyclic()) == [MEM_REDUCE_II]


# --- the shared iteration-control controller skeleton ------------------------


def _one_region(m):
    """The single done-driven region of `m` (the one that emits an `r<N>_fire`)."""
    ids = m.regions_with("fire")
    assert len(ids) == 1, f"expected one done-driven region, got {ids}"
    return ids[0]


def _hold_done(m, region):
    """The set-pulse of region `region`'s done latch.

    `holdDone` is `done = compreg(mux(start, false, mux(set, true, done)))`:
    cleared by the region start so a retriggered region re-edges, set by the
    completion pulse. Returns `set`, having checked the shape.
    """
    reg, inp = m.reg_named(f"r{region}_done")
    clear = m.mux(inp)
    assert clear and clear[1].startswith("false"), f"r{region}_done not cleared: {inp}"
    hold = m.mux(clear[2])
    assert (
        hold and hold[1].startswith("true") and hold[2] == reg
    ), f"r{region}_done is not a hold latch: {clear[2]}"
    return hold[0]


def _launch_cone(m, fire):
    """`fire`'s cone, stepping through one register level.

    Both counted cells register the ADVANCE path (the counter only settles the
    cycle after `advance`), so the advance pulse is always one register back,
    whether the whole launch is registered (a container) or only that arm is (a
    call node).
    """
    seen = set(m.cone(fire))
    for v in list(seen):
        if m.defs.get(v, "").startswith("seq.compreg"):
            seen |= m.cone(m.operands(v)[0])
    return seen


def _counted_skeleton(m, region, counter):
    """Check region `region`'s counted iteration controller.

    Returns `(complete, launch, counter_read)`. The skeleton is read back off
    the counter's next-value, which is the one place all four cells meet:
    `iv' = mux(start, lb, mux(advance, iv+step, iv))`.
    """
    _, inp = m.reg_named(counter)
    seed = m.mux(inp)
    assert seed, f"{counter}' is not seeded at start: {inp}"
    adv = m.mux(seed[2])
    assert adv, f"{counter}' does not advance conditionally: {seed[2]}"
    advance, stepped, read = adv
    assert m.defs[stepped].startswith("comb.add"), f"{counter} does not step: {stepped}"
    # `advance = complete & ~isLast`: the completion pulse is the operand of the
    # AND that is not the inverted bound test.
    conj = m.defs[advance]
    assert conj.startswith("comb.and"), f"advance is not a conjunction: {conj}"
    a, b = m.operands(advance)[:2]
    notlast = [v for v in (a, b) if m.defs.get(v, "").startswith("comb.xor")]
    assert len(notlast) == 1, f"advance has no ~isLast term: {conj}"
    complete = a if notlast[0] == b else b

    # Launch and finish are the two arms of the same boundary event.
    launch = m.signal(f"r{region}_fire")
    assert advance in _launch_cone(m, launch), "the fire pulse ignores advance"
    assert complete in m.cone(
        _hold_done(m, region)
    ), "the done latch does not key off the body's completion"
    return complete, launch, read


# A container (body = child regions) and a loop-over-call (body = one
# instance) emit the same counted skeleton, and differ only in launch policy.
# Sharing one skeleton is what makes the remaining difference legible as a
# deliberate decision rather than as drift between two controllers.
def test_container_and_loop_call_share_one_controller():
    @kernel
    def ct(A: i32[8, 4], B: i32[8]):
        for i in range(8):  # container: an inner reduction is its body
            acc: i32 = 0
            for j in range(4):
                acc += A[i, j]
            B[i] = acc

    @kernel
    def lc_body(A: i32[16], out: i32[16], i: index):
        out[i] = A[i] * 3

    @kernel
    def lc(A: i32[16], out: i32[16]):
        for i in range(16):  # call node: one child instance, fired 16 times
            lc_body(A, out, i)

    cm = Mod(_to_rtl(ct).mlir, "ct")
    lm = Mod(_to_rtl(lc).mlir, "lc")
    assert "hw.instance" in lm.text, "lc must instantiate its child"

    _, c_fire, c_read = _counted_skeleton(cm, _one_region(cm), "i")
    _, l_fire, l_read = _counted_skeleton(lm, _one_region(lm), "i")

    # The one difference: a container's fire pulse IS a register (its counter has
    # settled by the time a child samples it); a call node's is combinational off
    # `start`, over a bypassed counter, so start->done stays the scheduled span.
    assert cm.defs[c_fire].startswith("seq.compreg"), "a container launches registered"
    assert lm.defs[l_fire].startswith("comb.or"), "a call node launches at start"
    assert "start" in lm.cone(l_fire), "a call node's first launch is `start` itself"
    assert lm.mux(l_read), f"a call node bypasses its counter at start: {l_read}"
    assert not cm.mux(c_read), f"a container reads the counter register: {c_read}"

    A = (np.arange(32, dtype=np.int32) * 7 - 5).reshape(8, 4)
    B = np.zeros(8, np.int32)
    _to_rtl(ct).cosim(A, B)
    assert np.array_equal(B, A.sum(axis=1))

    A16i = np.arange(16, dtype=np.int32) * 3 + 1
    out = np.zeros(16, np.int32)
    _to_rtl(lc).cosim(A16i, out)
    assert np.array_equal(out, A16i * 3)


# A container whose trip is a runtime scalar is the same controller with
# resolved bounds, including the zero-trip case. Every bound is an ordinary
# datapath Source (a literal cell or a runtime value), so a dynamic trip
# changes what isLast compares against and nothing else, in particular not
# the empty path, which is the only reason the controller registers a
# completion pulse on the start cycle at all.
def test_runtime_bounds_ride_the_same_skeleton():
    @kernel
    def dyn(A: i32[16], B: i32[16], n: index):
        for i in range(4):
            for j in range(n):
                B[j] = B[j] + A[i]

    rtl = _to_rtl(dyn)
    m = Mod(rtl.mlir, "dyn")
    # The outer container is the counted skeleton; its child's bound is runtime.
    _counted_skeleton(m, _one_region(m), "i")

    A = (np.arange(16, dtype=np.int32) % 5) + 1
    for n in (0, 1, 16):  # n == 0 is the empty path
        B = np.zeros(16, np.int32)
        rtl.cosim(A, B, np.int32(n))
        gold = np.zeros(16, np.int32)
        gold[:n] = A[:4].sum()
        assert np.array_equal(B, gold), f"n={n}: {list(B)}"


# --- loop-over-calls bound generality -----------------------------------------


# for i: for j: child(i, j). The nest is deliberately NOT coalesced:
# flattening it would delinearize the two induction variables into address
# arithmetic sitting beside the call, forcing a decomposition where the
# uncoalesced form keeps a lone-call inner loop the fast controller drives
# directly. It also pins the controller's re-entry: the inner loop's counter
# register only carries lb from the cycle after start, while its first child
# fires ON start, so without the start-cycle bypass every outer iteration but
# the first would run its first child at the previous pass's final index.
def test_nested_loop_over_calls():
    @kernel
    def nl_child(A: i32[4, 4], B: i32[4, 4], i: index, j: index):
        B[i, j] = A[i, j] * 2

    @kernel
    def nl_top(A: i32[4, 4], B: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                nl_child(A, B, i, j)

    mod = _to_rtl(nl_top)
    # The nest survives as two levels: an outer container over an inner
    # loop-over-calls, not one coalesced 16-iteration loop.
    assert mod.dcp.count("allo.dcp.pipeline") == 2
    B = np.zeros((4, 4), np.int32)
    mod.cosim(A44, B)
    assert np.array_equal(B, A44 * 2)


# The outer level holds the inner loop-over-calls AND its own loose store,
# so the outer container sequences three children per iteration.
def test_nested_loop_over_calls_with_an_epilogue():
    @kernel
    def ne_child(A: i32[4, 4], B: i32[4, 4], i: index, j: index):
        B[i, j] = A[i, j] * 2

    @kernel
    def ne_top(A: i32[4, 4], B: i32[4, 4], C: i32[4]):
        for i in range(4):
            for j in range(4):
                ne_child(A, B, i, j)
            C[i] = A[i, 0] + 7

    B = np.zeros((4, 4), np.int32)
    C = np.zeros(4, np.int32)
    _to_rtl(ne_top).cosim(A44, B, C)
    assert np.array_equal(B, A44 * 2)
    assert np.array_equal(C, A44[:, 0] + 7)


# A recurrence in a body that also holds a call. The accumulator crosses
# iterations as a container iter-arg, the mechanism an imperfect nest uses,
# rather than as a shift-register tap whose depth assumes a per-cycle issue
# cadence the done-driven loop does not have.
def test_loop_carried_accumulate_beside_a_call():
    @kernel
    def ac_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def ac_top(A: i32[16], B: i32[16], S: i32[1]):
        acc: i32 = 0
        for i in range(16):
            ac_child(A, B, i)
            acc += A[i]
        S[0] = acc

    B = np.zeros(16, np.int32)
    S = np.zeros(1, np.int32)
    _to_rtl(ac_top).cosim(A16, B, S)
    assert np.array_equal(B, A16 * 2)
    assert S[0] == A16.sum()


# A sub-kernel call in a loop body is ONE child instance re-fired per
# iteration, not a pipelined operator: the next invocation cannot start until
# the previous one's done, plus the cycle the controller takes to re-arm.
# Charging it as a zero-occupancy latency node would report II=1 and a
# latency 20x short of the truth. Measured against cosim, so the occupancy
# model cannot drift from the controller it describes.
def test_loop_over_calls_reports_the_interval_it_runs_at():
    @kernel
    def li_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def li_top(A: i32[16], B: i32[16]):
        for i in range(16):
            li_child(A, B, i)

    child = _latency(li_child)
    regions = _to_rtl(li_top).schedule().func("li_top").regions
    assert [r.ii for r in regions if r.ii is not None] == [child + 1]
    B = np.zeros(16, np.int32)
    r = _to_rtl(li_top).cosim(A16, B)
    assert np.array_equal(B, A16 * 2 + 1)
    assert _latency(li_top) == r.cycles


# A straight-line span holding only a call is not one cycle deep: the region
# is not done until the child is. The parent below places its load at the
# callee's declared latency, so an undercount reads the buffer before the
# child has filled it (it reads the LAST-written element, so a stale read
# cannot pass by luck). The callee here is itself a loop over a call: 16
# invocations of a 3-cycle child, so a composition that reported the single
# child's 3 cycles would be an order of magnitude short.
def test_a_calls_latency_covers_the_call_itself():
    @kernel
    def cv_leaf(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def cv_mid(A: i32[16], B: i32[16]):
        for i in range(16):
            cv_leaf(A, B, i)

    @kernel
    def cv_top(A: i32[16], C: i32[16]):
        B: i32[16]
        cv_mid(A, B)
        C[0] = B[15] + 1

    assert _latency(cv_mid) > 16  # not the single child's 3 cycles
    C = np.zeros(16, np.int32)
    r = _to_rtl(cv_top).cosim(A16, C)
    assert C[0] == A16[15] * 2 + 1
    assert _latency(cv_top) == r.cycles


# The call walk sees calls only; the region sum sees regions only. A kernel
# with one of each needs both, since reporting just the call's completion
# stops the clock while the loop is still running.
def test_a_call_beside_a_plain_loop_counts_both():
    @kernel
    def bp_child(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] * 2

    @kernel
    def bp_top(A: i32[16], C: i32[16]):
        B: i32[16]
        bp_child(A, B)
        for i in range(16):
            C[i] = B[i] + 1

    C = np.zeros(16, np.int32)
    r = _to_rtl(bp_top).cosim(A16, C)
    assert np.array_equal(C, A16 * 2 + 1)
    # An upper bound is the contract (a caller may wait longer than needed, never
    # less), and the loop's own span has to be in it; the call alone is not.
    assert r.cycles <= _latency(bp_top) <= r.cycles + 2
    assert _latency(bp_top) > _latency(bp_child)


# A loop-over-calls body is a dcp.pipeline wrapping the call, so its call
# reifies to a dcp.instance and the container lowers to the leaf. One child
# instance is fired N times by a counter driving its index, each invocation
# advancing on the child's real done (throughput = one iteration per child
# latency, not the pipeline cadence).
def test_leaf_loop_over_calls_controller_paces_on_child_done():
    @kernel
    def lc_step(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def lc_top(A: i32[16], B: i32[16]):
        for i in range(16):
            lc_step(A, B, i)  # invoke the sub-kernel 16 times

    rtl = _to_rtl(lc_top)
    assert "allo.dcp.instance" in rtl.dcp  # leaf CallUnit path (structural lock)
    m = rtl.mlir
    assert "hw.instance" in m  # the single child instance
    assert "%i = seq.compreg" in m  # the loop counter, named after the source IV
    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.zeros(16, np.int32)
    r = rtl.cosim(A, B)
    assert r.cycles > 0
    assert np.array_equal(B, A * 2 + 1)


# The loop-over-calls controller takes its induction bounds from
# terminatorOf, the same source every other counted region uses, rather than
# a hardcoded 0 to N step 1: the counter seeds at lb, advances by step,
# terminates on iv + step >= ub, and an empty loop completes on the shared
# isEmpty escape. Each case below is one bound a hardcoded controller could
# not express. The child's index port reads the counter directly, so a wrong
# bound writes the wrong elements rather than merely mistiming.
def test_loop_over_calls_general_bounds():
    def child():
        @kernel
        def lc_step(A: i32[16], B: i32[16], i: index):
            B[i] = A[i] * 2 + 1

        return lc_step

    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F

    # A non-zero lower bound: the untouched prefix pins that the counter seeds
    # at `lb` instead of 0.
    lc_step = child()

    @kernel
    def lb_top(A: i32[16], B: i32[16]):
        for i in range(4, 16):
            lc_step(A, B, i)

    B = np.zeros(16, np.int32)
    _to_rtl(lb_top).cosim(A, B)
    ref = np.zeros(16, np.int32)
    ref[4:] = A[4:] * 2 + 1
    assert np.array_equal(B, ref)

    # A stride: only the even elements are written, so a counter that still
    # advanced by 1 would fill every slot.
    lc_step = child()

    @kernel
    def step_top(A: i32[16], B: i32[16]):
        for i in range(0, 16, 2):
            lc_step(A, B, i)

    B = np.zeros(16, np.int32)
    _to_rtl(step_top).cosim(A, B)
    ref = np.zeros(16, np.int32)
    ref[0::2] = A[0::2] * 2 + 1
    assert np.array_equal(B, ref)

    # A runtime trip: no `tripCount` at all, so the bound is a resolved Source.
    # The zero-trip drive additionally exercises the `isEmpty` escape: the
    # child never fires, so there is no `done` edge to complete on.
    lc_step = child()

    @kernel
    def dyn_top(A: i32[16], B: i32[16], n: index):
        for i in range(n):
            lc_step(A, B, i)

    rtl = _to_rtl(dyn_top)
    B = np.zeros(16, np.int32)
    rtl.cosim(A, B, 5)
    ref = np.zeros(16, np.int32)
    ref[:5] = A[:5] * 2 + 1
    assert np.array_equal(B, ref)

    B = np.zeros(16, np.int32)
    rtl.cosim(A, B, 0)
    assert np.array_equal(B, np.zeros(16, np.int32))


# --- pipeline directives ------------------------------------------------------


def _vadd_kernel():
    @kernel
    def v(A: i32[32], B: i32[32], C: i32[32]):
        for i in range(32, name="i"):
            C[i] = A[i] + B[i]

    return v


def _mac_kernel():
    @kernel
    def m(A: i32[8], B: i32[8], out: i32[8]):
        for i in range(8, name="i"):
            out[i] = A[i] * B[i]

    return m


def test_explicit_target_ii_floor_is_honored():
    assert _sched(_vadd_kernel()).cyclic()[0].ii == 1  # natural minimum

    s = _vadd_kernel().schedule()
    s.pipeline("i", ii=3)
    mod = s.export("rtl")
    assert mod.schedule().cyclic()[0].ii == 3  # target honored as a floor


def test_pipeline_disabled_runs_sequentially():
    pl = _sched(_mac_kernel()).cyclic()[0]

    s = _mac_kernel().schedule()
    s.pipeline("i", ii=-1)
    mod = s.export("rtl")
    npl = mod.schedule().cyclic()[0]

    assert npl.ii == npl.length  # no overlap: II = body length
    assert npl.latency == 8 * npl.length  # trip * depth
    assert pl.ii < npl.ii  # pipelining packs iterations tighter
    assert pl.latency < npl.latency


def test_pipeline_directive_preserves_result_cosim():
    # A pipeline directive changes the schedule (II), not the result: a forced
    # II=3 pipeline and a pipeline-off (sequential) loop both still compute the
    # elementwise op. Pins that the directive knob is correctness-neutral.
    A = np.arange(32, dtype=np.int32)
    B = np.arange(32, dtype=np.int32) * 3
    s = _vadd_kernel().schedule()
    s.pipeline("i", ii=3)  # forced II above the natural minimum
    C = np.zeros(32, np.int32)
    s.export("rtl").cosim(A, B, C)
    assert np.array_equal(C, A + B)

    A8 = np.arange(8, dtype=np.int32) + 1
    B8 = np.arange(8, dtype=np.int32) + 2
    s = _mac_kernel().schedule()
    s.pipeline("i", ii=-1)  # pipelining disabled -> sequential
    out = np.zeros(8, np.int32)
    s.export("rtl").cosim(A8, B8, out)
    assert np.array_equal(out, A8 * B8)


# --- pipelined-level scheduling under mixed loop ancestors --------------------


def test_pipelined_level_recurrence_under_mixed_loop_ancestor():
    # An imperfect pipelined nest schedules via the fused-overlap (level) path
    # (the outer loop pipelined over its inner loop as a fixed-latency node) when
    # `unroll_under_pipeline` is off. Here the pipelined level `j` sits under a
    # *runtime-bounded* outer loop `i` -- a memory-loaded trip makes `i` an
    # `scf.for`, so the nest MIXES scf.for and affine.for. The level carries a
    # memory recurrence (`acc[0]` divided every iteration). The affine dependence
    # components index only the affine loop `j` (an scf iv is no affine dim), so
    # attributing the recurrence to `j` by positional nesting depth would
    # over-count past the scf ancestor and silently drop the edge -- an unsound
    # II = the 4-cycle inner-loop occupancy. Matching the level to its component
    # by identity keeps the edge, so the II is the recurrence bound read -> div
    # -> write.
    RECUR_II = MEM + FDIV + MEM

    def mixed():
        @kernel
        def mix(A: f32[64, 64], acc: f32[1], B: f32[64, 2], nbuf: index[1]):
            n: index = nbuf[0]  # memory-loaded bound => scf.for ancestor
            allo.assume(n < 64)
            for i in range(n):
                for j in range(64, name="j"):  # pipelined affine level
                    acc[0] = acc[0] / A[i, j]  # level-carried recurrence
                    for p in range(2):  # inner loop kept rolled (level node)
                        B[j, p] = A[i, j]

        return mix

    s = mixed().schedule()
    s.pipeline("j")
    mod = s.export("rtl", unroll_under_pipeline=False)
    # Two cyclic regions: the pipelined level `j` and its inner loop `p` (II=1).
    # The level's II is the recurrence bound; without the identity-matched
    # projection it would collapse to the 4-cycle inner-loop occupancy.
    iis = _iis(mod.schedule().cyclic())
    assert max(iis) == RECUR_II


def test_outer_level_scheduling_is_timing_aware():
    # The outer-level scheduling problem is timing-aware: a combinational chain
    # among the outer level's loose ops is cut at the cycle boundary, exactly as
    # in the leaf body (an inner-loop node is a registered boundary, so it never
    # joins the chain). Here the level carries a 3-deep int-add chain (3 x 1.2ns
    # = 3.6ns); extra stores of the intermediates keep it from being
    # reassociated into a tree. At a tight clock the chain spans two cycles; at a
    # slack clock, one.
    def level_add_cycles(freq_mhz):
        @kernel
        def chain(
            A: i32[8, 8],
            b: i32[8],
            c: i32[8],
            d: i32[8],
            o0: i32[8],
            o1: i32[8],
            o2: i32[8],
        ):
            for i in range(8, name="i"):  # pipelined imperfect nest -> level path
                s0: i32 = b[i] + c[i]
                s1: i32 = s0 + d[i]
                s2: i32 = s1 + b[i]  # 3-deep combinational chain at the level
                o0[i] = s0  # extra uses -> not a reassociable single-use chain
                o1[i] = s1
                for p in range(8, name="p"):  # inner loop node (registered)
                    A[i, p] = s2
                o2[i] = s2

        s = chain.schedule()
        s.pipeline("i")
        mod = s.export("rtl", unroll_under_pipeline=False, freq_mhz=freq_mhz)
        res = mod.schedule()
        # The level region is the pipelined outer loop holding the 3-add chain.
        level = next(
            r for r in res.cyclic() if sum(o.kind == "addi" for o in r.ops) == 3
        )
        return len({o.t for o in level.ops if o.kind == "addi"})

    assert level_add_cycles(1000 / 3.0) == 2  # 3.6ns chain cut by a 3.0ns clock
    assert level_add_cycles(1000 / 6.0) == 1  # fits in one 6.0ns cycle -> not cut
