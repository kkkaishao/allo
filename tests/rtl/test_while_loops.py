# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`while`-loop scheduling and correctness: flushing pipelines, CHECK/RUN sequential control, nested whiles, and the various continue-condition shapes."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, f32, index

sys.path.insert(0, os.path.dirname(__file__))
from _common import Mod, _sched, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


# --- schedule shape -----------------------------------------------------------


def test_while_scheduling():
    # A counted while is raised to a for and schedules identically to one; a
    # data-dependent while stays conditional, scheduled as a flushing pipeline
    # with its trip -- and therefore latency -- left unknown.
    @kernel
    def wc(A: i32[128], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while i < 128:
            s = s + A[i]
            i = i + 1
        out[0] = s

    @kernel
    def fc(A: i32[128], out: i32[1]):
        s: i32 = 0
        for i in range(128):
            s = s + A[i]
        out[0] = s

    w = _sched(wc).cyclic()[0]
    f = _sched(fc).cyclic()[0]
    # Raised to a constant-bound for, so the schedule matches `range(128)`
    # exactly -- same II, length, and (non-bound) latency -- and is not
    # conditional (no flushing controller).
    assert (w.ii, w.length, w.latency) == (f.ii, f.length, f.latency)
    assert not w.conditional and not w.latency_is_bound

    @kernel
    def wr(n0: i32, out: i32[1]):
        x: i32 = n0
        c: i32 = 0
        while x > 1:
            x = x - 1
            c = c + 1
        out[0] = c

    mod = _to_rtl(wr)
    loop = mod.schedule().cyclic()[0]
    assert loop.conditional is True
    assert loop.latency is None
    assert "dcp.condition" in mod.dcp  # reified while terminator


def test_while_with_nested_while():
    # Two decreasing (hence un-raised) whiles nested. The inner while's
    # straight-line body schedules as a flushing pipeline; the outer while's
    # body is decomposed around it and the outer runs sequentially. Exercises
    # the nested-loop-in-while decomposition recursing through a while child.
    N = 64

    @kernel
    def nested_while(A: i32[N]) -> i32:
        total: i32 = 0
        s: i32 = N
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        return total

    mod = _to_rtl(nested_while)
    res = mod.schedule()
    assert len(res.cyclic()) >= 1  # the inner while pipelines
    assert res.func("nested_while").latency is None  # data-dependent trips
    # Both whiles close: the inner -> flushing pipeline, the outer -> sequential
    # while dcp.pipeline wrapping it. No raw scf.while; two dcp.condition ends.
    assert "scf.while" not in mod.dcp
    assert mod.dcp.count("dcp.condition") == 2


# --- flushing-pipeline correctness ---------------------------------------------


def test_while_flushing_pipeline_cosim():
    # The flushing pipeline emitted end-to-end: `running` gated by the exit
    # condition, each loop-carried iter-arg frozen into a survivor register at
    # exit, and the sibling store reading the frozen count. `x > 1` runs x-1
    # steps, so c = max(0, n0-1) -- including the zero-iteration case (n0<=1).
    @kernel
    def wr(n0: i32, out: i32[1]):
        x: i32 = n0
        c: i32 = 0
        while x > 1:
            x = x - 1
            c = c + 1
        out[0] = c

    mod = _to_rtl(wr)
    for n0 in (1, 2, 3, 7, 20):
        out = np.zeros(1, np.int32)
        r = mod.cosim(np.int32(n0), out)
        assert out[0] == max(0, n0 - 1)
        assert r.cycles > 0


def test_while_two_carried_accumulate_cosim():
    # A while carrying TWO recurrences whose result depends on both: acc folds
    # x while x counts down, so the frozen `acc` survivor is the triangular sum.
    @kernel
    def wacc(n0: i32, out: i32[1]):
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + x
            x = x - 1
        out[0] = acc

    mod = _to_rtl(wacc)
    for n0 in (0, 1, 5, 9):
        out = np.zeros(1, np.int32)
        mod.cosim(np.int32(n0), out)
        assert out[0] == n0 * (n0 + 1) // 2


def test_while_multistage_flush_cosim():
    # A store-less while whose *body* spans two stages (the `A[x-1]` load
    # pushes `next_acc` to stage 1) but whose condition `x > 0` is
    # combinational. The flushing pipeline drains the deeper survivor: `acc`
    # advances one cycle after each issue, and the exit is delayed to match, so
    # the frozen `acc` is the correct sum. Distinct from a memory-*dependent*
    # condition (still deferred).
    N = 64

    @kernel
    def wsum(n0: i32, A: i32[N], out: i32[1]):
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + A[x - 1]
            x = x - 1
        out[0] = acc

    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    mod = _to_rtl(wsum)
    for n0 in (0, 1, 4, 10, 25):
        out = np.zeros(1, np.int32)
        mod.cosim(np.int32(n0), A, out)
        assert out[0] == int(A[:n0].sum())


def test_while_in_loop_store_cosim():
    # A leaf flushing-while that writes memory in its body. The doomed exit
    # iteration is issued but must commit nothing: emitAccesses gates each
    # store's write-enable by the continue-condition (`issue & cond`), the same
    # rule the loop-carried survivors follow. Covers a single-stage store, a
    # multi-stage store fed by an in-loop carried scalar (deeper drain), and the
    # zero-trip case (no write). Unwritten output elements read back as the
    # memory init (0).
    N = 32

    @kernel
    def wstore(A: i32[N], B: i32[N], n0: i32):  # write-once per iteration
        x: i32 = n0
        while x > 0:
            B[x - 1] = A[x - 1] * 2
            x = x - 1

    @kernel
    def wscan(A: i32[N], B: i32[N], n0: i32):  # store the running prefix sum
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + A[x - 1]
            B[x - 1] = acc
            x = x - 1

    ma, mb = _to_rtl(wstore), _to_rtl(wscan)
    assert ma.schedule().cyclic()[0].conditional and "dcp.condition" in ma.dcp
    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    for n0 in (0, 1, 7, N):
        B = np.zeros(N, np.int32)
        ma.cosim(A, B, np.int32(n0))
        gold = np.zeros(N, np.int32)
        gold[:n0] = A[:n0] * 2
        assert np.array_equal(B, gold)

        B = np.zeros(N, np.int32)
        mb.cosim(A, B, np.int32(n0))
        gold = np.zeros(N, np.int32)
        gold[:n0] = np.cumsum(A[:n0][::-1])[::-1]  # acc counts x down from n0
        assert np.array_equal(B, gold)


# --- condition shapes: memory, IP, nested --------------------------------------


def test_while_mem_condition_cosim():
    # A while loop whose continue-condition reads memory (`A[i] != key`): the
    # loop index advances until the searched element is found, and the
    # loop-carried value is read after the loop. Covers a single-value carry, a
    # two-value carry (the index and a step counter), and a zero-iteration exit
    # (the condition false on entry).
    A = np.arange(16, dtype=np.int32)  # A[i] == i, so the found index equals key

    @kernel
    def linsearch(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        while A[i] != key:
            i = i + 1
        out[0] = i

    out = np.zeros(1, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(11), out)
    assert out[0] == 11

    @kernel
    def search_steps(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        c: i32 = 0
        while A[i] != key:
            i = i + 1
            c = c + 1
        out[0] = c

    out = np.zeros(1, np.int32)
    _to_rtl(search_steps).cosim(A, np.int32(9), out)
    assert out[0] == 9

    # A[0] == key: the condition is false on entry, so the body never runs and
    # the carried index holds its initial value.
    out = np.full(1, 999, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(0), out)
    assert out[0] == 0


def test_while_mem_condition_shared_array_cosim():
    # A while loop that reads the same array in BOTH its continue-condition
    # (`A[i] > 0`) and its body (`s += A[i]`). Each access is a distinct memory
    # read, so the condition and the body do not contend for a port.
    @kernel
    def wmem(A: i32[16], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while A[i] > 0:
            s = s + A[i]
            i = i + 1
        out[0] = s

    A = np.array([5, 3, 8, 2, 0] + [9] * 11, dtype=np.int32)  # sentinel 0 at idx 4
    out = np.zeros(1, np.int32)
    _to_rtl(wmem).cosim(A, out)
    assert out[0] == 5 + 3 + 8 + 2  # sum until A[4] == 0 stops the loop


def test_while_ip_condition_cosim():
    # A while whose continue-condition is a multi-cycle floating-point
    # operation rather than a memory read. The loop iterates until the float
    # condition settles false; the body advances a float-carried value. Covers
    # a single float comparison (`r > tol`) and a float subtraction feeding a
    # comparison (`x - b > 0`), the latter a multi-stage condition cone. The
    # condition is not settled in the issue cycle, so the loop runs
    # sequentially (a conditional region) rather than as a flushing pipeline.
    @kernel
    def fconverge(x: f32, tol: f32, out: f32[1]):
        r: f32 = x
        while r > tol:
            r = r * 0.5
        out[0] = r

    mod = _to_rtl(fconverge)
    assert mod.schedule().cyclic()[0].conditional
    assert "hw.module.extern @fcmp" in mod.mlir

    def gold_halve(x, tol):
        r = np.float32(x)
        while r > np.float32(tol):
            r = np.float32(r * np.float32(0.5))
        return r

    for x, tol in [(100.0, 1.0), (7.0, 1.0), (0.5, 1.0)]:  # last exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(x), np.float32(tol), out)
        assert out[0] == gold_halve(x, tol)

    @kernel
    def fcountdown(a: f32, b: f32, out: f32[1]):
        x: f32 = a
        while x - b > 0.0:
            x = x - 1.0
        out[0] = x

    mod = _to_rtl(fcountdown)
    assert mod.schedule().cyclic()[0].conditional

    def gold_count(a, b):
        x = np.float32(a)
        while np.float32(x - np.float32(b)) > np.float32(0.0):
            x = np.float32(x - np.float32(1.0))
        return x

    for a, b in [(10.0, 2.5), (5.0, 5.0), (3.0, 0.0)]:  # middle exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(a), np.float32(b), out)
        assert out[0] == gold_count(a, b)


def test_nested_while_cosim():
    # A sequential-wrapper while (outer `s`) around a flushing-pipeline while
    # (inner `t`), carrying a cross-region accumulator `total`. The outer while
    # is a conditional container: its iter-args are survivor registers advanced
    # by the children's results, the raw `s > 0` condition is evaluated over
    # those registers, and the children re-run each outer iteration. total ends
    # as sum_{s=1..N} sum_{t=1..s} A[t-1].
    N = 8

    @kernel
    def nested(A: i32[N], out: i32[1]):
        total: i32 = 0
        s: i32 = N
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        out[0] = total

    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    expected = sum(int(A[:s].sum()) for s in range(1, N + 1))
    out = np.zeros(1, np.int32)
    _to_rtl(nested).cosim(A, out)
    assert out[0] == expected


# --- call-in-while control drop ------------------------------------------------


def test_call_in_a_while_body():
    # A while whose body calls a sub-kernel cannot flushing-pipeline at all:
    # that schedule issues an iteration per cycle, which a child instance fired
    # and awaited per iteration can never follow. It drops to the sequential
    # CHECK/RUN controller, the same route a nested loop or a non-combinational
    # condition takes.
    @kernel
    def wc_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def wc_top(A: i32[16], B: i32[16]):
        i: i32 = 0
        while i < 16:
            wc_child(A, B, i)
            i += 1

    A16 = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.zeros(16, np.int32)
    _to_rtl(wc_top).cosim(A16, B)
    assert np.array_equal(B, A16 * 2)


# --- checked-iteration skeleton reuse -------------------------------------------


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


def test_checked_while_reuses_the_counted_skeleton():
    # A CHECK/RUN while (a conditional container wrapping a flushing-pipeline
    # inner while) keeps the same fire / done-latch pair a counted cell uses,
    # replacing only the counter-driven test with a delayed condition pulse:
    # no counter, no separate empty term, since the first CHECK already answers
    # it. Cosims a nested double-while summation.
    @kernel
    def nested(A: i32[8], out: i32[1]):
        total: i32 = 0
        s: i32 = 8
        while s > 0:  # conditional container
            t: i32 = s
            while t > 0:  # flushing-pipeline leaf
                total += A[t - 1]
                t -= 1
            s -= 1
        out[0] = total

    rtl = _to_rtl(nested)
    m = Mod(rtl.mlir, "nested")
    r = _one_region(m)

    check = m.signal(f"r{r}_check")
    fire = m.signal(f"r{r}_fire")
    finish = _hold_done(m, r)
    # Launch and finish are the two arms of ONE pulse: both are `check & (~)cond`
    # over the same settled CHECK, so the container cannot do both.
    assert check in m.cone(fire) and check in m.cone(finish)
    cond = [v for v in m.cone(fire) if m.defs.get(v, "").startswith("comb.icmp")]
    assert cond, "the fire pulse does not test the continue condition"
    assert any(c in m.cone(finish) for c in cond), "finish tests another condition"
    # No counter: termination is by condition alone, so no induction arithmetic
    # reaches the launch decision.
    assert not [v for v in m.cone(fire) if m.defs.get(v, "").startswith("comb.add")]

    A = (np.arange(8, dtype=np.int32) * 3 + 1) & 0xFF
    out = np.zeros(1, np.int32)
    rtl.cosim(A, out)
    assert out[0] == sum(int(A[:s].sum()) for s in range(1, 9))
