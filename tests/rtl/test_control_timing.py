# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline elasticity (the region-wide stall shell) and clock-frequency-aware chaining/timing."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, Stream
from allo.backend.rtl.device import builtin_device, MemoryKind

sys.path.insert(0, os.path.dirname(__file__))
from _common import Mod, _sched, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

_STALLS = [0.0, 0.5, 0.8]

# A cell carrying the region's TIME BASE: a valid-chain stage (`r1_v3`) or a
# register tap (`acc_d2`). Survivors (`r1_sv0`) are excluded, since a survivor
# is enabled by its own capture pulse, not by the shell.
_TIME_BASE = re.compile(r"^(r\d+_v\d+|.+_d\d+)$")


class _Mod(Mod):
    # Mod plus the time-base classification this file's locks read.

    def time_base(self):
        # (label, register, input) of every time-base cell.
        return [(lb, r, i) for lb, r, i in self.regs if _TIME_BASE.match(lb)]


# --- elasticity: one shell per region ----------------------------------------


# An elastic region's chain stages all ride ONE `chainEnable`. Each of these
# cells is built by a different helper (register chain, valid-delay, put/get
# pulses), and they agree only because each names the same region's shell.
def test_one_shell_enables_every_time_base_cell():
    @kernel
    def stage(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(16):
            y_out.put(x_in.get() + 7)

    rtl = _to_rtl(stage)
    m = _Mod(rtl.mlir, "stage")

    ce = m.hinted("r0_ce")
    assert m.hints_like(r"_ce$") == ["r0_ce"], "one region, one shell"

    cells = m.time_base()
    assert cells, "an elastic region must have time-base cells to freeze"
    enables = {m.enable_of(reg, inp) for _, reg, inp in cells}
    assert enables == {ce}, f"time-base cells not on one shell: {enables}"

    # G's half: issue is the run flag gated by the shell.
    issue = m.hinted("r0_issue")
    assert ce in m.cone(issue)

    # The done drain is held through back-pressure by the same signal, so the
    # region cannot report completion on a token that was never accepted.
    done_reg, done_in = m.reg_named("r0_done")
    assert done_reg  # the latch itself
    assert ce in m.cone(done_in)

    x = np.arange(16, dtype=np.int32) * 5 - 3
    for gap in _STALLS:
        y = np.zeros(16, dtype=np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, x + 7), f"gap={gap}: {list(y)}"


# A clock-enabled IP's `ce` port IS the region's `chainEnable`. The shell is
# consumed at the IP boundary too: a free-running IP would keep clocking
# while the shift chains are frozen and fold a stale result.
def test_ce_ip_rides_the_region_shell():
    @kernel
    def fstage(x_in: Stream[f32], y_out: Stream[f32]):
        for i in range(16):
            y_out.put(x_in.get() * 2.0 + 1.0)

    rtl = _to_rtl(fstage)
    m = _Mod(rtl.mlir, "fstage")
    ce = m.hinted("r0_ce")

    ports = re.findall(r"hw\.instance \"(\w+)\" @\w+\((.*?)\) ->", rtl.mlir)
    assert len(ports) >= 2, f"expected the fmul -> fadd chain, got {ports}"
    for name, args in ports:
        got = re.search(r"ce: %([\w.$-]+):", args)
        assert got, f"instance {name} has no ce port: {args}"
        assert got.group(1) == ce, f"instance {name} rides {got.group(1)}, not {ce}"

    fx = (np.arange(16, dtype=np.float32) * 0.5 - 3.0).astype(np.float32)
    for gap in _STALLS:
        fy = np.zeros(16, dtype=np.float32)
        rtl.cosim(fx, fy, stall_prob=gap)
        assert np.allclose(fy, fx * 2.0 + 1.0), f"gap={gap}: {list(fy)}"


# A banked memory read inside a stream region freezes with the chain. Both
# halves of the split (bank and offset) are held by the same enable: a
# disagreement about when to freeze would read the wrong element.
def test_held_read_address_rides_the_region_shell():
    @kernel
    def banked(A: i32[32], y_out: Stream[i32]):
        for i in range(32):
            y_out.put(A[i] * 3)

    s = banked.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=4)
    rtl = s.export("rtl")
    m = _Mod(rtl.mlir, "banked")
    ce = m.hinted("r0_ce")

    # Every self-holding cell in the region, chain stages and held address
    # halves alike, is enabled by the one shell.
    held = {m.enable_of(reg, inp) for _, reg, inp in m.regs}
    assert held - {None} == {ce}, f"not one shell: {held}"

    A = np.arange(32, dtype=np.int32) * 7 - 11
    for gap in _STALLS:
        y = np.zeros(32, dtype=np.int32)
        rtl.cosim(A, y, stall_prob=gap)
        assert np.array_equal(y, A * 3), f"gap={gap}: {list(y)}"


# No stream accesses => no shell, and no trace of one in the RTL. A rigid
# shell is the IDENTITY: every timing primitive reduces to its unconditional
# form, not a constant-true-enabled special case.
def test_rigid_region_emits_no_shell():
    @kernel
    def gemm(A: f32[8, 8], B: f32[8, 8], C: f32[8, 8]):
        for i, j in allo.grid(8, 8):
            acc: f32 = 0.0
            for k in range(8):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    rtl = _to_rtl(gemm)
    m = _Mod(rtl.mlir, "gemm")

    assert m.hints_like(r"_ce$") == [], "a rigid region derives no shell"
    cells = m.time_base()
    assert cells, "the deep f32 datapath must emit valid-chain stages"
    for label, reg, inp in cells:
        assert m.enable_of(reg, inp) is None, f"{label} is enabled under a rigid shell"

    A = np.random.rand(8, 8).astype(np.float32)
    B = np.random.rand(8, 8).astype(np.float32)
    C = np.zeros((8, 8), dtype=np.float32)
    rtl.cosim(A, B, C)
    assert np.allclose(C, A @ B, atol=1e-4), C


# --- multi-cycle write timing ------------------------------------------------


def _dev(write_latency: int):
    # The built-in device with the default on-chip storage rebound to a
    # write_latency-cycle write.
    d = builtin_device.copy()
    d.set_memory(MemoryKind.LUTRAM, 1, write_latency, 0.5, 0.5)
    d.set_default_memory(MemoryKind.LUTRAM)
    return d


# The deeper write is honored by the scheduler too, not just tolerated by
# the emitter: the memory-carried recurrence's II is read + add + write, so it
# grows one cycle per added write cycle.
def test_multi_cycle_write_costs_scheduled_cycles():
    @kernel
    def accumulate(A: i32[8], B: i32[8]):
        s: i32[8] = 0
        for i in range(8):
            s[0] = s[0] + A[i]
        for i in range(8):
            B[i] = s[0]

    iis = []
    for wr in (1, 2, 3):
        regions = _to_rtl(accumulate, device=_dev(wr)).schedule().func("accumulate")
        iis.append(max(r.ii for r in regions.cyclic()))
    assert iis == [iis[0], iis[0] + 1, iis[0] + 2], iis


# The registers that carry a multi-cycle write ride the region's clock
# enable, so a stream region's back-pressure freezes the in-flight write with
# the rest of the datapath instead of committing it a cycle early.
def test_multi_cycle_write_freezes_under_back_pressure():
    @kernel
    def strbuf(out: i32[8]):
        fifo: Stream[i32]

        @kernel(mapping=[2])
        def pe(out: i32[8], fifo: Stream[i32]):
            p = allo.get_wid(0)
            if p == 0:
                for i in range(8):
                    fifo.put(i * 3)
            else:
                buf: i32[8] = 0
                for i in range(8):
                    buf[i] = fifo.get() + 1
                for i in range(8):
                    out[i] = buf[i]

        pe(out, fifo)

    expect = np.arange(8, dtype=np.int32) * 3 + 1
    for wr in (1, 2, 3):
        for gap in (0.0, 0.6):
            out = np.zeros(8, dtype=np.int32)
            _to_rtl(strbuf, device=_dev(wr)).cosim(out, stall_prob=gap)
            assert np.array_equal(out, expect), f"wr_lat={wr} gap={gap}: {list(out)}"


# --- clock-frequency-aware chaining -------------------------------------------


# The timing/chaining model is clock-frequency sensitive: a 4-deep
# combinational int-add chain splits across more cycles under a tight clock
# than under a loose one.
def test_chaining_inserts_register():
    def chain():
        @kernel
        def c(A: i32[8], out: i32[8]):
            for i in range(8):
                x: i32 = A[i] + A[i]
                y: i32 = x + A[i]
                z: i32 = y + A[i]
                out[i] = z + A[i]

        return c

    # Four combinational int adds (1.2 ns each) cannot fit one 3.33 ns cycle, so
    # the chaining scheduler splits the chain across cycles -- more register
    # stages than under a huge cycle time, where the whole chain settles in one.
    tight = _sched(chain()).cyclic()[0]
    loose = _sched(chain(), freq_mhz=1.0).cyclic()[0]  # a 1000ns cycle
    assert tight.last_t() > loose.last_t()
