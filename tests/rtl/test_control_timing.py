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
from allo.lang import i32, f32, index, Stream
from allo.backend.rtl.device import builtin_device, Port

sys.path.insert(0, os.path.dirname(__file__))
from _common import Mod, _sched, _to_rtl, _iis, COMB, PERIOD_NS  # noqa: E402

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
    d.set_default_storage(
        d.add_storage(
            "lutram",
            ports=Port.T2P,
            read_latency=1,
            write_latency=write_latency,
            read_delay_ns=0.5,
            write_delay_ns=0.5,
        )
    )
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

    # The device's write latency is the subject, so `s` has to stay a memory:
    # the automatic complete partition would give it registers and time the
    # write at zero.
    iis = []
    for wr in (1, 2, 3):
        rtl = _to_rtl(accumulate, device=_dev(wr), scalarize_threshold=0)
        regions = rtl.schedule().func("accumulate")
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

    # The premise, stated against the device rather than assumed: four
    # combinational int adds do not fit one default cycle. A device whose adds
    # got faster would leave the test passing for the wrong reason.
    assert 4 * COMB["add"] > PERIOD_NS
    # So the chaining scheduler splits the chain across cycles -- more register
    # stages than under a huge cycle time, where the whole chain settles in one.
    tight = _sched(chain()).cyclic()[0]
    loose = _sched(chain(), freq_mhz=1.0).cyclic()[0]  # a 1000ns cycle
    assert tight.last_t() > loose.last_t()


def test_a_reified_bound_is_priced_against_the_clock(capfd):
    # What the reifier synthesizes AFTER the solve is combinational logic the
    # chaining scheduler never saw: a symbolic loop bound is expanded from its
    # affine map into arith ops stamped `start = 0`, so no chain break can land
    # in it and no register bounds its depth. It is only checkable at emission,
    # against the same clock the schedule was cut to.
    def band():
        @kernel
        def k(A: i32[64], out: i32[8]):
            for i in range(8):
                s: i32 = 0
                for j in range(i // 3 * 2 + 1):  # a floordiv bound map
                    s = s + A[j]
                out[i] = s

        return k

    # The premise, against the device: a signed floordiv expands to a divider
    # plus its sign correction (cmp, sub, select on each side), which alone
    # overruns the default period.
    assert COMB["div"] + COMB["sub"] + COMB["select"] > PERIOD_NS

    _to_rtl(band()).compile()
    text = "".join(capfd.readouterr())
    assert "AFTER the schedule was cut" in text
    assert "misses timing" in text

    # The clock is what decides, so a period the whole cone fits in reports
    # nothing. This is also the only remedy the message can offer: the
    # expression is the compiler's, not a binding the user can withdraw.
    _to_rtl(band(), freq_mhz=1.0).compile()  # a 1000ns cycle
    assert "AFTER the schedule was cut" not in "".join(capfd.readouterr())


def test_an_address_cone_is_charged_to_the_port_it_feeds():
    # An address never becomes an operation: it is folded into the access's
    # affine map, so no dependence carries its delay and only the access's own
    # operator type can account for it. These two kernels run the same four adds
    # over the same trip count and differ only in what it costs to reach the
    # element -- `flat` addresses with the bare counter, `cone` sums three
    # shifted terms. At 2 ns the compute alone fits and only the cone does not.
    @kernel
    def flat(A: i32[512], B: i32[512], out: i32[512]):
        for i in range(64):
            out[i] = A[i] + B[i] + A[i] + B[i]

    @kernel
    def cone(A: i32[8, 8, 8], B: i32[8, 8, 8], out: i32[8, 8, 8]):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    out[i + 1, j + 1, k + 1] = (
                        A[i, j, k] + B[i, j, k] + A[i, j, k] + B[i, j, k]
                    )

    at500 = {
        n: _sched(k, freq_mhz=500).cyclic()[0]
        for n, k in (("flat", flat), ("cone", cone))
    }
    assert at500["cone"].length > at500["flat"].length


def test_an_address_that_follows_the_counters_is_carried_in_a_register():
    # Address strength reduction. Every term of `i*400 + j*20 + k + c` is a
    # constant multiple of an enclosing counter, so consecutive iterations
    # differ by a constant: each term becomes a register the controller
    # advances beside the counter it follows, and the address is their sum. The
    # constant multiplies -- the arithmetic that dominates an address, and the
    # reason it was the widest cone in the datapath -- are gone entirely.
    @kernel
    def stencil(A: i32[20, 20, 20], out: i32[20, 20, 20]):
        for i in range(18):
            for j in range(18):
                for k in range(18):
                    out[i + 1, j + 1, k + 1] = A[i, j, k] + 1

    mod = _to_rtl(stencil)
    m = mod.mlir
    assert "comb.mul" not in m, "a constant stride survived on the address path"
    # One scaled counter per level, shared by the two accesses, except at the
    # outermost, where `out`'s own constant 421 rides in the register's reset
    # value instead of an adder on the address path. That fourth register is
    # what buys the adder off the memory port's setup, and a register is the
    # cheap side of that trade.
    assert sorted(set(re.findall(r"r\d+_addr\d+", m))) == [
        "r0_addr0",
        "r0_addr1",
        "r1_addr0",
        "r2_addr0",
    ]
    inits = dict(
        re.findall(r"%(r\d+_addr\d+) = seq\.compreg [^\n]*reset %rst, %(\w+)", m)
    )
    consts = dict(re.findall(r"%(\w+) = hw\.constant (-?\d+)", m))
    assert sorted(consts[inits[r]] for r in ("r0_addr0", "r0_addr1")) == ["0", "421"]

    A = (np.arange(8000, dtype=np.int32) % 251).reshape(20, 20, 20)
    out = np.zeros((20, 20, 20), np.int32)
    mod.cosim(A, out)
    exp = np.zeros((20, 20, 20), np.int32)
    exp[1:19, 1:19, 1:19] = A[0:18, 0:18, 0:18] + 1
    assert np.array_equal(out, exp)


def test_a_subscript_that_cannot_be_carried_keeps_the_row_its_register():
    # PARTIAL strength reduction. An address is not one decision. `A[i, c]` has a
    # row that follows a counter and a column that never can: `c` is a boundary
    # scalar, so no register advances with it. Taking the address as one decision
    # would cost the row its register as well, rebuilding `i*20` every cycle to
    # add `c` to it. The row reduces on its own.
    @kernel
    def colsum(A: i32[12, 20], c: index, out: i32[12]):
        for i in range(12):
            out[i] = A[i, c]

    mod = _to_rtl(colsum)
    m = mod.mlir
    # The row stride is a register the controller advances by 20, not a multiply
    # on the address path: 20 is no power of two, so one left there would be a
    # visible `comb.mul` by it (`mulConst` leaves the recoding to synthesis).
    # Asked of that constant rather than of `comb.mul` at large, since a runtime
    # loop bound negates with one too and that is control, not address.
    # Any width: the stride register is built at the range it walks, not at the
    # counter's width, so `20` is a constant of that register's own type.
    twenty = set(re.findall(r"(%c20_i\d+\w*) = hw\.constant 20", m))
    assert twenty, "no stride of 20 anywhere: the test measures nothing"
    assert any(
        re.search(rf"comb\.add %r0_addr\d+, {c}\b", m) for c in twenty
    ), "the row stride is not carried in a register"
    assert not any(
        re.search(rf"comb\.mul [^\n]*{c}\b", m) for c in twenty
    ), "a row stride survived beside a column that did"
    A = (np.arange(240, dtype=np.int32) % 251).reshape(12, 20)
    out = np.zeros(12, np.int32)
    mod.cosim(A, 7, out)
    assert np.array_equal(out, A[:, 7])


def test_normalizing_a_strided_loop_lets_its_nest_coalesce():
    # Loop normalization, and what it is FOR. Coalescing states a precondition
    # nothing else establishes (lower bound 0, step 1), so a nest `s.unroll`
    # left stepping by 2 would be refused for a property nothing fixes.
    # Normalized, the step moves into the subscript and the band coalesces into
    # one region running at II=1.
    #
    # The stride is on the INNER loop in the first kernel and on the OUTER loop
    # in the second, and the two are not the same case. Normalizing leaves the
    # original induction variable behind as an `affine.apply`; on an outer level
    # that op stands between the two loops, so the nest stops being perfect and
    # the normalization meant to open the band is what closes it. It only
    # coalesces because the leftover is sunk into the innermost body.
    @kernel
    def inner_stride(A: i32[8, 8], out: i32[8, 8]):
        for i in range(8):
            for j in range(0, 8, 2):
                out[i, j] = A[i, j] + 1
                out[i, j + 1] = A[i, j + 1] + 1

    @kernel
    def outer_stride(A: i32[8, 8], out: i32[8, 8]):
        for i in range(0, 8, 2):
            for j in range(8):
                out[i, j] = A[i, j] + 1
                out[i + 1, j] = A[i + 1, j] + 1

    A = (np.arange(64, dtype=np.int32) % 251).reshape(8, 8)
    for k in (inner_stride, outer_stride):
        mod = _to_rtl(k)
        assert (
            len(mod.schedule().func(k.__name__).cyclic(wrappers=True)) == 1
        ), f"{k.__name__}: the strided band did not coalesce into one region"
        assert _iis(mod.schedule().func(k.__name__).regions) == [1]
        out = np.zeros((8, 8), np.int32)
        mod.cosim(A, out)
        assert np.array_equal(out, A + 1)
