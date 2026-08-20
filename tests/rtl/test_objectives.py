# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler objective knobs: the ``O`` direction and the clock margin."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _impls, _to_rtl  # noqa: E402
from allo.backend.rtl.devices import default_device  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def _mixed_kernel():
    @kernel
    def mx(A: i32[32], B: i32[32], out: i32[1]):
        s: i32 = 0
        for i in range(32):
            s = s + A[i] * B[i]
        t: i32 = s * 3
        u: i32 = t * 5
        out[0] = t + u

    return mx


def _run(rtl):
    A = np.arange(32, dtype=np.int32)
    B = np.arange(32, dtype=np.int32) + 2
    out = np.zeros(1, dtype=np.int32)
    rtl.cosim(A, B, out)
    s = int((A.astype(np.int64) * B).sum())
    assert out[0] == np.int32(s * 18)


def test_area_objective_holds_the_span_leash():
    # O="area" ships no slower than the heuristic schedule (the leash), and
    # the design still computes.
    heuristic = _to_rtl(_mixed_kernel()).schedule().func("mx").latency
    rtl = _to_rtl(_mixed_kernel()).set_scheduler_opt(scheduler="exact", O="area")
    latency = rtl.schedule().func("mx").latency
    assert latency is not None and latency <= heuristic
    _run(rtl)


def test_area_objective_pins_an_explicit_pipeline_ii():
    # Under O="area" an explicit pipeline(ii=n) is a ceiling as well as a
    # floor, so the solved interval is exactly n.
    s = _mixed_kernel().schedule()
    s.pipeline("i", ii=2)
    rtl = s.export("rtl").set_scheduler_opt(scheduler="exact", O="area")
    assert rtl.schedule().cyclic()[0].interval == 2
    _run(rtl)


def test_heuristic_ignores_the_objective():
    # The heuristic solves spans only; O passes through without effect.
    rtl = _to_rtl(_mixed_kernel()).set_scheduler_opt(O="area")
    assert rtl.schedule().compiler.options.O == "area"
    _run(rtl)


def test_freq_objective_sweeps_the_period_and_writes_the_clock_back():
    # O="freq" probes periods below the requested clock, holds the span within
    # span_tolerance, and the handle's clock follows the winner; compile then
    # tightens it once more to the realized critical path, held under every
    # bound row's warranted period.
    rtl = _to_rtl(_mixed_kernel(), freq_mhz=50.0).set_scheduler_opt(O="freq")
    result = rtl.schedule()
    assert result.sweep and result.sweep[0].cycle_ns == pytest.approx(20.0)
    n0 = result.sweep[0].latency
    assert rtl.freq_mhz > 50.0
    fn = result.func("mx")
    assert fn.latency <= n0 * 1.1
    est = rtl.estimation  # compiles, which tightens the clock to fmax
    floors = {o.symbol: o.timing.min_period_ns for o in default_device.operators}
    cap = max(floors[i] for i in _impls(result))
    assert rtl.freq_mhz == pytest.approx(min(est.fmax, 1000.0 / cap))
    assert est.clock_mhz == pytest.approx(rtl.freq_mhz)
    _run(rtl)


def test_tighten_clock_moves_the_operating_clock_to_the_realized_path():
    # Any compiled design may be reclocked at its realized critical path
    # without recompiling; the report's clock follows.
    rtl = _to_rtl(_mixed_kernel())
    fmax = rtl.estimation.fmax
    mhz = rtl.tighten_clock()
    assert mhz == pytest.approx(fmax) and rtl.freq_mhz == pytest.approx(fmax)
    assert rtl.estimation.clock_mhz == pytest.approx(mhz)
    _run(rtl)


def test_clock_margin_splits_model_from_operating_period():
    # A margin cuts every chain to (1 - u) * cycle_ns while the design stays
    # clocked at cycle_ns; the QoR reports both periods.
    rtl = _to_rtl(_mixed_kernel(), freq_mhz=200.0).set_scheduler_opt(
        clock_margin=0.25
    )
    assert rtl.schedule().cycle_ns == pytest.approx(3.75)
    est = rtl.estimation
    assert est.fmax_target == pytest.approx(1000.0 / 3.75)
    assert est.clock_mhz == pytest.approx(200.0)
    assert "clocked at 200.0 MHz" in est.timing_report()
    _run(rtl)
