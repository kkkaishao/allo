# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the RTL tests.

The named latencies (``FADD``, ``FMUL``, ...) are read from the shipped built-in
operator library -- the one the RTL backend uses by default -- so the II
assertions read as the recurrence arithmetic they check while tracking the
library's real numbers.
"""

from __future__ import annotations

from allo.backend.rtl import OperatorLibrary, RTL, ScheduleResult

LIB = OperatorLibrary.builtin("builtin")

_DICT = LIB.to_dict()
_LAT = {(r.get("op"), r.get("dtype")): r["latency"] for r in _DICT["operators"]}
_PRIM = {p["name"]: p["latency"] for p in _DICT["memory"]["primitives"]}

FADD = FSUB = _LAT[("add", "float")]  # floating-point add/sub latency (cycles)
FMUL = _LAT[("mul", "float")]  # floating-point multiply latency
FDIV = _LAT[("div", "float")]  # floating-point divide latency
IMUL = _LAT[("mul", "int")]  # integer multiply latency
MEM = _PRIM["lutram"]["read"]  # default (LUTRAM) read / write latency

# A memory-carried accumulate (`M[x] += ...`) closes a distance-1 recurrence
# read -> add -> write, so its II is the sum; a scalar-carried accumulate keeps
# the partial in a register, so its II is just the add latency.
MEM_REDUCE_II = MEM + FADD + MEM


def _to_rtl(kernel, **kw) -> RTL:
    """Export ``kernel`` to the RTL backend (the default library is ``LIB``)."""
    return kernel.schedule().export("rtl", **kw)


def _sched(kernel, **kw) -> ScheduleResult:
    """Schedule ``kernel`` through the RTL backend."""
    return _to_rtl(kernel, **kw).schedule()


def _latency(kernel, **kw):
    """Whole-kernel latency (cycles) of ``kernel`` scheduled on its own; ``None``
    when a trip count is not statically known."""
    return _sched(kernel, **kw).func(kernel.__name__).latency


def _iis(regions):
    """Sorted IIs of ``regions``; a dynamic-trip sequential wrapper (``ii`` is
    ``None``) is skipped."""
    return sorted(r.ii for r in regions if r.ii is not None)
