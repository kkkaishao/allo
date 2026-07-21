# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the RTL tests.

The named latencies (``FADD``, ``FMUL``, ...) are read from the shipped built-in
device -- the one the RTL backend uses by default -- so the II assertions read as
the recurrence arithmetic they check while tracking the device's real numbers.
"""

from __future__ import annotations

from allo.backend.rtl import RTL, ScheduleResult, MemoryKind, builtin_device


# Operator latencies keyed by (kind, arg bit width), read off the built-in
# operator IPs (each an `@ip(optype=...)`).
def _key(op):
    dt = op.parse_argument_annotations()[0]
    return (op.optype.value, int(dt.primitive_width))


_LAT = {_key(o): o.timing.latency for o in builtin_device.operators}

FADD = FSUB = _LAT[("add", 32)]  # floating-point add/sub latency (cycles)
FMUL = _LAT[("mul", 32)]  # floating-point multiply latency
FDIV = _LAT[("div", 32)]  # floating-point divide latency
IMUL = 0  # integer multiply is combinational (latency 0)
MEM = builtin_device.memory[MemoryKind.LUTRAM].read_latency  # default read/write

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
