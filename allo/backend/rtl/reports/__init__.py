# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a compile produced, as data.

Four documents, split by the question each answers:

``schedule``   what the scheduler decided: per kernel a latency, per loop an
               interval and a trip count. Vitis vocabulary where the concept is
               the same one an HLS report names.
``microarch``  what the emitter BUILT: units, multiplexers, storage and the
               register ledger. The binding and everything downstream of it.
``compiler``   the compiler's account of itself: the scheduler's options and
               what one solve cost. Not a property of the design.
``compile``    the three joined, plus the boundary.

The rule they are pruned against: a report describes the design that EXISTS, not
the history of how it got there. A transformation the compiler applied is a log
line; what the array became is a field.

Fields a reader never wants but a cost model cannot work without are grouped
under a ``cost`` member rather than mixed in at the top level, so the shape of
each object says who it is for.
"""

from .compile import CompileReport
from .compiler import CompilerReport, SolveReport
from .microarch import (
    Call,
    FuncUarch,
    Memory,
    MemoryCost,
    MicroarchReport,
    MuxClass,
    RegClass,
    RegionCost,
    RegionUarch,
    RegRole,
    Stream,
    Unit,
)
from .schedule import (
    FuncSchedule,
    RegionKind,
    RegionSchedule,
    RegionScheduleCost,
    ScheduledOp,
    ScheduleResult,
    UnhonoredDirective,
)

__all__ = [
    "CompileReport",
    "CompilerReport",
    "SolveReport",
    "Call",
    "FuncUarch",
    "Memory",
    "MemoryCost",
    "MicroarchReport",
    "MuxClass",
    "RegClass",
    "RegionCost",
    "RegionUarch",
    "RegRole",
    "Stream",
    "Unit",
    "FuncSchedule",
    "RegionKind",
    "RegionSchedule",
    "RegionScheduleCost",
    "ScheduledOp",
    "ScheduleResult",
    "UnhonoredDirective",
]
