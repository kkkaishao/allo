# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RTL backend: SDC scheduling and hw/Verilog emission.

``kernel.schedule().export("rtl")`` returns an :class:`RTL` handle, the entry
point to the flow; the rest of this package is the operator timing library and
the schedule-result model that handle returns.
"""

from .operator_library import OperatorLibrary, OP_KINDS, OP_DTYPES
from .schedule import (
    ScheduleResult,
    FuncSchedule,
    RegionSchedule,
    ScheduledOp,
    RegionKind,
)
from .core import RTL
from .sim.shell import CosimResult

__all__ = [
    "OperatorLibrary",
    "OP_KINDS",
    "OP_DTYPES",
    "ScheduleResult",
    "FuncSchedule",
    "RegionSchedule",
    "ScheduledOp",
    "RegionKind",
    "RTL",
    "CosimResult",
]
