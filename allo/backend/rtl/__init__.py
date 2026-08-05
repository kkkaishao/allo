# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from . import reports
from .device import Device, Storage, builtin_device
from .interface import (
    Interfaces,
    ModuleInterface,
    Control,
    Scalar,
    FIFO,
    Memory,
    RegisterFile,
    Result,
    Operator,
)

# The report documents and the vocabulary a caller asserts on. `reports.Memory`
# is an ARRAY in the design and `Memory` above is a boundary port interface: two
# different things that cannot share one name here, so reach the former through
# the `reports` namespace.
from .reports import (
    CompileReport,
    CompilerReport,
    FuncSchedule,
    MicroarchReport,
    RegionKind,
    RegionSchedule,
    RegRole,
    ScheduledOp,
    ScheduleResult,
    ScheduleSettings,
)
from .schedule import has_exact_scheduler
from .qor import QoR, Utilization, estimate
from .core import RTL
from .sim.shell import CosimResult

__all__ = [
    "reports",
    "Device",
    "Storage",
    "builtin_device",
    "Interfaces",
    "ModuleInterface",
    "Control",
    "Scalar",
    "FIFO",
    "Memory",
    "RegisterFile",
    "Result",
    "Operator",
    "CompileReport",
    "CompilerReport",
    "FuncSchedule",
    "MicroarchReport",
    "RegionKind",
    "RegionSchedule",
    "RegRole",
    "ScheduledOp",
    "ScheduleResult",
    "ScheduleSettings",
    "has_exact_scheduler",
    "QoR",
    "Utilization",
    "estimate",
    "RTL",
    "CosimResult",
]
