# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary the device library is written in."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..device import CombKind, Resource


class Grade(NamedTuple):
    """A speed grade: which of a fabric's timing tables a part reads, and the
    clock that table was characterized at."""

    name: str  # "-2L", as the part number spells it
    default_freq_mhz: float


@dataclass(frozen=True)
class Part:
    """One die. ``capacity`` carries the PRIMARY resources only, in the fabric's
    own vocabulary; a resource the die does not have is ABSENT rather than zero,
    and every storage realization that needs it is then not declared either."""

    name: str  # the MLIR symbol the injected `dcp.device` carries
    part: str  # full vendor part number, the same string the vitis backend takes
    grade: Grade
    capacity: Mapping[str, int]


class StorageTiming(NamedTuple):
    """Access timing of one storage realization, or of a stream channel."""

    read_latency: int
    write_latency: int
    read_ns: float
    write_ns: float


class FabricTiming(NamedTuple):
    """Everything about a fabric that depends on the speed grade. One of these
    per grade the fabric has been characterized at."""

    comb: Mapping[CombKind, float]  # chaining delay, ns
    storage: Mapping[str, StorageTiming]
    stream: StorageTiming


class Derived(NamedTuple):
    """A resource a part does not quote, computed from one it does: an
    UltraScale+ die has one CARRY8 per eight LUTs, and nobody states it."""

    source: str
    divisor: int


class StorageSpec(NamedTuple):
    """What a fabric declares about one storage realization apart from its
    timing: the resources it cannot exist without, and what one instance spends
    over ``(depth, width)``. A die missing any of ``needs`` does not get the
    row, which is how a part with no UltraRAM says so."""

    needs: tuple[str, ...]
    uses: Callable[[Mapping[str, Resource]], dict]


class IPRow(NamedTuple):
    """What one operator core is on one fabric: how many cycles it takes, and
    what it spends. The two travel together because they are one measurement of
    one piece of hardware; a core pipelined to a different latency is different
    hardware with a different area, under its own symbol."""

    latency: int
    area: Mapping[str, int]  # resource name -> count, in the fabric's vocabulary
