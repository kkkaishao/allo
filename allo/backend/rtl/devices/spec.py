# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary the device library is written in."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..device import CombKind, Cost, Resource

#: Below this depth a delay chain stays in flip-flops; at or above it Vivado
#: extracts a shift register, even though the emitter resets every stage.
SRL_MIN_DEPTH = 4

#: SLICEM sites per bit of an extracted chain: an SRL32E holds 32 stages, so the
#: staircase is ``ceil(depth/32)`` from the extraction threshold on.
SRL_SITES_PER_BIT = {1: 0, SRL_MIN_DEPTH: 1}
SRL_SITES_PER_BIT.update({32 * i + 1: i + 1 for i in range(1, 17)})

#: LUTs per bit of a ``k``-source one-hot AND-OR select, measured for k = 2..40
#: and listed where the staircase steps. A LUT6 absorbs three (data, select)
#: pairs and ~2.5 more per further level, so the curve is LINEAR in k, not
#: logarithmic. One source is a wire and costs nothing.
MUX_LUT_PER_BIT = {
    1: 0,
    2: 1,
    4: 2,
    6: 3,
    9: 4,
    11: 5,
    14: 6,
    16: 7,
    19: 8,
    21: 9,
    24: 10,
    26: 11,
    29: 12,
    31: 13,
    34: 14,
    36: 15,
    39: 16,
}

#: Fabric LUTs per bit of an array that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of ``depth*width`` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0


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

    #: Chaining delay in ns, as a function of the OPERAND WIDTH, since a 32-bit
    #: divider was measured at 23.7 ns against an 8-bit one's 4.3.
    comb: Mapping[CombKind, Cost]
    storage: Mapping[str, StorageTiming]
    stream: StorageTiming
    reg_ns: float = 0.0


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
