# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types and measured constants of the device library."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..device import CombKind, Cost, Linear, Piecewise, Resource, Table

#: Below this depth a delay chain stays in flip-flops; at or above it Vivado
#: extracts a shift register, even with every stage reset. An SRL32E holds 32
#: stages, so an extracted chain occupies ``ceil(depth/32)`` SLICEM sites a bit.
SRL_MIN_DEPTH = 4

#: LUTs per bit of a ``k``-source one-hot AND-OR select, measured for k = 2..40
#: and listed where the staircase steps. A LUT6 absorbs three (data, select)
#: pairs and ~2.5 more per further level, so the curve is linear in k, not
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

#: LUTs per bit of a select, as a cost over its fan-in: the measured staircase
#: across the swept range, and past it the least-squares line through those
#: points (slope 0.408449, intercept 0.263495, within 0.20 LUT of the last
#: measurement). A region shares an operator over more sources than the sweep
#: covered, where a staircase would clamp flat and charge nothing for the rest.
MUX_LUT_COST = Piecewise(
    max(MUX_LUT_PER_BIT) + 1,
    Table(MUX_LUT_PER_BIT),
    Linear(0.4084490071, base=0.2634952767),
)

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
    """One die. ``capacity`` carries the primary resources only, named as the
    fabric names them. A resource the die lacks is absent rather than zero, and
    every storage realization that needs it is then left undeclared."""

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

    #: Chaining delay in ns as a function of the operand width, which matters:
    #: a 32-bit divider measures 23.7 ns against an 8-bit one's 4.3.
    comb: Mapping[CombKind, Cost]
    storage: Mapping[str, StorageTiming]
    stream: StorageTiming
    reg_ns: float = 0.0


class Derived(NamedTuple):
    """A resource a part does not quote, computed from one it does. An
    UltraScale+ die has one CARRY8 per eight LUTs."""

    source: str
    divisor: int


class StorageSpec(NamedTuple):
    """What a fabric declares about one storage realization apart from its
    timing: the resources it cannot exist without, and what one instance spends
    over ``(depth, width)``. A die missing any of ``needs`` does not get the
    row."""

    needs: tuple[str, ...]
    uses: Callable[[Mapping[str, Resource]], dict]


class IPRow(NamedTuple):
    """Latency and area of one operator core on one fabric, from a single
    measurement. A core pipelined to another latency is a separate row with its
    own symbol and area.

    Several rows of one archetype are candidates the library chooses between.
    ``mnemonic`` overrides the stem of the symbol, which is how two rows that
    share a latency are still named apart; ``None`` takes the archetype's own.
    """

    latency: int
    area: Mapping[str, int]  # resource name -> count, in the fabric's vocabulary
    mnemonic: str | None = None
