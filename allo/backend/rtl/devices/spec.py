# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types and measured constants of the device library."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

from ..device import CombKind, Const, Cost, Linear, Piecewise, Resource, Table, Tiled

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
#: measurement).
MUX_LUT_COST = Piecewise(
    max(MUX_LUT_PER_BIT) + 1,
    Table(MUX_LUT_PER_BIT),
    Linear(0.4084490071, base=0.2634952767),
)

#: Fabric LUTs per bit of an array that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of ``depth*width`` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0

#: Entries of a constant table one LUT covers, per output bit. A LUT6 computes
#: any function of six inputs, so it is a 64-entry one-bit lookup.
ROM_ENTRIES_PER_LUT = 64

#: LUT sites a ``depth`` x ``width`` constant table takes: one per 64 entries of
#: each bit, plus one more per eight of those to select between them (the
#: narrower selects ride the slice's own F7/F8 and take no site of their own).
#: Every fabric measures alike here: exactly 32 / 160 / 288 / 576 / 1152 LUT6
#: at 64 / 256 / 512 / 1024 / 2048 x 32, and within 1.4% at 4096 and 16384. A
#: table with regular contents minimizes below it.
ROM_LUT_COST = [
    (Tiled(ROM_ENTRIES_PER_LUT), Linear(1.0)),
    (
        Piecewise(2 * ROM_ENTRIES_PER_LUT, Const(0.0), Tiled(8 * ROM_ENTRIES_PER_LUT)),
        Linear(1.0),
    ),
]


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
    #: A channel's own timing. ``read_latency`` is 0 because ``seq.fifo`` is
    #: show-ahead: the head is on the wire in the cycle ``valid`` is high. The
    #: two delays are not characterized; they are copied from the ``lutram``
    #: row, an SRL-backed FIFO's output being a distributed-RAM read. Retake
    #: them with a FIFO DUT before trusting a chaining decision that turns on
    #: them.
    stream: StorageTiming
    reg_ns: float = 0.0
    #: The routed marginal delay of a one-hot select over its fan-in at a
    #: 32-bit reference width, and the unitless factor its actual width scales
    #: it by (pinned to 1.0 at 32). From the mux DUT sweep; None where that
    #: grade has not been measured.
    mux: Cost | None = None
    mux_w: Cost | None = None
    #: The routed delay of a constant table's read over its depth at the same
    #: 32-bit reference width, and the factor its width scales it by. The one
    #: read delay that grows with the array: a table deep enough not to close is
    #: held in a memory instead. None where the grade has not been measured,
    #: which leaves the table unrealizable there.
    rom: Cost | None = None
    rom_w: Cost | None = None


class Derived(NamedTuple):
    """A resource a part does not quote, computed from one it does. An
    UltraScale+ die has one CARRY8 per eight LUTs."""

    source: str
    divisor: int


class StorageSpec(NamedTuple):
    """What a fabric declares about one storage realization apart from its
    timing: the resources it cannot exist without, what one instance spends over
    ``(depth, width)``, and how many ports one instance has. A die missing any
    of ``needs`` does not get the row.

    A port limit of ``None`` is no limit, which is what the scatter row takes:
    one cell per element is not addressed. ``inst_ports`` is the pool the two
    directions draw on together, declared where a port serves a read or a write
    (a block RAM) and omitted where the directions are independent structures (a
    LUT RAM's write port against its one addressed read). All three are per
    instance, not per array: the compiler decides how many instances hold an
    array, every copy taking every write and serving ``inst_reads`` reads of its
    own.

    ``ram_style`` is the vendor attribute that pins an array to the row;
    ``can_init`` is whether the structure comes up holding contents."""

    needs: tuple[str, ...]
    uses: Callable[[Mapping[str, Resource]], dict]
    inst_reads: int | None = None
    inst_writes: int | None = None
    inst_ports: int | None = None
    ram_style: str | None = None
    can_init: bool = True
    #: Whether this is the constant table: a lookup built out of logic, no
    #: address bus and no port limit, priced and timed over the array's shape.
    is_table: bool = False


class IPRow(NamedTuple):
    """Latency and area of one operator core on one fabric, from a single
    measurement. A core pipelined to another latency is a separate row with its
    own symbol and area.

    Several rows of one archetype are candidates the library chooses between.
    ``mnemonic`` overrides the stem of the symbol so two rows sharing a latency
    are named apart; ``None`` takes the archetype's own. ``area`` splits
    state-holding LUT sites (a core's internal shift registers) into
    ``slicem_lut``, apart from the ``lut`` logic sites.
    """

    latency: int
    area: Mapping[str, int]  # resource name -> count, in the fabric's vocabulary
    mnemonic: str | None = None
    #: Measured input cone of this row, overriding the archetype's default.
    #: A single-cycle core is combinational up to its output register, so its
    #: cone is nearly the whole measured period, not the registered-input 0.5.
    in_delay_ns: float | None = None
    #: Least clock period the row's internal stages are warranted at. ``None``
    #: takes the grade's characterization period: a row is only ever warranted
    #: at the clock it was measured to close.
    min_period_ns: float | None = None
