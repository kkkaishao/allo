# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the `xcu55c` part has, and what a native operator spends of it.

The numbers are P6's measurements: Vivado 2023.2, ``xcu55c-fsvh2892-2L-e``,
out-of-context synthesis of one DUT per (kind, width), primitives counted off
the netlist. They lived in ``benchmark/area.py`` as a second model running
beside the compiler; this is the same table in the device's own vocabulary, so
there is one declaration and several readers.

Each cost is the measured SHAPE with a measured coefficient, never a curve fit:
an N-bit AND is N LUT6s so it is linear, an adder adds a carry chain, a divider
is quadratic. Where the shape is not structural, or is structural but has one
user and a handful of interesting widths, the measurement itself is the table.

Two places the declaration is coarser than ``area.py``'s arithmetic, both away
from the widths that were measured:

* A carry-chain count is ``ceil(w/8)`` there and ``0.125 * w`` rounded here.
  They agree at every multiple of eight and differ at 9 bits by one CARRY8 out
  of the part's 162,960.
* A table holds the value of the last point at or below its argument, so a
  width between two measured ones reads the lower row: a 48-bit barrel shift
  is priced as the 32-bit one, and anything under 8 bits as the 8-bit one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device import Device


#: What the part has. The card data sheet (DS978) gives 1,304K LUTs, 2,607K
#: registers, 9,024 DSP slices, 70.9 Mb of 36 Kb block RAM and 960 UltraRAM
#: blocks; the exact LUT and flip-flop counts are the Virtex UltraScale+ product
#: table's for XCVU47P, the die xcu55c is built on, and round to the card's.
CAPACITY = {
    "lut": 1_303_680,
    "ff": 2_607_360,
    "dsp": 9_024,
    # DERIVED, not quoted: no data sheet lists CARRY8. An UltraScale+ CLB holds
    # eight LUT6 and one CARRY8 (UG574), so the part has one per eight LUTs.
    "carry8": 162_960,
    "bram36": 2_016,
    # A tile counter, named for its size the way `bram36` is, because the
    # STORAGE realization an array binds to is `uram` and the two share the
    # device's one symbol table: the counter and the structure built out of it
    # are different things and cannot answer to one name.
    "uram288": 960,
}


def declare_xcu55c_area(device: "Device") -> None:
    """Declare P6's measured area model on ``device``: the resources the part
    has, and what one instance of each native combinational operator spends of
    them at a given operand width.

    The chaining delays are the device's own and are read back rather than
    restated, so this only ever adds the resource side of a row that exists.
    """
    # Deferred so the two modules do not import each other at load time:
    # `device` imports this one, and the vocabulary below is its.
    from .device import CombKind, Linear, Quadratic, Table

    lut = device.add_resource("lut", CAPACITY["lut"])
    dsp = device.add_resource("dsp", CAPACITY["dsp"])
    carry8 = device.add_resource("carry8", CAPACITY["carry8"])
    # Declared but unspent below: a delay chain and a storage realization spend
    # these, a combinational operator does not.
    device.add_resource("ff", CAPACITY["ff"])
    device.add_resource("bram36", CAPACITY["bram36"])
    device.add_resource("uram288", CAPACITY["uram288"])

    # A bitwise operator and a multiplexer are one LUT6 per bit exactly.
    logic = {lut: Linear(1.0)}
    # An adder is that plus a carry chain, one CARRY8 per eight bits. A compare
    # keeps no sum and packs two bits per carry stage, so its chain is half.
    addsub = {lut: Linear(1.0), carry8: Linear(0.125)}
    compare = {lut: Linear(1.0), carry8: Linear(0.0625)}
    # A barrel shift is w*ceil(log4 w) LUTs, which is structural but has one
    # user, so it stays the four points it was measured at.
    shift = {lut: Table({8: 16, 16: 32, 32: 96, 64: 192})}
    # A multiplier is DSP48E2s plus glue: one DSP holds up to 18 bits, above
    # which the partial products multiply up and the LUT glue shrinks. Its
    # carry chain is the compare's, over the partial-product adds.
    multiply = {
        lut: Table({8: 39, 16: 39, 32: 15, 64: 15}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10}),
        carry8: Linear(0.0625),
    }
    # Measured 75/286/1086 LUTs at w=8/16/32: quadratic is the structure of the
    # restoring divider, 1.06 is the measurement.
    divide = {lut: Quadratic(1.06), carry8: Linear(5.0)}

    # `neg` is a float sign flip and `icast` is a rename of bits, so neither
    # reaches a cell that synthesis charges for. They stay unpriced.
    for kinds, uses in (
        ((CombKind.AND, CombKind.OR, CombKind.XOR, CombKind.SELECT), logic),
        ((CombKind.ADD, CombKind.SUB), addsub),
        ((CombKind.CMP,), compare),
        ((CombKind.SHL, CombKind.SHR), shift),
        ((CombKind.MUL,), multiply),
        ((CombKind.DIV, CombKind.REM), divide),
    ):
        for kind in kinds:
            delay = device.comb.get(kind.value)
            assert delay is not None, f"{kind.value!r} has no delay to keep"
            device.set_comb_delay(kind, delay, uses=uses)
