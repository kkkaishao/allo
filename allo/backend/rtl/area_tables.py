# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the `xcu55c` part has, and what each thing it builds spends of it.

The numbers are P6's and P7's measurements: Vivado 2023.2,
``xcu55c-fsvh2892-2L-e``, out-of-context synthesis of one DUT per (kind, width),
one Xilinx Floating-Point core per device operator at its declared latency, and
a sweep per multiplexer fan-in, chain depth and array shape; primitives counted
off the netlist. They lived in ``benchmark/area.py`` as a second model running
beside the compiler; this is the same table in the device's own vocabulary, so
there is one declaration and several readers.

Each cost is the measured SHAPE with a measured coefficient, never a curve fit:
an N-bit AND is N LUT6s so it is linear, an adder adds a carry chain, a divider
is quadratic. Where the shape is not structural, or is structural but has one
user and a handful of interesting widths, the measurement itself is the table.

Two places the declaration is coarser than ``area.py``'s arithmetic:

* A table holds the value of the last point at or below its argument, so a
  width between two measured ones reads the lower row. A 48-bit barrel shift is
  priced as the 32-bit one, and anything under 8 bits as the 8-bit one.
* A ``Tiled`` cost is the array's BITS over a tile's, so it under-counts an
  array too shallow to fill a tile's depth: a 4x32 block RAM is one whole
  BRAM36, not ``ceil(128/36864)``. The same approximation for every tiled row.

And one shape the vocabulary cannot hold at all, which is written up where it
bites, on ``chain``.
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
    # DERIVED for the same reason: only a SLICEM LUT can hold a shift register
    # or a distributed RAM, and about half an UltraScale+ device's slices are
    # SLICEM (UG574), so the part has one SRL site per two LUTs.
    "slicem_lut": 651_840,
    "bram36": 2_016,
    # A tile counter, named for its size the way `bram36` is, because the
    # STORAGE realization an array binds to is `uram` and the two share the
    # device's one symbol table: the counter and the structure built out of it
    # are different things and cannot answer to one name.
    "uram288": 960,
}

#: The device operator IPs, each the Xilinx Floating-Point core at the latency
#: `device.py` declares, as (LUT, FF, DSP, CARRY8). An IP's signature fixes its
#: widths, so every one of these is a constant.
IP_AREA = {
    "fadd_l7": (247, 315, 2, 10),
    "fsub_l7": (247, 315, 2, 10),
    "fmul_l4": (115, 173, 2, 9),
    "fdiv_l12": (766, 1381, 0, 111),
    "fcmp_l1": (64, 12, 0, 7),
    "dadd_l14": (710, 872, 3, 30),
    "dsub_l14": (710, 872, 3, 30),
    "dmul_l9": (205, 542, 7, 16),
    "ddiv_l24": (3185, 6035, 0, 399),
    "dcmp_l1": (118, 12, 0, 12),
    "i2f_l3": (165, 228, 0, 11),
    "f2i_l3": (183, 232, 0, 6),
    "fcvt_l2": (50, 99, 0, 1),
    # bf16 has no measured core; priced from its f32 sibling by width.
    "bfadd_l4": (124, 158, 1, 5),
    "bfsub_l4": (124, 158, 1, 5),
    "bfmul_l2": (58, 87, 1, 5),
    "bf2f_l2": (25, 50, 0, 1),
}

#: LUTs per bit of a `k`-source one-hot AND-OR select, over the fan-ins P6
#: measured (k = 2 to 40), listed where the staircase steps. A LUT6 absorbs
#: three (data, select) pairs and about 2.5 more per further level, so the curve
#: is LINEAR in k rather than logarithmic; it is a table and not a `Linear`
#: because 2.5 pairs per level is a measurement and the ceiling it sits under is
#: what makes the per-bit cost whole. One source is a wire and costs nothing.
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

#: Below this depth a delay chain stays in flip-flops; at or above it Vivado
#: extracts an SRL, even though the emitter resets every stage. Measured here
#: exactly.
SRL_MIN_DEPTH = 4

#: SLICEM sites per bit of an extracted chain: an SRL32E holds 32 stages, so the
#: staircase is `ceil(depth/32)` and it starts at the extraction threshold.
#: `Tiled(32)` IS that ceiling, but a tiled cost reads the whole `(depth,
#: width)` tuple and this ceiling is per bit, so the staircase is spelled out.
SRL_SITES_PER_BIT = {1: 0, SRL_MIN_DEPTH: 1}
SRL_SITES_PER_BIT.update({32 * i + 1: i + 1 for i in range(1, 17)})

#: Fabric LUTs per bit of an array that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of `depth*width` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0


def declare_xcu55c_area(device: "Device") -> None:
    """Declare P6's and P7's measured area model on ``device``: the resources
    the part has, and what one instance of each thing it builds spends of them.

    The chaining delays and the storage timing are the device's own and are read
    back rather than restated, so this only ever adds the resource side of a row
    that exists.
    """
    # Deferred so the two modules do not import each other at load time:
    # `device` imports this one, and the vocabulary below is its.
    from .device import CombKind, Const, Linear, Quadratic, Step, Table, Tiled

    lut = device.add_resource("lut", CAPACITY["lut"])
    ff = device.add_resource("ff", CAPACITY["ff"])
    dsp = device.add_resource("dsp", CAPACITY["dsp"])
    carry8 = device.add_resource("carry8", CAPACITY["carry8"])
    # Its own counter rather than part of `@lut`, even though an SRL occupies a
    # LUT site: a design runs out of SLICEM with ordinary LUTs to spare, and a
    # delay chain competes for it with a distributed RAM and not with logic.
    # Folding the two would price them as interchangeable, which they are not.
    slicem = device.add_resource("slicem_lut", CAPACITY["slicem_lut"])
    bram36 = device.add_resource("bram36", CAPACITY["bram36"])
    uram288 = device.add_resource("uram288", CAPACITY["uram288"])

    # --- native combinational operators, over (width) -----------------------
    # A bitwise operator and a multiplexer are one LUT6 per bit exactly.
    logic = {lut: Linear(1.0)}
    # An adder is that plus a carry chain, one CARRY8 per eight bits. A compare
    # keeps no sum and packs two bits per carry stage, so its chain is half.
    # `Tiled` and not a linear coefficient: a carry chain is a CEILING, so an
    # 9-bit adder takes two CARRY8s and `0.125 * 9` rounded takes one.
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    # A barrel shift is w*ceil(log4 w) LUTs, which is structural but has one
    # user, so it stays the four points it was measured at.
    shift = {lut: Table({8: 16, 16: 32, 32: 96, 64: 192})}
    # A multiplier is DSP48E2s plus glue: one DSP holds up to 18 bits, above
    # which the partial products multiply up and the LUT glue shrinks. Its
    # carry chain is the compare's, over the partial-product adds.
    multiply = {
        lut: Table({8: 39, 16: 39, 32: 15, 64: 15}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10}),
        carry8: Tiled(16),
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

    # --- operator IPs, over (width) -----------------------------------------
    for operator in device.operators:
        measured = IP_AREA.get(operator.func_name)
        if measured is None:
            continue  # a user IP nobody has synthesized: unpriced, not free
        device.set_operator_uses(
            operator,
            {
                resource: Const(n)
                for resource, n in zip((lut, ff, dsp, carry8), measured)
                if n
            },
        )

    # --- the multiplexer, over (k, width) -----------------------------------
    # Priced as a STRUCTURE and not as the `and`/`or` cone the emitter writes: a
    # LUT6 absorbs three (data, select) pairs, so synthesis fuses the whole
    # thing and pricing the operations separately over-counts it about fivefold.
    device.set_mux_uses({lut: (Table(MUX_LUT_PER_BIT), Linear(1.0))})

    # --- the value delay chain, over (depth, width) -------------------------
    # Past the extraction threshold a chain stops being flip-flops: it becomes
    # `width` SRL sites per 32 stages, plus one LUT per bit of addressing and
    # output multiplexing, plus the head and tail stages. That cliff is what
    # `Step` is for, and it is the single largest disagreement between the
    # scheduling objective's register term (`depth * width` flip-flops) and the
    # part.
    #
    # Two terms of P7's measurement do NOT survive the translation, and both are
    # under-counts. A cost is one factor per parameter multiplied together, and
    # (a) the extracted chain's `2*width + depth - 1` flip-flops are a SUM of a
    # per-bit and a per-stage term, which no product is, so only the `2*width`
    # is declared; (b) a chain narrower than eight bits is left in flip-flops
    # whatever its depth, and that is a second cliff on the OTHER parameter,
    # which a factor over depth cannot see. Fixing either needs a cost over the
    # whole tuple, which is what `Tiled` is and what nothing else is.
    device.set_chain_uses(
        {
            ff: (Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0)),
            lut: (Step(SRL_MIN_DEPTH, 0.0, 1.0), Linear(1.0)),
            slicem: (Table(SRL_SITES_PER_BIT), Linear(1.0)),
        }
    )

    # --- storage realizations, over (depth, width) --------------------------
    # A register file is what an array that failed RAM inference becomes: every
    # word gets a data multiplexer and a write decode, so it costs the whole
    # array in flip-flops and twice it in LUTs. Measured at 512x32: one BRAM18
    # against 33,245 LUTs and 16,416 flip-flops.
    device.set_storage_uses(
        "register",
        {
            lut: (Linear(MULTIWRITE_LUT_PER_BIT), Linear(1.0)),
            ff: (Linear(1.0), Linear(1.0)),
        },
    )
    # The tiled rows: a structure holds so many bits however the array is cut.
    # Distributed RAM and shift registers live in SLICEM, block RAM and
    # UltraRAM in their own columns.
    device.set_storage_uses("lutram", {slicem: Tiled(64)})
    device.set_storage_uses("srl", {slicem: Tiled(32)})
    device.set_storage_uses("bram", {bram36: Tiled(36864)})
    device.set_storage_uses("uram", {uram288: Tiled(294912)})
