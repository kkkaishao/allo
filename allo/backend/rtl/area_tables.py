# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The xcu55c device's measured area/resource cost model: what the part has,
and what each structure it builds spends of it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device import Device


#: Part capacity, from the DS978 data sheet; exact LUT/FF counts are the Virtex
#: UltraScale+ product table's for XCVU47P, the die xcu55c is built on.
CAPACITY = {
    "lut": 1_303_680,
    "ff": 2_607_360,
    "dsp": 9_024,
    # DERIVED, not quoted: an UltraScale+ CLB holds eight LUT6 and one CARRY8
    # (UG574), so the part has one CARRY8 per eight LUTs.
    "carry8": 162_960,
    # DERIVED: only a SLICEM LUT holds a shift register or distributed RAM, and
    # about half an UltraScale+ device's slices are SLICEM (UG574).
    "slicem_lut": 651_840,
    "bram36": 2_016,
    # Tile counter, named for its size like `bram36`; the storage realization
    # an array binds to is `uram`, a distinct name in the same symbol table.
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

#: LUTs per bit of a `k`-source one-hot AND-OR select, measured for k = 2..40,
#: listed where the staircase steps. A LUT6 absorbs three (data, select) pairs
#: and ~2.5 more per further level, so the curve is LINEAR in k, not
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

#: Below this depth a delay chain stays in flip-flops; at or above it Vivado
#: extracts an SRL, even though the emitter resets every stage. Measured here
#: exactly.
SRL_MIN_DEPTH = 4

#: SLICEM sites per bit of an extracted chain: an SRL32E holds 32 stages, so the
#: staircase is `ceil(depth/32)` starting at the extraction threshold. Spelled
#: out per bit rather than as `Tiled(32)`, which reads the whole tuple.
SRL_SITES_PER_BIT = {1: 0, SRL_MIN_DEPTH: 1}
SRL_SITES_PER_BIT.update({32 * i + 1: i + 1 for i in range(1, 17)})

#: Fabric LUTs per bit of an array that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of `depth*width` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0


def declare_xcu55c_area(device: "Device") -> None:
    """Declare the measured area model on ``device``: the resources the part
    has, and what one instance of each thing it builds spends of them.

    Chaining delays and storage timing are the device's own and are read back
    rather than restated; this only adds the resource side of an existing row.
    """
    # Deferred: `device` imports this module, so importing back at load time
    # would cycle.
    from .device import CombKind, Const, Linear, Quadratic, Step, Table, Tiled

    lut = device.add_resource("lut", CAPACITY["lut"])
    ff = device.add_resource("ff", CAPACITY["ff"])
    dsp = device.add_resource("dsp", CAPACITY["dsp"])
    carry8 = device.add_resource("carry8", CAPACITY["carry8"])
    # A separate counter from `@lut`: a design can run out of SLICEM with
    # ordinary LUTs to spare, since a delay chain competes for SLICEM with a
    # distributed RAM, not with logic.
    slicem = device.add_resource("slicem_lut", CAPACITY["slicem_lut"])
    bram36 = device.add_resource("bram36", CAPACITY["bram36"])
    uram288 = device.add_resource("uram288", CAPACITY["uram288"])

    # --- native combinational operators, over (width) -----------------------
    # A bitwise operator and a multiplexer are one LUT6 per bit exactly.
    logic = {lut: Linear(1.0)}
    # An adder is that plus a carry chain, one CARRY8 per eight bits (a compare
    # packs two bits per stage, so its chain is half). `Tiled`, not linear,
    # because a carry chain is a ceiling: a 9-bit adder takes two CARRY8s.
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
    # Past the extraction threshold a chain becomes `width` SRL sites per 32
    # stages plus one LUT per bit of addressing/muxing plus a head and tail
    # stage; `Step` models that cliff. The flip-flop cost is a SUM of a per-bit
    # term and a per-stage term (`2*width + depth - 1`), split in two because a
    # term proportional to depth cannot also be gated on depth.
    per_stage = [
        (Linear(1.0, base=-1.0), Const(1.0)),
        (
            Table(
                {d: float(1 - d) for d in range(1, SRL_MIN_DEPTH)}
                | {SRL_MIN_DEPTH: 0.0}
            ),
            Const(1.0),
        ),
    ]
    # UNDER-count: a chain narrower than eight bits stays in flip-flops
    # whatever its depth, a cliff on the OTHER parameter that no sum of
    # per-parameter factors can express; not modelled here.
    device.set_chain_uses(
        {
            ff: [(Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0))] + per_stage,
            lut: (Step(SRL_MIN_DEPTH, 0.0, 1.0), Linear(1.0)),
            slicem: (Table(SRL_SITES_PER_BIT), Linear(1.0)),
        }
    )

    # --- storage realizations, over (depth, width) --------------------------
    # A register file (an array that failed RAM inference) costs the whole
    # array in flip-flops and twice it in LUTs (data mux + write decode).
    # Measured at 512x32: one BRAM18 against 33,245 LUTs, 16,416 flip-flops.
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
