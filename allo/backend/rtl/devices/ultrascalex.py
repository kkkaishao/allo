# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UltraScale+ fabric: what every UltraScale+ die builds, and what each
structure spends of it."""

from __future__ import annotations

from collections.abc import Mapping

from ....lang.ip import OperatorIP
from ..device import (
    CombKind,
    Const,
    Device,
    Linear,
    Quadratic,
    Resource,
    Step,
    Table,
    Tiled,
)
from . import ip
from .spec import (
    Derived,
    FabricTiming,
    Grade,
    IPRow,
    Part,
    StorageSpec,
    StorageTiming,
)

NAME = "ultrascalex"

#: Resources a part does not quote, each derived from one it does. An
#: UltraScale+ CLB holds eight LUT6 and one CARRY8 (UG574), so the die has one
#: CARRY8 per eight LUTs; and only a SLICEM LUT holds a shift register or a
#: distributed RAM, which is about half a device's slices.
DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
}

GRADE_2L = Grade("-2L", default_freq_mhz=300.0)

#: One entry per grade the fabric has been characterized at. A part binned at a
#: grade with no entry does not build: borrowing a neighbouring grade's numbers
#: is how a made-up delay reaches the scheduler.
TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2L: FabricTiming(
        # Integer arithmetic, mul/div/rem included, is combinational; float and
        # the float casts go through the operator cores below.
        comb={
            CombKind.ADD: 1.2,
            CombKind.SUB: 1.2,
            CombKind.MUL: 2.0,
            CombKind.DIV: 2.5,
            CombKind.REM: 2.5,
            CombKind.NEG: 1.0,
            CombKind.CMP: 1.0,
            CombKind.AND: 0.4,
            CombKind.OR: 0.4,
            CombKind.XOR: 0.4,
            CombKind.SHL: 0.5,
            CombKind.SHR: 0.5,
            CombKind.SELECT: 0.5,
            CombKind.INT_CAST: 0.3,
        },
        storage={
            "register": StorageTiming(0, 1, 0.1, 0.1),
            "lutram": StorageTiming(1, 1, 0.5, 0.5),
            "bram": StorageTiming(1, 1, 0.7, 0.7),
            "uram": StorageTiming(2, 1, 0.9, 0.9),
            "srl": StorageTiming(1, 1, 0.5, 0.5),
        },
        stream=StorageTiming(1, 1, 0.5, 0.5),
    ),
}

DEFAULT_STORAGE = "lutram"
#: The row that is not a memory: one cell per element, no address, no port
#: limit, which is what a completely partitioned array becomes.
SCATTER_STORAGE = "register"

#: Below this depth a delay chain stays in flip-flops; at or above it Vivado
#: extracts an SRL, even though the emitter resets every stage. Measured.
SRL_MIN_DEPTH = 4

#: SLICEM sites per bit of an extracted chain: an SRL32E holds 32 stages, so the
#: staircase is ``ceil(depth/32)`` from the extraction threshold on.
SRL_SITES_PER_BIT = {1: 0, SRL_MIN_DEPTH: 1}
SRL_SITES_PER_BIT.update({32 * i + 1: i + 1 for i in range(1, 17)})

#: LUTs per bit of a `k`-source one-hot AND-OR select, measured for k = 2..40 and
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

#: Fabric LUTs per bit of an array that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of ``depth*width`` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0

#: What one instance of each operator core takes and spends here. Each is the
#: Xilinx Floating-Point core at the latency named, so the area is a constant:
#: a core's signature fixes its widths. bf16 has no measured core and is priced
#: from its f32 sibling by width.
#:
#: The latency here is what names the core: `ip.fadd` at 7 injects as
#: `add_f32_f32_f32_l7`, so retuning a row renames the extern module with it.
IP: Mapping[OperatorIP, IPRow] = {
    ip.fadd: IPRow(7, {"lut": 247, "ff": 315, "dsp": 2, "carry8": 10}),
    ip.fsub: IPRow(7, {"lut": 247, "ff": 315, "dsp": 2, "carry8": 10}),
    ip.fmul: IPRow(4, {"lut": 115, "ff": 173, "dsp": 2, "carry8": 9}),
    ip.fdiv: IPRow(12, {"lut": 766, "ff": 1381, "carry8": 111}),
    ip.fcmp: IPRow(1, {"lut": 64, "ff": 12, "carry8": 7}),
    ip.dadd: IPRow(14, {"lut": 710, "ff": 872, "dsp": 3, "carry8": 30}),
    ip.dsub: IPRow(14, {"lut": 710, "ff": 872, "dsp": 3, "carry8": 30}),
    ip.dmul: IPRow(9, {"lut": 205, "ff": 542, "dsp": 7, "carry8": 16}),
    ip.ddiv: IPRow(24, {"lut": 3185, "ff": 6035, "carry8": 399}),
    ip.dcmp: IPRow(1, {"lut": 118, "ff": 12, "carry8": 12}),
    ip.bfadd: IPRow(4, {"lut": 124, "ff": 158, "dsp": 1, "carry8": 5}),
    ip.bfsub: IPRow(4, {"lut": 124, "ff": 158, "dsp": 1, "carry8": 5}),
    ip.bfmul: IPRow(2, {"lut": 58, "ff": 87, "dsp": 1, "carry8": 5}),
    ip.i2f: IPRow(3, {"lut": 165, "ff": 228, "carry8": 11}),
    ip.f2i: IPRow(3, {"lut": 183, "ff": 232, "carry8": 6}),
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry8": 1}),
    ip.bf2f: IPRow(2, {"lut": 25, "ff": 50, "carry8": 1}),
}


#: Per storage realization: the resources it needs to exist at all, and what one
#: instance spends over ``(depth, width)``. A row whose resources the die does
#: not have is not declared, which is how a part with no UltraRAM says so.
#:
#: A register file (an array that failed RAM inference) costs the whole array in
#: flip-flops and twice it in LUTs, a data multiplexer plus a write decode;
#: measured at 512x32 as one BRAM18 against 33,245 LUTs and 16,416 flip-flops.
#: The rest are tiled: a structure holds so many bits however the array is cut,
#: with distributed RAM and shift registers in SLICEM, block RAM and UltraRAM in
#: their own columns.
_STORAGE = {
    "register": StorageSpec(
        ("lut", "ff"),
        lambda r: {
            r["lut"]: (Linear(MULTIWRITE_LUT_PER_BIT), Linear(1.0)),
            r["ff"]: (Linear(1.0), Linear(1.0)),
        },
    ),
    "lutram": StorageSpec(("slicem_lut",), lambda r: {r["slicem_lut"]: Tiled(64)}),
    "srl": StorageSpec(("slicem_lut",), lambda r: {r["slicem_lut"]: Tiled(32)}),
    "bram": StorageSpec(("bram36",), lambda r: {r["bram36"]: Tiled(36864)}),
    "uram": StorageSpec(("uram288",), lambda r: {r["uram288"]: Tiled(294912)}),
}


def _comb_uses(r: Mapping[str, Resource]) -> dict[CombKind, dict | None]:
    """What one instance of each native operator kind spends, over its operand
    width. ``None`` is FREE and not unpriced: ``icast`` is a rename of bits and
    ``neg`` a float sign flip, so neither reaches a cell the part charges for."""
    lut, dsp, carry8 = r["lut"], r["dsp"], r["carry8"]
    # A bitwise operator and a multiplexer are one LUT6 per bit exactly. An
    # adder is that plus a carry chain, one CARRY8 per eight bits (a compare
    # packs two bits per stage, so its chain is half). `Tiled`, not linear,
    # because a carry chain is a ceiling: a 9-bit adder takes two CARRY8s.
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    # A barrel shift is w*ceil(log4 w) LUTs, which is structural but has one
    # user, so it stays the four points it was measured at.
    shift = {lut: Table({8: 16, 16: 32, 32: 96, 64: 192})}
    # A multiplier is DSP48E2s plus glue: one DSP holds up to 18 bits, above
    # which the partial products multiply up and the LUT glue shrinks. Its carry
    # chain is the compare's, over the partial-product adds.
    multiply = {
        lut: Table({8: 39, 16: 39, 32: 15, 64: 15}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10}),
        carry8: Tiled(16),
    }
    # Measured 75/286/1086 LUTs at w=8/16/32: quadratic is the structure of the
    # restoring divider, 1.06 is the measurement.
    divide = {lut: Quadratic(1.06), carry8: Linear(5.0)}
    return {
        CombKind.AND: logic,
        CombKind.OR: logic,
        CombKind.XOR: logic,
        CombKind.SELECT: logic,
        CombKind.ADD: addsub,
        CombKind.SUB: addsub,
        CombKind.CMP: compare,
        CombKind.SHL: shift,
        CombKind.SHR: shift,
        CombKind.MUL: multiply,
        CombKind.DIV: divide,
        CombKind.REM: divide,
        CombKind.NEG: None,
        CombKind.INT_CAST: None,
    }


def _chain_uses(r: Mapping[str, Resource]) -> dict:
    """What one ``depth``-stage, ``width``-bit value delay chain spends.

    Past the extraction threshold a chain becomes ``width`` SRL sites per 32
    stages plus one LUT per bit of addressing/muxing plus a head and tail stage;
    ``Step`` models that cliff. The flip-flop cost is a SUM of a per-bit term
    and a per-stage term (``2*width + depth - 1``), split in two because a term
    proportional to depth cannot also be gated on depth.

    UNDER-count: a chain narrower than eight bits stays in flip-flops whatever
    its depth, a cliff on the OTHER parameter that no sum of per-parameter
    factors can express; not modelled here.
    """
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
    return {
        r["ff"]: [(Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0))] + per_stage,
        r["lut"]: (Step(SRL_MIN_DEPTH, 0.0, 1.0), Linear(1.0)),
        r["slicem_lut"]: (Table(SRL_SITES_PER_BIT), Linear(1.0)),
    }


def build(part: Part) -> Device:
    """The :class:`Device` for one UltraScale+ die."""
    timing = TIMING.get(part.grade)
    if timing is None:
        raise ValueError(
            f"{NAME} has not been characterized at grade {part.grade.name!r}, so "
            f"{part.name!r} cannot be built; measure that grade and add it to "
            "TIMING rather than reading a neighbouring grade's delays"
        )
    d = Device(part.name, part=part.part, fabric=NAME, grade=part.grade.name)

    res = {name: d.add_resource(name, cap) for name, cap in part.capacity.items()}
    for name, derived in DERIVED.items():
        if derived.source in part.capacity:
            capacity = part.capacity[derived.source] // derived.divisor
            res[name] = d.add_resource(name, capacity)

    for name, t in timing.storage.items():
        needs, spend = _STORAGE[name]
        if not all(n in res for n in needs):
            continue
        d.add_storage(
            name,
            read_latency=t.read_latency,
            write_latency=t.write_latency,
            read_delay_ns=t.read_ns,
            write_delay_ns=t.write_ns,
            is_scatter=name == SCATTER_STORAGE,
            uses=spend(res),
        )
    d.set_default_storage(d.storage[DEFAULT_STORAGE])
    d.set_stream_timing(*timing.stream)

    comb = _comb_uses(res)
    for kind, delay in timing.comb.items():
        d.set_comb_delay(kind, delay, uses=comb[kind])

    for core, row in IP.items():
        operator = core.retimed(row.latency)
        d.add_operator(operator)
        d.set_operator_uses(operator, {res[n]: Const(v) for n, v in row.area.items()})

    d.set_mux_uses({res["lut"]: (Table(MUX_LUT_PER_BIT), Linear(1.0))})
    d.set_chain_uses(_chain_uses(res))
    d.set_default_frequency(part.grade.default_freq_mhz)
    return d.validate()
