# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 7-series fabric: Artix/Kintex/Zynq-7000"""

from __future__ import annotations

from collections.abc import Mapping

from ....lang.ip import OperatorIP
from ..device import (
    CombKind,
    Const,
    Device,
    Interp,
    Linear,
    Piecewise,
    Resource,
    Step,
    Table,
    Tiled,
)
from . import ip
from .spec import (
    MULTIWRITE_LUT_PER_BIT,
    MUX_LUT_COST,
    ROM_ENTRIES_PER_LUT,
    SRL_MIN_DEPTH,
    Derived,
    FabricTiming,
    Grade,
    IPRow,
    Part,
    StorageSpec,
    StorageTiming,
)

NAME = "series7"

DERIVED = {
    "carry4": Derived("lut", 4),
    "slicem_lut": Derived("lut", 2),
}

GRADE_1 = Grade("-1", default_freq_mhz=100.0)


TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_1: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 2.325, 16: 2.480, 32: 2.936, 64: 3.848, 96: 4.836, 128: 5.650}
            ),
            CombKind.SUB: Interp(
                {8: 2.325, 16: 2.480, 32: 2.936, 64: 3.848, 96: 4.836, 128: 5.650}
            ),
            CombKind.MUL: Interp(
                {8: 5.083, 16: 5.618, 32: 9.297, 64: 13.319, 96: 15.762, 128: 18.207}
            ),
            CombKind.DIV: Interp({8: 17.031, 16: 38.884, 32: 96.829, 64: 232.4}),
            CombKind.REM: Interp({8: 18.471, 16: 39.683, 32: 98.933, 64: 237.4}),
            CombKind.NEG: Interp({32: 1.086, 64: 1.254}),
            CombKind.MIN: Interp(
                {8: 3.389, 16: 3.454, 32: 4.166, 64: 4.678, 96: 5.343, 128: 5.911}
            ),
            CombKind.MAX: Interp(
                {8: 3.459, 16: 3.641, 32: 4.030, 64: 5.007, 96: 5.379, 128: 5.907}
            ),
            CombKind.CMP: Interp(
                {8: 2.036, 16: 2.284, 32: 2.438, 64: 2.789, 96: 3.241, 128: 3.701}
            ),
            CombKind.AND: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.OR: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.XOR: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.SHL: Interp(
                {8: 2.445, 16: 3.376, 32: 4.336, 64: 5.132, 96: 6.076, 128: 7.002}
            ),
            CombKind.SHR: Interp(
                {8: 2.847, 16: 3.685, 32: 4.568, 64: 6.255, 96: 6.255, 128: 6.619}
            ),
            CombKind.SELECT: Interp(
                {8: 1.736, 16: 1.982, 32: 2.088, 64: 2.088, 96: 2.224, 128: 2.224}
            ),
            CombKind.INT_CAST: Interp(
                {16: 1.216, 32: 1.216, 64: 1.510, 96: 1.510, 128: 1.510}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 1.086, 1.086),
            "lutram": StorageTiming(1, 1, 2.815, 3.449),
            "bram": StorageTiming(1, 1, 3.427, 1.390),
            "srl": StorageTiming(1, 1, 2.815, 3.449),
        },
        stream=StorageTiming(1, 1, 2.815, 3.449),
        reg_ns=1.086,
    ),
}

DEFAULT_STORAGE = "lutram"
SCATTER_STORAGE = "register"

#: Operator cores measured on this fabric, each inside a registered wrapper so
#: the number covers the whole path a caller sees. The trailing comment on each
#: row is that core's achieved Fmax in MHz, a record of the characterization run
#: and not an input to the cost model. Several rows under one archetype declare
#: several cores, which the library then chooses between; every one of them
#: closes at the part's default frequency, since a core that misses it is not a
#: realization the library may pick.
IP: Mapping[OperatorIP, IPRow | tuple[IPRow, ...]] = {
    ip.fadd: (
        IPRow(7, {"lut": 265, "ff": 238, "dsp": 2, "carry4": 19}),  # 181
        IPRow(5, {"lut": 376, "ff": 242, "carry4": 36}),  # 157
    ),
    ip.fsub: (
        IPRow(7, {"lut": 265, "ff": 238, "dsp": 2, "carry4": 19}),  # 181
        IPRow(5, {"lut": 376, "ff": 242, "carry4": 36}),  # 157
    ),
    ip.fmul: IPRow(4, {"lut": 115, "ff": 109, "dsp": 2, "carry4": 14}),  # 214
    ip.fdiv: IPRow(12, {"lut": 799, "ff": 477, "carry4": 194}),  # 114
    ip.fcmp: IPRow(1, {"lut": 64, "ff": 2, "carry4": 12}),  # 194
    ip.dadd: (
        IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry4": 51}),  # 210
        IPRow(6, {"lut": 735, "ff": 542, "carry4": 72}),  # 169
    ),
    ip.dsub: (
        IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry4": 51}),  # 210
        IPRow(6, {"lut": 735, "ff": 542, "carry4": 72}),  # 169
    ),
    ip.dmul: IPRow(9, {"lut": 172, "ff": 429, "dsp": 10, "carry4": 27}),  # 213
    ip.ddiv: IPRow(32, {"lut": 3267, "ff": 3027, "carry4": 794}),  # 119
    ip.dcmp: IPRow(1, {"lut": 118, "ff": 2, "carry4": 21}),  # 182
    ip.bfadd: IPRow(4, {"lut": 176, "ff": 118, "carry4": 24}),  # 175
    ip.bfsub: IPRow(4, {"lut": 176, "ff": 118, "carry4": 24}),  # 175
    ip.bfmul: IPRow(2, {"lut": 60, "ff": 34, "dsp": 1, "carry4": 9}),  # 185
    ip.i2f: IPRow(3, {"lut": 169, "ff": 99, "carry4": 20}),  # 163
    ip.f2i: IPRow(3, {"lut": 183, "ff": 127, "carry4": 11}),  # 186
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry4": 1}),  # 321
    ip.bf2f: IPRow(2, {"lut": 34, "ff": 53, "carry4": 1}),  # 363
    ip.imul16: (
        IPRow(4, {"ff": 16, "dsp": 1}),  # 514
        IPRow(1, {"dsp": 1}),  # 188
    ),
    ip.imul32: (
        IPRow(2, {"ff": 32, "dsp": 3}),  # 119
        IPRow(1, {"ff": 32, "dsp": 3}),  # 106
    ),
    ip.imul64: IPRow(6, {"lut": 64, "ff": 81, "dsp": 10}),  # 135
    ip.idiv8: IPRow(4, {"lut": 127, "ff": 166, "carry4": 27}),  # 102
    ip.udiv8: (
        IPRow(4, {"lut": 127, "ff": 166, "carry4": 27}),  # 102
        IPRow(2, {"lut": 110, "ff": 132, "carry4": 27}),  # 104
    ),
    ip.irem8: IPRow(4, {"lut": 127, "ff": 166, "carry4": 27}),  # 102
    ip.urem8: (
        IPRow(4, {"lut": 127, "ff": 166, "carry4": 27}),  # 102
        IPRow(2, {"lut": 110, "ff": 132, "carry4": 27}),  # 104
    ),
    ip.idiv16: IPRow(8, {"lut": 378, "ff": 578, "carry4": 93}),  # 103
    ip.udiv16: IPRow(8, {"lut": 378, "ff": 578, "carry4": 93}),  # 103
    ip.irem16: IPRow(8, {"lut": 378, "ff": 578, "carry4": 93}),  # 103
    ip.urem16: IPRow(8, {"lut": 378, "ff": 578, "carry4": 93}),  # 103
    ip.idiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry4": 313}),  # 102
    ip.udiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry4": 313}),  # 102
    ip.irem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry4": 313}),  # 102
    ip.urem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry4": 313}),  # 102
    ip.idiv64: IPRow(68, {"lut": 4614, "ff": 12808, "carry4": 1137}),  # 174
    ip.udiv64: (
        IPRow(68, {"lut": 4614, "ff": 12808, "carry4": 1137}),  # 174
        IPRow(66, {"lut": 4483, "ff": 12677, "carry4": 1105}),  # 186
    ),
    ip.irem64: IPRow(68, {"lut": 4614, "ff": 12808, "carry4": 1137}),  # 174
    ip.urem64: (
        IPRow(68, {"lut": 4614, "ff": 12808, "carry4": 1137}),  # 174
        IPRow(66, {"lut": 4483, "ff": 12677, "carry4": 1105}),  # 186
    ),
}


_STORAGE = {
    "register": StorageSpec(
        ("lut", "ff"),
        lambda r: {
            r["lut"]: (Linear(MULTIWRITE_LUT_PER_BIT), Linear(1.0)),
            r["ff"]: (Linear(1.0), Linear(1.0)),
        },
    ),
    # Distributed RAM has one write port and ONE addressed read, the two being
    # separate structures (no pool). A second read address is a whole further
    # copy of the array: measured 640 / 1280 / 1920 / 2560 LUT as memory at
    # 1024x32 for one through four reads. `max_reads` is then how many reads an
    # array here may be given at once, which is two copies' worth.
    "lutram": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: Tiled(64)},
        max_reads=2,
        max_writes=1,
        inst_reads=1,
        ram_style="distributed",
    ),
    # A shift register writes only at its head and reads one addressed tap.
    "srl": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: Tiled(32)},
        max_reads=1,
        max_writes=1,
    ),
    # Two ports, each reading OR writing in a cycle: hence the pool, which two
    # writers and a concurrent reader together exceed.
    "bram": StorageSpec(
        ("bram36",),
        lambda r: {r["bram36"]: Tiled(36864)},
        max_reads=2,
        max_writes=2,
        max_ports=2,
        ram_style="block",
    ),
    "uram": StorageSpec(
        ("uram288",),
        lambda r: {r["uram288"]: Tiled(294912)},
        max_reads=2,
        max_writes=2,
        max_ports=2,
        ram_style="ultra",
        can_init=False,
    ),
}


def _comb_uses(r: Mapping[str, Resource]) -> dict[CombKind, dict | None]:
    """What one instance of each native operator kind spends, over its operand
    width. ``None`` means free rather than unpriced: ``icast`` renames bits and
    ``neg`` flips a float sign, so neither reaches a cell the part charges for."""
    lut, dsp, carry4 = r["lut"], r["dsp"], r["carry4"]
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry4: Tiled(4)}
    compare = {lut: Linear(1.0), carry4: Tiled(8)}
    minmax = {lut: Linear(2.0), carry4: Tiled(8)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count is a whole number of slices and steps; the fabric logic
    # around them grows with the width. Past 64 bits the partial-product tree
    # takes more carry chains than `Tiled` charges, measured 57 against 16 at
    # 128 bits.
    multiply = {
        lut: Interp({8: 39, 16: 0, 32: 15, 64: 41, 96: 153, 128: 316}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10, 96: 21, 128: 36}),
        carry4: Tiled(8),
    }
    divide = {
        lut: Interp({8: 125, 16: 397, 32: 1344, 64: 4731, 96: 10188, 128: 17575}),
        carry4: Interp({8: 21, 16: 92, 32: 312, 64: 1136, 96: 2472, 128: 4320}),
    }
    return {
        CombKind.AND: logic,
        CombKind.OR: logic,
        CombKind.XOR: logic,
        CombKind.SELECT: logic,
        CombKind.ADD: addsub,
        CombKind.SUB: addsub,
        CombKind.CMP: compare,
        CombKind.MIN: minmax,
        CombKind.MAX: minmax,
        CombKind.SHL: shift,
        CombKind.SHR: shift,
        CombKind.MUL: multiply,
        CombKind.DIV: divide,
        CombKind.REM: divide,
        CombKind.NEG: None,
        CombKind.INT_CAST: None,
    }


def _chain_uses(r: Mapping[str, Resource]) -> dict:
    """What one delay chain spends, over its depth and bit width."""
    per_stage = [
        (Linear(1.0, base=-1.0), Const(1.0)),
        (Piecewise(SRL_MIN_DEPTH, Linear(-1.0, base=1.0), Const(0.0)), Const(1.0)),
    ]
    return {
        r["ff"]: [(Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0))] + per_stage,
        r["lut"]: (Step(SRL_MIN_DEPTH, 0.0, 1.0), Linear(1.0)),
        # An SRL32E holds 32 stages, so an extracted chain takes ceil(depth/32)
        # sites a bit and a shallower one takes none.
        r["slicem_lut"]: (
            Piecewise(SRL_MIN_DEPTH, Const(0.0), Tiled(32)),
            Linear(1.0),
        ),
    }


def build(part: Part) -> Device:
    """The :class:`Device` for one 7-series die."""
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
        spec = _STORAGE[name]
        if not all(n in res for n in spec.needs):
            continue
        d.add_storage(
            name,
            read_latency=t.read_latency,
            write_latency=t.write_latency,
            read_delay_ns=t.read_ns,
            write_delay_ns=t.write_ns,
            is_scatter=name == SCATTER_STORAGE,
            max_reads=spec.max_reads,
            max_writes=spec.max_writes,
            max_ports=spec.max_ports,
            inst_reads=spec.inst_reads,
            ram_style=spec.ram_style,
            can_init=spec.can_init,
            uses=spec.uses(res),
        )
    d.set_default_storage(d.storage[DEFAULT_STORAGE])
    d.set_stream_timing(*timing.stream)

    comb = _comb_uses(res)
    for kind, delay in timing.comb.items():
        d.set_comb_delay(kind, delay, uses=comb[kind])

    for core, rows in IP.items():
        for row in (rows,) if isinstance(rows, IPRow) else rows:
            operator = core.retimed(row.latency)
            if row.mnemonic is not None:
                operator.mnemonic = row.mnemonic
            d.add_operator(operator)
            d.set_operator_uses(
                operator, {res[n]: Const(v) for n, v in row.area.items()}
            )

    d.set_mux_uses({res["lut"]: (MUX_LUT_COST, Linear(1.0))})
    d.set_chain_uses(_chain_uses(res))
    # A constant table is logic, not storage: one LUT is a 64-entry lookup.
    d.set_rom_uses({res["lut"]: (Tiled(ROM_ENTRIES_PER_LUT), Linear(1.0))})
    d.set_register_floor(timing.reg_ns)
    d.set_default_frequency(part.grade.default_freq_mhz)
    return d.validate()


pynqz2 = build(
    Part(
        name="pynqz2",
        part="xc7z020clg400-1",
        grade=GRADE_1,
        capacity={
            "lut": 53_200,
            "ff": 106_400,
            "dsp": 220,
            "bram36": 140,
        },
    )
)

DEVICES = (pynqz2,)
