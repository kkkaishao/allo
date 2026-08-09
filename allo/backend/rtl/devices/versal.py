# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Versal fabric."""

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

NAME = "versal"

DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
}

GRADE_2MP = Grade("-2MP", default_freq_mhz=375.0)

TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2MP: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.860, 16: 0.933, 32: 1.069, 64: 1.179, 96: 1.364, 128: 1.561}
            ),
            CombKind.SUB: Interp(
                {8: 0.860, 16: 0.933, 32: 1.069, 64: 1.179, 96: 1.364, 128: 1.561}
            ),
            CombKind.MUL: Interp(
                {8: 1.439, 16: 2.320, 32: 3.324, 64: 4.079, 96: 4.932, 128: 5.203}
            ),
            CombKind.DIV: Interp({8: 6.018, 16: 13.603, 32: 28.187, 64: 59.2}),
            CombKind.REM: Interp({8: 6.651, 16: 14.861, 32: 30.021, 64: 63.0}),
            CombKind.NEG: Interp({32: 0.410, 64: 0.520}),
            CombKind.MIN: Interp(
                {8: 1.269, 16: 1.291, 32: 1.465, 64: 1.625, 96: 1.699, 128: 1.736}
            ),
            CombKind.MAX: Interp(
                {8: 1.244, 16: 1.441, 32: 1.441, 64: 1.583, 96: 1.733, 128: 1.792}
            ),
            CombKind.CMP: Interp(
                {8: 0.812, 16: 0.910, 32: 0.910, 64: 0.975, 96: 0.975, 128: 1.041}
            ),
            CombKind.AND: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.OR: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.XOR: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.SHL: Interp(
                {8: 1.014, 16: 1.133, 32: 1.441, 64: 1.628, 96: 1.960, 128: 2.100}
            ),
            CombKind.SHR: Interp(
                {8: 1.070, 16: 1.402, 32: 1.595, 64: 1.812, 96: 1.887, 128: 1.954}
            ),
            CombKind.SELECT: Interp(
                {8: 0.603, 16: 0.642, 32: 0.642, 64: 0.711, 96: 0.753, 128: 0.803}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.449, 32: 0.600, 64: 0.611, 96: 0.611, 128: 0.611}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.410, 0.410),
            "lutram": StorageTiming(1, 1, 1.268, 1.196),
            "bram": StorageTiming(1, 1, 1.299, 0.673),
            "uram": StorageTiming(2, 1, 1.057, 0.485),
            "srl": StorageTiming(1, 1, 1.268, 1.196),
        },
        stream=StorageTiming(1, 1, 1.268, 1.196),
        reg_ns=0.410,
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
    ip.fadd: IPRow(7, {"lut": 330, "ff": 238, "dsp": 2, "carry8": 10}),  # 436
    ip.fsub: IPRow(7, {"lut": 330, "ff": 238, "dsp": 2, "carry8": 10}),  # 436
    ip.fmul: IPRow(4, {"lut": 168, "ff": 109, "dsp": 2, "carry8": 9}),  # 502
    ip.fdiv: IPRow(16, {"lut": 1490, "ff": 699, "carry8": 111}),  # 452
    ip.fcmp: IPRow(1, {"lut": 105, "ff": 2, "carry8": 7}),  # 502
    ip.dadd: IPRow(14, {"lut": 954, "ff": 861, "dsp": 3, "carry8": 26}),  # 502
    ip.dsub: IPRow(14, {"lut": 954, "ff": 861, "dsp": 3, "carry8": 26}),  # 502
    ip.dmul: IPRow(9, {"lut": 360, "ff": 397, "dsp": 7, "carry8": 15}),  # 531
    ip.ddiv: IPRow(32, {"lut": 6288, "ff": 3027, "carry8": 396}),  # 388
    ip.dcmp: IPRow(1, {"lut": 197, "ff": 2, "carry8": 12}),  # 456
    ip.bfadd: IPRow(4, {"lut": 251, "ff": 118, "carry8": 12}),  # 440
    ip.bfsub: IPRow(4, {"lut": 251, "ff": 118, "carry8": 12}),  # 440
    ip.bfmul: IPRow(2, {"lut": 91, "ff": 34, "dsp": 1, "carry8": 6}),  # 482
    ip.i2f: IPRow(3, {"lut": 244, "ff": 99, "carry8": 11}),  # 478
    ip.f2i: IPRow(3, {"lut": 222, "ff": 127, "carry8": 6}),  # 612
    ip.fcvt: IPRow(2, {"lut": 54, "ff": 99, "carry8": 1}),  # 787
    ip.bf2f: IPRow(2, {"lut": 36, "ff": 53, "carry8": 1}),  # 792
    ip.imul16: (
        IPRow(3, {"dsp": 1}),  # 990
        IPRow(1, {"dsp": 1}),  # 470
    ),
    ip.imul32: IPRow(2, {"ff": 32, "dsp": 3}),  # 407
    ip.imul64: IPRow(3, {"lut": 23, "ff": 64, "dsp": 6}),  # 409
    ip.idiv8: IPRow(12, {"lut": 212, "ff": 264, "carry8": 18}),  # 737
    ip.udiv8: (
        IPRow(12, {"lut": 212, "ff": 264, "carry8": 18}),  # 737
        IPRow(10, {"lut": 195, "ff": 245, "carry8": 18}),  # 737
    ),
    ip.irem8: IPRow(12, {"lut": 212, "ff": 264, "carry8": 18}),  # 737
    ip.urem8: (
        IPRow(12, {"lut": 212, "ff": 264, "carry8": 18}),  # 737
        IPRow(10, {"lut": 195, "ff": 245, "carry8": 18}),  # 737
    ),
    ip.idiv16: IPRow(20, {"lut": 675, "ff": 904, "carry8": 55}),  # 672
    ip.udiv16: (
        IPRow(20, {"lut": 675, "ff": 904, "carry8": 55}),  # 672
        IPRow(8, {"lut": 643, "ff": 574, "carry8": 51}),  # 378
    ),
    ip.irem16: IPRow(20, {"lut": 675, "ff": 904, "carry8": 55}),  # 672
    ip.urem16: (
        IPRow(20, {"lut": 675, "ff": 904, "carry8": 55}),  # 672
        IPRow(8, {"lut": 643, "ff": 574, "carry8": 51}),  # 378
    ),
    ip.idiv32: IPRow(36, {"lut": 2372, "ff": 3336, "carry8": 173}),  # 650
    ip.udiv32: (
        IPRow(36, {"lut": 2372, "ff": 3336, "carry8": 173}),  # 650
        IPRow(16, {"lut": 2307, "ff": 2166, "carry8": 165}),  # 380
    ),
    ip.irem32: IPRow(36, {"lut": 2372, "ff": 3336, "carry8": 173}),  # 650
    ip.urem32: (
        IPRow(36, {"lut": 2372, "ff": 3336, "carry8": 173}),  # 650
        IPRow(16, {"lut": 2307, "ff": 2166, "carry8": 165}),  # 380
    ),
    ip.idiv64: IPRow(68, {"lut": 8838, "ff": 12808, "carry8": 601}),  # 528
    ip.udiv64: (
        IPRow(68, {"lut": 8838, "ff": 12808, "carry8": 601}),  # 528
        IPRow(66, {"lut": 8708, "ff": 12677, "carry8": 585}),  # 550
    ),
    ip.irem64: IPRow(68, {"lut": 8838, "ff": 12808, "carry8": 601}),  # 528
    ip.urem64: (
        IPRow(68, {"lut": 8838, "ff": 12808, "carry8": 601}),  # 528
        IPRow(66, {"lut": 8708, "ff": 12677, "carry8": 585}),  # 550
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
    # 1024x32 for one through four reads.
    "lutram": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: Tiled(64)},
        inst_reads=1,
        inst_writes=1,
        ram_style="distributed",
    ),
    # A shift register writes only at its head and reads one addressed tap.
    "srl": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: Tiled(32)},
        inst_reads=1,
        inst_writes=1,
    ),
    # Two ports, each reading OR writing in a cycle: hence the pool, which two
    # writers and a concurrent reader together exceed.
    "bram": StorageSpec(
        ("bram36",),
        lambda r: {r["bram36"]: Tiled(36864)},
        inst_reads=2,
        inst_writes=2,
        inst_ports=2,
        ram_style="block",
    ),
    "uram": StorageSpec(
        ("uram288",),
        lambda r: {r["uram288"]: Tiled(294912)},
        inst_reads=2,
        inst_writes=2,
        inst_ports=2,
        ram_style="ultra",
        can_init=False,
    ),
}


def _comb_uses(r: Mapping[str, Resource]) -> dict[CombKind, dict | None]:
    """What one instance of each native operator kind spends, over its operand
    width. ``None`` means free rather than unpriced: ``icast`` renames bits and
    ``neg`` flips a float sign, so neither reaches a cell the part charges for."""
    lut, dsp, carry8 = r["lut"], r["dsp"], r["carry8"]
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(0.5), carry8: Tiled(16)}
    minmax = {lut: Linear(1.5), carry8: Tiled(16)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count is a whole number of slices and steps; the fabric logic
    # around them grows with the width. Past 64 bits the partial-product tree
    # takes more carry chains than `Tiled` charges, measured 11 against 8 at
    # 128 bits.
    multiply = {
        lut: Interp({8: 36, 16: 0, 32: 10, 64: 19, 96: 82, 128: 255}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 6, 96: 15, 128: 21}),
        carry8: Tiled(16),
    }
    divide = {
        lut: Interp({8: 118, 16: 384, 32: 1301, 64: 4734, 96: 10189, 128: 17552}),
        carry8: Interp({8: 14, 16: 49, 32: 163, 64: 599, 96: 1283, 128: 2223}),
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
    """The :class:`Device` for one Versal die."""
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
            inst_reads=spec.inst_reads,
            inst_writes=spec.inst_writes,
            inst_ports=spec.inst_ports,
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


vck190 = build(
    Part(
        name="vck190",
        part="xcvc1902-vsva2197-2MP-e-S",
        grade=GRADE_2MP,
        capacity={
            "lut": 899_840,
            "ff": 1_799_680,
            "dsp": 1_968,
            "bram36": 967,
            "uram288": 463,
        },
    )
)

DEVICES = (vck190,)
