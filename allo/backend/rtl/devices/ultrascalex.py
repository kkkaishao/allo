# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UltraScale+ fabric."""

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

NAME = "ultrascalex"

DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
}

GRADE_2L = Grade("-2L", default_freq_mhz=300.0)
GRADE_2LV = Grade("-2LV", default_freq_mhz=300.0)

TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2L: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057, 96: 1.218, 128: 1.669}
            ),
            CombKind.SUB: Interp(
                {8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057, 96: 1.218, 128: 1.669}
            ),
            CombKind.MUL: Interp(
                {8: 1.653, 16: 2.353, 32: 3.241, 64: 4.970, 96: 5.759, 128: 6.564}
            ),
            CombKind.DIV: Interp({8: 5.106, 16: 10.788, 32: 24.732, 64: 59.4}),
            CombKind.REM: Interp({8: 5.409, 16: 11.212, 32: 25.144, 64: 60.3}),
            CombKind.NEG: Interp({32: 0.400, 64: 0.419}),
            CombKind.MIN: Interp(
                {8: 0.980, 16: 1.113, 32: 1.498, 64: 1.527, 96: 1.546, 128: 1.844}
            ),
            CombKind.MAX: Interp(
                {8: 0.947, 16: 1.332, 32: 1.562, 64: 1.562, 96: 1.689, 128: 1.725}
            ),
            CombKind.CMP: Interp(
                {8: 0.656, 16: 0.717, 32: 0.791, 64: 0.873, 96: 0.995, 128: 1.329}
            ),
            CombKind.AND: Interp(
                {8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.OR: Interp(
                {8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.XOR: Interp(
                {8: 0.446, 16: 0.446, 32: 0.495, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.SHL: Interp(
                {8: 0.727, 16: 1.004, 32: 1.537, 64: 1.915, 96: 2.096, 128: 2.239}
            ),
            CombKind.SHR: Interp(
                {8: 1.463, 16: 1.463, 32: 1.467, 64: 1.936, 96: 2.283, 128: 2.283}
            ),
            CombKind.SELECT: Interp(
                {8: 0.540, 16: 0.540, 32: 0.540, 64: 0.955, 96: 0.955, 128: 0.955}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.413, 32: 0.433, 64: 0.744, 96: 0.744, 128: 0.744}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.419, 0.419),
            "lutram": StorageTiming(1, 1, 1.574, 1.718),
            "bram": StorageTiming(1, 1, 1.345, 0.510),
            "uram": StorageTiming(2, 1, 1.379, 0.444),
            "srl": StorageTiming(1, 1, 1.574, 1.718),
        },
        stream=StorageTiming(1, 1, 1.574, 1.718),
        reg_ns=0.419,
    ),
    GRADE_2LV: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675, 96: 1.974, 128: 2.307}
            ),
            CombKind.SUB: Interp(
                {8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675, 96: 1.974, 128: 2.307}
            ),
            CombKind.MUL: Interp(
                {8: 2.032, 16: 2.933, 32: 4.158, 64: 6.524, 96: 7.716, 128: 8.892}
            ),
            # The 64-bit DIV and REM entries are extrapolated, not measured.
            CombKind.DIV: Interp({8: 6.598, 16: 15.439, 32: 36.402, 64: 87.4}),
            CombKind.REM: Interp({8: 6.830, 16: 15.289, 32: 37.631, 64: 90.3}),
            CombKind.NEG: Interp({32: 0.541, 64: 0.667}),
            CombKind.MIN: Interp(
                {8: 1.425, 16: 1.425, 32: 1.570, 64: 1.851, 96: 2.108, 128: 2.395}
            ),
            CombKind.MAX: Interp(
                {8: 1.425, 16: 1.425, 32: 1.590, 64: 1.853, 96: 2.238, 128: 2.683}
            ),
            CombKind.CMP: Interp(
                {8: 0.782, 16: 1.101, 32: 1.101, 64: 1.273, 96: 1.474, 128: 1.637}
            ),
            CombKind.AND: Interp(
                {8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655, 96: 0.681, 128: 0.681}
            ),
            CombKind.OR: Interp(
                {8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655, 96: 0.681, 128: 0.681}
            ),
            CombKind.XOR: Interp(
                {8: 0.653, 16: 0.653, 32: 0.653, 64: 0.653, 96: 0.681, 128: 0.681}
            ),
            CombKind.SHL: Interp(
                {8: 1.622, 16: 1.622, 32: 1.772, 64: 2.038, 96: 2.368, 128: 2.857}
            ),
            CombKind.SHR: Interp(
                {8: 1.035, 16: 1.420, 32: 2.046, 64: 2.324, 96: 2.436, 128: 2.922}
            ),
            CombKind.SELECT: Interp(
                {8: 0.685, 16: 0.685, 32: 0.718, 64: 0.948, 96: 1.057, 128: 1.057}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.555, 32: 0.638, 64: 0.697, 96: 0.845, 128: 0.846}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.638, 0.638),
            "lutram": StorageTiming(1, 1, 1.311, 1.698),
            "bram": StorageTiming(1, 1, 1.871, 0.646),
            "uram": StorageTiming(2, 1, 2.391, 0.754),
            "srl": StorageTiming(1, 1, 1.311, 1.698),
        },
        stream=StorageTiming(1, 1, 1.311, 1.698),
        reg_ns=0.638,
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
        IPRow(7, {"lut": 270, "ff": 238, "dsp": 2, "carry8": 10}),  # 432
        IPRow(5, {"lut": 383, "ff": 242, "carry8": 17}),  # 439
    ),
    ip.fsub: (
        IPRow(7, {"lut": 270, "ff": 238, "dsp": 2, "carry8": 10}),  # 432
        IPRow(5, {"lut": 383, "ff": 242, "carry8": 17}),  # 439
    ),
    ip.fmul: IPRow(4, {"lut": 115, "ff": 109, "dsp": 2, "carry8": 9}),  # 570
    ip.fdiv: IPRow(12, {"lut": 810, "ff": 477, "carry8": 109}),  # 374
    ip.fcmp: IPRow(1, {"lut": 63, "ff": 2, "carry8": 7}),  # 610
    ip.dadd: (
        IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry8": 30}),  # 575
        IPRow(6, {"lut": 735, "ff": 542, "carry8": 40}),  # 519
    ),
    ip.dsub: (
        IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry8": 30}),  # 575
        IPRow(6, {"lut": 735, "ff": 542, "carry8": 40}),  # 519
    ),
    ip.dmul: IPRow(9, {"lut": 262, "ff": 397, "dsp": 7, "carry8": 15}),  # 498
    ip.ddiv: IPRow(32, {"lut": 3267, "ff": 3027, "carry8": 398}),  # 398
    ip.dcmp: IPRow(1, {"lut": 117, "ff": 2, "carry8": 12}),  # 564
    ip.bfadd: IPRow(4, {"lut": 199, "ff": 118, "carry8": 12}),  # 537
    ip.bfsub: IPRow(4, {"lut": 199, "ff": 118, "carry8": 12}),  # 537
    ip.bfmul: IPRow(2, {"lut": 60, "ff": 34, "dsp": 1, "carry8": 6}),  # 521
    ip.i2f: IPRow(3, {"lut": 169, "ff": 99, "carry8": 11}),  # 490
    ip.f2i: IPRow(3, {"lut": 183, "ff": 127, "carry8": 6}),  # 678
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry8": 1}),  # 1032
    ip.bf2f: IPRow(2, {"lut": 34, "ff": 53, "carry8": 1}),  # 1181
    ip.imul16: (
        IPRow(3, {"dsp": 1}),  # 1073
        IPRow(1, {"dsp": 1}),  # 544
    ),
    ip.imul32: (
        IPRow(2, {"ff": 32, "dsp": 3}),  # 341
        IPRow(1, {"ff": 32, "dsp": 3}),  # 320
    ),
    ip.imul64: IPRow(6, {"lut": 64, "ff": 81, "dsp": 10}),  # 333
    ip.idiv8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
    ip.udiv8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
    ip.irem8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
    ip.urem8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
    ip.idiv16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 319
    ip.udiv16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 319
    ip.irem16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 319
    ip.urem16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 319
    ip.idiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 345
    ip.udiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 345
    ip.irem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 345
    ip.urem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 345
    ip.idiv64: IPRow(68, {"lut": 4614, "ff": 12808, "carry8": 601}),  # 579
    ip.udiv64: IPRow(32, {"lut": 4482, "ff": 8422, "carry8": 585}),  # 305
    ip.irem64: IPRow(68, {"lut": 4614, "ff": 12808, "carry8": 601}),  # 579
    ip.urem64: IPRow(32, {"lut": 4482, "ff": 8422, "carry8": 585}),  # 305
}


#: Rows that replace the base entry for their archetype at one grade, with a
#: tuple on either side standing for the whole candidate set.
IP_BY_GRADE: Mapping[Grade, Mapping[OperatorIP, IPRow | tuple[IPRow, ...]]] = {
    # Two shallower cores close on -2L and miss the same 300 MHz clock on -2LV,
    # so each is a candidate at the faster grade alone: the 2-cycle unsigned
    # 8-bit divider (311 MHz against 249) and the 24-cycle double divider (308
    # against 208).
    GRADE_2L: {
        ip.udiv8: (
            IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
            IPRow(2, {"lut": 110, "ff": 132, "carry8": 18}),  # 311
        ),
        ip.urem8: (
            IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 311
            IPRow(2, {"lut": 110, "ff": 132, "carry8": 18}),  # 311
        ),
        ip.ddiv: (
            IPRow(32, {"lut": 3267, "ff": 3027, "carry8": 398}),  # 398
            IPRow(24, {"lut": 3270, "ff": 2064, "carry8": 398}),  # 308
        ),
    },
    # The low-voltage grade closes nothing the -2L table declares for integer
    # division, and needs a deeper multiply besides, so most of its integer
    # arithmetic is a row of its own.
    GRADE_2LV: {
        ip.fdiv: IPRow(16, {"lut": 804, "ff": 699, "carry8": 111}),  # 361
        ip.imul32: IPRow(3, {"ff": 32, "dsp": 3}),  # 341
        ip.imul64: IPRow(8, {"lut": 113, "ff": 160, "dsp": 10}),  # 325
        ip.idiv8: IPRow(12, {"lut": 132, "ff": 264, "carry8": 18}),  # 804
        ip.irem8: IPRow(12, {"lut": 132, "ff": 264, "carry8": 18}),  # 804
        ip.udiv8: IPRow(4, {"lut": 114, "ff": 162, "carry8": 18}),  # 302
        ip.urem8: IPRow(4, {"lut": 114, "ff": 162, "carry8": 18}),  # 302
        ip.idiv16: IPRow(20, {"lut": 386, "ff": 904, "carry8": 55}),  # 745
        ip.irem16: IPRow(20, {"lut": 386, "ff": 904, "carry8": 55}),  # 745
        ip.udiv16: IPRow(8, {"lut": 354, "ff": 574, "carry8": 51}),  # 324
        ip.urem16: IPRow(8, {"lut": 354, "ff": 574, "carry8": 51}),  # 324
        ip.idiv32: IPRow(36, {"lut": 1284, "ff": 3336, "carry8": 173}),  # 572
        ip.irem32: IPRow(36, {"lut": 1284, "ff": 3336, "carry8": 173}),  # 572
        ip.udiv32: IPRow(34, {"lut": 1218, "ff": 3269, "carry8": 165}),  # 626
        ip.urem32: IPRow(34, {"lut": 1218, "ff": 3269, "carry8": 165}),  # 626
        ip.idiv64: IPRow(68, {"lut": 4614, "ff": 12808, "carry8": 601}),  # 439
        ip.irem64: IPRow(68, {"lut": 4614, "ff": 12808, "carry8": 601}),  # 439
        ip.udiv64: IPRow(66, {"lut": 4483, "ff": 12677, "carry8": 585}),  # 454
        ip.urem64: IPRow(66, {"lut": 4483, "ff": 12677, "carry8": 585}),  # 454
    },
}


_STORAGE = {
    "register": StorageSpec(
        ("lut", "ff"),
        lambda r: {
            r["lut"]: (Linear(MULTIWRITE_LUT_PER_BIT), Linear(1.0)),
            r["ff"]: (Linear(1.0), Linear(1.0)),
        },
    ),
    # Distributed RAM has one write port and one addressed read; the
    # synthesizer serves further reads by replicating the whole array, so the
    # read limit caps how many copies an array is worth rather than the
    # structure, and the two directions are separate structures (no pool).
    "lutram": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: Tiled(64)},
        max_reads=2,
        max_writes=1,
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
    lut, dsp, carry8 = r["lut"], r["dsp"], r["carry8"]
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    minmax = {lut: Linear(2.0), carry8: Tiled(16)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count is a whole number of slices and steps; the fabric logic
    # around them grows with the width. Past 64 bits the partial-product tree
    # takes more carry chains than `Tiled` charges, measured 31 against 8 at
    # 128 bits.
    multiply = {
        lut: Interp({8: 39, 16: 0, 32: 15, 64: 41, 96: 153, 128: 316}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10, 96: 21, 128: 34}),
        carry8: Tiled(16),
    }
    divide = {
        lut: Interp({8: 125, 16: 377, 32: 1344, 64: 4731, 96: 10188, 128: 17575}),
        carry8: Interp({8: 14, 16: 50, 32: 172, 64: 600, 96: 1284, 128: 2224}),
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
            ram_style=spec.ram_style,
            can_init=spec.can_init,
            uses=spec.uses(res),
        )
    d.set_default_storage(d.storage[DEFAULT_STORAGE])
    d.set_stream_timing(*timing.stream)

    comb = _comb_uses(res)
    for kind, delay in timing.comb.items():
        d.set_comb_delay(kind, delay, uses=comb[kind])

    for core, rows in {**IP, **IP_BY_GRADE.get(part.grade, {})}.items():
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
