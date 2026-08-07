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
    Linear,
    Resource,
    Step,
    Table,
    Tiled,
)
from . import ip
from .spec import (
    MULTIWRITE_LUT_PER_BIT,
    MUX_LUT_PER_BIT,
    SRL_MIN_DEPTH,
    SRL_SITES_PER_BIT,
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
            CombKind.ADD: Table({8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057}),
            CombKind.SUB: Table({8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057}),
            CombKind.MUL: Table({8: 1.653, 16: 2.353, 32: 3.241, 64: 4.970}),
            CombKind.DIV: Table({8: 5.106, 16: 10.788, 32: 24.732, 64: 59.4}),
            CombKind.REM: Table({8: 5.409, 16: 11.212, 32: 25.144, 64: 60.3}),
            CombKind.NEG: Table({32: 0.400, 64: 0.419}),
            CombKind.MIN: Table({8: 0.980, 16: 1.113, 32: 1.498, 64: 1.527}),
            CombKind.MAX: Table({8: 0.947, 16: 1.332, 32: 1.562, 64: 1.562}),
            CombKind.CMP: Table({8: 0.656, 16: 0.717, 32: 0.791, 64: 0.873}),
            CombKind.AND: Table({8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495}),
            CombKind.OR: Table({8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495}),
            CombKind.XOR: Table({8: 0.446, 16: 0.446, 32: 0.495, 64: 0.495}),
            CombKind.SHL: Table({8: 0.727, 16: 1.004, 32: 1.537, 64: 1.915}),
            CombKind.SHR: Table({8: 1.463, 16: 1.463, 32: 1.467, 64: 1.936}),
            CombKind.SELECT: Table({8: 0.540, 16: 0.540, 32: 0.540, 64: 0.955}),
            CombKind.INT_CAST: Table({16: 0.413, 32: 0.433, 64: 0.744}),
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
            CombKind.ADD: Table({8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675}),
            CombKind.SUB: Table({8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675}),
            CombKind.MUL: Table({8: 2.032, 16: 2.933, 32: 4.158, 64: 6.524}),
            # 64 bits extrapolated as above.
            CombKind.DIV: Table({8: 6.598, 16: 15.439, 32: 36.402, 64: 87.4}),
            CombKind.REM: Table({8: 6.830, 16: 15.289, 32: 37.631, 64: 90.3}),
            CombKind.NEG: Table({32: 0.541, 64: 0.667}),
            CombKind.MIN: Table({8: 1.425, 16: 1.425, 32: 1.570, 64: 1.851}),
            CombKind.MAX: Table({8: 1.425, 16: 1.425, 32: 1.590, 64: 1.853}),
            CombKind.CMP: Table({8: 0.782, 16: 1.101, 32: 1.101, 64: 1.273}),
            CombKind.AND: Table({8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655}),
            CombKind.OR: Table({8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655}),
            CombKind.XOR: Table({8: 0.653, 16: 0.653, 32: 0.653, 64: 0.653}),
            CombKind.SHL: Table({8: 1.622, 16: 1.622, 32: 1.772, 64: 2.038}),
            CombKind.SHR: Table({8: 1.035, 16: 1.420, 32: 2.046, 64: 2.324}),
            CombKind.SELECT: Table({8: 0.685, 16: 0.685, 32: 0.718, 64: 0.948}),
            CombKind.INT_CAST: Table({16: 0.555, 32: 0.638, 64: 0.697}),
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


IP: Mapping[OperatorIP, IPRow] = {
    ip.fadd: IPRow(7, {"lut": 270, "ff": 238, "dsp": 2, "carry8": 10}),  # 418
    ip.fsub: IPRow(7, {"lut": 270, "ff": 238, "dsp": 2, "carry8": 10}),  # 418
    ip.fmul: IPRow(4, {"lut": 115, "ff": 109, "dsp": 2, "carry8": 9}),  # 571
    ip.fdiv: IPRow(12, {"lut": 810, "ff": 477, "carry8": 109}),  # 356
    ip.fcmp: IPRow(1, {"lut": 63, "ff": 2, "carry8": 7}),  # combinational
    ip.dadd: IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry8": 30}),  # 629
    ip.dsub: IPRow(14, {"lut": 811, "ff": 872, "dsp": 3, "carry8": 30}),  # 629
    ip.dmul: IPRow(9, {"lut": 262, "ff": 397, "dsp": 7, "carry8": 15}),  # 593
    ip.ddiv: IPRow(32, {"lut": 3267, "ff": 3027, "carry8": 398}),  # 397
    ip.dcmp: IPRow(1, {"lut": 117, "ff": 2, "carry8": 12}),  # combinational
    ip.bfadd: IPRow(4, {"lut": 199, "ff": 118, "carry8": 12}),  # 664
    ip.bfsub: IPRow(4, {"lut": 199, "ff": 118, "carry8": 12}),  # 664
    ip.bfmul: IPRow(2, {"lut": 60, "ff": 34, "dsp": 1, "carry8": 6}),  # 739
    ip.i2f: IPRow(3, {"lut": 169, "ff": 99, "carry8": 11}),  # 527
    ip.f2i: IPRow(3, {"lut": 183, "ff": 127, "carry8": 6}),  # 701
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry8": 1}),  # 1149
    ip.bf2f: IPRow(2, {"lut": 34, "ff": 53, "carry8": 1}),  # 1232
    ip.imul16: IPRow(3, {"dsp": 1}),  # 1073
    ip.imul32: IPRow(2, {"ff": 32, "dsp": 3}),  # 468
    ip.imul64: IPRow(6, {"lut": 64, "ff": 81, "dsp": 10}),  # 333
    ip.idiv8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 310
    ip.udiv8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 310
    ip.irem8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 310
    ip.urem8: IPRow(4, {"lut": 127, "ff": 166, "carry8": 18}),  # 310
    ip.idiv16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 344
    ip.udiv16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 344
    ip.irem16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 344
    ip.urem16: IPRow(8, {"lut": 378, "ff": 578, "carry8": 55}),  # 344
    ip.idiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 352
    ip.udiv32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 352
    ip.irem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 352
    ip.urem32: IPRow(16, {"lut": 1266, "ff": 2170, "carry8": 173}),  # 352
    ip.idiv64: IPRow(32, {"lut": 4578, "ff": 8426, "carry8": 601}),  # 302
    ip.udiv64: IPRow(32, {"lut": 4578, "ff": 8426, "carry8": 601}),  # 302
    ip.irem64: IPRow(32, {"lut": 4578, "ff": 8426, "carry8": 601}),  # 302
    ip.urem64: IPRow(32, {"lut": 4578, "ff": 8426, "carry8": 601}),  # 302
}


IP_BY_GRADE: Mapping[Grade, Mapping[OperatorIP, IPRow]] = {
    GRADE_2LV: {ip.fdiv: IPRow(16, {"lut": 804, "ff": 699, "carry8": 111})},
}


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
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    minmax = {lut: Linear(2.0), carry8: Tiled(16)}
    shift = {lut: Table({8: 15, 16: 44, 32: 107, 64: 265})}
    multiply = {
        lut: Table({8: 39, 16: 0, 32: 15, 64: 41}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10}),
        carry8: Tiled(16),
    }
    divide = {
        lut: Table({8: 125, 16: 377, 32: 1344, 64: 4731}),
        carry8: Table({8: 14, 16: 50, 32: 172, 64: 600}),
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

    for core, row in {**IP, **IP_BY_GRADE.get(part.grade, {})}.items():
        operator = core.retimed(row.latency)
        d.add_operator(operator)
        d.set_operator_uses(operator, {res[n]: Const(v) for n, v in row.area.items()})

    d.set_mux_uses({res["lut"]: (Table(MUX_LUT_PER_BIT), Linear(1.0))})
    d.set_chain_uses(_chain_uses(res))
    d.set_register_floor(timing.reg_ns)
    d.set_default_frequency(part.grade.default_freq_mhz)
    return d.validate()
