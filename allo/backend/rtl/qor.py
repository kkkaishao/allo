# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prices a compile's structures against a device: what the emitter built,
costed through the device's measured area tables."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .device import CombKind, Device
from .devices import default_device
from .reports.microarch import RegRole, TimingPath, Unit
from .reports.compile import CompileReport

#: The device kind that prices a native combinational unit, keyed by the
#: realization mnemonic its identity carries. The device characterizes "an
#: integer add", not ``addi`` against ``subi``, so several mnemonics share a row.
#: A row of ``None`` is FREE and not unpriced: a resize is a rename of bits and a
#: float negate a sign flip, so neither reaches a cell the part charges for.
COMB_KIND = {
    "addi": CombKind.ADD,
    "subi": CombKind.SUB,
    "muli": CombKind.MUL,
    "divsi": CombKind.DIV,
    "divui": CombKind.DIV,
    "remsi": CombKind.REM,
    "remui": CombKind.REM,
    "andi": CombKind.AND,
    "ori": CombKind.OR,
    "xori": CombKind.XOR,
    "shli": CombKind.SHL,
    "shrsi": CombKind.SHR,
    "shrui": CombKind.SHR,
    "cmpi": CombKind.CMP,
    "select": CombKind.SELECT,
    "extsi": None,
    "extui": None,
    "trunci": None,
    "index_cast": None,
    "negf": None,
}

#: An identity's operand list, and the integer widths in it. A unit's cost width
#: comes from there and not from the result the report carries: a compare returns
#: one bit, and pricing it at one bit prices it at nothing.
_ARGS = re.compile(r"^[^(]*\(([^)]*)\)")
_INT_WIDTH = re.compile(r"\bi(\d+)\b")


@dataclass(frozen=True)
class Utilization:
    """Primitive counts in the device's own vocabulary.

    Built from what :meth:`Device.price` returns, so a part that declares other
    resources adds other rows and nothing here switches on the list; the fields
    are the ``xcu55c`` names spelled out for a reader."""

    lut: int = 0  # every LUT site, state-holding ones included
    logic_lut: int = 0  # of those, the ones not holding state
    #: LUT sites holding state (`slicem_lut`): a shift register or a distributed
    #: RAM. Vivado reports the two uses apart ("LUT as Shift Register" against
    #: "LUT as Distributed RAM").
    srl: int = 0
    ff: int = 0
    dsp: int = 0
    carry8: int = 0
    bram36: int = 0
    uram288: int = 0

    @classmethod
    def of(cls, spent: dict[str, int]) -> "Utilization":
        """From one :meth:`Device.price` result, keyed by resource name."""
        logic, srl = spent.get("lut", 0), spent.get("slicem_lut", 0)
        return cls(
            lut=logic + srl,
            logic_lut=logic,
            srl=srl,
            ff=spent.get("ff", 0),
            dsp=spent.get("dsp", 0),
            carry8=spent.get("carry8", 0),
            bram36=spent.get("bram36", 0),
            uram288=spent.get("uram288", 0),
        )

    def __add__(self, other: "Utilization") -> "Utilization":
        return Utilization(*(getattr(self, f) + getattr(other, f) for f in _FIELDS))

    def __mul__(self, n: int) -> "Utilization":
        """This many instances of the same structure."""
        return Utilization(*(getattr(self, f) * n for f in _FIELDS))

    def fraction_of(self, capacity: dict[str, int]) -> dict[str, float]:
        """What fraction of each resource this occupies. Keys are the device's
        own resource names, so a resource it does not declare is absent."""
        counts = {
            "lut": self.lut,
            "slicem_lut": self.srl,
            "ff": self.ff,
            "dsp": self.dsp,
            "carry8": self.carry8,
            "bram36": self.bram36,
            "uram288": self.uram288,
        }
        return {k: counts[k] / cap for k, cap in capacity.items() if k in counts}


_FIELDS = tuple(Utilization.__dataclass_fields__)


@dataclass(frozen=True)
# One field per thing a reader may quote, so the count tracks the vocabulary
# rather than any coupling between them.
class QoR:  # pylint: disable=too-many-instance-attributes
    """What a compile is worth: how long it runs, and what it occupies.

    The throughput half comes from the schedule result and the area half from the
    microarchitecture report. Nothing here is a substitute for synthesis, which
    is the only authority on either; a `Synthesis` is a sibling of this and never
    a variant of it, and the two meet only in a calibration."""

    #: the top kernel's span. ``latency`` is the EXACT one and is present only
    #: where the kernel publishes a static contract; ``latency_max`` carries the
    #: BOUND a bounded kernel publishes instead and ``latency_min`` the FLOOR a
    #: concurrent one does, so a variant with no exact span still carries the
    #: figure it does have, marked as what it is.
    latency: int | None
    latency_max: int | None
    latency_min: int | None
    #: what each region's solve decided, keyed ``"<func>#<order>"``.
    interval: dict[str, int]
    #: the clock this design's longest accountable combinational path holds,
    #: below ``fmax_target`` where the structures grown after the schedule ate
    #: the room it left. Estimated off summed device rows, with no placement or
    #: routing in it, so never a substitute for the part's own timing report.
    fmax: float
    fmax_target: float  # MHz, the period the schedule was cut to
    #: the paths that clock comes from, longest first and across every module,
    #: each decomposed into the cells the signal passes through.
    critical_paths: tuple[TimingPath, ...]
    area: Utilization
    #: the fabric total split by what spends it: units / muxes / regs / memories
    #: / control. The axis an allocation change trades along, since a fold drops
    #: a unit and grows the muxes feeding the one it folded onto.
    by_kind: dict[str, Utilization]
    by_func: dict[str, Utilization]  # per emitted module
    #: resources whose figure is a COUNT rather than a model of one. A QoR that
    #: cannot tell the two apart gets quoted as a utilization figure, which the
    #: estimated half is not.
    counted: frozenset[str]
    #: structures with no cost row, by kind. Reported, never dropped: a silently
    #: unpriced structure reads as a cheaper design.
    unmodelled: dict[str, int]
    mem_bits: int  # stored bits, every instance counted, apart from the fabric totals
    #: flip-flops the emitted design DECLARES, from the register ledger, and so a
    #: count. ``area.ff`` is the modelled figure beside it: what the part holds
    #: once the deep chains are extracted into SRLs.
    reg_bits: int
    #: what fraction of each declared resource ``area`` occupies, keyed by the
    #: device's own resource names. A resource the part does not declare is
    #: absent rather than zero.
    utilization: dict[str, float]

    @property
    def over_capacity(self) -> dict[str, float]:
        """Resources the design asks for more of than the part has, keyed to the
        fraction asked for. A design with any is not placeable there."""
        return {k: v for k, v in self.utilization.items() if v > 1.0}

    @property
    def latency_is_exact(self) -> bool:
        """Whether ``latency`` is a span the hardware must realize, and so a
        number two runs may be differenced on."""
        return self.latency is not None

    def timing_report(self, limit: int = 3) -> str:
        """The worst paths as text: what each step costs, what it runs through,
        and where it ends."""
        period = 1000.0 / self.fmax_target
        head = (
            f"estimated {self.fmax:.1f} MHz against a {self.fmax_target:.1f} MHz "
            f"target ({period:.2f} ns period)"
        )
        return "\n".join(
            [head] + [p.describe("  ") for p in self.critical_paths[:limit]]
        )


def _operator_costs(device: Device) -> dict[str, tuple]:
    """Each priced operator's declared cost and the operand width it is a
    function of. The width is fixed by the IP's signature, which is why the
    declaration is a constant, but it is still the parameter its kind carries."""
    out = {}
    for op in device.operators:
        uses = device.operator_uses.get(op.symbol)
        if uses:
            widths = [a.primitive_width for a in op.parse_argument_annotations()]
            out[op.symbol] = (uses, max(widths))
    return out


def _scatter_row(device: Device):
    """The device's ``is_scatter`` row: one cell per element, selected rather
    than addressed. What a queue between two children is priced at."""
    row = next((s for s in device.storage.values() if s.is_scatter), None)
    if row is None:
        raise ValueError(
            f"device {device.name!r} marks no scatter storage, so there is "
            "nothing to price a structure held one cell per element"
        )
    return row.uses


def _unit_width(unit: Unit) -> int:
    """The width a unit's cost is a function of: the widest of its result and its
    operands, since a compare returns one bit and costs its operands."""
    args = _ARGS.match(unit.identity)
    widths = [int(w) for w in _INT_WIDTH.findall(args.group(1))] if args else []
    return max([unit.width, *widths])


# One pass over the report, one bucket per kind of structure it publishes.
# pylint: disable-next=too-many-locals
def estimate(report: CompileReport, device: Device = default_device) -> QoR:
    """Price ``report``'s structures against ``device``.

    Every module the emission built is priced separately: a design with
    sub-kernels spends its area in all of them. Validated against real
    synthesis of four bed kernels, LUT lands between 1.02x and 1.28x and
    flip-flops between 1.07x and 1.75x of measured, so this is fit to COMPARE
    two schedules and not to quote as a utilization figure. Structures with no
    cost row are reported in :attr:`QoR.unmodelled` rather than dropped; the
    emitter's control glue (run/issue/done logic, memory port muxing) is
    aggregated away rather than priced structure by structure.
    """
    price = device.price
    ip_costs = _operator_costs(device)
    scatter = _scatter_row(device)

    by_kind: dict[str, Utilization] = {}
    by_func: dict[str, Utilization] = {}
    unmodelled: Counter = Counter()
    mem_bits = 0

    def charge(kind: str, func: str, spent: Utilization) -> None:
        by_kind[kind] = by_kind.get(kind, Utilization()) + spent
        by_func[func] = by_func.get(func, Utilization()) + spent

    def comb(kind: CombKind, width: int) -> Utilization:
        return Utilization.of(price(device.comb_uses.get(kind.value, ()), (width,)))

    for f in report.microarch.funcs:
        by_func.setdefault(f.func, Utilization())

        # A register RUN is the cost unit, not a register: past the extraction
        # threshold the flip-flop count stops tracking depth. The reset flag
        # picks the row, since the reset is what blocks extraction; the enable
        # is measured cell-identical either way and prices nothing.
        for c in f.regs:
            uses = device.chain_uses if c.reset else device.chain_uses_norst
            run = Utilization.of(price(uses, (c.depth, c.width)))
            charge("regs", f.func, run * c.count)

        # Every counter/stride advances by an adder and turns over on a compare;
        # its update selects are self-holds and reach no LUT, so the control
        # plane is priced from the counters it belongs to rather than counted.
        for c in f.regs:
            if c.role is RegRole.COUNTER:
                step = comb(CombKind.ADD, c.width) + comb(CombKind.CMP, c.width)
                charge("control", f.func, step * c.count)

        for r in f.regions:
            for unit in r.units:
                if unit.impl is not None:
                    cost = ip_costs.get(unit.impl)
                    if cost is None:
                        unmodelled[unit.impl] += 1  # an IP nobody has synthesized
                        continue
                    charge("units", f.func, Utilization.of(price(cost[0], (cost[1],))))
                    continue
                mnemonic = unit.identity.split("(", 1)[0]
                if mnemonic not in COMB_KIND:
                    unmodelled[mnemonic] += 1
                    continue
                kind = COMB_KIND[mnemonic]
                if kind is not None:
                    charge("units", f.func, comb(kind, _unit_width(unit)))
            for m in r.muxes:
                one = Utilization.of(price(device.mux_uses, (m.fanin, m.width)))
                charge("muxes", f.func, one * m.count)

        # The select cones the emitter built around storage (shared-port
        # selects, commit sinks, crossbars): part of what the memory plane
        # costs, so they are charged beside the storage rows.
        for m in f.mux_cones:
            one = Utilization.of(price(device.mux_uses, (m.fanin, m.width)))
            charge("memories", f.func, one * m.count)

        for m in f.mems:
            # A boundary port, or cells the register ledger already holds.
            if m.realization in {"boundary", "scatter"}:
                continue
            row = device.storage.get(m.storage)
            if row is None:
                # This part declares no such row, which happens when a report
                # is priced against a device other than the one it was built for.
                unmodelled[m.storage] += 1
                continue
            # An addressed array costs its row once per instance and once per
            # bank. A constant table is one lookup built out of logic, so it is
            # priced the same way and holds no memory bits.
            copies = m.cost.instances * m.banks
            if m.realization != "rom":
                mem_bits += m.bits * m.cost.instances
            charge(
                "memories",
                f.func,
                Utilization.of(price(row.uses, (m.depth_words, m.width))) * copies,
            )

        # A channel between two children is a queue this module builds, priced
        # one cell per element since the register ledger does not hold it. A
        # channel that does not cross a call is a boundary port or an
        # intra-module queue the report does not distinguish, so it is charged
        # nowhere.
        for s in f.streams:
            if s.crosses_call:
                charge(
                    "memories",
                    f.func,
                    Utilization.of(price(scatter, (s.depth, s.width))),
                )

    area = Utilization()
    for spent in by_kind.values():
        area = area + spent

    sched = report.schedule
    top = sched.func(report.microarch.top.func)
    exact = top.latency_is_exact
    # The design's paths are its modules', which run on one clock.
    critical_paths = tuple(
        sorted(
            (p for f in report.microarch.funcs for p in f.critical_paths),
            key=lambda p: p.total,
            reverse=True,
        )
    )
    assert critical_paths and critical_paths[0].total > 0, (
        "every emitted module holds at least one register hop, so the design "
        "has a longest combinational path"
    )
    return QoR(
        latency=top.latency if exact else None,
        latency_max=top.latency if exact or top.latency_is_bound else None,
        latency_min=top.latency if exact or top.determinacy == "concurrent" else None,
        interval={
            f"{f.name}#{r.order}": r.interval
            for f in sched.funcs
            for r in f.regions
            if r.interval is not None
        },
        fmax=1000.0 / critical_paths[0].total,
        fmax_target=1000.0 / report.microarch.cycle_time,
        critical_paths=critical_paths,
        area=area,
        by_kind=by_kind,
        by_func=by_func,
        counted=frozenset({"dsp"}),
        unmodelled=dict(unmodelled),
        mem_bits=mem_bits,
        reg_bits=report.microarch.reg_bits,
        utilization=area.fraction_of(
            {name: r.capacity for name, r in device.resources.items()}
        ),
    )
