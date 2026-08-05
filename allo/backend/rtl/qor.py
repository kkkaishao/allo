# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one compile COSTS: the report's structures, priced against the device.

A pure function of ``(CompileReport, Device)``. It is not an information source
of its own: the structures come from the microarchitecture report, which is what
the emitter BUILT, and every price comes from the device's own declaration in
``area_tables.py``, evaluated through the compiler's one cost evaluator. Nothing
here reads the emitted IR, so nothing here can drift from what was emitted.

The tables are MEASURED, not estimated: Vivado 2023.2, ``xcu55c-fsvh2892-2L-e``,
out-of-context synthesis of one DUT per (kind, width), one Xilinx Floating-Point
core per device operator at its declared latency, and a sweep per multiplexer
fan-in, chain depth and array shape, with primitives counted off the netlist.

Five properties a reader has to know before quoting a number:

  - **Provenance is per RESOURCE.** ``dsp`` is a count: an IP declares its own
    and a multiplier's is a measured constant per width, exact against synthesis
    on every design measured so far. :attr:`QoR.reg_bits` is a count too, of a
    different kind: every register an emitted module holds passes one line of the
    emitter, which charges the ledger the report publishes. ``lut``, ``carry8``
    and ``ff`` are a census of structures priced against a table, and an
    estimate. :attr:`QoR.counted` names the first kind.
  - **Accuracy, against real synthesis of four bed kernels** (``validate.py`` in
    ``drafts/p6-area/``): LUT lands between 1.02x and 1.28x and flip-flops
    between 1.07x and 1.75x, over-reading both. A census of structures cannot see
    LUT fusion or constant folding, and the flip-flop row does not know how much
    further than its own threshold the synthesizer will push a chain into an SRL.
    Use it to COMPARE two schedules, which is what it is for, and not as a
    utilization figure.
  - **The multiplexer is priced as a STRUCTURE.** ``EmitContext::oneHotSelect``
    emits ``or(and(v, replicate(sel)))`` and a LUT6 absorbs three (data, select)
    pairs, so synthesis fuses the whole cone; the report counts the cone once, as
    one mux of a fan-in and a width, which is what the device prices. Pricing the
    operations instead over-counts it about fivefold.
  - **Memory is priced by its WRITE PORT COUNT, and the cliff is enormous.** One
    write port infers a block RAM and costs no fabric at all; two infer a true
    dual port and are free again, but only where the schedule proved the writers
    never collide, since that is what lets each be described in its own ``always``
    block. Where it did not, the array becomes a register file with a data
    multiplexer in front of every word: measured at 512x32, one BRAM18 against
    33,245 LUTs and 16,416 flip-flops. Inferred RAM is reported as
    :attr:`QoR.mem_bits`, APART from the fabric totals, since no scheduling
    decision trades one for the other.
  - **A self-hold multiplexer costs nothing.** ``mux(enable, next, self)`` is how
    the emitter spells an enabled register and ``mux(load, init, ...)`` how it
    spells a load; synthesis maps the first onto the flip-flop's clock enable and
    the second onto its reset path, and neither reaches a LUT. So a counter is
    charged its incrementer and its bound test and nothing else, and a survivor
    latch is charged nothing at all. Charging those selects read 1.24x to 1.58x
    of measured synthesis where leaving them out reads 1.02x to 1.28x.

The one result here that contradicts the compiler: **a value delay chain deeper
than three does not cost flip-flops.** Vivado extracts it into SRLs, so its cost
is about ``w`` SRL sites plus ``2w`` flip-flops and is nearly INDEPENDENT of
depth, against the ``depth * width`` flip-flops the scheduling objective's
register term charges. That is why the report publishes a register RUN rather
than a register count, and why :attr:`QoR.reg_bits`, the flip-flops the design
DECLARES, sits beside ``area.ff``, the flip-flops the part is expected to hold.

What the report does not carry, and so is not priced: the emitter's glue. The
1-bit control cone (run / issue / done), the address and data multiplexers of a
shared memory port, and the ``affine.apply`` of a non-affine address are
operations the report deliberately aggregates away rather than structures it
knows. Measured over sixteen bed variants, leaving them out reads 0.29x to 0.89x
of the IR-walking scorer this replaced, and CLOSER to real synthesis than it on
all four designs that have been synthesized (1.02x to 1.28x against 1.58x to
1.82x), which is what says the missing operations are largely ones the
synthesizer fuses away rather than area this fails to charge.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .device import CombKind, Device, builtin_device
from .reports.microarch import Memory, RegRole, Unit
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

    lut: int = 0  # every LUT site, SRLs included
    logic_lut: int = 0  # of those, the ones not holding state
    srl: int = 0  # a LUT site holding a shift register (`slicem_lut`)
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
    #: the achieved clock. ``None`` until something retimes the emitted design:
    #: the schedule was CUT to ``fmax_target`` and nothing in the model says
    #: whether the part holds it.
    fmax: float | None
    fmax_target: float  # MHz, the period the schedule was cut to
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
    mem_bits: int  # inferred RAM, APART from the fabric totals
    regfile_arrays: int  # arrays that failed RAM inference
    #: flip-flops the emitted design DECLARES, from the register ledger, and so a
    #: count. ``area.ff`` is the modelled figure beside it: what the part holds
    #: once the deep chains are extracted into SRLs.
    reg_bits: int
    #: operator types whose timing row covers several operator IDENTITIES, and
    #: how many. Every operation of such a type was scheduled against one row
    #: though the rows they would each want differ, so a long list is an estimate
    #: over a coarser model than a short one.
    coarse_types: dict[str, int]

    @property
    def latency_is_exact(self) -> bool:
        """Whether ``latency`` is a span the hardware must realize, and so a
        number two runs may be differenced on."""
        return self.latency is not None


def _operator_costs(device: Device) -> dict[str, tuple]:
    """Each priced operator's declared cost and the operand width it is a
    function of. The width is fixed by the IP's signature, which is why the
    declaration is a constant, but it is still the parameter its kind carries."""
    out = {}
    for op in device.operators:
        uses = device.operator_uses.get(op.func_name)
        if uses:
            widths = [a.primitive_width for a in op.parse_argument_annotations()]
            out[op.func_name] = (uses, max(widths))
    return out


def _register_file(device: Device):
    """What an array that failed RAM inference falls back to: every word gets a
    data multiplexer and a write decode, which is what a complete partition
    builds too. The device's ``is_scatter`` row, since the compiler names no
    storage of its own and neither does this estimator."""
    row = next((s for s in device.storage.values() if s.is_scatter), None)
    if row is None:
        raise ValueError(
            f"device {device.name!r} marks no scatter storage, so there is "
            "nothing to price an array that failed RAM inference against"
        )
    return row.uses


def _unit_width(unit: Unit) -> int:
    """The width a unit's cost is a function of: the widest of its result and its
    operands, since a compare returns one bit and costs its operands."""
    args = _ARGS.match(unit.identity)
    widths = [int(w) for w in _INT_WIDTH.findall(args.group(1))] if args else []
    return max([unit.width, *widths])


def _infers_ram(mem: Memory, device: Device) -> bool:
    """Whether the synthesizer recognizes a RAM template for this array.

    One write port always does. Two do only where the schedule proved the writers
    never collide, which the emitter proves for writers of ONE region or ports of
    ONE child; that shape is what is read back here, since the decision itself is
    taken during emission and the report carries it only for a boundary array."""
    ports = mem.cost.ports_needed_write
    if ports <= 1:
        return True
    one_source = (mem.cost.writing_regions <= 1 and mem.cost.writing_calls == 0) or (
        mem.writes == 0 and mem.cost.writing_calls == 1
    )
    return ports <= device.max_writes and one_source


# One pass over the report, one bucket per kind of structure it publishes.
# pylint: disable-next=too-many-locals
def estimate(report: CompileReport, device: Device = builtin_device) -> QoR:
    """Price ``report``'s structures against ``device``.

    Every module the emission built is priced separately: a design with
    sub-kernels spends its area in all of them, and which one a schedule change
    lands in is the first thing a reader asks."""
    price = device.price
    ip_costs = _operator_costs(device)
    regfile = _register_file(device)

    by_kind: dict[str, Utilization] = {}
    by_func: dict[str, Utilization] = {}
    unmodelled: Counter = Counter()
    mem_bits = 0
    regfile_arrays = 0

    def charge(kind: str, func: str, spent: Utilization) -> None:
        by_kind[kind] = by_kind.get(kind, Utilization()) + spent
        by_func[func] = by_func.get(func, Utilization()) + spent

    def comb(kind: CombKind, width: int) -> Utilization:
        return Utilization.of(price(device.comb_uses.get(kind.value, ()), (width,)))

    for f in report.microarch.funcs:
        by_func.setdefault(f.func, Utilization())

        # A register RUN is the cost unit, not a register: past the extraction
        # threshold the flip-flop count stops tracking depth.
        for c in f.regs:
            run = Utilization.of(price(device.chain_uses, (c.depth, c.width)))
            charge("regs", f.func, run * c.count)

        # The control plane, as `emitScaledCounters` builds it: every counter and
        # address stride advances by an adder and turns over on a compare. The
        # selects of its update are self-holds and reach no LUT, and the report
        # aggregates the logic away, so it is priced from the counters it belongs
        # to rather than counted.
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

        for m in f.mems:
            if m.external or m.scattered:
                continue  # a boundary port, or cells the register ledger holds
            if _infers_ram(m, device):
                mem_bits += m.bits
                # A block RAM or UltraRAM the device was ASKED for is charged in
                # its own tiles; which primitive an INFERRED RAM lands in the
                # model cannot say, and the measurement says it costs no fabric.
                row = device.storage.get(m.storage)
                tiled = ("bram36", "uram288")
                if row is not None and any(r in tiled for r, _ in row.uses):
                    charge(
                        "memories",
                        f.func,
                        Utilization.of(price(row.uses, (m.depth_words, m.width))),
                    )
                continue
            bank = Utilization.of(price(regfile, (m.depth_words, m.width)))
            charge("memories", f.func, bank * m.banks)
            regfile_arrays += 1

        # A channel between two children is a queue this module builds, and its
        # words are the one storage the register ledger does not hold, so they
        # are MODELLED as a register file rather than counted. A channel that
        # does not cross a call is charged nowhere: it is either this module's
        # own boundary port, which is wires, or a queue between two of its
        # regions, which the report does not tell apart from one.
        for s in f.streams:
            if s.crosses_call:
                charge(
                    "memories",
                    f.func,
                    Utilization.of(price(regfile, (s.depth, s.width))),
                )

    area = Utilization()
    for spent in by_kind.values():
        area = area + spent

    sched = report.schedule
    top = sched.func(report.microarch.top.func)
    exact = top.latency_is_exact
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
        fmax=None,
        fmax_target=1000.0 / report.microarch.cycle_time,
        area=area,
        by_kind=by_kind,
        by_func=by_func,
        counted=frozenset({"dsp"}),
        unmodelled=dict(unmodelled),
        mem_bits=mem_bits,
        regfile_arrays=regfile_arrays,
        reg_bits=report.microarch.reg_bits,
        coarse_types={
            c.type: c.identities
            for c in sched.compiler.coarse_pricing
            if c.identities > 1
        },
    )
