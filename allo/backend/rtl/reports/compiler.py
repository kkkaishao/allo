# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiler's account of ITSELF.

What it was asked for, what the ask cost it, and where its own pricing model is
coarse. None of this is a property of the design, which is why it is a section
of its own rather than a few more fields beside the latency and the area: a
solve that took four seconds and a design that takes four cycles are not
comparable quantities and should not read as though they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScheduleSettings:
    """What the scheduler was ASKED for.

    Two results are comparable only under the same settings, and nothing else
    records them: the IR the scheduler leaves behind carries its decisions, not
    the knobs it was turned with. This is also the knob list itself, which
    ``RTL.set_scheduler_opt`` turns by field name.

    Args:
        scheduler: the solver that settles the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement; ``"exact"``
            is CP-SAT over the same problem, keeping the chain-breaking pre-pass;
            ``"exact-chaining"`` additionally decides where to break a too-long
            combinational chain in the solver. Both exact modes need OR-Tools.
        cycle_time_ns: the period every combinational chain was cut to. Derived
            from the handle's ``freq_mhz``, which the cosim clock also reads.
        allocate: decide how many copies of each operator a region builds, rather
            than leaving every operation its own. Derived from the binding, since
            an allocation is only worth deciding where a sharing binding builds
            it; exact schedulers only.
        float_reassoc: rebalance float reduction chains into logarithmic trees.
            Not bit-exact.
        accumulators: rotate float reductions across this many accumulators,
            dropping their II to ``ceil(latency / accumulators)`` (0 = off).
        unroll_under_pipeline: fully unroll the loops nested inside a pipelined
            loop, so the nest pipelines at one II (Vitis ``#pragma HLS pipeline``
            semantics). ``False`` keeps them rolled and the directive is then not
            honored.
        perfectize: sink an imperfect nest's prologue/epilogue into the inner
            loop under a guard, fusing it into one pipeline. A QoR alternative;
            the scheduler handles imperfect nests without it.
        scalarize_threshold: keep arrays of at most this many elements in
            registers rather than a memory (0 = off).
        budget: what ONE exact solve may spend, in the solver's deterministic
            time units (roughly a second of one core each); ``None`` takes the
            default. Raising it buys a better placement on the few regions large
            enough to exhaust it and costs nothing on the rest.
    """

    scheduler: str = "heuristic"
    cycle_time_ns: float = 5.0
    allocate: bool = False
    float_reassoc: bool = True
    accumulators: int = 0
    unroll_under_pipeline: bool = True
    perfectize: bool = False
    scalarize_threshold: int = 16
    budget: float | None = None


@dataclass(frozen=True)
class CarriedEdges:
    """The loop-carried memory edges one solve had to respect, split by what
    fixed each distance.

    An II resting on an ASSUMED distance is a scheduling-quality fact: the
    dependence test either proved the distance or fell back, and the problem
    that held the edges is gone by the time the schedule is reified, so nothing
    downstream can count them again."""

    total: int  # every carried edge in the problem
    non_affine: int  # of those, a pair the polyhedral test cannot model
    unknown: int  # of those, a direction it modelled but could not bound

    @classmethod
    def from_json(cls, d: dict) -> CarriedEdges:
        return cls(total=d["total"], non_affine=d["non_affine"], unknown=d["unknown"])

    @property
    def assumed(self) -> int:
        """Edges whose distance rests on an assumption rather than a proof."""
        return self.non_affine + self.unknown


@dataclass(frozen=True)
class SolveReport:
    """What one region's SOLVE cost: a measurement of the compiler, not a fact
    about the hardware.

    Deliberately not joined to a :class:`RegionSchedule`. A solve is keyed by the
    affine loop that owned the problem, and the regions above are read off the
    reified ``dcp`` ops, by which point that loop is gone. Both lists are in
    program order per func, which is as much of a correspondence as holds."""

    func: str
    where: str  # source location, as the scheduler's own log names it
    kind: str  # `cyclic` / `while` / `acyclic`
    ops: int  # operations in the problem
    limited_ops: int  # of those, holding at least one limited unit
    ms: float  # wall time of the solve
    interval: int | None = None  # what the solve decided; None for an acyclic span
    #: operations whose operator count the solve decided, and the units it
    #: decided to build for them. Both 0 unless the solve allocated, which only
    #: an exact solve with ``allocate`` does.
    allocated_ops: int = 0
    allocated_units: int = 0
    #: what the II had to respect across iterations; ``None`` for an acyclic
    #: span, which models no carried edge at all.
    carried_edges: CarriedEdges | None = None

    @classmethod
    def from_json(cls, d: dict) -> SolveReport:
        carried = d.get("carried_edges")
        return cls(
            func=d["func"],
            where=d["where"],
            kind=d["kind"],
            ops=d["ops"],
            limited_ops=d["limited_ops"],
            ms=d["ms"],
            interval=d.get("interval"),
            allocated_ops=d.get("allocated_ops", 0),
            allocated_units=d.get("allocated_units", 0),
            carried_edges=CarriedEdges.from_json(carried) if carried else None,
        )


@dataclass(frozen=True)
class OperatorClass:
    """One operator TYPE the library prices several operations under, and the
    number of distinct operator IDENTITIES those operations hold.

    ``identities > 1`` is where the pricing over-approximates: every operation
    of the type is charged one row though the rows they would each want differ.
    A cost estimate over this schedule is only as tight as this list is short."""

    type: str
    ops: int
    identities: int

    @classmethod
    def from_json(cls, d: dict) -> OperatorClass:
        return cls(type=d["type"], ops=d["ops"], identities=d["identities"])


@dataclass(frozen=True)
class CompilerReport:
    """Everything about the compile that is not about the hardware."""

    settings: ScheduleSettings | None = None
    #: per-region solve cost, in solve order (see :class:`SolveReport`).
    solves: list[SolveReport] = field(default_factory=list)
    #: operator types whose one pricing row covers several operator identities,
    #: which is exactly where the cost model OVER-approximates. The confidence
    #: annotation a QoR estimate carries (see :class:`OperatorClass`).
    coarse_pricing: list[OperatorClass] = field(default_factory=list)

    @property
    def solve_ms(self) -> float:
        """Wall time across every region's solve."""
        return sum(s.ms for s in self.solves)

    @classmethod
    def from_json(cls, d: dict, settings: ScheduleSettings | None) -> CompilerReport:
        return cls(
            settings=settings,
            solves=[SolveReport.from_json(s) for s in d.get("solves", [])],
            coarse_pricing=[
                OperatorClass.from_json(c) for c in d.get("operator_classes", [])
            ],
        )
