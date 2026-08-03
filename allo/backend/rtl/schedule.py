# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling driver and the schedule result it returns"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from ..base import run_pipeline
from ..._mlir.dialects.allo import run_sdc_scheduling

# re-exported through allo.backend.rtl so callers do not reach into _mlir
# pylint: disable-next=unused-import
from ..._mlir.dialects.allo import has_exact_scheduler

RTL_PREPARE_PIPELINE = """
builtin.module(
grid-mapping,
fold-constant-calls,
canonicalize,
cse,
materialize-topology,
canonicalize,
cse,
convert-allo-to-func,
elide-dead-init,
func.func(convert-linalg-to-affine-loops),legalize-arith,canonicalize,cse,
outline-loose-processes)
"""

# --- schedule result data model --------------------------------------------


class RegionKind(str, Enum):
    """The scheduling regime of a region. A ``str`` mixin keeps
    ``region.kind == "cyclic"`` working alongside ``region.kind is
    RegionKind.CYCLIC``."""

    CYCLIC = "cyclic"  # a pipelined loop (dcp.pipeline)
    ACYCLIC = "acyclic"  # a straight-line span (dcp.sequential)
    GUARD = "guard"  # a control select (dcp.select); carries no compute itself


@dataclass(frozen=True)
class ScheduledOp:
    """One scheduled operation inside a region."""

    kind: str  # operator mnemonic (addi/mulf/load/store/...); an open set, so str
    t: int  # start cycle within the region
    impl: str | None = None  # realization (device operator symbol / native)
    z: float | None = None  # SDC z-slack, when carried

    @classmethod
    def from_json(cls, d: dict) -> ScheduledOp:
        return cls(kind=d["kind"], t=d["t"], impl=d.get("impl"), z=d.get("z"))


@dataclass(frozen=True)
class RegionSchedule:
    """One scheduling region (a dcp.pipeline / dcp.sequential / dcp.select)."""

    kind: RegionKind
    order: int  # program order among the func's regions
    depth: int  # nesting depth among dcp regions (0 = outermost)
    container: bool  # nests another region (a loop / guard wrapper)
    ops: list[ScheduledOp] = field(default_factory=list)
    ii: int | None = None  # cyclic only; None for a dynamic-trip sequential wrapper
    trip: int | None = None  # constant trip count, when known
    length: int | None = None  # schedule depth: every op has completed by here
    # Terminal cycle: the last issue pulse to the deepest output committing, so
    # `done` rises a cycle later. What a span composes from, and not `length`,
    # which may carry slack the solver left above the last commit.
    drain: int | None = None
    latency: int | None = None  # region latency (cycles)
    latency_is_bound: bool = False  # latency is an upper bound, not exact
    conditional: bool = False  # while-pipeline (dcp.condition) or a guard
    # The controller family that paces this region: `counted_static`,
    # `conditional`, `indeterminate` or `concurrent`.
    determinacy: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> RegionSchedule:
        return cls(
            kind=RegionKind(d["kind"]),
            order=d["order"],
            depth=d["depth"],
            container=d["container"],
            ops=[ScheduledOp.from_json(o) for o in d["ops"]],
            ii=d.get("ii"),
            trip=d.get("trip"),
            length=d.get("length"),
            drain=d.get("drain"),
            latency=d.get("latency"),
            latency_is_bound=d["latency_bound"],
            conditional=d["conditional"],
            determinacy=d.get("determinacy"),
        )

    @property
    def is_wrapper(self) -> bool:
        """A container region carrying no compute of its own (a residual outer
        loop around leaf regions): a derived nesting node, not a scheduling
        decision."""
        return self.container and not self.ops

    @property
    def is_leaf(self) -> bool:
        return not self.container

    def op(self, kind: str) -> ScheduledOp:
        """The first op of the given kind (raises ``StopIteration`` if none)."""
        return next(o for o in self.ops if o.kind == kind)

    def has(self, kind: str) -> bool:
        return any(o.kind == kind for o in self.ops)

    def last_t(self) -> int:
        """The latest start cycle among this region's ops."""
        return max(o.t for o in self.ops)


@dataclass(frozen=True)
class FuncSchedule:
    """The schedule of one kernel (an ``allo.dcp.module``)."""

    name: str
    regions: list[RegionSchedule] = field(default_factory=list)
    latency: int | None = None  # whole-func latency (cycles), when static
    latency_is_bound: bool = False
    # Composition class: `counted_static` (`latency` is an exact start->done
    # span), `indeterminate` (consumers gate on `done`), or `concurrent`
    # (children paced by back-pressure, so `latency` is a floor).
    determinacy: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> FuncSchedule:
        return cls(
            name=d["name"],
            regions=[RegionSchedule.from_json(r) for r in d["regions"]],
            latency=d.get("latency"),
            latency_is_bound=d["latency_bound"],
            determinacy=d.get("determinacy"),
        )

    @property
    def latency_is_exact(self) -> bool:
        """Whether ``latency`` is an exact span the hardware must realize, and
        so a number a measured cycle count may be held to. A bounded, elastic or
        concurrent kernel publishes a figure that is deliberately not tight."""
        return (
            self.latency is not None
            and self.determinacy == "counted_static"
            and not self.latency_is_bound
        )

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        """This func's cyclic regions; pure sequential wrappers excluded unless
        ``wrappers=True``."""
        return [
            r
            for r in self.regions
            if r.kind is RegionKind.CYCLIC and (wrappers or not r.is_wrapper)
        ]


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
    ii: int | None = None  # what the solve decided; None for an acyclic span
    #: operations whose operator count the solve decided, and the units it
    #: decided to build for them. Both 0 unless the solve allocated, which only
    #: an exact solve with ``allocate`` does.
    allocated_ops: int = 0
    allocated_units: int = 0

    @classmethod
    def from_json(cls, d: dict) -> SolveReport:
        return cls(
            func=d["func"],
            where=d["where"],
            kind=d["kind"],
            ops=d["ops"],
            limited_ops=d["limited_ops"],
            ms=d["ms"],
            ii=d.get("ii"),
            allocated_ops=d.get("allocated_ops", 0),
            allocated_units=d.get("allocated_units", 0),
        )


@dataclass(frozen=True)
class ScheduleResult:
    """The whole-module schedule result: the schedule of every kernel."""

    funcs: list[FuncSchedule] = field(default_factory=list)
    #: per-region solve cost, in solve order (see :class:`SolveReport`).
    solves: list[SolveReport] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str | dict) -> ScheduleResult:
        """Parse the JSON schedule result the scheduler returns, either as the
        raw string or as an already-decoded object."""
        d = json.loads(text) if isinstance(text, str) else text
        return cls(
            funcs=[FuncSchedule.from_json(f) for f in d["funcs"]],
            solves=[SolveReport.from_json(s) for s in d.get("solves", [])],
        )

    def func(self, suffix: str) -> FuncSchedule:
        """The sub-function whose name ends with ``suffix`` (kernels compose by
        calling sub-kernels, so results carry ``top.sub`` funcs)."""
        return next(f for f in self.funcs if f.name.endswith(suffix))

    def regions(
        self, kind: RegionKind | None = None, *, wrappers: bool = False
    ) -> list[RegionSchedule]:
        """Regions across all funcs, optionally filtered by kind. Pure
        sequential wrappers are excluded by default (they carry a derived II, not
        a scheduling decision); pass ``wrappers=True`` for the full nesting
        tree."""
        return [
            r
            for f in self.funcs
            for r in f.regions
            if (kind is None or r.kind is kind) and (wrappers or not r.is_wrapper)
        ]

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        return self.regions(RegionKind.CYCLIC, wrappers=wrappers)


# --- driver ----------------------------------------------------------------


def run_schedule(
    top,
    module,
    *,
    cycle_time=None,
    float_reassoc=True,
    accumulators=0,
    perfectize=False,
    unroll_under_pipeline=True,
    scalarize_threshold=16,
    scheduler="heuristic",
    budget=None,
    allocate=False,
) -> ScheduleResult:
    """Schedule ``top`` and return the :class:`ScheduleResult`; ``module`` is
    rewritten in place, left holding the ``allo.dcp.*`` ops the schedule reifies
    into. Operator/device timing is read from the ``dcp.device`` / ``dcp.operator``
    ops injected into ``module`` before this call.

    Args:
        top: the name of the function to schedule.
        module: the MLIR module holding it.
        cycle_time: target clock period (ns); ``None`` falls back to 5.0.
        float_reassoc: rebalance float reduction chains into logarithmic trees.
        accumulators: rotate float reductions across this many accumulators (0 =
            off).
        perfectize: sink an imperfect nest's prologue/epilogue into the inner
            loop under a guard, fusing it into one pipeline.
        unroll_under_pipeline: fully unroll the loops nested inside a pipelined
            loop, so the nest pipelines at one II.
        scalarize_threshold: scalarize memory accesses to arrays with this many or
            fewer elements, so they are kept in registers rather than a memory.
            Set to 0 to disable.
        scheduler: the solver that settles the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement; ``"exact"``
            is CP-SAT over the same problem, keeping the chain-breaking pre-pass;
            ``"exact-chaining"`` additionally decides where to break a too-long
            combinational chain in the solver. Both exact modes need OR-Tools.
        budget: what ONE SOLVE may spend, in the solver's deterministic time
            units (roughly a second of one core each); ``None`` takes the
            default. Raising it buys a better placement on the few regions large
            enough to exhaust it and costs nothing on the rest, which finish
            orders of magnitude under it.
        allocate: decide how many copies of each operator a region builds,
            rather than leaving every operation its own. Exact schedulers only,
            and only useful under a sharing binding.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    reassoc = (
        "reassociate-reductions{float-reassoc="
        f"{'true' if float_reassoc else 'false'}}}"
    )
    rotate = f"rotate-reductions{{accumulators={int(accumulators)}}}"
    loops = (
        "loop-canonicalization{"
        f"unroll-under-pipeline={'true' if unroll_under_pipeline else 'false'} "
        f"perfectize={'true' if perfectize else 'false'}}}"
    )
    part = f"propagate-partition{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={scalarize_threshold}}}"
    pipeline = (
        f"builtin.module(canonicalize,cse,func.func(raise-to-affine,cse,"
        f"raise-counted-while,{loops},"
        f"canonicalize,fold-if-statements,cse,{scalarize},"
        f"{reassoc},{rotate}),drop-trivial-func,"
        f"{part},func.func(assign-banks))"
    )
    run_pipeline(module, pipeline)
    diagnostics: list[str] = []
    handler = module.context.attach_diagnostic_handler(
        lambda d: diagnostics.append(d.message) or True
    )
    try:
        result = run_sdc_scheduling(
            module, top, cycle_time or 5.0, scheduler, budget or 0.0, allocate
        )
    finally:
        handler.detach()
    if result is None:
        raise RuntimeError(
            "An error occurred during scheduling process:\n" + "\n".join(diagnostics)
        )
    return ScheduleResult.from_json(result)
