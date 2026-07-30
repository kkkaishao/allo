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
    impl: str | None = None  # realization (IP module name / native keyword)
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
    length: int | None = None  # single-iteration cycle span
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
            latency=d.get("latency"),
            latency_is_bound=d["latency_bound"],
            conditional=d["conditional"],
            determinacy=d.get("determinacy"),
        )

    @property
    def is_wrapper(self) -> bool:
        """A container region carrying no compute of its own (a residual outer
        loop around leaf regions) -- a derived nesting node, not a scheduling
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
class ScheduleResult:
    """The whole-module schedule result: the schedule of every kernel."""

    funcs: list[FuncSchedule] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str | dict) -> ScheduleResult:
        """Parse the JSON schedule result the scheduler returns, either as the
        raw string or as an already-decoded object."""
        d = json.loads(text) if isinstance(text, str) else text
        return cls(funcs=[FuncSchedule.from_json(f) for f in d["funcs"]])

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
            ``"heuristic"`` is the SDC simplex plus greedy placement;
            ``"exact"`` is CP-SAT, available only in a build with OR-Tools.
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
        f"builtin.module(canonicalize,cse,func.func(raise-counted-while,{loops},"
        f"canonicalize,fold-if-statements,cse,{scalarize},"
        f"{reassoc},{rotate}),"
        f"{part},func.func(assign-banks))"
    )
    run_pipeline(module, pipeline)
    result = run_sdc_scheduling(module, top, cycle_time or 5.0, scheduler)
    if result is None:
        raise RuntimeError(
            f"Scheduling step failed for {top}. Please check the log for details."
        )
    return ScheduleResult.from_json(result)
