# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiler's account of itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..options import SchedulerOptions


@dataclass(frozen=True)
class SolveReport:
    """What one region's SOLVE cost: a measurement of the compiler, not a fact
    about the hardware.

    Deliberately not joined to a :class:`RegionSchedule`. A solve is keyed by the
    affine loop that owned the problem, and the regions above are read off the
    reified ``dcp`` ops, by which point that loop is gone. The two lists do not
    line up positionally either: a container decomposes into its children and
    solves nothing of its own, and one solve covers a whole perfect band."""

    func: str
    where: str  # source location, as the scheduler's own log names it
    kind: str  # `cyclic` / `while` / `acyclic`
    ops: int  # operations in the problem
    limited_ops: int  # of those, holding at least one limited unit
    ms: float  # wall time of the solve
    #: the interval this solve searched to, and so the depth of the search that
    #: `ms` measures; `None` for an acyclic span. The interval the region runs
    #: at is `RegionSchedule.interval`, which the two need not agree on.
    interval: int | None = None
    #: `simplex` or `cpsat`; the config below applies to a cpsat solve only.
    solver: str = ""
    workers: int = 0
    seed: int = 0
    budget_s: float = 0.0
    #: every deciding CP-SAT status was OPTIMAL, so the shipped schedule is the
    #: model's optimum.
    proven: bool = False
    #: some solve hit its time limit, so the shipped schedule may differ
    #: between runs of the same compile.
    budget_exhausted: bool = False
    #: the exact solve decided nothing and the heuristic's schedule shipped.
    fallback: bool = False
    #: the interval whose solve exhausted the budget, ending the cyclic search.
    exhausted_at_ii: int | None = None
    #: whether re-running this compile reproduces the same schedule: a simplex
    #: solve always does, a cpsat one unless its budget ran out or its workers
    #: raced (``SchedulerOptions.deterministic`` off).
    deterministic: bool = True

    @classmethod
    def from_json(cls, d: dict) -> SolveReport:
        return cls(
            func=d["func"],
            where=d["where"],
            kind=d["kind"],
            ops=d["ops"],
            limited_ops=d["limited_ops"],
            ms=d["ms"],
            interval=d.get("interval"),
            solver=d.get("solver", ""),
            workers=d.get("workers", 0),
            seed=d.get("seed", 0),
            budget_s=d.get("budget_s", 0.0),
            proven=d.get("proven", False),
            budget_exhausted=d.get("budget_exhausted", False),
            fallback=d.get("fallback", False),
            exhausted_at_ii=d.get("exhausted_at_ii"),
            deterministic=d.get("deterministic", True),
        )


@dataclass(frozen=True)
class CompilerReport:
    """Everything about the compile that is not about the hardware."""

    #: the scheduler's knobs, as the solve ran under them.
    options: SchedulerOptions | None = None
    #: what each solve cost, in the order the scheduler solved: funcs
    #: callees-first, and within a func its solving regions in program order.
    #: Not one entry per region (see :class:`SolveReport`).
    solves: list[SolveReport] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        """Whether re-running this compile reproduces every schedule; False
        exactly when some region's solve ran out of budget or raced its
        workers."""
        return all(s.deterministic for s in self.solves)

    @property
    def budget_exhausted(self) -> int:
        """How many region solves ran out of budget."""
        return sum(s.budget_exhausted for s in self.solves)

    @classmethod
    def from_json(cls, d: dict, options: SchedulerOptions | None) -> CompilerReport:
        return cls(
            options=options,
            solves=[SolveReport.from_json(s) for s in d.get("solves", [])],
        )
