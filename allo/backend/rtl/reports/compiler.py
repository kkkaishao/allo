# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiler's account of ITSELF."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..options import SchedulerOptions


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
    #: the interval this solve searched to, and so the depth of the search that
    #: `ms` measures; `None` for an acyclic span. The interval the region RUNS
    #: at is `RegionSchedule.interval`, which the two need not agree on.
    interval: int | None = None

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
        )


@dataclass(frozen=True)
class CompilerReport:
    """Everything about the compile that is not about the hardware."""

    #: the scheduler's knobs, as the solve ran under them. The same object the
    #: caller turned, not a copy of it.
    options: SchedulerOptions | None = None
    #: per-region solve cost, in solve order (see :class:`SolveReport`).
    solves: list[SolveReport] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict, options: SchedulerOptions | None) -> CompilerReport:
        return cls(
            options=options,
            solves=[SolveReport.from_json(s) for s in d.get("solves", [])],
        )
