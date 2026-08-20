# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling driver. The result it returns lives in `reports`."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from ..base import run_pipeline
from ..._mlir.ir import Module
from ..._mlir.dialects.allo import run_sdc_scheduling

from .options import PrepassOptions, SchedulerOptions
from .reports.schedule import ScheduleResult, SweepPoint

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

# --- driver ----------------------------------------------------------------


def run_schedule(
    top,
    module,
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
) -> ScheduleResult:
    """Schedule ``top`` and return the :class:`ScheduleResult`. ``module`` is
    rewritten in place, left holding the ``allo.dcp.*`` ops the schedule reifies
    into. Operator/device timing is read from the ``dcp.device`` /
    ``dcp.operator`` ops injected into ``module`` before this call.

    ``prepass`` shapes the IR the scheduler is handed, ``options`` is what the
    scheduler itself is asked for, and ``allocate`` lets an exact solve decide
    how many copies of each operator a region builds.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    # The model period every period-dependent stage below reads: the operating
    # clock less the margin withheld. The cosim clock stays the operating one.
    if not 0.0 <= options.clock_margin < 1.0:
        raise ValueError(f"clock_margin must lie in [0, 1); got {options.clock_margin}")
    model_ns = options.cycle_ns * (1.0 - options.clock_margin)
    reassoc = (
        "reassociate-reductions{float-reassoc="
        f"{'true' if prepass.float_reassoc else 'false'}}}"
    )
    rotate = f"rotate-reductions{{accumulators={int(prepass.accumulators)}}}"
    loops = (
        "loop-canonicalization{"
        f"unroll-under-pipeline={'true' if prepass.unroll_under_pipeline else 'false'} "
        f"perfectize={'true' if prepass.perfectize else 'false'}}}"
    )
    part = f"reconcile-array-directives{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={prepass.scalarize_threshold}}}"
    pipeline = (
        f"builtin.module(canonicalize,cse,func.func(raise-to-affine,cse,"
        f"raise-counted-while,{loops},"
        f"canonicalize,fold-if-statements,cse,{scalarize},"
        f"{reassoc},{rotate},narrow-demanded-bits),drop-trivial-func,"
        f"{part},func.func(hoist-invariant-reads,assign-banks),canonicalize,cse,"
        f"func.func(expand-region-bounds),"
        f"legalize-arith{{expand-const-arith=true period-ns={model_ns}}},"
        f"canonicalize,cse)"
    )
    run_pipeline(module, pipeline)
    diagnostics: list[str] = []
    handler = module.context.attach_diagnostic_handler(
        lambda d: diagnostics.append(d.message) or True
    )
    try:
        result = run_sdc_scheduling(
            module,
            top,
            model_ns,
            options.scheduler,
            # "freq" and "wall" are period policies this driver sweeps; their
            # region solves run under the cycles order.
            "cycles" if options.O in ("freq", "wall") else options.O,
            options.budget,
            allocate,
            options.workers,
            options.seed,
            options.area_slack,
        )
    finally:
        handler.detach()
    if result is None:
        raise RuntimeError(
            "An error occurred during scheduling process:\n" + "\n".join(diagnostics)
        )
    return ScheduleResult.from_json(result, options)


# The period sweep's ladder: this many geometric rungs between the discovered
# device floor and the requested period, beyond the two endpoint probes.
_SWEEP_RUNGS = 8


def _region_vector(result) -> dict:
    """Every solved per-region quantity a span composes from, keyed stably
    across probes of one kernel. Trip counts are what a span-less kernel is
    missing, and none of these depend on one."""
    out = {}
    for f in result.funcs:
        for r in f.regions:
            for name, v in (
                ("ii", r.interval),
                ("len", r.iteration_latency),
                ("drain", r.cost.drain),
            ):
                if v is not None:
                    out[(f.name, r.order, name)] = v
    return out


def sweep_freq(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    floor_ns: float,
) -> tuple[Module, ScheduleResult]:
    """Minimize the operating period under ``O="freq"``: probe candidates below
    ``options.cycle_ns`` with the heuristic scheduler, keep those whose span
    stays within ``span_tolerance`` of the span at the requested clock, and
    solve once at the tightest survivor under the caller's own scheduler
    settings. A kernel with no composed span (unknown trip counts) is leashed
    per region instead: every solved quantity the span composition is monotone
    in must hold the same tolerance, which bounds the span for any trips.
    Every probe recompiles from pristine IR (``make_module``), since
    the legalized op set depends on the period; all probes are heuristic so the
    chosen clock is deterministic. ``floor_ns`` is the device's register floor,
    which bounds how deep the probes reach. Returns the scheduled module and
    its result, with the probed curve published as ``ScheduleResult.sweep``."""
    if options.span_tolerance < 0.0:
        raise ValueError(
            f"span_tolerance must be non-negative; got {options.span_tolerance}"
        )
    margin = 1.0 - options.clock_margin
    vectors: dict[float, dict] = {}

    def probe(period: float) -> SweepPoint:
        opts = replace(options, scheduler="heuristic", cycle_ns=period)
        result = run_schedule(top, make_module(), opts, prepass, allocate)
        vectors[period] = _region_vector(result)
        fn = result.func(top)
        return SweepPoint(
            cycle_ns=period,
            achieved_ns=result.cycle_ns / margin,
            latency=fn.latency,
            latency_is_bound=fn.latency_is_bound,
        )

    asked = probe(options.cycle_ns)
    # The aggressive probe: the least period whose cycle still holds logic,
    # twice the device's register floor (an 8x faster clock where the device
    # declares none). The derate lifts an unholdable ask, so what this one
    # achieves is the tightest clock on offer.
    anchor = 2.0 * floor_ns / margin if floor_ns > 0 else options.cycle_ns / 8.0
    points = [asked]
    lo = options.cycle_ns
    if anchor < options.cycle_ns:
        floor = probe(anchor)
        points.append(floor)
        lo = floor.achieved_ns
    if lo < options.cycle_ns:
        ratio = options.cycle_ns / lo
        points += [
            probe(lo * ratio ** (k / (_SWEEP_RUNGS + 1)))
            for k in range(1, _SWEEP_RUNGS + 1)
        ]
    # Candidates that derate onto the same achieved period are one design;
    # keep the laxest ask of each.
    seen: set[float] = set()
    curve: list[SweepPoint] = []
    for p in sorted(points, key=lambda p: p.cycle_ns, reverse=True):
        if (key := round(p.achieved_ns, 3)) not in seen:
            seen.add(key)
            curve.append(p)
    # A bounded span compares as its worst case; `asked` always qualifies, so
    # there is a winner. With no span to compare, hold the per-region vector:
    # the composed span is monotone in every entry, so a probe inside the
    # per-region leash spends no more than the tolerance at any trip counts.
    tol = 1.0 + options.span_tolerance
    if asked.latency is not None:
        leash = asked.latency * tol
        eligible = [p for p in curve if p.latency is not None and p.latency <= leash]
    else:
        ref = vectors[asked.cycle_ns]
        eligible = [
            p
            for p in curve
            if vectors[p.cycle_ns].keys() == ref.keys()
            and all(vectors[p.cycle_ns][k] <= v * tol for k, v in ref.items())
        ]
    winner = min(eligible, key=lambda p: (p.achieved_ns, p.latency or 0))
    module = make_module()
    result = run_schedule(
        top, module, replace(options, cycle_ns=winner.cycle_ns), prepass, allocate
    )
    return module, replace(result, sweep=tuple(curve))


# The wall sweep's ladder: this many geometric rungs between the aggressive
# anchor and the slowest period any device row is built for.
_WALL_RUNGS = 10


def sweep_wall(
    top,
    make_module: Callable[[], Module],
    options: SchedulerOptions,
    prepass: PrepassOptions,
    allocate: bool,
    floor_ns: float,
    cap_ns: float,
) -> tuple[Module, ScheduleResult]:
    """Minimize wall time under ``O="wall"``: probe candidate periods on both
    sides of the requested clock with the heuristic scheduler, take the one
    whose span times achieved period is least, and solve once there under the
    caller's own scheduler settings. The requested clock is a reference, not a
    bound: a slower clock that unlocks a shallower operator row can win.
    ``cap_ns`` tops the ladder at the slowest period any device row is built
    for, past which nothing new unlocks; ``floor_ns`` is the register floor
    the aggressive anchor stands on. A kernel with no composed span has no
    wall time to compare and is refused. Returns the scheduled module and its
    result, with the probed curve published as ``ScheduleResult.sweep``."""
    margin = 1.0 - options.clock_margin

    def probe(period: float) -> SweepPoint:
        opts = replace(options, scheduler="heuristic", cycle_ns=period)
        result = run_schedule(top, make_module(), opts, prepass, allocate)
        fn = result.func(top)
        return SweepPoint(
            cycle_ns=period,
            achieved_ns=result.cycle_ns / margin,
            latency=fn.latency,
            latency_is_bound=fn.latency_is_bound,
        )

    asked = probe(options.cycle_ns)
    if asked.latency is None:
        raise RuntimeError(
            f"O='wall' compares span times period, and '{top}' publishes no "
            "span at the requested clock; add allo.assume trip bounds or "
            "choose a different objective"
        )
    points = [asked]
    best = asked.latency * asked.achieved_ns
    lo = 2.0 * floor_ns / margin if floor_ns > 0 else options.cycle_ns / 8.0
    hi = max(cap_ns, options.cycle_ns, lo)
    # Descending walk with the incumbent: the laxest candidate's span bounds
    # every candidate's from below (feasible sets only grow with the period),
    # so a period that laxest span cannot win at is skipped unprobed.
    floor_span = None
    for k in range(_WALL_RUNGS):
        period = hi * (lo / hi) ** (k / (_WALL_RUNGS - 1))
        if abs(period - options.cycle_ns) < 1e-9:
            continue
        if floor_span is not None and floor_span * period >= best:
            continue
        p = probe(period)
        assert p.latency is not None, (
            "a span is a property of the kernel's trip structure, which no "
            "probed period changes"
        )
        floor_span = floor_span if floor_span is not None else p.latency
        best = min(best, p.latency * p.achieved_ns)
        points.append(p)
    # Candidates that derate onto the same achieved period are one design;
    # keep the laxest ask of each.
    seen: set[float] = set()
    curve: list[SweepPoint] = []
    for p in sorted(points, key=lambda p: p.cycle_ns, reverse=True):
        if (key := round(p.achieved_ns, 3)) not in seen:
            seen.add(key)
            curve.append(p)
    # Fewer cycles breaks a wall tie: the shorter schedule at the slower
    # clock spends less on pipeline registers.
    winner = min(curve, key=lambda p: (p.latency * p.achieved_ns, p.latency))
    module = make_module()
    result = run_schedule(
        top, module, replace(options, cycle_ns=winner.cycle_ns), prepass, allocate
    )
    return module, replace(result, sweep=tuple(curve))
