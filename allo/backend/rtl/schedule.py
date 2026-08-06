# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling driver. The result it returns lives in `reports`."""

from __future__ import annotations

from ..base import run_pipeline
from ..._mlir.dialects.allo import run_sdc_scheduling

# re-exported through allo.backend.rtl so callers do not reach into _mlir
# pylint: disable-next=unused-import
from ..._mlir.dialects.allo import has_exact_scheduler

from .options import PrepassOptions, SchedulerOptions
from .reports.schedule import ScheduleResult

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
    how many copies of each operator a region builds, which only pays off under
    a binding that folds them.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
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
    part = f"propagate-partition{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={prepass.scalarize_threshold}}}"
    pipeline = (
        f"builtin.module(canonicalize,cse,func.func(raise-to-affine,cse,"
        f"raise-counted-while,{loops},"
        f"canonicalize,fold-if-statements,cse,{scalarize},"
        f"{reassoc},{rotate},narrow-demanded-bits),drop-trivial-func,"
        f"{part},func.func(assign-banks),canonicalize,cse)"
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
            options.cycle_ns,
            options.scheduler,
            options.budget,
            allocate,
            options.workers,
            options.seed,
        )
    finally:
        handler.detach()
    if result is None:
        raise RuntimeError(
            "An error occurred during scheduling process:\n" + "\n".join(diagnostics)
        )
    return ScheduleResult.from_json(result, options)
