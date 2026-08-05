# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling driver. The result it returns lives in `reports`."""

from __future__ import annotations

from ..base import run_pipeline
from ..._mlir.dialects.allo import run_sdc_scheduling

# re-exported through allo.backend.rtl so callers do not reach into _mlir
# pylint: disable-next=unused-import
from ..._mlir.dialects.allo import has_exact_scheduler

from .reports.compiler import ScheduleSettings
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


def run_schedule(top, module, settings: ScheduleSettings) -> ScheduleResult:
    """Schedule ``top`` under ``settings`` and return the :class:`ScheduleResult`.
    ``module`` is rewritten in place, left holding the ``allo.dcp.*`` ops the
    schedule reifies into. Operator/device timing is read from the ``dcp.device``
    / ``dcp.operator`` ops injected into ``module`` before this call. The knobs
    are the fields of :class:`ScheduleSettings`.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    reassoc = (
        "reassociate-reductions{float-reassoc="
        f"{'true' if settings.float_reassoc else 'false'}}}"
    )
    rotate = f"rotate-reductions{{accumulators={int(settings.accumulators)}}}"
    loops = (
        "loop-canonicalization{"
        f"unroll-under-pipeline={'true' if settings.unroll_under_pipeline else 'false'} "
        f"perfectize={'true' if settings.perfectize else 'false'}}}"
    )
    part = f"propagate-partition{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={settings.scalarize_threshold}}}"
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
            module,
            top,
            settings.cycle_time_ns,
            settings.scheduler,
            settings.budget or 0.0,
            settings.allocate,
        )
    finally:
        handler.detach()
    if result is None:
        raise RuntimeError(
            "An error occurred during scheduling process:\n" + "\n".join(diagnostics)
        )
    return ScheduleResult.from_json(result, settings)
