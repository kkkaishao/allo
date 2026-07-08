# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver for the SDC ``allo-schedule`` pass.

``schedule(top, ...)`` prepares the kernel, runs ``allo-schedule`` with an
operator library, and returns the carried schedule as a dict. It replaces
hand-writing the pass-pipeline string: an :class:`OperatorLibrary` supplies both
the ``operator-library`` file and (via its declared frequency) the ``cycle-time``.
"""

from __future__ import annotations

import os
import tempfile

from pathlib import Path

from ..base import run_pipeline
from ..._mlir.schedule import collect_schedule_result
from .operator_library import OperatorLibrary


def _resolve_library(library):
    """Return ``(path_or_none, tmp_to_delete_or_none)`` for ``library``, which
    may be an :class:`OperatorLibrary`, a path-like, or ``None``."""
    if library is None:
        return None, None
    if isinstance(library, OperatorLibrary):
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="allo_oplib_")
        os.close(fd)
        library.to_yaml(path)
        return path, path
    return str(Path(library)), None  # a path-like, passed through as-is


def schedule(
    top,
    *,
    library=None,
    cycle_time=None,
    prepare=True,
    dump_region_graph=False,
    float_reassoc=False,
    accumulators=0,
):
    """Schedule ``top`` and return the result dict (see
    ``collect_schedule_result``). The schedule is also carried on the module as
    attributes.

    Args:
        top: a kernel (has ``.module``) or an MLIR module.
        library: an :class:`OperatorLibrary`, a path to a YAML library, or
            ``None`` for the built-in default.
        cycle_time: target clock period (ns); overrides the library's declared
            frequency.
        prepare: run the HLS preparation pipeline first (set ``False`` if the
            module is already lowered to affine form).
        dump_region_graph: print the coarse cross-region dependence graph.
        float_reassoc: permit reassociating floating-point reductions when
            rebalancing unrolled reduction chains (inexact; the fast-math knob).
        accumulators: rotate float reductions across this many accumulators so
            their II drops to ``ceil(latency / accumulators)`` (0 = off; set it
            to at least the reduction operator's latency for II=1).
    """
    module = getattr(top, "module", top)
    if prepare:
        # Shared HLS preparation (lowers allo IR to schedulable affine form).
        from ..vitis.core import HLS_PREPARE_PIPELINE

        run_pipeline(module, HLS_PREPARE_PIPELINE)

    if cycle_time is None and isinstance(library, OperatorLibrary):
        cycle_time = library.cycle_time()

    lib_path, tmp = _resolve_library(library)
    try:
        opts = []
        if cycle_time is not None:
            opts.append(f"cycle-time={cycle_time}")
        if lib_path is not None:
            opts.append(f"operator-library={lib_path}")
        if dump_region_graph:
            opts.append("dump-region-graph=true")
        schedule_pass = "allo-schedule{" + " ".join(opts) + "}"
        # Scheduling-path normalizations before the SDC solve (all deliberately
        # kept out of HLS_PREPARE_PIPELINE, since the Vitis path does its own):
        #   * flatten-perfect-loops -- coalesce constant-trip perfect nests so the
        #     whole nest pipelines at one II (canonicalize then folds the
        #     coalesced constant bound back into the loop);
        #   * if-hyperblock-conversion -- predicate affine.if / scf.if so no
        #     control flow remains inside a loop body;
        #   * cse -- fold the redundant loads/ops speculation introduces so they
        #     do not inflate the resource-bound II;
        #   * reassociate-reductions -- rebalance unrolled reduction chains so a
        #     loop-carried accumulator's recurrence spans one operator, not the
        #     whole unrolled chain.
        reassoc = (
            "reassociate-reductions{float-reassoc="
            f"{'true' if float_reassoc else 'false'}}}"
        )
        #   * rotate-reductions -- spread a float reduction across N accumulators
        #     so its recurrence spans N iterations and the II drops to 1.
        rotate = f"rotate-reductions{{accumulators={int(accumulators)}}}"
        pipeline = (
            "builtin.module(func.func(flatten-perfect-loops,canonicalize,"
            f"if-hyperblock-conversion,cse,{reassoc},{rotate},{schedule_pass}))"
        )
        run_pipeline(module, pipeline)
        return collect_schedule_result(module)
    finally:
        if tmp is not None:
            os.unlink(tmp)
