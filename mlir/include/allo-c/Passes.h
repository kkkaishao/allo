/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_C_PASSES_H
#define ALLO_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits Vivado HLS C++ for `module`, streaming the result through `callback`.
/// Returns failure if emission fails (in which case `callback` is not invoked).
/// `top` names the top function (emitted with `extern "C"` linkage and carrying
/// the global array_partition pragmas); pass an empty string for none.
MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitVivadoHLS(MlirModule module, bool enableApFloat, unsigned indexWidth,
                  bool withLocation, MlirStringRef top,
                  MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloDumpRegionDependenceAnalysis(MlirModule module, MlirStringRef funcName,
                                 MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitVerilog(MlirModule module, MlirStringCallback callback, void *userData);

/// Lowers every scheduled function in `module` to structural `hw.module`s (the
/// free-function form of `allo-datapath-to-hw`, mutating `module` in place),
/// and streams back through `callback` a single JSON object mapping each
/// emitted module's name to its port-interface JSON (the cosim manifest, with
/// concrete field names). `binding` names the resource-binding policy. Returns
/// failure (callback not invoked) if emission fails.
MLIR_CAPI_EXPORTED MlirLogicalResult alloEmitDatapathToHW(
    MlirModule module, MlirStringRef binding, MlirStringRef top,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitSplitVerilog(MlirModule module, MlirStringRef directory);

/// Schedules `top` and reifies the schedule into `module` in place as
/// `allo.dcp.*` ops, streaming back through `callback` the schedule report as
/// JSON: per-func regions with their per-op start times, plus the per-region
/// and whole-kernel latency. `scheduler` names the solver that settles the
/// resource half of every scheduling problem: "heuristic" (the SDC simplex plus
/// greedy placement) or "exact" (CP-SAT, only in a build with OR-Tools).
/// Returns failure (callback not invoked) if any phase fails.
MLIR_CAPI_EXPORTED MlirLogicalResult alloRunSDCSchedulingPipeline(
    MlirModule module, MlirStringRef top, float cycleTime,
    MlirStringRef scheduler, MlirStringCallback callback, void *userData);

/// Whether this build accepts `scheduler = "exact"`, i.e. links OR-Tools. The
/// option exists in both distributions, so this is what tells them apart.
MLIR_CAPI_EXPORTED bool alloHasExactScheduler(void);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_PASSES_H
