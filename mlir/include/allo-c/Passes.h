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

// NOTE: canonicalize / lower-to-llvm are intentionally NOT exposed here.
// General pass pipelines (incl. the registered `allo-lower-to-llvm` pipeline)
// run through upstream `mlir.passmanager.PassManager`, so no downstream wrapper
// is maintained. Vivado HLS emission stays because it is a translation, not a
// pass.

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
/// and streams back -- through `callback` -- a single JSON object mapping each
/// emitted module's name to its port-interface JSON (the cosim manifest, with
/// concrete field names). `binding` names the resource-binding policy. Returns
/// failure (callback not invoked) if emission fails.
MLIR_CAPI_EXPORTED MlirLogicalResult alloEmitDatapathToHW(
    MlirModule module, MlirStringRef binding, MlirStringRef top,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitSplitVerilog(MlirModule module, MlirStringRef directory);
#ifdef __cplusplus
}
#endif

#endif // ALLO_C_PASSES_H
