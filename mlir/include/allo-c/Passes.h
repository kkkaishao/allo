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
MLIR_CAPI_EXPORTED MlirLogicalResult alloEmitVivadoHLS(
    MlirModule module, bool enableApFloat, unsigned indexWidth,
    bool withLocation, MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_PASSES_H
