/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_C_REGISTRATION_H
#define ALLO_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Standard dialect registration handle for the `allo` dialect, usable with
/// `mlirDialectHandleRegisterDialect` / `mlirDialectHandleLoadDialect`.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Allo, allo);

/// Registers and loads every dialect Allo needs (including `allo`) into the
/// given context.
MLIR_CAPI_EXPORTED void alloMlirRegisterAllDialects(MlirContext context);

/// Registers and loads the transform dialect plus all Allo transform-dialect
/// extensions into the given context.
MLIR_CAPI_EXPORTED void alloMlirRegisterAllExtensions(MlirContext context);

/// Registers all Allo and upstream passes used by Allo with the global pass
/// registry so that they can be referenced from a textual pass pipeline.
MLIR_CAPI_EXPORTED void alloMlirRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_REGISTRATION_H
