/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Registration here is strictly ADDITIVE on top of the upstream
 * RegisterEverything bundled in the same package: it touches only the `allo`
 * dialect, the Allo transform-dialect extension, and Allo-specific passes.
 * Re-registering the upstream dialects/passes would collide with
 * RegisterEverything (single shared MLIR / global registries) -- in particular
 * re-running the upstream pass/pipeline registration aborts with
 * "<pipeline> registered multiple times".
 */

#include "allo-c/Registration.h"

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include <mutex>

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Allo, allo, ::mlir::allo::AlloDialect)

void alloMlirRegisterAllDialects(MlirContext context) {
  DialectRegistry registry;
  registry.insert<allo::AlloDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->getOrLoadDialect<allo::AlloDialect>();
}

void alloMlirRegisterAllExtensions(MlirContext context) {
  DialectRegistry registry;
  allo::registerTransformDialectExtension(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void alloMlirRegisterAllPasses() {
  static std::once_flag once;
  std::call_once(once, [] {
    allo::registerConversionPasses();
    allo::registerTransformsPasses();
    allo::registerAlloLLVMLoweringPipeline();
  });
}
