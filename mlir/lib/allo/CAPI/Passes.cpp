/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Passes.h"

#include "allo/Translation/VivadoHLSEmitter.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

MlirLogicalResult alloEmitVivadoHLS(MlirModule module, bool enableApFloat,
                                    unsigned indexWidth, bool withLocation,
                                    MlirStringRef top,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(allo::emitVivadoHLS(unwrap(module), stream, enableApFloat,
                                  indexWidth, withLocation, unwrap(top)));
}
