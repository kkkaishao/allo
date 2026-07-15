/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Passes.h"
#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/Utils.h"

#include "allo/Translation/VerilogEmitter.h"
#include "allo/Translation/VivadoHLSEmitter.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "llvm/ADT/StringMap.h"

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

MlirLogicalResult alloDumpRegionDependenceAnalysis(MlirModule module,
                                                   MlirStringRef funcName,
                                                   MlirStringCallback callback,
                                                   void *userData) {
  FailureOr<std::string> result = allo::dumpRegionDependenceAnaysis(
      unwrap(module), std::string(funcName.data, funcName.length));
  if (failed(result))
    return mlirLogicalResultFailure();
  callback(MlirStringRef{result->data(), result->size()}, userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult alloEmitVerilog(MlirModule module,
                                  MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(allo::emitVerilog(unwrap(module), stream));
}

MlirLogicalResult alloEmitSplitVerilog(MlirModule module,
                                       MlirStringRef directory) {
  return wrap(allo::emitSplitVerilog(unwrap(module), unwrap(directory)));
}

MlirLogicalResult alloEmitDatapathToHW(MlirModule module, MlirStringRef binding,
                                       MlirStringRef top,
                                       MlirStringCallback callback,
                                       void *userData) {
  llvm::StringMap<std::string> interfaces;
  if (failed(allo::uarch::emitDatapathToHW(unwrap(module), unwrap(binding),
                                           unwrap(top), interfaces)))
    return mlirLogicalResultFailure();
  // Combine the per-module interface JSON into one object keyed by module name.
  // Each value is already valid JSON, so it is embedded verbatim; module names
  // are plain identifiers (no JSON-escaping needed).
  std::string out = "{";
  bool first = true;
  for (const auto &kv : interfaces) {
    if (!first)
      out += ',';
    first = false;
    out += '"';
    out += kv.first();
    out += "\":";
    out += kv.second;
  }
  out += '}';
  callback(MlirStringRef{out.data(), out.size()}, userData);
  return mlirLogicalResultSuccess();
}
