/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Passes.h"
#include "allo/Microarch/EmitDriver.h"
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

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
                                       MlirStringRef top, double cycleTime,
                                       MlirStringCallback callback,
                                       void *userData) {
  llvm::StringMap<std::string> interfaces;
  if (failed(allo::uarch::emitDatapathToHW(unwrap(module), unwrap(binding),
                                           unwrap(top), (float)cycleTime,
                                           interfaces)))
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

MlirLogicalResult
alloRunSDCSchedulingPipeline(MlirModule module, MlirStringRef top,
                             float cycleTime, MlirStringRef scheduler,
                             double budget, bool allocate,
                             MlirStringCallback callback, void *userData) {
  ModuleOp mod = unwrap(module);
  StringRef topName = unwrap(top);
  StringRef schedulerName = unwrap(scheduler);
  std::optional<allo::SchedulerKind> kind =
      allo::parseSchedulerKind(schedulerName);
  if (!kind) {
    allo::logging::error(allo::logging::Stage::Sched, mod)
        << "Unknown scheduler '" << schedulerName
        << "'; expected \"heuristic\", \"exact\" or \"exact-chaining\"";
    return mlirLogicalResultFailure();
  }
  // The target clock period: the option, else a 5.0 ns default. Resolved once
  // here, since both halves price against it and a second copy of the default
  // is a second answer to what the target frequency is.
  float cycleTimeNs = cycleTime > 0.0f ? cycleTime : 5.0f;
  // Same rule for what one exact solve may spend: the option, else the default,
  // resolved once so no second copy of it exists downstream.
  allo::SchedulerOptions opts{
      *kind, budget > 0.0 ? budget : allo::kDefaultSolveBudget, allocate};
  if (failed(allo::runPreScheduleVerification(mod, topName, cycleTimeNs)))
    return mlirLogicalResultFailure();
  // The solved schedule travels between the two halves in memory rather than as
  // attributes on the IR, so it lives exactly as long as this pipeline does and
  // its `Operation *` keys cannot outlive the ops they name.
  allo::ScheduleModel model;
  if (failed(allo::runSDCScheduler(mod, topName, cycleTimeNs, opts, model)))
    return mlirLogicalResultFailure();
  allo::runPostScheduleConversion(mod, model);
  // The report the reify recorded, which is the only part of the model that
  // outlives the pipeline.
  std::string report = model.toJSON();
  callback(MlirStringRef{report.data(), report.size()}, userData);
  return mlirLogicalResultSuccess();
}

bool alloHasExactScheduler() { return allo::hasExactScheduler(); }
