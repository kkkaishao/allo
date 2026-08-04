/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ScheduleModel.h"

#include "allo/IR/AlloOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::dcp;

namespace {
// The arith mnemonic an IP operator's ABSTRACT kind (`add`/`div`/...) came
// from. IP compute is always floating-point, and an unmapped kind (a cast, say)
// already reads as its own mnemonic.
StringRef ipMnemonic(StringRef kind) {
  return llvm::StringSwitch<StringRef>(kind)
      .Case("add", "addf")
      .Case("sub", "subf")
      .Case("mul", "mulf")
      .Case("div", "divf")
      .Case("rem", "remf")
      .Case("cmp", "cmpf")
      .Case("neg", "negf")
      .Default(kind);
}

// The mnemonic a scheduled op is reported under: the source op it came from
// rather than the dcp op standing for it. \p kinds maps a `dcp.operator` symbol
// to its abstract kind.
std::string opKind(Operation *op, const llvm::StringMap<StringRef> &kinds) {
  if (auto compute = dyn_cast<DCPathComputeOp>(op)) {
    if (std::optional<StringRef> sym = compute.getOpType())
      return ipMnemonic(kinds.lookup(*sym)).str();
    assert(compute.getCombKind() && "a compute takes one realization path");
    return stringifyCombOpKindEnum(*compute.getCombKind()).str();
  }
  // A memory access reads as its bare mnemonic; anything else keeps the dcp
  // qualifier, which tells a `dcp.instance` from a `stream.get`.
  StringRef name = op->getName().getStringRef();
  if (isa<DCPathLoadOp, DCPathStoreOp>(op))
    return name.rsplit('.').second.str();
  return name.split('.').second.str();
}
} // namespace

void mlir::allo::ScheduleModel::record(ModuleOp module) {
  // A `dcp.operator` symbol to its abstract kind, for `opKind` below.
  llvm::StringMap<StringRef> kinds;
  for (DCPathOperatorOp op : module.getOps<DCPathOperatorOp>())
    kinds[op.getSymName()] = op.getKind();

  report.clear();
  for (DCPathModuleOp fn : module.getOps<DCPathModuleOp>()) {
    FuncReport f;
    f.name = fn.getSymName().str();
    if (std::optional<uint64_t> latency = fn.getLatency())
      f.latency = (int64_t)*latency;
    f.latencyBound = f.latency && fn.getLatencyBound();
    f.determinacy = stringifyDeterminacyEnum(fn.getDeterminacy()).str();

    // Pre-order, so a region is reported before the ones it nests and `order`
    // is program order.
    fn->walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto interface = dyn_cast<DCPathRegionOpInterface>(op);
      if (!interface)
        return;
      RegionReport r;
      r.order = (int64_t)f.regions.size();
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
        if (isa<DCPathRegionOpInterface>(p))
          ++r.depth;
      op->walk([&](Operation *nested) {
        if (nested == op || !isa<DCPathRegionOpInterface>(nested))
          return WalkResult::advance();
        r.container = true;
        return WalkResult::interrupt();
      });

      if (auto pipeline = dyn_cast<DCPathPipelineOp>(op)) {
        r.kind = "cyclic";
        // `ii` is absent for a data-dependent sequential wrapper (an enclosed
        // dynamic-trip loop); a pipelined region always has one.
        if (std::optional<uint64_t> ii = pipeline.getIi())
          r.ii = (int64_t)*ii;
        if (std::optional<uint64_t> length = pipeline.getLength())
          r.length = (int64_t)*length;
        r.conditional = pipeline.isWhileLoop();
      } else if (auto sequential = dyn_cast<DCPathSequentialOp>(op)) {
        r.kind = "acyclic";
        if (std::optional<uint64_t> length = sequential.getLength())
          r.length = (int64_t)*length;
      } else {
        // A control guard: it selects the active data path and carries no
        // compute of its own; its branch children are reported in turn.
        assert(isa<DCPathSelectOp>(op) && "a region is a pipeline, a "
                                          "sequential or a select");
        r.kind = "guard";
        r.conditional = true;
      }
      if (std::optional<uint64_t> trip = interface.getTrip())
        r.trip = (int64_t)*trip;
      if (std::optional<uint64_t> drain = interface.getDrain())
        r.drain = (int64_t)*drain;
      if (std::optional<uint64_t> latency = interface.getLatency())
        r.latency = (int64_t)*latency;
      r.latencyBound = r.latency && interface.getLatencyBound();
      if (std::optional<DeterminacyEnum> det = interface.getDeterminacy())
        r.determinacy = stringifyDeterminacyEnum(*det).str();

      // The DIRECT children only: a nested region's ops belong to that region,
      // so no op is reported twice.
      for (Operation &child : op->getRegion(0).front()) {
        auto start = child.getAttrOfType<IntegerAttr>("start");
        if (!start)
          continue;
        ScheduledOpReport o;
        o.kind = opKind(&child, kinds);
        o.start = start.getInt();
        if (auto compute = dyn_cast<DCPathComputeOp>(&child))
          if (std::optional<StringRef> sym = compute.getOpType())
            if (kinds.count(*sym))
              o.impl = sym->str();
        if (auto z = child.getAttrOfType<FloatAttr>("z"))
          o.z = (float)z.getValueAsDouble();
        r.ops.push_back(std::move(o));
      }
      f.regions.push_back(std::move(r));
    });
    report.push_back(std::move(f));
  }
}

std::string mlir::allo::ScheduleModel::toJSON() const {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  Array funcs;
  for (const FuncReport &f : report) {
    Array regions;
    for (const RegionReport &r : f.regions) {
      Array ops;
      for (const ScheduledOpReport &o : r.ops) {
        Object entry{{"kind", o.kind}, {"t", o.start}};
        if (!o.impl.empty())
          entry["impl"] = o.impl;
        if (o.z)
          entry["z"] = (double)*o.z;
        ops.push_back(std::move(entry));
      }
      Object entry{{"kind", r.kind},
                   {"order", r.order},
                   {"depth", r.depth},
                   {"container", r.container},
                   {"conditional", r.conditional},
                   {"latency_bound", r.latencyBound},
                   {"ops", std::move(ops)}};
      // A number the region does not have is an absent KEY, never a null.
      if (r.ii)
        entry["ii"] = *r.ii;
      if (r.trip)
        entry["trip"] = *r.trip;
      if (r.length)
        entry["length"] = *r.length;
      if (r.drain)
        entry["drain"] = *r.drain;
      if (r.latency)
        entry["latency"] = *r.latency;
      if (!r.determinacy.empty())
        entry["determinacy"] = r.determinacy;
      regions.push_back(std::move(entry));
    }
    Object entry{{"name", f.name},
                 {"latency_bound", f.latencyBound},
                 {"regions", std::move(regions)}};
    if (f.latency)
      entry["latency"] = *f.latency;
    if (!f.determinacy.empty())
      entry["determinacy"] = f.determinacy;
    funcs.push_back(std::move(entry));
  }

  Array solveEntries;
  for (const SolveReport &s : solves) {
    Object entry{{"func", s.func},
                 {"where", s.where},
                 {"kind", s.kind},
                 {"ops", s.ops},
                 {"limited_ops", s.limitedOps},
                 {"ms", s.millis}};
    if (s.ii)
      entry["ii"] = *s.ii;
    if (s.allocatedOps) {
      entry["allocated_ops"] = s.allocatedOps;
      entry["allocated_units"] = s.allocatedUnits;
    }
    solveEntries.push_back(std::move(entry));
  }

  Value root =
      Object{{"funcs", std::move(funcs)}, {"solves", std::move(solveEntries)}};
  std::string s;
  llvm::raw_string_ostream os(s);
  os << root;
  return s;
}
