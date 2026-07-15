/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/Scheduling/RegionGraph.h" // buildAndSortCallsiteGraph
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::allo {
#define GEN_PASS_DEF_PROPAGATEPARTITIONPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// The `allo.part` on a value's carrier: its defining op (a `memref.alloc`),
// else the enclosing function's argument attrs if it is a parameter. Mirrors
// `MemoryModel`'s `carrierAttr`, which is how the scheduler and emitter read
// the same attribute back.
static PartitionAttr partitionOn(Value memref) {
  if (Operation *def = memref.getDefiningOp())
    return def->getAttrOfType<PartitionAttr>(kPartitionAttr);
  if (auto arg = dyn_cast<BlockArgument>(memref))
    if (auto fn = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()))
      return fn.getArgAttrOfType<PartitionAttr>(arg.getArgNumber(),
                                                kPartitionAttr);
  return {};
}

struct PropagatePartitionPass
    : public allo::impl::PropagatePartitionPassBase<PropagatePartitionPass> {
  using PropagatePartitionPassBase::PropagatePartitionPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto topFunc = module.lookupSymbol<func::FuncOp>(top);
    if (!topFunc) {
      error(Stage::Prep, module) << "Top function '" << top << "' not found";
      return signalPassFailure();
    }
    // Callsites, callees-before-callers (the scheduler's own order). Reversed,
    // that is callers-before-callees: a callee's parameters carry their final
    // partition before the calls *inside* it are visited, so a partition
    // propagates the whole depth of the call graph in one pass.
    auto orderOr = buildAndSortCallsiteGraph(topFunc);
    if (failed(orderOr))
      return signalPassFailure();

    SymbolTableCollection syms;
    for (Operation *op : llvm::reverse(*orderOr)) {
      auto call = cast<func::CallOp>(op);
      auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        continue;
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
        if (!isa<MemRefType>(actual.getType()))
          continue;
        PartitionAttr part = partitionOn(actual);
        if (!part)
          continue;
        auto have = callee.getArgAttrOfType<PartitionAttr>(k, kPartitionAttr);
        if (have && have != part) {
          error(Stage::Prep, call)
              << "Array partitioning conflict detected on parameter " << k
              << " ; a same kernel cannot have incompatible partitions at "
                 "different callsites: "
              << "callsite '" << call.getCallee() << "' passes " << part
              << " but another callsite passes " << have;
          return signalPassFailure();
        }
        callee.setArgAttr(k, kPartitionAttr, part);
      }
    }
  }
};

} // namespace
