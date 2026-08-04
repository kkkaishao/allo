/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
#define ALLO_SCHEDULING_DEPENDENCEANALYSIS_H

#include "allo/Scheduling/RegionGraph.h"

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace mlir::allo {

/// A constant range `[lb, ub]` (inclusive) on an SSA value, either endpoint
/// open when unknown. Distilled from the `allo.assume.ssa` value facts.
struct AssumedRange {
  std::optional<int64_t> lb;
  std::optional<int64_t> ub;
};

/// The dependence distance carried by the counted loop at 1-based nesting depth
/// \p level among a dependence's shared enclosing loops, projected from its
/// components (outermost -> innermost). Sets \p drop when an OUTER loop carries
/// the dependence, whose sequential execution already satisfies it. Sets \p
/// valid = false when \p level is deeper than the shared loop nest. A
/// loop-independent dependence has no components and maps to distance 0.
int64_t
carriedDistanceAtLevel(llvm::ArrayRef<affine::DependenceComponent> comps,
                       unsigned level, bool &drop, bool &valid);

/// Memory + stream dependence analysis over a `func.func`. Affine memref
/// accesses and Allo stream get/put ops are recorded into one
/// MemoryDependenceResult that scheduling problem construction consumes.
class DependenceAnalysis {
public:
  explicit DependenceAnalysis(func::FuncOp funcOp);

  /// Dependences whose destination is \p op (may be empty).
  llvm::ArrayRef<circt::analysis::MemoryDependence>
  getDependences(Operation *op) {
    return results[op];
  }

  /// Redirect dependences of/to \p oldOp onto \p newOp (used when affine
  /// structures are lowered to their memref/std equivalents).
  void replaceOp(Operation *oldOp, Operation *newOp);

  /// The coarse cross-region dependence graph over the whole func, built and
  /// cached on first use. Analysis only. It does not affect scheduling.
  const RegionGraph &getRegionGraph();

  /// The constant range a value is known to lie in, distilled from the
  /// `allo.assume.ssa` facts, or nullopt when no such fact constrains it.
  std::optional<AssumedRange> getAssumedRange(Value v) const {
    auto it = assumedRanges.find(v);
    return it == assumedRanges.end() ? std::nullopt : std::optional(it->second);
  }

  /// All distilled value ranges, keyed by SSA value.
  const llvm::DenseMap<Value, AssumedRange> &getAssumedRanges() const {
    return assumedRanges;
  }

  /// Whether the polyhedral test cannot model \p op's access.
  bool isNonPolyhedral(Operation *op) const {
    return nonPolyhedral.contains(op);
  }

private:
  func::FuncOp func;
  circt::analysis::MemoryDependenceResult results;
  std::optional<RegionGraph> regionGraph;
  llvm::DenseMap<Value, AssumedRange> assumedRanges;
  llvm::SmallDenseSet<Operation *> nonPolyhedral;
};

/// Whether \p op carries a memory effect this analysis does not model
/// (`memref.copy`, `atomic_rmw`, `dma_*`). Such an op joins no access list, so
/// its dependences would be DROPPED and anything scheduled around it may race;
/// `verify-rtl-legality` rejects one before scheduling. The list here is the
/// complement of the access kinds the constructor's walk collects, so the two
/// must be edited together.
bool isUnmodeledMemoryAccess(Operation *op);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
