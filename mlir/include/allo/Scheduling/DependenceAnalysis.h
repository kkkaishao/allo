/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
#define ALLO_SCHEDULING_DEPENDENCEANALYSIS_H

#include "allo/Scheduling/RegionGraph.h"

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <optional>

namespace mlir::allo {

/// Memory + stream dependence analysis over a `func.func`. Mirrors CIRCT's
/// MemoryDependenceAnalysis for affine memref accesses and additionally
/// understands Allo stream get/put ops (streams are FIFOs; same-FIFO accesses
/// are serialized by a recurrence). Both flavors are recorded into one
/// MemoryDependenceResult that scheduling problem construction consumes
/// uniformly. Lifted out of the old convert-loop-to-schedule pass.
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

  /// The coarse cross-region dependence graph over the whole func (built and
  /// cached on first use). Analysis only -- does not affect scheduling.
  const RegionGraph &getRegionGraph();

private:
  func::FuncOp func;
  circt::analysis::MemoryDependenceResult results;
  std::optional<RegionGraph> regionGraph;
};

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_DEPENDENCEANALYSIS_H
