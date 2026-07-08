/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Memory / stream footprint summaries. A `Summary` aggregates, per memref root,
// whether a subtree reads / writes it (and the affine access ops, for
// polyhedral disjointness), plus the stream FIFOs it touches. Shared by the
// coarse cross-region graph (`RegionGraph`) and the per-level hierarchical
// analysis.
//===----------------------------------------------------------------------===//

#ifndef ALLO_SCHEDULING_FOOTPRINT_H
#define ALLO_SCHEDULING_FOOTPRINT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::allo {

/// Per-root access summary over a subtree.
struct Access {
  bool reads = false;
  bool writes = false;
  bool nonAffine =
      false; // >= 1 non-affine access (defeats sub-range refinement)
  llvm::SmallVector<Operation *> affine; // the affine load/store ops
};

/// Memory + stream footprint of a subtree.
struct Summary {
  llvm::DenseMap<Value, Access> mem; // memref root -> access
  llvm::DenseSet<Value> streams;     // stream roots touched (get or put)
};

/// Fold one op's memory / stream effect into \p s.
void summarizeOp(Operation *op, Summary &s);

/// Whether two accesses provably touch DISJOINT elements of a shared root (both
/// all-affine and no write-involving pair's polyhedral footprints intersect);
/// conservatively false otherwise.
bool footprintsDisjoint(const Access &ai, const Access &aj);

/// The ordering-hazard kind between an EARLIER access `a` and a LATER access
/// `b` on a shared memref root (program order a -> b).
enum class Conflict { None, RAW, WAR, WAW };

/// Classify the conflict on a shared root: `None` when both accesses are
/// read-only or their footprints are provably disjoint, otherwise the hazard
/// kind. Shared by the coarse cross-region graph and the per-level analysis.
Conflict footprintConflict(const Access &a, const Access &b);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_FOOTPRINT_H
