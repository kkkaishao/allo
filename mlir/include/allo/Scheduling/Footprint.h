/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_FOOTPRINT_H
#define ALLO_SCHEDULING_FOOTPRINT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::func {
class CallOp;
} // namespace mlir::func

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

/// Fold a synchronous sub-kernel call's footprint into \p s, keyed by the
/// CALLER's operand roots. `summarizeOp` cannot see through an opaque call, so
/// it marks every memref operand read+write+nonAffine. That makes any two
/// calls sharing an array conflict, even two readers of one input. This looks
/// INTO the callee instead and records, per parameter, the direction plus the
/// callee's own affine access ops (recursing through nested calls). Two calls
/// sharing an array can then be proven read-only or element-disjoint by
/// `callFootprintConflict` and left unordered, i.e. free to overlap.
///
/// Returns false when a construct defeats the summary (an unresolvable /
/// external callee, a call cycle, a view operand whose index space is offset
/// from its root's). \p s may then hold a partial record; it is subsumed by the
/// conservative `summarizeOp` marks the caller applies instead, so their union
/// stays conservative.
bool summarizeCall(func::CallOp call, Summary &s);

/// The ordering hazard between an EARLIER access \p a and a LATER access \p b
/// on a shared root, for accesses recorded by `summarizeCall`: their affine ops
/// live in the CALLEES, each naming its own parameter rather than one common
/// memref Value, so `footprintConflict`'s `MemRefAccess`-pair test does not
/// apply. Same classification, but disjointness compares polyhedral REGIONS
/// over the index space the parameters share with the array.
Conflict callFootprintConflict(const Access &a, const Access &b);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_FOOTPRINT_H
