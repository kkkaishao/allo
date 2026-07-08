/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Attribute names that carry an SDC schedule on the IR (the "carrier"). The
// schedule is a decision annotated onto the real ops; a separate stage
// materializes/exports it. See drafts/sdc-scheduling-design.md.
//===----------------------------------------------------------------------===//

#ifndef ALLO_SCHEDULING_SCHEDULEATTRS_H
#define ALLO_SCHEDULING_SCHEDULEATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::allo::sched {

/// Per-op: start time (cycle), relative to the op's scheduling region.
constexpr llvm::StringLiteral kStartTimeAttr = "allo.sched.t";
/// Per-op: id of the scheduling region the op belongs to.
constexpr llvm::StringLiteral kRegionIdAttr = "allo.sched.region";
/// Func-level: array of per-region dictionaries (see the kRegionKey* fields).
constexpr llvm::StringLiteral kRegionsAttr = "allo.sched.regions";

// Keys of a per-region dictionary in the kRegionsAttr array.
constexpr llvm::StringLiteral kRegionKeyId = "id";
constexpr llvm::StringLiteral kRegionKeyKind = "kind";     // "cyclic" | "acyclic"
constexpr llvm::StringLiteral kRegionKeyII = "ii";         // omitted for acyclic
constexpr llvm::StringLiteral kRegionKeyLength = "length"; // number of cycle slots
constexpr llvm::StringLiteral kRegionKeyOrder = "order";   // program order in func

} // namespace mlir::allo::sched

#endif // ALLO_SCHEDULING_SCHEDULEATTRS_H
