/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_ALIASANALYSIS_H
#define ALLO_SUPPORT_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir::allo {

/// Peel to the STORAGE IDENTITY of a buffer or stream: the one definition every
/// access to it must agree on. Peels view-like ops (subview / cast /
/// reinterpret_cast / view), and on DCP IR the region results and pipeline
/// iter-args that forward a buffer out of the region that allocated it.
/// Identity when \p v is already a root; distinct roots are assumed
/// non-aliasing (the Allo frontend has no pointers).
Value resolveRoot(Value v);

/// `resolveRoot`'s disjointness assumption as an `AliasAnalysis`
/// implementation: memrefs with distinct roots are `NoAlias`. It only ever ADDS
/// that answer, since equal roots and non-memref pairs come back `MayAlias` and
/// fall through to whichever implementation is asked next. `LocalAliasAnalysis`
/// cannot state it on its own: two distinct entry-block arguments may be the
/// same buffer as far as it knows.
struct DistinctRootAliasAnalysis {
  AliasResult alias(Value lhs, Value rhs);
  /// Nothing to add: this analysis is about storage identity, not effects.
  ModRefResult getModRef(Operation *, Value) {
    return ModRefResult::getModAndRef();
  }
};

/// An `AliasAnalysis` over \p scope carrying Allo's aliasing contract on top of
/// MLIR's local one. Every Allo caller of an MLIR utility taking an
/// `AliasAnalysis` builds it here, so a bare `AliasAnalysis` in this tree is a
/// defect.
AliasAnalysis alloAliasAnalysis(Operation *scope);

} // namespace mlir::allo

#endif // ALLO_SUPPORT_ALIASANALYSIS_H
