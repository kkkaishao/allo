/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_BITANALYSIS_H
#define ALLO_SUPPORT_BITANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/KnownBits.h"

namespace mlir::allo {

/// The bits of \p v a forward walk proves constant. `llvm::KnownBits` carries
/// the bit algebra; this is the dispatch onto it. Unknown for a value it cannot
/// follow and past \p depth, which is always the safe answer: an unknown bit
/// only forfeits a conclusion, never reaches a wrong one.
///
/// \p v must be integer-typed; a caller holding an `index` has no width to
/// reason in and must not ask.
llvm::KnownBits knownBits(Value v, unsigned depth = 8);

/// Whether \p op RENAMES bits rather than computing them, so it reaches no cell
/// and costs no logic. Two shapes qualify:
///
///   * a shift by a LITERAL amount, which `comb` canonicalizes into an extract
///     / concat. The device's shift row describes a barrel shifter, which is
///     the delay a RUNTIME amount pays;
///   * an `or` / `xor` whose operands share no set bit, which CONCATENATES
///     rather than combines: every result bit takes one side while the other
///     contributes a constant zero, and those zeros are the ones a synthesizer
///     propagates.
///
/// Asked by both places a datapath node is priced -- the chaining solve and the
/// binder's slack -- so they cannot disagree about what the schedule left.
bool isBitRename(Operation *op);

} // namespace mlir::allo

#endif // ALLO_SUPPORT_BITANALYSIS_H
