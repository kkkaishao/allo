/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Modulo-reservation-table (MRT) oracle: the exact compatibility judgment
// binding is built on. Under a fixed SDC/modulo schedule every op's occupancy
// of its functional unit is a compile-time constant (its issue residue mod II,
// or its absolute start in an acyclic region), so "can these two ops share one
// unit?" is an exact set-intersection test rather than a lifetime heuristic.
//
// This is the constraint side of binding (what may legally share); the
// objective side (what sharing costs) needs the operator area axis, which is
// separate.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_RESERVATION_H
#define ALLO_MICROARCH_RESERVATION_H

#include "allo/Microarch/Datapath.h"

namespace mlir::allo::uarch {

/// The resource cycles one bound op occupies on its functional unit, within its
/// region's schedule. For a pipelined unit only the issue slot is contended (a
/// single cycle); a non-pipelined unit is busy for its whole latency. Cyclic
/// regions count residues mod II (a wrapped window); acyclic regions count
/// absolute cycles (no wrap).
struct Reservation {
  RegionId region = 0;
  llvm::SmallVector<unsigned, 4> cycles; // occupied resource cycles
};

/// The reservation of an op bound to \p unit at issue \p residue in \p region.
/// \p residue is the value the binder already stored in FuncUnit::boundOps
/// (start mod II for cyclic, absolute start for acyclic).
Reservation reservationOf(const RegionBlock &region, const FuncUnit &unit,
                          unsigned residue);

/// Whether two reservations may coexist on one shared unit: their occupied
/// cycles must not intersect. Different regions conservatively conflict --
/// binding is within a region only (cross-region sharing is unsupported).
/// Combined with an operator-type match, this is the full share-compatibility
/// predicate a policy tests before merging two ops onto one unit.
bool reservationsDisjoint(const Reservation &a, const Reservation &b);

/// Whether two units realize the SAME operator type, a precondition for merging
/// their ops onto one physical unit: identical mnemonic + realization + result
/// type (an exact match; a covering ALU that realizes a superset is not
/// shared). Combined with `reservationsDisjoint`, this is the full
/// share-compatibility test.
bool sameOperatorType(const FuncUnit &a, const FuncUnit &b);

/// Assert the binding is legal: no two ops bound to the same unit contend for
/// it in the same resource cycle. A dev-time invariant (fail loudly) that a
/// buggy sharing policy trips immediately; vacuous under the trivial binding
/// (one op per unit).
void verifyBinding(const Datapath &dp);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_RESERVATION_H
