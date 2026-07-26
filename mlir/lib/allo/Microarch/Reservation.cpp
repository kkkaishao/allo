/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Reservation.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cassert>

namespace mlir::allo::uarch {

Reservation reservationOf(const RegionBlock &region, const FuncUnit &unit,
                          unsigned residue) {
  Reservation r;
  r.region = region.id;
  // A fully-pipelined unit only holds the issue slot (its internal stages carry
  // distinct data), so it is contended for one cycle regardless of latency; a
  // non-pipelined unit stays busy for its whole latency.
  unsigned len = unit.pipelined ? 1 : std::max(1u, unit.latency);
  // Cyclic regions wrap the occupancy mod II: a window that crosses the II
  // boundary self-overlaps when latency > II, correctly marking the unit busy
  // every cycle. Acyclic regions run once on a straight timeline, with no wrap.
  unsigned mod =
      region.kind == RegionBlock::Kind::Cyclic ? region.ii.value_or(1) : 0;
  for (unsigned i = 0; i < len; ++i)
    r.cycles.push_back(mod ? (residue + i) % mod : residue + i);
  return r;
}

bool reservationsDisjoint(const Reservation &a, const Reservation &b) {
  if (a.region != b.region)
    return false; // cross-region sharing isn't modelled; treated as a
                  // conflict
  llvm::SmallDenseSet<unsigned, 8> cyclesA(a.cycles.begin(), a.cycles.end());
  return llvm::none_of(b.cycles,
                       [&](unsigned c) { return cyclesA.contains(c); });
}

bool sameOperatorType(const FuncUnit &a, const FuncUnit &b) {
  if (a.opType != b.opType || a.impl != b.impl || a.resultType != b.resultType)
    return false;
  // opType/impl/resultType alone under-specify the operator (operand widths,
  // predicate, or map can differ). The emitter builds from `repOp()`, so
  // reject divergent merges here; verifyBinding backstops it post-merge.
  Operation *oa = a.repOp();
  Operation *ob = b.repOp();
  return std::equal(oa->getOperandTypes().begin(), oa->getOperandTypes().end(),
                    ob->getOperandTypes().begin(),
                    ob->getOperandTypes().end()) &&
         oa->getAttr("predicate") == ob->getAttr("predicate") &&
         oa->getAttr("map") == ob->getAttr("map");
}

void verifyBinding(const Datapath &dp) {
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      // The emitter builds one physical unit from boundOps.front() (operand
      // widths, predicate, map). A merge that differs in these would miscompile
      // the non-front ops; this backstops that (vacuous under trivial binding).
      if (Operation *f = u.boundOps.empty() ? nullptr : u.repOp())
        for (const auto &bo : u.boundOps)
          assert(
              std::equal(bo.first->getOperandTypes().begin(),
                         bo.first->getOperandTypes().end(),
                         f->getOperandTypes().begin(),
                         f->getOperandTypes().end()) &&
              bo.first->getAttr("predicate") == f->getAttr("predicate") &&
              bo.first->getAttr("map") == f->getAttr("map") &&
              "shared unit binds semantically divergent ops (operand widths / "
              "compare predicate / apply map differ); emit uses "
              "repOp() and miscompiles the others");
      for (unsigned i = 0, e = u.boundOps.size(); i < e; ++i) {
        auto ri = reservationOf(rb, u, u.boundOps[i].second);
        for (unsigned j = i + 1; j < e; ++j) {
          auto rj = reservationOf(rb, u, u.boundOps[j].second);
          (void)ri;
          (void)rj;
          assert(reservationsDisjoint(ri, rj) &&
                 "binding hazard: two ops share a unit in the same cycle");
        }
      }
    }
}

} // namespace mlir::allo::uarch
