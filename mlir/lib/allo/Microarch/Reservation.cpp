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
  // A pipelined unit holds only the issue slot, its stages carrying distinct
  // data; a non-pipelined unit stays busy for its whole latency.
  unsigned len = unit.pipelined ? 1 : std::max(1u, unit.latency);
  // Cyclic regions wrap occupancy mod II, so a latency at or above II marks the
  // unit busy on every residue. Acyclic regions run on a straight timeline.
  unsigned mod =
      region.kind == RegionBlock::Kind::Cyclic ? region.ii.value_or(1) : 0;
  for (unsigned i = 0; i < len; ++i)
    r.cycles.push_back(mod ? (residue + i) % mod : residue + i);
  return r;
}

bool reservationsDisjoint(const Reservation &a, const Reservation &b) {
  if (a.region != b.region)
    return false; // cross-region sharing isn't modelled; treated as a conflict
  llvm::SmallDenseSet<unsigned, 8> cyclesA(a.cycles.begin(), a.cycles.end());
  return llvm::none_of(b.cycles,
                       [&](unsigned c) { return cyclesA.contains(c); });
}

void verifyBinding(const Datapath &dp) {
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      // The emitter builds one operator from the unit's identity, so a bound op
      // of any other identity would be miscompiled.
      for (const auto &bo : u.boundOps)
        assert(operatorIdentity(cast<dcp::DCPathComputeOp>(bo.first)) ==
                   u.identity &&
               "shared unit binds an op of a different operator identity");
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
