/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Binding policies (pure decision). Compatibility is the exact MRT test from
// Reservation.h, since bind-after-schedule makes it a set intersection rather
// than a lifetime heuristic. See BindingPolicy.h.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/BindingPolicy.h"

#include "allo/Microarch/Reservation.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::allo::uarch {

std::vector<llvm::SmallVector<UnitId, 2>>
TrivialBinding::plan(const Datapath &) const {
  return {};
}

std::vector<llvm::SmallVector<UnitId, 2>>
GreedyShareBinding::plan(const Datapath &dp) const {
  std::vector<llvm::SmallVector<UnitId, 2>> groups;
  for (const RegionBlock &rb : dp.regions) {
    // Each bin is one physical unit's ops; a candidate joins the first bin of
    // its type whose every member is reservation-disjoint from it (left-edge).
    llvm::SmallVector<llvm::SmallVector<UnitId, 2>> bins;
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      auto ru = reservationOf(rb, u, u.boundOps.front().second);
      llvm::SmallVectorImpl<UnitId> *dest = nullptr;
      for (auto &bin : bins) {
        if (!sameOperatorType(dp.units[bin.front()], u))
          continue;
        bool ok = llvm::all_of(bin, [&](UnitId m) {
          const FuncUnit &mu = dp.units[m];
          return llvm::all_of(mu.boundOps,
                              [&](const std::pair<Operation *, unsigned> &bo) {
                                return reservationsDisjoint(
                                    reservationOf(rb, mu, bo.second), ru);
                              });
        });
        if (ok) {
          dest = &bin;
          break;
        }
      }
      if (dest)
        dest->push_back(uid);
      else
        bins.push_back({uid});
    }
    for (auto &bin : bins)
      if (bin.size() > 1)
        groups.emplace_back(bin.begin(), bin.end());
  }
  return groups;
}

std::unique_ptr<BindingPolicy> bindingPolicyFor(llvm::StringRef name) {
  if (name == "trivial")
    return std::make_unique<TrivialBinding>();
  if (name == "greedy-share")
    return std::make_unique<GreedyShareBinding>();
  return nullptr;
}

} // namespace mlir::allo::uarch
