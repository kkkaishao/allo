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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo::uarch {

std::vector<llvm::SmallVector<UnitId, 2>>
TrivialBinding::plan(const Datapath &, const BindingContext &) const {
  return {};
}

std::vector<llvm::SmallVector<UnitId, 2>>
GreedyShareBinding::plan(const Datapath &dp, const BindingContext &ctx) const {
  double level = muxLevelDelay(ctx.lib);
  std::vector<llvm::SmallVector<UnitId, 2>> groups;
  for (const RegionBlock &rb : dp.regions) {
    // Each bin is one physical unit's ops, carrying the tightest sub-cycle
    // slack any member has: growing the bin deepens one multiplexer in front
    // of all of them.
    struct Bin {
      llvm::SmallVector<UnitId, 2> units;
      double slack;
    };
    llvm::SmallVector<Bin> bins;
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      auto ru = reservationOf(rb, u, u.boundOps.front().second);
      double slack = unitSlack(u, ctx.cycleTime, ctx.lib);
      Bin *dest = nullptr;
      for (Bin &bin : bins) {
        if (dp.units[bin.units.front()].identity != u.identity)
          continue;
        if (muxLevels(bin.units.size() + 1) * level >
            std::min(bin.slack, slack))
          continue;
        bool ok = llvm::all_of(bin.units, [&](UnitId m) {
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
      if (dest) {
        dest->units.push_back(uid);
        dest->slack = std::min(dest->slack, slack);
      } else {
        bins.push_back({{uid}, slack});
      }
    }
    for (Bin &bin : bins)
      if (bin.units.size() > 1)
        groups.emplace_back(bin.units.begin(), bin.units.end());
  }
  return groups;
}

std::vector<llvm::SmallVector<UnitId, 2>>
PlannedBinding::plan(const Datapath &dp, const BindingContext &) const {
  // Instance symbol -> the units whose op the scheduler put on it. A
  // `MapVector` keeps the groups in first-use order, so two compiles agree.
  llvm::MapVector<llvm::StringRef, llvm::SmallVector<UnitId, 2>> byUnit;
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units) {
      auto comp = cast<dcp::DCPathComputeOp>(dp.units[uid].repOp());
      if (FlatSymbolRefAttr unit = comp.getUnitAttr())
        byUnit[unit.getValue()].push_back(uid);
    }
  std::vector<llvm::SmallVector<UnitId, 2>> groups;
  for (auto &[unit, units] : byUnit)
    if (units.size() > 1)
      groups.push_back(std::move(units));
  return groups;
}

std::unique_ptr<BindingPolicy> bindingPolicyFor(llvm::StringRef name) {
  if (name == "trivial")
    return std::make_unique<TrivialBinding>();
  if (name == "greedy-share")
    return std::make_unique<GreedyShareBinding>();
  if (name == "planned")
    return std::make_unique<PlannedBinding>();
  return nullptr;
}

} // namespace mlir::allo::uarch
