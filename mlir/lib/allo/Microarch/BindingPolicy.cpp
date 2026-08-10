/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Binding policies (pure decision). Compatibility is the exact MRT test from
// Reservation.h. See BindingPolicy.h.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/BindingPolicy.h"

#include "allo/Microarch/Reservation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::allo::uarch {

std::vector<llvm::SmallVector<UnitId, 2>>
TrivialBinding::plan(const Datapath &, const BindingContext &) const {
  return {};
}

namespace {

/// Whether \p op reads a loop recurrence: an un-latched iter-arg of its own
/// leaf pipeline. Such an operand costs a second mux arm, for the reduction
/// identity it re-injects (`Mux::Phase`). Asked of the IR because a policy runs
/// before the interconnect exists.
bool readsRecurrence(const RegionBlock &rb, Operation *op) {
  if (rb.container || rb.kind != RegionBlock::Kind::Cyclic)
    return false;
  Block *body = &cast<dcp::DCPathPipelineOp>(rb.op).getBody().front();
  return llvm::any_of(op->getOperands(), [&](Value v) {
    auto barg = dyn_cast<BlockArgument>(v);
    return barg && barg.getOwner() == body && barg.getArgNumber() >= 1;
  });
}

/// What one candidate binding costs a region's clock.
///
/// A multiplexer's delay does not stop at the unit it feeds: that unit's result
/// reaches whatever it combinationally drives in the same cycle, so two shared
/// units on one chain pay for both. The schedule proved `z(op) + inDelay(op) <=
/// period` over a mux-free datapath, leaving each unit `unitSlack` of room for
/// the whole cone reaching it, which is what a fold has to fit inside.
///
/// `checkCombPathsMeetPeriod` stays the authority; this is its conservative
/// pre-image, the same recursion read off the ops rather than the Sources it
/// has yet to build, and it can only over-count levels.
struct ShareCone {
  ShareCone(const Datapath &dp, const RegionBlock &rb,
            const BindingContext &ctx)
      : level(muxLevelDelay(ctx.lib)), fanin(rb.units.size(), 1),
        preds(rb.units.size()), slack(rb.units.size()) {
    // At plan time a unit is one op, so a producer names its unit directly.
    llvm::DenseMap<Operation *, unsigned> owner;
    for (auto [i, uid] : llvm::enumerate(rb.units))
      owner[dp.units[uid].repOp()] = i;
    for (auto [i, uid] : llvm::enumerate(rb.units)) {
      const FuncUnit &u = dp.units[uid];
      slack[i] = unitSlack(u, ctx.cycleTime);
      Operation *y = u.repOp();
      for (Value v : y->getOperands()) {
        Operation *x = v.getDefiningOp();
        auto it = x ? owner.find(x) : owner.end();
        if (it != owner.end() && !dcpLatency(x) && dcpStart(x) == dcpStart(y))
          preds[i].push_back(it->second);
      }
    }
  }

  /// Fold \p add into the bin holding \p members, whose input mux would then
  /// have \p arms sources. Keeps the fold iff every cone it deepens still meets
  /// the period, and reports whether it did.
  bool tryFold(llvm::ArrayRef<unsigned> members, unsigned add, unsigned arms) {
    llvm::SmallVector<unsigned, 4> saved(members.size());
    for (auto [k, m] : llvm::enumerate(members)) {
      saved[k] = fanin[m];
      fanin[m] = arms;
    }
    unsigned savedAdd = fanin[add];
    fanin[add] = arms;
    if (fits())
      return true;
    for (auto [k, m] : llvm::enumerate(members))
      fanin[m] = saved[k];
    fanin[add] = savedAdd;
    return false;
  }

private:
  /// The multiplexer delay reaching member \p i's inputs, its own mux included.
  double added(unsigned i) {
    if (memo[i] >= 0.0)
      return memo[i];
    // Seeded before the walk so a revisit reads 0 rather than recurring
    // forever: two bins may feed each other once ops issuing on different
    // cycles share units.
    memo[i] = 0.0;
    double in = 0.0;
    for (unsigned p : preds[i])
      in = std::max(in, added(p));
    return memo[i] = in + muxLevels(fanin[i]) * level;
  }

  /// Whether every member's cone fits the slack its schedule left it.
  bool fits() {
    memo.assign(fanin.size(), -1.0);
    for (unsigned i = 0, e = fanin.size(); i < e; ++i)
      if (added(i) > slack[i])
        return false;
    return true;
  }

  double level; // one LUT level of a one-hot select's AND-OR reduction
  llvm::SmallVector<unsigned> fanin; // input mux sources (1 = an unshared wire)
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> preds;
  llvm::SmallVector<double> slack;
  llvm::SmallVector<double> memo;
};

} // namespace

std::vector<llvm::SmallVector<UnitId, 2>>
GreedyShareBinding::plan(const Datapath &dp, const BindingContext &ctx) const {
  std::vector<llvm::SmallVector<UnitId, 2>> groups;
  for (const RegionBlock &rb : dp.regions) {
    ShareCone cone(dp, rb, ctx);
    // Each bin is one physical unit's ops, indexed as `rb.units`. `arms` is the
    // mux the bin has grown: one source per member, plus one for each member
    // that re-injects a reduction identity on its own arm.
    struct Bin {
      llvm::SmallVector<unsigned, 2> members;
      unsigned arms = 0;
    };
    llvm::SmallVector<Bin> bins;
    for (unsigned i = 0, e = rb.units.size(); i < e; ++i) {
      const FuncUnit &u = dp.units[rb.units[i]];
      auto ru = reservationOf(rb, u, u.boundOps.front().residue);
      unsigned own = readsRecurrence(rb, u.repOp()) ? 2 : 1;
      Bin *dest = nullptr;
      for (Bin &bin : bins) {
        if (dp.units[rb.units[bin.members.front()]].identity != u.identity)
          continue;
        bool free = llvm::all_of(bin.members, [&](unsigned m) {
          const FuncUnit &mu = dp.units[rb.units[m]];
          return reservationsDisjoint(
              reservationOf(rb, mu, mu.boundOps.front().residue), ru);
        });
        if (free && cone.tryFold(bin.members, i, bin.arms + own)) {
          dest = &bin;
          break;
        }
      }
      if (dest) {
        dest->members.push_back(i);
        dest->arms += own;
      } else {
        bins.push_back({{i}, own});
      }
    }
    for (Bin &bin : bins)
      if (bin.members.size() > 1) {
        llvm::SmallVector<UnitId, 2> group;
        for (unsigned m : bin.members)
          group.push_back(rb.units[m]);
        groups.push_back(std::move(group));
      }
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
