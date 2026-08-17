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
#include "allo/Scheduling/OperatorLibrary.h" // combParamWidth, muxCone's rows
#include "allo/Scheduling/Scheduler.h"       // solveSharing

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace mlir::allo::uarch {

std::vector<llvm::SmallVector<UnitId, 2>>
TrivialBinding::plan(const Datapath &, const BindingContext &) const {
  return {};
}

namespace {

/// Whether \p v is an un-latched iter-arg of \p rb's own leaf pipeline.
/// Reading it costs an extra mux arm on that port, for the reduction identity
/// it re-injects (`Mux::Phase`). Asked of the IR because a policy runs before
/// the interconnect exists.
bool carriedOperand(const RegionBlock &rb, Value v) {
  if (rb.container || rb.kind != RegionBlock::Kind::Cyclic)
    return false;
  Block *body = &cast<dcp::DCPathPipelineOp>(rb.op).getBody().front();
  auto barg = dyn_cast<BlockArgument>(v);
  return barg && barg.getOwner() == body && barg.getArgNumber() >= 1;
}

/// Each region-local unit's representative op mapped to its index in
/// `rb.units`. At plan time a unit is one op, so a producer names its unit.
llvm::DenseMap<Operation *, unsigned> unitOwners(const Datapath &dp,
                                                 const RegionBlock &rb) {
  llvm::DenseMap<Operation *, unsigned> owner;
  for (auto [i, uid] : llvm::enumerate(rb.units))
    owner[dp.units[uid].repOp()] = i;
  return owner;
}

/// The region-local unit driving \p v, when it produces \p v combinationally in
/// \p y's own issue cycle so its input mux lengthens \p y's cone.
std::optional<unsigned>
combPred(const llvm::DenseMap<Operation *, unsigned> &owner, Value v,
         Operation *y) {
  Operation *x = v.getDefiningOp();
  auto it = x ? owner.find(x) : owner.end();
  if (it != owner.end() && !dcpLatency(x) && dcpStart(x) == dcpStart(y))
    return it->second;
  return std::nullopt;
}

/// Whether \p op reads a loop recurrence on any operand.
bool readsRecurrence(const RegionBlock &rb, Operation *op) {
  return llvm::any_of(op->getOperands(),
                      [&](Value v) { return carriedOperand(rb, v); });
}

/// Whether \p v is held for the whole of \p rb's run: defined outside the
/// region, it reaches it as a literal, a boundary port, a survivor or an
/// enclosing counter, one source at every issue cycle. A value the region
/// schedules lands at a different register tap per consumer cycle.
bool heldOutside(const RegionBlock &rb, Value v) {
  Operation *at = v.getDefiningOp();
  if (!at)
    at = cast<BlockArgument>(v).getOwner()->getParentOp();
  return !rb.op->isAncestor(at);
}

/// What one candidate binding costs a region's clock.
///
/// A multiplexer's delay does not stop at the unit it feeds: that unit's result
/// reaches whatever it combinationally drives in the same cycle, so two shared
/// units on one chain pay for both. The schedule proved `z(op) + inDelay(op) <=
/// period` over a mux-free datapath, leaving each unit `unitSlack` of room for
/// the whole cone reaching it, which is what a fold has to fit inside.
///
/// `checkCombPathsMeetPeriod` stays the authority, and this is the same
/// recursion read off the ops rather than the Sources it has yet to build. In
/// particular a shared producer contributes the max over ALL its members'
/// inputs: the structural path through every mux arm exists whichever slot the
/// select sits in, and static timing knows no slots.
struct ShareCone {
  ShareCone(const Datapath &dp, const RegionBlock &rb,
            const BindingContext &ctx)
      : lib(ctx.lib), fanin(rb.units.size(), 1), width(rb.units.size(), 1),
        rep(rb.units.size()), pack(rb.units.size()), base(rb.units.size(), 1),
        preds(rb.units.size()), slack(rb.units.size()) {
    llvm::DenseMap<Operation *, unsigned> owner = unitOwners(dp, rb);
    for (auto [i, uid] : llvm::enumerate(rb.units)) {
      rep[i] = i;
      pack[i].push_back(i);
      const FuncUnit &u = dp.units[uid];
      // An unpriced unit (no `z`) gets no room, so no fold's cone may reach it.
      slack[i] = unitSlack(u, ctx.lib, ctx.cycleTime).value_or(0.0);
      Operation *y = u.repOp();
      // A recurrence identity is re-injected through a select the emitter
      // builds fold or no fold, so it is the unshared baseline, not an
      // addition: the emit gate walks it too.
      base[i] = fanin[i] = readsRecurrence(rb, y) ? 2 : 1;
      width[i] = std::max<int64_t>(1, combParamWidth(y));
      // One op feeding two operands takes two entries; `added` maxes over them.
      for (Value v : y->getOperands())
        if (auto p = combPred(owner, v, y))
          preds[i].push_back(*p);
    }
  }

  /// Fold \p add into the bin holding \p members, whose input mux would then
  /// have \p arms sources. Keeps the fold iff every cone it deepens still meets
  /// the period, and reports whether it did.
  bool tryFold(llvm::ArrayRef<unsigned> members, unsigned add, unsigned arms) {
    unsigned r = rep[members.front()];
    unsigned savedBin = fanin[r], savedAdd = fanin[add];
    fanin[r] = fanin[add] = arms;
    rep[add] = r;
    pack[r].push_back(add);
    if (fits())
      return true;
    pack[r].pop_back();
    rep[add] = add;
    fanin[r] = savedBin;
    fanin[add] = savedAdd;
    return false;
  }

  /// Load a whole assignment (unit -> representative, \p arms summed per
  /// representative) and report whether every cone fits: the exact solve's
  /// plan re-checked under this recursion before it is built.
  bool holds(llvm::ArrayRef<unsigned> assign, llvm::ArrayRef<unsigned> arms) {
    for (auto [i, r] : llvm::enumerate(assign)) {
      if (r == i)
        continue;
      rep[i] = r;
      pack[r].push_back(static_cast<unsigned>(i));
    }
    for (unsigned i = 0, e = rep.size(); i < e; ++i)
      fanin[i] = pack[rep[i]].size() > 1 ? arms[rep[i]] : base[i];
    return fits();
  }

private:
  /// The multiplexer delay reaching the bin member \p i folded into, its own
  /// select included: the max over every member's producers, each an arm.
  double added(unsigned i) {
    unsigned r = rep[i];
    if (memo[r] >= 0.0)
      return memo[r];
    // Seeded before the walk so a revisit reads 0 rather than recurring
    // forever: two bins may feed each other once ops issuing on different
    // cycles share units.
    memo[r] = 0.0;
    double in = 0.0;
    for (unsigned m : pack[r])
      for (unsigned p : preds[m])
        in = std::max(in, added(p));
    return memo[r] = in + muxCone(lib, fanin[r], width[r]);
  }

  /// Whether every member's cone fits the slack its schedule left it.
  bool fits() {
    memo.assign(fanin.size(), -1.0);
    for (unsigned i = 0, e = fanin.size(); i < e; ++i)
      if (added(i) > slack[i])
        return false;
    return true;
  }

  const OperatorLibrary &lib; // prices each select cone (`muxCone`)
  /// Input mux sources, meaningful at a bin's representative (1 = a wire).
  llvm::SmallVector<unsigned> fanin;
  llvm::SmallVector<unsigned> width; // the muxed operand's width, per unit
  llvm::SmallVector<unsigned> rep;   // the bin each unit folded into
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> pack; // members, per rep
  llvm::SmallVector<unsigned> base; // unshared arms: 2 past a recurrence init
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> preds;
  llvm::SmallVector<double> slack;
  llvm::SmallVector<double> memo; // per representative
};

/// First-fit sharing for one region: for each unit, the region-local index of
/// the unit it runs on, its own where unshared. Same shape `solveSharing`
/// returns, so this plan can seed it.
llvm::SmallVector<unsigned> greedyShare(const Datapath &dp,
                                        const RegionBlock &rb,
                                        const BindingContext &ctx) {
  ShareCone cone(dp, rb, ctx);
  // Each bin is one physical unit's ops, indexed as `rb.units`. `arms` is the
  // mux the bin has grown: one source per member, plus one for each member
  // that re-injects a reduction identity on its own arm.
  struct Bin {
    llvm::SmallVector<unsigned, 2> members;
    unsigned arms = 0;
  };
  llvm::SmallVector<Bin> bins;
  auto resOf = [&](const FuncUnit &u) {
    return reservationOf(rb, u, u.boundOps.front().residue);
  };
  for (unsigned i = 0, e = rb.units.size(); i < e; ++i) {
    const FuncUnit &u = dp.units[rb.units[i]];
    auto ru = resOf(u);
    unsigned own = readsRecurrence(rb, u.repOp()) ? 2 : 1;
    Bin *dest = nullptr;
    for (Bin &bin : bins) {
      if (dp.units[rb.units[bin.members.front()]].identity != u.identity)
        continue;
      bool free = llvm::all_of(bin.members, [&](unsigned m) {
        return reservationsDisjoint(resOf(dp.units[rb.units[m]]), ru);
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
  llvm::SmallVector<unsigned> assign(rb.units.size());
  for (Bin &bin : bins)
    for (unsigned m : bin.members)
      assign[m] = bin.members.front();
  return assign;
}

/// The groups \p assign folds, appended to \p groups in representative order.
void appendGroups(const RegionBlock &rb, llvm::ArrayRef<unsigned> assign,
                  std::vector<llvm::SmallVector<UnitId, 2>> &groups) {
  llvm::MapVector<unsigned, llvm::SmallVector<UnitId, 2>> byRep;
  for (auto [i, rep] : llvm::enumerate(assign))
    byRep[rep].push_back(rb.units[i]);
  for (auto &[rep, group] : byRep)
    if (group.size() > 1)
      groups.push_back(std::move(group));
}

/// One operand port's price table, indexed by arm count up to \p maxArms: the
/// mux area and the cone delay it adds, both at \p width. An unreachable count
/// (0, 1) stays zero.
SharingProblem::Port pricedPort(const OperatorLibrary &lib, unsigned maxArms,
                                unsigned width) {
  SharingProblem::Port port;
  port.muxPrice.assign(maxArms + 1, 0);
  port.conePicos.assign(maxArms + 1, 0);
  for (unsigned a = 2; a <= maxArms; ++a) {
    port.muxPrice[a] = lib.muxPrice(a, width);
    port.conePicos[a] =
        static_cast<int64_t>(std::ceil(muxCone(lib, a, width) * 1000.0));
  }
  return port;
}

/// One region as a `SharingProblem`: the arrays `ShareCone` reads, priced per
/// operand port at that port's own width with the rows the emit gate walks. The
/// cone tables round up and the slacks down, so an admitted fold clears
/// `checkCombPathsMeetPeriod` by construction.
SharingProblem sharingProblemOf(const Datapath &dp, const RegionBlock &rb,
                                const BindingContext &ctx) {
  SharingProblem problem;
  problem.units.resize(rb.units.size());
  llvm::DenseMap<Operation *, unsigned> owner = unitOwners(dp, rb);
  std::map<std::string, unsigned> classIdx;
  llvm::SmallVector<llvm::SmallVector<unsigned>> members;
  // Held-driver keys, interned per class: equal keys name one value.
  llvm::SmallVector<llvm::DenseMap<Value, unsigned>> heldKeys;
  for (auto [i, uid] : llvm::enumerate(rb.units)) {
    const FuncUnit &u = dp.units[uid];
    auto [it, isNew] = classIdx.try_emplace(u.identity.key(), members.size());
    if (isNew) {
      members.emplace_back();
      heldKeys.emplace_back();
    }
    members[it->second].push_back(i);
    SharingProblem::Unit &unit = problem.units[i];
    unit.cls = it->second;
    unit.slackPicos = std::max<int64_t>(
        0, std::floor(unitSlack(u, ctx.lib, ctx.cycleTime).value_or(0.0) *
                      1000.0));
    Operation *y = u.repOp();
    for (auto [k, v] : llvm::enumerate(y->getOperands())) {
      unit.initArms.push_back(carriedOperand(rb, v) ? 1 : 0);
      unsigned key = 0;
      if (heldOutside(rb, v))
        key = heldKeys[unit.cls]
                  .try_emplace(v, heldKeys[unit.cls].size() + 1)
                  .first->second;
      unit.drivers.push_back(key);
      if (auto p = combPred(owner, v, y))
        unit.preds.push_back({static_cast<unsigned>(k), *p});
    }
  }
  problem.classes.resize(members.size());
  auto resOf = [&](const FuncUnit &u) {
    return reservationOf(rb, u, u.boundOps.front().residue);
  };
  for (auto [cls, mem] : llvm::enumerate(members)) {
    Operation *y = dp.units[rb.units[mem.front()]].repOp();
    SharingProblem::UnitClass &c = problem.classes[cls];
    c.instancePrice =
        ctx.lib.instancePrice(dp.units[rb.units[mem.front()]].identity,
                              std::max<int64_t>(1, combParamWidth(y)));
    for (auto [k, t] : llvm::enumerate(y->getOperandTypes())) {
      unsigned maxArms = 0;
      for (unsigned m : mem) {
        assert(problem.units[m].initArms.size() == y->getNumOperands() &&
               "one identity, one signature");
        maxArms += 1 + problem.units[m].initArms[k];
      }
      auto width = static_cast<unsigned>(
          t.isIntOrIndexOrFloat() ? std::max<int64_t>(1, datapathWidth(t)) : 1);
      c.ports.push_back(pricedPort(ctx.lib, maxArms, width));
    }
    for (unsigned p = 0; p < mem.size(); ++p) {
      auto ra = resOf(dp.units[rb.units[mem[p]]]);
      for (unsigned q = p + 1; q < mem.size(); ++q)
        if (!reservationsDisjoint(ra, resOf(dp.units[rb.units[mem[q]]])))
          problem.conflicts.push_back({mem[p], mem[q]});
    }
  }
  return problem;
}

} // namespace

std::vector<llvm::SmallVector<UnitId, 2>>
ExactShareBinding::plan(const Datapath &dp, const BindingContext &ctx) const {
  std::vector<llvm::SmallVector<UnitId, 2>> groups;
  for (const RegionBlock &rb : dp.regions) {
    llvm::SmallVector<unsigned> assign = greedyShare(dp, rb, ctx);
    SharingProblem problem = sharingProblemOf(dp, rb, ctx);
    if (auto solved = solveSharing(problem, assign, rb.op)) {
      // The solve's cone constraint charges each member its own producers,
      // but the built mux is one structure whose every arm is a timed path.
      // Re-check its plan under the emit gate's recursion and keep the greedy
      // plan, admitted fold by fold, when a cross-member arm would bust.
      llvm::SmallVector<unsigned> arms(rb.units.size(), 0);
      for (auto [i, r] : llvm::enumerate(*solved))
        arms[r] += readsRecurrence(rb, dp.units[rb.units[i]].repOp()) ? 2 : 1;
      if (ShareCone(dp, rb, ctx).holds(*solved, arms))
        assign = std::move(*solved);
    }
    appendGroups(rb, assign, groups);
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
  if (name == "exact-share")
    return std::make_unique<ExactShareBinding>();
  if (name == "planned")
    return std::make_unique<PlannedBinding>();
  return nullptr;
}

} // namespace mlir::allo::uarch
