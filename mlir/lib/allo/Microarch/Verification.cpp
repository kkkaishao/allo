/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Verification.h"

#include "allo-c/Schedule.h"           // kPartitionAttr
#include "allo/Microarch/Naming.h"     // operatorModuleName
#include "allo/Microarch/Primitives.h" // combEmitted
#include "allo/Support/Logging.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GetGlobalOp
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// 1. Model well-formedness.
//===----------------------------------------------------------------------===//

// Supported subset: top-level siblings in program order, plus container loops
// whose children sequence within one outer iteration.
LogicalResult verifyDatapath(dcp::DCPathModuleOp func, const Datapath &dp) {
  // A kernel with no schedulable region computes nothing.
  if (dp.regions.empty())
    warn(Stage::Emit, func)
        << "Kernel '" << func.getSymName()
        << "' has no schedulable region: it emits as hardware that does "
           "nothing and completes immediately";
  // The builder already reported the offending edge; fail before any
  // hardware is built from the placeholder depths it left.
  if (dp.infeasible)
    return failure();

  // An unresolved (None) Source is a cross-region SSA hand-off the builder
  // could not thread; reject it here rather than asserting in `resolveSource`.
  // `forEachSource` owns the slot list and which slots may be empty.
  bool found = false;
  SourceSite badSite{};
  forEachSource(dp, [&](const Source &s, const SourceSite &site) {
    if (found || !site.required || s)
      return;
    found = true;
    badSite = site;
  });
  if (found) {
    // "cross-region value hand-off" is the stable phrase tests match on; the
    // builder's own hand-off rejects use the same wording.
    unsupported(Stage::Emit, badSite.op ? badSite.op : func.getOperation())
        << "A cross-region value hand-off is not lowered yet: "
        << badSite.describe() << " is unresolved";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// 2. Device-contract limits.
//===----------------------------------------------------------------------===//

namespace {

/// The combinational delay the SCHEDULE never accounted for, propagated along
/// the chains it can lengthen. Two contributors, tracked apart because their
/// remedies differ: the multiplexers a shared binding grows, and the cells the
/// reifier synthesizes after the solve.
///
/// The scheduler proved `z(op) + inDelay(op) <= period` over a datapath whose
/// unit inputs are all driven directly, and each addition shifts its consumer's
/// arrival by a constant. The delta is therefore additive along a combinational
/// path, so propagating it alone against each op's remaining sub-cycle slack is
/// exact.
struct AddedDelay {
  /// One arrival, split by what added it. `ns` is what a consumer waits for,
  /// whatever the composition.
  struct Delay {
    double mux = 0.0;  // levels of a shared unit's input multiplexer
    double cone = 0.0; // cells the scheduling stage never priced
    double ns() const { return mux + cone; }
  };

  AddedDelay(const Datapath &dp, const OperatorLibrary &lib, double level)
      : dp(dp), lib(lib), level(level) {}

  const Datapath &dp;
  const OperatorLibrary &lib;
  double level; // one LUT level of the one-hot select's AND-OR reduction
  llvm::DenseMap<UnitId, Delay> memo;

  /// What arrives at \p id's input ports, its own delay excluded.
  Delay ofUnit(UnitId id) {
    auto seen = memo.find(id);
    if (seen != memo.end())
      return seen->second;
    // Seeded before the walk, so a fused recurrence's self-referential input
    // terminates instead of recursing forever.
    memo[id] = Delay{};
    Delay added;
    for (const Source &in : dp.units[id].inputs)
      added = later(added, ofSource(in));
    memo[id] = added;
    return added;
  }

  Delay ofSource(const Source &s) {
    if (s.kind == Source::Kind::Mux) {
      const Mux &m = dp.muxes[s.id];
      Delay in;
      for (const Source &src : m.sources)
        in = later(in, ofSource(src));
      in.mux += muxLevels(m.sources.size()) * level;
      return in;
    }
    // Anything else is held when the cycle starts: a register tap, a port, a
    // literal, a survivor, or a unit whose own output is registered.
    if (s.kind != Source::Kind::Unit || dp.units[s.id].latency)
      return {};
    const FuncUnit &u = dp.units[s.id];
    Delay in = ofUnit(u.id);
    if (!scheduled(u)) {
      assert(u.identity.comb &&
             "only a combinational cell reaches the datapath unscheduled");
      in.cone += combDelay(u.identity.realization);
    }
    return in;
  }

  /// Whether the scheduling stage placed \p u: it stamps the sub-cycle start it
  /// proved (`z`) onto every op it schedules. The one thing left that carries
  /// none is a sequential while's condition, which the reify ASAP-walks after
  /// the solve, so no chain-break pass cut it and nothing charged its delay to
  /// the clock. A bound and a predicate the SCHEDULER expands now.
  static bool scheduled(const FuncUnit &u) {
    return llvm::all_of(u.boundOps, [](const auto &bound) {
      return bound.first->hasAttr("z");
    });
  }

  /// The device row a combinational realization is priced against.
  double combDelay(StringRef realization) const {
    std::optional<CombOpKindEnum> kind = symbolizeCombOpKindEnum(realization);
    assert(kind && "a combinational cell realizes a CombOpKind");
    return lib.combDelay(*kind);
  }

  static Delay later(const Delay &a, const Delay &b) {
    return a.ns() >= b.ns() ? a : b;
  }
};

/// Every unit's inputs still settle within the period, muxes and reified cones
/// included. Two faults with two remedies, so two levels: a binding that misses
/// timing is REJECTED, since binding is a choice the user can withdraw, while
/// logic the reifier synthesized after the solve is WARNED about, since no
/// option avoids it and only a slower clock or a simpler expression does.
LogicalResult checkCombPathsMeetPeriod(const Datapath &dp, float cycleTime,
                                       const OperatorLibrary &lib) {
  // One picosecond of slop, the resolution the scheduler's own model carries.
  constexpr double kSlop = 1e-3;
  AddedDelay added(dp, lib, muxLevelDelay(lib));
  llvm::DenseMap<UnitId, RegionId> unitRegion;
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units)
      unitRegion[uid] = rb.id;

  // The worst overrun of one reified expression, per region. Its cells are one
  // chain, so reporting each of them reports one fault as many times as it is
  // deep.
  struct Overrun {
    double path = 0.0, over = 0.0;
    Operation *op = nullptr;
  };
  llvm::SmallVector<Overrun> cones(dp.regions.size());
  auto record = [&](unsigned slot, double path, double over, Operation *op) {
    if (over > cones[slot].over)
      cones[slot] = {path, over, op};
  };

  bool ok = true;
  for (const FuncUnit &u : dp.units) {
    AddedDelay::Delay d = added.ofUnit(u.id);
    if (d.ns() == 0.0)
      continue;
    double slack = unitSlack(u, cycleTime, lib);
    if (d.ns() <= slack + kSlop)
      continue;
    // The multiplexer alone overruns, so the binding is the fault whatever else
    // rides the same path.
    if (d.mux > slack + kSlop) {
      // Anchor on the tightest bound op, the one the slack came from.
      Operation *worst = u.repOp();
      for (const auto &[op, residue] : u.boundOps) {
        auto z = op->getAttrOfType<FloatAttr>("z");
        auto wz = worst->getAttrOfType<FloatAttr>("z");
        if (z && (!wz || z.getValueAsDouble() > wz.getValueAsDouble()))
          worst = op;
      }
      // `d.mux` covers the whole input cone, so it may come from a shared
      // predecessor rather than from a multiplexer on this unit.
      unsupported(Stage::Emit, worst)
          << "Binding put " << llvm::format("%.2f", d.mux)
          << " ns of multiplexer on the path reaching this operation (its unit "
             "is shared between "
          << u.boundOps.size() << " operations), which is "
          << llvm::format("%.2f", d.mux - slack)
          << " ns more than the schedule left it against a "
          << llvm::format("%.2f", cycleTime)
          << " ns clock. The schedule was cut before the multiplexer existed, "
             "so this would miss timing in silicon. Use binding='trivial' for "
             "this kernel, or raise the target period";
      ok = false;
      continue;
    }
    // `cycleTime - slack` is the unit's own `z + inDelay`, so this is the whole
    // combinational path ending at it.
    record(unitRegion.lookup(u.id), cycleTime - slack + d.ns(), d.ns() - slack,
           u.repOp());
  }
  for (const Overrun &worst : cones) {
    if (!worst.op)
      continue;
    warn(Stage::Emit, worst.op)
        << "A while condition the compiler expanded AFTER the schedule was "
           "cut is "
        << llvm::format("%.2f", worst.path)
        << " ns of combinational logic against a "
        << llvm::format("%.2f", cycleTime) << " ns clock, "
        << llvm::format("%.2f", worst.over)
        << " ns over. The scheduling stage never saw it, so no chain break was "
           "placed in it and no simulation can show the violation: the emitted "
           "RTL misses timing in silicon. Lower the target frequency, or "
           "simplify the expression (a floordiv or mod in an affine map "
           "expands to a divider)";
  }
  return success(ok);
}

} // namespace

LogicalResult checkDeviceCapability(dcp::DCPathModuleOp func,
                                    const Datapath &dp, float cycleTime,
                                    const OperatorLibrary &lib) {
  // Memory rows the SCHEDULER honors, so a structure that silently realizes
  // them differently would place every consumer on the wrong cycle.
  for (const MemUnit &m : dp.mems) {
    // A partition on an initialized array is a silent no-op: `bankLayoutOf`
    // reads `allo.part` off the `memref.get_global` while the attribute rides
    // the `memref.global`.
    if (m.romInit) {
      assert(m.numBanks == 1 &&
             "an initialized array is laid out as one bank (allo.part on a "
             "memref.global never reaches the layout)");
      auto gg = m.memref.getDefiningOp<memref::GetGlobalOp>();
      assert(gg && "an initializer can only come from a memref.get_global");
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          gg, gg.getNameAttr());
      if (global && global->hasAttr(kPartitionAttr))
        warn(Stage::Emit, func)
            << "Partition on the initialized array '" << gg.getName()
            << "' is ignored: an array declared with compile-time contents is "
               "realized as a single bank";
    }
    assert(m.writeLatency >= 1 && "a 0-cycle write port reached emission");
  }

  // `ce` is the only IP port ABI the emitter realizes. `free` has no enable, so
  // it keeps clocking and desynchronizes in a back-pressured region, but is
  // fine elsewhere; `elastic` is rejected before scheduling.
  llvm::SmallDenseSet<unsigned> backPressured;
  for (const StreamChannel &s : dp.streams)
    for (const StreamChannel::Access &acc : s.accesses)
      backPressured.insert(acc.region);
  llvm::DenseMap<UnitId, unsigned> unitRegion;
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units)
      unitRegion[uid] = rb.id;
  for (const FuncUnit &u : dp.units) {
    if (u.identity.comb || u.stall == allo::StallContractEnum::Ce)
      continue;
    assert(u.stall != allo::StallContractEnum::Elastic &&
           "an elastic IP reached emission");
    if (backPressured.count(unitRegion.lookup(u.id))) {
      error(Stage::Emit, u.repOp())
          << "Operator IP '" << operatorModuleName(u)
          << "' is free-running (no clock enable) but sits in a stream region, "
             "whose datapath freezes under back-pressure; the IP would keep "
             "advancing and fold a stale result. Declare style='ce'";
      return failure();
    }
  }

  return checkCombPathsMeetPeriod(dp, cycleTime, lib);
}

//===----------------------------------------------------------------------===//
// 3. What this emitter lowers.
//===----------------------------------------------------------------------===//

// Whether a counted container holds no work of its own: the reifier gives every
// run of loose ops its own child region, so this checks what a unit READS
// rather than whether units exist. A conditional container is exempt.
[[maybe_unused]] static bool containerOwnsNoDatapath(const RegionBlock &rb,
                                                     const Datapath &dp) {
  if (!rb.memAccesses.empty() || !rb.streamAccesses.empty() ||
      !rb.callUnits.empty())
    return false;
  for (UnitId uid : rb.units)
    for (const Source &s : dp.units[uid].inputs)
      if (s.kind == Source::Kind::Survivor &&
          llvm::is_contained(rb.children, s.id))
        return false;
  return true;
}

LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp) {
  // Region shapes, mirroring `emitRegion`'s dispatch on the same stored
  // discriminant.
  for (const RegionBlock &rb : dp.regions) {
    // The op verifier already enforces that a counted `dcp.pipeline` carries
    // its trip either as the `trip` attribute or as the `dynamicBound` operand.
    assert((rb.kind != RegionBlock::Kind::Cyclic || rb.conditional ||
            rb.tripCount || rb.ubSource) &&
           "a counted cyclic region reached emission with neither a constant "
           "nor a dynamic trip; the reifier owns that");
    // `emitLoopCall` advances on the child's `done`, so it would silently drop
    // a second child or any loose datapath.
    assert(
        (rb.shape != RegionBlock::Shape::CallNode ||
         (rb.callUnits.size() <= 1 && rb.units.empty() && rb.regs.empty())) &&
        "a loop body holding a sub-kernel call alongside other work reached "
        "the leaf loop-over-calls controller; the scheduler must decompose "
        "it into sub-regions");
    assert((rb.shape != RegionBlock::Shape::Container || rb.conditional ||
            containerOwnsNoDatapath(rb, dp)) &&
           "a counted container reached emission carrying work of its own; the "
           "reifier gives every run of loose ops a child region");
  }

  // `verify-rtl-legality` owns the shapes a CONCURRENT container admits and the
  // caller/callee partition agreement, both settled before scheduling.

  // Stream protocol: a channel's {data,valid,ready} triple is time-shared by
  // all its accesses, sound only if the scheduler keeps them ordered and
  // non-overlapping. Ends are checked pre-schedule; timing is checked here.
  for (const StreamChannel &s : dp.streams) {
    // Distinct cycles in program order within a region, spanning under one II.
    // Per DIRECTION, since that is what shares a wire: a put drives
    // {data, valid} and a get {ready}, so a local channel's ends may coincide.
    for (const RegionBlock &rb : dp.regions)
      for (bool put : {false, true}) {
        const StreamChannel::Access *first = nullptr, *prev = nullptr;
        for (AccRef r : rb.streamAccesses) {
          if (r.id != s.id)
            continue;
          const StreamChannel::Access &acc = s.accesses[r.idx];
          if (acc.isPut != put)
            continue;
          assert((!prev || acc.stage > prev->stage) &&
                 "two accesses to one stream are scheduled on the same cycle, "
                 "or out of program order; they share a single handshake, so "
                 "their token order is lost. The scheduler owns this");
          prev = &acc;
          first = first ? first : &acc;
        }
        assert((!prev || !rb.ii || prev->stage - first->stage < *rb.ii) &&
               "accesses to one stream span a whole initiation interval, so "
               "successive iterations overlap on its handshake. The scheduler "
               "owns this");
      }
  }

  // Condition timing: a flushing leaf while or guard samples it in-cycle,
  // needing a stage-0 Unit or settled Survivor, while a sequential CHECK/RUN
  // while waits t_cond cycles. `verifyDatapath` already rejects a `None`.
  auto conditionOk = [&](const Source &s, bool sequential) {
    switch (s.kind) {
    // A scheduled prologue predicate is settled at the region start.
    case Source::Kind::Survivor:
      return true;
    case Source::Kind::Unit:
      return sequential || dcpStart(dp.producingOp(s)) == 0;
    default:
      return false; // a memory / IP / raw driver
    }
  };
  for (const RegionBlock &rb : dp.regions) {
    // Which of the two while controllers runs is the stored shape: a Container
    // while is the sequential CHECK/RUN one, a Leaf while the flushing one.
    if (rb.conditional && !conditionOk(rb.condition,
                                       /*sequential=*/rb.shape ==
                                           RegionBlock::Shape::Container)) {
      unsupported(Stage::Emit, rb.op)
          << "A while loop with a non-combinational (memory-/IP-dependent) "
             "condition is not lowered yet";
      return failure();
    }
    if (rb.shape == RegionBlock::Shape::Guard &&
        !conditionOk(rb.condition, /*sequential=*/false)) {
      unsupported(Stage::Emit, rb.op)
          << "A guard with a non-combinational predicate is not lowered yet";
      return failure();
    }
  }
  // A leaf `while` with an in-loop store needs no check: emitAccesses gates the
  // store's write-enable by `issue & cond`, so a doomed exit iteration commits
  // nothing.

  // Operator realizability is settled before scheduling: an op with neither an
  // IP row nor a `combKindOf` lowering never becomes a `dcp.compute`.
  for (const FuncUnit &u : dp.units)
    assert((u.identity.comb ? combEmitted(u.identity.realization)
                            : u.identity.realized()) &&
           "an unrealizable operator reached emission");
  return success();
}

//===----------------------------------------------------------------------===//
// 4. Structural backstops: invariants an upstream pass owns, asserted here so
// a regression in that pass fails at the seam rather than miscompiling.
//===----------------------------------------------------------------------===//

static void assertStructuralInvariants(const Datapath &dp) {
#ifndef NDEBUG
  // Every access is listed by exactly the region that issues it, and exactly
  // the EXTERNAL accesses hold a boundary port slot.
  unsigned listed = 0;
  for (const RegionBlock &rb : dp.regions) {
    listed += rb.memAccesses.size();
    for (AccRef r : rb.memAccesses)
      assert(dp.mems[r.id].accesses[r.idx].region == rb.id &&
             "a region lists an access another region issues");
  }
  for (const MemUnit &m : dp.mems) {
    // A scattered argument's ports are per ELEMENT, so its accesses hold no
    // port slot: each of them reads every element port and selects.
    assert(m.elemPorts.size() == (m.scattered ? m.depthWords : 0) &&
           "element ports belong to exactly the scattered memories, one per "
           "element");
    for (const MemUnit::Access &acc : m.accesses) {
      --listed;
      bool hasPort = acc.portIdx != MemUnit::Access::kNoPort;
      assert(hasPort == (m.external && !m.scattered) &&
             "a boundary port slot is held by exactly the addressed external "
             "accesses");
      assert((!hasPort || (acc.isWrite ? dp.writePorts : dp.readPorts).size() >
                              acc.portIdx) &&
             "an access's port slot is out of its boundary port list");
      (void)hasPort;
    }
  }
  assert(listed == 0 && "every memory access belongs to exactly one region");
  // An indeterminate call finishes at a data-dependent cycle, so nothing
  // statically scheduled may share its region; `enumerateRegions` isolates it.
  // A CONCURRENT container is exempt: nothing in it is placed against a child.
  for (const RegionBlock &rb : dp.regions) {
    if (rb.determinacy == DeterminacyEnum::Concurrent)
      continue;
    if (llvm::none_of(rb.callUnits,
                      [&](CallId cid) { return !dp.calls[cid].latency; }))
      continue;
    bool alone = rb.callUnits.size() == 1 && rb.units.empty() &&
                 rb.regs.empty() && rb.memAccesses.empty() &&
                 rb.streamAccesses.empty();
    assert(alone && "an indeterminate call shares its region with statically-"
                    "scheduled work; the region partitioner must isolate it");
  }
  // A constant table has no write port for anyone to master. A child may READ
  // one, but a writing port group would have nowhere to land.
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      assert(!(dp.mems[ma.mem].isRom && ma.isWrite) &&
             "a sub-kernel writes the ports of a constant table");
#else
  (void)dp;
#endif
}

LogicalResult validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp,
                               float cycleTime, const OperatorLibrary &lib) {
  if (failed(verifyDatapath(func, dp)) ||
      failed(checkDeviceCapability(func, dp, cycleTime, lib)) ||
      failed(checkEmitterSubset(func, dp)))
    return failure();
  assertStructuralInvariants(dp);
  return success();
}

} // namespace mlir::allo::uarch
