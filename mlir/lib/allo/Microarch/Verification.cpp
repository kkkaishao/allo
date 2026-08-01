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

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// 1. Model well-formedness.
//===----------------------------------------------------------------------===//

// A scheduled datapath always holds at least one region: a `dcp.module` is what
// a reified kernel closes into, and the builder's region walk collects every
// region the reify put in one.
LogicalResult verifyDatapath(dcp::DCPathModuleOp func, const Datapath &dp) {
  // Supported subset: top-level siblings in program order, plus container
  // loops whose children sequence within one outer iteration (crossing as a
  // survivor register).
  assert(!dp.regions.empty() &&
         "a scheduled kernel has no schedulable region; the builder's region "
         "walk found none where the reify built at least one");
  // The builder already reported the offending edge; fail before any
  // hardware is built from the placeholder depths it left.
  if (dp.infeasible)
    return failure();

  // An unresolved (None) Source is a cross-region SSA hand-off the builder
  // could not thread; reject it here rather than asserting in `resolveSource`.
  // ONE sweep: `forEachSource` owns the slot list and which slots may be empty.
  bool found = false;
  SourceSite badSite{};
  forEachSource(dp, [&](const Source &s, const SourceSite &site) {
    if (found || !site.required || s)
      return;
    found = true;
    badSite = site;
  });
  if (found) {
    // The "cross-region value hand-off" phrase is the stable part (tests and
    // the frontend match on it); the slot is the part that says WHERE. The
    // builder's three hand-off rejects use the same phrase.
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

LogicalResult checkDeviceCapability(dcp::DCPathModuleOp func,
                                    const Datapath &dp) {
  // Access latencies the emitted structure cannot realize. These are device
  // rows the SCHEDULER honors, so silently emitting a 1-cycle port instead
  // would place every consumer of that array on the wrong cycle.
  for (const MemUnit &m : dp.mems) {
    // A partition on an initialized array is a silent no-op: `bankLayoutOf`
    // reads `allo.part` off the `memref.get_global` while the attribute rides
    // the `memref.global`. Warn by name, since a WRITTEN table loses ports.
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
  // in a back-pressured region it keeps clocking and desynchronizes, but is
  // fine elsewhere; the `elastic` contract is rejected before scheduling.
  llvm::SmallDenseSet<unsigned> backPressured;
  for (const StreamChannel &s : dp.streams)
    for (const StreamChannel::Access &acc : s.accesses)
      backPressured.insert(acc.region);
  llvm::DenseMap<UnitId, unsigned> unitRegion;
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units)
      unitRegion[uid] = rb.id;
  for (const FuncUnit &u : dp.units) {
    if (u.comb || u.stall == allo::StallContractEnum::Ce)
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
  return success();
}

//===----------------------------------------------------------------------===//
// 3. What this emitter lowers.
//===----------------------------------------------------------------------===//

// A COUNTED container (`emitContainer`) drives child regions and has no
// per-iteration issue pulse to time work of its own against; the only thing it
// emits is a child guard's predicate before the children run. That is why the
// check below is what a unit READS rather than whether units exist.
//
// An INVARIANT, not a legality check: the reifier wraps every run of loose ops
// between a level's child loops in its own child region, so a counted container
// only ever holds declarations and predicates. Were one to arrive with work of
// its own, an external store would leave its boundary port undriven (a null
// operand reaching `Operation::create`) and a compute over a child's result
// would read that child's survivor before the child has emitted.
//
// A CONDITIONAL container is a different controller: `emitConditionalContainer`
// emits its own condition cone (`emitConditionRegion`), so its reads and units
// are expected. Its writes and stream accesses are not emitted, but no
// frontend shape puts one in a while condition, so that stays a gap rather
// than a check.
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
  // Region shapes. Mirrors `emitRegion`'s dispatch, reading the same stored
  // discriminant rather than re-deriving it.
  for (const RegionBlock &rb : dp.regions) {
    // A counted `dcp.pipeline` carries its trip either as the `trip` attribute
    // or as the `dynamicBound` operand; the op verifier enforces that, so a
    // cyclic non-while region always has one of the two by the time it is here.
    assert((rb.kind != RegionBlock::Kind::Cyclic || rb.conditional ||
            rb.tripCount || rb.ubSource) &&
           "a counted cyclic region reached emission with neither a constant "
           "nor a dynamic trip; the reifier owns that");
    // `emitLoopCall` advances on the child's `done`, so it would silently drop
    // a second child or any loose datapath. Keyed on the stored shape, so it
    // constrains exactly the regions that reach that controller.
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

  // `verify-rtl-legality` owns the shapes a CONCURRENT container admits (no
  // datapath of its own, each process instantiated once) and the caller/callee
  // partition agreement, both settled before scheduling.

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
    // A scheduled prologue predicate, and a func-scope cone combinational over
    // exactly such predicates and the module's ports: both are settled at the
    // region start.
    case Source::Kind::Survivor:
    case Source::Kind::Scope:
      return true;
    case Source::Kind::Unit:
      return sequential || dcpStart(dp.units[s.id].repOp()) == 0;
    default:
      return false; // a memory / IP / raw driver
    }
  };
  for (const RegionBlock &rb : dp.regions) {
    // Which of the two while controllers runs is the stored shape: a Container
    // while is the sequential CHECK/RUN one (it may wait `t_cond`), a Leaf
    // while the flushing one (it samples the condition in-cycle).
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
  // A leaf `while` with an in-loop store lowers: emitAccesses gates each
  // store's write-enable by `issue & cond`, so a doomed exit iteration
  // commits nothing, matching the non-speculative loop-carried-survivor rule.

  // Operator realizability is settled before scheduling: an op with neither an
  // IP row nor a `combKindOf` lowering never becomes a `dcp.compute`.
  for (const FuncUnit &u : dp.units)
    assert((u.comb ? combEmitted(u.opType) : !u.impl.empty()) &&
           "an unrealizable operator reached emission");
  // A func-scope cone is combinational by construction (`bindScopeOps` rejects
  // anything `combKindOf` does not name), so only `emitCompute`'s coverage is
  // left to check.
  for (const ScopeUnit &su : dp.scopeUnits)
    assert(combEmitted(su.opType) &&
           "an unrealizable func-scope expression reached emission");
  return success();
}

//===----------------------------------------------------------------------===//
// 4. Structural backstops: invariants an upstream pass owns, asserted here so
// a regression in that pass fails at the seam rather than miscompiling.
//===----------------------------------------------------------------------===//

static void assertStructuralInvariants(const Datapath &dp) {
#ifndef NDEBUG
  // Every access is listed by exactly the region that issues it, and exactly
  // the EXTERNAL accesses hold a boundary port slot. Consumers read both facts
  // off the model, so a builder that forgets one would miscompile silently.
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
  // one (that is a container-owned lookup table, and read-only is a property of
  // the USE), but a writing port group would have nowhere to land.
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      assert(!(dp.mems[ma.mem].isRom && ma.isWrite) &&
             "a sub-kernel writes the ports of a constant table");
#else
  (void)dp;
#endif
}

LogicalResult validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp) {
  if (failed(verifyDatapath(func, dp)) ||
      failed(checkDeviceCapability(func, dp)) ||
      failed(checkEmitterSubset(func, dp)))
    return failure();
  assertStructuralInvariants(dp);
  return success();
}

} // namespace mlir::allo::uarch
