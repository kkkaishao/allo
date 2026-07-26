/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Verify.h"

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

LogicalResult verifyDatapath(func::FuncOp func, const Datapath &dp) {
  // Supported subset: top-level siblings in program order, plus container
  // loops whose children sequence within one outer iteration (crossing as a
  // survivor register).
  if (dp.regions.empty())
    return func.emitError("allo-datapath-to-hw: no schedulable region");
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
    // the frontend match on it); the slot is the part that says WHERE.
    InFlightDiagnostic diag =
        badSite.op ? badSite.op->emitError() : func.emitError();
    return diag << "allo-datapath-to-hw: cross-region value hand-off not yet "
                   "supported: "
                << badSite.describe() << " is unresolved";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// 2. Device-contract limits.
//===----------------------------------------------------------------------===//

LogicalResult checkDeviceCapability(func::FuncOp func, const Datapath &dp) {
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
    // A boundary array's port latency is a contract with the driver, not
    // enforced by the emitted RTL; the interface manifest carries it, so any
    // latency >= 1 works, but 0 is rejected (an edge-triggered port can't).
    if (m.external && (m.readLatency < 1 || m.writeLatency < 1))
      return func.emitError("allo-datapath-to-hw: argument array with a ")
             << m.readLatency << "-cycle read / " << m.writeLatency
             << "-cycle write is unsupported; a boundary port is "
                "edge-triggered "
                "and needs at least 1 cycle. Use an internal buffer, or bind "
                "this argument to a storage impl with a >= 1 cycle access";
  }

  // `ce` is the only IP port ABI the emitter realizes. `free` has no enable, so
  // in a back-pressured region it keeps clocking and desynchronizes, but is
  // fine elsewhere; `elastic` is variable-latency, which nothing here honors.
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
    if (u.stall == allo::StallContractEnum::Elastic) {
      error(Stage::Emit, u.repOp())
          << "Operator IP '" << operatorModuleName(u)
          << "' declares the elastic (valid/ready, variable-latency) stall "
             "contract, which is not realized: its consumers are scheduled at "
             "the operator's fixed latency and the emitted instance has the "
             "free-running port shape. Declare style='ce'";
      return failure();
    }
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

LogicalResult checkEmitterSubset(func::FuncOp func, const Datapath &dp) {
  // Region shapes. Mirrors `emitRegion`'s dispatch, reading the same stored
  // discriminant rather than re-deriving it.
  for (const RegionBlock &rb : dp.regions) {
    if (rb.kind == RegionBlock::Kind::Cyclic && !rb.conditional &&
        !rb.tripCount && !rb.ubSource)
      return func.emitError("allo-datapath-to-hw: cyclic region needs a "
                            "constant or dynamic trip");
    // `emitLoopCall` advances on the child's `done`, so it would silently drop
    // a second child or any loose datapath. Keyed on the stored shape, so it
    // constrains exactly the regions that reach that controller.
    if (rb.shape == RegionBlock::Shape::CallNode &&
        (rb.callUnits.size() > 1 || !rb.units.empty() || !rb.regs.empty()))
      return func.emitError(
          "allo-datapath-to-hw: a loop body holding a sub-kernel call "
          "alongside other work reached the leaf loop-over-calls controller; "
          "the scheduler should have decomposed it into sub-regions");
  }

  // The shapes a CONCURRENT container admits. A structural composition has no
  // datapath of its own ("the top computes nothing", delivered by
  // `outline-loose-processes`) and instantiates each process exactly once.
  for (const RegionBlock &rb : dp.regions) {
    if (rb.determinacy != DeterminacyEnum::Concurrent)
      continue;
    if (!rb.units.empty() || !rb.memAccesses.empty() ||
        !rb.streamAccesses.empty())
      return func.emitError(
          "allo-datapath-to-hw: a concurrent (dataflow) container with its own "
          "datapath (loose load/store/compute beside the process network) is "
          "not supported; it composes child instances + channels only");
    if (rb.kind == RegionBlock::Kind::Cyclic)
      return func.emitError(
          "allo-datapath-to-hw: a dataflow process is spawned inside a loop; a "
          "process is instantiated once and runs concurrently, so spawn it "
          "once "
          "and let it iterate internally (move the loop into the process)");
  }

  // Stream protocol. A channel is one {data,valid,ready} triple time-shared by
  // every access to it, which two properties make sound. Both are the
  // scheduler's to deliver, but a violation would mis-order tokens silently.
  for (const StreamChannel &s : dp.streams) {
    // (a) One direction, but only where a boundary port forces it: a port is an
    // input or an output, so a channel both read and written has nothing to
    // lower to. Counted over this module's accesses and composed child ports.
    bool anyPut = false, anyGet = false;
    unsigned producers = 0;
    for (const StreamChannel::Access &acc : s.accesses) {
      anyPut |= acc.isPut;
      anyGet |= !acc.isPut;
    }
    for (const StreamChannel::CallEnd &e : s.callEnds) {
      bool reads = dp.calls[e.call].streamArgs[e.arg].isInput;
      anyPut |= !reads;
      anyGet |= reads;
      producers += !reads;
    }
    // Several READERS are a fan-out the emitter inserts (one FIFO each);
    // several WRITERS are a merge, whose token interleaving is not
    // deterministic.
    if (producers > 1)
      return func.emitError(
          "allo-datapath-to-hw: a stream channel is written by more than one "
          "process; a channel is single-producer and deterministic merge is "
          "not "
          "supported yet");
    if (anyPut && anyGet && !s.internal)
      return func.emitError(
          "allo-datapath-to-hw: a stream ARGUMENT both read and written inside "
          "one kernel is not yet lowered (a boundary channel lowers to one "
          "directional port); route the feedback through a second channel, or "
          "declare the channel inside the kernel");
    // A local channel with one end only is a stall by construction: the puts
    // fill it and block, or the first get waits on a token nothing produces.
    if (s.internal && !(anyPut && anyGet))
      return func.emitError("allo-datapath-to-hw: the kernel-local stream is ")
             << (anyPut ? "never read" : "never written")
             << "; a channel needs both ends inside the kernel that owns it";
    // A boundary argument nothing touches would leave a port undriven.
    if (!s.internal && !anyPut && !anyGet)
      return func.emitError("allo-datapath-to-hw: the stream argument is "
                            "neither read nor written");
    // (b) Distinct cycles in program order within a region, spanning under one
    // II. Per DIRECTION, since that is what shares a wire: a put drives
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
          if (prev && acc.stage <= prev->stage)
            return func.emitError(
                "allo-datapath-to-hw: two accesses to one stream are scheduled "
                "on the same cycle, or out of program order; they share a "
                "single handshake, so their token order would be lost");
          prev = &acc;
          first = first ? first : &acc;
        }
        if (prev && rb.ii && prev->stage - first->stage >= *rb.ii)
          return func.emitError(
              "allo-datapath-to-hw: accesses to one stream "
              "span a whole initiation interval, so successive "
              "iterations would overlap on its handshake");
      }
  }

  // Liveness. A directed cycle of channels with no initial tokens deadlocks, so
  // it suffices that the graph of UNSEEDED channels is acyclic. Insufficient
  // seeding (fewer tokens than the recurrence distance) surfaces as a hang.
  {
    llvm::DenseMap<CallId, SmallVector<CallId>> adj; // producer -> consumers
    for (const StreamChannel &s : dp.streams) {
      auto init = dyn_cast_or_null<ArrayAttr>(s.init);
      if (init && !init.empty())
        continue;
      std::optional<CallId> prod;
      for (const StreamChannel::CallEnd &e : s.callEnds)
        if (!dp.calls[e.call].streamArgs[e.arg].isInput)
          prod = e.call;
      if (!prod)
        continue; // fed from a boundary port: not part of a cycle
      for (const StreamChannel::CallEnd &e : s.callEnds)
        if (dp.calls[e.call].streamArgs[e.arg].isInput)
          adj[*prod].push_back(e.call);
    }
    llvm::DenseMap<CallId, int> color; // 0 white / 1 gray / 2 black
    SmallVector<CallId> cycle;
    // Self-parameter recursive lambda (`self(self, ...)`): a local DFS with no
    // std::function type-erasure.
    llvm::DenseMap<CallId, CallId> parent;
    auto visit = [&](auto &self, CallId u) -> bool {
      color[u] = 1;
      for (CallId v : adj[u]) {
        if (color[v] == 1) { // back edge -> the cycle v .. u -> v
          for (CallId x = u; x != v; x = parent[x])
            cycle.push_back(x);
          cycle.push_back(v);
          return true;
        }
        if (color[v] == 0) {
          parent[v] = u;
          if (self(self, v))
            return true;
        }
      }
      color[u] = 2;
      return false;
    };
    for (const CallUnit &cu : dp.calls)
      if (cycle.empty() && color[cu.id] == 0)
        visit(visit, cu.id);
    if (!cycle.empty()) {
      std::reverse(cycle.begin(), cycle.end()); // producer order
      std::string path;
      llvm::raw_string_ostream os(path);
      for (CallId x : cycle)
        os << dp.calls[x].callee << " -> ";
      os << dp.calls[cycle.front()].callee; // close the loop
      error(Stage::Emit, func)
          << "Dataflow feedback cycle [" << path
          << "] has no initial tokens and will deadlock; seed a channel on the "
             "cycle with an initializer, e.g. `s: Stream[T, depth] = [<init>]`";
      return failure();
    }
  }

  // Condition timing: a flushing leaf while or guard samples it in-cycle,
  // needing a stage-0 Unit or settled Survivor, while a sequential CHECK/RUN
  // while waits t_cond cycles. `verifyDatapath` already rejects a `None`.
  auto conditionOk = [&](const Source &s, bool sequential) {
    switch (s.kind) {
    case Source::Kind::Survivor:
      return true; // a scheduled prologue predicate, valid at the region start
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
    if (rb.conditional &&
        !conditionOk(rb.condition,
                     /*sequential=*/rb.shape == RegionBlock::Shape::Container))
      return func.emitError("allo-datapath-to-hw: a while loop with a non-"
                            "combinational (memory-/IP-dependent) condition is "
                            "not yet lowered");
    if (rb.shape == RegionBlock::Shape::Guard &&
        !conditionOk(rb.condition, /*sequential=*/false))
      return func.emitError("allo-datapath-to-hw: a guard with a "
                            "non-combinational predicate is not yet lowered");
  }
  // A leaf `while` with an in-loop store lowers: emitAccesses gates each
  // store's write-enable by `issue & cond`, so a doomed exit iteration
  // commits nothing, matching the non-speculative loop-carried-survivor rule.

  // Operator realizability: a combinational unit needs an EmitHW comb lowering,
  // an IP unit a non-empty module name. Fail by op name, not deep in emission.
  for (const FuncUnit &u : dp.units) {
    if (u.comb) {
      if (!combEmitted(u.opType)) {
        error(Stage::Emit, u.repOp())
            << "Combinational operator '" << u.opType
            << "' has no native EmitHW lowering; provide an IP or add native "
               "support";
        return failure();
      }
    } else if (u.impl.empty()) {
      error(Stage::Emit, u.repOp())
          << "Operator '" << u.opType
          << "' has no IP module realization; provide an IP for this operator "
             "or add native support";
      return failure();
    }
  }
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
  for (const MemUnit &m : dp.mems)
    for (const MemUnit::Access &acc : m.accesses) {
      --listed;
      bool hasPort = acc.portIdx != MemUnit::Access::kNoPort;
      assert(hasPort == m.external &&
             "a boundary port slot is held by exactly the external accesses");
      assert((!hasPort || (acc.isWrite ? dp.writePorts : dp.readPorts).size() >
                              acc.portIdx) &&
             "an access's port slot is out of its boundary port list");
      (void)hasPort;
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

LogicalResult validateDatapath(func::FuncOp func, const Datapath &dp) {
  if (failed(verifyDatapath(func, dp)) ||
      failed(checkDeviceCapability(func, dp)) ||
      failed(checkEmitterSubset(func, dp)))
    return failure();
  assertStructuralInvariants(dp);
  return success();
}

} // namespace mlir::allo::uarch
