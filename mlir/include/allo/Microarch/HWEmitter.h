/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// PER-REGION emission, split by role along the control/datapath (F/G) seam.
// ControlEmitter is control (G), DatapathEmitter is datapath (F), HWEmitter the
// orchestrator. Elasticity (H) is derived per region by `deriveStallShell` and
// handed to each side: G takes `issueEnable`, F takes `chainEnable`.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_HWEMIT_H
#define ALLO_MICROARCH_HWEMIT_H

#include "allo/IR/AlloOps.h" // kMemPortAttr
#include "allo/Microarch/Naming.h"
#include "allo/Microarch/Primitives.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::allo::iface {
struct ModuleInterface; // each callee's port model, read to wire its instance
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Terminator: what ends a pipelined region's run, a counter reaching an
// iteration bound (`Counted`) or a condition going false (`Conditional`).
//===----------------------------------------------------------------------===//
struct Terminator {
  enum class Kind { Counted, Conditional };
  Kind kind = Kind::Counted;
  // Induction bounds: `lb`/`step` seed the counter register (init lb, +=step),
  // `ub` ends it (iv+step >= ub). A while free-runs a 0-based counter
  // (lb=0/step=1, ub null), terminating on ~cond.
  Value lb, ub, step;
  Value cond; // Conditional: the i1 continue condition (a datapath value)

  static Terminator counted(Value lb, Value ub, Value step) {
    return {Kind::Counted, lb, ub, step, Value()};
  }
  static Terminator conditional(Value cond, Value zero, Value one) {
    return {Kind::Conditional, zero, Value(), one, cond};
  }

  /// The iteration issued at `iv` is the last one: `iv + step` reaches `ub`, or
  /// the continue-condition is false. \p ivStep is `iv + step`.
  Value isLast(EmitContext &c, Value ivStep) const {
    return kind == Kind::Conditional ? c.notBit(cond) : c.icmpSgeV(ivStep, ub);
  }
  /// The region is empty (issues nothing): lb >= ub. A while is never empty
  /// here; its zero-iteration case is the condition false on iteration 0,
  /// handled by the normal exit pulse.
  Value isEmpty(EmitContext &c) const {
    return kind == Kind::Conditional ? c.f1 : c.icmpSgeV(lb, ub);
  }
  /// The start pulse gated so an empty region issues nothing; a while passes
  /// through unconditionally.
  Value gateStart(EmitContext &c, Value start) const {
    return kind == Kind::Counted ? c.andBits(start, c.notBit(isEmpty(c)))
                                 : start;
  }
};

//===----------------------------------------------------------------------===//
// ControlEmitter (G): a per-region control regime plus its completion signal.
// It consumes a resolved Terminator and never itself walks a Source.
//===----------------------------------------------------------------------===//
struct RegionControl {
  Value issue; // pipeline issue / valid signal (already gated by the stall
               // shell's enable)
  Value
      counter; // iteration index (Source::Counter); null for an acyclic region
  Value wantIssue; // the UNgated issue desire (issue before `& enable`): the
                   // stall shell hazards a stage-0 stream access on this, not
                   // the gated issue, to stay combinationally acyclic
  Value running;   // the region is executing: the counter reloads its lower
                   // bound while low. Null for a done-driven controller, whose
                   // counter reloads on `start` instead.
  /// The modulo phase [0, ii), reloaded on `start` and advancing on every
  /// enabled cycle after it, drain included: an op landing at cycle `r` lands
  /// at phase `r % ii`. Null unless a schedule-paced controller runs at II > 1.
  Value phase;
  /// One register per `RegionBlock::addrStrides` entry, holding that multiple
  /// of `counter`. Updated by the same expression as the counter, so the two
  /// cannot fall out of step.
  llvm::SmallVector<Value> scaledCounters;
};

//===----------------------------------------------------------------------===//
// DatapathFeedback (F -> G): the store timing a control regime consumes to
// compute completion.
//===----------------------------------------------------------------------===//
struct DatapathFeedback {
  // The deepest store's commit cycle as the writes were emitted (a stream put
  // folds in too); 0 if the region stores nothing. `emitRegion` checks it
  // against `RegionBlock::drainStage`, the same number decided on the model.
  unsigned storeDrain = 0;
  // A CallUnit region's completion. For a CallNode (loop-over-call) region it
  // is the child's per-invocation completion PULSE, which paces the counter;
  // for a scheduled or concurrent composition it is the joined pass-scoped
  // done LEVEL, which IS (part of) the region's completion. Null for a
  // call-free region.
  Value callDone;
};

//===----------------------------------------------------------------------===//
// IterationControl: the output of a DONE-DRIVEN controller, one whose
// iterations are paced by the body COMPLETING rather than by the schedule's own
// cadence. `rc.issue` is the body-launch pulse; `done` is latched here rather
// than by `emitDone`, a done-driven region having no drain to count.
//===----------------------------------------------------------------------===//
struct IterationControl {
  RegionControl rc;
  Value done;
};

struct ControlEmitter {
  EmitContext &c;
  explicit ControlEmitter(EmitContext &c) : c(c) {}

  /// Pick the control shape for region \p rb, acyclic or a pipelined loop, and
  /// emit it, driven by \p start against the resolved \p term.
  RegionControl emitPipelineControl(const uarch::RegionBlock &rb,
                                    const Terminator &term, Value start,
                                    const StallShell &sh) const;
  /// The scaled counters of \p rb: one register per `addrStrides` entry, each
  /// at its OWN width (`AddrStride::width`) rather than the counter's. \p
  /// update is the counter's next-value expression with `lb` and `step` scaled,
  /// supplied by the caller since the two controller families disagree about
  /// it. \p bypassStart mirrors a done-driven counter's start-cycle bypass.
  llvm::SmallVector<Value> emitScaledCounters(
      const uarch::RegionBlock &rb, Value bypassStart,
      llvm::function_ref<Value(Value cur, Value stepped, Value init)> update)
      const;
  /// The one pipelined control skeleton for the free-running (II==1), modulo
  /// (II>1) and while (flushing) regimes: a `running` flag plus an iteration
  /// counter, differing only in \p term and, for II>1, a phase counter gating
  /// issue. Non-speculative for a conditional terminator (II >= t_cond, so no
  /// doomed iteration issues); no backpressure. \p sh gates issue as
  /// `wantIssue & sh.issueEnable` and runs the phase counter on
  /// `sh.chainEnable`; a rigid shell leaves both ungated.
  /// \p region names the emitted state cells (`r<id>_run` / `_iv` / `_phase`).
  RegionControl emitPipelined(unsigned region, int64_t ii,
                              const Terminator &term, Value start,
                              const StallShell &sh) const;
  /// The straight-line control skeleton: one pass, no counter. The pass is
  /// DEFERRED while `sh.issueEnable` is low rather than dropped, which lets a
  /// stage-0 stream access wait for its handshake. A rigid shell issues
  /// unconditionally and builds no state at all.
  RegionControl emitAcyclic(unsigned region, Value start, bool topLevel,
                            const StallShell &sh) const;

  /// The counted done-driven controller: `Container` and `CallNode` x
  /// `CountedStatic`. Runs one body pass per iteration of \p term, launching
  /// the next when \p complete pulses (the body's drain edge, a Backedge the
  /// caller resolves once the body has emitted). \p rb's `shape` decides
  /// whether the FIRST pass may launch on \p start itself.
  IterationControl emitCountedIteration(const uarch::RegionBlock &rb,
                                        const Terminator &term, Value start,
                                        Value complete) const;
  /// The conditional done-driven controller: `Container` x `Conditional`, a
  /// sequential-wrapper while. A CHECK pulse one cycle after \p start and after
  /// each body drain (\p complete) re-evaluates \p cond on the settled
  /// iter-args, \p tCond cycles later for a condition that reads memory or an
  /// IP, then forks to continue or finish. The region has no counter, so the
  /// returned `rc.counter` is null.
  IterationControl emitCheckedIteration(unsigned region, Value cond,
                                        unsigned tCond, Value start,
                                        Value complete) const;

  /// The region's completion signal, one latched level for every regime: it
  /// rises when \p lastIssue delayed `rb.drainStage` cycles lands, or
  /// immediately on \p emptyDone (null when unreachable). The latch's register
  /// cycle is the store/result commit cycle, so a sibling starting on this done
  /// reads every committed store and survivor. A \p retrig region resets on \p
  /// start and reads 0 on the \p start cycle itself, since a completion pulse
  /// coinciding with \p start would otherwise latch high on the first pass and
  /// never produce a later rising edge. \p sh holds the pulse through
  /// back-pressure, the last store or token not being committed until it is
  /// accepted.
  Value emitDone(const uarch::RegionBlock &rb, Value lastIssue, Value emptyDone,
                 Value start, bool retrig, const StallShell &sh) const;
};

//===----------------------------------------------------------------------===//
// DatapathEmitter (F): register chains, compute units, memory access, and
// Source resolution. Reads the controller's `issue`/`counter` (setControl) and
// returns the region's store drain.
//===----------------------------------------------------------------------===//
struct DatapathEmitter {
  EmitContext &c;
  // The sealed model. Emission is a pure function of it.
  const uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;
  const llvm::StringMap<Operation *> &opModules;

  // A region's controller outputs, the G->F seam. `counter` is null for an
  // acyclic region; `wantIssue` is null when the region has no stall shell.
  DenseMap<unsigned, RegionControl> controlOf;
  // Each region's counter widened to `kIndexWidth`. The counter REGISTER is
  // only as wide as its own induction range needs (`RegionBlock::counterType`);
  // a datapath READ of it is an ordinary index, so this is the second, wider
  // wire.
  DenseMap<unsigned, Value> counterIndex;
  // H's output per region. An access or a shared-unit mux is timed against the
  // shell of the region that OWNS it, which need not be the one currently
  // emitting. An unregistered region is RIGID (the default `StallShell`).
  DenseMap<unsigned, StallShell> shellOf;
  DenseMap<uint64_t, Value> streamReadData; // (channel id, access idx) -> the
                                            // input-stream data port value
  DenseMap<uint64_t, Value> survivorOf;    // (region id, result idx) -> latched
                                           // result (accKey-packed)
  DenseMap<uint64_t, Value> callResultVal; // (call id, result idx) -> the child
                                           // instance's scalar result output
                                           // (populated by emitCalls)
  /// Internal mem id -> the hlmem handles holding it, bank-major over the
  /// instances of each bank (`bank * instances + inst`). Empty for a scattered
  /// array, which holds no hlmem. Index it through `memReadCell` /
  /// `memWriteCells` rather than directly.
  DenseMap<unsigned, SmallVector<Value>> memBanks;
  /// The one instance of \p m's bank \p bank that answers read port \p port.
  /// Every instance holds the whole array, so one of them is enough.
  Value memReadCell(const uarch::MemUnit &m, unsigned bank, unsigned port) {
    return memBanks[m.id][bank * m.instances +
                          m.readInstance.lookup(
                              uarch::MemUnit::instanceKey(bank, port))];
  }
  /// Every instance of \p m's bank \p bank: a write reaches all of them, each
  /// copy needing it to stay the same array.
  ArrayRef<Value> memWriteCells(const uarch::MemUnit &m, unsigned bank) {
    return ArrayRef<Value>(memBanks[m.id])
        .slice(bank * m.instances, m.instances);
  }
  /// One backedge per element of a scattered internal array, in flat row-major
  /// order: the register's own output. Declared with the array so a read can
  /// select over the elements before the stores that drive them have emitted,
  /// and resolved by `finalizeScatteredPorts` once they all have.
  DenseMap<unsigned, SmallVector<circt::Backedge>> scatterElems; // by MemId
  DenseMap<unsigned, Value>
      romArray; // ROM mem id -> its hw.aggregate_constant array value
  DenseMap<unsigned, circt::Backedge> regHeadBE; // reg id -> chain head input
  DenseMap<unsigned, ShiftChain> regStages;      // reg id -> its tap chain
  DenseMap<uint64_t, Value> readData;            // (mem,access) -> read data
  DenseMap<unsigned, Value> unitVal;             // unit id -> result
  DenseMap<unsigned, circt::Backedge> unitBE;    // unit id -> result backedge
  DenseMap<unsigned, Value> muxVal;              // mux id -> resolved output

  /// One accessor's drive of a shared physical port: the terms it presents and
  /// the pulse that says it is presenting. A port is reached by several
  /// accessors (the accesses of different regions, a child mastering it) while
  /// each of `hw.output`, `seq.write` and an element register takes it exactly
  /// once, so an arm is built where its accessor emits and only combined by
  /// `commitSink` once every region has emitted.
  struct SinkArm {
    /// This accessor is driving now: a store's commit pulse, a region's
    /// accesses presenting, a child's run window. Null only where the arm holds
    /// the port alone and drives it unconditionally.
    Value fired;
    Value addr; // null on a sink that carries no address
    Value data; // null on a sink that carries no datum
  };
  /// What a shared port carries in a cycle no arm fires.
  enum class Idle {
    DontCare, // nothing samples it: a write whose enable is low
    Hold,     // it must keep the last value: an address bus another region owns
  };
  /// Reduce \p arms onto one driver per term, plus the OR of their pulses. At
  /// most one arm fires in a cycle (`portGraph` separates any two that could
  /// overlap), so the reduction is a one-hot select.
  SinkArm commitSink(ArrayRef<SinkArm> arms, Idle idle);

  /// One channel's port drives, accumulated over every access to it: a FIFO has
  /// a single {data,valid,ready} triple that several accesses time-share, and
  /// `hw.output` takes each port exactly once, so `finalizeStreamPorts` drives
  /// the ports after all regions have emitted.
  struct StreamDrive {
    Value valid;                  // OR of the puts' pulses
    Value ready;                  // OR of the gets' pulses
    SmallVector<SinkArm, 1> puts; // each put's pulse and the token it presents
  };
  SmallVector<StreamDrive> streamDrives; // by StreamId (sized on first use)

  /// Stores to a scattered memory, by MemId: `addr` is the element targeted, at
  /// the memory's address width, and the commit demuxes each arm onto every
  /// element. `finalizeScatteredPorts` drives an argument's element ports or
  /// builds an internal array's registers from them.
  DenseMap<unsigned, SmallVector<SinkArm, 1>> scatterWrites;

  /// One store to an internal array, held back so the stores coloured onto the
  /// same write port can be muxed onto one `seq.write`. A port per static write
  /// defeats block-RAM inference and drops the array into a register file; the
  /// colouring spreads the stores over at most two ports, which infer a true
  /// dual port.
  struct SharedWrite {
    unsigned bank; // the bank this store commits to (0 when unbanked)
    unsigned port; // the write port it was coloured onto
    SinkArm arm;   // `fired` already delayed for the device write latency
  };
  DenseMap<unsigned, SmallVector<SharedWrite, 2>> sharedWrites; // by MemId

  /// One shared read port, keyed by (memory, bank, port). `sharedReadPort`
  /// builds the `seq.read` on the first access to reach it, so its datum is
  /// available before the address that fetches it exists; the address and the
  /// read enable ride backedges `finalizeSharedReadPorts` resolves once every
  /// holder is known. An arm's `fired` is the second of two selects: within a
  /// region `sharedAddress` has already picked between that region's own
  /// accesses.
  struct SharedReadPort {
    Value data;
    circt::Backedge rdEnBE;
    circt::Backedge addr;
    SmallVector<SinkArm, 1> arms;
    /// The one region holding the port, when a region (not a child) does: the
    /// finalize reads its RESOLVED shell off `shellOf`, since the chainEnable
    /// at contribution time is still a promise.
    std::optional<unsigned> ownerRegion;
    unsigned owners = 0; // regions plus mastering children holding the port
  };
  /// A MapVector, not a DenseMap: the finalize iterates it to drive the ports,
  /// and the emitted module must not depend on a hash order.
  llvm::MapVector<std::tuple<unsigned, unsigned, unsigned>, SharedReadPort>
      sharedReads;

  /// Stores to an external array's port group, keyed by the group's base name
  /// (a StringRef into the immutable model's `portBase`/`topBase`) so a child
  /// mastering the same (bank, port) colour joins the accesses' arms. A
  /// MapVector, not a DenseMap: `finalizeBoundaryWritePorts` iterates it to
  /// drive the ports, and the emitted module must not depend on a hash order.
  llvm::MapVector<llvm::StringRef, SmallVector<SinkArm, 2>> boundaryWrites;

  /// The same for a boundary read port group's address output. A group several
  /// regions or children share is one module output, so only
  /// `finalizeSharedReadPorts` may drive it.
  llvm::MapVector<llvm::StringRef, SmallVector<SinkArm, 1>> boundaryReads;

  /// A kernel-local channel's body wires: what a boundary channel reads off its
  /// module ports, an internal one reads off its own `seq.fifo`. Backedges,
  /// since the FIFO can only be built once every access has contributed its
  /// drive.
  struct StreamWires {
    circt::Backedge data;  // the FIFO's show-ahead output
    circt::Backedge valid; // a token is available (~empty)
    circt::Backedge ready; // space is available (~full)
  };
  DenseMap<unsigned, StreamWires> streamWires; // internal channels only

  /// Body wires of a channel whose ends are CHILD PORTS (`callEnds`): the
  /// producer end's `ready` and, per CONSUMER end, its `{data, valid}`. Both
  /// halves are backedges because the child's input ports must exist before the
  /// FIFO that will drive them, and the FIFO needs the child's outputs. One
  /// entry per consumer: several readers are a FAN-OUT, each owning its own
  /// FIFO so per-consumer buffering decouples them.
  struct ComposedWires {
    circt::Backedge prodReady;
    llvm::SmallVector<circt::Backedge, 1> sinkData, sinkValid;
  };
  DenseMap<unsigned, ComposedWires> composedWires; // by StreamId
  /// Each instantiated child's output ports, by name. The channel realization
  /// reads these to find a producer's `{data, valid}` and a consumer's `ready`,
  /// since `emitCalls` builds the instances before the queues between them.
  DenseMap<unsigned, llvm::StringMap<Value>> callOuts; // by CallId

  // The child modules a `dcp.instance`'s CallUnit instantiates (null for
  // a plain leaf with no calls).
  const uarch::CalleeCtx &callees;

  DatapathEmitter(EmitContext &c, const uarch::Datapath &dp,
                  circt::hw::HWModulePortAccessor &pa,
                  const llvm::StringMap<Operation *> &opModules,
                  const uarch::CalleeCtx &callees)
      : c(c), dp(dp), pa(pa), opModules(opModules), callees(callees) {}

  static uint64_t accKey(unsigned m, unsigned a) {
    return (uint64_t(m) << 32) | a;
  }

  /// Resolve a datapath Source to the SSA value driving it.
  Value resolveSource(const uarch::Source &s);
  /// The window a recurrence input reads its reduction identities in: region
  /// \p rb's counter still inside its first \p dist iterations. A level, valid
  /// when the region issues, which a consumer delays to its own stage.
  Value firstIterations(const uarch::RegionBlock &rb, unsigned dist);
  /// The single iteration that reads identity \p iter: \p rb's counter at
  /// `lb + iter*step`. The same kind of level as `firstIterations`.
  Value atIteration(const uarch::RegionBlock &rb, unsigned iter);
  /// \p rb's counter and its lower bound, at the counter register's width.
  std::pair<Value, Value> counterAndLb(const uarch::RegionBlock &rb);
  /// The counter value \p rb's n-th iteration holds, `lb + n*step` at that
  /// width; null for n == 0, which is \p lb itself.
  Value ivAt(const uarch::RegionBlock &rb, unsigned n, Value lb);
  /// One cone \p r of this access's address as hardware at \p width: a
  /// constant, one register per strength-reduced term, and whatever did not
  /// reduce, evaluated.
  Value buildAddr(const uarch::MemUnit::Access &acc,
                  const uarch::MemUnit::Access::Reduced &r, unsigned width);
  /// The address hardware of an access: the element index within the bank it
  /// reaches, plus the bank digit when that is decided at runtime. The runtime
  /// dual of the static split (`dcp-resolve-banking`), routing an element to
  /// the same bank off the cones `planAddressGenerators` reduced.
  BankSplit bankAddress(const uarch::MemUnit &m,
                        const uarch::MemUnit::Access &acc);
  /// Narrow a linear address to a memory's clog2(depth)-bit index (hlmem).
  Value memAddr(const uarch::MemUnit &m, Value addr);
  /// Which element of a scattered memory an access names, at the datapath
  /// width (compared against literal element numbers, not used to index).
  Value scatterIndex(const uarch::MemUnit &m,
                     const uarch::MemUnit::Access &acc);
  /// The element registers of a scattered internal array, in element order.
  SmallVector<Value> scatterValues(unsigned id);
  /// \p v delayed to land with the datum of a read of \p m: a bank select and a
  /// constant table's own output both have to reach the consumer on the cycle
  /// the data does.
  Value atReadData(const uarch::MemUnit &m, Value v, const StallShell &sh);

  /// Bind external read-data input ports into readData (once, before regions).
  void bindReadPorts();
  /// Instantiate seq.hlmem storage for each internal (non-argument) memory.
  void createInternalMemories();
  /// Wire a region's controller output into the datapath, the G->F seam. Each
  /// field is absent where its controller publishes none, hence the
  /// field-by-field copy rather than a whole-struct assignment.
  void setControl(unsigned region, const RegionControl &rc) {
    RegionControl &slot = controlOf[region];
    if (rc.counter) {
      slot.counter = rc.counter;
      counterIndex[region] =
          resize(c.b, c.loc, rc.counter, kIndexWidth, /*isSigned=*/true);
    }
    slot.issue = rc.issue;
    if (rc.wantIssue)
      slot.wantIssue = rc.wantIssue;
    if (rc.running)
      slot.running = rc.running;
    if (rc.phase)
      slot.phase = rc.phase;
    if (!rc.scaledCounters.empty())
      slot.scaledCounters = rc.scaledCounters;
  }
  /// Record a region's latched result \p port so a sibling reading
  /// Source::Survivor{region, port} resolves to it.
  void setSurvivor(unsigned region, unsigned port, Value v) {
    survivorOf[accKey(region, port)] = v;
  }
  /// Register region \p region's stall shell, the H seam. The orchestrator
  /// registers a PROMISE (two backedges) before F and G emit against it, then
  /// re-registers the derived shell once `deriveStallShell` resolves them.
  void setShell(unsigned region, const StallShell &sh) { shellOf[region] = sh; }
  /// Region \p region's stall shell; rigid for an unregistered region.
  StallShell shellFor(unsigned region) const { return shellOf.lookup(region); }

  /// The part of \p rb's datapath that precedes the units, for both the leaf
  /// path (`emit`) and a container's condition cone (`emitConditionRegion`):
  /// the delay chains, the unit backedges (a read address may read a unit
  /// emitted later) and the reads, whose data the units consume.
  void emitBeforeUnits(const uarch::RegionBlock &rb, Value issue);
  /// The part that follows the units: the register heads, then the boundary
  /// read addresses, which may be computed by a unit and so need its filled
  /// value rather than its backedge.
  void emitAfterUnits(const uarch::RegionBlock &rb, Value issue);

  void emitRegisters(const uarch::RegionBlock &rb);
  /// Backedge every unit output before any consumer resolves it, so a read
  /// address or another unit input may reference a unit emitted later.
  void declareUnits(const uarch::RegionBlock &rb);
  /// Bind the datum of every read scheduled in region \p rb into `readData`,
  /// before `emitUnits` consumes it. One arm per `PortPlan`; the one it cannot
  /// serve is a boundary port group, whose address may be computed by a unit
  /// (`emitExternalReadAddrs`) and whose datum `bindReadPorts` already bound.
  void emitReads(const uarch::RegionBlock &rb, Value issue);
  /// The address one region's accesses on a port present: each drives it on its
  /// own issue cycle, held with the datapath so a read frozen by back-pressure
  /// keeps re-presenting its address. \p idxs indexes `m.accesses`, all in the
  /// region \p sh and \p issue belong to. \p fired, when given, additionally
  /// receives "one of them is presenting now", which a port another region also
  /// holds selects on; a lone region on a port drives it unconditionally.
  Value sharedAddress(const uarch::MemUnit &m, ArrayRef<unsigned> idxs,
                      Value issue, const StallShell &sh,
                      Value *fired = nullptr);
  /// Stamp an emitted `seq.read`/`seq.write` with the physical port it drives
  /// (`kMemPortAttr`), which puts a port's read and write in one `always`
  /// block and so makes them one port of a dual-port RAM.
  template <typename OpT> OpT atPort(OpT op, unsigned port) {
    op->setAttr(kMemPortAttr, c.b.getI64IntegerAttr(port));
    return op;
  }
  /// Drive the read-address port of each single-interface external port group
  /// in region \p rb (unbanked or statically banked); the data-dependent ones
  /// are `emitReads`. Runs after the units, so an address computed by
  /// one resolves to its filled value rather than a dangling backedge.
  void emitExternalReadAddrs(const uarch::RegionBlock &rb, Value issue);
  /// Region \p rb's compute units: native -> comb, IP -> an instance of the
  /// extern operator module. A loop-carried input re-injects `inputInits[k][n]`
  /// on the n-th iteration; a container's own units carry none
  /// (`assertModelInvariants`).
  void emitUnits(const uarch::RegionBlock &rb);
  /// Emit a sequential (CHECK/RUN) while's condition cone: the container's OWN
  /// condition memory reads plus its compute. Returns the settled condition
  /// with its ready latency `t_cond`, the cycles after CHECK-start at which it
  /// is valid (0 for a combinational condition). The read address is the frozen
  /// iter-arg survivor, so the loaded value is a stable wire across the CHECK
  /// window; the caller samples it at `delayValid(checkStart, t_cond)`.
  std::pair<Value, unsigned> emitConditionRegion(const uarch::RegionBlock &rb,
                                                 const uarch::Source &condSrc);
  void resolveRegHeads(const uarch::RegionBlock &rb);
  /// Every write scheduled in region \p rb, gated by \p issue, folding the
  /// deepest store's stage into \p fb.storeDrain. One arm per `PortPlan`, as
  /// `emitReads`.
  void emitWrites(const uarch::RegionBlock &rb, Value issue,
                  DatapathFeedback &fb);

  /// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
  /// `hw.instance` and fold the child's `done` into \p fb.callDone. Runs BEFORE
  /// the region's own register heads and accesses, since a call's scalar result
  /// is an ordinary datapath Source a register chain or a store may read.
  void emitCalls(const uarch::RegionBlock &rb, Value issue,
                 DatapathFeedback &fb);
  /// Master each memref operand of child \p cu from its instance outputs
  /// \p outs. One arm per `PortPlan`, as `emitReads` and `emitWrites`.
  /// \p rdBackedge holds the read-data promise each of the child's read ports
  /// waits on; \p runWindow is the window the child owns a port a second
  /// accessor also holds, built on demand.
  void masterCallPorts(const uarch::CallUnit &cu, llvm::StringMap<Value> &outs,
                       llvm::StringMap<circt::Backedge> &rdBackedge,
                       llvm::function_ref<Value()> runWindow,
                       const StallShell &sh);
  /// The start pulse of one child, from the start-policy table read on this
  /// node's contract and its region's composition class.
  Value startForCall(const uarch::CallUnit &cu, Value issue,
                     llvm::ArrayRef<Value> predDones, bool concurrent,
                     const StallShell &sh);
  /// The queue(s) behind a channel whose ends are child ports: one `seq.fifo`
  /// per consumer end (the fan-out tee), each optionally fronted by a seeded
  /// channel's init-prepend shim, and a pass-through where one end is a
  /// boundary port of this module rather than a child.
  void emitComposedChannel(const uarch::StreamChannel &s);

  /// Declare each kernel-local channel's body wires (`streamWires`) before any
  /// region reads them; `finalizeStreamPorts` builds the FIFO that resolves
  /// them.
  void declareInternalChannels();
  /// One channel's three handshake signals, wherever they live: a boundary
  /// channel's module ports, or a kernel-local channel's own FIFO.
  Value streamData(const uarch::StreamChannel &s);
  Value streamValid(const uarch::StreamChannel &s);
  Value streamReady(const uarch::StreamChannel &s);

  /// Bind each input stream's `_data` module port into `streamReadData` (once,
  /// before any consumer), so a Source::Stream resolves like a memory read.
  void bindStreamReads(const uarch::RegionBlock &rb);
  /// H for one region: wire region \p rb's stream handshakes and return the
  /// stall shell they derive. An input contributes its `_ready` (gated so a
  /// full output holds intake too), an output its `_data` plus `_valid`; the
  /// region's stalls become `{chainEnable, issueEnable}`, split by whether the
  /// blocked handshake belongs to an in-flight iteration or to the pass about
  /// to issue, and each put's stage folds into \p fb.storeDrain. Runs on the
  /// already-emitted (F, G) pair, timing its own deeper pulses against the
  /// region's registered promise; the caller resolves that promise with the
  /// result.
  StallShell deriveStallShell(const uarch::RegionBlock &rb, Value issue,
                              DatapathFeedback &fb);
  /// Drive every boundary channel's module ports, and build every local
  /// channel's `seq.fifo`, from the accumulated `streamDrives`. Call exactly
  /// once, after all regions have emitted and before `hw.output`.
  void finalizeStreamPorts();
  /// Drive each scattered argument's per-element data + write-enable outputs
  /// from the accumulated `scatterWrites`. Call exactly once, after all regions
  /// have emitted and before `hw.output`; a read-only scattered argument has no
  /// output port and drives nothing.
  void finalizeScatteredPorts();
  void finalizeSharedWritePorts();
  /// The `seq.read` of \p m's bank \p bank on read port \p port, built on the
  /// first access to reach it and reused by every later one. Its address is a
  /// backedge: the accesses sharing a port drive it on their own cycles, so it
  /// is only known once they have all emitted.
  Value sharedReadPort(const uarch::MemUnit &m, unsigned bank, unsigned port);
  /// Drive each shared read port's one address bus from the arms the regions
  /// holding it contributed, and each shared boundary read group's address
  /// output likewise. Call exactly once, with the same timing as the write
  /// finalizes.
  void finalizeSharedReadPorts();
  /// Whether read port \p port of \p m's bank \p bank is held by more than one
  /// accessor: a region whose own accesses reach it, or a child that masters
  /// it. Counts holders rather than regions, the two kinds saying they are
  /// driving in different ways.
  bool sharedInternalPort(const uarch::MemUnit &m, unsigned bank,
                          unsigned port) const;
  /// Drive each merged boundary write port group from the stores coloured onto
  /// it. Call exactly once, with the same timing as the two above.
  void finalizeBoundaryWritePorts();
  /// Build one kernel-local channel's `seq.fifo` from its accumulated drives
  /// (\p data is the puts' muxed token) and resolve its `streamWires`.
  void emitInternalChannel(const uarch::StreamChannel &s, Value data);

  /// Emit region \p rb's whole datapath (F) given the controller's \p issue;
  /// returns its store feedback. Times everything against the region's
  /// registered shell; deriving that shell (H) is the orchestrator's separate
  /// step, run on what this emits.
  DatapathFeedback emit(const uarch::RegionBlock &rb, Value issue);
};

//===----------------------------------------------------------------------===//
// HWEmitter: the orchestrator. Owns the context + both emitters and drives the
// region tree (sibling hand-off, container nesting), wiring the typed seam.
//===----------------------------------------------------------------------===//
struct HWEmitter {
  EmitContext ctx;
  ControlEmitter control;
  DatapathEmitter datapath;
  const uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;

  HWEmitter(OpBuilder &b, Location loc, const uarch::Datapath &dp,
            circt::hw::HWModulePortAccessor &pa,
            const llvm::StringMap<Operation *> &opModules,
            circt::BackedgeBuilder &bb, Type i1, Type i32,
            const uarch::CalleeCtx &callees)
      : ctx(b, loc, bb, i1, i32), control(ctx),
        datapath(ctx, dp, pa, opModules, callees), dp(dp), pa(pa) {
    ctx.countedDelayCycles = dp.countedDelayCycles;
  }

  /// The counted terminator of region \p rb: each bound resolved from its
  /// runtime Source (a dynamic trip) or the constant fast path. Empty for an
  /// acyclic region; a while builds its own Terminator::conditional.
  Terminator terminatorOf(const uarch::RegionBlock &rb);
  /// Emit one region and return its `done`. A leaf runs one imperative path for
  /// every regime (counted / dynamic-trip / while): control -> datapath ->
  /// resolve the F->G condition, capture results, done. A container runs its
  /// children once per outer iteration.
  Value emitRegion(const uarch::RegionBlock &rb, Value start, bool retrig);
  /// A loop-over-call region: a counted `dcp.pipeline` wrapping one
  /// `dcp.instance`. One child instance is fired \p tripCount times, a counter
  /// driving its index and each invocation advancing on the child's real
  /// `done`, so throughput is one iteration per child latency rather than the
  /// pipeline cadence.
  Value emitLoopCall(const uarch::RegionBlock &rb, Value start);
  /// The final iteration's issue pulse: a counted region's last iteration
  /// (counter+1 reaches the bound), a while's condition-false exit, or the
  /// issue pulse itself for an acyclic region. The pulse `emitDone` and
  /// `captureResults` both key off.
  Value lastIssuePulse(const RegionControl &rc, const Terminator &term);
  /// Capture LEAF region \p rb's results into the survivor registers a sibling
  /// reads, each at its own ready cycle relative to \p captureOn; returns the
  /// region's result-drain stage. \p captureOn is the last iteration's issue
  /// pulse for a counted loop and each continuing iteration's for a while.
  unsigned captureResults(const uarch::RegionBlock &rb, Value captureOn,
                          Value start);
  /// Run \p regions in program order, each starting when its predecessor drains
  /// (the first on \p start); returns the last region's done. The shared
  /// sequencer for func-scope siblings and a container's children.
  Value sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                 bool retrig);
  /// Compose the func-scope sibling regions by their dependence DAG
  /// (`rb.predecessors`): a predecessor-free region starts with the kernel
  /// \p start (independent siblings run concurrently), the rest on the rising
  /// edge of their predecessors' joined `done`. The returned kernel `done` is
  /// the conjunction of every region's, so it completes when the last does.
  Value composeSiblings(llvm::ArrayRef<uarch::RegionId> regions, Value start);
  /// Set up a container's loop-carried iter-args as frozen survivor registers
  /// (latch each `rb.results[k].init` at \p start, advance on \p advance),
  /// record each as Source::Survivor{rb, k}, and return the per-arg next-value
  /// backedges, set after the children emit since the next value comes from
  /// them.
  llvm::SmallVector<circt::Backedge>
  setupCarriedIterArgs(const uarch::RegionBlock &rb, Value start,
                       Value advance);
  /// A counted container: wire `emitCountedIteration` to a body that sequences
  /// its children, so the outer counter advances when the last child drains. A
  /// cross-region result crosses child-to-child as a survivor register.
  Value emitContainer(const uarch::RegionBlock &rb, Value start);
  /// A conditional container (a sequential-wrapper while): the same
  /// per-iteration child sequencing as emitContainer, but the outer iter-args
  /// are frozen survivor registers advanced by the children's results and the
  /// loop terminates on a continue-condition re-evaluated over them, so the
  /// controller is `emitCheckedIteration` rather than the counted one.
  Value emitConditionalContainer(const uarch::RegionBlock &rb, Value start);
  /// A guard region (a dcp.select): a predicated container whose children run
  /// once iff the held predicate (`rb.condition`) holds, else are skipped. The
  /// predicate start-gates child 0 (`start & cond`); a false predicate
  /// completes the region in one cycle (`start & ~cond`) without ever issuing
  /// the children, so their stores never fire.
  Value emitGuard(const uarch::RegionBlock &rb, Value start);
  /// Emit the whole module body: preamble + each top-level region in order.
  void emit();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_HWEMIT_H
