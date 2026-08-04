/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// PER-REGION emission, split by role along the control/datapath (F/G) seam.
//
//   * ControlEmitter is control (G): every per-region controller plus the
//     completion signal. SCHEDULE-paced controllers cover the free-running,
//     modulo and while regimes (`emitPipelined`); DONE-paced cover the counted
//     container and loop-over-call (`emitCountedIteration`) and the CHECK/RUN
//     while (`emitCheckedIteration`).
//   * DatapathEmitter is datapath (F): register chains, compute units, memory,
//     addressing, and Source resolution (`resolveSource`).
//   * HWEmitter is the orchestrator. It wires the two per region and composes
//     sibling and nested regions.
//
// The control/datapath seam is typed: a controller returns a `RegionControl
// {issue, counter}`; the datapath returns its store drain (the deepest
// store's stage) through `DatapathFeedback`.
//
// Elasticity (H) is a third derivation, not traffic across that seam. A
// region's `StallShell` (Primitives.h) is derived once by `deriveStallShell`
// and handed to each side explicitly: G takes `issueEnable`, F takes
// `chainEnable`. It is keyed per region (`DatapathEmitter::shellOf`), so a
// Source is timed against the shell of the region that OWNS it.
//
// The two layers this one sits between are their own headers.
//   * Primitives.h is the emission substrate (EmitContext, ShiftChain, the
//     banking crossbar, type/width and comb-lowering rules). It knows nothing
//     about regions or controllers.
//   * EmitDriver.h is the module-level driver: the pass entry, the call-DAG
//     walk, and port/operator declaration.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_HWEMIT_H
#define ALLO_MICROARCH_HWEMIT_H

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
// Terminator: what ends a pipelined region's run, either a counter reaching
// an iteration bound (`Counted`, a counted loop) or a data-dependent condition
// going false (`Conditional`, a while). The last-issue and start-gate tests,
// shared by the pipelined control skeleton (emitPipelined) and the
// survivor-capture path.
//===----------------------------------------------------------------------===//
struct Terminator {
  enum class Kind { Counted, Conditional };
  Kind kind = Kind::Counted;
  // Induction bounds. `lb`/`step` seed the counter register (init lb, +=step)
  // and `ub` its termination (iv+step >= ub). A while free-runs a 0-based
  // counter (lb=0/step=1, ub null), terminating on ~cond.
  Value lb, ub, step;
  Value cond; // Conditional: the i1 continue condition (a datapath value)

  static Terminator counted(Value lb, Value ub, Value step) {
    return {Kind::Counted, lb, ub, step, Value()};
  }
  static Terminator conditional(Value cond, Value zero, Value one) {
    return {Kind::Conditional, zero, Value(), one, cond};
  }

  /// The iteration issued at `iv` is the last one: the next induction value
  /// reaches the upper bound (iv+step >= ub), or the continue-condition is
  /// false (~cond). \p ivStep is `iv + step`.
  Value isLast(EmitContext &c, Value ivStep) const {
    return kind == Kind::Conditional ? c.notBit(cond) : c.icmpSgeV(ivStep, ub);
  }
  /// The region is empty (issues nothing): lb >= ub. A while is never "empty"
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
// ControlEmitter (G): a per-region control regime + its completion signal. The
// controller's output to the datapath is a RegionControl; it consumes a
// resolved Terminator (a datapath value for a dynamic bound / while condition)
// but never itself walks a Source.
//===----------------------------------------------------------------------===//
struct RegionControl {
  Value issue; // pipeline issue / valid signal (already gated by the stall
               // shell's enable)
  Value
      counter; // iteration index (Source::Counter); null for an acyclic region
  Value wantIssue; // the UNgated per-cycle issue desire (issue before `&
                   // enable`): the stall shell hazards a stage-0 stream access
                   // on this (not the gated issue) to stay combinationally
                   // acyclic. Equals `issue` when there is no stall.
  Value running;   // the region is executing: the level the counter reloads its
                   // lower bound while low. Null for a done-driven controller,
                   // whose counter reloads on `start` instead.
  /// The region's time base: the modulo phase [0, ii), reloaded on `start` and
  /// advancing on every enabled cycle after it, drain included. An iteration
  /// issues at phase 0, so an op landing at cycle `r` lands at phase `r % ii`.
  /// Null unless a schedule-paced controller runs this region at II > 1.
  Value phase;
  /// One register per `RegionBlock::addrStrides` entry, holding that multiple
  /// of `counter`. Emitted beside the counter and updated by the same
  /// expression, so the two cannot fall out of step.
  llvm::SmallVector<Value> scaledCounters;
};

//===----------------------------------------------------------------------===//
// DatapathFeedback (F -> G): the store timing a control regime consumes to
// compute completion.
//===----------------------------------------------------------------------===//
struct DatapathFeedback {
  // The deepest store's schedule stage (max dcpStart over the region's
  // stores); 0 if it stores nothing. The region's `done` waits until the last
  // iteration's last store has been presented + committed, so a multi-store
  // region cannot complete prematurely. A stream put folds into this too.
  unsigned storeDrain = 0;
  // The `done` of a CallUnit region's child instance. When set, it IS the
  // region's completion, bypassing the store-drain `emitDone`. Null for a
  // call-free region.
  Value callDone;
};

//===----------------------------------------------------------------------===//
// IterationControl: the output of a DONE-DRIVEN controller, one whose
// iterations are paced by the body COMPLETING rather than by the schedule's
// own cadence (as `emitPipelined`'s regimes are). A container, a
// loop-over-call and a CHECK/RUN while each launch the next pass only once
// the current one has drained, sharing one skeleton (boundary pulse, continue
// test, launch, done latch) that differs only in the cell of (shape x
// termination) they occupy.
//
// `rc` is the same G->F seam a leaf hands over: `issue` is the body-launch
// pulse and `counter` the IV. `done` is the region's completion level, latched
// here rather than by `emitDone`, since a done-driven region completes on an
// iteration boundary and has no drain to count.
//===----------------------------------------------------------------------===//
struct IterationControl {
  RegionControl rc;
  Value done;
};

struct ControlEmitter {
  EmitContext &c;
  explicit ControlEmitter(EmitContext &c) : c(c) {}

  /// Pick the control shape for region \p rb, acyclic (straight-line) or a
  /// pipelined loop, and emit it, driven by the \p start pulse against the
  /// resolved \p term (a counted bound or a while's continue-condition).
  /// Returns {issue, counter}.
  RegionControl emitPipelineControl(const uarch::RegionBlock &rb,
                                    const Terminator &term, Value start,
                                    const StallShell &sh) const;
  /// The scaled counters of \p rb: one register per `addrStrides` entry,
  /// holding that multiple of the region's counter, each at its OWN width
  /// (`AddrStride::width`) rather than the counter's. \p update is the
  /// counter's own next-value expression with `lb` and `step` scaled, supplied
  /// by the caller since the two controller families disagree about it. \p
  /// bypassStart mirrors a done-driven counter's start-cycle bypass (null for
  /// none).
  llvm::SmallVector<Value> emitScaledCounters(
      const uarch::RegionBlock &rb, Value bypassStart,
      llvm::function_ref<Value(Value cur, Value stepped, Value init)> update)
      const;
  /// The one pipelined control skeleton for the free-running (II==1), modulo
  /// (II>1), and while (flushing) regimes: they share a `running` flag + an
  /// iteration counter and differ only in \p term (a counter reaching a bound
  /// vs the continue-condition going false) and, for II>1, a phase counter
  /// gating issue. Non-speculative for a conditional terminator (II >= t_cond,
  /// so no doomed iteration issues); no backpressure (fixed-latency memory, no
  /// FIFO). A conditional \p term carries `cond` as a datapath value (F->G),
  /// typically a backedge resolved after the datapath emits its producer.
  /// \p sh is G's half of the region's elasticity (H): issue is gated
  /// `wantIssue & sh.issueEnable`, so a stalled cycle issues nothing. A rigid
  /// shell leaves issue ungated.
  /// \p region names the emitted state cells (`r<id>_run` / `_iv` / `_phase`).
  RegionControl emitPipelined(unsigned region, int64_t ii,
                              const Terminator &term, Value start,
                              const StallShell &sh) const;
  /// The straight-line control skeleton: one pass, no counter. \p sh is the
  /// same G-half contract `emitPipelined` honors: the single pass is DEFERRED
  /// while `issueEnable` is low rather than dropped, which lets a stage-0
  /// stream access wait for its handshake. A rigid shell issues unconditionally
  /// and builds no state at all.
  RegionControl emitAcyclic(unsigned region, Value start, bool topLevel,
                            const StallShell &sh) const;

  /// The counted done-driven controller: `Container` and `CallNode` x
  /// `CountedStatic`. Runs one body pass per iteration of \p term, launching
  /// the next when \p complete pulses (the body's drain edge, a Backedge the
  /// caller resolves once the body has emitted) and completing when the last
  /// one drains. \p rb supplies the counter's name and, through its `shape`,
  /// whether the FIRST pass may launch on \p start itself.
  IterationControl emitCountedIteration(const uarch::RegionBlock &rb,
                                        const Terminator &term, Value start,
                                        Value complete) const;
  /// The conditional done-driven controller: `Container` x `Conditional`, a
  /// sequential-wrapper while. A CHECK pulse one cycle after \p start and after
  /// each body drain (\p complete) re-evaluates \p cond on the settled
  /// iter-args, \p tCond cycles later for a condition that reads memory or an
  /// IP. The pulse then forks: continue (launch the body again) or finish.
  /// Non-speculative and stall-free, the same family as a leaf while. The
  /// region has no counter, so the returned `rc.counter` is null.
  IterationControl emitCheckedIteration(unsigned region, Value cond,
                                        unsigned tCond, Value start,
                                        Value complete) const;

  /// The region's completion signal: one latched level for every regime. It
  /// rises when \p lastIssue (the final iteration's issue pulse) delayed \p
  /// drainStage cycles (the deepest store or result stage), or immediately on
  /// \p emptyDone (an empty region completes in one cycle; null when
  /// unreachable). The latch's register cycle is the store/result commit
  /// cycle, so a sibling starting on this done reads every committed store and
  /// survivor. A \p retrig region resets its completion state on \p start, and
  /// its returned level additionally reads 0 on the \p start cycle itself: a
  /// region whose completion pulse COINCIDES with \p start (e.g. an acyclic
  /// one draining at stage 0) would otherwise latch high on the first pass and
  /// never produce a later rising edge.
  /// \p sh times the drain: under an elastic shell the completion pulse is
  /// held through back-pressure, since the last store/token is not committed
  /// until it is accepted.
  Value emitDone(unsigned region, unsigned drainStage, Value lastIssue,
                 Value emptyDone, Value start, bool retrig,
                 const StallShell &sh) const;
};

//===----------------------------------------------------------------------===//
// DatapathEmitter (F): register chains, compute units, memory access, and the
// uniform Source resolution `src`. Reads the controller's `issue`/`counter`
// (via setControl) and returns the region's store drain.
//===----------------------------------------------------------------------===//
struct DatapathEmitter {
  EmitContext &c;
  uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;
  const DenseMap<unsigned, Operation *> &unitModule;

  // A region's controller outputs (RegionControl: issue / counter / ungated
  // wantIssue), the G->F seam. `counter` is null for an acyclic region;
  // `wantIssue` is null when the region has no stall shell.
  DenseMap<unsigned, RegionControl> controlOf;
  // Each region's counter widened to `kIndexWidth`, built once at that seam.
  // The counter REGISTER is only as wide as its own induction range needs
  // (`RegionBlock::counterType`); a datapath READ of it (an address-cone
  // operand, a compute unit's `index` operand, a child's index port) is an
  // ordinary index, so this is the second, wider wire.
  DenseMap<unsigned, Value> counterIndex;
  // H's output per region: the stall shell that region's timing runs against.
  // An access or a shared-unit mux is timed against the shell of the region
  // that OWNS it, which need not be the one currently emitting. An
  // unregistered region is RIGID (the default `StallShell`).
  DenseMap<unsigned, StallShell> shellOf;
  DenseMap<uint64_t, Value> streamReadData; // (channel id, access idx) -> the
                                            // input-stream data port value
  DenseMap<uint64_t, Value> survivorOf;    // (region id, result idx) -> latched
                                           // result (accKey-packed)
  DenseMap<uint64_t, Value> callResultVal; // (call id, result idx) -> the child
                                           // instance's scalar result output
                                           // (populated by emitCalls)
  DenseMap<unsigned, SmallVector<Value>>
      memBanks; // internal mem id -> its bank hlmem handle(s) (one unless
                // banked)
  DenseMap<unsigned, Value>
      romArray; // ROM mem id -> its hw.aggregate_constant array value
  DenseMap<unsigned, circt::Backedge> regHeadBE; // reg id -> chain head input
  DenseMap<unsigned, ShiftChain> regStages;      // reg id -> its tap chain
  DenseMap<uint64_t, Value> readData;            // (mem,access) -> read data
  DenseMap<unsigned, Value> unitVal;             // unit id -> result
  DenseMap<unsigned, circt::Backedge> unitBE;    // unit id -> result backedge
  DenseMap<unsigned, Value> muxVal;              // mux id -> resolved output
  DenseMap<unsigned, Value> scopeVal; // ScopeUnit id -> emitted comb value

  /// One channel's port drives, accumulated over every access to it. A FIFO
  /// has a single {data,valid,ready} triple that several accesses time-share.
  /// Since `hw.output` takes each port exactly once, every region contributes
  /// its terms here and `finalizeStreamPorts` drives the ports once, after all
  /// regions have emitted.
  struct StreamDrive {
    Value valid;                                  // OR of the puts' pulses
    Value ready;                                  // OR of the gets' pulses
    SmallVector<std::pair<Value, Value>, 1> data; // (put pulse, its data)
  };
  SmallVector<StreamDrive> streamDrives; // by StreamId (sized on first use)

  /// One store to a scattered argument, as its element ports see it. The SAME
  /// N ports are shared by every write to that argument, so they cannot be
  /// driven where the store is emitted. Every write records its terms here and
  /// `finalizeScatteredPorts` drives each element once, after all regions have
  /// emitted, exactly as `StreamDrive` does for a channel's one handshake.
  ///
  /// The commit pulse is region-scoped (timed against that region's stall
  /// shell and issue), which is why it is BUILT at the store and only COMBINED
  /// here.
  struct ScatterWrite {
    Value we;    // this store's commit pulse
    Value index; // the element it targets, at the memory's address width
    Value data;  // the datum it presents
  };
  DenseMap<unsigned, SmallVector<ScatterWrite, 1>> scatterWrites; // by MemId

  /// One store to an internal array, held back so the stores sharing its write
  /// port can be muxed onto one `seq.write`. A port per static write defeats
  /// block-RAM inference and drops the array into a register file, which the
  /// bed pays 655k LUTs for. Combined by `finalizeSharedWritePorts`, for the
  /// same reason `ScatterWrite` exists: the writes come from different regions
  /// and calls, so a port can only be driven once all of them have emitted.
  struct SharedWrite {
    unsigned bank; // the bank this store commits to (0 when unbanked)
    unsigned port; // the write port it was coloured onto
    Value addr;
    Value data;
    Value we; // commit pulse, already delayed for the device write latency
  };
  DenseMap<unsigned, SmallVector<SharedWrite, 2>> sharedWrites; // by MemId

  /// Which write port each access of an internal array drives
  /// (`Datapath::writePortColouring`), for the arrays whose writes may share
  /// ports at all. An array is absent when the colouring refused it, and when
  /// it presents no single addressable write port to merge onto: an external,
  /// scattered or skewed one, or one a dynamically banked store drives every
  /// bank of behind a demux. Held because the colouring is a clique search and
  /// every store would otherwise redo it.
  DenseMap<unsigned, SmallVector<unsigned>> writePortOf; // by MemId

  /// A kernel-local channel's body wires: what a boundary channel reads off
  /// its module ports, an internal one reads off its own `seq.fifo`. Declared
  /// as backedges before any region emits (`declareInternalChannels`) and
  /// resolved by `finalizeStreamPorts`, since the FIFO can only be built once
  /// every access has contributed its drive.
  struct StreamWires {
    circt::Backedge data;  // the FIFO's show-ahead output
    circt::Backedge valid; // a token is available (~empty)
    circt::Backedge ready; // space is available (~full)
  };
  DenseMap<unsigned, StreamWires> streamWires; // internal channels only

  /// Body wires of a channel whose ends are CHILD PORTS (`callEnds`): the
  /// producer end's `ready` and, per CONSUMER end, its `{data, valid}`. Both
  /// halves are backedges because the child's input ports must exist before
  /// the FIFO that will drive them, and the FIFO needs the child's outputs.
  /// Declared by `declareInternalChannels`, resolved by `emitComposedChannel`.
  ///
  /// One entry per consumer: several readers are a FAN-OUT, each owning its
  /// own FIFO so per-consumer buffering decouples them.
  struct ComposedWires {
    circt::Backedge prodReady;
    llvm::SmallVector<circt::Backedge, 1> sinkData, sinkValid;
  };
  DenseMap<unsigned, ComposedWires> composedWires; // by StreamId
  /// Each instantiated child's output ports, by name. The channel realization
  /// reads these to find a producer's `{data, valid}` and a consumer's
  /// `ready`, since `emitCalls` builds the instances before the queues between
  /// them.
  DenseMap<unsigned, llvm::StringMap<Value>> callOuts; // by CallId

  // The child modules a `dcp.instance`'s CallUnit instantiates (null for
  // a plain leaf with no calls).
  const uarch::CalleeCtx *callees = nullptr;

  DatapathEmitter(EmitContext &c, uarch::Datapath &dp,
                  circt::hw::HWModulePortAccessor &pa,
                  const DenseMap<unsigned, Operation *> &unitModule,
                  const uarch::CalleeCtx *callees = nullptr)
      : c(c), dp(dp), pa(pa), unitModule(unitModule), callees(callees) {}

  static uint64_t accKey(unsigned m, unsigned a) {
    return (uint64_t(m) << 32) | a;
  }

  /// Resolve a datapath Source to the SSA value driving it.
  Value resolveSource(const uarch::Source &s);
  /// The cycle a freshly-produced Source's value lands, relative to the
  /// issuing pulse of the iteration that produced it: a compute unit's op slot
  /// + its latency, a memory read's slot + read latency, or 0 for an at-issue
  /// constant. Used by survivor capture.
  unsigned readyCycle(const uarch::Source &s) const;
  /// The resolved (already stage-delayed) index sources an access's affine map
  /// is evaluated over, dims then symbols.
  llvm::SmallVector<Value> addrSources(const uarch::MemUnit::Access &acc);
  /// One cone \p r of this access's address as hardware at \p width: a
  /// constant, one register per strength-reduced term, and whatever did not
  /// reduce, evaluated.
  Value buildAddr(const uarch::MemUnit::Access &acc,
                  const uarch::MemUnit::Access::Reduced &r, unsigned width);
  /// The address hardware of an access: the element index within the bank it
  /// reaches, plus the bank digit when that is decided at runtime. The runtime
  /// dual of the static split (`dcp-resolve-banking`), both deriving from the
  /// memref's `BankLayout` and routing an element to the same bank. Also
  /// covers an unbanked memref, whose one-bank offset is the flat index.
  BankSplit bankAddress(const uarch::MemUnit &m,
                        const uarch::MemUnit::Access &acc);
  /// Narrow a linear address to a memory's clog2(depth)-bit index (hlmem).
  Value memAddr(const uarch::MemUnit &m, Value addr);
  /// Which element of a scattered argument an access names, at the datapath
  /// width (compared against literal element numbers, not used to index).
  Value scatterIndex(const uarch::MemUnit &m,
                     const uarch::MemUnit::Access &acc);

  /// Bind external read-data input ports into readData (once, before regions).
  void bindReadPorts();
  /// Instantiate seq.hlmem storage for each internal (non-argument) memory.
  void createInternalMemories();
  /// Wire a region's controller output into the datapath, the G->F seam. The
  /// counter answers Source::Counter; the issue pulse times a fused
  /// accumulator's init injection; the phase paces a folded value chain. Each
  /// is absent where its controller publishes none, hence the field-by-field
  /// copy: a counter for an acyclic region, `wantIssue` without a stall shell,
  /// a phase outside a schedule-paced region at II > 1.
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
  /// Record a region's latched result \p port (from the orchestrator's
  /// survivor capture) so a sibling reading Source::Survivor{region, port}
  /// resolves to it.
  void setSurvivor(unsigned region, unsigned port, Value v) {
    survivorOf[accKey(region, port)] = v;
  }
  /// Register region \p region's stall shell, the H seam. The orchestrator
  /// registers a PROMISE (two backedges) before F and G emit against it, then
  /// re-registers the derived shell once `deriveStallShell` has resolved them.
  void setShell(unsigned region, const StallShell &sh) { shellOf[region] = sh; }
  /// Region \p region's stall shell; rigid for an unregistered region.
  StallShell shellFor(unsigned region) const { return shellOf.lookup(region); }

  void emitRegisters(const uarch::RegionBlock &rb);
  /// Backedge every unit output before any consumer resolves it, so a read
  /// address (emitInternalReads) or another unit input may reference a unit
  /// emitted later; emitUnits fills each backedge in when it wires the unit.
  void declareUnits(const uarch::RegionBlock &rb);
  void emitInternalReads(const uarch::RegionBlock &rb);
  /// The skewed halves of the two above: one port per bank per LANE instead of
  /// per bank per access, the bandwidth a skewed layout exists to buy.
  void emitSkewedInternalReads(const uarch::RegionBlock &rb);
  void emitSkewedInternalWrites(const uarch::RegionBlock &rb, Value commit,
                                DatapathFeedback &fb);
  /// Read crossbar for each data-dependent external (argument) read in region
  /// \p rb: drive every bank interface's address with the offset, read each
  /// bank's data port, and mux by the runtime bank (bound into readData before
  /// emitUnits consumes it, like emitInternalReads).
  void emitExternalReads(const uarch::RegionBlock &rb);
  /// Drive the read-address port of each SINGLE-INTERFACE external read in
  /// region \p rb (unbanked or statically banked). The data-dependent ones are
  /// `emitExternalReads`, which drives all of their bank addresses through the
  /// crossbar. Runs after the units, so an address computed by one resolves to
  /// its filled value rather than a dangling backedge.
  void emitExternalReadAddrs(const uarch::RegionBlock &rb);
  /// Where a region's units are being emitted from, which decides whether a
  /// recurrence input re-injects its reduction identity.
  enum class UnitMode {
    /// A leaf region: it has a per-iteration issue pulse, so a loop-carried
    /// input re-injects `inputInits[k]` on its first `inputInitDist[k]` runs.
    /// Its backedges are declared earlier, before the reads resolve.
    Leaf,
    /// A container's own PREDICATE units: a child guard's `i > j` over the
    /// counter / iter-arg survivors, reified as a start-0 combinational
    /// compute. A container has no issue pulse, and correspondingly no
    /// recurrence init to re-inject. Declares its own backedges. Called after
    /// the counter + iter-arg survivors are set and before the children are
    /// sequenced, so a child guard reads its parent's predicate as a
    /// Source::Unit.
    Container,
    /// A sequential while's own CONDITION cone (`emitConditionRegion`). Same
    /// no-recurrence rule as `Container`, but the cone may be MULTI-CYCLE (a
    /// memory read or an IP compare), which is what the CHECK/RUN regime's
    /// `t_cond` wait exists for, so it carries no `comb` restriction. Its
    /// backedges are declared by that caller, which emits the reads the cone
    /// addresses first.
    Condition,
  };
  void emitUnits(const uarch::RegionBlock &rb, UnitMode mode = UnitMode::Leaf);
  /// Emit a sequential (CHECK/RUN) while's condition cone: the container's OWN
  /// condition memory reads (address over the frozen iter-arg survivors) plus
  /// its compute. Returns the settled condition value together with its ready
  /// latency `t_cond`, the cycles after CHECK-start at which the condition is
  /// valid (0 for a combinational condition). A generalisation of
  /// `emitUnits(.., Container)` that also emits the container's own reads. The
  /// read address is the frozen iter-arg survivor, so the loaded value is a
  /// stable wire across the CHECK window; the caller samples it at
  /// `delayValid(checkStart, t_cond)`.
  std::pair<Value, unsigned> emitConditionRegion(const uarch::RegionBlock &rb,
                                                 const uarch::Source &condSrc);
  void resolveRegHeads(const uarch::RegionBlock &rb);
  /// External read addresses + all writes (external ports / internal
  /// seq.write), gated by \p issue, folding the deepest store's stage into
  /// \p fb.storeDrain.
  void emitAccesses(const uarch::RegionBlock &rb, Value issue,
                    DatapathFeedback &fb);

  /// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
  /// `hw.instance`: wire clk/rst/`start`; drive/read each mastered buffer's
  /// hlmem via the child's addr/data/we ports; fold the child's `done` into
  /// \p fb.callDone (the region's completion). Runs BEFORE the region's own
  /// register heads and accesses, since a call's scalar result is an ordinary
  /// datapath Source a register chain or a store may read.
  void emitCalls(const uarch::RegionBlock &rb, Value issue,
                 DatapathFeedback &fb);
  /// The start pulse of one child, from the composition operator's
  /// start-policy table read on this node's contract and its region's
  /// composition class. `EmitContext::startFor` is the same question on the
  /// region substrate.
  Value startForCall(const uarch::CallUnit &cu, Value issue,
                     llvm::ArrayRef<Value> predDones, bool concurrent,
                     const StallShell &sh);
  /// The queue(s) behind a channel whose ends are child ports: one `seq.fifo`
  /// per consumer end (the fan-out tee), each optionally fronted by the
  /// init-prepend shim of a seeded channel, and a pass-through where one end is
  /// a boundary port of this module rather than a child.
  void emitComposedChannel(const uarch::StreamChannel &s);

  /// Declare each kernel-local channel's body wires (`streamWires`) before any
  /// region reads them; `finalizeStreamPorts` builds the FIFO that resolves
  /// them.
  void declareInternalChannels();
  /// One channel's three handshake signals, wherever they live: a boundary
  /// channel's module ports, or a kernel-local channel's own FIFO. Every access
  /// reads the triple through these, so the shell logic is written once.
  Value streamData(const uarch::StreamChannel &s);
  Value streamValid(const uarch::StreamChannel &s);
  Value streamReady(const uarch::StreamChannel &s);

  /// Bind each input stream's `_data` module port into `streamReadData` (once,
  /// before any consumer), so a Source::Stream resolves like a memory read.
  void bindStreamReads(const uarch::RegionBlock &rb);
  /// H for one region: wire region \p rb's stream handshakes and RETURN the
  /// stall shell they derive. An input contributes its `_ready` (gated so a
  /// full output freezes intake too), an output its `_data` plus `_valid` (the
  /// put's activation pulse); the region's stall (input-empty | output-full)
  /// becomes `{chainEnable, issueEnable}`, and each put's stage folds into
  /// \p fb.storeDrain (a put is a store).
  ///
  /// Runs on the already-emitted (F, G) pair, timing its own deeper pulses
  /// against the region's registered PROMISE (`shellFor(rb.id)`); the caller
  /// resolves the promise with the result. A region with no stream accesses is
  /// rigid and derives nothing.
  StallShell deriveStallShell(const uarch::RegionBlock &rb, Value issue,
                              DatapathFeedback &fb);
  /// Drive every boundary channel's module ports, and build every local
  /// channel's `seq.fifo`, from the accumulated `streamDrives` once all regions
  /// have emitted. Call exactly once, before `hw.output`.
  void finalizeStreamPorts();
  /// Drive each scattered argument's per-element data + write-enable outputs
  /// from the accumulated `scatterWrites`, once all regions have emitted. Call
  /// exactly once, before `hw.output`; a read-only scattered argument has no
  /// output port and drives nothing.
  void finalizeScatteredPorts();
  void finalizeSharedWritePorts();
  /// Build one kernel-local channel's `seq.fifo` from its accumulated drives
  /// (\p data is the puts' muxed token) and resolve its `streamWires`.
  void emitInternalChannel(const uarch::StreamChannel &s, Value data);

  /// Emit region \p rb's whole datapath (F) given the controller's \p issue;
  /// returns its store feedback (see emitAccesses). Times everything against
  /// the region's registered shell; the shell's own derivation (H) is the
  /// orchestrator's separate step, run on what this emits.
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
  uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;

  HWEmitter(OpBuilder &b, Location loc, uarch::Datapath &dp,
            circt::hw::HWModulePortAccessor &pa,
            const DenseMap<unsigned, Operation *> &unitModule,
            circt::BackedgeBuilder &bb, Type i1, Type i32,
            const uarch::CalleeCtx *callees = nullptr)
      : ctx(b, loc, bb, i1, i32), control(ctx),
        datapath(ctx, dp, pa, unitModule, callees), dp(dp), pa(pa) {}

  /// The counted terminator of region \p rb: each bound resolved from its
  /// runtime Source (a dynamic trip) or the constant fast path
  /// (RegionBlock::lb/step/tripCount). Empty (default) for an acyclic region
  /// (no counter). A while builds its own Terminator::conditional from the
  /// resolved condition.
  Terminator terminatorOf(const uarch::RegionBlock &rb);
  /// Emit one region and return its `done`. A leaf runs one imperative path for
  /// every regime (counted / dynamic-trip / while): control -> datapath ->
  /// resolve the F->G condition, capture results, done. A container runs its
  /// children once per outer iteration.
  Value emitRegion(const uarch::RegionBlock &rb, Value start, bool retrig);
  /// A loop-over-call region: a counted `dcp.pipeline` wrapping one
  /// `dcp.instance`. One child instance is fired \p tripCount times, a counter
  /// driving its index and each invocation advancing on the child's real
  /// `done` (throughput = one iteration per child latency, not the pipeline
  /// cadence). The counter is the region's `rc.counter` (so `emitCalls` wires
  /// the child's index port to it) and the child start is `rc.issue`; `done`
  /// latches the last iteration.
  Value emitLoopCall(const uarch::RegionBlock &rb, Value start);
  /// The final iteration's issue pulse: a counted region's last iteration
  /// (counter+1 reaches the bound) or a while's condition-false exit; the issue
  /// pulse itself for an acyclic region (a single pass, no counter). The pulse
  /// `done` (emitDone) and the survivor captures (captureResults) both key off.
  Value lastIssuePulse(const RegionControl &rc, const Terminator &term);
  /// Capture LEAF region \p rb's results (`rb.results`) into the survivor
  /// registers a sibling reads, each at its own ready cycle relative to \p
  /// captureOn; returns the region's result-drain stage (the latest-landing
  /// result's ready cycle, folded into the region's `drainStage`). One path for
  /// both terminations: \p captureOn is the last iteration's issue pulse for a
  /// counted loop and each continuing iteration's for a while.
  unsigned captureResults(const uarch::RegionBlock &rb, Value captureOn,
                          Value start);
  /// Run \p regions in program order, each region starting when its
  /// predecessor drains (the first on \p start); returns the last region's
  /// done (a level). The shared sequencer for func-scope siblings and a
  /// container's children.
  Value sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                 bool retrig);
  /// Compose the func-scope sibling regions by their dependence DAG
  /// (`rb.predecessors`): a predecessor-free region starts with the kernel
  /// \p start (independent siblings run concurrently), the rest on the rising
  /// edge of their predecessors' joined `done`. The returned kernel `done` is
  /// the conjunction of every region's `done`, so it completes when the last
  /// region does, whichever that is. Degenerates to `sequence` when every
  /// region depends on its predecessor.
  Value composeSiblings(llvm::ArrayRef<uarch::RegionId> regions, Value start);
  /// Set up a container's loop-carried iter-args as frozen survivor registers
  /// (latch each `rb.results[k].init` at \p start, advance on \p advance),
  /// record each as Source::Survivor{rb, k}, and return the per-arg next-value
  /// backedges (set to `resolveSource(rb.results[k].value)` after the children
  /// emit). The container counterpart of `captureResults`, split in two
  /// because the next-value comes from children that emit later.
  llvm::SmallVector<circt::Backedge>
  setupCarriedIterArgs(const uarch::RegionBlock &rb, Value start,
                       Value advance);
  /// A counted container: wire `emitCountedIteration` to a body that sequences
  /// its children, so the outer counter advances when the last child drains. A
  /// cross-region result crosses child-to-child as a survivor register
  /// (captured in the producing child, read in the consumer).
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
  /// the children, so their stores never fire. Simpler than
  /// emitConditionalContainer: no iteration / iter-args, since the predicate
  /// does not depend on the children.
  Value emitGuard(const uarch::RegionBlock &rb, Value start);
  /// Emit the whole module body: preamble + each top-level region in order.
  void emit();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_HWEMIT_H
