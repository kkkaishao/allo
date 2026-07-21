/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Internal interface for structural `hw.module` emission, split by role along
// the control/datapath (F/G) seam:
//
//   * EmitContext    -- shared builder substrate (clock/reset, constants, and
//                       the low-level comb/seq helpers: reg, mux, delay, ...);
//   * ControlEmitter -- control (G): per-region regime FSMs (free-running /
//                       modulo / acyclic) + the completion signal;
//   * DatapathEmitter-- datapath (F): register chains, compute units, memory,
//                       addressing, and Source resolution (`src`);
//   * HWEmitter      -- the orchestrator wiring the two per region.
//
// The control<->datapath seam is a *typed interface*: a controller returns a
// `RegionControl {issue, counter}`, the datapath returns its store drain (the
// deepest store's stage), and the counter crosses via `setCounter` -- none of
// these are shared mutable members. Extending either side (a new regime, a new
// cell) is a local change.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_HWEMIT_H
#define ALLO_MICROARCH_HWEMIT_H

#include "allo/Microarch/Datapath.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <utility>

namespace mlir::allo::iface {
struct ModuleInterface; // the port model threaded to the structural-top emitter
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Shared free helpers (defined in EmitHW.cpp).
//===----------------------------------------------------------------------===//

/// Map an MLIR datapath type to its hardware integer type (index -> i32, a
/// float is carried as its bit pattern).
IntegerType hwType(Type t, OpBuilder &b);
/// The element type of a memory's memref.
IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b);
/// On-chip read latency of an internal memory's storage primitive (register: 0,
/// LUTRAM/BRAM/URAM: 1 -- the model the scheduler used for external memory).
unsigned memReadLatency(MemoryImplEnum impl);
/// A scheduled dcp op's pipeline stage (its start cycle within the region).
unsigned schedT(Operation *op);
/// Whether a native integer/logic mnemonic has an `emitCompute` comb lowering.
bool combEmitted(StringRef kind);
/// Evaluate an affine index expression to an i32 hw value, emitting comb ops.
/// \p idx holds the resolved value of each map operand (dims then symbols).
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims);
/// The comb op realizing a combinational integer compute unit (pre-checked by
/// `combEmitted`), reading as many of \p operands as the mnemonic's arity
/// needs.
/// \p resultType is the unit's hw result type -- unused by the width-preserving
/// binary ops, but the width the unary casts (extsi/extui/trunci) resize to.
/// \p srcOp is the source dcp.compute op, carrying any op-specific attribute
/// the mnemonic needs (e.g. arith.cmpi's `predicate`, preserved by the
/// reifier).
Value emitCompute(OpBuilder &b, Location loc, StringRef kind,
                  ValueRange operands, Type resultType, Operation *srcOp);

/// A memory access referenced as (mem id, access index) -- enumerated into the
/// module's read / write port lists.
struct AccRef {
  unsigned mem, idx;
};

/// Deterministic module-port base name for the kernel argument behind a memory
/// port, from the memref argument's NameLoc, so cosim/waveforms read in source
/// terms. \p role is "rd"/"wr"; a per-argument index is appended only when the
/// argument backs more than one port of that role (two reads of A ->
/// A_rd0/A_rd1), so single-port args stay clean (A_rd). Falls back to the
/// positional form (`<role><i>`) for an argument with no name. Both the port
/// declaration (EmitHW) and the port access (DatapathEmitter) call this, so the
/// name is defined once; CIRCT's LegalizeNames handles Verilog charset/keyword
/// legality and any residual collision.
std::string memPortBase(const uarch::Datapath &dp, ArrayRef<AccRef> ports,
                        unsigned i, StringRef role);
/// The boundary port base for a memref a CallUnit masters: the same
/// `<name>_<role>` a normal single-access boundary port gets (unindexed -- a
/// child-mastered arg has no parent access to disambiguate against), so the
/// interface declaration + emitCalls pass-through + cosim manifest agree.
std::string memBoundaryPortBase(const uarch::Datapath &dp, uarch::MemId mem,
                                llvm::StringRef role);
/// Deterministic name for a scalar-argument port, from its NameLoc (fallback
/// s<id>).
std::string scalarPortName(const uarch::IOPort &io);
/// Deterministic base name for a stream channel's FIFO ports, from the stream
/// argument's NameLoc (fallback stream<id>); the `_data`/`_valid`/`_ready`
/// suffixes are appended by the port declaration and access emitters alike.
std::string streamPortBase(const uarch::StreamChannel &s);

/// Attach a readable Verilog name to \p v, derived from \p loc's NameLoc, so a
/// frontend variable (acc, buf, i, ...) keeps its source name instead of
/// CIRCT's `_GEN` fallback. Picks the channel ExportVerilog reads: a register
/// (`seq.compreg`) names from its `name` attr, any other value from
/// `sv.namehint`. Best-effort: a no-op when \p loc carries no name (a
/// transform-generated value stays `_GEN`) or \p v is a block argument (named
/// by the port interface). CIRCT's LegalizeNames uniquifies any collision.
void nameValue(Value v, Location loc);
/// Attach \p name directly (no-op if empty or \p v is not an op result). For a
/// name held as a string (e.g. a region's counter name) not on a Location.
void nameValue(Value v, StringRef name);
/// The sanitized NameLoc name of \p loc, or \p fallback when it has none.
std::string cellName(Location loc, StringRef fallback);

/// Declare a module's boundary ports from its port model, in the canonical ABI
/// order: clk/rst/start, then scalar + stream-input + read-data *inputs*, done,
/// then stream-output + read-addr + write + result *outputs* (all module inputs
/// contiguous at the front, as HWModulePortAccessor requires). The single ABI
/// definition shared by the leaf datapath emitter and the structural top, both
/// of which own an `iface::ModuleInterface`.
llvm::SmallVector<circt::hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b);

/// Instantiate module \p mod (as instance \p name), wiring its input ports by
/// name from \p ins and returning its output ports by name. The instance-wiring
/// substrate shared by the leaf datapath emitter (`emitCalls`) and the
/// structural top (`instantiateProcesses`): both build the positional operand
/// vector in port order and collect the results by output-port name.
llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        circt::hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins);

//===----------------------------------------------------------------------===//
// Concurrent composition: a container whose scheduled body spawns concurrent
// processes (`func.call` with the `allo.async` carrier, or stream-wired calls)
// is lowered to a *structural* top -- not a datapath -- that instantiates each
// callee's `hw.module`, allocates a `seq.fifo` per internal channel, broadcasts
// `start`, and AND-reduces the child `done`s. Defined in ConcurrentTop.cpp.
//===----------------------------------------------------------------------===//

/// Emit the structural top for a concurrent container, wiring the already-
/// emitted callee modules (\p modules, keyed by
/// symbol name) into a thin `hw.module`. \p ifaceModels gives each callee's
/// port model in memory (arg <-> concrete port names), read to classify
/// boundaries/channels and each child's start policy. The per-child wiring --
/// broadcast start, static offset, or a `done` handshake; FIFO channel or
/// shared boundary -- is derived from the schedule and the callees'
/// determinacy, not a container-wide mode. A callee may be a leaf kernel or an
/// inner container already
/// emitted this pass -- the two are indistinguishable here, which is what lets
/// composition nest. Inserts the top module at \p b's insertion point; the
/// caller erases \p container afterward. On success fills \p modOut with the
/// emitted top module and \p ifaceOut with its port model (whose toJSON() is
/// the cosim manifest), so the caller can register them for an enclosing
/// container to consume.
/// \p scheduledFuncs maps every scheduled func to its (still un-erased) source
/// `func.func`, which is where a callee's determinacy is read from: an emitted
/// `hw.module` takes its callee's symbol name, so a symbol lookup from a
/// callsite is ambiguous while both live in the module.
LogicalResult
emitConcurrentTop(func::FuncOp container,
                  const llvm::StringMap<circt::hw::HWModuleOp> &modules,
                  const llvm::StringMap<iface::ModuleInterface> &ifaceModels,
                  const llvm::StringMap<func::FuncOp> &scheduledFuncs,
                  OpBuilder &b, circt::hw::HWModuleOp &modOut,
                  iface::ModuleInterface &ifaceOut);

//===----------------------------------------------------------------------===//
// Memory-banking crossbar: the reusable primitives that route an access to one
// of a cyclic-partitioned array's N banks when the bank is not statically known
// (dcp-resolve-banking left the array partitioned). Memory-primitive-agnostic
// -- the bank values feeding a read mux may come from on-chip `seq.read`s
// (internal, 2c) or module read ports (external, 2b); the write-enables gate
// `seq.write`s or port write-enables alike. Defined in DatapathEmitter.cpp.
//===----------------------------------------------------------------------===//
struct EmitContext;

/// A flat element address split into its cyclic bank index and in-bank offset.
struct BankSplit {
  Value bank;   // addr mod factor  (the bank the element lives in)
  Value offset; // addr div factor  (its index within that bank)
};
/// Decompose a flat element address for a cyclic partition of \p factor banks.
/// The factor is a power of two, so bank/offset are a mask/shift (no divider).
BankSplit splitBank(EmitContext &c, Value addr, unsigned factor);
/// N:1 result mux: select `bankValues[bank]` (a priority chain, bank in
/// [0,N) -- bank 0 falls through). Values are pre-read from every bank; the
/// caller aligns \p bank with the read latency.
Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank);
/// The write-enable for bank \p k: `we` gated by `bank == k`, so a write
/// reaches exactly the selected bank (an N-way demux over the banks).
Value bankWe(EmitContext &c, Value we, Value bank, unsigned k);

/// The banking of an *external* (argument) memory access on a
/// cyclic-partitioned array, so the boundary presents one interface per bank.
/// `factor == 1` is an unbanked memory (`bank == 0`); a banked access is either
/// statically routed
/// (`bank` set) or data-dependent (`bank` empty -> a crossbar over all `factor`
/// bank interfaces). Asserts the array is 1-D power-of-two cyclic.
struct ExternalBanking {
  unsigned factor = 1;          // physical banks (1 = unbanked)
  std::optional<unsigned> bank; // static bank, or empty = data-dependent
};
ExternalBanking externalBank(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc);

/// The boundary interfaces of one external memory access, as (bank, port base
/// name): one entry for an unbanked or statically-banked access, and one per
/// bank
/// (`<base>_b<k>`) for a data-dependent one (its crossbar drives every bank).
llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const uarch::Datapath &dp, ArrayRef<AccRef> ports, unsigned i,
         StringRef role);

//===----------------------------------------------------------------------===//
// ShiftChain: the taps of one shift-register chain. The index carries timing --
// `stages[k]` is the input delayed exactly k cycles (`stages[0]` = the
// undelayed input) -- so reads go through the named accessors, which keep the
// "index == cycles delayed" contract explicit and bounds-check every tap (a bad
// tap index is a silent timing bug otherwise).
//===----------------------------------------------------------------------===//
struct ShiftChain {
  llvm::SmallVector<Value> stages;
  /// The input delayed \p k cycles (k-cycle latency).
  Value tap(unsigned k) const {
    assert(k < stages.size() && "shift-chain tap out of range");
    return stages[k];
  }
  /// The deepest tap (delayed `depth()` cycles).
  Value last() const { return stages.back(); }
  /// The chain length in cycles (deepest delay).
  unsigned depth() const { return stages.size() - 1; }
};

//===----------------------------------------------------------------------===//
// EmitContext: the shared builder substrate. No F/G allegiance -- just the
// clock/reset/constants and the low-level combinational + sequential helpers
// both emitters build on.
//===----------------------------------------------------------------------===//
struct EmitContext {
  OpBuilder &b;
  Location loc;
  Type i1, i32;
  circt::BackedgeBuilder &bb;

  Value clk;    // seq.clock form (for compregs / hlmem)
  Value clkRaw; // i1 form (for extern operator instances)
  Value rst;
  Value zero32, one32, f1, t1; // set by initLiterals()

  // The clock-enable of the region currently being emitted (the
  // latency-insensitive stall shell): when set, every shift-register stage
  // (`shiftChain`, hence `delayValid` / `activationPulse` / the done drain)
  // advances only while it is high, so the whole datapath freezes coherently on
  // a stream stall and tap alignment is preserved. Null (the default) => an
  // unconditional pipeline, identical to a stall-free region. Set/cleared
  // by the orchestrator around a stream-touching region.
  Value regionEnable;

  EmitContext(OpBuilder &b, Location loc, circt::BackedgeBuilder &bb, Type i1,
              Type i32)
      : b(b), loc(loc), i1(i1), i32(i32), bb(bb) {}

  Value R(Operation *op) { return op->getResult(0); }
  /// Combinational (0-cycle) constant.
  Value konst(Type t, int64_t v);

  /// Registered (1-cycle): out[t+1] = in[t], sampled unconditionally on every
  /// clock; out = `rstVal` while in reset. The atomic state cell
  /// (`seq.compreg`).
  Value reg(Value in, Value rstVal);
  /// Clock-enabled register (1-cycle when enabled): out[t+1] = ce[t] ? in[t] :
  /// out[t] -- samples `in` on the clock edge only when `ce` is high, else
  /// holds; out = `rstVal` while in reset. Edge-triggered, NOT a
  /// level-sensitive latch.
  Value enabledReg(Value in, Value ce, Value rstVal);
  /// A while iter-arg's frozen result register: out[t+1] = load ? init :
  /// (advance ? next : out[t]). Loaded with `init` on `load` (the region
  /// start), advanced to `next` while the loop continues (`advance`), held
  /// (frozen) once it exits -- so it holds the loop's final carried value, or
  /// `init` for a zero-iteration loop. The survivor a sibling region reads
  /// (setSurvivor).
  Value latchReg(Value init, Value next, Value load, Value advance);
  /// Combinational (0-cycle) 2:1 mux: out = sel ? t : f.
  Value mux(Value sel, Value t, Value f);
  /// Unconditional shift register: every tap advances on each clock (no
  /// enable), so the tap alignment is valid only for a region issuing one item
  /// per cycle (II == 1 / free-running). Returns the taps -- `stages[k]` = `in`
  /// delayed k cycles (k-cycle latency), each stage reset to 0, `stages[0]` =
  /// `in` itself. Backs a tapped `Register` (consumers read distinct taps) and
  /// `delayValid` (last tap).
  ShiftChain shiftChain(Value in, unsigned depth);
  /// A 1-bit signal delayed `n` cycles (issue -> a store's pipeline stage): the
  /// last tap of an `n`-deep `shiftChain`. Resets to 0, so no spurious valid.
  Value delayValid(Value sig, unsigned n);
  /// A scheduled op's activation pulse: \p pulse delayed to the op's pipeline
  /// stage (its `schedT`). The one name for "this op fires now" -- a store's
  /// write-enable, a shared-unit input's mux select, and a fused accumulator's
  /// iteration-0 init gate are all this pulse at the op's stage.
  Value activationPulse(Value pulse, Operation *op);
  /// Combinational (0-cycle) equality of an i32 value `a` against a constant.
  Value icmpEq(Value a, int64_t c);
  /// Combinational (0-cycle) equality of two same-width values (a runtime
  /// compare, e.g. a counter against a data-dependent trip bound).
  Value icmpEqV(Value lhs, Value rhs);
  /// Combinational (0-cycle) unsigned `lhs >= rhs` of two same-width values
  /// (the induction bound test `iv+step >= ub` / empty test `lb >= ub`).
  Value icmpUgeV(Value lhs, Value rhs);
  /// Combinational (0-cycle) `v != 0` (a runtime non-empty / zero-trip test).
  Value isNonZero(Value v);
  /// Combinational (0-cycle) logical NOT of an i1 (`v XOR 1`).
  Value notBit(Value v);
  /// Combinational (0-cycle) AND of two i1s.
  Value andBits(Value lhs, Value rhs);
  /// Combinational (0-cycle) OR of two i1s.
  Value orBits(Value lhs, Value rhs);
  /// A 1-cycle pulse in the same cycle `level` rises 0->1 (out = level &
  /// ~(level delayed one cycle); 0 added latency). The delay reg resets to 0,
  /// so a level held high straight out of reset pulses on cycle 0.
  Value risingEdge(Value level);
  /// The start pulse of a schedulable node: its region-entry `regionStart` when
  /// it has no predecessors (independent -- runs with the kernel / container),
  /// else the rising edge of its predecessors' joined `done` (a handshake; the
  /// node waits for ALL predecessors). The ONE start policy the region composer
  /// (composeSiblings), the sequencer (sequence), and the leaf call chain
  /// (emitCalls) share.
  Value startFor(Value regionStart, ArrayRef<Value> predDones);
  /// A completion-latch level: set to 1 by \p setPulse, cleared to 0 by
  /// \p start (so a retriggered region re-edges each pass). out[t+1] = start ?
  /// 0 : (setPulse ? 1 : out[t]). The shared done-latch of the container
  /// regimes.
  Value holdDone(Value setPulse, Value start);
  /// Split a one-cycle \p when pulse by predicate \p cond into {taken,
  /// notTaken} = {when & cond, when & ~cond}. The predicated fork a run-once /
  /// per- iteration container uses: `taken` (re)starts the children, `notTaken`
  /// completes the region without issuing them.
  std::pair<Value, Value> branchPulse(Value when, Value cond);
  /// Materialize the shared literals (0/1 as i32, false/true as i1).
  void initLiterals();
};

//===----------------------------------------------------------------------===//
// Terminator: what ends a pipelined region's run -- a counter reaching an
// iteration bound (a counted loop) or a data-dependent condition going false (a
// while). This is the data-dependent-timing discriminant as a value:
// `Counted` -> free-running / modulo (a static regime), `Conditional` ->
// flushing. A counted bound is a compile-time
// constant (konst) or a runtime element count resolved from the datapath (a
// dynamic trip, F->G); a conditional's `cond` is a datapath value (typically a
// backedge resolved after the datapath emits its producer). One definition of
// the last-issue / start-gate tests, shared by the single pipelined control
// skeleton (emitPipelined) and the survivor-capture path.
//===----------------------------------------------------------------------===//
struct Terminator {
  enum class Kind { Counted, Conditional };
  Kind kind = Kind::Counted;
  // Induction bounds. `lb`/`step` seed the counter register (init lb, +=step)
  // and `ub` its termination (iv+step >= ub). A while free-runs a 0-based
  // counter (lb=0/step=1, ub null), terminating on ~cond. Values (konst for a
  // constant bound, a resolved datapath Source for a dynamic ub).
  Value lb, ub, step;
  bool dynamic = false; // Counted: runtime ub (may be empty) vs a constant
  Value cond; // Conditional: the i1 continue condition (a datapath value)

  static Terminator counted(Value lb, Value ub, Value step, bool dynamic) {
    return {Kind::Counted, lb, ub, step, dynamic, Value()};
  }
  static Terminator conditional(Value cond, Value zero, Value one) {
    return {Kind::Conditional, zero, Value(), one, false, cond};
  }

  /// The iteration issued at `iv` is the last one: the next induction value
  /// reaches the upper bound (iv+step >= ub), or the continue-condition is
  /// false
  /// (~cond). \p ivStep is `iv + step`.
  Value isLast(EmitContext &c, Value ivStep) const {
    return kind == Kind::Conditional ? c.notBit(cond) : c.icmpUgeV(ivStep, ub);
  }
  /// The region is empty (issues nothing): the lower bound already meets the
  /// upper (lb >= ub). A while is never "empty" here -- its zero-iteration case
  /// is the condition false on iteration 0, handled by the normal exit pulse.
  Value isEmpty(EmitContext &c) const {
    return kind == Kind::Conditional ? c.f1 : c.icmpUgeV(lb, ub);
  }
  /// The start pulse gated so an empty region issues nothing (an empty counted
  /// loop -- a runtime zero-trip or a static lb >= ub -- and a while, which
  /// passes through, its emptiness handled by the condition).
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
};

//===----------------------------------------------------------------------===//
// DatapathFeedback (F -> G): the store timing a control regime consumes to
// compute completion. The typed counterpart of RegionControl, so a new regime
// signal is a field add, not one more parameter on `emitDone`.
//===----------------------------------------------------------------------===//
struct DatapathFeedback {
  // The deepest store's schedule stage (max schedT over the region's stores);
  // 0 if it stores nothing. The region's `done` waits until the last
  // iteration's last store has been presented + committed (the latch adds the
  // commit cycle), which is what fixes a multi-store region's premature
  // completion. A stream put folds into this too (its `done` waits for the last
  // token pushed).
  unsigned storeDrain = 0;
  // The latency-insensitive shell's two control signals (i1; null for a region
  // with no stream accesses). Input starvation must NOT freeze the datapath --
  // that would hold a mid-flight output `valid` high across the stall and let a
  // ready consumer capture the same token twice -- so starvation only injects a
  // bubble (suppresses issue) while the pipeline keeps flowing; only output
  // back-pressure freezes.
  //   chainEnable = ~outputFull  -- the shift-chain clock-enable (freeze).
  //   issueEnable = ~outputFull & allInputsValid -- gates issue: a real
  //                 iteration on an available token, else a bubble.
  Value chainEnable;
  Value issueEnable;
  // The `done` of a CallUnit region's child instance. When set, it IS
  // the region's completion (a call region completes on the child's real done,
  // determinate or not), bypassing the store-drain `emitDone`. Null for a
  // call-free region.
  Value callDone;
};

struct ControlEmitter {
  EmitContext &c;
  explicit ControlEmitter(EmitContext &c) : c(c) {}

  /// Pick the control shape for region \p rb -- acyclic (straight-line) or a
  /// pipelined loop -- and emit it, driven by the \p start pulse against the
  /// resolved \p term (a counted bound or a while's continue-condition).
  /// Returns {issue, counter}.
  RegionControl emitPipelineControl(const uarch::RegionBlock &rb,
                                    const Terminator &term, Value start,
                                    Value enable);
  /// The one pipelined control skeleton for the free-running (II==1), modulo
  /// (II>1), and while (flushing) regimes: they share a `running` flag + an
  /// iteration counter and differ only in \p term (a counter reaching a bound
  /// vs the continue-condition going false) and, for II>1, a phase counter
  /// gating issue. Non-speculative for a conditional terminator (II >= t_cond,
  /// so no doomed iteration issues -> no squash); no backpressure
  /// (fixed-latency memory, no FIFO). A conditional \p term carries `cond` as a
  /// datapath value (F->G), typically a backedge resolved after the datapath
  /// emits its producer.
  /// \p enable is the stall shell's clock-enable (`~stall`): issue is gated
  /// `wantIssue & enable`, so a stalled cycle issues nothing and (with the
  /// enabled shift chains) the whole region freezes. Pass a constant true for a
  /// stall-free region.
  RegionControl emitPipelined(int64_t ii, const Terminator &term, Value start,
                              Value enable);
  RegionControl emitAcyclic(Value start, bool topLevel);

  /// The region's completion signal: one latched level for every regime. It
  /// rises when the last issued iteration's outputs have drained -- \p
  /// lastIssue (the final iteration's issue pulse) delayed \p drainStage cycles
  /// (the deepest store / result stage) -- or immediately on \p emptyDone (an
  /// empty region completes in one cycle; null when unreachable). The latch's
  /// register cycle is the store/result commit cycle, so a sibling starting on
  /// this done reads every committed store and every survivor. A \p retrig
  /// region resets its completion state on \p start.
  Value emitDone(unsigned drainStage, Value lastIssue, Value emptyDone,
                 Value start, bool retrig);
};

//===----------------------------------------------------------------------===//
// DatapathEmitter (F): register chains, compute units, memory access, and the
// uniform Source resolution `src`. Reads the controller's `issue`/`counter`
// (the latter via setCounter) and returns the region's store drain.
//===----------------------------------------------------------------------===//
struct DatapathEmitter {
  EmitContext &c;
  uarch::Datapath &dp;
  circt::hw::HWModulePortAccessor &pa;
  ArrayRef<AccRef> reads, writes;
  const DenseMap<unsigned, Operation *> &unitModule;

  // A region's controller outputs (RegionControl: issue / counter / ungated
  // wantIssue), the G->F seam. `counter` is null for an acyclic region;
  // `wantIssue` is null when the region has no stall shell (a stage-0 stream
  // access hazards on it).
  DenseMap<unsigned, RegionControl> controlOf;
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
  DenseMap<unsigned, circt::Backedge> regHeadBE; // reg id -> chain head input
  DenseMap<unsigned, ShiftChain> regStages;      // reg id -> its tap chain
  DenseMap<uint64_t, Value> readData;            // (mem,access) -> read data
  DenseMap<unsigned, Value> unitVal;             // unit id -> result
  DenseMap<unsigned, circt::Backedge> unitBE;    // unit id -> result backedge
  DenseMap<unsigned, Value> muxVal;              // mux id -> resolved output

  // The child modules a `dcp.instance`'s CallUnit instantiates (null for
  // a plain leaf with no calls).
  const uarch::CalleeCtx *callees = nullptr;

  DatapathEmitter(EmitContext &c, uarch::Datapath &dp,
                  circt::hw::HWModulePortAccessor &pa, ArrayRef<AccRef> reads,
                  ArrayRef<AccRef> writes,
                  const DenseMap<unsigned, Operation *> &unitModule,
                  const uarch::CalleeCtx *callees = nullptr)
      : c(c), dp(dp), pa(pa), reads(reads), writes(writes),
        unitModule(unitModule), callees(callees) {}

  static uint64_t accKey(unsigned m, unsigned a) {
    return (uint64_t(m) << 32) | a;
  }

  /// Resolve a datapath Source to the SSA value driving it.
  Value resolveSource(const uarch::Source &s);
  /// The cycle a freshly-produced Source's value lands, relative to the issuing
  /// pulse of the iteration that produced it: a compute unit's op slot + its
  /// latency, a memory read's slot + read latency, or 0 for an at-issue
  /// constant. The single definition of result-landing timing, used by survivor
  /// capture.
  unsigned readyCycle(const uarch::Source &s) const;
  /// The linear element address of a memory access (affine map + row-major
  /// linearization over the delayed index sources).
  Value computeAddr(const uarch::MemUnit &m, const uarch::MemUnit::Access &acc);
  /// Narrow a linear address to a memory's clog2(depth)-bit index (hlmem).
  Value memAddr(const uarch::MemUnit &m, Value addr);

  /// Bind external read-data input ports into readData (once, before regions).
  void bindReadPorts();
  /// Instantiate seq.hlmem storage for each internal (non-argument) memory.
  void createInternalMemories();
  /// Record a region's iteration counter (from its controller, or a container's
  /// materialized outer counter) for Source::Counter.
  void setCounter(unsigned region, Value iv) { controlOf[region].counter = iv; }
  /// Record a region's issue pulse (from its controller), used to time a fused
  /// accumulator's init injection (iteration-0 issue, delayed to the op's
  /// stage).
  void setIssue(unsigned region, Value issue) {
    controlOf[region].issue = issue;
  }
  /// Wire a region's controller output (counter + issue + ungated wantIssue)
  /// into the datapath in one call -- the G->F seam; the counter is absent for
  /// an acyclic region.
  void setControl(unsigned region, const RegionControl &rc) {
    if (rc.counter)
      setCounter(region, rc.counter);
    setIssue(region, rc.issue);
    if (rc.wantIssue)
      controlOf[region].wantIssue = rc.wantIssue;
  }
  /// Record a region's latched result \p port (from the orchestrator's survivor
  /// capture) so a sibling reading Source::Survivor{region, port} resolves to
  /// it.
  void setSurvivor(unsigned region, unsigned port, Value v) {
    survivorOf[accKey(region, port)] = v;
  }

  void emitRegisters(const uarch::RegionBlock &rb);
  /// Backedge every unit output before any consumer resolves it, so a read
  /// address (emitInternalReads) or another unit input may reference a unit
  /// emitted later; emitUnits fills each backedge in when it wires the unit.
  void declareUnits(const uarch::RegionBlock &rb);
  void emitInternalReads(const uarch::RegionBlock &rb);
  /// Read crossbar for each data-dependent external (argument) read in region
  /// \p rb: drive every bank interface's address with the offset, read each
  /// bank's data port, and mux by the runtime bank (bound into readData before
  /// emitUnits consumes it, like emitInternalReads).
  void emitExternalReads(const uarch::RegionBlock &rb);
  void emitUnits(const uarch::RegionBlock &rb);
  /// Emit region \p rb's own *combinational* units (start-0 computes with no
  /// recurrence init): a container's continue-condition or a guard predicate
  /// reified by the reifier. A restricted `emitUnits` with no
  /// reduction-identity re-injection (a container has no issue pulse) -- called
  /// after the counter
  /// + iter-arg survivors are set and before the children are sequenced, so a
  /// child guard reads its parent's predicate as a Source::Unit.
  void emitCombUnits(const uarch::RegionBlock &rb);
  void resolveRegHeads(const uarch::RegionBlock &rb);
  /// External read addresses + all writes (external ports / internal
  /// seq.write), gated by \p issue. Returns the region's store feedback (the
  /// deepest store's stage, `storeDrain`).
  DatapathFeedback emitAccesses(const uarch::RegionBlock &rb, Value issue);

  /// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
  /// `hw.instance`: wire clk/rst/`start`; drive/read each mastered
  /// buffer's hlmem via the child's addr/data/we ports; fold the child's `done`
  /// into \p fb.callDone (the region's completion). Runs after emitAccesses so
  /// the buffers' hlmems (createInternalMemories) and the region's own accesses
  /// are already emitted.
  void emitCalls(const uarch::RegionBlock &rb, Value issue,
                 DatapathFeedback &fb);

  /// The child induction-variable scalar port's type for a loop-over-call
  /// region: the counter must be built to this exact width so
  /// `resolveSource(Counter)` drives the port with no cast. The IV scalar
  /// operand is the one whose Source is this region's `Counter`.
  Type loopIndexPortType(const uarch::RegionBlock &rb);

  /// Bind each input stream's `_data` module port into `streamReadData` (once,
  /// before any consumer), so a Source::Stream resolves like a memory read.
  void bindStreamReads(const uarch::RegionBlock &rb);
  /// Drive region \p rb's stream ports -- an input's `_ready` (gated so a full
  /// output freezes intake too), an output's `_data` + `_valid` (the put's
  /// activation pulse) -- accumulating the region's stall (input-empty |
  /// output-full) into \p fb.stall and folding each put's stage into
  /// \p fb.storeDrain.
  void emitStreamAccesses(const uarch::RegionBlock &rb, Value issue,
                          DatapathFeedback &fb);

  /// Emit region \p rb's whole datapath given the controller's \p issue;
  /// returns its store feedback (see emitAccesses).
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
            circt::hw::HWModulePortAccessor &pa, ArrayRef<AccRef> reads,
            ArrayRef<AccRef> writes,
            const DenseMap<unsigned, Operation *> &unitModule,
            circt::BackedgeBuilder &bb, Type i1, Type i32,
            const uarch::CalleeCtx *callees = nullptr)
      : ctx(b, loc, bb, i1, i32), control(ctx),
        datapath(ctx, dp, pa, reads, writes, unitModule, callees), dp(dp),
        pa(pa) {}

  /// The counted terminator of region \p rb: each bound resolved from its
  /// runtime Source (a dynamic trip) or the constant fast path
  /// (RegionBlock::lb/step/tripCount). Empty (default) for an acyclic region
  /// (no counter) -- a while builds its own Terminator::conditional from the
  /// resolved condition.
  Terminator terminatorOf(const uarch::RegionBlock &rb);
  /// Emit one region and return its `done`. A leaf runs one imperative path for
  /// every regime (counted / dynamic-trip / while): control -> datapath ->
  /// resolve the F->G condition, capture results, done. A container runs its
  /// children once per outer iteration.
  Value emitRegion(const uarch::RegionBlock &rb, Value start, bool retrig);
  /// A loop-over-call region: a counted `dcp.pipeline` wrapping one
  /// `dcp.instance`. One child instance is fired \p tripCount times, a counter
  /// driving its index and each invocation advancing on the child's real `done`
  /// (throughput = one iteration per child latency, not the pipeline cadence).
  /// The counter is the region's `rc.counter` (so `emitCalls` wires the child's
  /// index port to it) and the child start is `rc.issue`; `done` latches the
  /// last iteration.
  Value emitLoopCall(const uarch::RegionBlock &rb, Value start);
  /// The final iteration's issue pulse: a counted region's last iteration
  /// (counter+1 reaches the bound) or a while's condition-false exit; the issue
  /// pulse itself for an acyclic region (a single pass, no counter). The one
  /// pulse the `done` (emitDone) and the survivor captures (captureResults)
  /// both key off.
  Value lastIssuePulse(const RegionControl &rc, const Terminator &term);
  /// Capture region \p rb's results into the survivor registers a sibling
  /// reads, and return the region's result-drain stage (the latest-landing
  /// result's ready cycle, folded into the region's `drainStage`). Dispatches
  /// the two survivor mechanisms by regime.
  unsigned captureResults(const uarch::RegionBlock &rb, const RegionControl &rc,
                          Value lastIssue, Value start);
  /// A counted / acyclic region's results, captured into a survivor register at
  /// each result's ready cycle (relative to \p lastIssue); returns the
  /// latest-landing (max) stage.
  unsigned captureCountedResults(const uarch::RegionBlock &rb, Value lastIssue,
                                 Value start);
  /// A while region's loop-carried results, each frozen into a latch (init at
  /// \p start, advanced while continuing); returns the deepest carried-value
  /// stage.
  unsigned captureWhileResults(const uarch::RegionBlock &rb,
                               const RegionControl &rc, Value start);
  /// Run \p regions in program order, each region starting when its predecessor
  /// drains (the first on \p start); returns the last region's done (a level).
  /// The shared sequencer for func-scope siblings and a container's children.
  Value sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                 bool retrig);
  /// Compose the func-scope sibling regions by their dependence DAG
  /// (`rb.predecessors`): a predecessor-free region starts with the kernel
  /// \p start (independent siblings run concurrently), the rest on the rising
  /// edge of their predecessors' joined `done`. Regions emit in program order
  /// (SSA dominance), and the returned kernel `done` is the conjunction of
  /// every region's `done` -- it completes when the last region does, whichever
  /// that is. Degenerates to `sequence` when every region depends on its
  /// predecessor.
  Value composeSiblings(llvm::ArrayRef<uarch::RegionId> regions, Value start);
  /// Set up a container's loop-carried iter-args as frozen survivor registers
  /// (latch each `inits[k]` at \p start, advance on \p advance), record each as
  /// Source::Survivor{rb, k}, and return the per-arg next-value backedges (set
  /// to `resolveSource(nexts[k])` after the children emit). Shared by the
  /// counted and conditional container regimes.
  llvm::SmallVector<circt::Backedge>
  setupCarriedIterArgs(const uarch::RegionBlock &rb,
                       llvm::ArrayRef<uarch::Source> inits, Value start,
                       Value advance);
  /// A counted container: sequence its children within each outer iteration,
  /// advancing the outer counter when the last child drains. A cross-region
  /// result crosses child-to-child as a survivor register (captured in the
  /// producing child, read in the consumer).
  Value emitContainer(const uarch::RegionBlock &rb, Value start);
  /// A conditional container (a sequential-wrapper while): the same
  /// per-iteration child sequencing as emitContainer, but the outer iter-args
  /// are frozen survivor registers advanced by the children's results, the loop
  /// terminates on the (combinational) continue-condition instead of an
  /// iteration count, and the condition + iter-arg re-checks are timed by a
  /// done-based CHECK/RUN FSM.
  Value emitConditionalContainer(const uarch::RegionBlock &rb, Value start);
  /// A guard region (a dcp.select): a predicated container whose children run
  /// once iff the held predicate (`dp.guardCond`) holds, else are skipped. The
  /// predicate start-gates child 0 (`start & cond`); a false predicate
  /// completes the region in one cycle (`start & ~cond`) without ever issuing
  /// the children, so their stores never fire. Simpler than
  /// emitConditionalContainer: no iteration / iter-args (the predicate does not
  /// depend on the children).
  Value emitGuard(const uarch::RegionBlock &rb, Value start);
  /// Emit the whole module body: preamble + each top-level region in order.
  void emit();
};

/// Lower the scheduled `func.func`s reachable from \p top to structural
/// `hw.module`s (leaf datapaths + dataflow/sequential tops), erasing the source
/// funcs -- the free function behind the `allo-datapath-to-hw` pass. Emission
/// is rooted at \p top and runs bottom-up over the call DAG (callees before
/// callers), mirroring the scheduler. \p binding names the resource-binding
/// policy. On success \p interfaces maps each emitted module's symbol name to
/// its port-interface JSON (the cosim manifest), so a caller gets the boundary
/// directly without reading any IR attribute.
LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top,
                               llvm::StringMap<std::string> &interfaces);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_HWEMIT_H
