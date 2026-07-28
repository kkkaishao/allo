/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The emission substrate: everything that builds hardware but knows nothing
// about regions, controllers, or the datapath's structure.
//
//   * free helpers: the type/width rule, storage declaration, a compute unit's
//     comb lowering, affine index evaluation, and the module-boundary ABI
//     (declare ports / instantiate);
//   * banking: the read crossbar and write demux over a partitioned array's N
//     banks;
//   * ShiftChain: the taps of one shift register, index == cycles;
//   * EmitContext: clock/reset/constants plus the low-level comb and
//     sequential primitives (reg, mux, delay, pulse, ...).
//
// Nothing here reads a `RegionBlock` or a controller, which is why it is its
// own header: it is the layer under both F and G.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_PRIMITIVES_H
#define ALLO_MICROARCH_PRIMITIVES_H

#include "allo/Microarch/Datapath.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h" // seq::HLMemOp
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>
#include <utility>

namespace mlir::allo::iface {
struct ModuleInterface; // the port model both emitters declare their ports from
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Type / width / storage-declaration rules, the comb lowering of a compute
// unit, and the module-boundary ABI. Shared by every emitter.
//===----------------------------------------------------------------------===//

/// Map an MLIR datapath type to its hardware integer type (index -> i32, a
/// float is carried as its bit pattern).
IntegerType hwType(Type t, OpBuilder &b);
/// The element type of a memory's memref.
IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b);
/// The depth on-chip storage holding \p words elements is DECLARED with. All
/// three realizations address their storage with exactly clog2(depth) bits
/// (`seq.hlmem` via HLMemType::getAddressTypes, `hw.array_get` by its
/// IndexBitWidthConstraint, `seq.fifo` through its read/write pointers), so a
/// single-element store would need a 0-bit address, a width `hw`/`comb` cannot
/// carry. Declaring one spare word makes the address 1 bit; every access
/// addresses element 0, so the spare is never read, and extra FIFO slack only
/// ever adds buffering. One rule, since a leaf's accesses and the child ports
/// of a composed container must not disagree on one buffer's address width.
unsigned declaredDepth(unsigned words);
/// The element bit patterns of a compile-time array initializer, in NATURAL
/// order (element 0 first), each resized to \p width and the spare words
/// `declaredDepth` reserves padded with zero, so the result is exactly
/// \p depth long. A float table carries its values as their IEEE bit patterns,
/// the same convention the datapath gives every float (`hwType`). One
/// definition for all three consumers of a `memref.global`'s contents: the two
/// constant-table realizations (leaf and container) and `recordMemoryInit`.
llvm::SmallVector<llvm::APInt> initWords(ElementsAttr init, unsigned width,
                                         unsigned depth);
/// Record \p words as the power-on contents of \p mem (`kMemoryInitAttr`).
/// `seq.hlmem` has no initializer, so an initialized array that is also
/// WRITTEN, a real memory that merely starts with contents rather than a
/// constant table, carries them as a discardable attribute until the seq->SV
/// pipeline turns them into the `initial` block a simulator starts from and a
/// synthesis tool reads back as a BRAM INIT.
void recordMemoryInit(circt::seq::HLMemOp mem,
                      llvm::ArrayRef<llvm::APInt> words);
/// Whether a native integer/logic mnemonic has an `emitCompute` comb lowering.
bool combEmitted(StringRef kind);
/// The datapath's width for an index value (`uarch::hwWidth` of an `index`).
/// An address expression may be carried narrower than this (see `evalAffine`),
/// but its operands arrive at this width and a divider is computed at it.
inline constexpr unsigned kDatapathAddressWidth = 32;

/// Evaluate an affine index expression to a \p width -bit hw value, emitting
/// comb ops. \p idx holds the resolved value of each map operand (dims then
/// symbols), each `kDatapathAddressWidth` wide.
///
/// Carrying an address at the `clog2(depth)` bits it actually needs is exact,
/// not an approximation: truncation is reduction modulo 2^width and `+`, `-`,
/// `*` commute with it. `floordiv` / `mod` do not, so they are computed at
/// `kDatapathAddressWidth` and their result narrowed.
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims, unsigned width = kDatapathAddressWidth);
/// The comb op realizing a combinational integer compute unit (pre-checked by
/// `combEmitted`), reading as many of \p operands as the mnemonic's arity
/// needs.
/// \p resultType is the unit's hw result type. The width-preserving binary ops
/// ignore it; the unary casts (extsi/extui/trunci) resize to it.
/// \p srcOp is the source dcp.compute op, carrying any op-specific attribute
/// the mnemonic needs (e.g. arith.cmpi's `predicate`, preserved by the
/// reifier).
Value emitCompute(OpBuilder &b, Location loc, StringRef kind,
                  ValueRange operands, Type resultType, Operation *srcOp);

/// Declare a module's boundary ports from its port model, in the canonical ABI
/// order: clk/rst/start, then scalar + stream-input + read-data *inputs*, done,
/// then stream-output + read-addr + write + result *outputs* (all module inputs
/// contiguous at the front, as HWModulePortAccessor requires). The single ABI
/// definition every emitted module's ports are declared from, off its
/// `iface::ModuleInterface`.
llvm::SmallVector<circt::hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b);

/// Instantiate module \p mod (as instance \p name), wiring its input ports by
/// name from \p ins and returning its output ports by name. The instance-wiring
/// substrate under `emitCalls`: it builds the positional operand vector in port
/// order and collects the results by output-port name.
llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        circt::hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins);

//===----------------------------------------------------------------------===//
// Memory-banking crossbar: the reusable primitives that route an access to one
// of a cyclic-partitioned array's N banks when the bank is not statically known
// (dcp-resolve-banking left the array partitioned). These are
// memory-primitive-agnostic: the bank values feeding a read mux may come from
// on-chip `seq.read`s or from module read ports, and the write-enables gate
// `seq.write`s or port write-enables alike. `externalBank`, the pure-model
// half, lives in Datapath.h.
//===----------------------------------------------------------------------===//
struct EmitContext;

/// An element address split into its bank index and in-bank offset, per the
/// memref's `BankLayout` (see `DatapathEmitter::bankAddress`).
struct BankSplit {
  Value bank;   // which of the layout's banks holds the element; null when the
                // access is statically banked and the caller routes it itself
  Value offset; // its linear index inside that bank (over `bankShape`)
};
/// N:1 result mux: select `bankValues[bank]` (a priority chain, bank in [0,N),
/// with bank 0 falling through). Values are pre-read from every bank; the
/// caller aligns \p bank with the read latency.
Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank);

/// The 1:N write mirror of `readCrossbar`: the write-enable of bank \p k when
/// one address/datum is broadcast to every bank interface and only the
/// addressed bank may commit. \p bank is the runtime bank index, or null for a
/// statically-routed / unbanked write, whose single interface takes \p we
/// verbatim. Caller aligns \p bank and \p we to the same cycle.
Value writeDemux(EmitContext &c, Value we, Value bank, unsigned k);

//===----------------------------------------------------------------------===//
// ShiftChain: the taps of one shift-register chain. The index carries timing:
// `stages[k]` is the input delayed exactly k cycles, and `stages[0]` is the
// undelayed input. Reads go through the named accessors, which keep the
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
// StallShell (H): the elasticity derivation's one output object.
//
// A schedule is RIGID: every op fires at its stamped cycle. H turns a rigid
// (F, G) pair into a latency-insensitive one, and is orthogonal to both. It
// changes neither WHAT is computed nor WHEN relative to issue (F), nor WHICH
// controller runs (G). It stretches the region's time base coherently, so a
// stalled cycle advances nothing and every tap stays aligned.
//
//   chainEnable is F's half. Every shift-register stage, every held read
//     address, and every clock-enabled IP's `ce` advances only while it is
//     high, so the whole datapath freezes together.
//   issueEnable is G's half. The controller issues only while it is high, and
//     DEFERS the denied pass (a latched `running` / `pend`) rather than
//     dropping it, which is what lets a stage-0 access wait for its handshake.
//     It coincides with `chainEnable` today, since a starved stage-0 input
//     freezes the whole region rather than injecting a bubble; the two stay
//     separate fields because a bubbling regime is the one E1 would add.
//
// Both null is a RIGID shell, the identity: every primitive below reduces to
// its unconditional form, which is exactly what a region with no stream
// accesses wants. Neither F nor G computes this object.
// `DatapathEmitter::deriveStallShell` does, from the region's stream
// handshakes.
//
// It is never AMBIENT. Every timing primitive takes one, so a caller must name
// the shell it is timing against. The caller in turn gets it from the region
// that OWNS the cell (`DatapathEmitter::shellFor`, keyed like `controlOf`),
// which for a shared-unit mux or a memory access is not necessarily the region
// currently emitting.
//===----------------------------------------------------------------------===//
struct StallShell {
  Value chainEnable; // F consumes; null => rigid
  Value issueEnable; // G consumes; null => rigid (issue ungated)
  /// Whether this region is latency-insensitive at all (has a stall shell).
  explicit operator bool() const { return chainEnable != Value(); }
};

//===----------------------------------------------------------------------===//
// EmitContext: the shared builder substrate. No F/G allegiance, just the
// clock/reset/constants and the low-level combinational and sequential helpers
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

  // The region being emitted, as a naming prefix (`r3`). Naming only, with no
  // semantics attached. It exists so the anonymous cells these helpers build
  // (valid chains, activation pulses) read as `r3_v2` rather than `_GEN_41`.
  std::string regionTag;

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
  /// out[t]. It samples `in` on the clock edge only when `ce` is high, else
  /// holds; out = `rstVal` while in reset. Edge-triggered, NOT a
  /// level-sensitive latch.
  Value enabledReg(Value in, Value ce, Value rstVal);
  /// Stall-hold: transparent (combinational passthrough) while \p sh's
  /// `chainEnable` is high, holds its last enabled value while low. out = ce ?
  /// in : held; held[t+1] = out[t]. Unlike `enabledReg` (which delays by a
  /// cycle), this adds NO latency when enabled. A read address therefore stays
  /// == the counter in steady state, but freezes on back-pressure so a stalled
  /// memory keeps presenting the un-consumed element and the in-flight read is
  /// not lost. A no-op (returns `in`) under a rigid shell.
  Value stallHold(Value in, const StallShell &sh);
  /// A while iter-arg's frozen result register: out[t+1] = load ? init :
  /// (advance ? next : out[t]). Loaded with `init` on `load` (the region
  /// start), advanced to `next` while the loop continues (`advance`), held
  /// (frozen) once it exits, so it holds the loop's final carried value, or
  /// `init` for a zero-iteration loop. The survivor a sibling region reads
  /// (setSurvivor).
  Value latchReg(Value init, Value next, Value load, Value advance);
  /// Combinational (0-cycle) 2:1 mux: out = sel ? t : f.
  Value mux(Value sel, Value t, Value f);
  /// Shift register on \p sh's time base: each tap advances every clock under a
  /// rigid shell, and only while `chainEnable` is high under an elastic one, so
  /// the taps freeze together and the "index == cycles delayed" contract holds
  /// under stall too. Returns the taps: `stages[k]` = `in` delayed k cycles
  /// (k-cycle latency), each stage reset to 0, `stages[0]` = `in` itself. Backs
  /// a tapped `Register` (consumers read distinct taps) and `delayValid` (last
  /// tap).
  ShiftChain shiftChain(Value in, unsigned depth, const StallShell &sh);
  /// A 1-bit signal delayed `n` cycles (issue -> a store's pipeline stage): the
  /// last tap of an `n`-deep `shiftChain`. Resets to 0, so no spurious valid.
  Value delayValid(Value sig, unsigned n, const StallShell &sh);
  /// A scheduled op's activation pulse: \p pulse delayed to the op's pipeline
  /// stage (its `dcpStart`). The one name for "this op fires now": a store's
  /// write-enable, a shared-unit input's mux select, and a fused accumulator's
  /// iteration-0 init gate are all this pulse at the op's stage.
  Value activationPulse(Value pulse, Operation *op, const StallShell &sh);
  /// Combinational (0-cycle) equality of an i32 value `a` against a constant.
  Value icmpEq(Value a, int64_t c);
  /// Combinational (0-cycle) equality of two same-width values (a runtime
  /// compare, e.g. a counter against a data-dependent trip bound).
  Value icmpEqV(Value lhs, Value rhs);
  /// Combinational (0-cycle) unsigned `lhs >= rhs` of two same-width values.
  Value icmpUgeV(Value lhs, Value rhs);
  /// Combinational (0-cycle) SIGNED `lhs >= rhs` of two same-width values (the
  /// induction bound test `iv+step >= ub` / empty test `lb >= ub`): signed so a
  /// negative compile-time lower bound (`affine.for %i = -4 to 4`) compares
  /// correctly. Identical to the unsigned test for a non-negative counter.
  Value icmpSgeV(Value lhs, Value rhs);
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
  /// it has no predecessors (independent, so it runs with the kernel or
  /// container), else the rising edge of its predecessors' joined `done` (a
  /// handshake: the node waits for ALL predecessors). The ONE start policy the
  /// region composer (composeSiblings), the sequencer (sequence), and the leaf
  /// call chain (emitCalls) share.
  Value startFor(Value regionStart, ArrayRef<Value> predDones);
  /// A completion-latch level: set to 1 by \p setPulse, cleared to 0 by
  /// \p start (so a retriggered region re-edges each pass). out[t+1] = start ?
  /// 0 : (setPulse ? 1 : out[t]). The shared done-latch of the container
  /// regimes.
  Value holdDone(Value setPulse, Value start);
  /// Split a one-cycle \p when pulse by predicate \p cond into {taken,
  /// notTaken} = {when & cond, when & ~cond}. The predicated fork a run-once or
  /// per-iteration container uses: `taken` (re)starts the children, `notTaken`
  /// completes the region without issuing them.
  std::pair<Value, Value> branchPulse(Value when, Value cond);
  /// Materialize the shared literals (0/1 as i32, false/true as i1).
  void initLiterals();
};

/// Scoped setter for `EmitContext::regionTag`, restoring the enclosing
/// container's tag on exit so a nested region's cells carry its own prefix.
struct RegionTag {
  EmitContext &c;
  std::string saved;
  RegionTag(EmitContext &c, unsigned region) : c(c), saved(c.regionTag) {
    c.regionTag = "r" + std::to_string(region);
  }
  ~RegionTag() { c.regionTag = saved; }
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_PRIMITIVES_H
