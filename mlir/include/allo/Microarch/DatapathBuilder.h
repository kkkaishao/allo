/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// DatapathBuilder: constructs the L2 `Datapath` model from a function carrying
// materialized `allo.dcp.*` ops, in two composable phases:
//
//   * Allocation & binding: a `RegionBlock` per dcp region op, and the trivial
//     resource binding (every compute op its own `FuncUnit`, every memref its
//     own `MemUnit`, each literal a `ConstCell`, scalar args `IOPort`s).
//     `bindResource` is the seam for non-trivial binding (sharing + muxes).
//   * Interconnect derivation: `resolveOperand` applies the register-depth rule
//     `d*II + (tY - tX) - lat`, and `insertRegister` materializes the
//     shift-register chains.
//
// The build-time scratch maps (producerOf / ioOf / regionIdxOf / memOf) are
// MEMBERS, not threaded arguments -- so a new piece of build state (e.g. the
// survivor's resultOf) is a member, not one more parameter on every method.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_DATAPATHBUILDER_H
#define ALLO_MICROARCH_DATAPATHBUILDER_H

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/Datapath.h"

#include "llvm/ADT/MapVector.h"

#include <deque>

namespace mlir::allo::uarch {

/// The producer of a value plus the register depth a consumer needs to read it.
struct Resolved {
  Source base;      // the producing cell output
  Value key;        // register key (the produced SSA value; null => never reg)
  unsigned depth;   // pipeline-register depth for this edge
  bool ok = false;  // false => producer outside this region / not modelled
  Source init = {}; // reduction identity iff this edge reads a loop-carried
                    // iter_arg (a recurrence input); None otherwise
  unsigned initDist =
      1; // recurrence distance (iterations) for `init`: the init
         // must be re-injected for the first `initDist` runs (a
         // distance-d carry, e.g. a 2nd-order shift register,
         // reads d undefined past values before its own outputs)
};

//===----------------------------------------------------------------------===//
// DatapathBuilder. One instance builds one function's Datapath: `build()`
// drives allocation then interconnect derivation and returns the model.
//===----------------------------------------------------------------------===//
struct DatapathBuilder {
  Datapath &dp;
  func::FuncOp func;

  // Build-time scratch (NOT part of the result): value/op provenance maps.
  llvm::DenseMap<Value, MemId> memOf;
  llvm::DenseMap<Value, StreamId> streamOf;
  llvm::DenseMap<Operation *, Source> producerOf;
  llvm::DenseMap<Value, Source> ioOf;
  llvm::DenseMap<Operation *, unsigned> regionIdxOf;
  llvm::StringMap<unsigned> boundaryBaseSeq; // CallUnit boundary port groups:
                                             // running accessor index per base

  // Interconnect-derivation scratch (transient; see deriveInterconnect).
  // A register is keyed by (held value, consuming region): the SAME value (an
  // enclosing loop's counter, delayed to a later stage) is read in several
  // nested regions, and each needs its OWN delay chain built in its own region
  // -- a single shared register would be emitted in one region and tapped,
  // unbuilt, from a sibling emitted earlier.
  using RegKey = std::pair<::mlir::Value, unsigned>;
  struct RegDepth { // a register-fed input slot, patched once its chain exists
    Source *slot;
    RegKey key;
    unsigned depth;
  };
  struct MuxBuild { // a shared unit port's per-op drivers, muxed after chains
    UnitId unit;
    unsigned port;
    RegionId region;
    llvm::SmallVector<Operation *, 2> ops;
    llvm::SmallVector<Source, 2> sources; // parallel to ops
  };
  llvm::MapVector<RegKey, llvm::SmallVector<unsigned>> depthsByKey;
  llvm::DenseMap<RegKey, Source> baseByKey;
  llvm::SmallVector<RegDepth> pending;
  std::deque<MuxBuild> muxBuilds; // a deque so `record`'s slot pointers into
                                  // `sources` survive later pushes

  const BindingPolicy &policy; // decides resource sharing
  const CalleeCtx *callees;    // child modules/ifaces for a dcp.instance
                               // (null for a plain leaf, no calls)

  DatapathBuilder(Datapath &dp, func::FuncOp func, const BindingPolicy &policy,
                  const CalleeCtx *callees = nullptr)
      : dp(dp), func(func), policy(policy), callees(callees) {}

  /// build the datapath model
  void build();

  // -- Allocation & binding --------------------------------------
  /// Register every literal as a tie-off ConstCell (func-wide, so a hoisted
  /// constant resolves the same as an in-body one).
  void collectConstants();
  /// Create a RegionBlock for \p regionOp (id \p ridx): kind/ii/length/trip and
  /// the parent/child linkage. Returned by value; pushed by `build`.
  RegionBlock addRegion(Operation *regionOp, RegionId ridx);
  /// Bind one body op to its resource: compute -> FuncUnit, memref access ->
  /// MemUnit port, constant -> already registered, nested region -> skipped.
  void bindResource(Operation *op, RegionBlock &rb);
  /// Allocate (or reuse) a MemUnit for \p memref (external iff a func
  /// argument).
  MemId getOrCreateMem(Value memref);
  /// Allocate (or reuse) a StreamChannel for the `!allo.stream` value \p stream
  /// (a func block arg). \p isInput sets the channel direction on first
  /// touch (a get => input, a put => output).
  StreamId getOrCreateStream(Value stream, bool isInput);
  /// Record region \p rb's result (its `uncondition` operand's producing
  /// Source) in `dp.regionResult`, so a sibling consuming it reads a Survivor.
  /// No-op for a result-less region. Runs after the region body is bound.
  void recordRegionResult(const RegionBlock &rb, Operation *regionOp);
  /// Record each pipeline's runtime induction bounds (lb/ub/step) as Sources on
  /// its RegionBlock (from the `lbBound`/`dynamicBound`/`stepBound` operands).
  /// Runs after bindIOArgs so a scalar-IO bound resolves; a prologue-survivor
  /// bound needs only regionIdxOf (populated as regions are added, prologue
  /// before loop).
  void recordRegionBounds(llvm::ArrayRef<Operation *> regionOps);
  /// Record each iter-arg-carrying container's / while's loop-carried
  /// recurrence in `dp.carryInfo` (per-carried init/next, plus a while's
  /// condition). Runs after bindIOArgs so a scalar-IO init resolves. A leaf
  /// counted reduction is skipped (its accumulator is fused).
  void recordCarryInfo(llvm::ArrayRef<Operation *> regionOps);
  /// Record each guard (dcp.select) region's i1 predicate Source in
  /// `dp.guardCond` (its `$condition` operand, a preceding condition region's
  /// survivor). Runs after region recording so the survivor resolves. A
  /// result-mux guard (else branch) is asserted out -- unsupported.
  void recordGuards(llvm::ArrayRef<Operation *> regionOps);
  /// Record each top-level region's composition predecessors
  /// (`rb.predecessors`): the earlier top-level siblings it must start after,
  /// from a shared memref (any access -- hazard or read-port conflict) or a
  /// cross-region SSA edge (a scalar survivor). The emitter starts a
  /// predecessor-free region concurrently and gates the rest on their
  /// producers' `done`. Runs last (needs the final memref accesses + region
  /// tree).
  void recordSiblingDeps(llvm::ArrayRef<Operation *> regionOps);
  /// The Source driving a pipeline's runtime bound value \p v (a region-result
  /// survivor, a hoisted producer, or a scalar IOPort); None if unmodelled.
  Source boundSource(Value v);
  /// Scalar (non-memref) function arguments become input IOPorts.
  void bindIOArgs();
  /// Scalar (non-memref) function results become `dp.results` output ports,
  /// each driven by the Source of its `func.return` operand (a returning
  /// region's survivor / a passthrough input / a constant). Runs after
  /// bindIOArgs + region recording so the operand resolves. Array results are
  /// out-params by now (buffer-results-to-out-params), so every remaining
  /// operand is a scalar.
  void recordResults();

  /// Apply the policy's sharing decision: fold each group's units onto its
  /// first (moving their bound ops + rebinding `opToUnit`/`producerOf`), then
  /// drop the emptied units from their region. Runs after the trivial
  /// allocation and before interconnect derivation, which then grows the
  /// sharing muxes.
  void applyBinding(llvm::ArrayRef<llvm::SmallVector<UnitId, 2>> groups);

  // -- Interconnect derivation -----------------------------------
  /// Resolve an operand \p v consumed by \p consumer (in a region with
  /// initiation interval \p ii) to its producing Source + register depth.
  Resolved resolveOperand(Value v, Operation *consumer, unsigned ii);
  /// Resolve a region-external recurrence init \p v (a reduction identity) to a
  /// Source -- a hoisted Const or a scalar IOPort; None if unmodelled.
  Source initSource(Value v);
  /// Materialize one shift-register chain carrying \p key, deep enough for the
  /// largest of \p depths, with a tap at each distinct depth. Returns its id.
  RegId insertRegister(Value key, ArrayRef<unsigned> depths, Source input,
                       RegionId region);
  /// Resolve every unit input / memory address / store-data / stream driver,
  /// threading non-zero-depth edges through inserted register chains. Drives
  /// the four phases below.
  void deriveInterconnect();
  /// Size the (empty) input-slot vectors every resolve phase fills.
  void allocateInputSlots();
  /// Record a resolved edge into \p slot: a depth-0 edge ties directly, a
  /// deeper one is deferred (its register chain is built in insertRegisters).
  void recordEdge(Resolved r, Source &slot, unsigned regionIdx);
  /// Resolve every unit input (single, or shared-then-muxed).
  void resolveUnitInputs();
  /// Resolve every memory address / store data and stream data + predicate.
  void resolveAccessOperands();
  /// Build the register chains the deferred edges need, patch their slots, and
  /// materialize the shared-unit muxes.
  void insertRegisters();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATHBUILDER_H
