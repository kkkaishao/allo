/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_DATAPATHBUILDER_H
#define ALLO_MICROARCH_DATAPATHBUILDER_H

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/Datapath.h"

#include "allo/IR/AlloOps.h"

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
  unsigned initDist = 1; // recurrence distance (iterations): `init` must be
                         // re-injected for the first `initDist` runs
};

//===----------------------------------------------------------------------===//
// DatapathBuilder. One instance builds one function's Datapath: `build()`
// drives allocation then interconnect derivation and returns the model.
//===----------------------------------------------------------------------===//
struct DatapathBuilder {
  Datapath &dp;
  dcp::DCPathModuleOp func;

  // Build-time scratch (NOT part of the result): value/op provenance maps.
  llvm::DenseMap<Value, MemId> memOf;
  llvm::DenseMap<Value, StreamId> streamOf;
  // Keyed by the produced VALUE, not its op: a multi-result producer (a
  // sub-kernel call returning several scalars) drives one Source per result,
  // and every lookup site has the Value in hand anyway.
  llvm::DenseMap<Value, Source> producerOf;
  llvm::DenseMap<Value, Source> ioOf;
  llvm::DenseMap<Operation *, unsigned> regionIdxOf;

  // Interconnect-derivation scratch (transient; see deriveInterconnect).
  // A register is keyed by (held value, consuming region): the SAME value (an
  // enclosing loop's counter) can be read in several nested regions, and each
  // needs its own delay chain built in its own region.
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
  const OperatorLibrary &lib;  // device storage + operator timing
  float cycleTime;             // the period the schedule was cut against
  const CalleeCtx *callees;    // child modules/ifaces for a dcp.instance
                               // (null for a plain leaf, no calls)

  DatapathBuilder(Datapath &dp, dcp::DCPathModuleOp func,
                  const BindingPolicy &policy, const OperatorLibrary &lib,
                  float cycleTime, const CalleeCtx *callees = nullptr)
      : dp(dp), func(func), policy(policy), lib(lib), cycleTime(cycleTime),
        callees(callees) {}

  /// build the datapath model
  void build();

  // Allocation & binding -----------------------------------------
  /// Register every literal as a tie-off ConstCell (func-wide, so a hoisted
  /// constant resolves the same as an in-body one).
  void collectConstants();
  /// Create a RegionBlock for \p regionOp (id \p ridx): kind/ii/length/trip and
  /// the parent/child linkage. Returned by value; pushed by `build`.
  RegionBlock addRegion(Operation *regionOp, RegionId ridx);
  /// Derive every region's `shape` discriminant (see `RegionBlock::Shape`) and
  /// assert the structural invariants each shape carries. Runs right after the
  /// region walk, the earliest point where both the parent/child edges and
  /// the CallUnits it reads are complete (unlike `addRegion`, which sees
  /// neither for the region it is building).
  void deriveShapes();
  /// Bind one body op to its resource: the dispatch, one arm per resource kind
  /// (below) plus the kinds that bind nothing (a nested region, a literal, a
  /// declaration). An op matching none of them is reported and marks the build
  /// infeasible, rather than being silently dropped from the hardware.
  void bindResource(Operation *op, RegionBlock &rb);
  /// A `dcp.instance` -> a CallUnit owned by \p rb: one MemArg per child memory
  /// port (the child masters them), one scalar-input slot per scalar operand
  /// (its driver resolved later, by `recordCallScalars`), and a `Source::Call`
  /// producer per scalar result.
  void bindCall(dcp::DCPathInstanceOp inv, RegionBlock &rb);
  /// A `stream.get` / `stream.put` -> one StreamChannel access. Both directions
  /// bind identically; only a get produces a token.
  void bindStream(Operation *op, RegionBlock &rb);
  /// A `dcp.load` / `dcp.store` on \p memref -> one MemUnit access. Asserts the
  /// two contracts this binding assumes: no store to a memory classified
  /// read-only, and the scheduled access latency equalling the device model's.
  void bindMemory(Operation *op, Value memref, RegionBlock &rb);
  /// A `dcp.compute` -> a FuncUnit, combinational or IP-realized, holding the
  /// op at its reservation slot (its issue cycle, modulo II when cyclic).
  void bindCompute(dcp::DCPathComputeOp comp, RegionBlock &rb);
  /// Bind the func-scope arith cone: every op the reifier left in the module
  /// body outside a region (a top-level loop's affine bound expression, a
  /// top-level guard's predicate) -> a `ScopeUnit` (see its declaration).
  void bindScopeOps();
  /// Allocate (or reuse) a MemUnit for \p memref (external iff a func
  /// argument).
  MemId getOrCreateMem(Value memref);
  /// Allocate (or reuse) a StreamChannel for the `!allo.stream` value \p stream
  /// (a func block arg). \p isInput sets the channel direction on first
  /// touch (a get => input, a put => output).
  StreamId getOrCreateStream(Value stream, bool isInput);
  /// Record how each region produces its results (`rb.results`) and, for the
  /// two regimes that have one, its control predicate (`rb.condition`). A
  /// region result is a survivor register; a loop's k-th result is the final
  /// value of its k-th iter-arg (see `RegionResult`). No-op for a
  /// result-less counted region.
  void recordRegionResults(llvm::ArrayRef<Operation *> regionOps);
  /// Resolve each `dcp.instance`'s scalar operands into its CallUnit's
  /// `scalarIns`. Separate from `bindResource`, which creates the CallUnit
  /// during the region walk: a Source resolution needs the complete region
  /// model (see `resolveValue`).
  void recordCallScalars();
  /// Record every CallUnit's composition predecessors (`cu.predecessors`), the
  /// instance-substrate counterpart of `recordSiblingDeps`. A SCHEDULED
  /// composition orders its children by their placed `start` and only gates
  /// an earlier or indeterminate producer; a CONCURRENT one places every child
  /// at 0 and must read the hazard direction (RAW / WAW / WAR) to order them
  /// at all. Runs after `recordCallScalars`, whose Sources carry the
  /// scalar-result edges.
  void recordCallDeps();
  /// Re-decide which initialized arrays are constant TABLES, once the
  /// children's port directions are known. `isConstantTable` (the scheduler's
  /// predicate, read at `getOrCreateMem` time) conservatively disqualifies any
  /// array handed to a sub-kernel, since before the callee interfaces exist it
  /// cannot see which way the child touches it; here, with every access and
  /// child port group bound, an array nothing writes is a ROM.
  void reclassifyRoms();
  /// Derive every cyclic region's `counterType`, the width its iteration
  /// counter and therefore its bounds are built at, from that loop's own
  /// induction range. It reads nothing but the region op, so it runs before the
  /// Source-resolving passes; a consumer wanting another width adapts at its
  /// own end.
  void deriveCounterTypes();
  /// Record each pipeline's induction bounds (lb/ub/step) as Sources on its
  /// RegionBlock: a runtime bound from the `lbBound`/`dynamicBound`/`stepBound`
  /// operand, a compile-time one as a synthesized literal cell. Needs
  /// `counterType`, which is the width those literals are tied in at.
  void recordRegionBounds(llvm::ArrayRef<Operation *> regionOps);
  /// A literal \p v of type \p t as a Source, appending the ConstCell that
  /// holds it. For a value the model needs but no `arith.constant` in the body
  /// produces, such as an induction bound written as an attribute rather than
  /// an operand.
  Source constant(int64_t v, Type t);
  /// Enumerate the module's boundary memory ports: `dp.{read,write}Ports`, each
  /// external access's slot and port-group NAME
  /// (`MemUnit::Access::{portIdx, portBase}`), and each call-mastered boundary
  /// argument's group name (`CallUnit::MemArg::topBase`), all off ONE
  /// per-(memory, role) counter so parent accesses are numbered first and
  /// child ports continue the same sequence in call order. Runs once every
  /// access and call is bound. The owner name comes from `uniqueOwnerOf`
  /// against the module's whole memref list, so two arguments sharing one
  /// source name still get distinct port-group names.
  void enumerateBoundaryPorts();
  /// Record each top-level region's composition predecessors
  /// (`rb.predecessors`): the earlier top-level siblings it must start after,
  /// from a shared memref (any access, whether a hazard or a read-port
  /// conflict) or a cross-region SSA edge (a scalar survivor). The emitter
  /// starts a predecessor-free region concurrently and gates the rest on their
  /// producers' `done`. Runs last (needs the final memref accesses + region
  /// tree).
  void recordSiblingDeps(llvm::ArrayRef<Operation *> regionOps);
  /// Scalar (non-memref) function arguments become input IOPorts.
  void bindIOArgs();
  /// Scalar (non-memref) function results become `dp.results` output ports,
  /// each driven by the Source of its `func.return` operand (a returning
  /// region's survivor / a passthrough input / a constant). Array results are
  /// out-params by this stage (buffer-results-to-out-params), so every
  /// remaining operand is a scalar.
  void recordResults();

  /// Settle the allocation: fold each group named by \p groups onto one unit
  /// and REBUILD the table densely, so a unit with no bound op never exists.
  ///
  /// Runs immediately after the region walk, which is the last point at which
  /// a `UnitId` is held only by `producerOf` and `dp.opToUnit`, both rewritten
  /// here. Every pass below resolves Values to Sources against the table, so a
  /// decision taken after any of them would leave those Sources naming a unit
  /// that no longer has ops (see the class comment on the phase order).
  void allocateUnits(llvm::ArrayRef<llvm::SmallVector<UnitId, 2>> groups);

  // Value resolution ---------------------------------------------
  /// The ONE Value -> Source resolution: the channel through which \p v can be
  /// read, or None if this datapath cannot read it (every caller reports that
  /// as an unresolved slot rather than dropping it silently).
  ///
  /// Four disjoint cases:
  ///   * a scheduled producer bound during the region walk (`producerOf`): a
  ///     compute unit, a memory / stream read port, a call result;
  ///   * a nested region's result -> that region's Survivor register;
  ///   * a scalar function argument (`ioOf`) -> an IOPort;
  ///   * a `dcp.pipeline` block argument -> its region's Counter (arg 0), or
  ///     the matching Survivor where the region LATCHES its iter-args (a
  ///     container / while); a childless counted reduction instead fuses its
  ///     accumulator into the datapath, readable only through the recurrence
  ///     edge `resolveOperand` builds.
  ///
  /// Needs the COMPLETE region model (to know which regions latch), so every
  /// caller runs after the region walk.
  Source resolveValue(Value v);

  // Interconnect derivation ---------------------------------------
  /// Resolve an operand \p v consumed by \p consumer (in a region with
  /// initiation interval \p ii) to its producing Source + register depth:
  /// `resolveValue` plus the depth rule, plus the one edge that does not read
  /// \p v at all (an un-latched own iter-arg = the loop recurrence).
  Resolved resolveOperand(Value v, Operation *consumer, unsigned ii);
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
  /// Group a skewed memory's accesses into lanes that can share one port per
  /// bank, or leave it crossbarring when they cannot.
  void assignLanes(MemUnit &m);
  /// Record a resolved edge into \p slot: a depth-0 edge ties directly, a
  /// deeper one is deferred (its register chain is built in insertRegisters).
  void recordEdge(Resolved r, Source &slot, unsigned regionIdx);
  /// Resolve every unit input (single, or shared-then-muxed).
  void resolveUnitInputs();
  /// Resolve every memory address / store data and stream data + predicate.
  void resolveAccessOperands();
  /// Decide which accesses carry their address in a register that advances
  /// with the loop counters, and record the scaled counters that needs.
  void planAddressGenerators();
  /// Build the register chains the deferred edges need, patch their slots, and
  /// materialize the shared-unit muxes.
  void insertRegisters();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATHBUILDER_H
