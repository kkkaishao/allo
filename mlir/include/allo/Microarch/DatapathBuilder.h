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

/// The memref a `dcp.load` / `dcp.store` accesses; null for any other op.
Value dcpMemref(Operation *op);
/// Every op of \p regionOp that binds a resource: its body, plus a guard's ELSE
/// branch, which the body alone does not report. A nested region is visited in
/// its own turn, so this does not recurse. Shared by the facts sweep and the
/// binding walk, which must enumerate the same ops in the same order.
void forEachBodyOp(Operation *regionOp,
                   llvm::function_ref<void(Operation *)> fn);

/// The producer of a value plus the register depth a consumer needs to read it.
struct Resolved {
  Source base;    // the producing cell output
  Value key;      // register key (the produced SSA value; null => never reg)
  unsigned depth; // pipeline-register depth for this edge
  unsigned ready = 0; // cycle `base` lands at within its iteration
  bool ok = false;    // false => producer outside this region / not modelled
  // Reduction identities of the loop-carried iter_arg this edge reads, one per
  // iteration: `inits[n]` is re-injected at iteration n, and `base` carries the
  // edge from `inits.size()` on, that size being the recurrence distance. Empty
  // for every other edge.
  llvm::SmallVector<Source, 1> inits = {};
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
  // sub-kernel call) drives one Source per result.
  llvm::DenseMap<Value, Source> producerOf;
  llvm::DenseMap<Value, Source> ioOf;
  llvm::DenseMap<Operation *, unsigned> regionIdxOf;

  // Interconnect-derivation scratch (transient; see resolveEdges).
  // A delay chain is keyed by (held value, consuming region): the SAME value
  // (an enclosing loop's counter) can be read in several nested regions, each
  // needing its own chain built in its own region.
  using RegKey = std::pair<::mlir::Value, unsigned>;
  struct Edge {  // an input slot's driver, and the delay it owes before landing
    Source base; // what drives the head of the chain
    RegKey key;  // the chain this slot taps
    unsigned depth; // the tap it reads
    unsigned ready; // cycle `base` lands at within its iteration
    // The address reduction folded this operand into a scaled counter, so no
    // chain owes it a tap and its slot stays empty.
    bool reduced = false;
  };
  struct MuxBuild { // one slot's per-op drivers, muxed after the chains exist
    Source *slot;
    RegionId region;
    Type type;
    llvm::SmallVector<Operation *, 2> ops;
    llvm::SmallVector<Source, 2> sources;    // parallel to ops
    llvm::SmallVector<Mux::Phase, 2> phases; // parallel to ops
  };
  // Keyed by the slot each edge will patch, which takes at most one: the
  // address reduction withdraws by slot, and chains are built in record order.
  llvm::MapVector<Source *, Edge> edges;
  std::deque<MuxBuild> muxBuilds; // a deque so `record`'s slot pointers into
                                  // `sources` survive later pushes

  const BindingPolicy &policy; // decides resource sharing
  const DeviceModel &dev;      // device storage + operator timing
  float cycleTime;             // the period the schedule was cut against
  const CalleeCtx &callees;    // child modules/ifaces for a dcp.instance
                               // (null for a plain leaf, no calls)

  DatapathBuilder(Datapath &dp, dcp::DCPathModuleOp func,
                  const BindingPolicy &policy, const DeviceModel &dev,
                  float cycleTime, const CalleeCtx &callees)
      : dp(dp), func(func), policy(policy), dev(dev), cycleTime(cycleTime),
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
  /// assert the structural invariants each shape carries. Runs after the region
  /// walk, the earliest point where the parent/child edges and the CallUnits it
  /// reads are complete.
  void deriveShapes();
  /// Bind one body op to its resource: one arm per resource kind, plus the
  /// kinds that bind nothing (a nested region, a literal, a declaration). An op
  /// matching none is reported and marks the build infeasible.
  void bindResource(Operation *op, RegionBlock &rb);
  /// A `dcp.instance` -> a CallUnit owned by \p rb: one MemArg per child memory
  /// port, one scalar-input slot per scalar operand (its driver resolved later
  /// by `recordCallScalars`), and a `Source::Call` producer per scalar result.
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
  /// Build one MemUnit per array the function touches, holding everything the
  /// device and the layout say about it. Runs before the region walk, so an
  /// array's WHOLE use is in view when a property of the use is decided: an
  /// initialized array nothing writes is a constant table, and only a sweep
  /// that has seen every access and every child port group can say so. Must
  /// reach the same SET of ops the binding walk does, which only looks a
  /// MemUnit up; the order here is what fixes MemId order, and the order that
  /// fixes the boundary ports is `m.accesses`, which the binding walk alone
  /// decides.
  void collectStorageFacts(llvm::ArrayRef<Operation *> regionOps);
  /// The MemUnit backing \p memref. A lookup, not a factory:
  /// `collectStorageFacts` has already built every one of them.
  MemId memIdOf(Value memref);
  /// Allocate (or reuse) a StreamChannel for the `!allo.stream` value \p stream
  /// (a func block arg). \p isInput sets the channel direction on first
  /// touch (a get => input, a put => output).
  StreamId getOrCreateStream(Value stream, bool isInput);
  /// Record how each region produces its results (`rb.results`) and, where it
  /// has one, its control predicate (`rb.condition`). A region result is a
  /// survivor register; a loop's k-th result is its k-th iter-arg's last value.
  void recordRegionResults(llvm::ArrayRef<Operation *> regionOps);
  /// Resolve each `dcp.instance`'s scalar operands into its CallUnit's
  /// `scalarIns`. Separate from `bindResource`: a Source resolution needs the
  /// complete region model (see `resolveValue`).
  void recordCallScalars();
  /// Record every CallUnit's composition predecessors (`cu.predecessors`). A
  /// SCHEDULED composition orders its children by their placed `start` and only
  /// gates an earlier or indeterminate producer; a CONCURRENT one places every
  /// child at 0 and orders them by hazard direction (RAW / WAW / WAR) instead.
  /// Runs after `recordCallScalars`, whose Sources carry the scalar-result
  /// edges.
  void recordCallDeps();
  /// Derive every cyclic region's `counterType`, the width its iteration
  /// counter and therefore its bounds are built at, from that loop's own
  /// induction range. A consumer wanting another width adapts at ITS end (an
  /// ordinary datapath read widens back to `kIndexWidth`, a child's index port
  /// takes the port's width, an address cone takes the memory's).
  void deriveCounterTypes();
  /// Record each pipeline's induction bounds (lb/ub/step) as Sources on its
  /// RegionBlock: a runtime bound from the `lbBound`/`dynamicBound`/`stepBound`
  /// operand, a compile-time one as a literal cell. Needs `counterType`, the
  /// width those literals are tied in at.
  void recordRegionBounds(llvm::ArrayRef<Operation *> regionOps);
  /// A literal \p v of type \p t as a Source, appending the ConstCell that
  /// holds it. For a value the model needs but no `arith.constant` in the body
  /// produces, such as an induction bound written as an attribute.
  Source constant(int64_t v, Type t);
  /// Enumerate the module's boundary memory ports: `dp.{read,write}Ports`, each
  /// external access's `MemUnit::Access::{portIdx, portBase}` and each
  /// call-mastered boundary argument's `CallUnit::MemArg::topBase`, all off ONE
  /// per-(memory, role) counter so parent accesses are numbered first and child
  /// ports continue the same sequence in call order. Runs once every access and
  /// call is bound. Owner names come from `uniqueOwnerOf` against the module's
  /// whole memref list, so two arguments sharing a source name still differ.
  void enumerateBoundaryPorts();
  /// Bind every memory access and child port to a port of its bank
  /// (`MemUnit::Access::port`, `CallUnit::MemArg::port`) and record how many
  /// ports each bank is built with. The boundary port enumeration, the emitter
  /// and the report all read it. Runs after `planAccessPorts`, which settles
  /// how each access reaches its memory.
  void bindMemoryPorts();
  /// Group a skewed memory's accesses into lanes that can share one port per
  /// bank (`MemUnit::skewed`, `Access::lane`), or leave it crossbarring when
  /// they cannot. The first of the port decisions, since the plan below reads
  /// whether the skew held.
  void assignLanes();
  /// Decide how each access and each child-mastered port reaches its memory
  /// (`MemUnit::Access::plan`, `CallUnit::MemArg::plan`). Runs before
  /// `bindMemoryPorts`, which hands out ports along the plan it settles.
  void planAccessPorts();
  /// Ports one bank comes out of a `planPorts` colouring with: split by
  /// direction, and counted outright, which is below their sum wherever a port
  /// carries both. `colours` is the whole memory's count, where a second
  /// colouring of the other direction starts numbering.
  struct PortCounts {
    unsigned reads = 0, writes = 0, total = 0, colours = 0;
  };
  /// One candidate binding of a memory's ports, decided without touching the
  /// model so two of them can be compared before either is committed.
  struct PortAssignment {
    /// The port each vertex of `Datapath::portGraph`'s order takes, already
    /// offset by the `base` it was planned at.
    llvm::SmallVector<unsigned> colour;
    PortCounts counts;
    /// Whether the writes landed on ports separate enough to be separate
    /// `always` blocks. Meaningful only when `writes` covered them.
    bool writesIndependent = false;
    /// The direction this covers, or none for a both-directions colouring.
    /// `commitPorts` replays the same vertex walk from it.
    std::optional<bool> writes;
  };
  /// Colour one memory's port graph. \p writes picks a direction, or nullopt
  /// takes both together. \p base offsets the numbering so a second, separate
  /// colouring cannot collide with the first.
  ///
  /// Returns nullopt only from a both-directions pass whose writes did not
  /// split, which it cannot express. A pass given a direction always plans.
  std::optional<PortAssignment>
  planPorts(const MemUnit &m, std::optional<bool> writes, unsigned base);
  /// Write \p pa into `MemUnit::Access::port` / `CallUnit::MemArg::port`, and
  /// `MemUnit::writesIndependent` where it covered the writes.
  void commitPorts(MemUnit &m, const PortAssignment &pa);
  /// Record what each memory's ports cost against what its schedule asks:
  /// `MemUnit::{readConcurrency, writeConcurrency, boundaryPorts}`. Measures
  /// only; what the measurements are worth reporting is `checkInputLegality`.
  /// Runs after `enumerateBoundaryPorts`, whose groups it counts.
  void measurePorts();
  /// Record each top-level region's composition predecessors
  /// (`rb.predecessors`): the earlier top-level siblings it must start after.
  /// Needs the bound accesses and the region tree, so it runs after the region
  /// walk; and it runs BEFORE the port passes, which read the ordering it
  /// establishes (`Datapath::portGraph`).
  void recordSiblingDeps(llvm::ArrayRef<Operation *> regionOps);
  /// Scalar (non-memref) function arguments become input IOPorts.
  void bindIOArgs();
  /// Scalar (non-memref) function results become `dp.results` output ports,
  /// each driven by the Source of its `func.return` operand. Array results are
  /// out-params by this stage (buffer-results-to-out-params).
  void recordResults();

  /// Settle the allocation: fold each group named by \p groups onto one unit
  /// and REBUILD the table densely, so a unit with no bound op never exists.
  ///
  /// Runs immediately after the region walk, the last point at which a
  /// `UnitId` is held only by `producerOf` and `dp.opToUnit`, both rewritten
  /// here. A fold taken later would leave already-resolved Sources naming a
  /// unit with no ops.
  void allocateUnits(llvm::ArrayRef<llvm::SmallVector<UnitId, 2>> groups);

  // Value resolution ---------------------------------------------
  /// The ONE Value -> Source resolution: the channel through which \p v can be
  /// read, or None if this datapath cannot read it (every caller reports that
  /// as an unresolved slot rather than dropping it silently). Needs the
  /// COMPLETE region model, to know which regions latch their iter-args, so
  /// every caller runs after the region walk.
  Source resolveValue(Value v);

  // Interconnect derivation ---------------------------------------
  /// Resolve an operand \p v consumed by \p consumer (in a region with
  /// initiation interval \p ii) to its producing Source plus register depth,
  /// plus the one edge that does not read \p v at all (an un-latched own
  /// iter-arg = the loop recurrence).
  Resolved resolveOperand(Value v, Operation *consumer, unsigned ii);
  /// Resolve every unit input / memory address / store-data / stream driver
  /// into an `Edge`. Records what each slot reads and how late; materializes
  /// nothing, so what the delays cost is still open when it returns.
  void resolveEdges();
  /// Realize the delays the edges owe: the address reduction first, since a
  /// term it folds into a scaled counter withdraws its edge, then the chains
  /// over what is left, then the muxes that pick between them.
  ///
  /// The LAST derivation `build()` runs, after the ports, which read nothing it
  /// produces. Not the other way round because withdrawing an edge is the one
  /// thing here that is not additive, so no pass may resolve a Source after it.
  void realizeDelays();
  /// Record a resolved edge into \p slot: a depth-0 edge ties directly, a
  /// deeper one is deferred to `edges` and patched by `insertRegisters`.
  void recordEdge(const Resolved &r, Source &slot, unsigned regionIdx);
  /// `recordEdge` for a slot that is a bare Source with no input port beside it
  /// to hold a recurrence identity (`FuncUnit::inputInits`): an address, a
  /// store datum, a stream token. A recurrence edge on \p operand of \p
  /// consumer is muxed against its identities, phased on that op's issue cycle;
  /// every other edge is recorded unchanged.
  void recordCarriedEdge(const Resolved &r, Value operand, Operation *consumer,
                         Source &slot, unsigned regionIdx);
  /// Resolve every unit input (single, or shared-then-muxed).
  void resolveUnitInputs();
  /// Resolve every memory address / store data and stream data + predicate.
  void resolveAccessOperands();
  /// Decide which accesses carry their address in a register that advances
  /// with the loop counters, record the scaled counters that needs, and
  /// withdraw the edge of every operand no residual is left reading.
  void planAddressGenerators();
  /// Build one chain per (value, region) the surviving edges tap, patch their
  /// slots, and materialize the shared-unit muxes.
  void insertRegisters();
  /// Record each region's terminal cycle (`RegionBlock::drainStage`). Runs
  /// after `resolveEdges`, which re-stamps a stream put's stage and so is the
  /// last pass that moves one.
  void recordDrainStages();
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATHBUILDER_H
