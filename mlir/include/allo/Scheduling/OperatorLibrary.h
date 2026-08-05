/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORLIBRARY_H
#define ALLO_SCHEDULING_OPERATORLIBRARY_H

#include "allo/IR/AlloOps.h"                  // kAlloAsyncAttr
#include "allo/Scheduling/MemoryModel.h"      // MemoryLibrary
#include "allo/Scheduling/OperatorIdentity.h" // OperatorIdentity
#include "allo/Scheduling/RegionGraph.h"      // calleeStaticLatency
#include "allo/Scheduling/Scheduler.h"

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // func::CallOp
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Abstract operator vocabulary (hardware-facing, independent of MLIR op names).
//===----------------------------------------------------------------------===//

/// The abstract operator kind timing is characterized against, spelled in the
/// IR by `dcp.comb` and a built-in `dcp.operator` (see `OpKindEnum` in
/// `AlloAttrs.td`). `stringifyOpKindEnum` / `symbolizeOpKindEnum` convert, the
/// latter returning nullopt for an advanced mnemonic such as `sqrt`.
using OpKind = OpKindEnum;

/// Classify \p op into its abstract kind.
OpKind classify(Operation *op);

/// The abstract kind a combinational realization is priced under, for a caller
/// holding the realization after the `arith` op is gone. Strictly coarser than
/// `classify`: signed and unsigned mnemonics share a row, as do the four
/// integer casts. `Unknown` for a realization no abstract row covers.
OpKind opKindOf(CombOpKindEnum kind);

/// The combinational realization kind of \p op, nullopt for an op with no comb
/// lowering (a float/cast IP, a memory access, an unrelated op). Every case
/// here has an `emitCompute` lowering in the emitter.
std::optional<CombOpKindEnum> combKindOf(Operation *op);

//===----------------------------------------------------------------------===//
// Library entries (built from the injected `dcp.operator` / `dcp.device` IR).
//===----------------------------------------------------------------------===//

/// One row of the operator library. A comb row (`comb == true`) matches by
/// `kind` + all-integer operands, at any width; an IP row by `kind` + an exact
/// operand/result element-type list; an advanced row additionally by `mlirOp`.
struct OperatorEntry {
  OpKind kind = OpKind::Unknown; // abstract kind (Unknown on an advanced row).
  std::string mlirOp;            // advanced: raw MLIR op name (else empty).
  bool comb = false;             // a synthesized combinational row.
  llvm::SmallVector<Type> argTypes; // IP/advanced: exact operand element types.
  llvm::SmallVector<Type> resTypes; // IP/advanced: exact result element types.

  uint32_t latency = 0;  // cycles
  bool pipelined = true; // accepts a new input every cycle
  double inDelay = 0.0;  // ns
  double outDelay = 0.0;
  std::string symbol; // the injected `dcp.operator` sym_name (IP rows only).
  ArrayAttr uses;     // what one instance spends, null where the device is
                      // silent (see `OperatorLibrary::priceOf`).
};

/// What a lookup resolves for one operation. Two separate keys: the scheduling
/// problem prices `typeName`, while an allocation limit, the binder's share
/// test and the emitted module name all key on `identity`.
struct OperatorChar {
  std::string typeName; // stable: one Problem::OperatorType per matched entry
  uint32_t latency = 0;
  /// Whether one instance accepts a new input every cycle. False bounds a
  /// cyclic region's interval (`populateOperatorOccupancy`) and keeps the
  /// operator out of a cyclic allocation (`populateOperatorAllocation`).
  bool pipelined = true;
  double inDelay = 0.0;
  double outDelay = 0.0;
  /// What ONE instance of the matched row costs in the device's currency
  /// (`OperatorLibrary::priceOf`), at this operation's own width. Zero where
  /// the device prices the row at nothing and where it prices it not at all,
  /// which are the same thing to an objective.
  int64_t price = 0;
  OperatorIdentity identity; // empty for an op no functional unit is built for
};

/// The operator library, built from the injected device IR: comb rows from
/// `dcp.device.comb`, IP rows from `dcp.operator` symbols, storage timing from
/// `dcp.device.memory`.
class OperatorLibrary {
public:
  /// Build the library from a module's injected `dcp.device` + `dcp.operator`
  /// ops. A module with no `dcp.device` yields an empty (all-default) library.
  static OperatorLibrary fromModule(ModuleOp module);

  /// Resolve the characterization for \p op: the matching row (advanced first,
  /// then abstract last-wins), else the default.
  OperatorChar lookup(Operation *op) const;

  /// Whether \p op needs an IP realization (a float or advanced compute op) but
  /// no library row matched, so the caller can report an error instead of
  /// scheduling it at the default zero latency.
  bool requiresUnmatchedIP(Operation *op) const;

  /// Whether the device provides a direct realization for \p op: a matching IP
  /// operator or comb row. `legalize-arith` keeps a composite arith op
  /// (max/min/maxnum/minnum/ceildiv/floordiv) when this holds and expands it
  /// into primitive arith otherwise.
  bool hasDirectRealization(Operation *op) const;

  /// The storage-timing view of the device.
  const MemoryLibrary &memoryLibrary() const { return memory; }

  /// The chaining delay of the device's combinational row for \p kind, or 0.0
  /// when the device declares none. For a caller with no `Operation *` to hand
  /// `lookup` (`addressDelaysOf`).
  double combDelay(OpKind kind) const;

  /// The same, for a caller holding a reified realization (a `dcp.compute`'s
  /// `comb_kind`). Falls back to the DEFAULT row, not to 0.0, so an
  /// `affine.apply` is priced the way it was scheduled.
  double combDelay(CombOpKindEnum kind) const;

  //===--------------------------------------------------------------------===//
  // Area, in the objective's currency.
  //
  // One unit of a resource costs `kPriceResolution` times the largest capacity
  // the device declares, over its own: a resource the part has less of costs
  // more, which is the only ranking a scheduler can have between a LUT and a
  // DSP slice. The scale itself is arbitrary and cancels, since every term of
  // the objective is in it; what it buys is resolution, the most plentiful
  // resource pricing at `kPriceResolution` and everything else rounding
  // against that.
  //
  // A capacity is a price input and NOT a budget: regions are solved
  // independently, so no single solve can see a whole-device total (see
  // `dcp.resource`).
  //===--------------------------------------------------------------------===//

  /// What `k` sources of `width` bits cost to select between, which is what
  /// sharing one functional unit puts in front of each of its operand ports.
  int64_t muxPrice(int64_t sources, int64_t width) const;

  /// What carrying a `width`-bit value across `depth` cycles costs. Zero at
  /// depth 0, which is a wire and not a chain.
  int64_t chainPrice(int64_t depth, int64_t width) const;

  /// What ONE cycle of an activation pulse chain costs: one more stage of a
  /// one-bit chain, which is a flip-flop wherever the device says so without
  /// this layer having to know the symbol it says it under.
  int64_t pulsePrice() const;

private:
  /// What \p uses spends at \p params, every resource at its price. Null
  /// \p uses is free, which is what a device saying nothing about a row means.
  int64_t priceOf(ArrayAttr uses, ArrayRef<int64_t> params) const;

  /// The device's combinational row for \p kind, null when it declares none.
  /// Last wins, like `matchEntry`.
  const OperatorEntry *combEntry(OpKind kind) const;

  std::vector<OperatorEntry> advancedEntries; // matched first (raw name)
  std::vector<OperatorEntry> entries;         // abstract rows
  OperatorEntry defaultEntry;
  MemoryLibrary memory;
  llvm::StringMap<int64_t> resourcePrices; // one `dcp.resource`, priced
  ArrayAttr muxUses;                       // `dcp.mux`, over (k, width)
  ArrayAttr chainUses;                     // `dcp.chain`, over (depth, width)
};

/// What the most plentiful resource on a device prices at, and so how much
/// resolution every other price keeps. See the area block above.
inline constexpr int64_t kPriceResolution = 8;

/// Record which operator TYPE prices each of \p problem's operations, and which
/// operator IDENTITY that operation actually holds. \p model accumulates them
/// whole-module and publishes the types covering several identities, which is
/// where the library's pricing over-approximates.
void recordOperatorClasses(circt::scheduling::Problem &problem,
                           const OperatorLibrary &lib, ScheduleModel &model);

//===----------------------------------------------------------------------===//
// Scheduled-call latency
//
// A plain (non-async) call to an already-scheduled callee is a fixed-latency
// node priced through `calleeStaticLatency`; nullopt for any other op.
//===----------------------------------------------------------------------===//
inline std::optional<std::pair<int64_t, std::string>>
scheduledCallLatency(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call || op->hasAttr(kAlloAsyncAttr))
    return std::nullopt;
  Operation *callee = calleeOf(op);
  if (!callee)
    return std::nullopt;
  std::optional<int64_t> lat = calleeStaticLatency(callee);
  if (!lat)
    return std::nullopt;
  return std::make_pair(*lat, ("call." + call.getCallee()).str());
}

//===----------------------------------------------------------------------===//
// Operator model: apply a library to a scheduling problem.
//===----------------------------------------------------------------------===//

/// Assign an operator type (latency + chaining delays) to every operation
/// \p problem holds, sourced from \p lib. A scheduled sync call is a
/// fixed-latency node between registered boundaries.
///
/// Over the problem's OWN operations and not a second walk of the IR: each
/// builder registers every op it walks, and nothing runs between a build and
/// this to change that set, so the problem is what holds the list.
template <class ProblemT>
void populateOperatorTypes(ProblemT &problem, const OperatorLibrary &lib) {
  using namespace circt::scheduling;
  for (Operation *op : problem.getOperations()) {
    if (isSyncSubKernelCall(op)) {
      // Timed by its callee, between registered boundaries. An INDETERMINATE
      // callee has no length to charge, so the node takes zero and its region
      // waits on the child's `done` instead (`isIndeterminateCall`).
      std::optional<std::pair<int64_t, std::string>> cl =
          scheduledCallLatency(op);
      Problem::OperatorType opr = problem.getOrInsertOperatorType(
          cl ? cl->second
             : ("call." + cast<func::CallOp>(op).getCallee() + ".open").str());
      problem.setLatency(opr, cl ? static_cast<unsigned>(cl->first) : 0);
      problem.setIncomingDelay(opr, 0.0);
      problem.setOutgoingDelay(opr, 0.0);
      problem.setLinkedOperatorType(op, opr);
      continue;
    }
    OperatorChar c = lib.lookup(op);
    Problem::OperatorType opr = problem.getOrInsertOperatorType(c.typeName);
    problem.setLatency(opr, c.latency);
    problem.setIncomingDelay(opr, c.inDelay);
    problem.setOutgoingDelay(opr, c.outDelay);
    problem.setLinkedOperatorType(op, opr);
  }
}

/// Reserve a limit-1 resource, held for `latency + 1` cycles, for every sync
/// sub-kernel call in a counted loop body: it is one child instance re-fired
/// per iteration, not a pipelined operator, and the loop controller starts the
/// next invocation on the previous one's `done` plus the cycle it takes to
/// re-arm. Keyed per callsite, since distinct calls are distinct instances.
///
/// A straight-line region needs none: each of its callsites issues once, so
/// there is no second invocation for an occupancy window to hold off.
inline void populateCallOccupancy(ChainingModuloProblem &problem) {
  using P = circt::scheduling::Problem;
  unsigned idx = 0;
  for (Operation *op : problem.getOperations()) {
    std::optional<std::pair<int64_t, std::string>> cl =
        scheduledCallLatency(op);
    if (!cl)
      continue;
    P::ResourceType rsrc =
        problem.getOrInsertResourceType(cl->second + "#" + std::to_string(idx));
    problem.setLimit(rsrc, 1);
    problem.setLinkedResourceTypes(op, SmallVector<P::ResourceType>{rsrc});
    problem.setResourceCycles(op, cl->first + 1);
    ++idx;
  }
}

/// Reserve a PRIVATE limit-1 resource, held for the operator's whole latency,
/// for every operation on a NON-PIPELINED operator. Such a unit takes one input
/// per latency window, so a modulo schedule that re-issues the same operation
/// every II cycles needs `II >= latency`. Without this the model lets a
/// non-pipelined IP run at II=1 and the emitter builds a datapath that feeds it
/// faster than it can accept.
///
/// The window is the latency itself, the span `reservationOf` marks the unit
/// busy for, so what bounds the interval here and what the binder checks a unit
/// against are ONE number.
///
/// Private per operation, because this prices an operation against ITSELF one
/// iteration on, which holds however many units the region builds. What a unit
/// SHARED between two operations costs is `populateOperatorAllocation`'s, and it
/// declines a non-pipelined operator in a cyclic region for want of a
/// circular-arc colouring, leaving every such operation the unit this bounds.
///
/// Only an IP row can be non-pipelined: a comb row and the default row are
/// zero-latency and pipelined, and a memory access is timed by its storage.
///
/// A straight-line region needs none: it issues each operation once, so there
/// is no second issue for a window to hold off.
inline void populateOperatorOccupancy(ChainingModuloProblem &problem,
                                      const OperatorLibrary &lib) {
  using P = circt::scheduling::Problem;
  unsigned idx = 0;
  for (Operation *op : problem.getOperations()) {
    if (isSyncSubKernelCall(op))
      continue; // a re-fired child instance, `populateCallOccupancy`'s window
    OperatorChar c = lib.lookup(op);
    // A one-cycle window is what a pipelined unit already holds, and bounds no
    // interval.
    if (c.pipelined || c.latency < 2)
      continue;
    assert(c.identity.realized() &&
           "only an IP row is non-pipelined, and an IP row names a realization");
    P::ResourceType rsrc = problem.getOrInsertResourceType(
        c.identity.key() + "#" + std::to_string(idx));
    problem.setLimit(rsrc, 1);
    SmallVector<P::ResourceType> units;
    if (auto linked = problem.getLinkedResourceTypes(op))
      units.assign(linked->begin(), linked->end());
    units.push_back(rsrc);
    problem.setLinkedResourceTypes(op, units);
    problem.setResourceCycles(op, c.latency);
    ++idx;
  }
}

//===----------------------------------------------------------------------===//
// Allocation model: how many copies of an operator a region builds. Keyed on
// the operator identity rather than the timing row, since only one physical
// operator can host two operations.
//===----------------------------------------------------------------------===//

/// Declare one allocatable resource per operator identity this region could
/// build fewer copies of than it has operations. Scope:
///
///   * IP identities only. Folding a combinational operator pays for a
///     multiplexer nearly as wide as the operator itself (a 32-bit adder is
///     ~32 LUTs against ~64 of mux).
///   * At least two operations, or there is nothing to fold.
///   * In a cyclic region, a one-cycle occupancy. Past one cycle the
///     reservation window wraps the initiation interval and a count per
///     congruence class no longer implies that many units suffice
///     (circular-arc colouring). Acyclic windows form an interval graph, where
///     the count is exactly the chromatic number, so any occupancy is fine.
///
/// What `n` instances cost is what the DEVICE charges for `n` of them, in the
/// currency the rest of the objective is in: `n` copies of the measured core,
/// plus the multiplexer that many puts in front of every operand port. An
/// UPPER bound on the multiplexer, since two operations sharing a driver need
/// no select between them and the emitter builds one only where the drivers
/// differ.
template <class ProblemT>
void populateOperatorAllocation(ProblemT &problem, const OperatorLibrary &lib) {
  using namespace circt::scheduling;
  constexpr bool isCyclic = std::is_base_of_v<CyclicProblem, ProblemT>;
  // One identity's operations, in problem order. Sorted keying, not insertion
  // order, so two compiles declare the resources in the same order.
  struct OperatorClass {
    llvm::SmallVector<Operation *> ops;
    unsigned occupancy = 1;
    int64_t unitPrice = 0;
    int64_t ports = 0;     // operand ports one instance multiplexes
    int64_t portWidth = 0; // bits each of them carries
  };
  std::map<std::string, OperatorClass> byIdentity;
  for (Operation *op : problem.getOperations()) {
    if (isSyncSubKernelCall(op))
      continue; // one child instance per callsite; no unit to fold it onto
    OperatorChar c = lib.lookup(op);
    if (!c.identity.realized() || c.identity.comb)
      continue;
    // A non-pipelined unit is busy for its whole latency; a pipelined one
    // contends only for its issue slot.
    unsigned occ = c.pipelined ? 1 : std::max(1u, c.latency);
    if (isCyclic && occ > 1)
      continue; // a count alone is not sufficient modulo the II
    OperatorClass &cls = byIdentity[c.identity.key()];
    cls.ops.push_back(op);
    cls.occupancy = occ;
    cls.unitPrice = c.price;
    cls.ports = op->getNumOperands();
    cls.portWidth = 0;
    for (Type t : op->getOperandTypes())
      if (t.isIntOrFloat())
        cls.portWidth =
            std::max<int64_t>(cls.portWidth, t.getIntOrFloatBitWidth());
  }

  for (auto &[key, cls] : byIdentity) {
    if (cls.ops.size() < 2)
      continue;
    auto ceiling = static_cast<unsigned>(cls.ops.size());
    llvm::SmallVector<int64_t> price(ceiling + 1, 0);
    for (unsigned n = 1; n <= ceiling; ++n) {
      // Round-robin, the rule `assignUnits` hands the operations out by:
      // `ceiling % n` instances host one more than the rest.
      unsigned busy = ceiling % n, share = ceiling / n;
      price[n] = n * cls.unitPrice +
                 cls.ports * (busy * lib.muxPrice(share + 1, cls.portWidth) +
                              (n - busy) * lib.muxPrice(share, cls.portWidth));
    }
    Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
    problem.setAllocatable(
        rsrc, typename ProblemT::AllocatableUnit{ceiling, std::move(price)});
    for (Operation *op : cls.ops) {
      llvm::SmallVector<Problem::ResourceType> units;
      if (auto linked = problem.getLinkedResourceTypes(op))
        units.assign(linked->begin(), linked->end());
      units.push_back(rsrc);
      problem.setLinkedResourceTypes(op, units);
      problem.setResourceCycles(op, cls.occupancy);
    }
  }
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORLIBRARY_H
