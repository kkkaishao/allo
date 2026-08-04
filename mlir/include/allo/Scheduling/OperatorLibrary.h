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

/// The abstract operator kind timing is characterized against; `classify` maps
/// concrete IR ops onto these. The three cast kinds are separate because their
/// hardware timing differs.
enum class OpKind {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Max,      // maximumf / maxsi / maxui (NaN-propagating or integer maximum)
  Min,      // minimumf / minsi / minui
  MaxNum,   // maxnumf (maxNum: returns the non-NaN operand)
  MinNum,   // minnumf
  CeilDiv,  // ceildivsi / ceildivui
  FloorDiv, // floordivsi
  Neg,
  Cmp,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  Select,
  ICastI, // integer resize (sext / zext / trunc / index_cast)
  FCastI, // int <-> float conversion (si/ui-to-fp, fp-to-si/ui)
  FCastF, // float resize (extf / truncf)
  MemRead,
  MemWrite,
  StreamRead,
  StreamWrite,
  Unknown // op the classifier does not recognize (e.g. math.sqrt).
};

/// Classify \p op into its abstract kind.
OpKind classify(Operation *op);

/// The abstract-kind string a device/operator uses (`add`/`sub`/.../`select`),
/// and its inverse; `parseOpKind` returns nullopt for a non-abstract name (an
/// advanced mnemonic such as `sqrt`).
llvm::StringRef opKindString(OpKind kind);
std::optional<OpKind> parseOpKind(llvm::StringRef s);

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
};

/// What a lookup resolves for one operation. Two separate keys: the scheduling
/// problem prices `typeName`, while an allocation limit, the binder's share
/// test and the emitted module name all key on `identity`.
struct OperatorChar {
  std::string typeName; // stable: one Problem::OperatorType per matched entry
  uint32_t latency = 0;
  /// Whether one instance accepts a new input every cycle. Read only by
  /// `populateOperatorAllocation`.
  bool pipelined = true;
  double inDelay = 0.0;
  double outDelay = 0.0;
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

private:
  /// The device's combinational row for \p kind, null when it declares none.
  /// Last wins, like `matchEntry`.
  const OperatorEntry *combEntry(OpKind kind) const;

  std::vector<OperatorEntry> advancedEntries; // matched first (raw name)
  std::vector<OperatorEntry> entries;         // abstract rows
  OperatorEntry defaultEntry;
  MemoryLibrary memory;
};

/// Log, for one solved region, every operator type covering more than one
/// operator identity. Measurement only: nothing consumes the log.
void reportOperatorClassSplit(circt::scheduling::Problem &problem,
                              const OperatorLibrary &lib);

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

/// Assign an operator type (latency + delays) to every op reached by \p walkFn,
/// sourced from \p lib. A `ChainingProblem` also receives incoming/outgoing
/// delays; a scheduled sync call is a fixed-latency node.
template <class ProblemT, class WalkFn>
LogicalResult populateOperatorTypesImpl(ProblemT &problem, WalkFn walkFn,
                                        const OperatorLibrary &lib) {
  using namespace circt::scheduling;
  constexpr bool isChaining = std::is_base_of_v<ChainingProblem, ProblemT>;

  walkFn([&](Operation *op) {
    if (std::optional<std::pair<int64_t, std::string>> cl =
            scheduledCallLatency(op)) {
      Problem::OperatorType opr = problem.getOrInsertOperatorType(cl->second);
      problem.setLatency(opr, static_cast<unsigned>(cl->first));
      if constexpr (isChaining) {
        problem.setIncomingDelay(opr, 0.0); // registered boundary
        problem.setOutgoingDelay(opr, 0.0);
      }
      problem.setLinkedOperatorType(op, opr);
      return;
    }
    OperatorChar c = lib.lookup(op);
    Problem::OperatorType opr = problem.getOrInsertOperatorType(c.typeName);
    problem.setLatency(opr, c.latency);
    if constexpr (isChaining) {
      problem.setIncomingDelay(opr, c.inDelay);
      problem.setOutgoingDelay(opr, c.outDelay);
    }
    problem.setLinkedOperatorType(op, opr);
  });
  return success();
}

/// Populate operator types for every op reachable from \p body (a loop body).
template <class ProblemT>
LogicalResult populateOperatorTypes(Block &body, ProblemT &problem,
                                    const OperatorLibrary &lib) {
  return populateOperatorTypesImpl(
      problem, [&](auto handle) { body.walk(handle); }, lib);
}

/// Populate operator types over the (walked) top-level ops of a straight-line
/// region.
template <class ProblemT>
LogicalResult populateOperatorTypes(ArrayRef<Operation *> ops,
                                    ProblemT &problem,
                                    const OperatorLibrary &lib) {
  return populateOperatorTypesImpl(
      problem,
      [&](auto handle) {
        for (Operation *top : ops)
          top->walk(handle);
      },
      lib);
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
/// The cost is `latency * width` flip-flops, in the unit the objective's
/// register tie-break counts. It under-estimates a real IP, so the objective
/// cannot over-value a fold.
template <class ProblemT, class WalkFn>
LogicalResult populateOperatorAllocationImpl(ProblemT &problem, WalkFn walkFn,
                                             const OperatorLibrary &lib) {
  using namespace circt::scheduling;
  if constexpr (!std::is_base_of_v<OccupancyProblem, ProblemT>) {
    return success();
  } else {
    constexpr bool isCyclic = std::is_base_of_v<CyclicProblem, ProblemT>;
    // One identity's operations, in walk order. Sorted keying, not insertion
    // order, so two compiles declare the resources in the same order.
    struct OperatorClass {
      llvm::SmallVector<Operation *> ops;
      unsigned occupancy = 1;
      unsigned cost = 0;
    };
    std::map<std::string, OperatorClass> byIdentity;
    walkFn([&](Operation *op) {
      OperatorChar c = lib.lookup(op);
      if (!c.identity.realized() || c.identity.comb)
        return;
      // A non-pipelined unit is busy for its whole latency; a pipelined one
      // contends only for its issue slot.
      unsigned occ = c.pipelined ? 1 : std::max(1u, c.latency);
      if (isCyclic && occ > 1)
        return; // a count alone is not sufficient modulo the II
      OperatorClass &cls = byIdentity[c.identity.key()];
      cls.ops.push_back(op);
      cls.occupancy = occ;
      cls.cost = c.latency * c.identity.resultType.getIntOrFloatBitWidth();
    });

    for (auto &[key, cls] : byIdentity) {
      if (cls.ops.size() < 2)
        continue;
      Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
      problem.setAllocatable(
          rsrc, typename ProblemT::AllocatableUnit{
                    static_cast<unsigned>(cls.ops.size()), cls.cost});
      for (Operation *op : cls.ops) {
        llvm::SmallVector<Problem::ResourceType> units;
        if (auto linked = problem.getLinkedResourceTypes(op))
          units.assign(linked->begin(), linked->end());
        units.push_back(rsrc);
        problem.setLinkedResourceTypes(op, units);
        problem.setResourceCycles(op, cls.occupancy);
      }
    }
    return success();
  }
}

/// Populate the allocation model for every op reachable from \p body.
template <class ProblemT>
LogicalResult populateOperatorAllocation(Block &body, ProblemT &problem,
                                         const OperatorLibrary &lib) {
  return populateOperatorAllocationImpl(
      problem, [&](auto handle) { body.walk(handle); }, lib);
}

/// The same over the (walked) top-level ops of a straight-line region.
template <class ProblemT>
LogicalResult populateOperatorAllocation(ArrayRef<Operation *> ops,
                                         ProblemT &problem,
                                         const OperatorLibrary &lib) {
  return populateOperatorAllocationImpl(
      problem,
      [&](auto handle) {
        for (Operation *top : ops)
          top->walk(handle);
      },
      lib);
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORLIBRARY_H
