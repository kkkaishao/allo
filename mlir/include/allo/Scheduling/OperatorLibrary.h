/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORLIBRARY_H
#define ALLO_SCHEDULING_OPERATORLIBRARY_H

#include "allo/IR/AlloOps.h"             // kAlloAsyncAttr, dcp ops
#include "allo/Scheduling/MemoryModel.h" // MemoryLibrary + populateMemoryResources
#include "allo/Scheduling/Scheduler.h"
#include "allo/Scheduling/Utils.h" // sched::kLatencyAttr

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // func::CallOp (scheduled-call latency)
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
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Abstract operator vocabulary (hardware-facing, independent of MLIR op names).
//===----------------------------------------------------------------------===//

/// The abstract operator kind timing is characterized against; `classify` maps
/// concrete IR ops onto these. The three cast kinds are separate because
/// integer resize, integer/float conversion, and float resize have distinct
/// hardware timing.
enum class OpKind {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
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

/// Classify \p op into its abstract kind -- the mapping from a concrete IR op
/// to the abstract vocabulary. Matching then compares the op's concrete operand
/// and result types against a library row's types.
OpKind classify(Operation *op);

/// The abstract-kind string a device/operator uses (`add`/`sub`/.../`select`),
/// and its inverse; `parseOpKind` returns nullopt for a non-abstract name (an
/// advanced mnemonic such as `sqrt`).
llvm::StringRef opKindString(OpKind kind);
std::optional<OpKind> parseOpKind(llvm::StringRef s);

/// The combinational realization kind of \p op. This enumerates the emitter's
/// native `comb` coverage: every case has an `emitCompute` lowering. Nullopt
/// for an op with no comb lowering (a float/cast IP, a memory access, or an
/// unrelated op).
std::optional<CombOpKindEnum> combKindOf(Operation *op);

//===----------------------------------------------------------------------===//
// Library entries (built from the injected `dcp.operator` / `dcp.device` IR).
//===----------------------------------------------------------------------===//

/// One row of the operator library. A comb row (`comb == true`) matches by
/// `kind` + all-integer operands (integer arithmetic is uniformly
/// combinational, any width); an IP row matches by `kind` + an exact operand /
/// result element-type list (`argTypes`/`resTypes`), carrying its injected
/// `dcp.operator` symbol; an advanced row additionally keys on the raw MLIR
/// mnemonic (`mlirOp`). Comb rows come from `dcp.device.comb`, IP/advanced rows
/// from injected `dcp.operator` symbols.
struct OperatorEntry {
  OpKind kind = OpKind::Unknown; // abstract kind (Unknown on an advanced row).
  std::string mlirOp;            // advanced: raw MLIR op name (else empty).
  bool comb = false;             // a synthesized combinational row.
  llvm::SmallVector<Type> argTypes; // IP/advanced: exact operand element types.
  llvm::SmallVector<Type> resTypes; // IP/advanced: exact result element types.

  uint32_t latency = 0; // cycles
  double inDelay = 0.0; // ns
  double outDelay = 0.0;
  bool pipelined = true;
  std::string symbol; // the injected `dcp.operator` sym_name (IP rows only).
};

/// The timing characterization resolved for a specific operation.
struct OperatorChar {
  std::string typeName; // stable: one Problem::OperatorType per matched entry
  uint32_t latency = 0;
  double inDelay = 0.0;
  double outDelay = 0.0;
  bool pipelined = true;
  std::string symbol; // the matched `dcp.operator` sym_name (IP), else empty
};

/// The operator library, built from the injected device IR: comb rows from
/// `dcp.device.comb`, IP rows from `dcp.operator` symbols, storage timing from
/// `dcp.device.memory`. `lookup` returns the matching row's characterization
/// (advanced first, then abstract last-wins, else the default).
class OperatorLibrary {
public:
  /// Build the library from a module's injected `dcp.device` + `dcp.operator`
  /// ops. A module with no `dcp.device` yields an empty (all-default) library.
  static OperatorLibrary fromModule(ModuleOp module);

  /// Resolve the characterization for \p op: the matching row, else default.
  OperatorChar lookup(Operation *op) const;

  /// Whether \p op needs an IP realization (a float or advanced compute op) but
  /// no library row matched, so the caller can report an error instead of
  /// scheduling it at the default zero latency.
  bool requiresUnmatchedIP(Operation *op) const;

  /// The storage-timing view of the device.
  const MemoryLibrary &memoryLibrary() const { return memory; }

private:
  std::vector<OperatorEntry> advancedEntries; // matched first (raw name)
  std::vector<OperatorEntry> entries;         // abstract rows
  OperatorEntry defaultEntry;
  MemoryLibrary memory;
};

//===----------------------------------------------------------------------===//
// Scheduled-call latency: a scheduling helper, separate from operator
// characterization. A plain (non-async) call to an already-scheduled callee is
// a fixed-latency node in the enclosing problem -- the callee's whole-kernel
// latency (its `sched.latency`, annotated bottom-up) with registered
// boundaries. Returns {latency, stable operator-type name} for such a call,
// else nullopt.
//===----------------------------------------------------------------------===//
inline std::optional<std::pair<int64_t, std::string>>
scheduledCallLatency(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call || op->hasAttr(kAlloAsyncAttr))
    return std::nullopt;
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      op, call.getCalleeAttr());
  if (!callee)
    return std::nullopt;
  auto lat = callee->getAttrOfType<IntegerAttr>(sched::kLatencyAttr);
  if (!lat)
    return std::nullopt;
  return std::make_pair(lat.getInt(), ("call." + call.getCallee()).str());
}

//===----------------------------------------------------------------------===//
// Operator model: apply a library to a scheduling problem.
//===----------------------------------------------------------------------===//

/// Assign an operator type (latency + delays) to every op reached by \p walkFn,
/// sourced from \p lib. A `ChainingProblem` also receives incoming/outgoing
/// delays. A scheduled sync call is a fixed-latency node (see
/// `scheduledCallLatency`); everything else is characterized by the library.
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

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORLIBRARY_H
