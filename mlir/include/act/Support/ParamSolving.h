#ifndef ACT_SUPPORT_PARAM_SOLVING_H
#define ACT_SUPPORT_PARAM_SOLVING_H

#include "act/Support/SemanticMatching.h"
#include "act/Support/SymbolicExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <cassert>
#include <utility>

namespace mlir::act {

enum class AddrParamKind {
  Shape,
  Offset,
  Mixed,
};

enum class AccessRole {
  Read,
  Write,
  ReadWrite,
};

inline bool accessReads(AccessRole role) {
  return role == AccessRole::Read || role == AccessRole::ReadWrite;
}

inline bool accessWrites(AccessRole role) {
  return role == AccessRole::Write || role == AccessRole::ReadWrite;
}

struct SymbolicRegion {
  SymShape basis;
  SymShape counts;
  SymShape strides;
};

struct SymbolicAccess {
  unsigned operandIdx = 0;
  StringAttr bufferName;
  BufferTypeInterface bufferType;
  AccessRole role = AccessRole::Read;
  SymbolicRegion storage;
  SymShape visibleShape;
};

struct InstructionParamModel {
  DefineOp defineOp;
  SmallVector<SymbolicAccess, 4> accesses;
  DenseMap<unsigned, AddrParamKind> paramKinds;
};

struct ParamSolution {
  SemanticGraphNode &node;
  InstructionParamModel model;
  DenseMap<unsigned, int64_t> solvedParams;
  bool isValid = false;

  ParamSolution(SemanticGraphNode &node, InstructionParamModel model)
      : node(node), model(std::move(model)) {}
  ParamSolution(const ParamSolution &) = default;
  ParamSolution(ParamSolution &&) = default;

  ParamSolution &operator=(const ParamSolution &other) {
    assert(&node == &other.node && "ParamSolution cannot rebind node");
    model = other.model;
    solvedParams = other.solvedParams;
    isValid = other.isValid;
    return *this;
  }

  ParamSolution &operator=(ParamSolution &&other) {
    assert(&node == &other.node && "ParamSolution cannot rebind node");
    model = std::move(other.model);
    solvedParams = std::move(other.solvedParams);
    isValid = other.isValid;
    return *this;
  }
};

using GraphParamSolution = SmallVector<ParamSolution, 4>;

FailureOr<InstructionParamModel> buildInstructionParamModel(DefineOp defineOp,
                                                            ModuleOp module);

FailureOr<GraphParamSolution> runParamSolving(SemanticGraph &graph);
} // namespace mlir::act

#endif // ACT_SUPPORT_PARAM_SOLVING_H
