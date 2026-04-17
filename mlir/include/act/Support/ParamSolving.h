#ifndef ACT_SUPPORT_PARAM_SOLVING_H
#define ACT_SUPPORT_PARAM_SOLVING_H

#include "act/Support/SemanticMatching.h"
#include "act/Support/SymbolicExpr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::act {

/// Classification of addr parameters.
enum class AddrParamKind {
  Shape,  // appears in counts/output_shape — controls computation size
  Offset, // appears in basis — controls data position
  Mixed,  // appears in both
};

/// Solved parameters for one matched node.
struct ParamSolution {
  SemanticsGraphNode *node = nullptr;

  /// Solved shape params: addr block arg index -> concrete value.
  DenseMap<unsigned, int64_t> solvedParams;

  /// Param classifications.
  DenseMap<unsigned, AddrParamKind> paramKinds;

  /// Whether the solution is valid (all shapes match).
  bool isValid = false;

  DefineOp getInstruction() const { return node ? node->instruction : nullptr; }
};

using GraphParamSolution = SmallVector<ParamSolution, 4>;

/// Run parameter solving on a SemanticsGraph.
/// For each matched node, solve addr parameters so source shapes fit the
/// instruction. Rejects (marks invalid) when shapes don't fit — no tiling.
FailureOr<GraphParamSolution> runParamSolving(SemanticsGraph &graph,
                                              ModuleOp module);

} // namespace mlir::act

#endif // ACT_SUPPORT_PARAM_SOLVING_H
