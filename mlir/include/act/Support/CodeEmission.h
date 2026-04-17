#ifndef ACT_SUPPORT_CODE_EMISSION_H
#define ACT_SUPPORT_CODE_EMISSION_H

#include "act/Support/ParamSolving.h"
#include "act/Support/Planning.h"
#include "act/Support/SemanticMatching.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::act {

struct RegionLoweringPlan {
  SemanticsGraph graph;
  GraphParamSolution paramSolution;
  LogicalPlan logicalPlan;
  ResourcePlan resourcePlan;
};

struct FunctionLoweringPlan {
  func::FuncOp func;
  SemanticsGraph graph;
  GraphParamSolution solution;
  LogicalPlan logicalPlan;
  ResourcePlan resourcePlan;
  DenseSet<Operation *> coveredSemanticOps;
  bool isComplete = false;
  explicit FunctionLoweringPlan(func::FuncOp func) : func(func) {}
};

FailureOr<FunctionLoweringPlan>
buildFunctionLoweringPlan(func::FuncOp func, ModuleOp module,
                          InstructionCatalog &catalog);

LogicalResult emitInstructionSequence(RewriterBase &rewriter,
                                      FunctionLoweringPlan &plan);

} // namespace mlir::act

#endif // ACT_SUPPORT_CODE_EMISSION_H
