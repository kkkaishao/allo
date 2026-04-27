#include "act/Support/CodeEmission.h"

#include "act/IR/ActOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "code-emission"

using namespace mlir;
using namespace mlir::act;

using llvm::dbgs;

static FailureOr<SmallVector<int64_t, 4>>
getStaticAddrParams(DefineOp instruction,
                    DenseMap<unsigned, int64_t> &paramBindings,
                    Operation *errorOp) {
  unsigned numParams = instruction.getAccessBlock().getNumArguments();
  SmallVector<int64_t, 4> params(numParams, 0);
  for (unsigned idx = 0; idx < numParams; ++idx) {
    auto it = paramBindings.find(idx);
    if (it == paramBindings.end())
      return errorOp->emitError() << "missing static addr param p" << idx
                                  << " for @" << instruction.getSymName();
    params[idx] = it->second;
  }
  return params;
}

static FailureOr<SmallVector<int64_t, 4>>
getStaticComputeParams(DefineOp instruction, Operation *errorOp) {
  unsigned numParams = instruction.getExtraComputeArgs().size();
  if (numParams != 0)
    return errorOp->emitError()
           << "static compute params are not supported yet for @"
           << instruction.getSymName();
  return SmallVector<int64_t, 4>{};
}

static LogicalResult emitInstruction(RewriterBase &rewriter, Location loc,
                                     DefineOp instruction,
                                     DenseMap<unsigned, int64_t> &addrBindings,
                                     Operation *errorOp) {
  auto addrParams = getStaticAddrParams(instruction, addrBindings, errorOp);
  if (failed(addrParams))
    return failure();

  auto computeParams = getStaticComputeParams(instruction, errorOp);
  if (failed(computeParams))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  LLVM_DEBUG({
    dbgs() << "emit @" << instruction.getSymName() << " addr(";
    llvm::interleaveComma(*addrParams, dbgs());
    dbgs() << ")\n";
  });

  EmitOp::create(
      rewriter, loc, FlatSymbolRefAttr::get(ctx, instruction.getSymName()),
      ValueRange{}, ValueRange{}, DenseI64ArrayAttr::get(ctx, *addrParams),
      DenseI64ArrayAttr::get(ctx, *computeParams));
  return success();
}

static LogicalResult emitScheduleStep(RewriterBase &rewriter,
                                      ExecutionPlan &plan,
                                      PlanScheduleStep &step) {
  Location loc = plan.func.getLoc();
  switch (step.kind) {
  case PlanScheduleKind::Compute: {
    assert(step.nodeIdx < plan.nodes.size() && "invalid compute node id");
    PlanNode &node = plan.nodes[step.nodeIdx];
    return emitInstruction(rewriter, loc, node.instruction, node.paramBindings,
                           plan.func);
  }
  case PlanScheduleKind::Move: {
    assert(step.nodeIdx < plan.moveNodes.size() && "invalid move node id");
    PlanMoveNode &node = plan.moveNodes[step.nodeIdx];
    return emitInstruction(rewriter, loc, node.instruction, node.paramBindings,
                           plan.func);
  }
  }
  llvm_unreachable("unknown schedule step kind");
}

LogicalResult act::emitInstructionSequence(RewriterBase &rewriter,
                                           ExecutionPlan &plan) {
  func::FuncOp func = plan.func;
  assert(func && "expected function in execution plan");

  rewriter.setInsertionPointAfter(func);
  auto sequence =
      SequenceOp::create(rewriter, func.getLoc(), func.getSymNameAttr());
  sequence->setDiscardableAttrs(func->getDiscardableAttrDictionary());
  Block *entryBlock = sequence.addEntryBlock();

  rewriter.setInsertionPointToStart(entryBlock);
  for (PlanScheduleStep &step : plan.schedule) {
    if (failed(emitScheduleStep(rewriter, plan, step))) {
      rewriter.eraseOp(sequence);
      return failure();
    }
  }

  rewriter.eraseOp(func);
  return success();
}
