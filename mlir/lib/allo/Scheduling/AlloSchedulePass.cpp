/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/Passes.h"

#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/OperatorModel.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/ScheduleResult.h"
#include "allo/Scheduling/Scheduler.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::allo {
#define GEN_PASS_DEF_ALLOSCHEDULEPASS
#include "allo/Scheduling/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace circt::scheduling;

namespace {
// Whether \p loop contains a nested affine loop (i.e. is not truly innermost).
bool hasNestedLoop(AffineForOp loop) {
  bool found = false;
  loop.getBody()->walk([&](AffineForOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct AlloSchedulePass
    : public allo::impl::AlloSchedulePassBase<AlloSchedulePass> {
  using AlloSchedulePassBase::AlloSchedulePassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getFunctionBody().empty())
      return;

    // Whole-func memory + stream dependence analysis.
    auto &deps = getAnalysis<DependenceAnalysis>();

    // Coarse cross-region dependence graph (analysis only; does not affect the
    // per-loop scheduling below).
    if (dumpRegionGraph)
      printRegionGraphDot(deps.getRegionGraph(), funcOp, llvm::errs());

    // Schedule each region: loops as cyclic (modulo) problems, straight-line
    // spans as acyclic problems. Cross-region composition is by program order /
    // SSA only -- no cross-region overlap in v1. Scheduling only adds
    // attributes, so iterating a region snapshot is safe.
    for (const SchedRegion &region : enumerateRegions(funcOp)) {
      if (region.kind == allo::RegionKind::Loop) {
        auto loop = cast<AffineForOp>(region.anchor());
        SmallVector<AffineForOp> nestedLoops;
        getPerfectlyNestedLoops(nestedLoops, loop);
        // Schedule the innermost body of a perfect nest. Composing the outer
        // levels into a schedule (loop-as-node) and imperfect nesting are
        // deferred, so skip any body that still contains a loop.
        AffineForOp body = nestedLoops.back();
        if (hasNestedLoop(body)) {
          body.emitRemark("allo-schedule: imperfect loop nesting not scheduled");
          continue;
        }

        ModuloProblem problem = buildCyclicProblem(body, deps);
        if (failed(populateOperatorTypes(*body.getBody(), problem)))
          return signalPassFailure();
        if (failed(solveSchedulingProblem(problem,
                                          body.getBody()->getTerminator())))
          return signalPassFailure();
        annotateRegion(problem, funcOp, region.id, "cyclic",
                       problem.getInitiationInterval(), region.id);
      } else {
        SharedOperatorsProblem problem = buildAcyclicProblem(region.ops, deps);
        if (failed(populateOperatorTypes(ArrayRef<Operation *>(region.ops),
                                         problem)))
          return signalPassFailure();
        if (failed(solveSchedulingProblem(problem, region.ops.back())))
          return signalPassFailure();
        annotateRegion(problem, funcOp, region.id, "acyclic", std::nullopt,
                       region.id);
      }
    }
  }
};
} // namespace
