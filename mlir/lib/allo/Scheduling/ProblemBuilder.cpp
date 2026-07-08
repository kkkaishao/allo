/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ProblemBuilder.h"

#include "allo/IR/AlloOps.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace mlir::allo {

ModuloProblem buildCyclicProblem(AffineForOp forOp, DependenceAnalysis &deps) {
  ModuloProblem problem(forOp);

  // Insert memory and stream dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    problem.insertOperation(op);

    for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Only model dependences whose source is also inside this loop. Whole-func
      // analysis may surface cross-region dependences whose source is not part
      // of this loop's scheduling problem (its endpoints are scheduled
      // elsewhere); those are handled by cross-region analysis, not here.
      if (!forOp.getBody()->findAncestorOpInBlock(*memoryDep.source))
        continue;

      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. A
      // loop-independent (intra-iteration) dependence carries no components and
      // maps to distance 0.
      unsigned distance = memoryDep.dependenceComponents.empty()
                              ? 0
                              : *memoryDep.dependenceComponents.back().lb;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Problem::Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Anchor: side-effecting ops (stores and stream accesses) must be scheduled
  // before the loop terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (!isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
      // If it's not an operation, we don't need to model the dependence.
      if (!iterArgDefiner)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(iterArgDefiner, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  return problem;
}

SharedOperatorsProblem buildAcyclicProblem(ArrayRef<Operation *> ops,
                                           DependenceAnalysis &deps) {
  assert(!ops.empty() && "straight-line region must have at least one op");
  SharedOperatorsProblem problem(ops.front());

  // Collect the span's op set (all nested ops) for intra-span dep filtering.
  DenseSet<Operation *> spanOps;
  for (Operation *top : ops)
    top->walk([&](Operation *op) { spanOps.insert(op); });

  // Insert ops + intra-span memory/stream dependences. Acyclic: no distance,
  // and no loop-carried / iter_arg handling.
  for (Operation *top : ops)
    top->walk([&](Operation *op) {
      problem.insertOperation(op);

      for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
        if (!hasDependence(memoryDep.dependenceType))
          continue;
        // Only intra-span dependences belong to this problem.
        if (!spanOps.contains(memoryDep.source))
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
    });

  // Make the last program-order op a unique sink via auxiliary dependences, so
  // that minimizing its start time yields an ASAP schedule for the whole span.
  Operation *sink = ops.back();
  for (Operation *op : problem.getOperations()) {
    if (op == sink)
      continue;
    (void)problem.insertDependence(Problem::Dependence(op, sink));
  }

  return problem;
}

} // namespace mlir::allo
