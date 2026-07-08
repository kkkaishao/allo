/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULER_H
#define ALLO_SCHEDULING_SCHEDULER_H

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::allo {

/// A cyclic, resource-constrained, chaining-enabled scheduling problem: the
/// composition of CIRCT's `ChainingProblem` and `ModuloProblem`. Defined here
/// (rather than in CIRCT) so the Allo scheduler is self-contained; it derives
/// from CIRCT's public base problems and mirrors CIRCT's
/// `ChainingCyclicProblem` diamond. Solving it yields an integer II, integer
/// start times, and per-op sub-cycle start times that respect a target cycle
/// time, under modulo resource constraints.
class ChainingModuloProblem : public virtual circt::scheduling::ChainingProblem,
                              public virtual circt::scheduling::ModuloProblem {
public:
  static constexpr auto name = "ChainingModuloProblem";
  using circt::scheduling::ChainingProblem::ChainingProblem;

protected:
  ChainingModuloProblem() = default;

public:
  LogicalResult checkDefUse(circt::scheduling::Problem::Dependence dep);
  LogicalResult check() override;
  LogicalResult verify() override;
};

/// An acyclic, resource-constrained, chaining-enabled scheduling problem: the
/// composition of CIRCT's `ChainingProblem` and `SharedOperatorsProblem`. The
/// straight-line twin of `ChainingModuloProblem` (no initiation interval / no
/// inter-iteration distance). Solving it yields integer start times and per-op
/// sub-cycle start times that respect a target cycle time, under per-cycle
/// resource limits.
class ChainingSharedOperatorsProblem
    : public virtual circt::scheduling::ChainingProblem,
      public virtual circt::scheduling::SharedOperatorsProblem {
public:
  static constexpr auto name = "ChainingSharedOperatorsProblem";
  using circt::scheduling::ChainingProblem::ChainingProblem;

protected:
  ChainingSharedOperatorsProblem() = default;

public:
  LogicalResult check() override;
  LogicalResult verify() override;
};

//===----------------------------------------------------------------------===//
// SDC simplex schedulers.
//
// Self-contained fork of CIRCT's `scheduleSimplex` family (implementation in
// Scheduler.cpp). Reuses CIRCT's public Problem data model; the solver is ours
// to instrument and extend. Callers should use these via
// `solveSchedulingProblem` below or by fully-qualified name
// (`mlir::allo::scheduleSimplex`) to avoid ambiguity with the CIRCT overloads.
//===----------------------------------------------------------------------===//

LogicalResult scheduleSimplex(circt::scheduling::Problem &prob,
                              Operation *lastOp);
LogicalResult scheduleSimplex(circt::scheduling::CyclicProblem &prob,
                              Operation *lastOp);
LogicalResult scheduleSimplex(circt::scheduling::SharedOperatorsProblem &prob,
                              Operation *lastOp);
LogicalResult scheduleSimplex(circt::scheduling::ModuloProblem &prob,
                              Operation *lastOp);
LogicalResult scheduleSimplex(circt::scheduling::ChainingProblem &prob,
                              Operation *lastOp, float cycleTime);
LogicalResult scheduleSimplex(circt::scheduling::ChainingCyclicProblem &prob,
                              Operation *lastOp, float cycleTime);
/// \p minII is a lower bound on the initiation interval (from a pipeline
/// directive); the achieved II is max(\p minII, the natural minimum). The
/// default 1 imposes no additional bound.
LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, unsigned minII = 1);
LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime);

/// Check, solve (via our SDC simplex), and verify \p problem, minimizing the
/// start time of \p anchor. Templated so the static problem type selects the
/// right scheduleSimplex overload (e.g. ModuloProblem -> modulo scheduler).
template <typename ProblemT>
LogicalResult solveSchedulingProblem(ProblemT &problem, Operation *anchor) {
  if (failed(problem.check()))
    return failure();
  if (failed(mlir::allo::scheduleSimplex(problem, anchor)))
    return failure();
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Chaining variant: the scheduler additionally enforces the target clock
/// period
/// \p cycleTime (ns) by breaking combinational chains across cycle boundaries.
template <typename ProblemT>
LogicalResult solveSchedulingProblem(ProblemT &problem, Operation *anchor,
                                     float cycleTime) {
  if (failed(problem.check()))
    return failure();
  if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime)))
    return failure();
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Chaining modulo variant with a target-II lower bound (from a pipeline
/// directive): the achieved II is max(\p minII, the natural minimum). \p minII
/// == 1 imposes no additional bound.
inline LogicalResult solveSchedulingProblem(ChainingModuloProblem &problem,
                                            Operation *anchor, float cycleTime,
                                            unsigned minII) {
  if (failed(problem.check()))
    return failure();
  if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime, minII)))
    return failure();
  if (failed(problem.verify()))
    return failure();
  return success();
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
