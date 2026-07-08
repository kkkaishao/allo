/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULER_H
#define ALLO_SCHEDULING_SCHEDULER_H

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::allo {

/// Check, solve (via CIRCT's SDC simplex), and verify \p problem, minimizing the
/// start time of \p anchor. Templated so the static problem type selects the
/// right scheduleSimplex overload (e.g. ModuloProblem -> modulo scheduler).
template <typename ProblemT>
LogicalResult solveSchedulingProblem(ProblemT &problem, Operation *anchor) {
  if (failed(problem.check()))
    return failure();
  if (failed(circt::scheduling::scheduleSimplex(problem, anchor)))
    return failure();
  if (failed(problem.verify()))
    return failure();
  return success();
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
