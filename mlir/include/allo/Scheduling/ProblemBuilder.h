/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_PROBLEMBUILDER_H
#define ALLO_SCHEDULING_PROBLEMBUILDER_H

#include "allo/Scheduling/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::allo {

/// Build a cyclic (modulo) scheduling problem for one innermost affine loop:
/// registers the body ops, their memory/stream dependences (with inter-iteration
/// distances), conditional value-flow, a terminator anchor for side-effecting
/// ops, and loop-carried (iter_arg) recurrences.
circt::scheduling::ModuloProblem
buildCyclicProblem(affine::AffineForOp forOp, DependenceAnalysis &deps);

/// Build an acyclic resource problem for a straight-line region (the top-level
/// \p ops of a maximal non-loop run). Registers the ops with their intra-span
/// memory/stream dependences (no inter-iteration distance) and makes the last
/// program-order op the unique sink (so minimizing it schedules the span ASAP).
/// SSA def-use is modeled implicitly by the scheduling problem.
circt::scheduling::SharedOperatorsProblem
buildAcyclicProblem(ArrayRef<Operation *> ops, DependenceAnalysis &deps);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_PROBLEMBUILDER_H
