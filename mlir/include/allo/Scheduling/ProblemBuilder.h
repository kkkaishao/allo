/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_PROBLEMBUILDER_H
#define ALLO_SCHEDULING_PROBLEMBUILDER_H

#include "allo/Scheduling/DependenceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::allo {

class OperatorLibrary;

/// Build a cyclic scheduling problem for one counted loop (`affine.for` or
/// `scf.for`): registers the body ops, their memory/stream dependences (with
/// inter-iteration distances), conditional value-flow, a terminator anchor for
/// side-effecting ops, and loop-carried (iter_arg) recurrences. \p ProblemT is
/// a `CyclicProblem` subclass; the pass instantiates it for
/// `ChainingModuloProblem`.
template <class ProblemT>
ProblemT buildCyclicProblem(LoopLikeOpInterface loop, DependenceAnalysis &deps);

/// Whether an `scf.while` forwards all before-args to the after region 1:1
/// (identity forwarding, equal arity): the shape `buildWhileProblem` schedules,
/// aligning inits/before-args/after-args/yield/results by one slot index.
bool whileHasIdentityForwarding(scf::WhileOp w);

/// Whether an `scf.while`'s continue-condition is combinational, i.e. settled
/// the cycle the loop issues, so the while can flushing-pipeline. False when
/// the condition cone (the before region, which under identity forwarding only
/// computes the condition) holds a multi-cycle op per \p lib: a memory read
/// (`while (A[i] != key)`) or a latency IP (a float compare, `while (r >
/// tol)`). A non-combinational condition routes to the sequential CHECK/RUN
/// controller, which waits for the condition to settle. The scheduler and the
/// reifier share this one predicate, so they route in lockstep.
bool conditionIsCombinational(scf::WhileOp w, const OperatorLibrary &lib);

/// Build a cyclic scheduling problem for an uncounted `scf.while` (its before +
/// after regions scheduled as one iteration): registers both regions' ops +
/// memory/stream deps, the non-speculative condition gate (`cond -> after`,
/// dist 0), the loop-carried state recurrence (`yield[j] -> readers(state[j])`,
/// dist 1), and a side-effect anchor before `scf.yield`. Requires
/// `whileHasIdentityForwarding(w)`. \p ProblemT is a `CyclicProblem` subclass.
template <class ProblemT>
ProblemT buildWhileProblem(scf::WhileOp w, DependenceAnalysis &deps);

/// Build an acyclic scheduling problem for a straight-line region (the
/// top-level
/// \p ops of a maximal non-loop run). Registers the ops with their intra-span
/// memory/stream dependences (no inter-iteration distance) and makes the last
/// program-order op the unique sink (so minimizing it schedules the span ASAP).
/// SSA def-use is modeled implicitly. The pass instantiates \p ProblemT for
/// `ChainingSharedOperatorsProblem`.
template <class ProblemT>
ProblemT buildAcyclicProblem(ArrayRef<Operation *> ops,
                             DependenceAnalysis &deps);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_PROBLEMBUILDER_H
