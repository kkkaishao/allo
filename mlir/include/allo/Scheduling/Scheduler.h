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

/// A resource-constrained problem whose shared instances need not be fully
/// pipelined. CIRCT's `SharedOperatorsProblem` assumes they are: a limited
/// operation holds its unit for exactly the cycle it issues in, so the only
/// resource fact the problem carries is a per-type limit. Real operators break
/// that assumption. A non-pipelined memory port holds its port for the whole
/// access, a synchronous sub-kernel call holds its child instance until the
/// child is done, and a child loop standing in for a whole nested pipeline
/// holds it for that pipeline's latency. Two such operations may not overlap
/// on one unit even though they start in different cycles.
///
/// This problem carries that occupancy window per operation, so everything the
/// resource model knows lives in the problem: nothing is stashed on the IR, and
/// `verifyOccupancy` sees the whole reservation rather than only its first
/// cycle.
///
/// An operation may hold several units at once (a memory port and a shared
/// functional unit, say). It takes them all at its start time and holds each
/// for the same window, so a cycle is feasible for it only where every one of
/// them has room. `setLinkedResourceTypes` states an operation's complete unit
/// list; the schedulers reserve all or none of it.
///
/// It also drops CIRCT's rule that a limited operation must have a non-zero
/// latency. A combinational access still occupies its port for the cycle it
/// issues in and contends like any other; rejecting it would force the port
/// model to leave those accesses unconstrained.
class OccupancyProblem
    : public virtual circt::scheduling::SharedOperatorsProblem {
public:
  static constexpr auto name = "OccupancyProblem";
  using circt::scheduling::SharedOperatorsProblem::SharedOperatorsProblem;

protected:
  OccupancyProblem() = default;
  /// A limited operation may have zero latency (see the class comment).
  LogicalResult checkLatency(Operation *op) override;

public:
  /// The number of consecutive cycles \p op holds its resource unit, counting
  /// from its start time. One (the fully-pipelined case) unless set.
  unsigned getResourceCycles(Operation *op) {
    return resourceCycles.lookup(op).value_or(1);
  }
  void setResourceCycles(Operation *op, unsigned cycles) {
    resourceCycles[op] = cycles;
  }

  /// No limited resource is oversubscribed in any cycle, counting each
  /// operation's whole occupancy window. \p ii == 0 checks an acyclic
  /// schedule; a non-zero \p ii checks the windows modulo the initiation
  /// interval. This is not an override of `verifyUtilization`, which
  /// `ModuloProblem` already claims: the concrete problems below call it from
  /// their `verify`.
  LogicalResult verifyOccupancy(unsigned ii);

private:
  OperationProperty<unsigned> resourceCycles;
};

/// The cyclic twin: CIRCT's `ModuloProblem` with occupancy windows, i.e.
/// reservations that span several congruence classes modulo the II.
class ModuloOccupancyProblem : public virtual circt::scheduling::ModuloProblem,
                               public virtual OccupancyProblem {
public:
  static constexpr auto name = "ModuloOccupancyProblem";
  using circt::scheduling::ModuloProblem::ModuloProblem;

protected:
  ModuloOccupancyProblem() = default;

public:
  LogicalResult verify() override;
};

/// A cyclic, resource-constrained, chaining-enabled scheduling problem: the
/// composition of CIRCT's `ChainingProblem` and `ModuloOccupancyProblem`.
/// Defined here (rather than in CIRCT) so the Allo scheduler is self-contained;
/// it derives from CIRCT's public base problems and mirrors CIRCT's
/// `ChainingCyclicProblem` diamond. Solving it yields an integer II, integer
/// start times, and per-op sub-cycle start times that respect a target cycle
/// time, under modulo resource constraints.
class ChainingModuloProblem : public virtual circt::scheduling::ChainingProblem,
                              public virtual ModuloOccupancyProblem {
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
/// composition of CIRCT's `ChainingProblem` and `OccupancyProblem`. The
/// straight-line twin of `ChainingModuloProblem` (no initiation interval / no
/// inter-iteration distance). Solving it yields integer start times and per-op
/// sub-cycle start times that respect a target cycle time, under per-cycle
/// resource limits.
class ChainingSharedOperatorsProblem
    : public virtual circt::scheduling::ChainingProblem,
      public virtual OccupancyProblem {
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
LogicalResult scheduleSimplex(OccupancyProblem &prob, Operation *lastOp);
LogicalResult scheduleSimplex(ModuloOccupancyProblem &prob, Operation *lastOp);
LogicalResult scheduleSimplex(circt::scheduling::ChainingProblem &prob,
                              Operation *lastOp, float cycleTime);
LogicalResult scheduleSimplex(circt::scheduling::ChainingCyclicProblem &prob,
                              Operation *lastOp, float cycleTime);
/// What the SDC heuristic contributes to a solve that is not its own: the II
/// bound it settles before placing anything, and whether its greedy placement
/// then reached a schedule worth hinting from and falling back to.
///
/// Asking for one also makes a PLACEMENT failure advisory. The greedy leaves
/// the problem unscheduled and `placed` stays false, but the call still
/// succeeds, because the caller is going to place the region itself and the
/// regions the greedy cannot place are exactly the ones an exact solver is for.
/// A failure underneath it is not advisory and still fails the call: the
/// resource-free LP is exact, so its infeasibility says no schedule exists at
/// any II, which no solver can repair.
struct SimplexWarmStart {
  /// The largest II any bound justifies before resources are placed: the
  /// resource-min II, a loop-carried recurrence, and the pipeline directive's
  /// floor, whichever is largest. Everything past that point is greedy
  /// placement, so this is where an exact II search has to start.
  unsigned lowerBoundII = 1;
  /// Whether the greedy placement reached a schedule, i.e. whether the problem
  /// now carries start times and an initiation interval.
  bool placed = false;
};

/// \p minII is a lower bound on the initiation interval (from a pipeline
/// directive); the achieved II is max(\p minII, the natural minimum). The
/// default 1 imposes no additional bound.
///
/// \p warm, when given, receives the warm start above and switches placement
/// failures to advisory.
LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, unsigned minII = 1,
                              SimplexWarmStart *warm = nullptr);
LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime);

//===----------------------------------------------------------------------===//
// CP-SAT exact schedulers.
//
// Which solver settles the RESOURCE half of a problem. The SDC simplex is
// exact for the difference constraints either way (a network matrix is totally
// unimodular, so its LP optimum is already integral); what differs is only the
// resource placement, which the simplex path leaves to the MRT plus a greedy
// heuristic and this path solves as one constraint program.
//===----------------------------------------------------------------------===//

enum class SchedulerKind {
  /// The SDC simplex plus greedy modulo / shared-operator placement.
  Heuristic,
  /// CP-SAT over the same problem: exact under the model, and available only
  /// in a build with OR-Tools.
  Exact,
};

/// \p name ("heuristic" / "exact") as a kind, or nullopt when it names neither.
std::optional<SchedulerKind> parseSchedulerKind(StringRef name);

/// Whether this build has the CP-SAT exact scheduler compiled in.
bool hasExactScheduler();

/// Solve \p prob exactly with CP-SAT, minimizing the start time of \p lastOp
/// under the target clock period \p cycleTime. Reports `unsupported` and fails
/// in a build without OR-Tools, so callers dispatch on the requested kind and
/// never on the build configuration.
LogicalResult scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                            Operation *lastOp, float cycleTime);
/// Cyclic twin; \p minII lower-bounds the initiation interval.
LogicalResult scheduleCPSAT(ChainingModuloProblem &prob, Operation *lastOp,
                            float cycleTime, unsigned minII);

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
/// == 1 imposes no additional bound. \p kind selects the resource solver; both
/// paths are wrapped by the same `check` and `verify`, so an exact solve is
/// held to the model the heuristic is held to.
inline LogicalResult solveSchedulingProblem(ChainingModuloProblem &problem,
                                            Operation *anchor, float cycleTime,
                                            unsigned minII,
                                            SchedulerKind kind) {
  if (failed(problem.check()))
    return failure();
  if (kind == SchedulerKind::Exact) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime, minII)))
      return failure();
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime,
                                                minII))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Acyclic twin of the variant above.
inline LogicalResult
solveSchedulingProblem(ChainingSharedOperatorsProblem &problem,
                       Operation *anchor, float cycleTime, SchedulerKind kind) {
  if (failed(problem.check()))
    return failure();
  if (kind == SchedulerKind::Exact) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime)))
      return failure();
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

LogicalResult runPreScheduleVerification(ModuleOp module, StringRef top);
LogicalResult runSDCScheduler(ModuleOp module, StringRef top, float cycleTime,
                              SchedulerKind kind);
void runPostScheduleConversion(ModuleOp module);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
