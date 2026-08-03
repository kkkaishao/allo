/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULER_H
#define ALLO_SCHEDULING_SCHEDULER_H

#include "allo/Scheduling/ScheduleModel.h"

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::allo {

/// A resource-constrained problem whose shared instances need not be fully
/// pipelined: unlike CIRCT's `SharedOperatorsProblem` (a per-type limit only,
/// assuming an op holds its unit for exactly its issue cycle), this problem
/// carries a per-operation occupancy window, so a synchronous call that holds
/// its callee's instance until the callee is done can be modeled
/// (`populateCallOccupancy`).
///
/// An operation may hold several units at once (a memory port and a shared
/// functional unit, say); `setLinkedResourceTypes` states its complete unit
/// list, and a cycle is feasible for it only where every unit in that list has
/// room across the whole window.
///
/// A limited operation may also have zero latency here (CIRCT requires
/// non-zero): a combinational access still occupies its port for the cycle it
/// issues in and contends like any other.
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

  /// Whether \p op holds at least one unit whose count is capped, i.e. whether
  /// it contends for anything. An unlimited link constrains nothing and no
  /// reservation tracks it, so this is the half of a problem a resource solver
  /// actually decides.
  bool holdsLimitedUnit(Operation *op);

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

/// The chain-breaking edges \p prob needs to meet \p cycleTime: for every
/// combinational path whose accumulated delay would not fit the period, an
/// auxiliary edge from the path's ORIGIN to the operation, which both solvers
/// weigh one cycle more than a plain dependence. Schedule-independent, so a
/// caller may run it before or after solving.
///
/// Visits operations in topological order (`Problem::getOperations()` is only
/// insertion-ordered) and marks one "handled" (`chains[op][op] = 0`) only once
/// every predecessor's chain map is complete, so a successor never inherits a
/// half-built map. A registered predecessor's outgoing chain is maxed against
/// any longer chain reaching the same origin through another route.
LogicalResult computeChainBreaks(
    circt::scheduling::ChainingProblem &prob, float cycleTime,
    SmallVectorImpl<circt::scheduling::Problem::Dependence> &result);

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
/// reached a schedule.
///
/// Passing one also makes a PLACEMENT failure advisory: the call still
/// succeeds with `placed == false`, since the caller will place the region
/// itself. A failure in the resource-free LP below placement is not advisory
/// and still fails the call, since that LP is exact: infeasible there means no
/// schedule exists at any II.
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
// What a solve is charged: the span objective.
//===----------------------------------------------------------------------===//

/// One region OUTPUT's contribution to the region's drain: it commits at
/// `start(op) + offset`.
///
/// The drain is the max over these (`drainOf`), and the exact scheduler bounds
/// its own drain variable below by each one, so the quantity a solve minimizes
/// and the quantity `leafSpan` charges are ONE expression rather than two
/// derivations that happen to agree. Only `SDC.cpp` classifies the outputs, it
/// being the layer that still holds the region's results.
struct DrainTerm {
  Operation *op;
  int64_t offset;
};

/// The drain of a SOLVED problem: the cycle its deepest output commits.
inline int64_t drainOf(circt::scheduling::Problem &problem,
                       ArrayRef<DrainTerm> terms) {
  int64_t drain = 0;
  for (const DrainTerm &term : terms)
    drain =
        std::max(drain, static_cast<int64_t>(*problem.getStartTime(term.op)) +
                            term.offset);
  return drain;
}

/// One value a region spends a DELAY REGISTER chain on. The chain is as long as
/// its deepest reader needs and costs one flip-flop per bit per cycle of that:
///
/// ```
/// depth(v) = max over reads ( t_read + ii * distance ) - ( t_def + latency )
/// cost(v)  = width * depth(v)
/// ```
///
/// No register is shared between two values (`insertRegister` keys one chain
/// per value and region), which is what makes this a SUM over values that is
/// linear in the schedule rather than a MAXLIVE over time coupled to an
/// allocation, and so a term an objective can carry directly.
struct RegisterTerm {
  Operation *def;
  /// Cycles after `def` issues before the value is readable.
  int64_t latency;
  /// Flip-flops one cycle of delay costs.
  int64_t width;
  /// Each reader, and the iteration distance its read spans.
  SmallVector<std::pair<Operation *, int64_t>> reads;
};

/// What a region's span is charged, and so what the exact scheduler minimizes:
/// `(trip - 1) * ii + drain`, the part of `leafSpan` a solve controls, with the
/// region's register cost as the tie-break below it.
///
/// The heuristic ignores this and keeps minimizing the anchor's start time,
/// which is a max over every SINK where the drain is a max over the OUTPUTS
/// alone, i.e. an over-constrained proxy for the quantity actually charged.
struct SpanObjective {
  /// The region's outputs.
  ArrayRef<DrainTerm> drain;
  /// The values it spends a delay register on.
  ArrayRef<RegisterTerm> regs;
  /// The region's trip count, when it is a compile-time constant. Empty leaves
  /// the exact scheduler on the anchor-start objective, which is the right one
  /// wherever no span composes off this solve (a `while`, a dynamic bound) or
  /// wherever iterations do not overlap and it is the schedule DEPTH, not the
  /// drain, that the trip multiplies (`s.pipeline(ii=-1)`).
  std::optional<int64_t> trip;
};

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
  /// in a build with OR-Tools. The chain breaks stay the pre-pass's, so the
  /// only thing that differs from the heuristic is resource placement, and any
  /// difference between the two schedules is attributable to it.
  Exact,
  /// As above, but the chain breaks are decided in the constraint program too.
  /// The pre-pass breaks a too-long chain at its ORIGIN, which forces origin
  /// and operation into different cycles where a register mid-chain would have
  /// done; deciding it in the model lets the solver put the break where it is
  /// cheapest, and pay for it against the same span and area objective.
  ExactChaining,
};

/// Whether \p kind solves the resource half with CP-SAT, i.e. needs OR-Tools.
inline bool usesExactScheduler(SchedulerKind kind) {
  return kind != SchedulerKind::Heuristic;
}

/// How much work ONE SOLVE may consume, in OR-Tools' DETERMINISTIC time units
/// (roughly one core-second on current hardware). Deterministic rather than
/// wall-clock so a compile is reproducible across machines.
///
/// Charged per solve, not per region: a cyclic region's search spends this
/// budget again for every initiation interval it probes.
inline constexpr double kDefaultSolveBudget = 30.0;

/// What the caller asked the scheduler for.
struct SchedulerOptions {
  SchedulerKind kind = SchedulerKind::Heuristic;
  double budget = kDefaultSolveBudget;
};

/// \p name ("heuristic" / "exact" / "exact-chaining") as a kind, or nullopt
/// when it names none of them.
std::optional<SchedulerKind> parseSchedulerKind(StringRef name);

/// Whether this build has the CP-SAT exact scheduler compiled in.
bool hasExactScheduler();

/// Solve \p prob exactly with CP-SAT, minimizing \p span under the target clock
/// period \p cycleTime. Reports `unsupported` and fails in a build without
/// OR-Tools, so callers dispatch on the requested kind and never on the build
/// configuration.
///
/// \p opts carries which chaining form the model states (see
/// `SchedulerKind::ExactChaining`) and what the region's solve may spend.
LogicalResult scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                            Operation *lastOp, float cycleTime,
                            const SpanObjective &span,
                            const SchedulerOptions &opts);
/// Cyclic twin; \p minII lower-bounds the initiation interval, and the search
/// over intervals is a branch and bound on \p span.
LogicalResult scheduleCPSAT(ChainingModuloProblem &prob, Operation *lastOp,
                            float cycleTime, unsigned minII,
                            const SpanObjective &span,
                            const SchedulerOptions &opts);

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
/// == 1 imposes no additional bound. \p opts selects the resource solver; both
/// paths are wrapped by the same `check` and `verify`, so an exact solve is
/// held to the model the heuristic is held to.
inline LogicalResult solveSchedulingProblem(ChainingModuloProblem &problem,
                                            Operation *anchor, float cycleTime,
                                            unsigned minII,
                                            const SchedulerOptions &opts,
                                            const SpanObjective &span) {
  if (failed(problem.check()))
    return failure();
  if (usesExactScheduler(opts.kind)) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime, minII, span, opts)))
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
inline LogicalResult solveSchedulingProblem(
    ChainingSharedOperatorsProblem &problem, Operation *anchor, float cycleTime,
    const SchedulerOptions &opts, const SpanObjective &span) {
  if (failed(problem.check()))
    return failure();
  if (usesExactScheduler(opts.kind)) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime, span, opts)))
      return failure();
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Reject a kernel the backend cannot schedule at all: an unmodelled memory
/// effect, an unrealizable operator, an illegal channel or partition, and an
/// address cone that does not fit in \p cycleTime. Everything here is a
/// property of the input rather than of a scheduling decision, so it is settled
/// before a single problem is built.
///
/// \p cycleTime is the RESOLVED target period in ns (the caller applies the
/// default), so this and `runSDCScheduler` price against one number.
LogicalResult runPreScheduleVerification(ModuleOp module, StringRef top,
                                         float cycleTime);

/// Solve the schedule of every func reachable from \p top, recording it in
/// \p model. The IR is left in affine/scf form; nothing is materialized.
/// \p cycleTime is the resolved target period in ns, as above.
LogicalResult runSDCScheduler(ModuleOp module, StringRef top, float cycleTime,
                              const SchedulerOptions &opts,
                              ScheduleModel &model);

/// Reify \p model onto the IR as `dcp.*` regions. It runs immediately after the
/// scheduler over the same module, which is what makes the model's `Operation
/// *` keys valid; it also ADDS to the model, for the condition cones and
/// symbolic bounds it schedules itself.
void runPostScheduleConversion(ModuleOp module, ScheduleModel &model);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
