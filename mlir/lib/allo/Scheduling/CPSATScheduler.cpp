/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"

#ifdef ALLO_ENABLE_ORTOOLS
#include "circt/Scheduling/Utilities.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#endif

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

std::optional<SchedulerKind> mlir::allo::parseSchedulerKind(StringRef name) {
  return llvm::StringSwitch<std::optional<SchedulerKind>>(name)
      .Case("heuristic", SchedulerKind::Heuristic)
      .Case("exact", SchedulerKind::Exact)
      .Default(std::nullopt);
}

bool mlir::allo::hasExactScheduler() {
#ifdef ALLO_ENABLE_ORTOOLS
  return true;
#else
  return false;
#endif
}

#ifdef ALLO_ENABLE_ORTOOLS

using namespace circt::scheduling;
using namespace operations_research::sat;

namespace {

/// How much work one region's solve may consume, in OR-Tools' DETERMINISTIC
/// time units. A deterministic budget (rather than a wall-clock one) is what
/// keeps a compile reproducible: the same problem consumes the same budget on
/// any machine, so a schedule never depends on how loaded the host was. The
/// unit is roughly a second of a single core on current hardware.
constexpr double kDeterministicBudget = 10.0;

/// The solver configuration every Allo solve uses. Single-worker search with a
/// fixed seed and a deterministic limit, so two identical compiles emit
/// identical RTL. That reproducibility is what the content-keyed simulation
/// cache, a reproducible bug report and an A/B between two compiler revisions
/// all rest on; the only thing it costs is parallel search, which problems of
/// this size do not need.
SatParameters solverParameters() {
  SatParameters params;
  params.set_num_workers(1);
  params.set_random_seed(0);
  params.set_max_deterministic_time(kDeterministicBudget);
  return params;
}

/// The weight of a precedence edge out of \p src: the cycles a dependent must
/// wait after \p src issues before the value has arrived.
int64_t latencyOf(Problem &prob, Operation *src) {
  return *prob.getLatency(*prob.getLinkedOperatorType(src));
}

/// Whether \p op holds at least one unit whose count is capped, i.e. whether it
/// contends for anything. An unlimited link constrains nothing and no
/// reservation tracks it.
bool holdsLimitedUnit(SharedOperatorsProblem &prob, Operation *op) {
  auto linked = prob.getLinkedResourceTypes(op);
  return linked && llvm::any_of(*linked, [&](Problem::ResourceType rsrc) {
           return prob.getLimit(rsrc).value_or(0) > 0;
         });
}

/// The anchor's start time weighted above the most the tie-break total can
/// reach, so `WeightedSum` over start times orders the two objectives strictly:
/// minimize the anchor first, then pull everything else as early as the result
/// allows. See the acyclic scheduler for why the tie-break is load-bearing.
SmallVector<int64_t> objectiveWeights(size_t numOps, unsigned anchorIndex,
                                      unsigned horizon) {
  SmallVector<int64_t> weights(numOps, 1);
  weights[anchorIndex] +=
      static_cast<int64_t>(numOps) * (static_cast<int64_t>(horizon) + 1);
  return weights;
}

/// Report a solve that produced nothing usable and leave the heuristic's
/// schedule in place. `warn`, not `error` or `unsupported`: the compile is
/// still correct and still finishes, it just did not get the better schedule
/// it asked for.
void reportUnsolved(Problem &prob, const CpSolverResponse &response) {
  assert(response.status() != CpSolverStatus::INFEASIBLE &&
         response.status() != CpSolverStatus::MODEL_INVALID &&
         "the heuristic's schedule satisfies every constraint this encoding "
         "states, so the model is satisfiable by construction; an infeasible "
         "or invalid model means the encoding disagrees with the problem");
  warn(Stage::Sched, prob.getContainingOp())
      << "Exact scheduling gave up after "
      << llvm::format("%g", kDeterministicBudget)
      << " deterministic time units (solver status "
      << CpSolverStatus_Name(response.status())
      << "); keeping the heuristic schedule";
}

} // namespace

/// Refines the heuristic's acyclic schedule to the CP-SAT optimum. The
/// heuristic runs first as both a feasibility check and a warm-start hint:
/// its resource-free LP is the only thing that can fail, and no exact solver
/// repairs an infeasible LP either, so a failure here is fatal (unlike the
/// cyclic path, where placement is optional). The horizon is the whole region
/// laid out end to end, which is guaranteed to contain the heuristic's own
/// schedule; bounding it by the heuristic's depth instead would exclude a
/// schedule that reaches the anchor sooner at the cost of some other
/// operation finishing later. The objective minimizes the anchor's start
/// time, with the sum of all starts as a tie-break (weighted below the anchor
/// so the two never interact) so an off-path operation cannot drift and
/// lengthen the region for free; the anchor is also upper-bounded by the
/// heuristic's own start time, pruning like a branch-and-bound incumbent
/// without excluding the optimum, unlike the horizon which would be a hard
/// ceiling.
LogicalResult mlir::allo::scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                                        Operation *lastOp, float cycleTime) {
  // First-fit placement here cannot fail (a cycle with room always exists),
  // so a failure is the resource-free LP declaring infeasibility, which no
  // exact solver repairs either.
  if (failed(mlir::allo::scheduleSimplex(prob, lastOp, cycleTime)))
    return failure();

  // The same chain-breaking edges the heuristic used (the pre-pass is
  // schedule-independent), so any difference between the two schedules is
  // purely the cost of greedy resource placement.
  SmallVector<Problem::Dependence> chainBreaks;
  auto broke = computeChainBreakingDependences(prob, cycleTime, chainBreaks);
  assert(succeeded(broke) && "chain breaking is a pure function of the problem "
                             "and the cycle time, and the heuristic just ran "
                             "it on both successfully");
  (void)broke;

  const auto &ops = prob.getOperations();

  // Horizon: the whole region laid out end to end (each op after the
  // previous one's end, its occupancy window, plus a spare cycle), wide
  // enough that every precedence, chain break and reservation is satisfiable.
  unsigned horizon = 0;
  for (Operation *op : ops)
    horizon += latencyOf(prob, op) + prob.getResourceCycles(op) + 1;

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> startVars;
  // The same variables in problem order, for the objective. `ops` is a
  // SetVector, so this order is stable across runs and the model built from it
  // is byte-identical for a given problem.
  SmallVector<IntVar> orderedStarts;
  unsigned anchorIndex = 0;
  orderedStarts.reserve(ops.size());
  for (Operation *op : ops) {
    IntVar var = model.NewIntVar(operations_research::Domain(0, horizon));
    model.AddHint(var, *prob.getStartTime(op));
    startVars.try_emplace(op, var);
    if (op == lastOp)
      anchorIndex = orderedStarts.size();
    orderedStarts.push_back(var);
  }

  // Precedence, in the same two flavours `buildTableau` emits: a dependence
  // separates its endpoints by the source's latency, and a chain-breaking edge
  // by one cycle more.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op))
      model.AddLessOrEqual(startVars.at(dep.getSource()) +
                               latencyOf(prob, dep.getSource()),
                           startVars.at(dep.getDestination()));
  for (auto &dep : chainBreaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             latencyOf(prob, dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));

  // Resources: an op occupies one instance of every limited unit it links to
  // for its whole window, so a cumulative constraint per resource matches
  // `verifyOccupancy`. Multi-unit ops contribute the same window to each.
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    if (limit == 0)
      continue;
    CumulativeConstraint cumulative = model.AddCumulative(limit);
    for (Operation *op : ops) {
      auto linked = prob.getLinkedResourceTypes(op);
      if (!linked || !llvm::is_contained(*linked, rsrc))
        continue;
      cumulative.AddDemand(model.NewFixedSizeIntervalVar(
                               startVars.at(op), prob.getResourceCycles(op)),
                           1);
    }
  }

  // Same objective as the simplex: minimize the anchor's start time, then
  // the sum of all starts as a tie-break, weighted below the anchor so the
  // two never interact. Bounded above by the heuristic's own anchor start.
  unsigned anchorBound = *prob.getStartTime(lastOp);
  model.AddLessOrEqual(orderedStarts[anchorIndex],
                       static_cast<int64_t>(anchorBound));
  model.Minimize(LinearExpr::WeightedSum(
      orderedStarts, objectiveWeights(ops.size(), anchorIndex, horizon)));

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters());
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    reportUnsolved(prob, response);
    return success();
  }

  unsigned anchorStart = SolutionIntegerValue(response, startVars.at(lastOp));
  assert(anchorStart <= anchorBound && "the model bounds the anchor by the "
                                       "heuristic's own start time");
  if (anchorStart < anchorBound)
    info(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling improved the schedule: the last operation moved "
           "from cycle "
        << anchorBound << " to " << anchorStart;

  for (Operation *op : ops)
    prob.setStartTime(op, SolutionIntegerValue(response, startVars.at(op)));
  return computeStartTimesInCycle(prob);
}

namespace {

/// What one fixed-II solve settled. `Infeasible` is a PROOF that the initiation
/// interval admits no schedule, which is the answer the whole II search is
/// built to get; `Exhausted` is the solver giving up, which proves nothing.
enum class ModuloOutcome { Scheduled, Infeasible, Exhausted };

/// Solve \p prob at the FIXED initiation interval \p ii, writing the start
/// times into \p starts when one exists.
///
/// Fixing the II is what keeps the model linear. With the II as a variable,
/// `ii * distance` in a precedence edge becomes a product of two variables and
/// the congruence below acquires a variable modulus; with it constant, both are
/// plain linear constraints and the outer search pays for the difference in
/// solves rather than in solver capability.
///
/// \p hint says the start times \p prob already carries are feasible at this
/// II, i.e. that the greedy reached exactly this II, and hands them to the
/// solver as a starting solution. At any other II they are not a schedule and
/// hinting them would only buy a repair.
ModuloOutcome solveAtII(ChainingModuloProblem &prob, Operation *lastOp,
                        ArrayRef<Problem::Dependence> chainBreaks, unsigned ii,
                        unsigned horizon, bool hint,
                        DenseMap<Operation *, unsigned> &starts) {
  const auto &ops = prob.getOperations();

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> startVars;
  SmallVector<IntVar> orderedStarts;
  unsigned anchorIndex = 0;
  orderedStarts.reserve(ops.size());
  for (Operation *op : ops) {
    IntVar var = model.NewIntVar(operations_research::Domain(0, horizon));
    if (hint)
      model.AddHint(var, *prob.getStartTime(op));
    startVars.try_emplace(op, var);
    if (op == lastOp)
      anchorIndex = orderedStarts.size();
    orderedStarts.push_back(var);
  }

  // Precedence. An edge spanning `distance` iterations is relaxed by one II per
  // iteration it spans, which is the cyclic constraint row `buildTableau`
  // emits; a chain-breaking edge is intra-iteration and costs one cycle more.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      int64_t separation =
          latencyOf(prob, src) -
          static_cast<int64_t>(ii) * prob.getDistance(dep).value_or(0);
      model.AddLessOrEqual(startVars.at(src) + separation,
                           startVars.at(dep.getDestination()));
    }
  for (auto &dep : chainBreaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             latencyOf(prob, dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));

  // One-hot congruence class per contending op. `t = ii*lap + sum(p*slot[p])`
  // over a one-hot slot defines class and modulo at once, with no
  // reification: slot[p] IS membership in class p, which the sums below need.
  DenseMap<Operation *, SmallVector<BoolVar>> slotsOf;
  SmallVector<int64_t> classes(ii);
  for (unsigned p = 0; p < ii; ++p)
    classes[p] = p;
  for (Operation *op : ops) {
    if (!holdsLimitedUnit(prob, op))
      continue;
    SmallVector<BoolVar> slots;
    slots.reserve(ii);
    for (unsigned p = 0; p < ii; ++p)
      slots.push_back(model.NewBoolVar());
    model.AddExactlyOne(slots);
    IntVar lap = model.NewIntVar(operations_research::Domain(0, horizon / ii));
    model.AddEquality(startVars.at(op),
                      lap * static_cast<int64_t>(ii) +
                          LinearExpr::WeightedSum(slots, classes));
    slotsOf.try_emplace(op, std::move(slots));
  }

  // Modulo reservation: an op holding a unit for `occ` cycles wraps the II
  // table floor(occ/ii) times (every class) plus `occ % ii` more from its
  // own slot, exactly what `MRT::enter` counts, so the two models cross-check.
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    if (limit == 0)
      continue;
    for (unsigned slot = 0; slot < ii; ++slot) {
      LinearExpr used;
      for (Operation *op : ops) {
        auto linked = prob.getLinkedResourceTypes(op);
        if (!linked || !llvm::is_contained(*linked, rsrc))
          continue;
        unsigned occ = prob.getResourceCycles(op);
        used += static_cast<int64_t>(occ / ii);
        const SmallVector<BoolVar> &slots = slotsOf.at(op);
        for (unsigned k = 0, partial = occ % ii; k < partial; ++k)
          used += slots[(slot + ii - k) % ii];
      }
      model.AddLessOrEqual(used, static_cast<int64_t>(limit));
    }
  }

  model.Minimize(LinearExpr::WeightedSum(
      orderedStarts, objectiveWeights(ops.size(), anchorIndex, horizon)));

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters());
  if (response.status() == CpSolverStatus::INFEASIBLE)
    return ModuloOutcome::Infeasible;
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    assert(response.status() != CpSolverStatus::MODEL_INVALID &&
           "the encoding built an ill-formed model");
    return ModuloOutcome::Exhausted;
  }
  for (Operation *op : ops)
    starts[op] = SolutionIntegerValue(response, startVars.at(op));
  return ModuloOutcome::Scheduled;
}

} // namespace

/// Refines the heuristic's modulo (cyclic) schedule to the CP-SAT optimum by
/// searching fixed II values from the heuristic's own II lower bound upward.
/// The only thing this path needs from the heuristic is that lower bound,
/// settled by the resource-free LP before any placement runs; without it the
/// search would start at II=1 and spend its budget refuting intervals no
/// bound ever allowed. The heuristic's own placement is optional context
/// (`SimplexWarmStart`), not a precondition: what must still succeed is the
/// LP underneath, which is exact regardless of whether placement did.
LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII) {
  // Needs only the heuristic's II lower bound from the LP; its placement is
  // optional (see SimplexWarmStart) and not required to succeed.
  SimplexWarmStart warm;
  if (failed(
          mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, minII, &warm)))
    return failure();

  unsigned greedyII = warm.placed ? *prob.getInitiationInterval() : 0;
  assert((!warm.placed || greedyII >= warm.lowerBoundII) &&
         "placement only ever grows the II");

  SmallVector<Problem::Dependence> chainBreaks;
  auto broke = computeChainBreakingDependences(prob, cycleTime, chainBreaks);
  assert(succeeded(broke) && "chain breaking is a pure function of the problem "
                             "and the cycle time, and the heuristic just ran "
                             "it successfully");
  (void)broke;

  // Window: region laid out end to end (satisfies precedence + chain
  // breaks) plus one II per contending op, widened to the heuristic's own
  // reach. Must be provably sufficient, since INFEASIBLE here counts as proof.
  const auto &ops = prob.getOperations();
  int64_t sequential = 0;
  int64_t greedyReach = 0;
  unsigned contending = 0;
  for (Operation *op : ops) {
    sequential += latencyOf(prob, op) + 1;
    if (holdsLimitedUnit(prob, op))
      ++contending;
    if (warm.placed)
      greedyReach = std::max(greedyReach, int64_t(*prob.getStartTime(op)));
  }
  int64_t window = std::max(sequential, greedyReach);

  // Search bound: with a greedy incumbent, scan through its own II (the II
  // alone isn't sufficient; placement there must still be solved). With no
  // incumbent, bound by total occupancy, where every op gets its own slot.
  unsigned totalOccupancy = 0;
  for (Operation *op : ops)
    if (holdsLimitedUnit(prob, op))
      totalOccupancy += prob.getResourceCycles(op);
  unsigned upperII =
      warm.placed ? greedyII : std::max(warm.lowerBoundII, totalOccupancy);

  for (unsigned ii = warm.lowerBoundII; ii <= upperII; ++ii) {
    DenseMap<Operation *, unsigned> starts;
    switch (solveAtII(prob, lastOp, chainBreaks, ii, window + ii * contending,
                      /*hint=*/warm.placed && ii == greedyII, starts)) {
    case ModuloOutcome::Scheduled: {
      auto d = info(Stage::Sched, prob.getContainingOp());
      d << "Exact scheduling placed the region at II=" << ii
        << ", the proven minimum";
      if (!warm.placed)
        d << ": the greedy placement could not place it at all";
      else if (ii < greedyII)
        d << ", down from the heuristic's II=" << greedyII
          << ": the gap was greedy resource placement";
      else
        d << ", the II the heuristic also reached, with the placement inside "
             "it solved exactly";
      prob.setInitiationInterval(ii);
      for (Operation *op : ops)
        prob.setStartTime(op, starts.at(op));
      return computeStartTimesInCycle(prob);
    }
    case ModuloOutcome::Infeasible:
      assert((!warm.placed || ii < greedyII) &&
             "the greedy's own schedule satisfies every constraint this "
             "encoding states at the II it achieved, so INFEASIBLE there means "
             "the encoding and the reservation model disagree");
      continue;
    case ModuloOutcome::Exhausted:
      if (!warm.placed) {
        unsupported(Stage::Sched, prob.getContainingOp())
            << "Neither scheduler could place this region: the greedy modulo "
               "placement gave up, and the exact one ran out of budget at II="
            << ii << " without deciding it";
        return failure();
      }
      warn(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling ran out of budget at II=" << ii
          << " without deciding it; falling back to the heuristic's schedule "
             "at II="
          << greedyII << ", which is therefore not known to be minimal";
      return success();
    }
  }

  // Only reachable without an incumbent: with one, the II it achieved is in the
  // search range and feasible there by construction.
  assert(!warm.placed && "the incumbent II is inside the search range");
  unsupported(Stage::Sched, prob.getContainingOp())
      << "Neither scheduler could place this region: the greedy modulo "
         "placement gave up, and every initiation interval from "
      << warm.lowerBoundII << " to " << upperII << " is infeasible";
  return failure();
}

#else // !ALLO_ENABLE_ORTOOLS

namespace {
/// Unreachable through `runSDCScheduler`, which rejects `scheduler="exact"` on
/// an OR-Tools-free build before any region is solved. Kept so the two entry
/// points exist in both configurations: conditional compilation switches the
/// implementation, never the interface.
LogicalResult noExactScheduler(Operation *containingOp, StringRef which) {
  unsupported(Stage::Sched, containingOp)
      << "This build has no exact scheduler: it was configured without "
         "OR-Tools. Rebuild with -DALLO_ENABLE_ORTOOLS=ON, or schedule with "
         "scheduler=\"heuristic\" (the default). Requested for the "
      << which << " problem";
  return failure();
}
} // namespace

LogicalResult mlir::allo::scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                                        Operation *lastOp, float cycleTime) {
  return noExactScheduler(prob.getContainingOp(), "acyclic");
}

LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII) {
  return noExactScheduler(prob.getContainingOp(), "cyclic");
}

#endif // ALLO_ENABLE_ORTOOLS
