/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h" // the device the area is priced at
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"

#ifdef ALLO_ENABLE_ORTOOLS
#include "circt/Scheduling/Utilities.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

#include <cmath>
#endif

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

std::optional<SchedulerKind> mlir::allo::parseSchedulerKind(StringRef name) {
  return llvm::StringSwitch<std::optional<SchedulerKind>>(name)
      .Case("heuristic", SchedulerKind::Heuristic)
      .Case("exact", SchedulerKind::Exact)
      .Case("exact-chaining", SchedulerKind::ExactChaining)
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

/// Solver configuration for every Allo solve: single-worker, fixed seed, and a
/// deterministic (not wall-clock) time limit of \p budget, so two identical
/// compiles emit identical RTL.
SatParameters solverParameters(double budget) {
  SatParameters params;
  params.set_num_workers(1);
  params.set_random_seed(0);
  params.set_max_deterministic_time(budget);
  return params;
}

/// How the model states the clock period: either the chain-breaking edges the
/// pre-pass computed (each costs a cycle on top of plain precedence), or the
/// period itself, which `addChaining` encodes as sub-cycle start times.
struct Chaining {
  SmallVector<Problem::Dependence> breaks;
  std::optional<float> period;
};

/// Build the chaining constraint: the pre-pass's break edges, or \p cycleTime
/// itself when \p exactChaining.
template <class ProblemT>
Chaining chainingFor(ProblemT &prob, float cycleTime, bool exactChaining) {
  Chaining chaining;
  if (exactChaining) {
    chaining.period = cycleTime;
    return chaining;
  }
  auto broke = mlir::allo::computeChainBreaks(prob, cycleTime, chaining.breaks);
  assert(succeeded(broke) && "chain breaking is a pure function of the problem "
                             "and the cycle time, and the heuristic just ran "
                             "it successfully");
  (void)broke;
  return chaining;
}

/// Sub-cycle time in picoseconds, rounded to the nearest: CP-SAT is integer.
/// Device delays are given to a hundredth of a nanosecond, so a picosecond has
/// enough resolution and the rounding only absorbs float representation error.
/// Round-to-nearest and not up, because a chain that fills the period exactly
/// is common and rounding up would reject a schedule the clock accepts.
constexpr double kPicosPerNs = 1000.0;
int64_t picos(double ns) { return std::llround(ns * kPicosPerNs); }

/// States the period as a model constraint rather than as pre-pass edges: one
/// sub-cycle start time `z` per operation, in picoseconds from the start of its
/// cycle, matching what `computeStartTimesInCycle` computes afterwards.
/// `z(v) <= P - inDelay(v)`, and where a def-use producer u ends in the cycle v
/// starts, `z(v) >= (lat(u) == 0 ? z(u) : 0) + outDelay(u)`.
///
/// Precedence already forces `t_v - t_u >= lat(u)`, so gating on the `<=` half
/// alone (via `sameCycle`) detects "ends in the same cycle".
///
/// Only def-use edges carry a combinational path; an auxiliary edge (memory
/// order, stream order, loop-carried recurrence) always passes through a port
/// or register, so both forms of chaining see the same chains.
template <class ProblemT>
void addSubCycleTimes(CpModelBuilder &model, ProblemT &prob,
                      DenseMap<Operation *, IntVar> &startVars,
                      float cycleTime) {
  int64_t period = picos(cycleTime);
  DenseMap<Operation *, IntVar> inCycle;
  for (Operation *op : prob.getOperations()) {
    int64_t in = picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
    assert(in <= period &&
           "an operator whose own delay exceeds the period is rejected by the "
           "chain-breaking pre-pass, which the heuristic ran before this");
    inCycle.try_emplace(
        op, model.NewIntVar(operations_research::Domain(0, period - in)));
  }

  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      if (!dep.isDefUse())
        continue;
      if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>)
        assert(prob.getDistance(dep).value_or(0) == 0 &&
               "a distance rides an AUXILIARY edge here (`ProblemBuilder` "
               "inserts every carried dependence as one), so a def-use edge is "
               "always intra-iteration and its endpoints share a cycle");
      Operation *src = dep.getSource();
      Problem::OperatorType srcOpr = *prob.getLinkedOperatorType(src);
      int64_t lat = *prob.getLatency(srcOpr);
      int64_t out = picos(*prob.getOutgoingDelay(srcOpr));
      LinearExpr separation = startVars.at(op) - startVars.at(src);
      BoolVar sameCycle = model.NewBoolVar();
      model.AddLessOrEqual(separation, lat).OnlyEnforceIf(sameCycle);
      model.AddGreaterOrEqual(separation, lat + 1)
          .OnlyEnforceIf(sameCycle.Not());
      // A multi-cycle producer contributes only its outgoing delay: its last
      // register stage is what the cycle starts from.
      LinearExpr ready = lat == 0 ? inCycle.at(src) + out : LinearExpr(out);
      model.AddGreaterOrEqual(inCycle.at(op), ready).OnlyEnforceIf(sameCycle);
    }
}

/// State \p chaining on the model: a chain-breaking edge widens an existing
/// precedence by one cycle; a period uses the sub-cycle encoding above.
template <class ProblemT>
void addChaining(CpModelBuilder &model, ProblemT &prob,
                 DenseMap<Operation *, IntVar> &startVars,
                 const Chaining &chaining) {
  for (const Problem::Dependence &dep : chaining.breaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             prob.latencyOf(dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));
  if (chaining.period)
    addSubCycleTimes(model, prob, startVars, *chaining.period);
}

/// Every operation's inputs settle within the period.
[[maybe_unused]] bool chainsFitCycleTime(ChainingProblem &prob,
                                         float cycleTime) {
  // Slop of one picosecond, the model's own resolution, to absorb float error.
  constexpr float kSlop = 1e-3f;
  for (Operation *op : prob.getOperations()) {
    float in = *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
    if (*prob.getStartTimeInCycle(op) + in > cycleTime + kSlop)
      return false;
  }
  return true;
}

/// Derive the solved schedule's sub-cycle start times and check the period.
/// `ChainingProblem` does not carry the period itself, so this is the only
/// place that verifies chains fit it.
LogicalResult finishSchedule(ChainingProblem &prob, float cycleTime) {
  if (failed(computeStartTimesInCycle(prob)))
    return failure();
  assert(chainsFitCycleTime(prob, cycleTime) &&
         "a combinational chain crosses more than one clock period");
  return success();
}

/// The region's drain as a variable: the max of `start(op) + offset` over the
/// same terms `drainOf` maxes over, stated as lower bounds only, which is tight
/// since the objective minimizes it.
///
/// \p bound caps it at an incumbent's, so the solver only searches schedules
/// that would beat it; an INFEASIBLE result then means "nothing beats the
/// incumbent" rather than "the interval is impossible".
IntVar drainVariable(CpModelBuilder &model,
                     DenseMap<Operation *, IntVar> &startVars,
                     ArrayRef<DrainTerm> terms, int64_t horizon,
                     std::optional<int64_t> bound) {
  assert((!bound || *bound >= 0) && "incumbent cut before building the model");
  IntVar drain = model.NewIntVar(operations_research::Domain(
      0, bound ? std::min(*bound, horizon) : horizon));
  for (const DrainTerm &term : terms)
    model.AddLessOrEqual(startVars.at(term.op) + term.offset, drain);
  return drain;
}

/// One allocatable resource in the model: the unit count to decide, and what
/// building that many of it costs.
struct AllocationVar {
  Problem::ResourceType rsrc;
  IntVar units;
  /// Priced off `units` through the resource's own table, so the plateaus in
  /// what a fold saves and what it grows are in the model exactly.
  IntVar price;
  int64_t maxPrice = 0;
};

/// Declare `N_r` for every allocatable resource: how many copies of one
/// operator this region builds, in `[1, ceiling]`. The caller states the
/// capacity constraint against it.
///
/// \p hint says the heuristic's start times are being hinted too, and then the
/// count hinted is the TIGHTEST one those start times admit rather than the
/// trivial one. Both are consistent with the hinted schedule and the tight one
/// is what the greedy binder would have built from it, so the solver starts
/// where the area-agnostic policy ends instead of having to search its way
/// there; on a region whose budget runs out, the hint is what ships.
SmallVector<AllocationVar> allocationVars(CpModelBuilder &model,
                                          OccupancyProblem &prob, unsigned ii,
                                          bool hint) {
  SmallVector<AllocationVar> allocs;
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    auto unit = prob.getAllocatable(rsrc);
    if (!unit)
      continue;
    assert(unit->ceiling > 0 && "an allocatable resource with no operation");
    IntVar n = model.NewIntVar(operations_research::Domain(1, unit->ceiling));
    if (hint)
      model.AddHint(n, prob.demandFor(rsrc, ii));
    std::vector<int64_t> table(unit->price.begin(), unit->price.end());
    int64_t hi = *llvm::max_element(unit->price);
    IntVar price = model.NewIntVar(
        operations_research::Domain(*llvm::min_element(table), hi));
    model.AddElement(n, table, price);
    allocs.push_back({rsrc, n, price, hi});
  }
  return allocs;
}

/// Add \p price at \p size to a weighted sum, for a price tabulated at every
/// value the size can take. A piecewise-linear price is its FIRST slope on the
/// size, plus at every change of slope that change charged on how far the size
/// runs past the point it changes at: `max(size - b, 0)`. Every variable this
/// adds is determined by the size through a propagator, where selecting one of
/// the segments instead would be a disjunction per structure for the search to
/// branch on, and a device has a change of slope per shift register.
///
/// The sum is an identity, so what it contributes is bounded by the price's own
/// maximum however the slopes cancel.
void addPiecewiseCost(CpModelBuilder &model, IntVar size,
                      ArrayRef<int64_t> price, SmallVectorImpl<IntVar> &vars,
                      SmallVectorImpl<int64_t> &weights) {
  auto hi = static_cast<int64_t>(price.size()) - 1;
  if (hi < 1)
    return;
  int64_t slope = price[1] - price[0];
  vars.push_back(size);
  weights.push_back(slope);
  for (int64_t d = 2; d <= hi; ++d) {
    int64_t next = price[d] - price[d - 1];
    if (next == slope)
      continue;
    IntVar over = model.NewIntVar(operations_research::Domain(0, hi - d + 1));
    model.AddMaxEquality(over, {LinearExpr(size) - (d - 1), LinearExpr(0)});
    vars.push_back(over);
    weights.push_back(next - slope);
    slope = next;
  }
}

/// Minimize \p primary, with the region's area as the tie-break below it,
/// weighted so the two never interact: `primary` is settled first, and the
/// tie-break decides only among schedules that reach it.
///
/// The tie-break is the region's AREA, every term of it in what the device
/// spends: the delay chain each value carried across slack costs
/// (`RegisterTerm`), one stage of a one-bit activation pulse chain per cycle of
/// every start offset, and the table above per allocatable operator. A chain is
/// NOT `width * depth` flip-flops: past a measured depth the synthesizer stops
/// building them and extracts a shift register instead, so the cost stops
/// rising with the width and the term keeps only the gradient the part keeps.
void minimizeCost(CpModelBuilder &model, IntVar primary,
                  ArrayRef<IntVar> starts, const SpanObjective &span,
                  DenseMap<Operation *, IntVar> &startVars,
                  ArrayRef<AllocationVar> allocs, int64_t ii, int64_t horizon) {
  int64_t pulse = span.device.pulsePrice();
  SmallVector<IntVar> vars(starts.begin(), starts.end());
  SmallVector<int64_t> weights(starts.size(), pulse);
  // Bounds how far the tie-break can reach, so `primary`'s weight below
  // strictly dominates it. Loose on purpose: a price is already an area and
  // needs no horizon factor, but taking it out makes the tie-break a
  // comparable share of the objective and costs four times the search for a
  // third of a percent of area.
  int64_t area = static_cast<int64_t>(starts.size()) * pulse;
  // One chain price table per width: a region carries many values of the same
  // type, and tabulating the device's cost is the expensive half.
  DenseMap<int64_t, SmallVector<int64_t>> chainPrices;
  for (const RegisterTerm &term : span.regs) {
    auto [entry, isNew] = chainPrices.try_emplace(term.width);
    if (isNew)
      for (int64_t d = 0; d <= horizon; ++d)
        entry->second.push_back(span.device.chainPrice(d, term.width));
    ArrayRef<int64_t> table = entry->second;
    IntVar depth = model.NewIntVar(operations_research::Domain(0, horizon));
    IntVar def = startVars.at(term.def);
    for (auto [reader, distance] : term.reads)
      model.AddLessOrEqual(startVars.at(reader) + distance * ii - term.latency,
                           def + depth);
    addPiecewiseCost(model, depth, table, vars, weights);
    area += *llvm::max_element(table);
  }
  for (const AllocationVar &alloc : allocs) {
    vars.push_back(alloc.price);
    weights.push_back(1);
    area += alloc.maxPrice;
  }
  vars.push_back(primary);
  // A device that declares no area model prices every term at nothing, which
  // is the honest reading of saying nothing; the floor keeps `primary` from
  // being weighted at nothing along with it.
  weights.push_back(std::max<int64_t>(area, 1) * (horizon + 1));
  model.Minimize(LinearExpr::WeightedSum(vars, weights));
}

/// The unit counts one solve decided. Held apart from the problem because the
/// cyclic search runs many solves and only the adopted one's counts stand.
using Allocated = SmallVector<std::pair<Problem::ResourceType, unsigned>>;

Allocated readAllocation(const CpSolverResponse &response,
                         ArrayRef<AllocationVar> allocs) {
  Allocated decided;
  for (const AllocationVar &alloc : allocs)
    decided.push_back({alloc.rsrc, static_cast<unsigned>(SolutionIntegerValue(
                                       response, alloc.units))});
  return decided;
}

/// What \p decided costs the device: every resource, at the price of the count
/// it settled on.
int64_t areaOf(OccupancyProblem &prob, const Allocated &decided) {
  int64_t area = 0;
  for (auto [rsrc, units] : decided)
    area += prob.getAllocatable(rsrc)->price[units];
  return area;
}

/// Write \p decided onto the problem and derive which instance each operation
/// runs on. \p ii is 0 for a straight-line region.
void applyAllocation(OccupancyProblem &prob, const Allocated &decided,
                     unsigned ii) {
  if (decided.empty())
    return;
  int64_t built = 0, ops = 0;
  for (auto [rsrc, units] : decided) {
    prob.setAllocation(rsrc, units);
    built += units;
    ops += prob.getAllocatable(rsrc)->ceiling;
  }
  // Counts alone are not buildable; this derives the per-operation instance.
  prob.assignUnits(ii);
  info(Stage::Sched, prob.getContainingOp())
      << "Allocated: " << ops << " operations onto " << built
      << " instances of " << decided.size() << " shared operator types";
}

/// Fall back to the tightest allocation the schedule already on the problem
/// admits, for a solve that decided none. Without it a region whose budget ran
/// out keeps the TRIVIAL allocation and the emitter builds one instance per
/// operation, which is strictly worse than what the same schedule supports and
/// worse than what an area-agnostic greedy binder would fold it to.
void applyDemandAllocation(OccupancyProblem &prob, unsigned ii) {
  Allocated decided;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (prob.getAllocatable(rsrc))
      decided.push_back({rsrc, prob.demandFor(rsrc, ii)});
  applyAllocation(prob, decided, ii);
}

/// Report a solve that produced nothing usable and leave the heuristic's
/// schedule in place. A `warn`: the compile is still correct, it just did not
/// get a better schedule.
void reportUnsolved(Problem &prob, const CpSolverResponse &response,
                    double budget) {
  assert(response.status() != CpSolverStatus::INFEASIBLE &&
         response.status() != CpSolverStatus::MODEL_INVALID &&
         "the heuristic's schedule satisfies this encoding, so the model is "
         "satisfiable by construction");
  warn(Stage::Sched, prob.getContainingOp())
      << "Exact scheduling gave up after " << llvm::format("%g", budget)
      << " deterministic time units (solver status "
      << CpSolverStatus_Name(response.status())
      << "); keeping the heuristic schedule";
}

} // namespace

//===----------------------------------------------------------------------===//
// The acyclic solve.
//===----------------------------------------------------------------------===//

/// Refines the heuristic's acyclic schedule to the CP-SAT optimum.
///
/// The heuristic runs first as a feasibility check and a warm-start hint: its
/// resource-free LP is the only thing that can fail, so a failure here is
/// fatal, unlike the cyclic path where placement is optional.
///
/// A straight-line region runs once, so its whole cost is its drain, which the
/// objective minimizes, upper-bounded by the heuristic's own drain so the
/// search prunes like a branch and bound.
LogicalResult mlir::allo::scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  // First-fit placement here cannot fail (a cycle with room always exists),
  // so a failure is the resource-free LP declaring infeasibility, which no
  // exact solver repairs either.
  if (failed(mlir::allo::scheduleSimplex(prob, lastOp, cycleTime)))
    return failure();

  // The pre-pass is schedule-independent, so taking its edges hands CP-SAT the
  // chain breaks the heuristic just used.
  Chaining chaining =
      chainingFor(prob, cycleTime, opts.kind == SchedulerKind::ExactChaining);

  const auto &ops = prob.getOperations();

  // Horizon: the whole region laid out end to end (each op after the previous
  // one's end, its occupancy window, plus a spare cycle), wide enough that
  // every precedence, chain break and reservation is satisfiable.
  unsigned horizon = 0;
  for (Operation *op : ops)
    horizon += prob.latencyOf(op) + prob.getResourceCycles(op) + 1;

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> startVars;
  // The same variables in problem order, for the objective; `ops` is a
  // SetVector so this order is stable across runs.
  SmallVector<IntVar> orderedStarts;
  orderedStarts.reserve(ops.size());
  for (Operation *op : ops) {
    IntVar var = model.NewIntVar(operations_research::Domain(0, horizon));
    model.AddHint(var, *prob.getStartTime(op));
    startVars.try_emplace(op, var);
    orderedStarts.push_back(var);
  }

  // Precedence, as `buildTableau` emits it: a dependence separates its
  // endpoints by the source's latency.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op))
      model.AddLessOrEqual(startVars.at(dep.getSource()) +
                               prob.latencyOf(dep.getSource()),
                           startVars.at(dep.getDestination()));
  addChaining(model, prob, startVars, chaining);

  // An op occupies one instance of every unit it links to for its whole window,
  // so a cumulative constraint per resource matches `verifyOccupancy`. A
  // multi-unit op contributes the same window to each.
  auto cumulativeOn = [&](Problem::ResourceType rsrc, LinearExpr capacity) {
    CumulativeConstraint cumulative = model.AddCumulative(std::move(capacity));
    for (Operation *op : ops)
      if (prob.usesResource(op, rsrc))
        cumulative.AddDemand(model.NewFixedSizeIntervalVar(
                                 startVars.at(op), prob.getResourceCycles(op)),
                             1);
  };
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (unsigned limit = prob.getLimit(rsrc).value_or(0))
      cumulativeOn(rsrc, limit);

  // An allocatable operator takes the same shape, with the count being decided
  // as the capacity. Occupancy windows on a line form an interval graph, so a
  // capacity is an assignment: `N` units suffice when no cycle needs more.
  SmallVector<AllocationVar> allocs =
      allocationVars(model, prob, /*ii=*/0, /*hint=*/true);
  for (const AllocationVar &alloc : allocs)
    cumulativeOn(alloc.rsrc, alloc.units);

  // What the region is charged, bounded by what the heuristic already reached.
  int64_t heuristicDrain = span.drainOf(prob);
  IntVar drain =
      drainVariable(model, startVars, span.drain, horizon, heuristicDrain);
  minimizeCost(model, drain, orderedStarts, span, startVars, allocs, /*ii=*/0,
               horizon);

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts.budget));
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    reportUnsolved(prob, response, opts.budget);
    applyDemandAllocation(prob, /*ii=*/0);
    return success();
  }

  // FEASIBLE and not OPTIMAL means the budget stopped short of proving it, so
  // what ships is an incumbent.
  if (response.status() != CpSolverStatus::OPTIMAL)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget before proving this region's "
           "placement optimal; it shipped the best schedule it had found, "
           "which is no worse than the heuristic's and is not known to be best";

  int64_t solvedDrain = SolutionIntegerValue(response, drain);
  assert(solvedDrain <= heuristicDrain &&
         "the model bounds the drain by the heuristic's own");
  if (solvedDrain < heuristicDrain)
    info(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling shortened the region: its deepest output now "
           "commits at cycle "
        << solvedDrain << " instead of " << heuristicDrain;

  for (Operation *op : ops)
    prob.setStartTime(op, SolutionIntegerValue(response, startVars.at(op)));
  applyAllocation(prob, readAllocation(response, allocs), /*ii=*/0);
  return finishSchedule(prob, cycleTime);
}

//===----------------------------------------------------------------------===//
// The cyclic solve: a branch and bound over initiation intervals.
//===----------------------------------------------------------------------===//

namespace {

/// What one fixed-II solve settled. `Infeasible` is a proof that the
/// initiation interval admits no schedule; `Exhausted` is the solver giving
/// up, which proves nothing.
enum class ModuloOutcome { Scheduled, Infeasible, Exhausted };

/// A lower bound on the region's drain at ANY initiation interval: the longest
/// chain of intra-iteration (distance-0) edges reaching an output. An edge
/// spanning iterations is relaxed by one II per iteration it spans, so only the
/// distance-0 subgraph bounds a start time regardless of interval width, and
/// resources only push starts later. This is what keeps the branch and bound's
/// cut tight once the drain dwarfs the trip.
///
/// Only \p chaining's break edges lengthen a path here; where the period is
/// stated in the model instead there are no break edges, and the bound is
/// simply looser, still sound.
int64_t drainFloor(ChainingModuloProblem &prob, const Chaining &chaining,
                   ArrayRef<DrainTerm> terms) {
  // Incoming edges by destination, weighted as the model weights them.
  DenseMap<Operation *, SmallVector<std::pair<Operation *, int64_t>>> incoming;
  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        incoming[op].push_back(
            {dep.getSource(), prob.latencyOf(dep.getSource())});
  for (auto &dep : chaining.breaks)
    incoming[dep.getDestination()].push_back(
        {dep.getSource(), prob.latencyOf(dep.getSource()) + 1});

  // Longest path, memoized; the seeded zero keeps a cycle from recursing
  // forever if the distance-0 subgraph were ever not acyclic.
  DenseMap<Operation *, int64_t> asap;
  auto reach = [&](auto &self, Operation *op) -> int64_t {
    auto seen = asap.find(op);
    if (seen != asap.end())
      return seen->second;
    asap[op] = 0;
    int64_t longest = 0;
    auto edges = incoming.find(op);
    if (edges != incoming.end())
      for (auto [src, weight] : edges->second)
        longest = std::max(longest, self(self, src) + weight);
    asap[op] = longest;
    return longest;
  };

  int64_t bound = 0;
  for (const DrainTerm &term : terms)
    bound = std::max(bound, reach(reach, term.op) + term.offset);
  return bound;
}

/// Solve \p prob at the FIXED initiation interval \p ii, writing the start
/// times into \p starts when one exists. Fixing the II keeps the model linear:
/// `ii * distance` in a precedence edge and the modulo congruence below would
/// otherwise need a variable modulus.
///
/// \p hint is only valid when the greedy placement itself reached this II; at
/// any other II its start times are not a schedule.
///
/// \p proven is OPTIMAL against FEASIBLE, which the II search cannot otherwise
/// tell apart, and an unproven placement's drain is still what the region's
/// span gets charged.
///
/// \p drainBound is the incumbent's, so INFEASIBLE here means nothing beats the
/// incumbent at this II rather than a proof the interval is impossible.
ModuloOutcome solveAtII(ChainingModuloProblem &prob, Operation *lastOp,
                        const Chaining &chaining, const SpanObjective &span,
                        const SchedulerOptions &opts,
                        std::optional<int64_t> drainBound, unsigned ii,
                        unsigned horizon, bool hint,
                        DenseMap<Operation *, unsigned> &starts,
                        Allocated &decided, bool &proven, int64_t &drain) {
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
  // emits. A chain-breaking edge is intra-iteration, so `addChaining` states it
  // without the II term.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      int64_t separation =
          prob.latencyOf(src) -
          static_cast<int64_t>(ii) * prob.getDistance(dep).value_or(0);
      model.AddLessOrEqual(startVars.at(src) + separation,
                           startVars.at(dep.getDestination()));
    }
  addChaining(model, prob, startVars, chaining);

  // One-hot congruence class per contending op. `t = ii*lap + sum(p*slot[p])`
  // defines class and modulo at once with no reification: slot[p] IS membership
  // in class p, which the sums below need.
  DenseMap<Operation *, SmallVector<BoolVar>> slotsOf;
  SmallVector<int64_t> classes(ii);
  for (unsigned p = 0; p < ii; ++p)
    classes[p] = p;
  for (Operation *op : ops) {
    if (!prob.contendsForUnit(op))
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
  auto usesIn = [&](Problem::ResourceType rsrc, unsigned slot) {
    LinearExpr used;
    for (Operation *op : ops) {
      if (!prob.usesResource(op, rsrc))
        continue;
      unsigned occ = prob.getResourceCycles(op);
      used += static_cast<int64_t>(occ / ii);
      const SmallVector<BoolVar> &slots = slotsOf.at(op);
      for (unsigned k = 0, partial = occ % ii; k < partial; ++k)
        used += slots[(slot + ii - k) % ii];
    }
    return used;
  };
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    if (limit == 0)
      continue;
    for (unsigned slot = 0; slot < ii; ++slot)
      model.AddLessOrEqual(usesIn(rsrc, slot), static_cast<int64_t>(limit));
  }

  // The same sum against the count being decided. Allocatable operators occupy
  // one cycle here, so an op sits in one class and a per-class count is
  // realizable as an assignment. `N_r >= ceil(total/ii)` is implied, cut here.
  SmallVector<AllocationVar> allocs = allocationVars(model, prob, ii, hint);
  for (const AllocationVar &alloc : allocs) {
    int64_t total = 0;
    for (Operation *op : ops)
      if (prob.usesResource(op, alloc.rsrc))
        total += prob.getResourceCycles(op);
    model.AddGreaterOrEqual(alloc.units, (total + ii - 1) / ii);
    for (unsigned slot = 0; slot < ii; ++slot)
      model.AddLessOrEqual(usesIn(alloc.rsrc, slot), alloc.units);
  }

  // `(trip - 1) * ii` is constant at a fixed II, so minimizing the span here is
  // minimizing the drain; the outer search carries the II term. With no span to
  // compose, the anchor's start time takes the primary slot instead.
  std::optional<IntVar> drainVar;
  if (span.trip)
    drainVar = drainVariable(model, startVars, span.drain, horizon, drainBound);
  minimizeCost(model, drainVar.value_or(orderedStarts[anchorIndex]),
               orderedStarts, span, startVars, allocs, ii, horizon);

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts.budget));
  if (response.status() == CpSolverStatus::INFEASIBLE)
    return ModuloOutcome::Infeasible;
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    assert(response.status() != CpSolverStatus::MODEL_INVALID &&
           "the encoding built an ill-formed model");
    return ModuloOutcome::Exhausted;
  }
  proven = response.status() == CpSolverStatus::OPTIMAL;
  for (Operation *op : ops)
    starts[op] = SolutionIntegerValue(response, startVars.at(op));
  decided = readAllocation(response, allocs);
  drain = drainVar ? SolutionIntegerValue(response, *drainVar) : 0;
  return ModuloOutcome::Scheduled;
}

} // namespace

/// Refines the heuristic's modulo (cyclic) schedule by searching fixed II
/// values from the heuristic's own II lower bound upward, as a branch and bound
/// on the region's span. Only that lower bound (from the resource-free LP) is
/// needed; the heuristic's placement is optional context (`SimplexWarmStart`).
///
/// The search cannot return at the first feasible II: what the region is
/// charged is `(trip - 1) * ii + drain`, and a larger II can still win by
/// admitting a shorter drain. So it keeps the best span seen and cuts at the
/// first interval whose II term alone already reaches it.
LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  SimplexWarmStart warm;
  if (failed(
          mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, minII, &warm)))
    return failure();

  unsigned greedyII = warm.placed ? *prob.getInitiationInterval() : 0;
  assert((!warm.placed || greedyII >= warm.lowerBoundII) &&
         "placement only ever grows the II");

  // The heuristic ran the pre-pass whichever form this takes, so the schedule
  // this falls back to meets the period either way.
  Chaining chaining =
      chainingFor(prob, cycleTime, opts.kind == SchedulerKind::ExactChaining);

  // Window: region laid out end to end (satisfying precedence and chain breaks)
  // plus one II per contending op, widened to the heuristic's own reach. Must
  // be provably sufficient, since INFEASIBLE here counts as proof.
  const auto &ops = prob.getOperations();
  int64_t sequential = 0;
  int64_t greedyReach = 0;
  unsigned contending = 0;
  for (Operation *op : ops) {
    sequential += prob.latencyOf(op) + 1;
    if (prob.contendsForUnit(op))
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
    if (prob.holdsLimitedUnit(op))
      totalOccupancy += prob.getResourceCycles(op);
  unsigned upperII =
      warm.placed ? greedyII : std::max(warm.lowerBoundII, totalOccupancy);

  // The part of `leafSpan` this solve controls. With no trip there is no span
  // to compare across intervals, so the search takes the first feasible II,
  // placed as shallowly as the anchor objective can manage.
  bool bySpan = span.trip.has_value();
  int64_t iiWeight = bySpan ? *span.trip - 1 : 0;

  // The incumbent: bounds every model below, and is the fallback if none beats
  // it. Without it, a budget-limited placement at a new II is unbounded and can
  // ship a schedule worse than the heuristic's.
  std::optional<int64_t> heuristicSpan;
  if (bySpan && warm.placed)
    heuristicSpan = iiWeight * greedyII + span.drainOf(prob);
  std::optional<int64_t> best = heuristicSpan;
  int64_t floorDrain = bySpan ? drainFloor(prob, chaining, span.drain) : 0;

  // Whether this region has an allocation to decide at all, which the cut
  // below admits a span tie for.
  bool allocates = false;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    allocates |= prob.getAllocatable(rsrc).has_value();

  DenseMap<Operation *, unsigned> bestStarts;
  Allocated bestAllocation;
  int64_t bestArea = 0;
  unsigned bestII = 0;
  bool bestProven = false;
  bool adopted = false;
  std::optional<unsigned> exhaustedAt;

  for (unsigned ii = warm.lowerBoundII; ii <= upperII; ++ii) {
    // Cut: this interval's span already reaches the incumbent's before a single
    // operation is placed in it, and every interval past it is worse. Where an
    // allocation is decided, an interval that only ties on span can still win
    // on area, so the cut admits the tie.
    if (best && iiWeight * ii + floorDrain >= *best + (allocates ? 1 : 0))
      break;
    std::optional<int64_t> drainBound;
    if (best)
      drainBound = *best - iiWeight * ii;

    DenseMap<Operation *, unsigned> starts;
    Allocated decided;
    bool proven = false;
    int64_t drain = 0;
    ModuloOutcome outcome = solveAtII(prob, lastOp, chaining, span, opts,
                                      drainBound, ii, window + ii * contending,
                                      /*hint=*/warm.placed && ii == greedyII,
                                      starts, decided, proven, drain);
    if (outcome == ModuloOutcome::Infeasible) {
      // INFEASIBLE is a proof only where nothing bounded the solve; under the
      // incumbent's bound it is the weaker "nothing here beats it".
      assert((!warm.placed || ii < greedyII || drainBound) &&
             "the greedy's own schedule satisfies this encoding at the II it "
             "achieved");
      continue;
    }
    if (outcome == ModuloOutcome::Exhausted) {
      // Stop rather than try a wider interval: the budget just proved this
      // problem hard.
      exhaustedAt = ii;
      break;
    }
    // Adopt on a strict improvement, or on the first exact schedule at all.
    // Improvement is lexicographic: span first, then the instances built.
    int64_t solved = iiWeight * ii + drain;
    int64_t area = areaOf(prob, decided);
    if (!adopted || solved < *best || (solved == *best && area < bestArea)) {
      best = solved;
      bestArea = area;
      bestII = ii;
      bestProven = proven;
      bestStarts = std::move(starts);
      bestAllocation = std::move(decided);
      adopted = true;
    }
    if (!bySpan)
      break;
  }

  if (!adopted) {
    if (!warm.placed) {
      auto d = unsupported(Stage::Sched, prob.getContainingOp());
      d << "Neither scheduler could place this region: the greedy modulo "
           "placement gave up, and ";
      if (exhaustedAt)
        d << "the exact one ran out of budget at II=" << *exhaustedAt
          << " without deciding it";
      else
        d << "every initiation interval from " << warm.lowerBoundII << " to "
          << upperII << " is infeasible";
      return failure();
    }
    // Both arms leave the problem exactly as the simplex left it.
    if (exhaustedAt)
      warn(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling ran out of budget at II=" << *exhaustedAt
          << " without deciding it; falling back to the heuristic's schedule "
             "at II="
          << greedyII << ", which is therefore not known to be minimal";
    else
      info(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling found nothing shorter than the heuristic's "
             "schedule at II="
          << greedyII << "; keeping it";
    applyDemandAllocation(prob, greedyII);
    return success();
  }

  prob.setInitiationInterval(bestII);
  for (Operation *op : ops)
    prob.setStartTime(op, bestStarts.at(op));
  applyAllocation(prob, bestAllocation, bestII);

  {
    auto d = info(Stage::Sched, prob.getContainingOp());
    d << "Exact scheduling placed the region at II=" << bestII;
    if (!warm.placed)
      d << ": the greedy placement could not place it at all";
    else if (bestII < greedyII)
      d << ", down from the heuristic's II=" << greedyII
        << ": the gap was greedy resource placement";
    else
      d << ", the II the heuristic also reached";
    if (bySpan) {
      d << "; span " << *best;
      if (heuristicSpan)
        d << " against the heuristic's " << *heuristicSpan;
    }
  }
  // An exhausted budget leaves the placement inside the interval unproven,
  // and that placement is what the region's span is charged.
  if (!bestProven)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget placing the region at II="
        << bestII
        << ", so it shipped the best schedule it had found rather than the "
           "cheapest one; what it reached is no worse than the heuristic's but "
           "is not known to be minimal in span, registers or instances";
  if (exhaustedAt)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget at II=" << *exhaustedAt
        << " without deciding it, so the search stopped there; what it kept is "
           "the best of the intervals it did decide";
  return finishSchedule(prob, cycleTime);
}

#else // !ALLO_ENABLE_ORTOOLS

namespace {
/// Unreachable through `runSDCScheduler`, which rejects `scheduler="exact"` on
/// an OR-Tools-free build before any region is solved. Kept so the two entry
/// points exist in both configurations.
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
                                        Operation *lastOp, float cycleTime,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  return noExactScheduler(prob.getContainingOp(), "acyclic");
}

LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  return noExactScheduler(prob.getContainingOp(), "cyclic");
}

#endif // ALLO_ENABLE_ORTOOLS
