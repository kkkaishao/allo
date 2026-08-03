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

/// The solver configuration every Allo solve uses: single-worker search with a
/// fixed seed and a deterministic (not wall-clock) limit of \p budget, so two
/// identical compiles emit identical RTL. That reproducibility is what the
/// content-keyed simulation cache, a reproducible bug report and an A/B between
/// two compiler revisions all rest on; the only thing it costs is parallel
/// search, which problems of this size do not need.
SatParameters solverParameters(double budget) {
  SatParameters params;
  params.set_num_workers(1);
  params.set_random_seed(0);
  params.set_max_deterministic_time(budget);
  return params;
}

/// The weight of a precedence edge out of \p src: the cycles a dependent must
/// wait after \p src issues before the value has arrived.
int64_t latencyOf(Problem &prob, Operation *src) {
  return *prob.getLatency(*prob.getLinkedOperatorType(src));
}

/// How a model states the clock period. Exactly one form is in force: either
/// the chain-breaking edges the pre-pass computed, which cost a cycle each on
/// top of the plain precedence, or the period itself, which `addChaining`
/// encodes as sub-cycle start times and lets the solver break chains against.
struct Chaining {
  SmallVector<Problem::Dependence> breaks;
  std::optional<float> period;
};

/// The chaining constraint \p prob is solved under: the pre-pass's edges, or
/// \p cycleTime itself when \p exactChaining.
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

/// Sub-cycle time is a REAL quantity (ns) where CP-SAT is integer, so the model
/// carries it in PICOSECONDS, rounded to the nearest. A device states delays to
/// a hundredth of a nanosecond and an address cone sums a few of them, so a
/// picosecond resolves every value the model can see and the rounding only
/// absorbs the float representation error (1.2f is 1.20000004768 ns). Rounding
/// to nearest rather than up matters: a chain that fills the period EXACTLY is
/// common at these delays, and rounding each link up would make the model
/// reject a schedule the clock accepts.
constexpr double kPicosPerNs = 1000.0;
int64_t picos(double ns) { return std::llround(ns * kPicosPerNs); }

/// The period as a constraint on the model rather than as edges from a
/// pre-pass. Adds one sub-cycle start time `z` per operation, in picoseconds
/// from the start of its cycle, and reads exactly as
/// `computeStartTimesInCycle` computes them afterwards:
///
///   * `z(v) <= P - inDelay(v)`, i.e. v's inputs reach its first register (or
///     its output, if it is combinational) inside the period;
///   * where a def-use producer u ENDS in the cycle v starts,
///     `z(v) >= (lat(u) == 0 ? z(u) : 0) + outDelay(u)`, i.e. v waits for the
///     chain it sits on.
///
/// "Ends in the cycle v starts" is `t_v - t_u == lat(u)`, and precedence
/// already forces `>=`, so the `<=` half alone reifies it; two
/// half-implications on that inequality propagate where an
/// equality/not-equality pair would not.
///
/// Only DEF-USE edges carry a combinational path. An auxiliary edge is a memory
/// or stream order, or a loop-carried recurrence, and every one of them passes
/// through a port or a register; the pre-pass skips them for the same reason,
/// so both forms see the same chains and their difference stays attributable to
/// WHERE each breaks them.
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

/// State \p chaining on the model, whichever form it takes: a chain-breaking
/// edge is a precedence one cycle wider than the dependence the caller has
/// already stated, and a period is the encoding above.
template <class ProblemT>
void addChaining(CpModelBuilder &model, ProblemT &prob,
                 DenseMap<Operation *, IntVar> &startVars,
                 const Chaining &chaining) {
  for (const Problem::Dependence &dep : chaining.breaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             latencyOf(prob, dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));
  if (chaining.period)
    addSubCycleTimes(model, prob, startVars, *chaining.period);
}

/// Every operation's inputs settle within the period.
[[maybe_unused]] bool chainsFitCycleTime(ChainingProblem &prob,
                                         float cycleTime) {
  // One picosecond: the model's own resolution (`picos`), and more than the
  // float error the pre-pass and this accumulate differently.
  constexpr float kSlop = 1e-3f;
  for (Operation *op : prob.getOperations()) {
    float in = *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
    if (*prob.getStartTimeInCycle(op) + in > cycleTime + kSlop)
      return false;
  }
  return true;
}

/// Derive the solved schedule's sub-cycle start times, and check the period the
/// whole encoding exists to hold. Nothing else checks it: `ChainingProblem`
/// does not carry the period, so neither its `verify` nor
/// `computeStartTimesInCycle` can test it, and a schedule that misses it is not
/// wrong in simulation, only slow in silicon. The check is a cross-check on the
/// pre-pass's edges having reached the model intact, and the sub-cycle
/// encoding's own guard.
LogicalResult finishSchedule(ChainingProblem &prob, float cycleTime) {
  if (failed(computeStartTimesInCycle(prob)))
    return failure();
  assert(chainsFitCycleTime(prob, cycleTime) &&
         "the schedule this returned puts a combinational chain across more "
         "than one clock period, which nothing downstream detects and which "
         "silicon reports as a timing failure");
  return success();
}

/// The region's drain as a variable: the max of `start(op) + offset` over the
/// same terms `drainOf` maxes over. Lower bounds alone, which is tight because
/// the objective minimizes the drain above everything else.
///
/// \p bound caps it at an incumbent's, so the solver spends its budget only on
/// schedules that would WIN. A solve that then exhausts its budget comes back
/// with something better than the incumbent rather than with something merely
/// legal, and one that comes back INFEASIBLE has said the weaker thing the
/// search asked rather than proved the interval impossible.
IntVar drainVariable(CpModelBuilder &model,
                     DenseMap<Operation *, IntVar> &startVars,
                     ArrayRef<DrainTerm> terms, int64_t horizon,
                     std::optional<int64_t> bound) {
  assert((!bound || *bound >= 0) && "an incumbent the interval cannot beat is "
                                    "cut before a model is built for it");
  IntVar drain = model.NewIntVar(operations_research::Domain(
      0, bound ? std::min(*bound, horizon) : horizon));
  for (const DrainTerm &term : terms)
    model.AddLessOrEqual(startVars.at(term.op) + term.offset, drain);
  return drain;
}

/// Minimize \p primary, with the region's AREA as the tie-break below it. The
/// two are weighted so they never interact: `primary` is what the region's span
/// is charged and is settled first; the tie-break decides only among schedules
/// that reach it.
///
/// The tie-break is two terms, both flip-flops:
///
///   * `width(v) * depth(v)` per value carried in a delay chain
///     (`RegisterTerm`), which is 84% of the bed's register bits;
///   * the sum of all start times at weight ONE, which is what a 1-bit
///     ACTIVATION PULSE chain costs per cycle of an op's start offset, and what
///     an index-typed address register costs to within its own width. It is the
///     part of a region's area the term above does not carry.
///
/// The relative weights are therefore the hardware's, not a tuning knob: a
/// value chain is `width` times a pulse chain because it is `width` flip-flops
/// wide. Both directions matter and they oppose each other. Register depth
/// wants a producer LATE and its readers EARLY where the sum of starts wants
/// everything early, so a chain's interior is settled by the pulse term while
/// its endpoints are settled by the value term.
void minimizeCost(CpModelBuilder &model, IntVar primary,
                  ArrayRef<IntVar> starts, const SpanObjective &span,
                  DenseMap<Operation *, IntVar> &startVars, int64_t ii,
                  int64_t horizon) {
  SmallVector<IntVar> vars(starts.begin(), starts.end());
  SmallVector<int64_t> weights(starts.size(), 1);
  // How far the whole tie-break can reach per cycle of horizon, so the weight
  // below orders `primary` strictly above it.
  int64_t bits = starts.size();
  for (const RegisterTerm &term : span.regs) {
    IntVar depth = model.NewIntVar(operations_research::Domain(0, horizon));
    IntVar def = startVars.at(term.def);
    // Lower bounds only, tight because the objective minimizes the depth.
    for (auto [reader, distance] : term.reads)
      model.AddLessOrEqual(startVars.at(reader) + distance * ii - term.latency,
                           def + depth);
    vars.push_back(depth);
    weights.push_back(term.width);
    bits += term.width;
  }
  vars.push_back(primary);
  weights.push_back(bits * (horizon + 1));
  model.Minimize(LinearExpr::WeightedSum(vars, weights));
}

/// Report a solve that produced nothing usable and leave the heuristic's
/// schedule in place. `warn`, not `error` or `unsupported`: the compile is
/// still correct and still finishes, it just did not get the better schedule
/// it asked for.
void reportUnsolved(Problem &prob, const CpSolverResponse &response,
                    double budget) {
  assert(response.status() != CpSolverStatus::INFEASIBLE &&
         response.status() != CpSolverStatus::MODEL_INVALID &&
         "the heuristic's schedule satisfies every constraint this encoding "
         "states, so the model is satisfiable by construction; an infeasible "
         "or invalid model means the encoding disagrees with the problem");
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

/// Refines the heuristic's acyclic schedule to the CP-SAT optimum. The
/// heuristic runs first as both a feasibility check and a warm-start hint:
/// its resource-free LP is the only thing that can fail, and no exact solver
/// repairs an infeasible LP either, so a failure here is fatal (unlike the
/// cyclic path, where placement is optional). The horizon is the whole region
/// laid out end to end, which is guaranteed to contain the heuristic's own
/// schedule; bounding it by the heuristic's depth instead would exclude a
/// schedule that drains sooner at the cost of some other operation finishing
/// later.
///
/// A straight-line region arms at the top boundary and runs once, so its whole
/// cost is its DRAIN, and that is what the objective minimizes, with the sum of
/// all starts as a tie-break so an off-path operation cannot drift for free.
/// The drain is upper-bounded by the heuristic's own, pruning like a
/// branch-and-bound incumbent without excluding the optimum, unlike the horizon
/// which would be a hard ceiling.
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
  // chain breaks the heuristic just used, leaving greedy resource placement as
  // the only difference between the two schedules.
  Chaining chaining =
      chainingFor(prob, cycleTime, opts.kind == SchedulerKind::ExactChaining);

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
                               latencyOf(prob, dep.getSource()),
                           startVars.at(dep.getDestination()));
  addChaining(model, prob, startVars, chaining);

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

  // What the region is charged, bounded by what the heuristic already reached.
  int64_t heuristicDrain = drainOf(prob, span.drain);
  IntVar drain =
      drainVariable(model, startVars, span.drain, horizon, heuristicDrain);
  minimizeCost(model, drain, orderedStarts, span, startVars, /*ii=*/0, horizon);

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts.budget));
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    reportUnsolved(prob, response, opts.budget);
    return success();
  }

  // A FEASIBLE result is a legal schedule the budget stopped short of proving
  // optimal, so what ships is an incumbent rather than the shortest schedule.
  // Said out loud, because the drain bound means it can never be WORSE than
  // the heuristic's, which makes a silent one indistinguishable from a win.
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
  return finishSchedule(prob, cycleTime);
}

//===----------------------------------------------------------------------===//
// The cyclic solve: a branch and bound over initiation intervals.
//===----------------------------------------------------------------------===//

namespace {

/// What one fixed-II solve settled. `Infeasible` is a PROOF that the initiation
/// interval admits no schedule, which is the answer the whole II search is
/// built to get; `Exhausted` is the solver giving up, which proves nothing.
enum class ModuloOutcome { Scheduled, Infeasible, Exhausted };

/// A lower bound on the region's drain at ANY initiation interval: the longest
/// chain of INTRA-iteration edges reaching an output. An edge spanning
/// iterations is relaxed by one II per iteration it spans, so only the
/// distance-0 subgraph bounds a start time from below however wide the interval
/// gets, and resources only push starts later.
///
/// This is what makes the branch and bound's cut tight. Cutting on
/// `(trip - 1) * ii` alone is sound but useless wherever the drain dwarfs the
/// trip: a 25-iteration region draining at 215 admits every interval up to 25
/// as a candidate, so the search scans them at a full budget each and comes
/// back with the schedule it started from.
///
/// Only \p chaining's break edges lengthen a path here. Where the period is
/// stated in the model instead, there are no break edges before the solve and
/// the bound is simply looser, which the cut tolerates and a break edge could
/// only tighten.
int64_t drainFloor(ChainingModuloProblem &prob, const Chaining &chaining,
                   ArrayRef<DrainTerm> terms) {
  // Incoming edges by destination, weighted as the model weights them.
  DenseMap<Operation *, SmallVector<std::pair<Operation *, int64_t>>> incoming;
  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        incoming[op].push_back(
            {dep.getSource(), latencyOf(prob, dep.getSource())});
  for (auto &dep : chaining.breaks)
    incoming[dep.getDestination()].push_back(
        {dep.getSource(), latencyOf(prob, dep.getSource()) + 1});

  // Longest path, memoized. The distance-0 subgraph is acyclic, and the seeded
  // zero keeps a cycle from recursing forever if it ever were not: the answer
  // stays a lower bound either way, which is all the cut needs.
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
///
/// \p proven distinguishes the two ways a schedule comes back. `OPTIMAL` means
/// the placement inside this II minimizes the objective; `FEASIBLE` means the
/// budget ran out with an incumbent in hand, which is a legal schedule and
/// nobody's optimum. The II search cannot tell them apart from the outcome
/// alone, and the difference is not cosmetic: what an unproven placement drains
/// at is what the region's span is charged.
///
/// \p drainBound is the incumbent's, keeping even an unproven placement inside
/// what the search would accept. It makes INFEASIBLE the weaker statement "no
/// schedule here beats what we have" rather than a proof about the interval.
ModuloOutcome solveAtII(ChainingModuloProblem &prob, Operation *lastOp,
                        const Chaining &chaining, const SpanObjective &span,
                        const SchedulerOptions &opts,
                        std::optional<int64_t> drainBound, unsigned ii,
                        unsigned horizon, bool hint,
                        DenseMap<Operation *, unsigned> &starts, bool &proven,
                        int64_t &drain) {
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
          latencyOf(prob, src) -
          static_cast<int64_t>(ii) * prob.getDistance(dep).value_or(0);
      model.AddLessOrEqual(startVars.at(src) + separation,
                           startVars.at(dep.getDestination()));
    }
  addChaining(model, prob, startVars, chaining);

  // One-hot congruence class per contending op. `t = ii*lap + sum(p*slot[p])`
  // over a one-hot slot defines class and modulo at once, with no
  // reification: slot[p] IS membership in class p, which the sums below need.
  DenseMap<Operation *, SmallVector<BoolVar>> slotsOf;
  SmallVector<int64_t> classes(ii);
  for (unsigned p = 0; p < ii; ++p)
    classes[p] = p;
  for (Operation *op : ops) {
    if (!prob.holdsLimitedUnit(op))
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

  // `(trip - 1) * ii` is a constant at a fixed II, so minimizing the span here
  // is minimizing the drain, and the outer search carries the II term. Where no
  // span composes off this solve the anchor's start time takes the primary
  // slot; the area tie-break underneath it is the same either way.
  std::optional<IntVar> drainVar;
  if (span.trip)
    drainVar = drainVariable(model, startVars, span.drain, horizon, drainBound);
  minimizeCost(model, drainVar.value_or(orderedStarts[anchorIndex]),
               orderedStarts, span, startVars, ii, horizon);

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
  drain = drainVar ? SolutionIntegerValue(response, *drainVar) : 0;
  return ModuloOutcome::Scheduled;
}

} // namespace

/// Refines the heuristic's modulo (cyclic) schedule by searching fixed II
/// values from the heuristic's own II lower bound upward, as a BRANCH AND BOUND
/// on the region's span. The only thing this path needs from the heuristic is
/// that lower bound, settled by the resource-free LP before any placement runs;
/// without it the search would start at II=1 and spend its budget refuting
/// intervals no bound ever allowed. The heuristic's own placement is optional
/// context (`SimplexWarmStart`), not a precondition: what must still succeed is
/// the LP underneath, which is exact regardless of whether placement did.
///
/// The search cannot return at the first feasible II, because a SMALLER II is
/// only better for a large trip: what the region is charged is
/// `(trip - 1) * ii + drain`, and a longer interval admits a shorter drain. So
/// it keeps the best span seen and cuts at the first interval whose II term
/// alone already reaches it.
LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  // Needs only the heuristic's II lower bound from the LP; its placement is
  // optional (see SimplexWarmStart) and not required to succeed.
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

  // Window: region laid out end to end (satisfies precedence + chain
  // breaks) plus one II per contending op, widened to the heuristic's own
  // reach. Must be provably sufficient, since INFEASIBLE here counts as proof.
  const auto &ops = prob.getOperations();
  int64_t sequential = 0;
  int64_t greedyReach = 0;
  unsigned contending = 0;
  for (Operation *op : ops) {
    sequential += latencyOf(prob, op) + 1;
    if (prob.holdsLimitedUnit(op))
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
  // to compare across intervals and the search keeps its old shape: the first
  // feasible II, placed as shallowly as the anchor objective can manage.
  bool bySpan = span.trip.has_value();
  int64_t iiWeight = bySpan ? *span.trip - 1 : 0;

  // The INCUMBENT, and the whole safety of this path. The heuristic's own span
  // is the first one, so an exact solve can only ship a region that beats it:
  // as a bound inside every model below, and as the fallback if none does.
  // Without it a budget-limited placement at a NEW II is bounded by nothing at
  // all, which is how a region 60% slower than the heuristic's shipped.
  std::optional<int64_t> heuristicSpan;
  if (bySpan && warm.placed)
    heuristicSpan = iiWeight * greedyII + drainOf(prob, span.drain);
  std::optional<int64_t> best = heuristicSpan;
  int64_t floorDrain = bySpan ? drainFloor(prob, chaining, span.drain) : 0;

  DenseMap<Operation *, unsigned> bestStarts;
  unsigned bestII = 0;
  bool bestProven = false;
  bool adopted = false;
  std::optional<unsigned> exhaustedAt;

  for (unsigned ii = warm.lowerBoundII; ii <= upperII; ++ii) {
    // The bound: this interval's span already reaches the incumbent's before a
    // single operation is placed in it, and every interval past it is worse.
    if (best && iiWeight * ii + floorDrain >= *best)
      break;
    std::optional<int64_t> drainBound;
    if (best)
      drainBound = *best - iiWeight * ii;

    DenseMap<Operation *, unsigned> starts;
    bool proven = false;
    int64_t drain = 0;
    ModuloOutcome outcome = solveAtII(prob, lastOp, chaining, span, opts,
                                      drainBound, ii, window + ii * contending,
                                      /*hint=*/warm.placed && ii == greedyII,
                                      starts, proven, drain);
    if (outcome == ModuloOutcome::Infeasible) {
      // A PROOF that the interval admits no schedule only where nothing bounded
      // the solve. Under the incumbent's bound it is the weaker statement the
      // search asked for, and the greedy's own schedule may well be outside it.
      assert((!warm.placed || ii < greedyII || drainBound) &&
             "the greedy's own schedule satisfies every constraint this "
             "encoding states at the II it achieved, so INFEASIBLE there means "
             "the encoding and the reservation model disagree");
      continue;
    }
    if (outcome == ModuloOutcome::Exhausted) {
      // Stop rather than try a wider interval: the budget just proved this
      // problem hard, and the incumbent is what a wider one would have to beat.
      exhaustedAt = ii;
      break;
    }
    // Adopt on a strict improvement, or on the first exact schedule at all:
    // every model here is bounded by the incumbent, so an equal span is that
    // same span reached with the tie-break minimized too.
    int64_t solved = iiWeight * ii + drain;
    if (!adopted || solved < *best) {
      best = solved;
      bestII = ii;
      bestProven = proven;
      bestStarts = std::move(starts);
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
    // The guard: the heuristic's schedule stands unless an exact one beats the
    // span it composes to. Both arms leave the problem exactly as the simplex
    // left it.
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
    return success();
  }

  prob.setInitiationInterval(bestII);
  for (Operation *op : ops)
    prob.setStartTime(op, bestStarts.at(op));

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
  // What an exhausted budget leaves unproven is the placement inside the
  // interval, and that placement is what the region's drain, and so its span,
  // is charged. The bound keeps it from being worse than the incumbent; nothing
  // makes it best.
  if (!bestProven)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget placing the region at II="
        << bestII
        << ", so it shipped the best schedule it had found rather than the "
           "shortest one; the span it reached is nobody's optimum";
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
                                        Operation *lastOp, float cycleTime,
                                        const SpanObjective &span,
                                        bool exactChaining) {
  return noExactScheduler(prob.getContainingOp(), "acyclic");
}

LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII,
                                        const SpanObjective &span,
                                        bool exactChaining) {
  return noExactScheduler(prob.getContainingOp(), "cyclic");
}

#endif // ALLO_ENABLE_ORTOOLS
