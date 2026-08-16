/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h" // the device the area is priced at
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

#include <cmath>
#include <limits>
#include <type_traits>

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

using namespace circt::scheduling;
using namespace operations_research::sat;

namespace {

/// Solver configuration for one solve. The time limit is deterministic rather
/// than wall-clock, which is what lets two identical compiles emit identical
/// RTL. A solve that exhausts that limit has been seen to differ run to run
/// even so, which is unexplained.
SatParameters solverParameters(const SchedulerOptions &opts) {
  SatParameters params;
  params.set_num_workers(opts.workers);
  params.set_random_seed(opts.seed);
  params.set_max_deterministic_time(opts.budget);
  // Several workers otherwise race, and which incumbent the budget stops on
  // would depend on thread timing. Interleaved, the portfolio advances in a
  // fixed order under the deterministic limit above.
  if (opts.workers > 1)
    params.set_interleave_search(true);
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
Chaining chainingFor(ProblemT &prob, float cycleTime, float regFloor,
                     bool exactChaining) {
  Chaining chaining;
  if (exactChaining) {
    chaining.period = cycleTime;
    return chaining;
  }
  auto broke = mlir::allo::computeChainBreaks(prob, cycleTime, regFloor,
                                              chaining.breaks);
  assert(succeeded(broke) && "chain breaking is a pure function of the problem "
                             "and the cycle time, and the heuristic just ran "
                             "it successfully");
  (void)broke;
  return chaining;
}

/// Sub-cycle time in picoseconds, rounded to nearest: CP-SAT is integer, and a
/// picosecond has enough resolution against delays given to a hundredth of a
/// nanosecond. Round-to-nearest rather than up, since a chain that fills the
/// period exactly is common and rounding up would reject it.
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
/// or register.
///
/// Returns the per-operation sub-cycle variables, for constraints stated on
/// top of the system (`addAllocationHeadroom`).
template <class ProblemT>
DenseMap<Operation *, IntVar>
addSubCycleTimes(CpModelBuilder &model, ProblemT &prob,
                 DenseMap<Operation *, IntVar> &startVars, float cycleTime,
                 float regFloor) {
  int64_t period = picos(cycleTime);
  // Nothing in a cycle starts before its operands leave a register, so the
  // fabric floor is every `z`'s lower bound. A chain from a registered producer
  // then costs `max(floor, that producer's outgoing delay)`.
  int64_t floor = picos(regFloor);
  DenseMap<Operation *, IntVar> inCycle;
  for (Operation *op : prob.getOperations()) {
    int64_t in = picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
    assert(in + floor <= period &&
           "an operator whose own delay exceeds the period is rejected by the "
           "chain-breaking pre-pass, which the heuristic ran before this");
    inCycle.try_emplace(
        op, model.NewIntVar(operations_research::Domain(floor, period - in)));
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
  return inCycle;
}

/// State \p chaining on the model: a chain-breaking edge widens an existing
/// precedence by one cycle; a period uses the sub-cycle encoding above, whose
/// variables are passed back (empty in the break-edge form).
template <class ProblemT>
DenseMap<Operation *, IntVar>
addChaining(CpModelBuilder &model, ProblemT &prob,
            DenseMap<Operation *, IntVar> &startVars, const Chaining &chaining,
            float regFloor) {
  for (const Problem::Dependence &dep : chaining.breaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             prob.latencyOf(dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));
  if (chaining.period)
    return addSubCycleTimes(model, prob, startVars, *chaining.period, regFloor);
  return DenseMap<Operation *, IntVar>();
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
LogicalResult finishSchedule(ChainingProblem &prob, float cycleTime,
                             float regFloor) {
  if (failed(mlir::allo::computeStartTimesInCycle(prob, regFloor)))
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
  /// The fullest instance's select-cone delay in picoseconds, `headroomNs`
  /// read at `units`; absent where no count this resource can take builds a
  /// select.
  std::optional<IntVar> headroom;
};

/// The tightest count of \p rsrc the schedule CURRENTLY on \p prob admits:
/// the busiest-slot demand, opened until the select cone fits the sub-cycle
/// slack that schedule leaves the resource's operations. Pairing this count
/// with those start times satisfies the headroom constraint, so as a hint it
/// stays a feasible point and as a fallback it stays buildable.
template <class ProblemT>
unsigned demandWithHeadroom(ProblemT &prob, Problem::ResourceType rsrc,
                            unsigned ii, float cycleTime) {
  auto unit = prob.getAllocatable(rsrc);
  double slack = cycleTime;
  for (Operation *op : prob.getOperations())
    if (prob.usesResource(op, rsrc))
      slack = std::min(
          slack, double(cycleTime) - *prob.getStartTimeInCycle(op) -
                     *prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
  unsigned n = prob.demandFor(rsrc, ii);
  while (n < unit->ceiling && unit->headroomNs[n] > slack)
    ++n;
  return n;
}

/// Declare `N_r` for every allocatable resource: how many copies of one
/// operator this region builds, in `[1, ceiling]`. The caller states the
/// capacity constraint against it.
///
/// \p hint says the heuristic's start times are being hinted too, and then the
/// count hinted is the TIGHTEST one those start times admit with the select
/// cone charged (what the greedy binder could have built from them); on a
/// region whose budget runs out, `applyDemandAllocation` ships that same
/// count.
template <class ProblemT>
SmallVector<AllocationVar> allocationVars(CpModelBuilder &model, ProblemT &prob,
                                          unsigned ii, bool hint,
                                          float cycleTime) {
  SmallVector<AllocationVar> allocs;
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    auto unit = prob.getAllocatable(rsrc);
    if (!unit)
      continue;
    assert(unit->ceiling > 0 && "an allocatable resource with no operation");
    IntVar n = model.NewIntVar(operations_research::Domain(1, unit->ceiling));
    if (hint)
      model.AddHint(n, demandWithHeadroom(prob, rsrc, ii, cycleTime));
    std::vector<int64_t> table(unit->price.begin(), unit->price.end());
    int64_t hi = *llvm::max_element(unit->price);
    IntVar price = model.NewIntVar(
        operations_research::Domain(*llvm::min_element(table), hi));
    model.AddElement(n, table, price);
    AllocationVar alloc{rsrc, n, price, hi, std::nullopt};
    std::vector<int64_t> cone;
    cone.reserve(unit->headroomNs.size());
    for (double ns : unit->headroomNs)
      cone.push_back(picos(ns));
    if (int64_t top = *llvm::max_element(cone)) {
      alloc.headroom = model.NewIntVar(operations_research::Domain(0, top));
      model.AddElement(n, cone, *alloc.headroom);
    }
    allocs.push_back(alloc);
  }
  return allocs;
}

/// Hold every operation of an allocatable operator to the period with the
/// select cone its decided count implies: `z + inDelay + headroom(N) <=
/// period`. This is what lets a `planned` binding realize the allocation as
/// built: a count only shrinks where its multiplexer fits inside the slack the
/// same solve leaves, so the emit-side gate has nothing left to refuse.
///
/// The break-edge chaining form carries no sub-cycle variables, so they are
/// created here on demand; the break edges already keep the plain system
/// satisfiable at any placement, so adding it tightens the model only by the
/// headroom itself.
template <class ProblemT>
void addAllocationHeadroom(CpModelBuilder &model, ProblemT &prob,
                           DenseMap<Operation *, IntVar> &startVars,
                           DenseMap<Operation *, IntVar> &inCycle,
                           ArrayRef<AllocationVar> allocs, float cycleTime,
                           float regFloor) {
  if (llvm::none_of(allocs, [](const AllocationVar &a) {
        return a.headroom.has_value();
      }))
    return;
  if (inCycle.empty())
    inCycle = addSubCycleTimes(model, prob, startVars, cycleTime, regFloor);
  int64_t period = picos(cycleTime);
  for (const AllocationVar &alloc : allocs) {
    if (!alloc.headroom)
      continue;
    for (Operation *op : prob.getOperations()) {
      if (!prob.usesResource(op, alloc.rsrc))
        continue;
      int64_t in =
          picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
      model.AddLessOrEqual(inCycle.at(op) + *alloc.headroom, period - in);
    }
  }
}

/// Add \p price at \p size to a weighted sum, for a price tabulated at every
/// value the size can take. A piecewise-linear price is its FIRST slope on the
/// size, plus at every change of slope that change charged on how far the size
/// runs past the point it changes at: `max(size - b, 0)`. Every variable this
/// adds is determined by the size through a propagator, avoiding a
/// per-segment disjunction for the search to branch on.
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
/// The tie-break is the region's area, every term of it in what the device
/// spends: the delay chain each value carried across slack costs
/// (`RegisterTerm`), one stage of a one-bit activation pulse chain per cycle of
/// every start offset, and the table above per allocatable operator. A chain is
/// not `width * depth` flip-flops: neither run holds a reset, so past a
/// measured depth the synthesizer extracts a shift register and the cost stops
/// rising with the depth outside the steps a new site adds.
void minimizeCost(CpModelBuilder &model, IntVar primary,
                  ArrayRef<IntVar> starts, const SpanObjective &span,
                  DenseMap<Operation *, IntVar> &startVars,
                  ArrayRef<AllocationVar> allocs, int64_t ii, int64_t horizon) {
  int64_t pulse = span.device.pulsePrice();
  SmallVector<IntVar> vars(starts.begin(), starts.end());
  SmallVector<int64_t> weights(starts.size(), pulse);
  // Bounds the tie-break so `primary`'s weight below dominates it. Loose on
  // purpose: a tight bound turns the tie-break into a comparable share of the
  // objective, at a large search cost for negligible area.
  int64_t area = static_cast<int64_t>(starts.size()) * pulse;
  // At II > 1 the emitter folds every chain onto the region's phase, holding
  // `depth` taps in `ceil(depth / ii)` registers (`EmitContext::foldedChain`).
  // The variable below is therefore the registers BUILT rather than the cycles
  // spanned, and the table is indexed by that same count.
  int64_t fold = std::max<int64_t>(ii, 1);
  int64_t stages = (horizon + fold - 1) / fold;
  // One chain price table per width: a region carries many values of the same
  // type, and tabulating the device's cost is the expensive half.
  DenseMap<int64_t, SmallVector<int64_t>> chainPrices;
  for (const RegisterTerm &term : span.regs) {
    auto [entry, isNew] = chainPrices.try_emplace(term.width);
    if (isNew)
      for (int64_t n = 0; n <= stages; ++n)
        entry->second.push_back(span.device.chainPrice(n, term.width));
    ArrayRef<int64_t> table = entry->second;
    IntVar built = model.NewIntVar(operations_research::Domain(0, stages));
    IntVar def = startVars.at(term.def);
    // Only bounded from below. A chain price is nondecreasing in its length, so
    // a minimizing solve lands `built` on the fold of the deepest read.
    for (auto [reader, distance] : term.reads)
      model.AddLessOrEqual(startVars.at(reader) + distance * ii - term.latency,
                           def + LinearExpr::Term(built, fold));
    addPiecewiseCost(model, built, table, vars, weights);
    area += *llvm::max_element(table);
  }
  for (const AllocationVar &alloc : allocs) {
    vars.push_back(alloc.price);
    weights.push_back(1);
    area += alloc.maxPrice;
  }
  // A device that declares no area model prices every term at nothing, which
  // is the honest reading of saying nothing; the floor keeps `primary` from
  // being weighted at nothing along with it.
  int64_t dominating = std::max<int64_t>(area, 1) * (horizon + 1);
#ifndef NDEBUG
  // Recomputes the tie-break's max reach from each variable's domain and
  // weight; a term added above without a matching charge to `area` trips this
  // rather than letting the tie-break outweigh `primary`.
  int64_t reach = 0;
  for (auto [var, weight] : llvm::zip(vars, weights)) {
    operations_research::Domain d = var.Domain();
    reach += std::max(weight * d.Min(), weight * d.Max());
  }
  assert(dominating > reach &&
         "the area tie-break reaches past the weight that is supposed to "
         "dominate it, so `area` above has stopped covering its own terms");
#endif
  vars.push_back(primary);
  weights.push_back(dominating);
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
/// admits, for a solve that decided none: the busiest-cycle demand with the
/// select cone held against the slack that schedule leaves. Without it, a
/// region whose budget ran out keeps the trivial allocation (one instance per
/// operation) instead of what the schedule actually supports.
template <class ProblemT>
void applyDemandAllocation(ProblemT &prob, unsigned ii, float cycleTime) {
  Allocated decided;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (prob.getAllocatable(rsrc))
      decided.push_back({rsrc, demandWithHeadroom(prob, rsrc, ii, cycleTime)});
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

/// Lower bound on the drain of ANY schedule of \p prob.
///
/// Two facts bound where an output can commit. Its own longest path is one. The
/// other is resource contention: for any set S of operations that must all pass
/// one capped resource before the output commits,
///
/// ```
/// start(v) >= minHead(S) + ceil( sum demand(u) / limit ) - 1 + minTail(S, v)
/// ```
///
/// since every member of S issues between the earliest head in it and
/// `start(v)` less the shortest path onward, a window whose capacity has to
/// cover them all. The longest path is this with S a singleton, where the
/// middle term vanishes.
///
/// Valid at every initiation interval, so the cyclic search computes it once:
/// within one iteration a window of length L touches `min(L, ii)` congruence
/// classes, each admitting `limit` units from that iteration, and work above
/// `ii * limit` is an interval `computeResMinII` already ruled out.
template <typename ProblemT>
int64_t drainFloor(ProblemT &prob, const Chaining &chaining,
                   ArrayRef<DrainTerm> terms) {
  constexpr int64_t kUnreached = std::numeric_limits<int64_t>::min();

  // The edges the model imposes, weighted as it weights them, in both
  // directions: heads are read off one end and tails off the other. Only the
  // edges that stay WITHIN one iteration bound this iteration's outputs, which
  // is every edge of a straight-line region and the distance-0 ones of a
  // modulo problem.
  DenseMap<Operation *, SmallVector<std::pair<Operation *, int64_t>>> in, out;
  auto edge = [&](Operation *src, Operation *dst, int64_t weight) {
    in[dst].push_back({src, weight});
    out[src].push_back({dst, weight});
  };
  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      if constexpr (std::is_same_v<ProblemT, ChainingModuloProblem>)
        if (prob.getDistance(dep).value_or(0) != 0)
          continue;
      edge(dep.getSource(), op, prob.latencyOf(dep.getSource()));
    }
  // A chain break is intra-iteration whichever problem this is.
  for (auto &dep : chaining.breaks)
    edge(dep.getSource(), dep.getDestination(),
         prob.latencyOf(dep.getSource()) + 1);

  // Longest path in, memoized; the seeded zero keeps a cycle from recursing
  // forever if the distance-0 subgraph were ever not acyclic.
  DenseMap<Operation *, int64_t> heads;
  auto head = [&](auto &self, Operation *op) -> int64_t {
    auto seen = heads.find(op);
    if (seen != heads.end())
      return seen->second;
    heads[op] = 0;
    int64_t longest = 0;
    auto edges = in.find(op);
    if (edges != in.end())
      for (auto [src, weight] : edges->second)
        longest = std::max(longest, self(self, src) + weight);
    heads[op] = longest;
    return longest;
  };

  int64_t bound = 0;
  for (const DrainTerm &term : terms)
    bound = std::max(bound, head(head, term.op) + term.offset);

  SmallVector<std::pair<Problem::ResourceType, int64_t>> capped;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (unsigned limit = prob.getLimit(rsrc).value_or(0))
      capped.push_back({rsrc, limit});
  if (capped.empty())
    return bound;

  struct Contender {
    int64_t head, tail, demand;
  };
  // The strongest bound over the threshold sets of a group. At fixed
  // thresholds widening a set only adds work, so the maximum lies on one of
  // them and the subsets themselves need no enumerating.
  auto strongest = [](SmallVectorImpl<Contender> &group, int64_t limit) {
    llvm::sort(group, [](const Contender &a, const Contender &b) {
      return a.tail > b.tail;
    });
    int64_t best = 0;
    for (const Contender &first : group) {
      int64_t work = 0;
      for (const Contender &c : group) {
        if (c.head < first.head)
          continue;
        work += c.demand;
        best = std::max(best,
                        first.head + (work + limit - 1) / limit - 1 + c.tail);
      }
    }
    return best;
  };

  DenseSet<Operation *> feeding;
  for (const DrainTerm &term : terms) {
    // Longest path on to this output, absent for an operation that cannot
    // reach it.
    DenseMap<Operation *, int64_t> tails;
    auto tail = [&](auto &self, Operation *op) -> int64_t {
      if (op == term.op)
        return 0;
      auto seen = tails.find(op);
      if (seen != tails.end())
        return seen->second;
      tails[op] = kUnreached;
      int64_t longest = kUnreached;
      auto edges = out.find(op);
      if (edges != out.end())
        for (auto [dst, weight] : edges->second) {
          int64_t onward = self(self, dst);
          if (onward != kUnreached)
            longest = std::max(longest, weight + onward);
        }
      tails[op] = longest;
      return longest;
    };
    for (auto [rsrc, limit] : capped) {
      SmallVector<Contender> group;
      for (Operation *op : prob.getOperations()) {
        if (!prob.usesResource(op, rsrc))
          continue;
        int64_t onward = tail(tail, op);
        if (onward == kUnreached)
          continue;
        feeding.insert(op);
        group.push_back({head(head, op), onward, prob.getResourceDemand(op)});
      }
      if (!group.empty())
        bound = std::max(bound, strongest(group, limit) + term.offset);
    }
  }

  // Every operation feeding any output issues by the drain whatever path it
  // takes there, which bounds the drain where no single output orders them all.
  for (auto [rsrc, limit] : capped) {
    SmallVector<Contender> group;
    for (Operation *op : feeding)
      if (prob.usesResource(op, rsrc))
        group.push_back({head(head, op), 0, prob.getResourceDemand(op)});
    if (!group.empty())
      bound = std::max(bound, strongest(group, limit));
  }
  return bound;
}

} // namespace

//===----------------------------------------------------------------------===//
// The acyclic solve.
//===----------------------------------------------------------------------===//

/// Refines the heuristic's acyclic schedule to the CP-SAT optimum.
///
/// The heuristic runs first as a feasibility check and a warm-start hint: its
/// resource-free LP is the only thing that can fail, so a failure here is
/// fatal.
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
  if (failed(
          mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, opts.regFloor)))
    return failure();

  // The pre-pass is schedule-independent, so taking its edges hands CP-SAT the
  // chain breaks the heuristic just used.
  Chaining chaining = chainingFor(prob, cycleTime, opts.regFloor,
                                  opts.kind == SchedulerKind::ExactChaining);

  const auto &ops = prob.getOperations();

  // The cyclic search's entry cut, at the one interval a straight-line region
  // has. `drainFloor` bounds the drain of any schedule from below, so reaching
  // it proves this one is as short as the region gets and leaves only the area
  // tie-break. `scheduleSimplex` has already written the start times and their
  // sub-cycle offsets, so shipping its schedule needs nothing further. An
  // allocation still to decide is worth the solve anyway.
  bool allocates = false;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    allocates |= prob.getAllocatable(rsrc).has_value();
  if (!allocates &&
      drainFloor(prob, chaining, span.drain) >= span.drainOf(prob))
    return success();

  // Horizon: the whole region laid out end to end (each op after the previous
  // one's end, its occupancy window, plus a spare cycle), wide enough that
  // every precedence, chain break and reservation is satisfiable.
  int64_t horizon = 0;
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
  DenseMap<Operation *, IntVar> inCycle =
      addChaining(model, prob, startVars, chaining, opts.regFloor);

  // An op occupies one instance of every unit it links to for its whole window,
  // so a cumulative constraint per resource matches `verifyOccupancy`. A
  // multi-unit op contributes the same window to each.
  auto cumulativeOn = [&](Problem::ResourceType rsrc, LinearExpr capacity) {
    CumulativeConstraint cumulative = model.AddCumulative(std::move(capacity));
    for (Operation *op : ops)
      if (prob.usesResource(op, rsrc))
        cumulative.AddDemand(model.NewFixedSizeIntervalVar(
                                 startVars.at(op), prob.getResourceCycles(op)),
                             prob.getResourceDemand(op));
  };
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (unsigned limit = prob.getLimit(rsrc).value_or(0))
      cumulativeOn(rsrc, limit);

  // An allocatable operator takes the same shape, with the count being decided
  // as the capacity. Occupancy windows on a line form an interval graph, so a
  // capacity is an assignment: `N` units suffice when no cycle needs more.
  SmallVector<AllocationVar> allocs =
      allocationVars(model, prob, /*ii=*/0, /*hint=*/true, cycleTime);
  for (const AllocationVar &alloc : allocs)
    cumulativeOn(alloc.rsrc, alloc.units);
  addAllocationHeadroom(model, prob, startVars, inCycle, allocs, cycleTime,
                        opts.regFloor);

  // What the region is charged, bounded by what the heuristic already reached.
  int64_t heuristicDrain = span.drainOf(prob);
  assert(heuristicDrain <= horizon &&
         "the horizon must cover the schedule the heuristic just found, or "
         "capping the drain variable at it cuts that schedule out and the "
         "solve comes back INFEASIBLE against a model that has one");
  IntVar drain =
      drainVariable(model, startVars, span.drain, horizon, heuristicDrain);
  minimizeCost(model, drain, orderedStarts, span, startVars, allocs, /*ii=*/0,
               horizon);

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts));
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    reportUnsolved(prob, response, opts.budget);
    applyDemandAllocation(prob, /*ii=*/0, cycleTime);
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
  return finishSchedule(prob, cycleTime, opts.regFloor);
}

//===----------------------------------------------------------------------===//
// The cyclic solve: a branch and bound over initiation intervals.
//===----------------------------------------------------------------------===//

namespace {

/// What one fixed-II solve settled. `Infeasible` is a proof that the
/// initiation interval admits no schedule; `Exhausted` is the solver giving
/// up, which proves nothing.
enum class ModuloOutcome { Scheduled, Infeasible, Exhausted };

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
                        const Chaining &chaining, float cycleTime,
                        const SpanObjective &span, const SchedulerOptions &opts,
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

  // Precedence. An edge spanning `distance` iterations is relaxed by one II
  // per iteration it spans, matching the cyclic constraint row `buildTableau`
  // emits; a chain-breaking edge is intra-iteration and carries no II term.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      int64_t separation =
          prob.latencyOf(src) -
          static_cast<int64_t>(ii) * prob.getDistance(dep).value_or(0);
      model.AddLessOrEqual(startVars.at(src) + separation,
                           startVars.at(dep.getDestination()));
    }
  DenseMap<Operation *, IntVar> inCycle =
      addChaining(model, prob, startVars, chaining, opts.regFloor);

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
      auto held = static_cast<int64_t>(prob.getResourceDemand(op));
      used += static_cast<int64_t>(occ / ii) * held;
      const SmallVector<BoolVar> &slots = slotsOf.at(op);
      for (unsigned k = 0, partial = occ % ii; k < partial; ++k)
        used += LinearExpr::Term(slots[(slot + ii - k) % ii], held);
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
  SmallVector<AllocationVar> allocs =
      allocationVars(model, prob, ii, hint, cycleTime);
  for (const AllocationVar &alloc : allocs) {
    int64_t total = 0;
    for (Operation *op : ops)
      if (prob.usesResource(op, alloc.rsrc))
        total += prob.getResourceCycles(op);
    model.AddGreaterOrEqual(alloc.units, (total + ii - 1) / ii);
    for (unsigned slot = 0; slot < ii; ++slot)
      model.AddLessOrEqual(usesIn(alloc.rsrc, slot), alloc.units);
  }
  addAllocationHeadroom(model, prob, startVars, inCycle, allocs, cycleTime,
                        opts.regFloor);

  // `(trip - 1) * ii` is constant at a fixed II, so minimizing the span here is
  // minimizing the drain; the outer search carries the II term. With no span to
  // compose, the anchor's start time takes the primary slot instead.
  std::optional<IntVar> drainVar;
  if (span.trip)
    drainVar = drainVariable(model, startVars, span.drain, horizon, drainBound);
  minimizeCost(model, drainVar.value_or(orderedStarts[anchorIndex]),
               orderedStarts, span, startVars, allocs, ii, horizon);

  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts));
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
/// The search cannot stop at the first feasible II: what the region is charged
/// is `(trip - 1) * ii + drain`, and a larger II can still win with a shorter
/// drain. It keeps the best span seen and cuts once an interval's II term
/// alone already reaches it.
LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  SimplexWarmStart warm;
  if (failed(mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, opts.regFloor,
                                         minII, &warm)))
    return failure();

  unsigned greedyII = warm.placed ? *prob.getInitiationInterval() : 0;
  assert((!warm.placed || greedyII >= warm.lowerBoundII) &&
         "placement only ever grows the II");

  // The heuristic ran the pre-pass whichever form this takes, so the schedule
  // this falls back to meets the period either way.
  Chaining chaining = chainingFor(prob, cycleTime, opts.regFloor,
                                  opts.kind == SchedulerKind::ExactChaining);

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
    // Cut: this interval's span already reaches the incumbent's before a
    // single operation is placed, and every later interval is worse. Where an
    // allocation is decided, admit a tie since it can still win on area.
    if (best && iiWeight * ii + floorDrain >= *best + (allocates ? 1 : 0))
      break;
    std::optional<int64_t> drainBound;
    if (best)
      drainBound = *best - iiWeight * ii;

    DenseMap<Operation *, unsigned> starts;
    Allocated decided;
    bool proven = false;
    int64_t drain = 0;
    ModuloOutcome outcome = solveAtII(
        prob, lastOp, chaining, cycleTime, span, opts, drainBound, ii,
        window + ii * contending,
        /*hint=*/warm.placed && ii == greedyII, starts, decided, proven, drain);
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
      auto d = unsupported(Stage::Sched, Code::PlacementFailed,
                           prob.getContainingOp());
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
    applyDemandAllocation(prob, greedyII, cycleTime);
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
  return finishSchedule(prob, cycleTime, opts.regFloor);
}

//===----------------------------------------------------------------------===//
// Exact operator sharing: one bind-time solve per region.
//===----------------------------------------------------------------------===//

/// Deterministic time budget for one region's sharing solve. Small next to a
/// schedule's: the model is a few booleans per same-class unit pair.
static constexpr double kSharingSolveBudget = 10.0;

std::optional<SmallVector<unsigned>>
mlir::allo::solveSharing(SharingProblem &problem, ArrayRef<unsigned> hint,
                         Operation *anchor) {
  auto n = static_cast<unsigned>(problem.units.size());
  llvm::DenseSet<uint64_t> collide;
  for (auto [a, b] : problem.conflicts)
    collide.insert(uint64_t(a) * n + b);
  // Who may fold onto whom: same class, no collision, onto a smaller index
  // only, so a group's representative is its first member.
  SmallVector<SmallVector<unsigned>> cands(n), joiners(n);
  SmallVector<unsigned> assign(n);
  bool foldable = false;
  for (unsigned i = 0; i < n; ++i) {
    assign[i] = i;
    for (unsigned j = 0; j < i; ++j)
      if (problem.units[i].cls == problem.units[j].cls &&
          !collide.contains(uint64_t(j) * n + i)) {
        cands[i].push_back(j);
        joiners[j].push_back(i);
        foldable = true;
      }
  }
  if (!foldable)
    return assign;

  CpModelBuilder model;
  SmallVector<BoolVar> rep(n);            // the unit keeps its own instance
  llvm::DenseMap<uint64_t, BoolVar> join; // j * n + i: unit i runs on unit j
  for (unsigned i = 0; i < n; ++i)
    rep[i] = model.NewBoolVar();
  for (unsigned i = 0; i < n; ++i) {
    SmallVector<BoolVar> choice{rep[i]};
    for (unsigned j : cands[i]) {
      BoolVar x = model.NewBoolVar();
      model.AddImplication(x, rep[j]);
      join[uint64_t(j) * n + i] = x;
      choice.push_back(x);
    }
    model.AddExactlyOne(choice);
  }
  auto lit = [&](unsigned i, unsigned j) {
    return i == j ? rep[i] : join.find(uint64_t(j) * n + i)->second;
  };
  // A colliding pair may not meet through a common representative either.
  for (auto [a, b] : problem.conflicts)
    for (unsigned j : cands[a])
      if (auto x = join.find(uint64_t(j) * n + b); x != join.end())
        model.AddAtMostOne({lit(a, j), x->second});

  // Per potential representative and operand port: the arms its select grew
  // (zero while it shares nothing), with the cone and price read off the
  // port's tables at that count. A port whose candidates all read one held
  // value stays a wire in every fold, so it is skipped whole; the emitter
  // collapses exactly that case.
  int64_t horizon = 0;
  for (SharingProblem::Unit &u : problem.units)
    horizon = std::max(horizon, u.slackPicos);
  SmallVector<IntVar> arrive(n);
  for (unsigned j = 0; j < n; ++j)
    arrive[j] = model.NewIntVar(operations_research::Domain(0, horizon));
  llvm::DenseMap<std::pair<unsigned, unsigned>, IntVar> coneAt; // (host, port)
  // Area dominates; below it, fewer folds win ties, so a free device shares
  // nothing rather than folding at whim.
  int64_t w = n + 1;
  LinearExpr objective;
  for (unsigned i = 0; i < n; ++i)
    objective += LinearExpr::Term(
        rep[i], problem.classes[problem.units[i].cls].instancePrice * w - 1);
  for (unsigned j = 0; j < n; ++j) {
    if (joiners[j].empty())
      continue;
    const SharingProblem::Unit &uj = problem.units[j];
    const SharingProblem::UnitClass &cls = problem.classes[uj.cls];
    BoolVar shared = model.NewBoolVar();
    SmallVector<BoolVar> in;
    for (unsigned i : joiners[j]) {
      in.push_back(lit(i, j));
      model.AddImplication(in.back(), shared);
    }
    model.AddBoolOr(in).OnlyEnforceIf(shared);
    for (unsigned p = 0, e = cls.ports.size(); p < e; ++p) {
      unsigned key = uj.drivers[p];
      if (key && llvm::all_of(joiners[j], [&](unsigned i) {
            return problem.units[i].drivers[p] == key;
          }))
        continue; // one held driver across every candidate: a wire
      int64_t maxArms = 1 + uj.initArms[p];
      LinearExpr arms = LinearExpr::Term(shared, 1 + uj.initArms[p]);
      for (unsigned i : joiners[j]) {
        unsigned add = 1 + problem.units[i].initArms[p];
        arms += LinearExpr::Term(lit(i, j), add);
        maxArms += add;
      }
      IntVar armCount =
          model.NewIntVar(operations_research::Domain(0, maxArms));
      model.AddEquality(armCount, arms);
      const SharingProblem::Port &port = cls.ports[p];
      std::vector<int64_t> cones(port.conePicos.begin(),
                                 port.conePicos.begin() + maxArms + 1);
      if (int64_t top = *llvm::max_element(cones)) {
        IntVar c = model.NewIntVar(operations_research::Domain(0, top));
        model.AddElement(armCount, cones, c);
        coneAt.try_emplace({j, p}, c);
        model.AddGreaterOrEqual(arrive[j], c);
      }
      std::vector<int64_t> prices(port.muxPrice.begin(),
                                  port.muxPrice.begin() + maxArms + 1);
      if (int64_t top = *llvm::max_element(prices)) {
        IntVar price = model.NewIntVar(operations_research::Domain(0, top));
        model.AddElement(armCount, prices, price);
        objective += LinearExpr::Term(price, w);
      }
    }
  }
  model.Minimize(objective);

  // The gate's recursion (`AddedDelay`), over bins instead of built sources:
  // a producer's cone arrives through the select of the port it drives, and
  // every member's slack must hold its whole bin's cone.
  for (unsigned y = 0; y < n; ++y)
    for (auto [port, p] : problem.units[y].preds) {
      SmallVector<unsigned> ys(cands[y]);
      ys.push_back(y);
      SmallVector<unsigned> ps(cands[p]);
      ps.push_back(p);
      for (unsigned jy : ys)
        for (unsigned jp : ps) {
          if (jy == jp)
            continue;
          LinearExpr reach = arrive[jp];
          if (auto c = coneAt.find({jy, port}); c != coneAt.end())
            reach += c->second;
          model.AddLessOrEqual(reach, arrive[jy])
              .OnlyEnforceIf({lit(y, jy), lit(p, jp)});
        }
    }
  for (unsigned i = 0; i < n; ++i) {
    model.AddLessOrEqual(arrive[i], problem.units[i].slackPicos)
        .OnlyEnforceIf(rep[i]);
    for (unsigned j : cands[i])
      model.AddLessOrEqual(arrive[j], problem.units[i].slackPicos)
          .OnlyEnforceIf(lit(i, j));
  }

  // The greedy plan seeds the search. It may sit outside this model where its
  // own cone test under-counted (that is what this solve is for), which only
  // costs the hint.
  for (unsigned i = 0; i < n; ++i) {
    model.AddHint(rep[i], hint[i] == i);
    if (hint[i] != i)
      model.AddHint(lit(i, hint[i]), true);
  }

  SchedulerOptions opts;
  opts.budget = kSharingSolveBudget;
  CpSolverResponse response =
      SolveWithParameters(model.Build(), solverParameters(opts));
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    assert(response.status() != CpSolverStatus::INFEASIBLE &&
           response.status() != CpSolverStatus::MODEL_INVALID &&
           "every unit keeping its own instance satisfies this encoding, so "
           "the model is satisfiable by construction");
    warn(Stage::Emit, anchor)
        << "Exact sharing gave up after " << llvm::format("%g", opts.budget)
        << " deterministic time units (solver status "
        << CpSolverStatus_Name(response.status())
        << "); keeping the greedy plan";
    return std::nullopt;
  }
  unsigned folded = 0;
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j : cands[i])
      if (SolutionBooleanValue(response, lit(i, j))) {
        assign[i] = j;
        ++folded;
        break;
      }
  if (response.status() != CpSolverStatus::OPTIMAL)
    warn(Stage::Emit, anchor)
        << "Exact sharing ran out of budget before proving this region's fold "
           "optimal; it shipped the best plan it had found";
  info(Stage::Emit, anchor)
      << "Exact sharing folded " << folded << " of " << n
      << " units away (spent "
      << llvm::format("%.3f", response.deterministic_time()) << " of "
      << llvm::format("%g", opts.budget) << " deterministic time units)";
  return assign;
}
