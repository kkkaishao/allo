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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::allo {

/// The device view. Declared rather than included: `OperatorLibrary` is built
/// on the problems below, so the dependence only runs one way.
class OperatorLibrary;

/// A resource-constrained problem whose shared instances need not be fully
/// pipelined: it carries a per-operation occupancy window, so a synchronous
/// call that holds its callee's instance until the callee is done can be
/// modeled (`populateCallOccupancy`).
///
/// An operation may hold several units at once; `setLinkedResourceTypes` states
/// its complete unit list, and a cycle is feasible for it only where every unit
/// in that list has room across the whole window.
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

  /// How many units of every resource linked to \p op it holds at once. One
  /// unless set: a write to an array held in several copies reaches all of
  /// them, taking a port of each.
  unsigned getResourceDemand(Operation *op) {
    return resourceDemand.lookup(op).value_or(1);
  }
  void setResourceDemand(Operation *op, unsigned units) {
    resourceDemand[op] = units;
  }

  /// The cycles a dependent waits after \p op issues before its result has
  /// arrived: the latency of the operator type \p op is linked to. Every
  /// operation carries one, `populateOperatorTypes` having linked every
  /// operation the problem holds.
  ///
  /// Signed, though the underlying latency is not: every caller composes it
  /// into an expression that subtracts, and an `unsigned` one silently
  /// evaluates `latencyOf(op) - 1` on a combinational operator as 2^32 - 1.
  int64_t latencyOf(Operation *op);

  /// The schedule DEPTH of a SOLVED problem: the cycle by which every operation
  /// has completed. A REPORT only, since a span composes from the drain
  /// instead, which the solver may leave below the depth. A combinational
  /// operation still occupies the cycle it issues in, hence the floor of one.
  int64_t scheduleDepth();

  /// Whether \p op holds at least one unit whose count is capped. An unlimited
  /// link constrains nothing and no reservation tracks it.
  bool holdsLimitedUnit(Operation *op);

  /// Whether \p op holds a unit of \p rsrc.
  bool usesResource(Operation *op, ResourceType rsrc) {
    auto linked = getLinkedResourceTypes(op);
    return linked && llvm::is_contained(*linked, rsrc);
  }

  /// The operations holding a unit of \p rsrc, earliest start first, so a
  /// derived assignment is a function of the schedule rather than of walk
  /// order. Every operation must be scheduled.
  SmallVector<Operation *> usersOf(ResourceType rsrc);

  //===--------------------------------------------------------------------===//
  // Allocatable resources: how many units to build, as opposed to how many
  // exist. An allocatable resource carries no limit, so `holdsLimitedUnit`
  // stays false for it and no reservation table of the heuristic ever sees it.
  //===--------------------------------------------------------------------===//

  /// What one allocatable resource may cost and how many of it may exist.
  struct AllocatableUnit {
    /// The trivial allocation: one unit per operation linked to the resource,
    /// so declaring a resource never makes a problem infeasible.
    unsigned ceiling = 0;
    /// What building `n` instances costs, indexed by `n` over `[0, ceiling]`:
    /// the instances themselves plus the multiplexers that many of them puts
    /// in front of the operations sharing each one. A TABLE and not a
    /// coefficient because a multiplexer's cost per bit rises in plateaus (a
    /// LUT6 absorbs three source/select pairs), so the total is not monotone
    /// in the count and no linear term can stand for it.
    llvm::SmallVector<int64_t> price;
    /// The delay of the select cone in front of the fullest instance at `n`
    /// instances, in ns, indexed like `price`. Zero at the ceiling, where
    /// nothing shares. A solve charges it on every linked operation's sub-cycle
    /// start, so a count only shrinks where the cone fits the slack the same
    /// schedule leaves.
    llvm::SmallVector<double> headroomNs;
  };

  void setAllocatable(ResourceType rsrc, AllocatableUnit unit) {
    allocatable[rsrc] = unit;
  }
  std::optional<AllocatableUnit> getAllocatable(ResourceType rsrc) {
    return allocatable.lookup(rsrc);
  }

  /// How many units a solve decided to build. Absent until one does, leaving
  /// the trivial allocation in force.
  void setAllocation(ResourceType rsrc, unsigned units) {
    allocation[rsrc] = units;
  }
  std::optional<unsigned> getAllocation(ResourceType rsrc) {
    return allocation.lookup(rsrc);
  }

  /// Which instance of its allocatable operator \p op runs on: an index below
  /// `getAllocation` of that operator's resource. Absent until `assignUnits`
  /// derives it, and for every operation on nothing allocatable.
  std::optional<unsigned> getAssignedUnit(Operation *op) {
    return assignedUnit.lookup(op);
  }

  /// Turn every decided count into an assignment of operations to instances,
  /// spread round-robin over all the instances the decision bought rather than
  /// packed into the fewest that would fit.
  ///
  /// Valid at the occupancies an allocation is offered for: cyclic (\p ii > 0)
  /// occupancy is one cycle, so handing out 0, 1, 2, ... within each congruence
  /// class fits the count the model bounded that class by; acyclic (\p ii == 0)
  /// windows form an interval graph, so as many instances as the busiest cycle
  /// needs suffice.
  void assignUnits(unsigned ii);

  /// Whether \p op contends for a resource whose count is being decided.
  bool holdsAllocatableUnit(Operation *op);

  /// The fewest units of \p rsrc the CURRENT schedule needs: the busiest cycle
  /// of its occupancy windows, or busiest congruence class at a non-zero \p ii.
  /// Every operation must be scheduled. This is the count `assignUnits` can
  /// still place, since windows on a line form an interval graph and first fit
  /// in start order colours one exactly.
  ///
  /// The same histogram `verifyOccupancy` compares against a limit, so what a
  /// resource is checked to fit and what it is decided to need are one count.
  unsigned demandFor(ResourceType rsrc, unsigned ii);

  /// Whether \p op contends for anything at all: a capped unit, an allocated
  /// one, or both. This is what needs a congruence class in a modulo model.
  bool contendsForUnit(Operation *op) {
    return holdsLimitedUnit(op) || holdsAllocatableUnit(op);
  }

  /// No two operations assigned to one instance contend for it in the same
  /// cycle, and no instance index exceeds the count decided. Vacuous where no
  /// solve set an allocation.
  LogicalResult verifyAllocation(unsigned ii);

  /// No limited resource is oversubscribed in any cycle, i.e. no resource
  /// demands more than it has. \p ii == 0 checks an acyclic schedule; a
  /// non-zero \p ii checks the windows modulo the initiation interval. Not an
  /// override: the concrete problems below call it from their `verify`.
  LogicalResult verifyOccupancy(unsigned ii);

private:
  OperationProperty<unsigned> resourceCycles;
  OperationProperty<unsigned> resourceDemand;
  ResourceTypeProperty<AllocatableUnit> allocatable;
  ResourceTypeProperty<unsigned> allocation;
  OperationProperty<unsigned> assignedUnit;
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
/// Solving it yields an integer II, integer start times, and per-op sub-cycle
/// start times that respect a target cycle time, under modulo resource
/// constraints.
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
/// straight-line twin of `ChainingModuloProblem`, with no initiation interval
/// and no inter-iteration distance.
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
/// The edges state the period exactly over integer start times: an
/// over-period pair must sit a cycle apart in any schedule (same-cycle
/// endpoints pull every zero-latency intermediate into their cycle), and a
/// schedule separating every such pair leaves no cycle holding an over-period
/// chain. Register placement inside a broken chain stays the solver's.
///
/// Visits operations in topological order and marks one "handled" only once
/// every predecessor's chain map is complete, so a successor never inherits a
/// half-built map.
/// \p regFloor is the earliest sub-cycle time any operation may start at, so
/// every chain begins having already spent it.
/// Every operator fits \p cycleTime on its own (asserted): `runSDCScheduler`
/// derates the period before any problem is built.
LogicalResult computeChainBreaks(
    circt::scheduling::ChainingProblem &prob, float cycleTime, float regFloor,
    SmallVectorImpl<circt::scheduling::Problem::Dependence> &result);

/// `circt::scheduling::computeStartTimesInCycle` with a floor: an operation's
/// sub-cycle start is at least \p regFloor, where CIRCT's takes zero. CIRCT
/// models an ideal register whose result is available at time 0.0 of the cycle
/// it is read in; a real one costs clock-to-out plus routing (0.419 ns on
/// xcu55c, against a 3.333 ns period). A chain from a registered node then
/// costs `max(regFloor, that node's outgoing delay)`.
LogicalResult computeStartTimesInCycle(circt::scheduling::ChainingProblem &prob,
                                       float regFloor);

//===----------------------------------------------------------------------===//
// SDC simplex schedulers.
//
// Fork of CIRCT's `scheduleSimplex` family (implementation in Scheduler.cpp).
// Call these via `solveSchedulingProblem` below or by fully-qualified name
// (`mlir::allo::scheduleSimplex`) to avoid ambiguity with the CIRCT overloads.
//
// Two entries, for the two problems this backend builds. The resource-free and
// non-chaining rungs of CIRCT's family have no caller here: every Allo region
// is solved against a clock period.
//===----------------------------------------------------------------------===//

/// What the SDC heuristic contributes to a solve that is not its own: the II
/// bound it settles before placing anything, and whether its greedy placement
/// reached a schedule.
///
/// Passing one also makes a PLACEMENT failure advisory: the call still succeeds
/// with `placed == false`. A failure in the resource-free LP below placement is
/// not advisory and still fails the call, since that LP is exact: infeasible
/// there means no schedule exists at any II.
struct SimplexWarmStart {
  /// The largest II any bound justifies before resources are placed: the
  /// resource-min II, a loop-carried recurrence, and the pipeline directive's
  /// floor, whichever is largest. Where an exact II search has to start.
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
                              float cycleTime, float regFloor,
                              unsigned minII = 1,
                              SimplexWarmStart *warm = nullptr);
LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime,
                              float regFloor);

//===----------------------------------------------------------------------===//
// What a solve is charged: the span objective.
//===----------------------------------------------------------------------===//

/// One region OUTPUT's contribution to the region's drain: it commits at
/// `start(op) + offset`, plus the linked operator's latency where
/// `plusLatency`. A value handed onward commits when it lands, and WHICH row
/// produces it may itself be a solver decision, so its latency is read off the
/// problem at composition time rather than baked into the offset.
struct DrainTerm {
  Operation *op;
  int64_t offset;
  bool plusLatency = false;
};

/// The drain of a SOLVED problem: the cycle its deepest output commits.
inline int64_t drainOf(circt::scheduling::Problem &problem,
                       ArrayRef<DrainTerm> terms) {
  int64_t drain = 0;
  for (const DrainTerm &term : terms) {
    int64_t at =
        static_cast<int64_t>(*problem.getStartTime(term.op)) + term.offset;
    if (term.plusLatency)
      at += *problem.getLatency(*problem.getLinkedOperatorType(term.op));
    drain = std::max(drain, at);
  }
  return drain;
}

/// One value a region spends a delay register chain on. The chain is as long as
/// its deepest reader needs, and costs what the device charges for a chain of
/// that many stages at this width:
///
/// ```
/// depth(v) = max over reads ( t_read + ii * distance ) - ( t_def + latency )
/// cost(v)  = chainPrice( stages(v), width )
/// ```
///
/// No register is shared between two values (`insertRegister` keys one chain
/// per value and region), which makes this a sum over values that is linear in
/// the schedule rather than a MAXLIVE coupled to an allocation, and so a term
/// an objective can carry directly.
///
/// `stages(v)` is `depth(v)`, except at II > 1 where the emitter folds the
/// chain onto the region's phase: one register holds a tap for a whole
/// interval, so `depth` cycles of delay are built from `ceil(depth / ii)` of
/// them (`EmitContext::foldedChain`).
///
/// `latency` above is the definer's, read live off the model rather than held
/// here: which row realizes the definer may itself be a solver decision.
struct RegisterTerm {
  Operation *def;
  /// Flip-flops one cycle of delay costs.
  int64_t width;
  /// Each reader, and the iteration distance its read spans.
  SmallVector<std::pair<Operation *, int64_t>> reads;
};

/// What a region's span is charged, and so what the exact scheduler minimizes:
/// `(trip - 1) * ii + drain`, the part of `leafSpan` a solve controls, with
/// the region's area decided in a second solve under the span the first one
/// settles.
///
/// The heuristic ignores this and keeps minimizing the anchor's start time, an
/// over-constrained proxy for the quantity actually charged.
struct SpanObjective {
  /// Read one region's charge off \p problem, which needs its operator types
  /// but not a solution: what a term costs is a property of the region, and
  /// only where each term LANDS is a property of the schedule.
  ///
  /// \p results are the values escaping the region, \p carried the counted-loop
  /// body whose block arguments after the induction variable are its iter_args
  /// (null where there is no such recurrence to price: a straight-line span, a
  /// `while`), and \p device what the area terms are priced against.
  SpanObjective(OccupancyProblem &problem, ValueRange results, Block *carried,
                std::optional<int64_t> trip, const OperatorLibrary &device);

  /// The region's outputs.
  SmallVector<DrainTerm> drain;
  /// The values it spends a delay register on.
  SmallVector<RegisterTerm> regs;
  /// The region's trip count, when it is a compile-time constant. Empty leaves
  /// the exact scheduler on the anchor-start objective, which is the right one
  /// wherever no span composes off this solve (a `while`, a dynamic bound) or
  /// wherever iterations do not overlap and the trip multiplies the schedule
  /// DEPTH rather than the drain (`s.pipeline(ii=-1)`).
  std::optional<int64_t> trip;
  /// The device the area terms are priced against. Every one of them costs
  /// what the part spends on it, so a register, a multiplexer and an operator
  /// are comparable; without it the objective would be ranking flip-flops
  /// against DSP slices in a unit neither is measured in.
  const OperatorLibrary &device;

  /// Where this region's deepest output commits in a SOLVED \p problem.
  int64_t drainOf(circt::scheduling::Problem &problem) const {
    return mlir::allo::drainOf(problem, drain);
  }
};

//===----------------------------------------------------------------------===//
// CP-SAT exact schedulers.
//
// Which solver settles the RESOURCE half of a problem. The SDC simplex is exact
// for the difference constraints either way; only the resource placement
// differs, greedy over an MRT there and one constraint program here.
//===----------------------------------------------------------------------===//

enum class SchedulerKind {
  /// The SDC simplex plus greedy modulo / shared-operator placement.
  Heuristic,
  /// CP-SAT over the same problem: exact under the model. The chain breaks
  /// stay the pre-pass's, which state the period exactly (see
  /// `computeChainBreaks`), so only resource placement differs from the
  /// heuristic. Where a device offers several usable rows for one operation,
  /// which row realizes it is also this solver's decision; the heuristic keeps
  /// the library's own pick.
  Exact,
};

/// Whether \p kind solves the resource half with CP-SAT.
inline bool usesExactScheduler(SchedulerKind kind) {
  return kind != SchedulerKind::Heuristic;
}

/// Defaults for one solve. The budget is in OR-Tools deterministic time units
/// (roughly a core-second) and is charged per solve, shared by a solve's span
/// and area passes, so a cyclic search spends it again at every initiation
/// interval it probes. Reproducibility comes from
/// the fixed seed plus the interleaved portfolio `solverParameters` selects
/// above one worker; a solve that exhausts its budget can still differ run to
/// run. The worker count is not only a speed knob: the same deterministic
/// budget buys more search, so a budget-limited region can settle on a
/// different schedule at a different worker count.
inline constexpr double kDefaultSolveBudget = 30.0;
inline constexpr int kDefaultSolveWorkers = 8;
inline constexpr int kDefaultSolveSeed = 0;

/// What the caller asked the scheduler for.
struct SchedulerOptions {
  SchedulerKind kind = SchedulerKind::Heuristic;
  double budget = kDefaultSolveBudget;
  /// Whether to decide how many copies of each operator a region builds
  /// (`populateOperatorAllocation`) rather than leave every operation its own.
  /// Only meaningful alongside a binding that folds them: with the trivial
  /// binding the emitter builds one unit per operation anyway. The heuristic
  /// ignores it. An operation whose realization the exact solver decides
  /// (`selectionCandidates`) is composed differently: a straight-line solve
  /// puts it in the class of whichever row it decides (a shared class), while
  /// a modulo solve leaves it its own instance, where bind-time sharing may
  /// still fold equal decided rows.
  bool allocate = false;
  int workers = kDefaultSolveWorkers;
  int seed = kDefaultSolveSeed;
  /// The fabric's register-to-register floor (ns): the earliest sub-cycle time
  /// any operation may start at. Combinational rows carry their measured delay
  /// less the floor, so a cycle pays it once however many operators chain.
  float regFloor = 0.0f;
};

/// \p name ("heuristic" / "exact") as a kind, or nullopt when it names
/// neither.
std::optional<SchedulerKind> parseSchedulerKind(StringRef name);

/// One region's operator-sharing problem, decided at bind time with the
/// schedule already fixed: which same-class units to fold onto one instance.
/// Numeric throughout, so the emitter hands one over without this header
/// knowing its model. A shared instance grows one select per operand port,
/// with one arm per member plus each member's re-injected recurrence
/// identities, so tables are per port and indexed by arms; a select of one
/// arm is a wire, so indices 0 and 1 are zero.
struct SharingProblem {
  struct Port {
    /// The select at this port's own width, by arms.
    llvm::SmallVector<int64_t> muxPrice;
    /// Its delay in picoseconds, by arms.
    llvm::SmallVector<int64_t> conePicos;
  };
  struct UnitClass {
    /// One instance of the operator, in the device's currency.
    int64_t instancePrice = 0;
    llvm::SmallVector<Port, 0> ports;
  };
  struct Unit {
    unsigned cls = 0; // index into `classes`; only equal ones may fold
    /// The room the schedule left for this operation's whole input cone.
    int64_t slackPicos = 0;
    /// Same-cycle combinational producers, as (port, unit): a producer's cone
    /// arrives through the select of the port it drives.
    llvm::SmallVector<std::pair<unsigned, unsigned>, 2> preds;
    /// Per port: select arms past its own data arm, one per recurrence
    /// identity the operation re-injects there.
    llvm::SmallVector<unsigned, 2> initArms;
    /// Per port: a nonzero key marks a held operand (a wire at any issue
    /// cycle), equal keys naming equal values. A port whose members all carry
    /// one key collapses to that wire and builds no select; 0 marks a
    /// scheduled or carried operand, which never collapses.
    llvm::SmallVector<unsigned, 2> drivers;
  };
  llvm::SmallVector<UnitClass, 0> classes;
  llvm::SmallVector<Unit, 0> units;
  /// Same-class pairs `(i, j)`, `i < j`, that may not share an instance: their
  /// reservations collide.
  llvm::SmallVector<std::pair<unsigned, unsigned>> conflicts;
};

/// Decide the fold exactly: returns, for each unit, the unit it runs on (the
/// smallest member of its group; itself where unshared). Minimizes the
/// modelled area, instances plus multiplexers with fewer folds breaking ties,
/// holding every unit's input cone within `slackPicos` under the recursion the
/// emit gate walks (`AddedDelay`): a bin's select plus everything a same-cycle
/// producer's bin adds. \p hint seeds the search as an incumbent. \p anchor is
/// where diagnostics land. Returns nullopt when the budget expires with
/// nothing usable.
std::optional<SmallVector<unsigned>> solveSharing(SharingProblem &problem,
                                                  ArrayRef<unsigned> hint,
                                                  Operation *anchor);

/// Solve \p prob exactly with CP-SAT, minimizing \p span under the target clock
/// period \p cycleTime.
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

/// Check, solve, and verify \p problem, minimizing the span \p span charges.
/// The chaining modulo variant, with a target-II lower bound (from a pipeline
/// directive): the achieved II is max(\p minII, the natural minimum). \p minII
/// == 1 imposes no additional bound. \p opts selects the resource solver; both
/// paths go through the same `check` and `verify`.
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
                                                opts.regFloor, minII))) {
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
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime,
                                                opts.regFloor))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Reject a kernel the backend cannot schedule at all: an unmodelled memory
/// effect, an unrealizable operator, an illegal channel or partition.
/// Everything here is a property of the input, so it is settled before a
/// single problem is built. Timing is not among the refusals: an operator or
/// address cone past the clock period derates the period at schedule time
/// instead.
LogicalResult runPreScheduleVerification(ModuleOp module, StringRef top);

/// Solve the schedule of every func reachable from \p top, recording it in
/// \p model. The IR is left in affine/scf form; nothing is materialized.
/// \p cycleTime is the RESOLVED target period in ns (the caller applies the
/// default). A target no single operator fits is raised to the least period
/// every device row does, with a warning naming the rows; the period the
/// schedule holds is published as `model.cycleTimeNs` either way.
LogicalResult runSDCScheduler(ModuleOp module, StringRef top, float cycleTime,
                              const SchedulerOptions &opts,
                              ScheduleModel &model);

/// Reify \p model onto the IR as `dcp.*` regions. It runs immediately after the
/// scheduler over the same module, which is what keeps the model's `Operation
/// *` keys valid; it also ADDS to the model, for the condition cones and
/// symbolic bounds it schedules itself.
void runPostScheduleConversion(ModuleOp module, ScheduleModel &model);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
