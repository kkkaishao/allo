/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Fork of CIRCT's linear-programming (SDC) simplex schedulers
// (externals/circt/lib/Scheduling/SimplexSchedulers.cpp). Only the solver lives
// here; the Problem data model and the chaining utilities stay CIRCT's.
//
// A PARTIAL fork, deliberately: only the two rungs this backend solves against
// survive, the chaining modulo and chaining shared-operators ones. Every Allo
// region carries a clock period and a resource model, so CIRCT's resource-free
// and non-chaining schedulers have nothing here to solve.
//
// Portions derived from LLVM/CIRCT, Apache-2.0 WITH LLVM-exception.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "allo-simplex-schedulers"

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;
using namespace mlir::allo::logging;
using mlir::allo::ChainingModuloProblem;
using mlir::allo::ChainingSharedOperatorsProblem;
using mlir::allo::ModuloOccupancyProblem;
using mlir::allo::OccupancyProblem;

using llvm::dbgs;
using llvm::format;

namespace {

/// A dependence circuit that binds the initiation interval. `latency` sums each
/// edge's source latency plus the extra cycle a chain-breaking constraint adds;
/// `distance` sums the iterations the edges span.
struct Recurrence {
  SmallVector<Operation *> ops; // the circuit, in dependence order
  int64_t latency = 0;
  int64_t distance = 0;
  explicit operator bool() const { return !ops.empty(); }
};

/// One-line rendering: the circuit as an arrow chain closing on itself,
/// followed by its two sums. The II it forces is `ceil(latency / distance)`.
static std::string render(const Recurrence &rec) {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (Operation *op : rec.ops)
    os << op->getName().getStringRef() << " -> ";
  os << rec.ops.front()->getName().getStringRef() << " (total latency "
     << rec.latency << " over distance " << rec.distance << ")";
  return s;
}

/// Models a scheduling problem as a lexico-parametric linear program (LP),
/// solved with an extended dual simplex algorithm, per:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// A resource-free ("central") problem's ILP has a totally unimodular
/// constraint matrix, so the plain (non-integer) simplex already returns an
/// integer optimum.
class SimplexSchedulerBase {
protected:
  /// The objective is to minimize the start time of this operation.
  Operation *lastOp;

  /// S is part of a mechanism to assign fixed values to the LP variables.
  int parameterS;

  /// T represents the initiation interval (II). Its minimally-feasible value is
  /// computed by the algorithm.
  int parameterT;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                        ◀───nColumns────▶
  ///           nParameters────┐
  ///                        ◀─┴─▶
  ///                       ┌─────┬───────────┬ ─ ─ ─ ─ ┐
  ///                      ▲│. . .│. . ... . .│    0        ▲
  ///           nObjectives││. . .│. . ... . .│         │   │
  ///                      ▼│. . .│. . ... . .│             │
  ///                       ├─────┼───────────┼ ─ ─ ─ ─ ┤   │
  ///  firstConstraintRow > │. . .│. . ... . .│1            │nRows
  ///                       │. . .│. . ... . .│  1      │   │
  ///                       │. . .│. . ... . .│    1        │
  ///                       │. . .│. . ... . .│      1  │   │
  ///                       │. . .│. . ... . .│        1    ▼
  ///                       └─────┴───────────┴ ─ ─ ─ ─ ┘
  ///       parameter1Column ^
  ///         parameterSColumn ^
  ///           parameterTColumn ^
  ///  firstNonBasicVariableColumn ^
  ///                              ─────────── ──────────
  ///                       nonBasicVariables   basicVariables
  SmallVector<SmallVector<int>> tableau;

  /// During the pivot operation, one column in the elided part of the tableau
  /// is modified; this vector temporarily catches the changes.
  SmallVector<int> implicitBasicVariableColumnVector;

  /// The linear program models the operations' start times as variables, which
  /// we identify here as 0, ..., |ops|-1.
  /// Additionally, for each dependence (precisely, the inequality modeling the
  /// precedence constraint), a slack variable is required; these are identified
  /// as |ops|, ..., |ops|+|deps|-1.
  ///
  /// This vector stores the numeric IDs of non-basic variables. A variable's
  /// index *i* in this vector corresponds to the tableau *column*
  /// `firstNonBasicVariableColumn`+*i*.
  SmallVector<unsigned> nonBasicVariables;

  /// This vector store the numeric IDs of basic variables. A variable's index
  /// *i* in this vector corresponds to the tableau *row*
  /// `firstConstraintRow`+*i*.
  SmallVector<unsigned> basicVariables;

  /// An operation's start time variable id.
  DenseMap<Operation *, unsigned> startTimeVariables;

  /// This vector keeps track of the current locations (i.e. row or column) of
  /// a start time variable in the tableau. We encode column numbers as positive
  /// integers, and row numbers as negative integers. We do not track the slack
  /// variables.
  SmallVector<int> startTimeLocations;

  /// Non-basic variables can be "frozen" to a specific value, which prevents
  /// them from being pivoted into basis again.
  DenseMap<unsigned, unsigned> frozenVariables;

  /// Number of rows in the tableau = |obj| + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = |params| + |ops|.
  unsigned nColumns;

  // Number of objective rows.
  unsigned nObjectives;
  /// All other rows encode linear constraints.
  unsigned &firstConstraintRow = nObjectives;

  // Number of parameters.
  static constexpr unsigned nParameters = 3;
  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameter1Column = 0;
  /// The second column corresponds to the variable-freezing parameter S.
  static constexpr unsigned parameterSColumn = 1;
  /// The third column corresponds to the parameter T, i.e. the current II.
  static constexpr unsigned parameterTColumn = 2;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = nParameters;

  /// Allow subclasses to collect additional constraints that are not part of
  /// the input problem, but should be modeled in the linear problem.
  SmallVector<Problem::Dependence> additionalConstraints;

  virtual Problem &getProblem() = 0;
  /// Iteration distance a dependence spans. The base answers 0 (the acyclic
  /// `distance == 0` special case); the cyclic subclasses override.
  virtual unsigned distanceOf(Problem::Dependence dep);
  /// The dependence circuit that binds the II at \p ii: the constraints are
  /// `t_dst - t_src >= latency(src) + extra - ii*distance`, so a schedule
  /// exists iff no circuit's weights sum positive. A positive circuit forces
  /// `ii >= ceil(latency / distance)`, and one with `distance == 0` can never
  /// be satisfied. Empty when no circuit binds. O(|ops| * |deps|) Bellman-Ford.
  Recurrence bindingRecurrence(unsigned ii);
  /// Report a failed initial solve, naming the recurrence responsible.
  void reportInfeasible();
  virtual LogicalResult checkLastOp();
  /// The objective rows, optimized lexicographically in this order. The second
  /// one is a TIEBREAK: minimizing the last operation's start time leaves a
  /// whole face of optimal solutions, and the emitter builds an operation's
  /// start pulse as `delayValid(regionStart, t)`, one flip-flop per cycle of
  /// `t`, so a slack-bearing node placed late costs registers for no latency.
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  virtual bool fillObjectiveRow(SmallVector<int> &row, unsigned obj);
  virtual void fillConstraintRow(SmallVector<int> &row,
                                 Problem::Dependence dep);
  virtual void fillAdditionalConstraintRow(SmallVector<int> &row,
                                           Problem::Dependence dep);
  void buildTableau();

  int getParametricConstant(unsigned row);
  SmallVector<int> getObjectiveVector(unsigned column);
  std::optional<unsigned> findDualPivotRow();
  std::optional<unsigned> findDualPivotColumn(unsigned pivotRow,
                                              bool allowPositive = false);
  std::optional<unsigned> findPrimalPivotColumn();
  std::optional<unsigned> findPrimalPivotRow(unsigned pivotColumn);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);
  LogicalResult solveTableau();
  LogicalResult restoreDualFeasibility();
  bool isInBasis(unsigned startTimeVariable);
  unsigned freeze(unsigned startTimeVariable, unsigned timeStep);
  void translate(unsigned column, int factor1, int factorS, int factorT);
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);
  void moveBy(unsigned startTimeVariable, unsigned amount);
  unsigned getStartTime(unsigned startTimeVariable);

  /// A restorable copy of the linear program's mutable state. A failed
  /// `solveTableau` cannot be undone by inverting the moves that led to it, so
  /// a speculative transform must keep a copy. Holds everything a pivot, a
  /// `translate` or a `moveBy` touches; `implicitBasicVariableColumnVector` is
  /// pivot scratch and the tableau's dimensions never change.
  struct LPState {
    SmallVector<SmallVector<int>> tableau;
    SmallVector<unsigned> nonBasicVariables, basicVariables;
    SmallVector<int> startTimeLocations;
    DenseMap<unsigned, unsigned> frozenVariables;
    int parameterS, parameterT;
  };
  LPState saveLP();
  void restoreLP(LPState &saved);

  void dumpTableau();

public:
  explicit SimplexSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SimplexSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  CyclicProblem &prob;

protected:
  Problem &getProblem() override { return prob; }
  unsigned distanceOf(Problem::Dependence dep) override {
    return prob.getDistance(dep).value_or(0);
  }
  void fillConstraintRow(SmallVector<int> &row,
                         Problem::Dependence dep) override;

public:
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves acyclic, resource-constrained `OccupancyProblem` with a
// simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  OccupancyProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(OccupancyProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

/// What set the resource-min II: the pool that needed the most cycles, its
/// demand against its per-cycle limit, and one operation holding it, so a
/// diagnostic can point at source rather than at an internal resource key.
struct BindingResource {
  circt::scheduling::Problem::ResourceType rsrc;
  unsigned demand = 0, limit = 0;
  Operation *witness = nullptr;
};

// This class solves the `ModuloOccupancyProblem` using the iterative heuristic
// presented in [2].
class ModuloSimplexScheduler : public CyclicSimplexScheduler {
private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    // Modulo slot -> number of resource instances occupied there. A count (not
    // a set of ops) so a non-pipelined window wider than the II, which wraps
    // and lands in a slot more than once, contributes its true multiplicity.
    using TableType = SmallDenseMap<unsigned, unsigned>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::ResourceType, TableType> tables;
    SmallDenseMap<Problem::ResourceType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep);
    void release(Operation *op);
    void clear() {
      tables.clear();
      reverseTables.clear();
    }
  };

  ModuloOccupancyProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;
  // Lower bound on the II from a pipeline directive. The search only ever grows
  // the II, so the achieved II is max(this, the natural minimum).
  unsigned minII = 1;
  // Set when any limited op occupies its unit for >1 cycle (non-pipelined). The
  // de Dinechin II-increment assumes fully-pipelined (1-slot) reservations, so
  // a problem with blocking ops uses a conservative II-growth path instead.
  bool hasBlockingOps = false;
  // Sum of occupancies over limited ops; the conservative II-growth path
  // must converge within this bound (all ops fit in disjoint windows by then).
  unsigned totalResourceCycles = 0;
  // The largest II any bound justifies before resources are placed: the
  // resource-min II, a loop-carried recurrence, and the pipeline directive's
  // floor, whichever is largest. Greedy placement can only grow the II past it.
  unsigned lowerBoundII = 1;
  // Whether the resource-free solve that settles `lowerBoundII` got that far.
  bool boundSettled = false;
  // Whether a caller places this region itself if the greedy cannot (see
  // `SimplexWarmStart`). It changes only what a placement failure is reported
  // AS, never what the placement does.
  bool placementAdvisory = false;

protected:
  Problem &getProblem() override { return prob; }
  LogicalResult checkLastOp() override;
  void updateMargins();
  LogicalResult scheduleOperation(Operation *n);
  LogicalResult growIIByDeDinechin(Operation *n);
  LogicalResult growIIUniformly(Operation *n);
  /// The fewest cycles one iteration's resource demand can be issued in.
  /// \p binding receives what set it, untouched where nothing does.
  unsigned computeResMinII(BindingResource &binding);

public:
  ModuloSimplexScheduler(ModuloOccupancyProblem &prob, Operation *lastOp,
                         unsigned minII = 1)
      : CyclicSimplexScheduler(prob, lastOp), prob(prob), mrt(*this),
        minII(minII) {}
  LogicalResult schedule() override;
  /// See `lowerBoundII`. Settled before placement, so it is meaningful even
  /// after `schedule` fails, but only once `hasLowerBound` holds.
  unsigned getLowerBoundII() const { return lowerBoundII; }
  bool hasLowerBound() const { return boundSettled; }
  void setPlacementAdvisory() { placementAdvisory = true; }
};

// This class solves the resource-constrained, cyclic, chaining-enabled
// `ChainingModuloProblem` on top of the `ModuloSimplexScheduler`: a pre-pass
// fills the chain-breaking dependences (consumed by `buildTableau`), and a
// post-pass fills the sub-cycle start times.
class ChainingModuloSimplexScheduler : public ModuloSimplexScheduler {
private:
  ChainingModuloProblem &prob;
  float cycleTime;
  float regFloor;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override {
    // Inherited (cyclic) constraint row: latency + II*distance, plus one
    // extra time step to break the combinational chain.
    fillConstraintRow(row, dep);
    row[parameter1Column] -= 1;
  }

public:
  ChainingModuloSimplexScheduler(ChainingModuloProblem &prob, Operation *lastOp,
                                 float cycleTime, float regFloor,
                                 unsigned minII = 1)
      : ModuloSimplexScheduler(prob, lastOp, minII), prob(prob),
        cycleTime(cycleTime), regFloor(regFloor) {}
  LogicalResult schedule() override {
    if (failed(mlir::allo::computeChainBreaks(prob, cycleTime, regFloor,
                                              additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adding pipeline register stages / latency)";
    if (failed(ModuloSimplexScheduler::schedule()))
      return failure();
    return mlir::allo::computeStartTimesInCycle(prob, regFloor);
  }
};

// This class solves the resource-constrained, acyclic, chaining-enabled
// `ChainingSharedOperatorsProblem` on top of the
// `SharedOperatorsSimplexScheduler`. The acyclic mirror of
// `ChainingModuloSimplexScheduler`.
class ChainingSharedOperatorsSimplexScheduler
    : public SharedOperatorsSimplexScheduler {
private:
  ChainingSharedOperatorsProblem &prob;
  float cycleTime;
  float regFloor;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override {
    // Acyclic constraint row (latency only, no II term), plus one extra time
    // step to break the combinational chain.
    fillConstraintRow(row, dep);
    row[parameter1Column] -= 1;
  }

public:
  ChainingSharedOperatorsSimplexScheduler(ChainingSharedOperatorsProblem &prob,
                                          Operation *lastOp, float cycleTime,
                                          float regFloor)
      : SharedOperatorsSimplexScheduler(prob, lastOp), prob(prob),
        cycleTime(cycleTime), regFloor(regFloor) {}
  LogicalResult schedule() override {
    if (failed(mlir::allo::computeChainBreaks(prob, cycleTime, regFloor,
                                              additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adds pipeline register stages / latency)";
    if (failed(SharedOperatorsSimplexScheduler::schedule()))
      return failure();
    return mlir::allo::computeStartTimesInCycle(prob, regFloor);
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Chain breaking
//===----------------------------------------------------------------------===//

LogicalResult mlir::allo::computeStartTimesInCycle(ChainingProblem &prob,
                                                   float regFloor) {
  prob.clearStartTimeInCycle();
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // The floor, not zero: an operand reaches `op` no earlier than its own
    // register can drive it.
    float startTimeInCycle = regFloor;
    unsigned startTime = *prob.getStartTime(op);

    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary()) // carries no value
        continue;
      Operation *pred = dep.getSource();
      auto predStartTimeInCycle = prob.getStartTimeInCycle(pred);
      if (!predStartTimeInCycle)
        return failure(); // a predecessor is still pending

      auto predOpr = *prob.getLinkedOperatorType(pred);
      unsigned predEnd = *prob.getStartTime(pred) + *prob.getLatency(predOpr);
      if (predEnd < startTime)
        continue; // registered a whole step earlier

      // `pred` ends in the cycle `op` starts in. A multi-cycle producer
      // contributes only its outgoing delay, its last register stage being
      // what the cycle starts from.
      float predEndInCycle =
          (*prob.getStartTime(pred) == predEnd ? *predStartTimeInCycle : 0.0f) +
          *prob.getOutgoingDelay(predOpr);
      startTimeInCycle = std::max(predEndInCycle, startTimeInCycle);
    }

    prob.setStartTimeInCycle(op, startTimeInCycle);
    return success();
  });
}

LogicalResult
mlir::allo::computeChainBreaks(ChainingProblem &prob, float cycleTime,
                               float regFloor,
                               SmallVectorImpl<Problem::Dependence> &result) {
  // Every operator fits a cycle of its own: `runSDCScheduler` raises the
  // period to the least every row does before any problem is built, so a
  // violation here is an operation the derate walk did not price.
  assert(llvm::all_of(prob.getOperatorTypes(),
                      [&](Problem::OperatorType opr) {
                        return regFloor + *prob.getIncomingDelay(opr) <=
                                   cycleTime &&
                               *prob.getOutgoingDelay(opr) <= cycleTime;
                      }) &&
         "an operator exceeds the derated period; `minSchedulablePeriod` "
         "prices every operation a problem registers");

  // chains[v][u]: the delay arriving at `v` along the longest combinational
  // chain starting at `u`. A key is also the "handled" marker, so nothing is
  // written for an operation until every predecessor of it is complete.
  DenseMap<Operation *, SmallDenseMap<Operation *, float>> chains;

  // Problem order, which is the IR's. `chains` is keyed by pointer, so its
  // iteration order is one of ADDRESSES, and the edges below would otherwise
  // vary between two compiles of one kernel.
  DenseMap<Operation *, unsigned> order;
  for (Operation *op : prob.getOperations())
    order.try_emplace(op, order.size());

  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    for (auto dep : prob.getDependences(op))
      if (dep.isDefUse() && !chains.count(dep.getSource()))
        return failure(); // a predecessor is still pending; retry `op` later

    // `op` is the origin of its own chain, and every chain arriving at it is
    // one of its combinational predecessors' extended by that predecessor. A
    // chain starts at the floor, not at zero: its operands leave a register.
    chains[op][op] = regFloor;
    for (auto dep : prob.getDependences(op)) {
      if (!dep.isDefUse()) // an auxiliary edge transports no value
        continue;
      Operation *pred = dep.getSource();
      auto predOpr = *prob.getLinkedOperatorType(pred);
      float outgoing = *prob.getOutgoingDelay(predOpr);
      if (*prob.getLatency(predOpr) > 0) {
        // Registered: the chain restarts at `pred` carrying its output delay,
        // maxed against any longer chain that also reaches here through `pred`,
        // and against the floor, which is `pred`'s own clock-to-out.
        chains[op][pred] =
            std::max(chains[op][pred], std::max(regFloor, outgoing));
        continue;
      }
      for (auto [origin, delay] : chains[pred])
        chains[op][origin] = std::max(delay + outgoing, chains[op][origin]);
    }

    // Break every chain `op` cannot be appended to within the period. Erasing
    // it here is what keeps `op`'s successors from inheriting a chain the edge
    // has just cut.
    float incoming = *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
    SmallVector<Operation *, 4> tooLong;
    for (auto [origin, delay] : chains[op])
      if (delay + incoming > cycleTime)
        tooLong.push_back(origin);
    llvm::sort(tooLong, [&](Operation *a, Operation *b) {
      return order.at(a) < order.at(b);
    });
    for (Operation *origin : tooLong) {
      result.emplace_back(origin, op);
      chains[op].erase(origin);
    }
    return success();
  });
}

//===----------------------------------------------------------------------===//
// SimplexSchedulerBase
//===----------------------------------------------------------------------===//

unsigned SimplexSchedulerBase::distanceOf(Problem::Dependence) { return 0; }

Recurrence SimplexSchedulerBase::bindingRecurrence(unsigned ii) {
  auto &prob = getProblem();
  DenseMap<Operation *, unsigned> index;
  SmallVector<Operation *> nodes;
  for (auto *op : prob.getOperations()) {
    index[op] = nodes.size();
    nodes.push_back(op);
  }

  // One edge per constraint row `buildTableau` would emit, carrying the same
  // latency / distance / chain-break terms.
  struct Edge {
    unsigned src, dst;
    int64_t latency, distance;
  };
  SmallVector<Edge> edges;
  auto weightOf = [&](const Edge &e) {
    return e.latency - static_cast<int64_t>(ii) * e.distance;
  };
  auto addEdge = [&](Problem::Dependence dep, int extra) {
    auto srcIt = index.find(dep.getSource());
    auto dstIt = index.find(dep.getDestination());
    if (srcIt == index.end() || dstIt == index.end())
      return;
    int64_t latency =
        *prob.getLatency(*prob.getLinkedOperatorType(dep.getSource())) + extra;
    edges.push_back({srcIt->second, dstIt->second, latency, distanceOf(dep)});
  };
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      addEdge(dep, /*extra=*/0);
  // A chain-breaking constraint costs one extra time step (see the
  // `fillAdditionalConstraintRow` overrides).
  for (auto &dep : additionalConstraints)
    addEdge(dep, /*extra=*/1);

  // Bellman-Ford for a positive circuit, every node a source (`dist` starts at
  // zero) so a circuit anywhere in the graph is found. Settling early means
  // there is none.
  SmallVector<int64_t> dist(nodes.size(), 0);
  SmallVector<int> pred(nodes.size(), -1), predEdge(nodes.size(), -1);
  int relaxed = -1;
  for (unsigned round = 0; round < nodes.size(); ++round) {
    relaxed = -1;
    for (auto [e, edge] : llvm::enumerate(edges))
      if (dist[edge.src] + weightOf(edge) > dist[edge.dst]) {
        dist[edge.dst] = dist[edge.src] + weightOf(edge);
        pred[edge.dst] = edge.src;
        predEdge[edge.dst] = e;
        relaxed = edge.dst;
      }
    if (relaxed < 0)
      return {}; // settled: every circuit's weights sum non-positive
  }

  // A node still relaxing after |ops| rounds is reachable from a positive
  // circuit; |ops| predecessor steps land inside the circuit itself.
  unsigned v = relaxed;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (pred[v] < 0)
      return {};
    v = pred[v];
  }
  Recurrence rec;
  for (unsigned u = v;;) {
    rec.ops.push_back(nodes[u]);
    const Edge &in = edges[predEdge[u]];
    rec.latency += in.latency;
    rec.distance += in.distance;
    u = pred[u];
    if (u == v)
      break;
  }
  std::reverse(rec.ops.begin(), rec.ops.end());
  return rec;
}

void SimplexSchedulerBase::reportInfeasible() {
  auto &prob = getProblem();
  // The initial solve grows the II freely, so failing it means no II works:
  // some circuit carries positive latency over zero distance. Search at an II
  // large enough that any distance-carrying circuit is comfortably negative.
  unsigned bigII = 1 + additionalConstraints.size();
  for (auto *op : prob.getOperations())
    if (auto opr = prob.getLinkedOperatorType(op))
      bigII += prob.getLatency(*opr).value_or(0);
  Recurrence rec = bindingRecurrence(bigII);
  auto diag =
      error(Stage::Sched, Code::DependenceInfeasible, prob.getContainingOp());
  if (!rec) {
    // No circuit binds, so the infeasibility comes from the constraints layered
    // on top of the dependences (a fixed start time, a resource reservation).
    diag << "Problem is infeasible: no dependence recurrence explains it, so a "
            "fixed start time or a resource reservation does";
    return;
  }
  diag << "Problem is infeasible: the dependence cycle " << render(rec)
       << " must complete within one iteration, but takes " << rec.latency
       << " cycle(s); break it with a loop-carried value (an iter-arg), a "
          "faster operator, or an allo.assume.nodep hint if the dependence is "
          "spurious";
}

LogicalResult SimplexSchedulerBase::checkLastOp() {
  auto &prob = getProblem();
  if (!prob.hasOperation(lastOp)) {
    assert(false && "the scheduling problem does not include its last "
                    "operation; ProblemBuilder constructs both, so no input "
                    "can reach this");
    return failure();
  }
  return success();
}

bool SimplexSchedulerBase::fillObjectiveRow(SmallVector<int> &row,
                                            unsigned obj) {
  switch (obj) {
  case OBJ_LATENCY:
    // Minimize start time of user-specified last operation.
    row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
    return true;
  case OBJ_AXAP:
    // Minimize sum of start times of all-but-the-last operation.
    for (auto *op : getProblem().getOperations())
      if (op != lastOp)
        row[startTimeLocations[startTimeVariables[op]]] = 1;
    return false;
  default:
    llvm_unreachable("Unsupported objective requested");
  }
}

void SimplexSchedulerBase::fillConstraintRow(SmallVector<int> &row,
                                             Problem::Dependence dep) {
  auto &prob = getProblem();
  auto *src = dep.getSource();
  auto *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  row[parameter1Column] = -latency; // note the negation
  if (src != dst) {                 // coefficients zero out for self-arcs.
    row[startTimeLocations[startTimeVariables[src]]] = 1;
    row[startTimeLocations[startTimeVariables[dst]]] = -1;
  }
}

void SimplexSchedulerBase::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  // Handling is subclass-specific, so do nothing by default.
  (void)row;
  (void)dep;
}

void SimplexSchedulerBase::buildTableau() {
  auto &prob = getProblem();

  // The initial tableau is constructed so that operations' start time variables
  // are out of basis, whereas all slack variables are in basis. We will number
  // them accordingly.
  unsigned var = 0;
  for (auto *op : prob.getOperations()) {
    nonBasicVariables.push_back(var);
    startTimeVariables[op] = var;
    startTimeLocations.push_back(firstNonBasicVariableColumn + var);
    ++var;
  }

  // one column for each parameter (1,S,T), and for all operations
  nColumns = nParameters + nonBasicVariables.size();

  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  nObjectives = 0;
  bool hasMoreObjectives;
  do {
    auto &objRowVec = addRow();
    hasMoreObjectives = fillObjectiveRow(objRowVec, nObjectives);
    ++nObjectives;
  } while (hasMoreObjectives);

  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      auto &consRowVec = addRow();
      fillConstraintRow(consRowVec, dep);
      basicVariables.push_back(var);
      ++var;
    }
  }
  for (auto &dep : additionalConstraints) {
    auto &consRowVec = addRow();
    fillAdditionalConstraintRow(consRowVec, dep);
    basicVariables.push_back(var);
    ++var;
  }

  // one row per objective + one row per dependence
  nRows = tableau.size();
}

int SimplexSchedulerBase::getParametricConstant(unsigned row) {
  auto &rowVec = tableau[row];
  // Compute the dot-product ~B[row] * u between the constant matrix and the
  // parameter vector.
  return rowVec[parameter1Column] + rowVec[parameterSColumn] * parameterS +
         rowVec[parameterTColumn] * parameterT;
}

SmallVector<int> SimplexSchedulerBase::getObjectiveVector(unsigned column) {
  SmallVector<int> objVec;
  // Extract the column vector C^T[column] from the cost matrix.
  for (unsigned obj = 0; obj < nObjectives; ++obj)
    objVec.push_back(tableau[obj][column]);
  return objVec;
}

std::optional<unsigned> SimplexSchedulerBase::findDualPivotRow() {
  // Find the first row in which the parametric constant is negative.
  for (unsigned row = firstConstraintRow; row < nRows; ++row)
    if (getParametricConstant(row) < 0)
      return row;

  return std::nullopt;
}

std::optional<unsigned>
SimplexSchedulerBase::findDualPivotColumn(unsigned pivotRow,
                                          bool allowPositive) {
  SmallVector<int> maxQuot(nObjectives, std::numeric_limits<int>::min());
  std::optional<unsigned> pivotCol;

  // Among nonzero entries in the constraint matrix (~A part of the tableau),
  // pick the one with the lexicographical maximum (over objective rows) of
  // the quotient tableau[<objective row>][col] / pivotCand.
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    int pivotCand = tableau[pivotRow][col];
    // Only negative candidates bring us closer to the optimal solution.
    // However, when freezing variables to a certain value, we accept that the
    // value of the objective function degrades.
    if (pivotCand < 0 || (allowPositive && pivotCand > 0)) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand * pivotCand == 1);

      SmallVector<int> quot;
      for (unsigned obj = 0; obj < nObjectives; ++obj)
        quot.push_back(tableau[obj][col] / pivotCand);

      if (std::lexicographical_compare(maxQuot.begin(), maxQuot.end(),
                                       quot.begin(), quot.end())) {
        maxQuot = quot;
        pivotCol = col;
      }
    }
  }

  return pivotCol;
}

std::optional<unsigned> SimplexSchedulerBase::findPrimalPivotColumn() {
  // Find the first lexico-negative column in the cost matrix.
  SmallVector<int> zeroVec(nObjectives, 0);
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    auto objVec = getObjectiveVector(col);
    if (std::lexicographical_compare(objVec.begin(), objVec.end(),
                                     zeroVec.begin(), zeroVec.end()))
      return col;
  }

  return std::nullopt;
}

std::optional<unsigned>
SimplexSchedulerBase::findPrimalPivotRow(unsigned pivotColumn) {
  int minQuot = std::numeric_limits<int>::max();
  std::optional<unsigned> pivotRow;

  // Among positive entries in the constraint matrix (~A part of the tableau),
  // pick the one minimizing the quotient parametricConstant(row) / pivotCand.
  for (unsigned row = firstConstraintRow; row < nRows; ++row) {
    int pivotCand = tableau[row][pivotColumn];
    if (pivotCand > 0) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand == 1);
      int quot = getParametricConstant(row) / pivotCand;
      if (quot < minQuot) {
        minQuot = quot;
        pivotRow = row;
      }
    }
  }

  return pivotRow;
}

void SimplexSchedulerBase::multiplyRow(unsigned row, int factor) {
  assert(factor != 0);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[row][col] *= factor;
  // Also multiply the corresponding entry in the temporary column vector.
  implicitBasicVariableColumnVector[row] *= factor;
}

void SimplexSchedulerBase::addMultipleOfRow(unsigned sourceRow, int factor,
                                            unsigned targetRow) {
  assert(factor != 0 && sourceRow != targetRow);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[targetRow][col] += tableau[sourceRow][col] * factor;
  // Again, perform row operation on the temporary column vector as well.
  implicitBasicVariableColumnVector[targetRow] +=
      implicitBasicVariableColumnVector[sourceRow] * factor;
}

/// The pivot operation applies elementary row operations to the tableau in
/// order to make the \p pivotColumn (corresponding to a non-basic variable) a
/// unit vector (only the \p pivotRow'th entry is 1). Then, a basis exchange is
/// performed: the non-basic variable is swapped with the basic variable
/// associated with the pivot row.
void SimplexSchedulerBase::pivot(unsigned pivotRow, unsigned pivotColumn) {
  // The implicit columns are part of an identity matrix.
  implicitBasicVariableColumnVector[pivotRow] = 1;

  int pivotElem = tableau[pivotRow][pivotColumn];
  // The constraint matrix has only {-1, 0, 1} entries by construction.
  assert(pivotElem * pivotElem == 1);
  // Make `tableau[pivotRow][pivotColumn]` := 1
  multiplyRow(pivotRow, 1 / pivotElem);

  for (unsigned row = 0; row < nRows; ++row) {
    if (row == pivotRow)
      continue;

    int elem = tableau[row][pivotColumn];
    if (elem == 0)
      continue; // nothing to do

    // Make `tableau[row][pivotColumn]` := 0.
    addMultipleOfRow(pivotRow, -elem, row);
  }

  // Swap the pivot column with the implicitly constructed column vector.
  // We really only need to copy in one direction here, as the former pivot
  // column is a unit vector, which is not stored explicitly.
  for (unsigned row = 0; row < nRows; ++row) {
    tableau[row][pivotColumn] = implicitBasicVariableColumnVector[row];
    implicitBasicVariableColumnVector[row] = 0; // Reset for next pivot step.
  }

  unsigned &nonBasicVar =
      nonBasicVariables[pivotColumn - firstNonBasicVariableColumn];
  unsigned &basicVar = basicVariables[pivotRow - firstConstraintRow];

  // Keep track of where start time variables are; ignore slack variables.
  if (nonBasicVar < startTimeLocations.size())
    startTimeLocations[nonBasicVar] = -pivotRow; // ...going into basis.
  if (basicVar < startTimeLocations.size())
    startTimeLocations[basicVar] = pivotColumn; // ...going out of basis.

  std::swap(nonBasicVar, basicVar);
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  // "Solving" technically means performing dual pivot steps until primal
  // feasibility is reached, i.e. the parametric constants in all constraint
  // rows are non-negative.
  while (auto pivotRow = findDualPivotRow()) {
    if (auto pivotCol = findDualPivotColumn(*pivotRow)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // No pivot column: infeasible unless the parameterT entry is positive,
    // which lets growing II rescue it, but only when parameterS == 0 (initial
    // solves). scheduleAt sets parameterS != 0 and instead fails, rolling back.
    int entry1Col = tableau[*pivotRow][parameter1Column];
    int entryTCol = tableau[*pivotRow][parameterTColumn];
    if (parameterS == 0 && entryTCol > 0) {
      // The negation of `entry1Col` is not in the paper, likely an oversight:
      // it is always negative here (else this would not be a valid pivot row),
      // so omitting the negation would make the new II negative.
      assert(entry1Col < 0);
      int newParameterT = (-entry1Col - 1) / entryTCol + 1;
      if (newParameterT > parameterT) {
        // Name the circuit that forces the bump. The search is
        // O(|ops| * |deps|), so only run it when the message will be printed.
        auto diag = info(Stage::Sched, getProblem().getContainingOp());
        diag << "II=" << parameterT
             << " is not achievable: a loop-carried recurrence requires II>="
             << newParameterT << ", increasing II to " << newParameterT;
        if (Recurrence rec = bindingRecurrence(parameterT))
          diag << "; the binding recurrence is " << render(rec);
        parameterT = newParameterT;
        continue;
      }
    }

    // Otherwise, the linear program is infeasible.
    return failure();
  }

  // Optimal solution found!
  return success();
}

LogicalResult SimplexSchedulerBase::restoreDualFeasibility() {
  // Dual feasibility requires all columns in the cost matrix to be
  // non-lexico-negative. Changing the order of the objective rows can violate
  // that; primal pivot steps restore it.
  while (auto pivotCol = findPrimalPivotColumn()) {
    if (auto pivotRow = findPrimalPivotRow(*pivotCol)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // Otherwise, the linear program is unbounded.
    return failure();
  }

  // Optimal solution found!
  return success();
}

bool SimplexSchedulerBase::isInBasis(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());
  int loc = startTimeLocations[startTimeVariable];
  if (-loc >= (int)firstConstraintRow)
    return true;
  if (loc >= (int)firstNonBasicVariableColumn)
    return false;
  llvm_unreachable("Invalid variable location");
}

unsigned SimplexSchedulerBase::freeze(unsigned startTimeVariable,
                                      unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  frozenVariables[startTimeVariable] = timeStep;

  if (!isInBasis(startTimeVariable))
    // That's all for non-basic variables.
    return startTimeLocations[startTimeVariable];

  // We need to pivot this variable one out of basis.
  unsigned pivotRow = -startTimeLocations[startTimeVariable];

  // Here, positive pivot elements can be considered as well, hence finding a
  // suitable column should not fail.
  auto pivotCol = findDualPivotColumn(pivotRow, /* allowPositive= */ true);
  assert(pivotCol);
  pivot(pivotRow, *pivotCol);

  // After the exchange, `startTimeVariable` is represented by `pivotCol`.
  return *pivotCol;
}

void SimplexSchedulerBase::translate(unsigned column, int factor1, int factorS,
                                     int factorT) {
  for (unsigned row = 0; row < nRows; ++row) {
    auto &rowVec = tableau[row];
    int elem = rowVec[column];
    if (elem == 0)
      continue;

    rowVec[parameter1Column] += -elem * factor1;
    rowVec[parameterSColumn] += -elem * factorS;
    rowVec[parameterTColumn] += -elem * factorT;
  }
}

LogicalResult SimplexSchedulerBase::scheduleAt(unsigned startTimeVariable,
                                               unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  unsigned frozenCol = freeze(startTimeVariable, timeStep);
  translate(frozenCol, /* factor1= */ 0, /* factorS= */ 1, /* factorT= */ 0);

  // Temporarily set S to the desired value, and attempt to solve.
  parameterS = timeStep;
  auto solved = solveTableau();
  parameterS = 0;

  if (failed(solved)) {
    // The LP is infeasible with the new constraint. Other values of S could
    // be tried, but instead this rolls back and signals failure to the driver.
    translate(frozenCol, /* factor1= */ 0, /* factorS= */ -1, /* factorT= */ 0);
    frozenVariables.erase(startTimeVariable);
    auto solvedAfterRollback = solveTableau();
    assert(succeeded(solvedAfterRollback));
    (void)solvedAfterRollback;
    return failure();
  }

  // Zero the S-column again via factor1=timeStep, factorS=1 (negating the
  // factors, which isn't in the paper's text but is implied by its example).
  // This doesn't change the parametric constants, so no re-solve is needed.
  translate(parameterSColumn, /* factor1= */ -timeStep, /* factorS= */ 1,
            /* factorT= */ 0);

  return success();
}

void SimplexSchedulerBase::moveBy(unsigned startTimeVariable, unsigned amount) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(frozenVariables.count(startTimeVariable));

  frozenVariables[startTimeVariable] += amount;

  // Translate by the desired amount; solving to restore primal feasibility is
  // deferred to the caller, which typically batch-moves several operations
  // (an intermediate solve could see a still-infeasible tableau).
  translate(startTimeLocations[startTimeVariable], /* factor1= */ amount,
            /* factorS= */ 0, /* factorT= */ 0);
}

SimplexSchedulerBase::LPState SimplexSchedulerBase::saveLP() {
  return {
      tableau,         nonBasicVariables, basicVariables, startTimeLocations,
      frozenVariables, parameterS,        parameterT};
}

void SimplexSchedulerBase::restoreLP(LPState &saved) {
  tableau = std::move(saved.tableau);
  nonBasicVariables = std::move(saved.nonBasicVariables);
  basicVariables = std::move(saved.basicVariables);
  startTimeLocations = std::move(saved.startTimeLocations);
  frozenVariables = std::move(saved.frozenVariables);
  parameterS = saved.parameterS;
  parameterT = saved.parameterT;
}

unsigned SimplexSchedulerBase::getStartTime(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());

  if (!isInBasis(startTimeVariable))
    // Non-basic variables that are not already fixed to a specific time step
    // are 0 at the end of the simplex algorithm.
    return frozenVariables.lookup(startTimeVariable);

  // For a variable in basis, look up the solution in the tableau.
  return getParametricConstant(-startTimeLocations[startTimeVariable]);
}

void SimplexSchedulerBase::dumpTableau() {
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  for (unsigned i = 0; i < nRows; ++i) {
    if (i == firstConstraintRow) {
      for (unsigned j = 0; j < nColumns; ++j) {
        if (j == firstNonBasicVariableColumn)
          dbgs() << "-+";
        dbgs() << "----";
      }
      dbgs() << '\n';
    }
    for (unsigned j = 0; j < nColumns; ++j) {
      if (j == firstNonBasicVariableColumn)
        dbgs() << " |";
      dbgs() << format(" %3d", tableau[i][j]);
    }
    if (i >= firstConstraintRow)
      dbgs() << format(" |< %2d", basicVariables[i - firstConstraintRow]);
    dbgs() << '\n';
  }
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  dbgs() << format(" %3d %3d %3d | ", 1, parameterS, parameterT);
  for (unsigned j = firstNonBasicVariableColumn; j < nColumns; ++j)
    dbgs() << format(" %2d^",
                     nonBasicVariables[j - firstNonBasicVariableColumn]);
  dbgs() << '\n';
}

//===----------------------------------------------------------------------===//
// CyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void CyclicSimplexScheduler::fillConstraintRow(SmallVector<int> &row,
                                               Problem::Dependence dep) {
  SimplexSchedulerBase::fillConstraintRow(row, dep);
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

LogicalResult CyclicSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau())) {
    reportInfeasible();
    return failure();
  }

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Optimal solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

static bool isLimited(Operation *op, SharedOperatorsProblem &prob) {
  auto maybeRsrcs = prob.getLinkedResourceTypes(op);
  if (!maybeRsrcs)
    return false;
  return llvm::any_of(*maybeRsrcs, [&](Problem::ResourceType rsrc) {
    return prob.getLimit(rsrc).value_or(0) > 0;
  });
}

/// The limited units \p op holds, in link order. An operation takes all of them
/// at its start time and releases them together, so a cycle is feasible for it
/// only if every one has room. An unlimited link is dropped: it constrains
/// nothing, so no reservation table tracks it.
static SmallVector<Problem::ResourceType>
limitedUnits(SharedOperatorsProblem &prob, Operation *op) {
  auto maybeRsrcs = prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "operation must have linked resource types");
  SmallVector<Problem::ResourceType> units;
  for (Problem::ResourceType rsrc : *maybeRsrcs)
    if (prob.getLimit(rsrc).value_or(0) > 0)
      units.push_back(rsrc);
  return units;
}

LogicalResult SharedOperatorsSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau())) {
    reportInfeasible();
    return failure();
  }

  LLVM_DEBUG(dbgs() << "After solving resource-free problem:\n"; dumpTableau());

  // Heuristic phase: greedily fix start times for shared-operator ops within
  // allocation limits, re-solving the LP with each added constraint. Each solve
  // is optimal given prior fixes; overall optimality is not guaranteed.

  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Placement order: earliest first, then the largest reservation first among
  // operations starting at the same time. Earliest-first is a topological
  // order, which keeps the acyclic problem feasible under pinning; the scan
  // below is first fit over rectangles, which needs largest-first to behave.
  //
  // Slack is not available as a further tie-break here: an ALAP would maximize
  // the start times, and with dependences the only rows in this tableau an
  // operation without an outgoing one (any store) is unbounded above.
  auto rectangle = [&](Operation *op) {
    return prob.getResourceCycles(op) * prob.getResourceDemand(op);
  };
  llvm::stable_sort(limitedOps, [&](Operation *a, Operation *b) {
    unsigned ta = getStartTime(startTimeVariables[a]);
    unsigned tb = getStartTime(startTimeVariables[b]);
    if (ta != tb)
      return ta < tb;
    return rectangle(a) > rectangle(b);
  });

  // Store the number of operations using a resource type in a particular time
  // step.
  SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    SmallVector<Problem::ResourceType> units = limitedUnits(prob, op);
    assert(!units.empty() && "a limited operation holds a limited unit");

    // Find the first time step (from the current start time) where every unit
    // the op holds is free for its whole occupancy window (occ consecutive
    // cycles; occ == 1 when pipelined).
    unsigned occ = prob.getResourceCycles(op);
    unsigned slots = prob.getResourceDemand(op);
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    auto hasRoom = [&](unsigned t) {
      for (Problem::ResourceType rsrc : units) {
        unsigned limit = *prob.getLimit(rsrc);
        for (unsigned i = 0; i < occ; ++i)
          if (reservationTable[rsrc].lookup(t + i) + slots > limit)
            return false;
      }
      return true;
    };
    while (!hasRoom(candTime))
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the use of every unit across the occupancy window.
    for (Problem::ResourceType rsrc : units)
      for (unsigned i = 0; i < occ; ++i)
        reservationTable[rsrc][candTime + i] += slots;

    LLVM_DEBUG(dbgs() << "After scheduling " << startTimeVar
                      << " to t=" << candTime << ":\n";
               dumpTableau());
  }

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Feasible solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ModuloSimplexScheduler::checkLastOp() {
  if (!prob.hasOperation(lastOp)) {
    assert(false && "the scheduling problem does not include its last "
                    "operation; ProblemBuilder constructs both, so no input "
                    "can reach this");
    return failure();
  }

  // Determine which operations have no outgoing *intra*-iteration dependences.
  auto &ops = prob.getOperations();
  DenseSet<Operation *> sinks(ops.begin(), ops.end());
  for (auto *op : ops)
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());

  if (!sinks.contains(lastOp)) {
    assert(false && "the problem's last operation is not a sink; "
                    "ProblemBuilder anchors it, so no input can reach this");
    return failure();
  }
  if (sinks.size() > 1) {
    assert(false && "the problem has several sinks; ProblemBuilder anchors "
                    "exactly one, so no input can reach this");
    return failure();
  }

  return success();
}

LogicalResult ModuloSimplexScheduler::MRT::enter(Operation *op,
                                                 unsigned timeStep) {
  SmallVector<Problem::ResourceType> units = limitedUnits(sched.prob, op);
  assert(!units.empty() && "a limited operation holds a limited unit");

  // A non-pipelined op occupies `occ` consecutive modulo slots; a window wider
  // than II wraps, hitting one slot twice, which a per-slot set would hide. The
  // window is the same on every unit, all taken at the op's start time.
  unsigned occ = sched.prob.getResourceCycles(op);
  unsigned slots = sched.prob.getResourceDemand(op);
  unsigned base = timeStep % sched.parameterT;
  SmallDenseMap<unsigned, unsigned> want;
  for (unsigned i = 0; i < occ; ++i)
    want[(base + i) % sched.parameterT] += slots;

  // Admit only if every touched slot of every unit fits, then commit to all of
  // them: an op that fits in one unit but not another must leave no partial
  // reservation behind.
  for (Problem::ResourceType rsrc : units) {
    auto &table = tables[rsrc];
    for (const auto &[slot, cnt] : want)
      if (table.lookup(slot) + cnt > *sched.prob.getLimit(rsrc))
        return failure();
  }
  for (Problem::ResourceType rsrc : units) {
    auto &table = tables[rsrc];
    for (const auto &[slot, cnt] : want)
      table[slot] += cnt;
    auto &revTab = reverseTables[rsrc];
    assert(!revTab.count(op));
    revTab[op] = base;
  }
  return success();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  unsigned occ = sched.prob.getResourceCycles(op);
  unsigned slots = sched.prob.getResourceDemand(op);
  // Undo enter's per-slot increments on every unit it reserved, recomputed from
  // the stored base + occ so a wrapped slot is decremented once per lap. The
  // reverse tables record exactly the units entered, unlimited links skipped.
  bool held = false;
  for (auto &[rsrc, revTab] : reverseTables) {
    auto it = revTab.find(op);
    if (it == revTab.end())
      continue;
    auto &table = tables[rsrc];
    for (unsigned i = 0; i < occ; ++i) {
      unsigned &cnt = table[(it->second + i) % sched.parameterT];
      assert(cnt >= slots && "releasing an MRT slot that was never reserved");
      cnt -= slots;
    }
    revTab.erase(it);
    held = true;
  }
  assert(held && "releasing an operation that holds no unit");
  (void)held;
}

void ModuloSimplexScheduler::updateMargins() {
  // Assumes the current secondary objective is "ASAP". Negating the objective
  // row maximizes the sum of start times, yielding the "ALAP" times; negating
  // again restores "ASAP". Both sets of times are stored.
  for (auto *axapTimes : {&alapTimes, &asapTimes}) {
    multiplyRow(OBJ_AXAP, -1);
    // This should not fail for a feasible tableau.
    auto dualFeasRestored = restoreDualFeasibility();
    auto solved = solveTableau();
    assert(succeeded(dualFeasRestored) && succeeded(solved));
    (void)dualFeasRestored, (void)solved;

    for (unsigned stv = 0; stv < startTimeLocations.size(); ++stv)
      (*axapTimes)[stv] = getStartTime(stv);
  }
}

/// Tries `n` at its current time step and the II-1 slots after it, then grows
/// the II if none admit it (`growIIByDeDinechin` first, `growIIUniformly` as
/// fallback). The de Dinechin move assumes fully-pipelined reservations
/// (`hasBlockingOps` bypasses it) and is only SPECULATIVE, so a failed re-solve
/// there is rolled back rather than asserted.
LogicalResult ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  unsigned stvN = startTimeVariables[n];

  // Try the op's current time step in the partial solution and the II-1
  // following ones. A later step may increase the overall latency, but that is
  // preferred over incrementing the II to resolve resource conflicts.
  unsigned stN = getStartTime(stvN);
  unsigned ubN = stN + parameterT - 1;

  LLVM_DEBUG(dbgs() << "Attempting to schedule in [" << stN << ", " << ubN
                    << "]: " << *n << '\n');

  for (unsigned ct = stN; ct <= ubN; ++ct)
    if (succeeded(mrt.enter(n, ct))) {
      auto fixedN = scheduleAt(stvN, ct);
      if (succeeded(fixedN)) {
        LLVM_DEBUG(dbgs() << "Success at t=" << ct << " " << *n << '\n');
        return success();
      }
      // Problem became infeasible with `n` at `ct`, roll back the MRT
      // assignment. Also, no later time can be feasible, so stop the search
      // here.
      mrt.release(n);
      break;
    }

  // `n` does not fit at this II, so the II has to grow.
  if (!hasBlockingOps) {
    LPState savedLP = saveLP();
    auto savedTables = mrt.tables;
    auto savedReverseTables = mrt.reverseTables;
    if (succeeded(growIIByDeDinechin(n)))
      return success();
    restoreLP(savedLP);
    mrt.tables = std::move(savedTables);
    mrt.reverseTables = std::move(savedReverseTables);
    info(Stage::Sched, n) << "The targeted II increment did not carry the "
                             "partial schedule over; growing the II uniformly "
                             "instead, which may cost more than one cycle";
  }
  return growIIUniformly(n);
}

// Grow the II without assuming anything about the partial schedule: every
// scheduled op is shifted by its own `phi`, which keeps it in its modulo slot
// once the II is one larger, then the reservation table is rebuilt and `n` is
// retried. The path for non-pipelined ops, whose multi-slot reservations the
// targeted move cannot express, and the fallback when that move fails.
LogicalResult ModuloSimplexScheduler::growIIUniformly(Operation *n) {
  unsigned stvN = startTimeVariables[n];
  info(Stage::Sched, n) << "II=" << parameterT << " is not achievable for "
                        << n->getName().getStringRef()
                        << ", growing the II uniformly until it fits";
  // Where the compile stops on the default path, and only advice when an exact
  // solver is going to place the region itself.
  auto placementFailed = [&](Operation *at) {
    return placementAdvisory
               ? warn(Stage::Sched, at)
               : unsupported(Stage::Sched, Code::PlacementFailed, at);
  };
  while (true) {
    SmallVector<std::pair<unsigned, unsigned>> phis;
    for (Operation *j : scheduled) {
      unsigned stvJ = startTimeVariables[j];
      phis.push_back({stvJ, getStartTime(stvJ) / parameterT});
    }
    for (auto [stvJ, phiJ] : phis)
      moveBy(stvJ, phiJ);
    ++parameterT;
    // Every op fits in a disjoint window by II=totalResourceCycles; 2x+2
    // leaves slack for cross-window fragmentation. Past that, growth is not
    // converging: a scheduler limit, not a fact about the kernel.
    if (parameterT > 2 * static_cast<int>(totalResourceCycles) + 2 ||
        failed(solveTableau())) {
      auto d = placementFailed(n);
      d << "The modulo scheduler could not place "
        << n->getName().getStringRef()
        << ": resource placement is greedy, and the operations it already "
           "pinned leave this one no feasible cycle, which growing the "
           "initiation interval (tried up to II="
        << parameterT << ") does not undo";
      if (placementAdvisory)
        d << "; the exact scheduler places the region instead";
      else
        d << ". Partitioning the array it contends for, or reducing how many "
             "times one iteration accesses that array, gives the placement "
             "room";
      return failure();
    }

    mrt.clear();
    for (Operation *j : scheduled)
      if (failed(mrt.enter(j, getStartTime(startTimeVariables[j])))) {
        placementFailed(n)
            << "The modulo scheduler could not rebuild its reservation table "
               "after growing the initiation interval to II="
            << parameterT;
        return failure();
      }

    unsigned lo = getStartTime(stvN);
    for (unsigned ct = lo; ct <= lo + parameterT - 1; ++ct)
      if (succeeded(mrt.enter(n, ct))) {
        if (succeeded(scheduleAt(stvN, ct)))
          return success();
        mrt.release(n);
      }
  }
}

// De Dinechin's Theorem 1: move the ops that precede `n` on its own resource
// one modulo slot right, grow the II by one, and place `n` in the slot they
// vacate. Fails, leaving the caller to roll back, when the partial schedule
// does not survive the moves.
LogicalResult ModuloSimplexScheduler::growIIByDeDinechin(Operation *n) {
  // `n` contends for the physical units it holds (a memref port, a unit pool);
  // the reservation table arbitrates those, not the operator type. Every op
  // reaching here is limited, so it holds at least one.
  SmallVector<Problem::ResourceType> unitsN = limitedUnits(prob, n);
  unsigned stvN = startTimeVariables[n];
  unsigned stN = getStartTime(stvN);

  info(Stage::Sched, n) << "II=" << parameterT
                        << " is not achievable: a shared-resource conflict for "
                        << n->getName().getStringRef()
                        << ", trying II=" << parameterT + 1;
  info(Stage::Sched) << "Incrementing II to " << (parameterT + 1)
                     << " to resolve resource conflict for " << *n;

  // Fully-pipelined operators mean incrementing the II by one always suffices
  // here (the paper's general case may need more).
  unsigned phiN = stN / parameterT;
  unsigned tauN = stN % parameterT;

  // Track whether the following moves free an operator instance in the slot
  // the current op wants, so it can stay there.
  unsigned deltaN = 1;

  SmallVector<Operation *> moved;
  for (Operation *j : scheduled) {
    unsigned stvJ = startTimeVariables[j];
    unsigned stJ = getStartTime(stvJ);
    unsigned phiJ = stJ / parameterT;
    unsigned tauJ = stJ % parameterT;
    unsigned deltaJ = 0;

    // `j` stands in `n`'s way only where they hold a unit in common.
    if (llvm::any_of(limitedUnits(prob, j), [&](Problem::ResourceType rsrc) {
          return llvm::is_contained(unitsN, rsrc);
        })) {
      // Resolve conflicts by moving ops contending for a unit `n` also needs
      // (e.g. a load/store pair shares a memref port despite distinct
      // operator types) that are "preceded" (de Dinechin's ≺ relation) right.
      if (tauN < tauJ || (tauN == tauJ && phiN > phiJ) ||
          (tauN == tauJ && phiN == phiJ && stvN < stvJ)) {
        // TODO: Replace the last condition with a proper graph analysis.

        deltaJ = 1;
        moved.push_back(j);
        if (tauN == tauJ)
          deltaN = 0;
      }
    }

    // Move: add `phiJ` to keep `j` in its modulo slot `tauJ` after II grows
    // (stJ + phiJ = phiJ*(parameterT+1) + tauJ), plus `deltaJ` to shift it to
    // a different slot when it conflicts with the op that triggered growth.
    moveBy(stvJ, phiJ + deltaJ);
  }

  // Finally, increment the II and solve to apply the moves.
  ++parameterT;
  if (failed(solveTableau()))
    return failure();

  // Re-enter moved operations into their new slots.
  for (auto *m : moved)
    mrt.release(m);
  for (auto *m : moved)
    if (failed(mrt.enter(m, getStartTime(startTimeVariables[m]))))
      return failure();

  // Finally, schedule the operation. Again, adding `phiN` accounts for the
  // implicit shift caused by incrementing the II.
  if (failed(scheduleAt(stvN, stN + phiN + deltaN)))
    return failure();
  return mrt.enter(n, tauN + deltaN);
}

unsigned ModuloSimplexScheduler::computeResMinII(BindingResource &binding) {
  unsigned resMinII = 1;
  SmallDenseMap<Problem::ResourceType, unsigned> uses;
  SmallDenseMap<Problem::ResourceType, Operation *> witness;
  for (auto *op : prob.getOperations()) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    for (auto rsrc : *maybeRsrcs) {
      if (prob.getLimit(rsrc).value_or(0) > 0) {
        // occupancy: the whole window a non-pipelined unit is held for, times
        // the units the operation holds at once
        uses[rsrc] += prob.getResourceCycles(op) * prob.getResourceDemand(op);
        // The operation list is in a stable order, so the witness a diagnostic
        // points at is deterministic.
        witness.try_emplace(rsrc, op);
      }
    }
  }

  // Integer ceil: enough parallel units to cover total occupancy in one II.
  // (unsigned `a / b` floors, so an explicit integer ceil is needed once
  // limit >= 2.)
  for (auto pair : uses) {
    unsigned limit = *prob.getLimit(pair.first);
    unsigned need = (pair.second + limit - 1) / limit;
    if (need <= resMinII)
      continue;
    resMinII = need;
    binding = {pair.first, pair.second, limit, witness.lookup(pair.first)};
  }

  return resMinII;
}

/// Seeds the II at the larger of the resource-min II and the pipeline
/// directive's floor, then iteratively fixes limited operations to time steps
/// in earliest-first, least-slack-breaks-ties order. That order matters:
/// pinning a consumer caps how late its operands may issue, and once a
/// resource saturates at this II there is no cycle left for the last of them.
LogicalResult ModuloSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  // Seed the II at the resource-min II, but never below the pipeline
  // directive's target; the search only grows it from there.
  BindingResource binding;
  unsigned resMinII = computeResMinII(binding);
  parameterT = std::max(resMinII, minII);
  info(Stage::Sched, prob.getContainingOp())
      << "Initiation interval search seeded at II=" << parameterT
      << " (resource-min II=" << resMinII
      << ", pipeline-directive floor minII=" << minII << ")";
  LLVM_DEBUG(dbgs() << "ResMinII = " << parameterT << " (minII=" << minII
                    << ")\n");
  buildTableau();
  asapTimes.resize(startTimeLocations.size());
  alapTimes.resize(startTimeLocations.size());

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau())) {
    reportInfeasible();
    return failure();
  }
  // The resource-free solve already raises the II to any loop-carried
  // recurrence's minimum, so `parameterT` here is the best lower bound anything
  // downstream can justify.
  lowerBoundII = parameterT;
  boundSettled = true;

  // Report what set the bound, so it can be acted on: banking or replicating an
  // array lowers a port-bound interval, reassociating a reduction lowers a
  // recurrence-bound one.
  if (lowerBoundII > 1) {
    if (lowerBoundII > std::max(resMinII, minII))
      info(Stage::Sched, prob.getContainingOp())
          << "II cannot go below " << lowerBoundII
          << " here: a loop-carried recurrence takes that long to come round";
    else if (resMinII >= minII && binding.witness)
      info(Stage::Sched, binding.witness)
          << "II cannot go below " << resMinII << " here: one iteration takes "
          << binding.demand << " slots of a resource serving " << binding.limit
          << " per cycle. Banking or replicating what this access reaches is "
             "what lowers that bound";
  }

  // Determine which operations are subject to resource constraints, and whether
  // any of them is non-pipelined (occupies its unit for more than one cycle).
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob)) {
      unscheduled.push_back(op);
      unsigned occ = prob.getResourceCycles(op);
      totalResourceCycles += occ;
      if (occ > 1)
        hasBlockingOps = true;
    }

  // Main loop: iteratively fix limited operations to time steps.
  while (!unscheduled.empty()) {
    // ASAP/ALAP margins, refreshed against the operations pinned so far.
    updateMargins();

    // Earliest-first, least slack breaking the tie (see the doc comment above).
    auto priority = [&](Operation *op) {
      unsigned stv = startTimeVariables[op];
      return std::make_pair(asapTimes[stv], alapTimes[stv] - asapTimes[stv]);
    };
    auto *opIt = std::min_element(unscheduled.begin(), unscheduled.end(),
                                  [&](Operation *opA, Operation *opB) {
                                    return priority(opA) < priority(opB);
                                  });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    if (failed(scheduleOperation(op)))
      return failure();
    scheduled.push_back(op);
  }

  // Resource placement is greedy, so an II above the LP's bound may be the
  // problem's real minimum or just what the heuristic cost; nothing here can
  // tell the two apart.
  if (parameterT > static_cast<int>(lowerBoundII))
    warn(Stage::Sched, prob.getContainingOp())
        << "Scheduled at II=" << parameterT
        << " against a lower bound of II=" << lowerBoundII
        << " (resource-min II=" << resMinII
        << "): resource placement is a greedy heuristic, so this gap is not "
           "known to be necessary";

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// OccupancyProblem / ModuloOccupancyProblem (declared in Scheduler.h): CIRCT's
// resource problems with a per-operation occupancy window.
//===----------------------------------------------------------------------===//

LogicalResult OccupancyProblem::checkLatency(Operation *op) {
  // Deliberately NOT SharedOperatorsProblem::checkLatency, which rejects a
  // zero-latency operation on a limited resource. A combinational access holds
  // its port for the cycle it issues in and contends like any other.
  return Problem::checkLatency(op);
}

int64_t OccupancyProblem::latencyOf(Operation *op) {
  std::optional<OperatorType> opr = getLinkedOperatorType(op);
  assert(opr && "an operation the operator model never characterized");
  std::optional<unsigned> latency = getLatency(*opr);
  assert(latency && "an operator type with no latency");
  return *latency;
}

int64_t OccupancyProblem::scheduleDepth() {
  int64_t depth = 1;
  for (Operation *op : getOperations())
    if (std::optional<unsigned> start = getStartTime(op))
      depth = std::max(depth, static_cast<int64_t>(*start) +
                                  std::max<int64_t>(1, latencyOf(op)));
  return depth;
}

bool OccupancyProblem::holdsLimitedUnit(Operation *op) {
  auto linked = getLinkedResourceTypes(op);
  return linked && llvm::any_of(*linked, [&](ResourceType rsrc) {
           return getLimit(rsrc).value_or(0) > 0;
         });
}

bool OccupancyProblem::holdsAllocatableUnit(Operation *op) {
  auto linked = getLinkedResourceTypes(op);
  return linked && llvm::any_of(*linked, [&](ResourceType rsrc) {
           return getAllocatable(rsrc).has_value();
         });
}

SmallVector<Operation *> OccupancyProblem::usersOf(ResourceType rsrc) {
  SmallVector<Operation *> users;
  for (Operation *op : getOperations())
    if (usesResource(op, rsrc))
      users.push_back(op);
  llvm::stable_sort(users, [&](Operation *a, Operation *b) {
    return *getStartTime(a) < *getStartTime(b);
  });
  return users;
}

unsigned OccupancyProblem::demandFor(ResourceType rsrc, unsigned ii) {
  SmallDenseMap<unsigned, unsigned> used;
  unsigned peak = 0;
  for (Operation *op : getOperations()) {
    if (!usesResource(op, rsrc))
      continue;
    unsigned start = *getStartTime(op);
    unsigned slots = getResourceDemand(op);
    for (unsigned k = 0, occ = getResourceCycles(op); k < occ; ++k) {
      unsigned &cnt = used[ii ? (start + k) % ii : start + k];
      cnt += slots;
      peak = std::max(peak, cnt);
    }
  }
  return peak;
}

void OccupancyProblem::assignUnits(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    std::optional<unsigned> units = getAllocation(rsrc);
    if (!units)
      continue;
    SmallVector<Operation *> users = usersOf(rsrc);
    // Both rules round-robin over all the instances rather than packing into
    // the fewest that fit, so the count decided is the count built.
    unsigned cursor = 0;
    if (ii) {
      // Occupancy is one cycle here, so an instance is available iff it is
      // free in the operation's congruence class.
      llvm::DenseSet<std::pair<unsigned, unsigned>> taken;
      for (Operation *op : users) {
        unsigned cls = *getStartTime(op) % ii;
        unsigned k = cursor % *units;
        for (unsigned tried = 1; taken.count({k, cls}) && tried < *units;
             ++tried)
          k = (k + 1) % *units;
        assert(!taken.count({k, cls}) &&
               "the busiest congruence class needs more instances than the "
               "allocation decided");
        taken.insert({k, cls});
        assignedUnit[op] = k;
        cursor = k + 1;
      }
    } else {
      // First fit over occupancy windows in start order, rotating the instance
      // scanned first so the load spreads.
      SmallVector<unsigned> freeAt(*units, 0);
      for (Operation *op : users) {
        unsigned start = *getStartTime(op);
        unsigned k = cursor % *units;
        for (unsigned tried = 1; freeAt[k] > start && tried < *units; ++tried)
          k = (k + 1) % *units;
        assert(freeAt[k] <= start && "the busiest cycle needs more instances "
                                     "than the allocation decided");
        assignedUnit[op] = k;
        freeAt[k] = start + getResourceCycles(op);
        cursor = k + 1;
      }
    }
  }
}

LogicalResult OccupancyProblem::verifyAllocation(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    std::optional<unsigned> units = getAllocation(rsrc);
    if (!units)
      continue; // no solve decided one, so the trivial allocation stands
    // (instance, cycle) pairs already taken.
    llvm::DenseSet<std::pair<unsigned, unsigned>> busy;
    for (Operation *op : getOperations()) {
      if (!usesResource(op, rsrc))
        continue;
      std::optional<unsigned> unit = getAssignedUnit(op);
      if (!unit || *unit >= *units) {
        assert(false && "an operation on an allocated operator has no instance "
                        "to run on, or one past the count decided");
        return failure();
      }
      unsigned start = *getStartTime(op);
      for (unsigned k = 0, occ = getResourceCycles(op); k < occ; ++k)
        if (!busy.insert({*unit, ii ? (start + k) % ii : start + k}).second) {
          assert(false && "two operations share one operator instance in the "
                          "same cycle");
          return failure();
        }
    }
  }
  return success();
}

LogicalResult OccupancyProblem::verifyOccupancy(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    unsigned limit = getLimit(rsrc).value_or(0);
    if (limit && demandFor(rsrc, ii) > limit) {
      assert(false && "a resource is oversubscribed across its occupancy "
                      "windows; the reservation table admits an operation "
                      "only when every slot it touches fits, so a solved "
                      "schedule cannot reach this");
      return failure();
    }
  }
  return success();
}

LogicalResult ModuloOccupancyProblem::verify() {
  if (failed(ModuloProblem::verify()))
    return failure();
  unsigned ii = *getInitiationInterval();
  if (failed(verifyOccupancy(ii)))
    return failure();
  return verifyAllocation(ii);
}

//===----------------------------------------------------------------------===//
// ChainingModuloProblem (declared in Scheduler.h): the composition of CIRCT's
// ChainingProblem and ModuloOccupancyProblem.
//===----------------------------------------------------------------------===//

LogicalResult ChainingModuloProblem::checkDefUse(Dependence dep) {
  if (!dep.isAuxiliary() && (getDistance(dep).value_or(0) != 0)) {
    assert(false && "a def-use dependence carries a non-zero distance; the "
                    "edges are ours to insert, so no input can reach this");
    return failure();
  }
  return success();
}

LogicalResult ChainingModuloProblem::check() {
  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(checkDefUse(dep)))
        return failure();

  if (ChainingProblem::check().succeeded() &&
      ModuloProblem::check().succeeded())
    return success();
  return failure();
}

LogicalResult ChainingModuloProblem::verify() {
  if (ChainingProblem::verify().succeeded() &&
      ModuloOccupancyProblem::verify().succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// ChainingSharedOperatorsProblem (declared in Scheduler.h): the composition of
// CIRCT's ChainingProblem and OccupancyProblem. The acyclic twin of
// ChainingModuloProblem (no distance, so no def-use distance check).
//===----------------------------------------------------------------------===//

LogicalResult ChainingSharedOperatorsProblem::check() {
  if (ChainingProblem::check().succeeded() &&
      SharedOperatorsProblem::check().succeeded())
    return success();
  return failure();
}

LogicalResult ChainingSharedOperatorsProblem::verify() {
  if (ChainingProblem::verify().succeeded() &&
      SharedOperatorsProblem::verify().succeeded() &&
      verifyOccupancy(/*ii=*/0).succeeded() &&
      verifyAllocation(/*ii=*/0).succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, float regFloor, unsigned minII,
                              SimplexWarmStart *warm) {
  ChainingModuloSimplexScheduler simplex(prob, lastOp, cycleTime, regFloor,
                                         minII);
  if (warm)
    simplex.setPlacementAdvisory();
  LogicalResult scheduled = simplex.schedule();
  if (!warm)
    return scheduled;
  warm->lowerBoundII = simplex.getLowerBoundII();
  warm->placed = succeeded(scheduled);
  // A placement failure is the caller's to recover from; a resource-free one
  // means no II admits a schedule, and nothing downstream can repair that.
  return success(simplex.hasLowerBound());
}

LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime,
                              float regFloor) {
  ChainingSharedOperatorsSimplexScheduler simplex(prob, lastOp, cycleTime,
                                                  regFloor);
  return simplex.schedule();
}

} // namespace mlir::allo
