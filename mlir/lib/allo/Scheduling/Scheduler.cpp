/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Self-contained fork of CIRCT's linear-programming (SDC) simplex schedulers
// (externals/circt/lib/Scheduling/SimplexSchedulers.cpp), vendored into the
// Allo tree so the scheduling engine is ours to instrument (debugging /
// inspection) and extend without growing a CIRCT diff. It reuses CIRCT's public
// Problem data model (circt/Scheduling/Problems.h) and chaining utilities
// (circt/Scheduling/Utilities.h); only the solver lives here. Portions derived
// from LLVM/CIRCT, Apache-2.0 WITH LLVM-exception.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/Scheduler.h"
#include "allo/Scheduling/Utils.h"
#include "allo/Support/Logging.h"

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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

using llvm::dbgs;
using llvm::format;

namespace {

/// Number of consecutive cycles \p op occupies its resource unit: its latency
/// for a non-pipelined multi-cycle unit (stamped by the operator model), else 1
/// (fully pipelined). Governs how many reservation-table slots it holds.
static unsigned resourceCycles(Operation *op) {
  using mlir::allo::sched::kResourceCyclesAttr;
  if (auto a = op->getAttrOfType<IntegerAttr>(kResourceCyclesAttr))
    return static_cast<unsigned>(a.getInt());
  return 1;
}

/// A dependence circuit that binds the initiation interval: the ops around it,
/// plus the sums the II bound is read off. `latency` counts each edge's source
/// latency (and the extra cycle a chain-breaking constraint adds); `distance`
/// counts the iterations each edge spans.
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

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// Resource-free scheduling problems (called "central problems" in the papers)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
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

  /// Used to conveniently retrieve an operation's start time variable. The
  /// alternative would be to find the op's index in the problem's list of
  /// operations.
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
  /// Iteration distance a dependence spans. Only a cyclic problem carries one;
  /// an acyclic problem is the `distance == 0` special case, so the base
  /// answers 0 and the cyclic subclasses override, mirroring how
  /// `fillConstraintRow` adds the parameter-T term.
  virtual unsigned distanceOf(Problem::Dependence dep);
  /// The dependence circuit that binds the II at \p ii: the constraints are
  /// `t_dst - t_src >= latency(src) + extra - ii*distance`, so a schedule
  /// exists iff no circuit's weights sum positive. A positive circuit forces
  /// `ii >= ceil(latency / distance)`, and one with `distance == 0` can never
  /// be satisfied, which is exactly what "the problem is infeasible" means
  /// here. Empty when no circuit binds. O(|ops| * |deps|) Bellman-Ford.
  Recurrence bindingRecurrence(unsigned ii);
  /// Report a failed initial solve, naming the recurrence responsible. Shared
  /// by every scheduler's `schedule()`; the message is the only thing a user
  /// sees when their kernel has an unsatisfiable dependence cycle.
  void reportInfeasible();
  virtual LogicalResult checkLastOp();
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

  void dumpTableau();

public:
  explicit SimplexSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SimplexSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase {
private:
  Problem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SimplexScheduler(Problem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}

  LogicalResult schedule() override;
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

// This class solves acyclic, resource-constrained `SharedOperatorsProblem` with
// a simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  SharedOperatorsProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(SharedOperatorsProblem &prob,
                                  Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves the `ModuloProblem` using the iterative heuristic presented
// in [2].
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

  ModuloProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;
  // Lower bound on the II from a pipeline directive; the search seeds the II at
  // max(this, the resource-min II) and only ever grows it, so the achieved II
  // is max(this, the natural minimum). 1 imposes no additional bound.
  unsigned minII = 1;
  // Set when any limited op occupies its unit for >1 cycle (non-pipelined). The
  // de Dinechin II-increment assumes fully-pipelined (1-slot) reservations, so
  // a problem with blocking ops uses a conservative II-growth path instead.
  bool hasBlockingOps = false;
  // Sum of occupancies over limited ops; the conservative II-growth path
  // must converge within this bound (all ops fit in disjoint windows by then).
  unsigned totalResourceCycles = 0;

protected:
  Problem &getProblem() override { return prob; }
  LogicalResult checkLastOp() override;
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  bool fillObjectiveRow(SmallVector<int> &row, unsigned obj) override;
  void updateMargins();
  void scheduleOperation(Operation *n);
  unsigned computeResMinII();

public:
  ModuloSimplexScheduler(ModuloProblem &prob, Operation *lastOp,
                         unsigned minII = 1)
      : CyclicSimplexScheduler(prob, lastOp), prob(prob), mrt(*this),
        minII(minII) {}
  LogicalResult schedule() override;
};

// This class solves the `ChainingProblem` by relying on pre-computed
// chain-breaking constraints.
class ChainingSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingSimplexScheduler(ChainingProblem &prob, Operation *lastOp,
                           float cycleTime)
      : SimplexSchedulerBase(lastOp), prob(prob), cycleTime(cycleTime) {}
  LogicalResult schedule() override;
};

// This class solves the resource-free `ChainingCyclicProblem` by relying on
// pre-computed chain-breaking constraints. The optimal initiation interval (II)
// is determined as a side product of solving the parametric problem, and
// corresponds to the "RecMII" (= recurrence-constrained minimum II) usually
// considered as one component in the lower II bound used by modulo schedulers.
class ChainingCyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingCyclicProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  unsigned distanceOf(Problem::Dependence dep) override {
    return prob.getDistance(dep).value_or(0);
  }
  void fillConstraintRow(SmallVector<int> &row,
                         Problem::Dependence dep) override;
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingCyclicSimplexScheduler(ChainingCyclicProblem &prob, Operation *lastOp,
                                 float cycleTime)
      : SimplexSchedulerBase(lastOp), prob(prob), cycleTime(cycleTime) {}
  LogicalResult schedule() override;
};

// This class solves the resource-constrained, cyclic, chaining-enabled
// `ChainingModuloProblem` by reusing the `ModuloSimplexScheduler` (MRT + II
// increment) and layering the orthogonal chaining constraints around it: a
// pre-pass fills the chain-breaking dependences (consumed by `buildTableau`),
// and a post-pass fills the sub-cycle start times.
class ChainingModuloSimplexScheduler : public ModuloSimplexScheduler {
private:
  ChainingModuloProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override {
    // Inherited (cyclic) constraint row: latency + II*distance ...
    fillConstraintRow(row, dep);
    // ... plus one extra time step to break the combinational chain.
    row[parameter1Column] -= 1;
  }

public:
  ChainingModuloSimplexScheduler(ChainingModuloProblem &prob, Operation *lastOp,
                                 float cycleTime, unsigned minII = 1)
      : ModuloSimplexScheduler(prob, lastOp, minII), prob(prob),
        cycleTime(cycleTime) {}
  LogicalResult schedule() override {
    if (failed(computeChainBreakingDependences(prob, cycleTime,
                                               additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adding pipeline register stages / latency)";
    if (failed(ModuloSimplexScheduler::schedule()))
      return failure();
    return computeStartTimesInCycle(prob);
  }
};

// This class solves the resource-constrained, acyclic, chaining-enabled
// `ChainingSharedOperatorsProblem` by reusing the
// `SharedOperatorsSimplexScheduler` (per-cycle resource reservation) and
// layering the orthogonal chaining constraints around it. It is the acyclic
// mirror of `ChainingModuloSimplexScheduler`: a pre-pass fills the
// chain-breaking dependences (consumed by `buildTableau`), and a post-pass
// fills the sub-cycle start times.
class ChainingSharedOperatorsSimplexScheduler
    : public SharedOperatorsSimplexScheduler {
private:
  ChainingSharedOperatorsProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override {
    // Acyclic constraint row (latency only, no II term) ...
    fillConstraintRow(row, dep);
    // ... plus one extra time step to break the combinational chain.
    row[parameter1Column] -= 1;
  }

public:
  ChainingSharedOperatorsSimplexScheduler(ChainingSharedOperatorsProblem &prob,
                                          Operation *lastOp, float cycleTime)
      : SharedOperatorsSimplexScheduler(prob, lastOp), prob(prob),
        cycleTime(cycleTime) {}
  LogicalResult schedule() override {
    if (failed(computeChainBreakingDependences(prob, cycleTime,
                                               additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adds pipeline register stages / latency)";
    if (failed(SharedOperatorsSimplexScheduler::schedule()))
      return failure();
    return computeStartTimesInCycle(prob);
  }
};

} // anonymous namespace

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
  auto diag = error(Stage::Sched, prob.getContainingOp());
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
  assert(obj == 0);
  // Minimize start time of user-specified last operation.
  row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
  return false;
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

  // Assign column and variable numbers to the operations' start times.
  for (auto *op : prob.getOperations()) {
    nonBasicVariables.push_back(var);
    startTimeVariables[op] = var;
    startTimeLocations.push_back(firstNonBasicVariableColumn + var);
    ++var;
  }

  // one column for each parameter (1,S,T), and for all operations
  nColumns = nParameters + nonBasicVariables.size();

  // Helper to grow both the tableau and the implicit column vector.
  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  // Set up the objective rows.
  nObjectives = 0;
  bool hasMoreObjectives;
  do {
    auto &objRowVec = addRow();
    hasMoreObjectives = fillObjectiveRow(objRowVec, nObjectives);
    ++nObjectives;
  } while (hasMoreObjectives);

  // Now set up rows/constraints for the dependences.
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

  // Look up numeric IDs of variables involved in this pivot operation.
  unsigned &nonBasicVar =
      nonBasicVariables[pivotColumn - firstNonBasicVariableColumn];
  unsigned &basicVar = basicVariables[pivotRow - firstConstraintRow];

  // Keep track of where start time variables are; ignore slack variables.
  if (nonBasicVar < startTimeLocations.size())
    startTimeLocations[nonBasicVar] = -pivotRow; // ...going into basis.
  if (basicVar < startTimeLocations.size())
    startTimeLocations[basicVar] = pivotColumn; // ...going out of basis.

  // Record the swap in the variable lists.
  std::swap(nonBasicVar, basicVar);
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  // "Solving" technically means perfoming dual pivot steps until primal
  // feasibility is reached, i.e. the parametric constants in all constraint
  // rows are non-negative.
  while (auto pivotRow = findDualPivotRow()) {
    // Look for pivot elements.
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
        // Name the circuit that forces the bump; it is what a user would have
        // to shorten to get the II back. The search is O(|ops| * |deps|), so
        // only run it when the message will actually be printed.
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
    // Look for pivot elements.
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

  // Mark variable.
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

  // Perform the exchange.
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

  // Freeze variable and translate its column by parameter S.
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

  // Bookkeeping.
  frozenVariables[startTimeVariable] += amount;

  // Moving an already frozen variable means translating it by the desired
  // amount, and solving the tableau to restore primal feasibility...
  translate(startTimeLocations[startTimeVariable], /* factor1= */ amount,
            /* factorS= */ 0, /* factorT= */ 0);

  // ... however, we typically batch-move multiple operations (otherwise, the
  // tableau may become infeasible on intermediate steps), so actually defer
  // solving to the caller.
}

unsigned SimplexSchedulerBase::getStartTime(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());

  if (!isInBasis(startTimeVariable))
    // Non-basic variables that are not already fixed to a specific time step
    // are 0 at the end of the simplex algorithm.
    return frozenVariables.lookup(startTimeVariable);

  // For the variables currently in basis, we look up the solution in the
  // tableau.
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
// SimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult SimplexScheduler::schedule() {
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

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
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

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Build a priority list of limited ops (sorted by resource-free start
  // time, a topological order); fixing operators in this order keeps the
  // acyclic problem feasible. TODO: use a better priority (ASAP/ALAP, height).
  std::stable_sort(limitedOps.begin(), limitedOps.end(),
                   [&](Operation *a, Operation *b) {
                     return getStartTime(startTimeVariables[a]) <
                            getStartTime(startTimeVariables[b]);
                   });

  // Store the number of operations using a resource type in a particular time
  // step.
  SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    assert(maybeRsrcs && "Limited operation must have linked resource types");

    auto &rsrcs = *maybeRsrcs;
    assert(rsrcs.size() == 1 &&
           "an operation is linked to several resource types; ProblemBuilder "
           "links exactly one, and the scheduler indexes by that one");

    auto rsrc = rsrcs[0];
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    assert(limit > 0);

    // Find the first time step (from the current start time) where an
    // operator instance is free for the whole occupancy window (occ
    // consecutive cycles; occ == 1 when pipelined).
    unsigned occ = resourceCycles(op);
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    auto hasRoom = [&](unsigned t) {
      for (unsigned i = 0; i < occ; ++i)
        if (reservationTable[rsrc].lookup(t + i) == limit)
          return false;
      return true;
    };
    while (!hasRoom(candTime))
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the operator use across the occupancy window.
    for (unsigned i = 0; i < occ; ++i)
      ++reservationTable[rsrc][candTime + i];

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
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "Operation must have linked resource types");

  auto &rsrcs = *maybeRsrcs;
  assert(rsrcs.size() == 1 &&
         "an operation is linked to several resource types; ProblemBuilder "
         "links exactly one, and the scheduler indexes by that one");

  auto rsrc = rsrcs[0];
  auto lim = *sched.prob.getLimit(rsrc);
  assert(lim > 0);

  auto &revTab = reverseTables[rsrc];
  assert(!revTab.count(op));

  // A non-pipelined op occupies `occ` consecutive modulo slots (occ == 1
  // when pipelined); a window wider than II wraps, hitting one slot twice,
  // which a per-slot set would hide. Admit only if every touched slot fits.
  unsigned occ = resourceCycles(op);
  unsigned base = timeStep % sched.parameterT;
  auto &table = tables[rsrc];
  SmallDenseMap<unsigned, unsigned> want;
  for (unsigned i = 0; i < occ; ++i)
    ++want[(base + i) % sched.parameterT];
  for (const auto &[slot, cnt] : want)
    if (table.lookup(slot) + cnt > lim)
      return failure();
  for (const auto &[slot, cnt] : want)
    table[slot] += cnt;
  revTab[op] = base;
  return success();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "Operation must have linked resource types");

  auto &rsrcs = *maybeRsrcs;
  assert(rsrcs.size() == 1 &&
         "an operation is linked to several resource types; ProblemBuilder "
         "links exactly one, and the scheduler indexes by that one");

  auto rsrc = rsrcs[0];
  auto &revTab = reverseTables[rsrc];
  auto it = revTab.find(op);
  assert(it != revTab.end());
  unsigned occ = resourceCycles(op);
  auto &table = tables[rsrc];
  // Undo enter's per-slot increments (recomputed from stored base + occ, so a
  // wrapped slot is decremented once per lap, symmetric with `want` above).
  for (unsigned i = 0; i < occ; ++i) {
    unsigned &cnt = table[(it->second + i) % sched.parameterT];
    assert(cnt > 0 && "releasing an MRT slot that was never reserved");
    --cnt;
  }
  revTab.erase(it);
}

bool ModuloSimplexScheduler::fillObjectiveRow(SmallVector<int> &row,
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

void ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  // `n` contends for a single physical resource (its memref port or unit
  // pool); the reservation table arbitrates that resource, not the operator
  // type. Every op reaching here is limited, so it has exactly one.
  auto rsrcN = (*prob.getLinkedResourceTypes(n))[0];
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
        return;
      }
      // Problem became infeasible with `n` at `ct`, roll back the MRT
      // assignment. Also, no later time can be feasible, so stop the search
      // here.
      mrt.release(n);
      break;
    }

  // Non-pipelined ops reserve multiple slots, which the single-slot move
  // logic can't handle. Instead grow II via the base transform (each op
  // keeps its valid modulo slot), rebuild the MRT, and retry until `n` fits.
  if (hasBlockingOps) {
    while (true) {
      SmallVector<std::pair<unsigned, unsigned>> phis;
      for (Operation *j : scheduled) {
        unsigned stvJ = startTimeVariables[j];
        phis.push_back({stvJ, getStartTime(stvJ) / parameterT});
      }
      for (auto [stvJ, phiJ] : phis)
        moveBy(stvJ, phiJ);
      info(Stage::Sched, n)
          << "II=" << parameterT
          << " is not achievable: a non-pipelined operator ("
          << n->getName().getStringRef()
          << ") cannot fit its occupancy window, trying II=" << parameterT + 1;
      ++parameterT;
      // Every op fits in a disjoint occupancy window by II =
      // totalResourceCycles; 2x + 2 leaves slack for consecutive-window
      // fragmentation.
      assert(parameterT <= 2 * static_cast<int>(totalResourceCycles) + 2 &&
             "non-pipelined II growth did not converge");
      auto solved = solveTableau();
      assert(succeeded(solved));
      (void)solved;

      mrt.clear();
      for (Operation *j : scheduled) {
        auto entered = mrt.enter(j, getStartTime(startTimeVariables[j]));
        assert(succeeded(entered) && "re-entry after II growth must succeed");
        (void)entered;
      }

      unsigned lo = getStartTime(stvN);
      for (unsigned ct = lo; ct <= lo + parameterT - 1; ++ct)
        if (succeeded(mrt.enter(n, ct))) {
          if (succeeded(scheduleAt(stvN, ct)))
            return;
          mrt.release(n);
        }
    }
  }

  // As a last resort, increase II to make room for the op. De Dinechin's
  // Theorem 1 lays out conditions/guidelines to transform the current partial
  // schedule for II to a valid one for a larger II'.

  info(Stage::Sched, n) << "II=" << parameterT
                        << " is not achievable: a shared-resource conflict for "
                        << n->getName().getStringRef()
                        << ", trying II=" << parameterT + 1;
  info(Stage::Sched) << "Incrementing II to " << (parameterT + 1)
                     << " to resolve resource conflict for " << *n;

  // This is simpler than the paper's general approach because operators
  // here are fully pipelined, so incrementing the II by one always suffices.

  // Decompose start time.
  unsigned phiN = stN / parameterT;
  unsigned tauN = stN % parameterT;

  // Track whether the following moves free an operator instance in the slot
  // the current op wants, so it can stay there.
  unsigned deltaN = 1;

  // We're going to revisit the current partial schedule.
  SmallVector<Operation *> moved;
  for (Operation *j : scheduled) {
    auto rsrcJ = (*prob.getLinkedResourceTypes(j))[0];
    unsigned stvJ = startTimeVariables[j];
    unsigned stJ = getStartTime(stvJ);
    unsigned phiJ = stJ / parameterT;
    unsigned tauJ = stJ % parameterT;
    unsigned deltaJ = 0;

    if (rsrcN == rsrcJ) {
      // Resolve conflicts by moving ops contending for `n`'s *same resource*
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
  auto solved = solveTableau();
  assert(succeeded(solved));
  (void)solved;

  // Re-enter moved operations into their new slots.
  for (auto *m : moved)
    mrt.release(m);
  for (auto *m : moved) {
    auto enteredM = mrt.enter(m, getStartTime(startTimeVariables[m]));
    assert(succeeded(enteredM));
    (void)enteredM;
  }

  // Finally, schedule the operation. Again, adding `phiN` accounts for the
  // implicit shift caused by incrementing the II.
  auto fixedN = scheduleAt(stvN, stN + phiN + deltaN);
  auto enteredN = mrt.enter(n, tauN + deltaN);
  assert(succeeded(fixedN) && succeeded(enteredN));
  (void)fixedN, (void)enteredN;
}

unsigned ModuloSimplexScheduler::computeResMinII() {
  unsigned resMinII = 1;
  SmallDenseMap<Problem::ResourceType, unsigned> uses;
  for (auto *op : prob.getOperations()) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    for (auto rsrc : *maybeRsrcs) {
      if (prob.getLimit(rsrc).value_or(0) > 0)
        uses[rsrc] += resourceCycles(op); // occupancy: latency if non-pipelined
    }
  }

  // Integer ceil: enough parallel units to cover total occupancy in one II.
  // (unsigned `a / b` floors, so an explicit integer ceil is needed once
  // limit >= 2.)
  for (auto pair : uses) {
    unsigned limit = *prob.getLimit(pair.first);
    resMinII = std::max(resMinII, (pair.second + limit - 1) / limit);
  }

  return resMinII;
}

LogicalResult ModuloSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  // Seed the II at the resource-min II, but never below the pipeline
  // directive's target: the search only grows the II, so the result is
  // max(minII, natural).
  unsigned resMinII = computeResMinII();
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

  // Determine which operations are subject to resource constraints, and whether
  // any of them is non-pipelined (occupies its unit for more than one cycle).
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob)) {
      unscheduled.push_back(op);
      unsigned occ = resourceCycles(op);
      totalResourceCycles += occ;
      if (occ > 1)
        hasBlockingOps = true;
    }

  // Main loop: Iteratively fix limited operations to time steps.
  while (!unscheduled.empty()) {
    // Update ASAP/ALAP times.
    updateMargins();

    // Heuristically (here: least amount of slack) pick the next operation to
    // schedule.
    auto *opIt =
        std::min_element(unscheduled.begin(), unscheduled.end(),
                         [&](Operation *opA, Operation *opB) {
                           auto stvA = startTimeVariables[opA];
                           auto stvB = startTimeVariables[opB];
                           auto slackA = alapTimes[stvA] - asapTimes[stvA];
                           auto slackB = alapTimes[stvB] - asapTimes[stvB];
                           return slackA < slackB;
                         });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    scheduleOperation(op);
    scheduled.push_back(op);
  }

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingSimplexScheduler::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain (the latency is negative in the
  // tableau).
  row[parameter1Column] -= 1;
}

LogicalResult ChainingSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau())) {
    reportInfeasible();
    return failure();
  }

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn)); // Problem is known to be acyclic at this point.
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingCyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingCyclicSimplexScheduler::fillConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  SimplexSchedulerBase::fillConstraintRow(row, dep);
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

void ChainingCyclicSimplexScheduler::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain (the latency is negative in the
  // tableau).
  row[parameter1Column] -= 1;
}

LogicalResult ChainingCyclicSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
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

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn));
  (void)filledIn;

  return success();
}

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// ChainingModuloProblem (declared in Scheduler.h): the composition of CIRCT's
// ChainingProblem and ModuloProblem. Mirrors CIRCT's ChainingCyclicProblem.
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
      ModuloProblem::verify().succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// ChainingSharedOperatorsProblem (declared in Scheduler.h): the composition of
// CIRCT's ChainingProblem and SharedOperatorsProblem. The acyclic twin of
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
      SharedOperatorsProblem::verify().succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduleSimplex(Problem &prob, Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(CyclicProblem &prob, Operation *lastOp) {
  CyclicSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(SharedOperatorsProblem &prob, Operation *lastOp) {
  SharedOperatorsSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(ModuloProblem &prob, Operation *lastOp) {
  ModuloSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(ChainingProblem &prob, Operation *lastOp,
                              float cycleTime) {
  ChainingSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(ChainingCyclicProblem &prob, Operation *lastOp,
                              float cycleTime) {
  ChainingCyclicSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, unsigned minII) {
  ChainingModuloSimplexScheduler simplex(prob, lastOp, cycleTime, minII);
  return simplex.schedule();
}

LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime) {
  ChainingSharedOperatorsSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

} // namespace mlir::allo
