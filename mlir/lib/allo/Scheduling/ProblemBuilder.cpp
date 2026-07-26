/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ProblemBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/Footprint.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/Scheduler.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace mlir::allo {

// Project a memory dependence's components onto the innermost scheduled
// loop (its deepest, last component), setting `drop` when an ENCLOSING loop
// carries the dependence (satisfied by that loop's sequential execution, so
// it does not constrain the innermost modulo schedule). A thin wrapper over
// the shared `carriedDistanceAtLevel`: for the innermost loop the target
// level is exactly the number of components.
static unsigned
innermostCarriedDistance(ArrayRef<affine::DependenceComponent> comps,
                         bool &drop) {
  bool valid = true;
  return static_cast<unsigned>(
      carriedDistanceAtLevel(comps, comps.size(), drop, valid));
}

// A dependence carried by an enclosing loop (a positive distance at some
// level) is satisfied by that loop's sequential execution, so it does not
// order two ops within a single straight-line instance. Unlike a
// modulo-scheduled loop, an acyclic span has no scheduled loop of its own,
// so while the cyclic builder keeps innermost-carried edges (with their
// distance), a span must drop every carried edge and keep only
// loop-independent (all-zero) ones.
static bool
isLoopCarriedDependence(ArrayRef<affine::DependenceComponent> comps) {
  for (const affine::DependenceComponent &c : comps)
    if (c.lb.value_or(0) > 0)
      return true;
  return false;
}

// Trace an iter_arg's incoming value to the operation that actually defines it,
// following any chain of iter_arg-to-iter_arg shifts (a yield operand that is
// itself an iter_arg of this loop, as produced by accumulator rotation) and
// counting one loop-carried distance per hop. Returns {definer, distance}, or
// {nullptr, 0} for a pure shift cycle (loop-invariant, no recurrence) or a
// value defined outside the loop.
static std::pair<Operation *, unsigned>
traceIterArgSource(Block *body, Operation *yield, unsigned iterArg) {
  auto v = yield->getOperand(iterArg);
  unsigned distance = 0;
  llvm::SmallDenseSet<unsigned> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    // iter_args are the body block arguments after the induction variable.
    if (arg.getOwner() != body || arg.getArgNumber() == 0 ||
        !seen.insert(arg.getArgNumber()).second)
      return {nullptr, 0};
    ++distance;
    v = yield->getOperand(arg.getArgNumber() - 1);
  }
  auto *definer = v.getDefiningOp();
  return definer ? std::make_pair(definer, distance + 1)
                 : std::make_pair<Operation *, unsigned>(nullptr, 0);
}

static bool isSyncCall(Operation *op); // a plain (non-async) sub-kernel call

// Anchor every remaining dependence-DAG sink to \p anchor with a
// loop-independent (distance-0) edge, making the anchor the problem's unique
// sink. The modulo scheduler requires that: it minimizes the anchor's start
// time, and `ModuloSimplexScheduler::checkLastOp` rejects a problem outright
// if any other operation has no distance-0 successor.
//
// The explicit side-effect anchoring in the builders covers the common case
// (a store / stream access / sync call has no results at all, so nothing can
// depend on it), but it enumerates op kinds and so cannot be complete. A sink
// is a graph property: any op whose consumers are all loop-carried, or a
// nested region's result-less terminator, is one too. Computing the set is
// exact and makes the rejection structurally unreachable rather than a
// user-facing limit. The same shape appears in the pipelined-level problem,
// which anchors every level node to the terminator.
//
// Anchoring is sound: the anchor is the loop body's terminator, so an edge
// `sink -> anchor` only states that the iteration is not complete until the
// sink has produced its result, which is exactly the region's own semantics.
template <class ProblemT>
static void anchorSinks(ProblemT &problem, Operation *anchor) {
  DenseSet<Operation *> sinks(problem.getOperations().begin(),
                              problem.getOperations().end());
  for (Operation *op : problem.getOperations())
    for (auto &dep : problem.getDependences(op))
      if (problem.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());
  // Collect in the problem's insertion order, not the hash set's, so the edges
  // and the solved schedule are deterministic. Snapshot before inserting:
  // `insertDependence` registers its endpoints into the set being iterated.
  SmallVector<Operation *> unanchored;
  for (Operation *op : problem.getOperations())
    if (op != anchor && sinks.contains(op))
      unanchored.push_back(op);
  for (Operation *op : unanchored)
    (void)problem.insertDependence(Problem::Dependence(op, anchor));
}

template <class ProblemT>
ProblemT buildCyclicProblem(LoopLikeOpInterface loop,
                            DependenceAnalysis &deps) {
  ProblemT problem(loop.getOperation());
  Block *body = &loop.getLoopRegions().front()->front();

  // Insert memory and stream dependences into the problem.
  body->walk([&](Operation *op) {
    problem.insertOperation(op);

    for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Only model dependences whose source is inside this loop. Whole-func
      // analysis may surface cross-region dependences whose endpoints are
      // scheduled elsewhere; cross-region analysis handles those.
      if (!body->findAncestorOpInBlock(*memoryDep.source))
        continue;

      bool drop = false;
      unsigned distance =
          innermostCarriedDistance(memoryDep.dependenceComponents, drop);
      if (drop)
        continue;

      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // One pair may carry both an intra-iteration (dist 0) and a loop-carried
      // edge (e.g. `A[2*i]`/`A[i]`, which alias only at i == 0): keep the
      // tightest (smallest) distance so the same-iteration ordering survives.
      unsigned cur = problem.getDistance(dep).value_or(distance);
      problem.setDistance(dep, std::min(cur, distance));
    }
  });

  // Insert conditional dependences into the problem.
  body->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Problem::Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Anchor: side-effecting ops (stores, streams, a sync sub-kernel call) must
  // be scheduled before the loop terminator, making it the problem's unique
  // sink; a call in the body would otherwise be a second, unordered sink.
  auto *anchor = body->getTerminator();
  body->walk([&](Operation *op) {
    if (!isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op) &&
        !isSyncCall(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = loop.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      // The value carried into iter_arg `i` may reach its real definer through
      // a chain of shifts; the distance is the number of iterations it spans (1
      // for a direct recurrence, P for a P-slot rotated accumulator).
      auto [definer, distance] = traceIterArgSource(body, anchor, i);
      if (!definer)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(definer, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
        problem.setDistance(dep, distance);
      }
    }
  }

  // Every other sink joins the terminator too. Run last, so the sink set is
  // computed over the finished graph.
  anchorSinks(problem, anchor);

  return problem;
}

bool whileHasIdentityForwarding(scf::WhileOp w) {
  auto &before = w.getBefore().front();
  auto &after = w.getAfter().front();
  auto cond = w.getConditionOp();
  unsigned n = before.getNumArguments();
  if (cond.getArgs().size() != n || after.getNumArguments() != n ||
      w.getYieldOp().getNumOperands() != n)
    return false;
  for (auto [i, arg] : llvm::enumerate(cond.getArgs()))
    if (arg != before.getArgument(i))
      return false;
  return true;
}

bool conditionIsCombinational(scf::WhileOp w, const OperatorLibrary &lib) {
  // The continue-test settles the cycle the loop issues iff every op in its
  // cone (the before region; `scf.condition` is a pure wire) is 0-latency,
  // else the while needs a sequential CHECK/RUN controller, not a pipeline.
  bool comb = true;
  auto *term = w.getConditionOp().getOperation();
  w.getBefore().walk([&](Operation *op) {
    if (op == term || lib.lookup(op).latency == 0)
      return WalkResult::advance();
    comb = false;
    return WalkResult::interrupt();
  });
  return comb;
}

template <class ProblemT>
ProblemT buildWhileProblem(scf::WhileOp w, DependenceAnalysis &deps) {
  assert(whileHasIdentityForwarding(w) && "while must forward args 1:1");
  ProblemT problem(w.getOperation());
  auto &before = w.getBefore().front();
  auto &after = w.getAfter().front();
  auto condOp = w.getConditionOp();
  auto yieldOp = w.getYieldOp();
  auto *condProducer = condOp.getCondition().getDefiningOp();

  // Register every op in both regions first, so a later-walked back-edge
  // source still resolves. The before terminator (`scf.condition`) is a pure
  // forwarding wire; excluding it keeps `scf.yield` the unique sink.
  before.walk([&](Operation *op) {
    if (op != condOp.getOperation())
      problem.insertOperation(op);
  });
  after.walk([&](Operation *op) { problem.insertOperation(op); });

  // Memory / stream dependences over both regions (intra-`while` only; SSA
  // def-use is modeled implicitly by the problem).
  auto addMemDeps = [&](Block &blk) {
    blk.walk([&](Operation *op) {
      for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
        if (!hasDependence(memoryDep.dependenceType))
          continue;
        if (!w->isProperAncestor(memoryDep.source))
          continue;
        bool drop = false;
        unsigned distance =
            innermostCarriedDistance(memoryDep.dependenceComponents, drop);
        if (drop)
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        if (failed(problem.insertDependence(dep)))
          continue;
        // Keep the tightest distance when the pair also has another edge.
        unsigned cur = problem.getDistance(dep).value_or(distance);
        problem.setDistance(dep, std::min(cur, distance));
      }
    });
  };
  addMemDeps(before);
  addMemDeps(after);

  // Non-speculative condition gate: the whole after body (state update +
  // stores) waits for the condition, so the state recurrence runs through it
  // (II >= t_cond, no speculation). dist 0 (intra-iteration).
  if (condProducer)
    after.walk([&](Operation *op) {
      (void)problem.insertDependence(Problem::Dependence(condProducer, op));
    });

  // Loop-carried state recurrence: the next value of slot `j` (yield operand
  // j) feeds back one iteration later to that slot's readers: the users of
  // before-arg[j] and after-arg[j], excluding the forwarding terminators.
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    auto *definer = yieldOp.getOperand(j).getDefiningOp();
    if (!definer)
      continue; // block-arg / invariant: no recurrence
    SmallVector<Operation *> readers;
    for (Operation *u : before.getArgument(j).getUsers())
      if (u != condOp.getOperation())
        readers.push_back(u);
    for (Operation *u : after.getArgument(j).getUsers())
      if (u != yieldOp.getOperation())
        readers.push_back(u);
    for (Operation *r : readers) {
      Problem::Dependence dep(definer, r);
      if (succeeded(problem.insertDependence(dep)))
        problem.setDistance(dep, 1);
    }
  }

  // Side-effect anchor: stores / streams in the body precede the yield.
  auto *anchor = yieldOp.getOperation();
  after.walk([&](Operation *op) {
    if (isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op))
      (void)problem.insertDependence(Problem::Dependence(op, anchor));
  });

  // Every other sink joins the yield too. The `before` region normally reaches
  // the anchor through the condition gate above, which is empty when the
  // condition is a block argument or the after region is bare.
  anchorSinks(problem, anchor);

  return problem;
}

// A synchronous sub-kernel call: a plain (non-async) func.call, which the
// parent schedules as an opaque fixed-latency node in program order. An async
// call is composed structurally as dataflow (ordered by its streams, not the
// SDC schedule), so it is not treated as a sync call here.
static bool isSyncCall(Operation *op) {
  return isa<func::CallOp>(op) && !op->hasAttr(kAlloAsyncAttr);
}

template <class ProblemT>
ProblemT buildAcyclicProblem(ArrayRef<Operation *> ops,
                             DependenceAnalysis &deps) {
  assert(!ops.empty() && "straight-line region must have at least one op");
  ProblemT problem(ops.front());

  // Collect the span's op set (all nested ops) for intra-span dep filtering.
  DenseSet<Operation *> spanOps;
  for (Operation *top : ops)
    top->walk([&](Operation *op) { spanOps.insert(op); });

  // Insert ops + intra-span memory/stream dependences. Only loop-INDEPENDENT
  // (distance-0) edges are modeled; adding a carried edge distance-less
  // would falsely close a cycle with the forward edge (spurious infeasibility).
  for (Operation *top : ops)
    top->walk([&](Operation *op) {
      problem.insertOperation(op);

      for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
        if (!hasDependence(memoryDep.dependenceType))
          continue;
        // Only intra-span dependences belong to this problem.
        if (!spanOps.contains(memoryDep.source))
          continue;
        // A loop-carried dependence is satisfied across iterations of the
        // enclosing loop, not within this single instance; drop it.
        if (isLoopCarriedDependence(memoryDep.dependenceComponents))
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
    });

  // DependenceAnalysis misses call ops; sync calls are instead ordered by
  // memory footprint: a shared write serializes, disjoint/read-only don't,
  // and an opaque callee falls back to a conservative (safe) record.
  auto summarize = [](Operation *top, Summary &s) {
    top->walk([&](Operation *op) {
      if (isSyncCall(op) && summarizeCall(cast<func::CallOp>(op), s))
        return;
      summarizeOp(op, s);
    });
  };
  for (unsigned i = 0, e = ops.size(); i < e; ++i)
    for (unsigned j = i + 1; j < e; ++j) {
      if (!isSyncCall(ops[i]) && !isSyncCall(ops[j]))
        continue;
      Summary si, sj;
      summarize(ops[i], si);
      summarize(ops[j], sj);
      for (const auto &kv : si.mem) {
        auto it = sj.mem.find(kv.first);
        if (it != sj.mem.end() &&
            callFootprintConflict(kv.second, it->second) != Conflict::None)
          (void)problem.insertDependence(Problem::Dependence(ops[i], ops[j]));
      }
    }

  // Make the last program-order op a unique sink via auxiliary dependences, so
  // that minimizing its start time yields an ASAP schedule for the whole span.
  auto *sink = ops.back();
  for (Operation *op : problem.getOperations()) {
    if (op == sink)
      continue;
    // Two sync calls are already ordered by the footprint edges above; a
    // blanket edge here would falsely serialize data-independent calls.
    // Every other pair keeps the ASAP-sink edge (zero latency, no ordering).
    if (isSyncCall(op) && isSyncCall(sink))
      continue;
    (void)problem.insertDependence(Problem::Dependence(op, sink));
  }

  return problem;
}

// Explicit instantiations. The scheduler pass builds only the resource-aware
// chaining problems: loops as `ChainingModuloProblem`, straight-line spans as
// `ChainingSharedOperatorsProblem`.
template ChainingModuloProblem
buildCyclicProblem<ChainingModuloProblem>(LoopLikeOpInterface,
                                          DependenceAnalysis &);
template ChainingModuloProblem
buildWhileProblem<ChainingModuloProblem>(scf::WhileOp, DependenceAnalysis &);
template ChainingSharedOperatorsProblem
buildAcyclicProblem<ChainingSharedOperatorsProblem>(ArrayRef<Operation *>,
                                                    DependenceAnalysis &);

} // namespace mlir::allo
