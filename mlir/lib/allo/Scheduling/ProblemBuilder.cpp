/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ProblemBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/Footprint.h"
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

// Project a memory dependence's components onto the innermost scheduled loop --
// its deepest (last) component -- setting `drop` when an ENCLOSING loop carries
// the dependence (satisfied by that loop's sequential execution, so it does not
// constrain the innermost modulo schedule). A thin wrapper over the shared
// `carriedDistanceAtLevel`: for the innermost loop the target level is exactly
// the number of components.
static unsigned
innermostCarriedDistance(ArrayRef<affine::DependenceComponent> comps,
                         bool &drop) {
  bool valid = true;
  return static_cast<unsigned>(
      carriedDistanceAtLevel(comps, comps.size(), drop, valid));
}

// A dependence carried by an enclosing loop (a positive distance at some level)
// is satisfied by that loop's sequential execution, so it does not order two
// ops within a single straight-line instance. Unlike a modulo-scheduled loop,
// an acyclic span has no scheduled loop of its own -- so, whereas the cyclic
// builder keeps innermost-carried edges (with their distance), a span must drop
// every carried edge and keep only loop-independent (all-zero) ones.
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
  Value v = yield->getOperand(iterArg);
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
  Operation *definer = v.getDefiningOp();
  return definer ? std::make_pair(definer, distance + 1)
                 : std::make_pair<Operation *, unsigned>(nullptr, 0);
}

static bool isSyncCall(Operation *op); // a plain (non-async) sub-kernel call

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

      // Only model dependences whose source is also inside this loop.
      // Whole-func analysis may surface cross-region dependences whose source
      // is not part of this loop's scheduling problem (its endpoints are
      // scheduled elsewhere); those are handled by cross-region analysis, not
      // here.
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

  // Anchor: side-effecting ops (stores, stream accesses, and a sync sub-kernel
  // call -- an opaque node touching whole memrefs) must be scheduled before the
  // loop terminator, so the terminator is the problem's unique sink (a call in
  // the body would otherwise be a second, unordered sink).
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

  return problem;
}

bool whileHasIdentityForwarding(scf::WhileOp w) {
  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  scf::ConditionOp cond = w.getConditionOp();
  unsigned n = before.getNumArguments();
  if (cond.getArgs().size() != n || after.getNumArguments() != n ||
      w.getYieldOp().getNumOperands() != n)
    return false;
  for (auto [i, arg] : llvm::enumerate(cond.getArgs()))
    if (arg != before.getArgument(i))
      return false;
  return true;
}

template <class ProblemT>
ProblemT buildWhileProblem(scf::WhileOp w, DependenceAnalysis &deps) {
  assert(whileHasIdentityForwarding(w) && "while must forward args 1:1");
  ProblemT problem(w.getOperation());
  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  scf::ConditionOp condOp = w.getConditionOp();
  scf::YieldOp yieldOp = w.getYieldOp();
  Operation *condProducer = condOp.getCondition().getDefiningOp();

  // Register every op in both regions first (so a loop-carried back-edge whose
  // source is walked later still resolves). The before terminator
  // (`scf.condition`) is a pure forwarding wire; leaving it out keeps
  // `scf.yield` the problem's unique sink.
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

  // Loop-carried state recurrence: the next value of slot `j` (yield operand j,
  // if an after op defines it) feeds back to the readers of that slot's state
  // -- the users of before-arg[j] and after-arg[j], excluding the forwarding
  // terminators -- one iteration later.
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    Operation *definer = yieldOp.getOperand(j).getDefiningOp();
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
  Operation *anchor = yieldOp.getOperation();
  after.walk([&](Operation *op) {
    if (isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op))
      (void)problem.insertDependence(Problem::Dependence(op, anchor));
  });

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

  // Insert ops + intra-span memory/stream dependences. A span is one straight-
  // line instance, so it models only loop-INDEPENDENT (distance-0) dependences:
  // a carried edge is dropped, not added distance-less (which would miscast it
  // as a same-cycle constraint and, together with the intra-iteration forward
  // edge on the same pair, close a false positive cycle -> spurious
  // infeasibility).
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
        // enclosing loop, not within this single instance -- drop it.
        if (isLoopCarriedDependence(memoryDep.dependenceComponents))
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
    });

  // Op-level DependenceAnalysis tracks only load/store/stream accesses, so a
  // func.call -- an opaque sub-kernel touching whole memrefs -- carries no
  // memory edge. Order a sync call against another call (or a raw access) in
  // the same span by their memory footprints, at op granularity. A shared-array
  // producer/consumer serializes; two calls that are data-independent, share
  // only reads, or write provably DISJOINT elements get no edge (they overlap).
  // Only pairs with at least one sync call need this -- the rest are already
  // modeled above.
  //
  // A sync call contributes its CALLEE's per-argument footprint (direction plus
  // the callee's own affine accesses, in the caller's terms): the conservative
  // `summarizeOp` cannot see through a call, so it marks every memref operand
  // read+write, which would falsely serialize even two readers of one input. A
  // callee the summary cannot see through falls back to that conservative
  // record, which subsumes any partial one, so the pair stays ordered.
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
  Operation *sink = ops.back();
  for (Operation *op : problem.getOperations()) {
    if (op == sink)
      continue;
    // Two sync calls are ordered exactly by the footprint edges above; a
    // blanket auxiliary edge between them would falsely serialize
    // data-independent calls (the sink call's start forced past the other's
    // whole latency). Every other pair keeps the ASAP-sink edge (a zero-latency
    // sink serializes nothing).
    if (isSyncCall(op) && isSyncCall(sink))
      continue;
    (void)problem.insertDependence(Problem::Dependence(op, sink));
  }

  return problem;
}

// Explicit instantiations: the resource-aware chaining problems are the only
// problem types the scheduler pass builds -- loops as `ChainingModuloProblem`,
// straight-line spans as `ChainingSharedOperatorsProblem`.
template ChainingModuloProblem
buildCyclicProblem<ChainingModuloProblem>(LoopLikeOpInterface,
                                          DependenceAnalysis &);
template ChainingModuloProblem
buildWhileProblem<ChainingModuloProblem>(scf::WhileOp, DependenceAnalysis &);
template ChainingSharedOperatorsProblem
buildAcyclicProblem<ChainingSharedOperatorsProblem>(ArrayRef<Operation *>,
                                                    DependenceAnalysis &);

} // namespace mlir::allo
