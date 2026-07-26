/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Perfectize an imperfect loop nest so the perfect-nest scheduler can pipeline
// it, instead of the driver skipping it. An outer loop (affine.for / scf.for)
// whose body is `{ prologue..., inner loop, epilogue... }` is made perfect by
// sinking the surrounding ops into the inner loop:
//   * epilogue (after the inner loop) -> guarded by the last iteration,
//     remapping uses of the inner loop's results to its final (yielded) values;
//   * prologue (before it), always loop-invariant w.r.t. the inner loop:
//       - a store -> guarded by the first iteration (runs once, e.g. an
//                    accumulator init);
//       - pure ops / a load of a memref not written in the nest -> sunk
//                    unguarded (recompute).
// The guard follows the inner loop's dialect: an affine.for inner uses
// `affine.if (iv == const)` (its trip is constant); an scf.for inner uses
// `scf.if` on a runtime `cmpi` (`iv == lb` first, `iv + step >= ub` last).
// `fold-if-statements` then predicates the guards. Bail (leave the nest
// untouched, so the driver skips it) on anything not covered: sibling inner
// loops, an scf.while, escaping/aliasing values, other side effects.
//===----------------------------------------------------------------------===//

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::allo {
#define GEN_PASS_DEF_PERFECTIZELOOPNESTPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// One matched imperfect nest: the inner loop and the prologue/epilogue ops
// surrounding it in the outer body (all validated sinkable).
struct Match {
  Operation *lin = nullptr;          // inner loop (affine.for / scf.for)
  SmallVector<Operation *> prologue; // before lin
  SmallVector<Operation *> epilogue; // after lin
};

// The single body block of a counted loop (affine.for / scf.for).
static Block *loopBody(Operation *loop) {
  return &cast<LoopLikeOpInterface>(loop).getLoopRegions().front()->front();
}

// Does `loop`'s body contain a nested loop (i.e. `loop` is not innermost)?
static bool hasNestedLoop(Operation *loop) {
  bool found = false;
  loopBody(loop)->walk([&](Operation *op) {
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static Value storedMemRef(Operation *op) {
  if (auto s = dyn_cast<affine::AffineStoreOp>(op))
    return s.getMemRef();
  if (auto s = dyn_cast<memref::StoreOp>(op))
    return s.getMemRef();
  return {};
}

static Value loadedMemRef(Operation *op) {
  if (auto l = dyn_cast<affine::AffineLoadOp>(op))
    return l.getMemRef();
  if (auto l = dyn_cast<memref::LoadOp>(op))
    return l.getMemRef();
  return {};
}

// Match `outer` as a sinkable imperfect nest, or return nullopt. When `outer`
// has the shape of an imperfect nest (a counted inner loop plus surrounding
// ops) but an unsupported feature blocks it, `reason` is set so the caller can
// report the skip; a loop that is simply not an imperfect nest leaves `reason`
// empty (stay silent).
static std::optional<Match> matchImperfect(Operation *outer,
                                           std::string &reason) {
  Match m;
  unsigned innerCount = 0;
  bool hasWhile = false;
  for (Operation &op : loopBody(outer)->without_terminator()) {
    if (isa<affine::AffineForOp, scf::ForOp>(op)) {
      if (!m.lin)
        m.lin = &op;
      ++innerCount;
      continue;
    }
    if (isa<scf::WhileOp>(op)) {
      hasWhile = true;
      continue;
    }
    (m.lin ? m.epilogue : m.prologue).push_back(&op);
  }
  // Not an imperfect nest at all (no counted inner loop, or nothing to sink):
  // stay silent, since there is nothing this pass was meant to handle here.
  if (!m.lin || (m.prologue.empty() && m.epilogue.empty()))
    return std::nullopt;

  // From here `outer` is a genuine imperfect nest, so every bail explains why
  // the scheduler will skip it.
  if (innerCount > 1)
    return (reason = "it has sibling inner loops"), std::nullopt;
  if (hasWhile)
    return (reason = "it contains an uncounted (scf.while) inner loop"),
           std::nullopt;
  if (outer->getNumResults() != 0)
    return (reason =
                "the outer loop carries a result (an accumulator escapes)"),
           std::nullopt;
  if (hasNestedLoop(m.lin))
    return (reason = "the inner loop is itself a nest (not innermost)"),
           std::nullopt;

  // Inner-loop guard feasibility. affine.for: a constant last-iteration IV
  // (normalized, constant trip) for the `affine.if`. scf.for: a runtime guard
  // needs a known positive step for the last-iteration test (`iv+step >= ub`).
  if (auto af = dyn_cast<affine::AffineForOp>(m.lin)) {
    if (!af.hasConstantLowerBound() || af.getConstantLowerBound() != 0 ||
        af.getStepAsInt() != 1 || !affine::getConstantTripCount(af))
      return (reason = "the inner loop is not a normalized constant-trip loop"),
             std::nullopt;
  } else {
    auto sf = cast<scf::ForOp>(m.lin);
    std::optional<int64_t> step = getConstantIntValue(sf.getStep());
    if (!m.epilogue.empty() && (!step || *step <= 0))
      return (reason =
                  "the inner loop has a non-constant or non-positive step"),
             std::nullopt;
  }

  // Epilogue: straight-line, no result escaping the epilogue set (so every use
  // lands inside the inner loop after sinking; stores have no results).
  DenseSet<Operation *> epiSet(m.epilogue.begin(), m.epilogue.end());
  for (Operation *op : m.epilogue) {
    if (op->getNumRegions() != 0)
      return (reason = "an epilogue op has a nested region"), std::nullopt;
    for (Value r : op->getResults())
      for (Operation *user : r.getUsers())
        if (!epiSet.contains(user))
          return (reason = "an epilogue value is used outside the nest"),
                 std::nullopt;
  }

  // Prologue: each op pure (unguarded), a store (guard first), or a load of a
  // memref not written anywhere in the nest (recompute-safe).
  DenseSet<Value> written;
  loopBody(outer)->walk([&](Operation *op) {
    if (Value mr = storedMemRef(op))
      written.insert(mr);
  });
  for (Operation *op : m.prologue) {
    if (op->getNumRegions() != 0)
      return (reason = "a prologue op has a nested region"), std::nullopt;
    if (isMemoryEffectFree(op) || storedMemRef(op))
      continue;
    Value mr = loadedMemRef(op);
    if (!mr || written.contains(mr))
      return (reason = "a prologue op has an unschedulable side effect or an "
                       "aliased load"),
             std::nullopt;
  }
  return m;
}

// Sink `ops` into a guard inserted at `insertPt` inside `lin`'s body that fires
// only at the first (or last) iteration: an `affine.if` (constant IV) for an
// affine.for inner, an `scf.if` (runtime `cmpi`) for an scf.for inner.
static void sinkGuarded(Operation *lin, ArrayRef<Operation *> ops, bool first,
                        Operation *insertPt) {
  OpBuilder b(insertPt);
  Location loc = lin->getLoc();
  Operation *thenTerm;
  if (auto af = dyn_cast<affine::AffineForOp>(lin)) {
    int64_t v = first ? 0 : (*affine::getConstantTripCount(af) - 1);
    AffineExpr d0 = b.getAffineDimExpr(0);
    IntegerSet set = IntegerSet::get(/*dimCount=*/1, /*symbolCount=*/0,
                                     {d0 - v}, /*eqFlags=*/{true});
    auto ifOp = affine::AffineIfOp::create(b, loc, set,
                                           ValueRange{af.getInductionVar()},
                                           /*withElseRegion=*/false);
    thenTerm = ifOp.getThenBlock()->getTerminator();
  } else {
    auto sf = cast<scf::ForOp>(lin);
    Value iv = sf.getInductionVar();
    Value cond;
    if (first) {
      cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, iv,
                                   sf.getLowerBound());
    } else {
      Value next = arith::AddIOp::create(b, loc, iv, sf.getStep());
      cond = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, next,
                                   sf.getUpperBound());
    }
    auto ifOp = scf::IfOp::create(b, loc, cond, /*withElseRegion=*/false);
    thenTerm = ifOp.thenBlock()->getTerminator();
  }
  for (Operation *op : ops)
    op->moveBefore(thenTerm);
}

static void perfectize(Match &m) {
  Operation *lin = m.lin;
  Block *body = loopBody(lin);

  // Epilogue -> last-iteration guard before the terminator; the inner loop's
  // results equal their yields at the last iteration.
  if (!m.epilogue.empty()) {
    Operation *term = body->getTerminator();
    sinkGuarded(lin, m.epilogue, /*first=*/false, term);
    for (auto [res, yv] : llvm::zip(lin->getResults(), term->getOperands()))
      res.replaceAllUsesWith(yv);
  }

  // Prologue -> body top (before the original first op): a store runs once
  // under the first-iteration guard; a pure op / safe load is recomputed
  // unguarded.
  Operation *anchor = &body->front();
  for (Operation *op : m.prologue) {
    if (storedMemRef(op))
      sinkGuarded(lin, {op}, /*first=*/true, anchor);
    else
      op->moveBefore(anchor);
  }

  info(Stage::Prep, lin) << "Perfectizing imperfect loop nest by sinking "
                         << (m.prologue.size() + m.epilogue.size())
                         << " surrounding ops into the inner loop";
}

struct PerfectizeLoopNestPass
    : public allo::impl::PerfectizeLoopNestPassBase<PerfectizeLoopNestPass> {
  void runOnOperation() override {
    // Collect matches first; each sink only mutates its own inner loop, and the
    // innermost-inner-loop restriction rules out nested candidates.
    SmallVector<Match> work;
    getOperation().walk([&](Operation *outer) {
      if (!isa<affine::AffineForOp, scf::ForOp>(outer))
        return;
      std::string reason;
      if (std::optional<Match> m = matchImperfect(outer, reason))
        work.push_back(std::move(*m));
      else if (!reason.empty())
        warn(Stage::Prep, outer)
            << "imperfect loop nest not perfectized because " << reason
            << "; the scheduler schedules its body as sequential sub-regions "
               "instead of one fused pipeline";
    });
    for (Match &m : work)
      perfectize(m);
  }
};

} // namespace
