/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RAISECOUNTEDWHILEPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A matched counted-while: which iter-arg is the induction variable, the tested
// bound and step, and whether the test was inclusive (`<=`, so ub = bound + 1).
struct CountedWhileMatch {
  unsigned ivIndex;
  Value bound;
  Value step;
  bool inclusive;
};

// Whether `v` is loop-invariant w.r.t. `w`: not defined inside either region.
static bool isInvariant(Value v, scf::WhileOp w) {
  Operation *def = v.getDefiningOp();
  return !def || !w->isProperAncestor(def);
}

// Match an increasing counted-while: a pure ordered test of one monotonically
// increasing induction variable against a loop-invariant bound. Returns nullopt
// (leave the loop alone) on any deviation.
static std::optional<CountedWhileMatch> matchCountedWhile(scf::WhileOp w) {
  if (!w.getBefore().hasOneBlock() || !w.getAfter().hasOneBlock())
    return std::nullopt;
  Block &before = w.getBefore().front();
  scf::ConditionOp cond = w.getConditionOp();
  scf::YieldOp yield = w.getYieldOp();

  // Identity forwarding: the condition passes the before args through 1:1, so
  // before / after / inits / results share one index space.
  if (cond.getArgs().size() != before.getNumArguments())
    return std::nullopt;
  for (auto [i, arg] : llvm::enumerate(cond.getArgs()))
    if (arg != before.getArgument(i))
      return std::nullopt;

  // `before` is pure and holds only the comparison + the condition terminator.
  auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp->getBlock() != &before)
    return std::nullopt;
  for (Operation &op : before.without_terminator())
    if (&op != cmp.getOperation())
      return std::nullopt;

  // One cmp operand is a before block-arg (the IV), the other is invariant.
  auto ivArg = dyn_cast<BlockArgument>(cmp.getLhs());
  Value bound = cmp.getRhs();
  bool ivOnLhs = true;
  if (!ivArg || ivArg.getOwner() != &before) {
    ivArg = dyn_cast<BlockArgument>(cmp.getRhs());
    bound = cmp.getLhs();
    ivOnLhs = false;
  }
  if (!ivArg || ivArg.getOwner() != &before || !isInvariant(bound, w))
    return std::nullopt;

  // Terminating, ordered predicate; normalize the IV to the "less-than" side.
  using P = arith::CmpIPredicate;
  P pred = cmp.getPredicate();
  if (!ivOnLhs) {
    switch (pred) { // bound <pred> iv  ==  iv <flipped> bound
    case P::ugt:
      pred = P::ult;
      break;
    case P::sgt:
      pred = P::slt;
      break;
    case P::uge:
      pred = P::ule;
      break;
    case P::sge:
      pred = P::sle;
      break;
    default:
      return std::nullopt;
    }
  }
  bool inclusive;
  switch (pred) {
  case P::ult:
  case P::slt:
    inclusive = false;
    break;
  case P::ule:
  case P::sle:
    inclusive = true;
    break;
  default:
    return std::nullopt; // eq/ne or decreasing: not a counted increasing loop
  }

  // IV self-update: yield[k] = addi(after.arg[k], constant step > 0).
  unsigned k = ivArg.getArgNumber();
  auto add = yield.getOperand(k).getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  Value ivAfter = w.getAfterArguments()[k];
  Value step;
  if (add.getLhs() == ivAfter)
    step = add.getRhs();
  else if (add.getRhs() == ivAfter)
    step = add.getLhs();
  else
    return std::nullopt;
  std::optional<int64_t> stepC = getConstantIntValue(step);
  if (!stepC || *stepC <= 0)
    return std::nullopt;

  // Restricted to an index-typed IV whose result is unused.
  if (!ivArg.getType().isIndex() || !w.getResult(k).use_empty())
    return std::nullopt;

  return CountedWhileMatch{k, bound, step, inclusive};
}

struct RaiseCountedWhile : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp w,
                                PatternRewriter &rewriter) const override {
    std::optional<CountedWhileMatch> m = matchCountedWhile(w);
    if (!m)
      return failure();

    Location loc = w.getLoc();
    rewriter.setInsertionPoint(w);

    Value lb = w.getInits()[m->ivIndex];
    Value ub = m->bound;
    if (m->inclusive) {
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      ub = arith::AddIOp::create(rewriter, loc, m->bound, one);
    }

    // Carried (non-IV) iter-args, in order.
    SmallVector<Value> carriedInits;
    SmallVector<unsigned> carriedIdx;
    for (auto [i, in] : llvm::enumerate(w.getInits()))
      if (i != m->ivIndex) {
        carriedInits.push_back(in);
        carriedIdx.push_back(i);
      }

    scf::YieldOp whileYield = w.getYieldOp();
    Block &after = w.getAfter().front();

    // Move the after-region body into the for body, mapping the IV to the for's
    // induction var and the carried args to its iter-args. The IV self-update
    // clones as dead code (nothing yields it) and canonicalize removes it.
    auto build = [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
      IRMapping map;
      map.map(after.getArgument(m->ivIndex), iv);
      for (auto [r, idx] : llvm::enumerate(carriedIdx))
        map.map(after.getArgument(idx), iterArgs[r]);
      for (Operation &op : after.without_terminator())
        b.clone(op, map);
      SmallVector<Value> yields;
      for (unsigned idx : carriedIdx)
        yields.push_back(map.lookupOrDefault(whileYield.getOperand(idx)));
      scf::YieldOp::create(b, l, yields);
    };

    auto forOp =
        scf::ForOp::create(rewriter, loc, lb, ub, m->step, carriedInits, build);

    info(Stage::Prep, forOp)
        << "Raising counted while loop into a counted for loop";

    for (auto [r, idx] : llvm::enumerate(carriedIdx))
      rewriter.replaceAllUsesWith(w.getResult(idx), forOp.getResult(r));
    rewriter.eraseOp(w);
    return success();
  }
};

struct RaiseCountedWhilePass
    : public allo::impl::RaiseCountedWhilePassBase<RaiseCountedWhilePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseCountedWhile>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
