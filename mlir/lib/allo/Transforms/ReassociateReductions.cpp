/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_REASSOCIATEREDUCTIONSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A loop-carried iter_arg (not the induction variable) of an enclosing
// affine.for. Such an accumulator is folded in last so its recurrence stays a
// single operator.
bool isLoopCarried(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg)
    return false;
  auto forOp = dyn_cast<affine::AffineForOp>(arg.getOwner()->getParentOp());
  return forOp && llvm::is_contained(forOp.getRegionIterArgs(), v);
}

struct ReductionChain {
  SmallVector<ReductionStep> steps; // chain steps, tail first
  SmallVector<Value> leaves;        // the operands the chain folds together
};

// Flatten the maximal chain of `proto`'s operator: recursively absorb any
// single-use step of the same operator/idiom, collecting every non-chain
// operand (peeled through the idiom's extends) as a leaf. Absorbed steps are
// recorded so their ops can be erased once the chain is rebalanced.
void flatten(Value v, const ReductionStep &proto, ReductionChain &chain) {
  ReductionStep s = matchReductionStep(v);
  if (s && sameReduction(s, proto) && v.hasOneUse()) {
    chain.steps.push_back(s);
    auto [a, b] = reductionOperands(s);
    flatten(a, proto, chain);
    flatten(b, proto, chain);
    return;
  }
  chain.leaves.push_back(v);
}

// Erase a rewritten step's ops (idiom: trunc, core, both extends), once their
// results are dead. Steps are erased tail-first, so each op is use-empty by the
// time it is reached.
void eraseStep(RewriterBase &b, const ReductionStep &s) {
  Operation *e0 = s.widened() ? s.core->getOperand(0).getDefiningOp() : nullptr;
  Operation *e1 = s.widened() ? s.core->getOperand(1).getDefiningOp() : nullptr;
  for (Operation *op : {s.trunc, s.core, e0, e1})
    if (op && op->use_empty())
      b.eraseOp(op);
}

struct ReassociateReductionsPass
    : public allo::impl::ReassociateReductionsPassBase<
          ReassociateReductionsPass> {
  using ReassociateReductionsPassBase::ReassociateReductionsPassBase;

  void runOnOperation() override {
    // Process tails first (reverse program order) so each chain is rebalanced
    // from its outermost step inward and its absorbed links are skipped. Only
    // the integer widening idiom is exactly associative; float needs opt-in.
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (op->getNumResults() == 1 && matchReductionStep(op->getResult(0)))
        candidates.push_back(op);
    });

    DenseSet<Operation *> consumed;
    IRRewriter b(&getContext());
    for (Operation *op : llvm::reverse(candidates)) {
      if (consumed.contains(op))
        continue;
      ReductionStep tail = matchReductionStep(op->getResult(0));
      if (!tail.widened() && !floatReassoc) // a bare step is always float
        continue;

      ReductionChain chain;
      chain.steps.push_back(tail);
      auto [lhs, rhs] = reductionOperands(tail);
      flatten(lhs, tail, chain);
      flatten(rhs, tail, chain);
      if (chain.steps.size() < 2) // nothing absorbed: a lone step, no chain
        continue;

      // A loop-carried accumulator is folded in last so its recurrence spans
      // one operator; the remaining leaves form a balanced tree.
      SmallVector<Value> carried, rest;
      for (Value leaf : chain.leaves)
        (isLoopCarried(leaf) ? carried : rest).push_back(leaf);

      // Rewrite only when the depth strictly improves: a carried chain drops
      // its recurrence from N operators to 1; a straight-line chain drops its
      // depth from N to ceil(log2(N)).
      unsigned n = chain.leaves.size();
      bool improves = carried.empty() ? llvm::Log2_32_Ceil(n) < n - 1 : n >= 3;
      if (!improves)
        continue;

      for (const ReductionStep &s : chain.steps) {
        consumed.insert(s.core);
        if (s.trunc)
          consumed.insert(s.trunc);
      }

      b.setInsertionPoint(op);
      Value acc = rest.empty() ? buildBalancedTree(b, tail, carried)
                               : buildBalancedTree(b, tail, rest);
      if (!rest.empty())
        for (Value c : carried)
          acc = buildReductionStep(b, tail, acc, c);

      op->getResult(0).replaceAllUsesWith(acc);
      info(Stage::Prep, acc.getDefiningOp())
          << "Rebalancing associative reduction chain of " << n
          << " terms into a balanced tree";
      for (const ReductionStep &s : chain.steps)
        eraseStep(b, s);
    }
  }
};

} // namespace
