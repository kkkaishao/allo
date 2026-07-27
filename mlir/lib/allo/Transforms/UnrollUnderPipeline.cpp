/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::allo {
#define GEN_PASS_DEF_UNROLLUNDERPIPELINEPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

static bool isCounted(Operation *op) {
  return isa<affine::AffineForOp, scf::ForOp>(op);
}

// A loop pipelined by `s.pipeline(ii != -1)`; its body must become loop-free.
static bool isPipelined(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>(kPipelineIIAttr);
  return attr && attr.getInt() != -1;
}

// A counted loop whose trip count is a compile-time constant (fully
// unrollable).
static bool isConstantTrip(Operation *op) {
  if (auto af = dyn_cast<affine::AffineForOp>(op))
    return affine::getConstantTripCount(af).has_value();
  auto sf = cast<scf::ForOp>(op);
  std::optional<int64_t> lb = getConstantIntValue(sf.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(sf.getUpperBound());
  std::optional<int64_t> st = getConstantIntValue(sf.getStep());
  return lb && ub && st && *st != 0;
}

// A leaf loop strictly inside `root` (a counted loop with no further nested
// loop), or null if `root` has no nested loop.
static Operation *innermostNestedLoop(Operation *root) {
  Operation *leaf = nullptr;
  root->walk([&](Operation *op) {
    if (op == root || !isCounted(op))
      return WalkResult::advance();
    bool hasInner = false;
    op->walk([&](Operation *n) {
      if (n != op && isCounted(n)) {
        hasInner = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasInner) {
      leaf = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return leaf;
}

// Every loop strictly inside `root` is a constant-trip counted loop.
static bool innerLoopsUnrollable(Operation *root) {
  bool ok = true;
  root->walk([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op) &&
        !(isCounted(op) && isConstantTrip(op))) {
      ok = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return ok;
}

struct UnrollUnderPipelinePass
    : public allo::impl::UnrollUnderPipelinePassBase<UnrollUnderPipelinePass> {
  void runOnOperation() override {
    // Outermost pipelined loops only (a pipelined loop nested in another
    // pipelined one is consumed when the outer one is unrolled).
    SmallVector<Operation *> targets;
    getOperation().walk([&](Operation *op) {
      if (!isPipelined(op))
        return;
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
        if (isPipelined(p))
          return;
      targets.push_back(op);
    });

    for (Operation *loop : targets) {
      if (!innermostNestedLoop(loop))
        continue; // already a single loop, nothing to unroll
      if (!innerLoopsUnrollable(loop)) {
        warn(Stage::Prep, loop)
            << "Pipelined loop has a dynamic or uncounted inner loop; not "
               "unrolled, so it falls back to pipelining the innermost loop "
               "only";
        continue;
      }
      auto loc =
          cast<LoopLikeOpInterface>(loop).getSingleInductionVar()->getLoc();
      auto loopName = logging::detail::describe(loc);
      if (loopName.empty())
        loopName = "<unnamed>";
      // Fully unroll the inner loops, innermost-first; each unroll strictly
      // reduces the nested-loop count (a leaf has no loop to replicate).
      while (Operation *leaf = innermostNestedLoop(loop)) {
        LogicalResult r =
            isa<affine::AffineForOp>(leaf)
                ? affine::loopUnrollFull(cast<affine::AffineForOp>(leaf))
                : mlir::loopUnrollFull(cast<scf::ForOp>(leaf));
        assert(succeeded(r) && "constant-trip loop must fully unroll");
        info(Stage::Prep, leaf)
            << "Automatically fully unrolled the loop implied by pipelining on "
            << loopName;
        (void)r;
      }
    }
  }
};

} // namespace
