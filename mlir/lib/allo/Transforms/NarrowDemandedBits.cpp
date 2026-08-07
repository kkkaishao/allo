/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/BitAnalysis.h" // knownBits
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::allo {
#define GEN_PASS_DEF_NARROWDEMANDEDBITSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

// The two's-complement ring operators, whose low `w` result bits are a function
// of the low `w` bits of the operands alone, which is what makes sinking a
// truncation through them exact. Division, remainder, right shift and compare
// read the high bits, so the demand stops at those.
bool isRingOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
}

// trunc_w(a `op` b) -> trunc_w(a) `op` trunc_w(b), moving the truncation toward
// the leaves so the operator is built at the width its consumer reads. The
// truncations left behind meet the extends bit growth introduced and fold,
// exposing the next operator up to the same rewrite.
struct SinkTruncThroughRingOp : OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp trunc,
                                PatternRewriter &rewriter) const override {
    Operation *op = trunc.getIn().getDefiningOp();
    // Without a single use the wide result stays live, so the wide operator
    // survives and this only adds truncations.
    if (!op || !isRingOp(op) || !op->hasOneUse())
      return failure();

    Type narrow = trunc.getType();
    Location loc = op->getLoc();
    OperationState state(loc, op->getName());
    for (Value operand : op->getOperands())
      state.addOperands(
          arith::TruncIOp::create(rewriter, loc, narrow, operand).getResult());
    state.addTypes(narrow);
    rewriter.replaceOp(trunc, rewriter.create(state)->getResult(0));
    return success();
  }
};

// `x & y` -> `x` when every bit `y` would clear is already zero in `x`: a mask
// over a field the value cannot hold. Writing a bit field splices with such a
// mask on every field after the first, and the splices chain, so each mask
// removed takes a whole AND off the critical path.
struct DropRedundantMask : OpRewritePattern<arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<IntegerType>(op.getType()))
      return failure(); // an index mask has no width to reason in
    llvm::KnownBits lhs = knownBits(op.getLhs());
    llvm::KnownBits rhs = knownBits(op.getRhs());
    // Bit by bit: the mask keeps this one, or the value never sets it.
    if ((rhs.One | lhs.Zero).isAllOnes()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if ((lhs.One | rhs.Zero).isAllOnes()) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

struct NarrowDemandedBitsPass
    : public allo::impl::NarrowDemandedBitsPassBase<NarrowDemandedBitsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SinkTruncThroughRingOp, DropRedundantMask>(ctx);
    // The cast folds are what make the rewrite chain: without them a sunk
    // truncation stops on top of an extend instead of collapsing into it.
    arith::TruncIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtSIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtUIOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
