/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>

namespace mlir::allo {
#define GEN_PASS_DEF_LEGALIZEARITHPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

//===----------------------------------------------------------------------===//
// Bit fields -> integer arithmetic.
//
// `allo.bit.get_slice` / `set_slice` name a field of an integer, which no phase
// below this one models: the operator library prices arith and math, and the
// datapath realizes what that library covers. Expanding here, BEFORE the
// schedule is cut, is what lets the chaining solve see the field access at its
// real combinational depth; expanding at reify would grow a cone the cut never
// saw.
//
// The field WIDTH is the result type's (a get) or the value operand's (a set),
// so `hi` says nothing the width does not and is left to die. The OFFSET may be
// dynamic, which is why this shifts rather than selects: `comb.extract` takes
// its low bit as an ATTRIBUTE and cannot name a runtime one. A constant offset
// still arrives there, since a shift by a literal folds back into the extract /
// concat the field access really is.
//===----------------------------------------------------------------------===//

/// The offset when it is a literal. Worth asking because it makes every shift
/// and mask below constant, which is the difference between a field access that
/// costs the schedule a barrel shifter and one that costs it wiring.
std::optional<uint64_t> constantOffset(Value lo) {
  APInt cst;
  return matchPattern(lo, m_ConstantInt(&cst))
             ? std::optional<uint64_t>(cst.getZExtValue())
             : std::nullopt;
}

// result = trunc(src >> lo), at the result type's width.
struct LowerBitGetSlice : OpRewritePattern<BitGetSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BitGetSliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = cast<IntegerType>(src.getType());
    auto resTy = cast<IntegerType>(op.getType());

    std::optional<uint64_t> at = constantOffset(op.getLo());
    Value bits = src;
    if (!at || *at) { // a field at bit zero is already in place
      Value lo =
          at ? arith::ConstantOp::create(
                   rewriter, loc, rewriter.getIntegerAttr(srcTy, (int64_t)*at))
                   .getResult()
             : arith::IndexCastOp::create(rewriter, loc, srcTy, op.getLo())
                   .getResult();
      bits = arith::ShRUIOp::create(rewriter, loc, src, lo);
    }
    if (resTy.getWidth() < srcTy.getWidth())
      bits = arith::TruncIOp::create(rewriter, loc, resTy, bits);
    else if (resTy.getWidth() > srcTy.getWidth())
      bits = arith::ExtUIOp::create(rewriter, loc, resTy, bits);
    rewriter.replaceOp(op, bits);
    return success();
  }
};

// result = (src & ~(mask << lo)) | (value << lo), with `mask` the low `width`
// bits. The value needs no mask of its own: widening it to the source zeroes
// every bit outside the field, so only the hole it fills is masked.
struct LowerBitSetSlice : OpRewritePattern<BitSetSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BitSetSliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    auto srcTy = cast<IntegerType>(src.getType());
    unsigned srcW = srcTy.getWidth();
    unsigned valW = cast<IntegerType>(op.getValue().getType()).getWidth();
    unsigned width = std::min(valW, srcW);
    auto konst = [&](const APInt &v) -> Value {
      return arith::ConstantOp::create(rewriter, loc,
                                       rewriter.getIntegerAttr(srcTy, v));
    };

    Value value = op.getValue();
    if (valW < srcW)
      value = arith::ExtUIOp::create(rewriter, loc, srcTy, value);
    else if (valW > srcW)
      value = arith::TruncIOp::create(rewriter, loc, srcTy, value);

    std::optional<uint64_t> at = constantOffset(op.getLo());
    Value hole, placed = value;
    if (at) {
      // A field running off the top keeps only the bits that land, which is
      // what the shift does on the dynamic path too.
      hole = konst(
          ~APInt::getBitsSet(srcW, *at, std::min<uint64_t>(*at + width, srcW)));
      if (*at)
        placed = arith::ShLIOp::create(rewriter, loc, value,
                                       konst(APInt(srcW, *at)));
    } else {
      Value lo = arith::IndexCastOp::create(rewriter, loc, srcTy, op.getLo());
      Value mask = arith::ShLIOp::create(
          rewriter, loc, konst(APInt::getLowBitsSet(srcW, width)), lo);
      hole = arith::XOrIOp::create(rewriter, loc, mask,
                                   konst(APInt::getAllOnes(srcW)));
      placed = arith::ShLIOp::create(rewriter, loc, value, lo);
    }
    Value cleared = arith::AndIOp::create(rewriter, loc, src, hole);
    rewriter.replaceOp(op,
                       arith::OrIOp::create(rewriter, loc, cleared, placed));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant divisors -> shifts.
//
// A divide or a remainder by a power of two is not a divider: it is a shift and
// a mask.
//===----------------------------------------------------------------------===//

/// The base-2 log of \p divisor when it is a constant power of two. Only then
/// is every shift below by a literal, which is wiring rather than a barrel
/// shifter.
static std::optional<unsigned> powerOfTwoDivisor(Value divisor) {
  APInt cst;
  if (!matchPattern(divisor, m_ConstantInt(&cst)) || !cst.isPowerOf2())
    return std::nullopt;
  return cst.logBase2();
}

/// `x sdiv 2^k`, which is NOT `x >> k`: the shift floors and the division
/// truncates toward zero, so a negative numerator is nudged by `2^k - 1` first.
/// A compare, a select and an add, none of which is a divider.
static Value signedQuotient(PatternRewriter &rewriter, Location loc, Value x,
                            unsigned k) {
  Type ty = x.getType();
  auto cst = [&](uint64_t v) {
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getIntegerAttr(ty, v))
        .getResult();
  };
  Value zero = cst(0);
  Value isNeg =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, x, zero);
  Value bias = arith::SelectOp::create(rewriter, loc, isNeg,
                                       cst((uint64_t(1) << k) - 1), zero);
  return arith::ShRSIOp::create(
      rewriter, loc, arith::AddIOp::create(rewriter, loc, x, bias), cst(k));
}

struct ReduceDivUI : OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<unsigned> k = powerOfTwoDivisor(op.getRhs());
    if (!k)
      return failure();
    Value amount = arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getIntegerAttr(op.getType(), *k));
    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(), amount);
    return success();
  }
};

struct ReduceRemUI : OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<unsigned> k = powerOfTwoDivisor(op.getRhs());
    if (!k)
      return failure();
    Value mask = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(op.getType(), (uint64_t(1) << *k) - 1));
    rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(), mask);
    return success();
  }
};

struct ReduceDivSI : OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<unsigned> k = powerOfTwoDivisor(op.getRhs());
    if (!k)
      return failure();
    rewriter.replaceOp(op,
                       signedQuotient(rewriter, op.getLoc(), op.getLhs(), *k));
    return success();
  }
};

struct ReduceRemSI : OpRewritePattern<arith::RemSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<unsigned> k = powerOfTwoDivisor(op.getRhs());
    if (!k)
      return failure();
    Location loc = op.getLoc();
    Value x = op.getLhs();
    // The remainder takes the dividend's sign, which `x - (q << k)` already
    // does, so the quotient above is the only thing that needs care.
    Value q = signedQuotient(rewriter, loc, x, *k);
    Value amount = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(op.getType(), *k));
    rewriter.replaceOpWithNewOp<arith::SubIOp>(
        op, x, arith::ShLIOp::create(rewriter, loc, q, amount));
    return success();
  }
};

// The RTL-path, device-IP-aware replacement for `arith-expand`. A composite
// arith op the device can realize directly (a matching `dcp.operator`) is KEPT,
// so the scheduler binds it to that IP; every other one is EXPANDED into
// primitive arith by the upstream arith-expand patterns. Integer max/min are
// native combinational ops and are left alone (never marked illegal).
struct LegalizeArithPass
    : public allo::impl::LegalizeArithPassBase<LegalizeArithPass> {
  using LegalizeArithPassBase::LegalizeArithPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Built from the injected `dcp.device` / `dcp.operator` IR
    OperatorLibrary lib = OperatorLibrary::fromModule(module);

    // Reuse the upstream expansion patterns
    RewritePatternSet patterns(&getContext());
    arith::populateArithExpandOpsPatterns(patterns);
    patterns.add<LowerBitGetSlice, LowerBitSetSlice, ReduceDivUI, ReduceRemUI,
                 ReduceDivSI, ReduceRemSI>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    // A bit field is named by the frontend and modelled by nothing below, so it
    // never survives this pass.
    target.addIllegalOp<BitGetSliceOp, BitSetSliceOp>();

    // A composite op is legal (kept) iff the device realizes it directly;
    // otherwise it is illegal and the patterns decompose it into primitives.
    auto keepIfRealizable = [&lib](Operation *op) {
      return lib.hasDirectRealization(op);
    };
    target.addDynamicallyLegalOp<arith::CeilDivSIOp, arith::CeilDivUIOp,
                                 arith::FloorDivSIOp, arith::MaximumFOp,
                                 arith::MinimumFOp, arith::MaxNumFOp,
                                 arith::MinNumFOp>(keepIfRealizable);
    // A power-of-two divisor makes the op a shift, so it does not survive
    // either. Every other divisor stays: the device has a core for it, or it
    // is priced as the divider it really is.
    target.addDynamicallyLegalOp<arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
                                 arith::RemUIOp>([](Operation *op) {
      return !powerOfTwoDivisor(op->getOperand(1)).has_value();
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
