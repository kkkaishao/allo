/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // magicMultiplier
#include "allo/Scheduling/MemoryModel.h"  // kIndexWidth
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

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
// datapath realizes what that library covers. Expanding before the schedule is
// cut lets the chaining solve see the field access at its real combinational
// depth.
//
// The field width is the result type's (a get) or the value operand's (a set),
// so `hi` is not read. The offset may be dynamic, so this shifts rather than
// selects: `comb.extract` takes its low bit as an attribute and cannot name a
// runtime one. A shift by a literal folds back into the extract or concat the
// field access really is.
//===----------------------------------------------------------------------===//

/// The offset when it is a literal, which makes every shift and mask below
/// constant: wiring rather than a barrel shifter.
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

/// The base-2 log of \p divisor when it is a constant power of two.
static std::optional<unsigned> powerOfTwoDivisor(Value divisor) {
  APInt cst;
  if (!matchPattern(divisor, m_ConstantInt(&cst)) || !cst.isPowerOf2())
    return std::nullopt;
  return cst.logBase2();
}

/// `x sdiv 2^k`, which is not `x >> k`: the shift floors while the division
/// truncates toward zero, so a negative numerator is biased by `2^k - 1` first.
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
    // gives once the quotient truncates toward zero.
    Value q = signedQuotient(rewriter, loc, x, *k);
    Value amount = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(op.getType(), *k));
    rewriter.replaceOpWithNewOp<arith::SubIOp>(
        op, x, arith::ShLIOp::create(rewriter, loc, q, amount));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant divisors, general case -> reciprocal multiply.
//
// A constant divisor never needs a divider: `n divui d` is
// `(n * M) >> (w + ceil(log2 d))` with `M` the rounded-up reciprocal, exact
// for every `n` below `2^w`. The dividend's range decides `w`: an
// induction-variable expression usually proves a narrow bound, and the
// multiply then fits the clock. A signed division carries the sign around the
// magnitude's multiply, which preserves its truncation toward zero.
//
// Who expands is a fit question. An index op matches no device IP row (a row
// is declared at concrete widths) and is priced as a full-width combinational
// divider, one operator no register can split, so it expands regardless. A
// typed op has a pipelined divider IP to fall back on, so it expands only
// where the reciprocal's product multiply fits the stated clock period, and
// keeps the IP everywhere else.
//===----------------------------------------------------------------------===//

/// Largest value the expression \p v can take, read as unsigned bits: walked
/// over constants, affine induction variables and the arithmetic an index
/// expression is built from. nullopt where the walk meets anything it cannot
/// bound.
static std::optional<uint64_t> unsignedBound(Value v, unsigned depth = 0) {
  if (depth > 8)
    return std::nullopt;
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.getActiveBits() > 63
               ? std::nullopt
               : std::optional<uint64_t>(cst.getZExtValue());
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    auto loop = dyn_cast<affine::AffineForOp>(barg.getOwner()->getParentOp());
    if (loop && barg == loop.getInductionVar() &&
        loop.hasConstantLowerBound() && loop.hasConstantUpperBound() &&
        loop.getConstantLowerBound() >= 0 &&
        loop.getConstantUpperBound() > loop.getConstantLowerBound())
      return static_cast<uint64_t>(loop.getConstantUpperBound()) - 1;
    return std::nullopt;
  }
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto operand = [&](unsigned k) {
    return unsignedBound(op->getOperand(k), depth + 1);
  };
  // The right operand when it is a constant below 2^63.
  auto konst = [&]() -> std::optional<uint64_t> {
    APInt k;
    if (matchPattern(op->getOperand(1), m_ConstantInt(&k)) &&
        k.getActiveBits() <= 63)
      return k.getZExtValue();
    return std::nullopt;
  };
  // Everything the result type can hold: a truncation wraps rather than
  // clamps, but whatever it wraps to still fits the type.
  auto typeMax = [&]() -> uint64_t {
    Type t = op->getResult(0).getType();
    unsigned w = t.isIndex() ? 64 : t.getIntOrFloatBitWidth();
    return w >= 64 ? UINT64_MAX : (uint64_t(1) << w) - 1;
  };
  return llvm::TypeSwitch<Operation *, std::optional<uint64_t>>(op)
      .Case<arith::AddIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0), b = operand(1);
        if (!a || !b)
          return std::nullopt;
        return llvm::SaturatingAdd(*a, *b);
      })
      .Case<arith::MulIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0), b = operand(1);
        if (!a || !b)
          return std::nullopt;
        return llvm::SaturatingMultiply(*a, *b);
      })
      .Case<arith::ShLIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0);
        auto k = konst();
        if (!a || !k || *k >= 64)
          return std::nullopt;
        return llvm::SaturatingMultiply(*a, uint64_t(1) << *k);
      })
      .Case<arith::ShRUIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0);
        auto k = konst();
        return a && k && *k < 64 ? std::optional<uint64_t>(*a >> *k) : a;
      })
      .Case<arith::DivUIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0);
        auto d = konst();
        return a && d && *d ? std::optional<uint64_t>(*a / *d) : a;
      })
      .Case<arith::RemUIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0);
        auto d = konst();
        if (d && *d)
          return std::min(a.value_or(UINT64_MAX), *d - 1);
        return std::nullopt;
      })
      .Case<arith::AndIOp>([&](auto) -> std::optional<uint64_t> {
        auto a = operand(0);
        auto m = konst();
        if (m)
          return std::min(a.value_or(UINT64_MAX), *m);
        return a;
      })
      .Case<arith::SelectOp>([&](auto) -> std::optional<uint64_t> {
        auto a = unsignedBound(op->getOperand(1), depth + 1);
        auto b = unsignedBound(op->getOperand(2), depth + 1);
        if (!a || !b)
          return std::nullopt;
        return std::max(*a, *b);
      })
      .Case<arith::IndexCastUIOp, arith::ExtUIOp, arith::TruncIOp>(
          [&](auto) -> std::optional<uint64_t> {
            return std::min(operand(0).value_or(UINT64_MAX), typeMax());
          })
      .Default([](auto) { return std::nullopt; });
}

/// Bits of the dividend's proven range, capped at \p cap bits.
static unsigned boundWidth(Value n, unsigned cap) {
  uint64_t bound = std::min<uint64_t>(
      unsignedBound(n).value_or(UINT64_MAX),
      cap >= 64 ? UINT64_MAX : (uint64_t(1) << cap) - 1);
  return bound ? llvm::Log2_64(bound) + 1 : 1;
}

/// One planned expansion: the divisor, the width `w` the reciprocal form is
/// exact below, and whether the dividend is proven non-negative, which lets a
/// signed op take the unsigned form directly.
struct MagicPlan {
  uint64_t divisor;
  unsigned width;
  bool nonneg;
  /// Whether the whole dividend range sits below the divisor, where the
  /// quotient folds to zero and the remainder to the dividend, no multiply
  /// built.
  bool folds() const {
    return width < 64 && ((uint64_t(1) << width) - 1) < divisor;
  }
};

/// The plan when the reciprocal patterns apply to \p op, or nullopt. Shared by
/// the patterns and the conversion target, so what is marked illegal is
/// exactly what rewrites.
///
/// TODO: a negative constant divisor keeps its op today; it could expand as
/// the magnitude's reciprocal with one more negation of the quotient.
static std::optional<MagicPlan> magicPlan(Operation *op, bool isSigned,
                                          unsigned maxMulWidth) {
  APInt cst;
  if (!matchPattern(op->getOperand(1), m_ConstantInt(&cst)) ||
      (isSigned && cst.isNegative()) || cst.getActiveBits() > 32)
    return std::nullopt;
  uint64_t d = cst.getZExtValue();
  if (d == 0 || llvm::isPowerOf2_64(d))
    return std::nullopt;
  Type t = op->getResult(0).getType();
  if (t.isIndex()) {
    if (isSigned)
      return std::nullopt;
    return MagicPlan{d, boundWidth(op->getOperand(0), kIndexWidth), true};
  }
  auto it = dyn_cast<IntegerType>(t);
  if (!it || it.getWidth() < 2 || it.getWidth() > 64)
    return std::nullopt;
  unsigned n = it.getWidth();
  MagicPlan p{d, 0, true};
  if (isSigned) {
    // A magnitude takes one bit more than the non-negative range, so a signed
    // dividend computes at the full type width unless proven non-negative.
    std::optional<uint64_t> b = unsignedBound(op->getOperand(0));
    p.nonneg = b && n < 64 && *b < (uint64_t(1) << (n - 1));
    p.width = p.nonneg ? boundWidth(op->getOperand(0), n) : n;
  } else {
    p.width = boundWidth(op->getOperand(0), n);
  }
  if (p.folds() || 2 * p.width + 1 <= maxMulWidth)
    return p;
  return std::nullopt;
}

/// Zero-extend or truncate \p v to \p width bits, index through the unsigned
/// cast. A truncation is only built over a value proven to fit.
static Value resizeUInt(PatternRewriter &rewriter, Location loc, Value v,
                        unsigned width) {
  Type t = rewriter.getIntegerType(width);
  if (v.getType().isIndex())
    return arith::IndexCastUIOp::create(rewriter, loc, t, v);
  unsigned n = cast<IntegerType>(v.getType()).getWidth();
  if (n == width)
    return v;
  if (n < width)
    return arith::ExtUIOp::create(rewriter, loc, t, v);
  return arith::TruncIOp::create(rewriter, loc, t, v);
}

/// \p v back at \p type, the op's own result type.
static Value resizeToType(PatternRewriter &rewriter, Location loc, Value v,
                          Type type) {
  if (type.isIndex())
    return arith::IndexCastUIOp::create(rewriter, loc, type, v);
  return resizeUInt(rewriter, loc, v, cast<IntegerType>(type).getWidth());
}

/// A constant of \p type carrying the low bits of \p k.
static Value konstOf(PatternRewriter &rewriter, Location loc, Type type,
                     uint64_t k) {
  unsigned w = cast<IntegerType>(type).getWidth();
  return arith::ConstantOp::create(rewriter, loc,
                                   IntegerAttr::get(type, APInt(w, k)))
      .getResult();
}

/// `(n * magic) >> shift` at the product's width, `2w+1` bits. Only called
/// with `d <= 2^w - 1`, which keeps the shift below that width.
static Value magicQuotient(PatternRewriter &rewriter, Location loc, Value n,
                           uint64_t d, unsigned w) {
  unsigned shift;
  uint64_t magic = magicMultiplier(d, w, shift);
  Type wide = rewriter.getIntegerType(2 * w + 1);
  Value nw = resizeUInt(rewriter, loc, n, 2 * w + 1);
  Value prod = arith::MulIOp::create(rewriter, loc, nw,
                                     konstOf(rewriter, loc, wide, magic));
  return arith::ShRUIOp::create(rewriter, loc, prod,
                                konstOf(rewriter, loc, wide, shift));
}

/// `n - (n divui d) * d`, all in the plan's width: the quotient times the
/// divisor never exceeds the dividend.
static Value magicRemainder(PatternRewriter &rewriter, Location loc, Value n,
                            const MagicPlan &p, Type type) {
  Type narrow = rewriter.getIntegerType(p.width);
  Value q = arith::TruncIOp::create(
      rewriter, loc, narrow,
      magicQuotient(rewriter, loc, n, p.divisor, p.width));
  Value nn = resizeUInt(rewriter, loc, n, p.width);
  Value qd = arith::MulIOp::create(
      rewriter, loc, q, konstOf(rewriter, loc, narrow, p.divisor));
  Value r = arith::SubIOp::create(rewriter, loc, nn, qd);
  return resizeToType(rewriter, loc, r, type);
}

/// The signed quotient at the op's own type: the sign carried around the
/// magnitude's reciprocal multiply, which preserves truncation toward zero.
/// `|INT_MIN|` survives as its own bit pattern, the magnitude read unsigned.
static Value signedMagicQuotient(PatternRewriter &rewriter, Location loc,
                                 Value n, const MagicPlan &p, Type type) {
  Value zero = konstOf(rewriter, loc, type, 0);
  Value isNeg = arith::CmpIOp::create(rewriter, loc,
                                      arith::CmpIPredicate::slt, n, zero);
  Value neg = arith::SubIOp::create(rewriter, loc, zero, n);
  Value mag = arith::SelectOp::create(rewriter, loc, isNeg, neg, n);
  Value qa = resizeToType(
      rewriter, loc, magicQuotient(rewriter, loc, mag, p.divisor, p.width),
      type);
  Value qneg = arith::SubIOp::create(rewriter, loc, zero, qa);
  return arith::SelectOp::create(rewriter, loc, isNeg, qneg, qa);
}

template <typename OpT> struct MagicBase : OpRewritePattern<OpT> {
  MagicBase(MLIRContext *ctx, unsigned maxMulWidth)
      : OpRewritePattern<OpT>(ctx), maxMulWidth(maxMulWidth) {}
  unsigned maxMulWidth;
};

struct MagicDivUI : MagicBase<arith::DivUIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p =
        magicPlan(op, /*isSigned=*/false, maxMulWidth);
    if (!p)
      return failure();
    if (p->folds()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, op.getType(), rewriter.getZeroAttr(op.getType()));
      return success();
    }
    Value q =
        magicQuotient(rewriter, op.getLoc(), op.getLhs(), p->divisor, p->width);
    rewriter.replaceOp(op,
                       resizeToType(rewriter, op.getLoc(), q, op.getType()));
    return success();
  }
};

struct MagicRemUI : MagicBase<arith::RemUIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p =
        magicPlan(op, /*isSigned=*/false, maxMulWidth);
    if (!p)
      return failure();
    if (p->folds()) {
      rewriter.replaceOp(op, op.getLhs()); // n < d, so the remainder is n
      return success();
    }
    rewriter.replaceOp(op, magicRemainder(rewriter, op.getLoc(), op.getLhs(),
                                          *p, op.getType()));
    return success();
  }
};

struct MagicDivSI : MagicBase<arith::DivSIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p = magicPlan(op, /*isSigned=*/true, maxMulWidth);
    if (!p)
      return failure();
    Location loc = op.getLoc();
    if (p->folds()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, op.getType(), rewriter.getZeroAttr(op.getType()));
      return success();
    }
    Value q = p->nonneg
                  ? resizeToType(rewriter, loc,
                                 magicQuotient(rewriter, loc, op.getLhs(),
                                               p->divisor, p->width),
                                 op.getType())
                  : signedMagicQuotient(rewriter, loc, op.getLhs(), *p,
                                        op.getType());
    rewriter.replaceOp(op, q);
    return success();
  }
};

struct MagicRemSI : MagicBase<arith::RemSIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p = magicPlan(op, /*isSigned=*/true, maxMulWidth);
    if (!p)
      return failure();
    Location loc = op.getLoc();
    if (p->folds()) {
      rewriter.replaceOp(op, op.getLhs()); // |n| < d, so the remainder is n
      return success();
    }
    if (p->nonneg) {
      rewriter.replaceOp(op, magicRemainder(rewriter, loc, op.getLhs(), *p,
                                            op.getType()));
      return success();
    }
    // n - (n divsi d) * d at the full width: the sign follows the dividend,
    // and the product never exceeds it, so nothing wraps.
    Value q = signedMagicQuotient(rewriter, loc, op.getLhs(), *p, op.getType());
    Value qd = arith::MulIOp::create(
        rewriter, loc, q, konstOf(rewriter, loc, op.getType(), p->divisor));
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getLhs(), qd);
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

    // The widest multiply the clock takes whole, register floor included: a
    // typed division keeps its divider IP where the reciprocal's product
    // multiply would not fit. With no period stated every typed division
    // stays on its IP; an index division has no IP and expands regardless.
    unsigned maxMulWidth = 0;
    if (expandConstDiv && periodNs > 0.0)
      for (unsigned w = 2; w <= 129; ++w)
        if (std::optional<double> delay = lib.measuredCombDelay(OpKind::Mul, w))
          if (*delay <= periodNs)
            maxMulWidth = w;

    // Reuse the upstream expansion patterns
    RewritePatternSet patterns(&getContext());
    arith::populateArithExpandOpsPatterns(patterns);
    patterns.add<LowerBitGetSlice, LowerBitSetSlice, ReduceDivUI, ReduceRemUI,
                 ReduceDivSI, ReduceRemSI>(&getContext());
    if (expandConstDiv)
      patterns.add<MagicDivUI, MagicRemUI, MagicDivSI, MagicRemSI>(
          &getContext(), maxMulWidth);

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
    // either; a constant one the reciprocal patterns plan for goes to them
    // when the flag asks. Every other divisor stays and is bound to a device
    // core or priced as a divider.
    bool expand = expandConstDiv;
    target.addDynamicallyLegalOp<arith::DivUIOp, arith::RemUIOp>(
        [expand, maxMulWidth](Operation *op) {
          if (powerOfTwoDivisor(op->getOperand(1)))
            return false;
          return !(expand && magicPlan(op, /*isSigned=*/false, maxMulWidth));
        });
    target.addDynamicallyLegalOp<arith::DivSIOp, arith::RemSIOp>(
        [expand, maxMulWidth](Operation *op) {
          if (powerOfTwoDivisor(op->getOperand(1)))
            return false;
          return !(expand && magicPlan(op, /*isSigned=*/true, maxMulWidth));
        });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
