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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DivisionByConstantInfo.h"
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
// divider, one operator no register can split, so it expands regardless; one
// the reciprocal cannot take moves to the typed width instead
// (`IndexDivToTyped`). A typed op has a pipelined divider IP to fall back on,
// so it expands only where the reciprocal's product multiply fits the stated
// clock period, and keeps the IP everywhere else.
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
/// signed op take the unsigned form directly. `viaIp` picks the classic mulh
/// form whose `2w`-bit product a pipelined multiplier row carries registered,
/// where the narrow form's combinational multiply would not fit the clock.
struct MagicPlan {
  uint64_t divisor;
  unsigned width;
  bool nonneg;
  bool viaIp = false;
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
                                          unsigned maxMulWidth,
                                          unsigned maxIpMulWidth) {
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
      return std::nullopt; // `IndexDivToTyped` routes it through the i32 form
    MagicPlan p{d, boundWidth(op->getOperand(0), kIndexWidth), true};
    // The narrow form regardless when nothing better fits: even its wide
    // combinational multiply beats the full combinational divider an `index`
    // op would otherwise become.
    if (!p.folds() && 2 * p.width + 1 > maxMulWidth &&
        2 * kIndexWidth <= maxIpMulWidth) {
      p.viaIp = true;
      p.width = kIndexWidth;
    }
    return p;
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
  if (2 * n <= maxIpMulWidth) {
    p.viaIp = true;
    p.width = n;
    return p;
  }
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
  unsigned w = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
  return arith::ConstantOp::create(rewriter, loc,
                                   IntegerAttr::get(type, APInt(w, k)))
      .getResult();
}

/// One nonzero digit of a literal factor's non-adjacent form.
struct NafDigit {
  unsigned shift;
  bool negative;
};

/// The digits of \p cst's non-adjacent form, where the shift-add network is
/// small enough to beat a multiplier. Digits at or above the factor's width
/// contribute nothing modulo two to that width and are dropped, which keeps
/// the recoding exact for either sign; a wider factor keeps its multiplier,
/// which a DSP slice serves better than a deep adder tree.
static std::optional<SmallVector<NafDigit, 5>> nafDigits(const APInt &cst) {
  constexpr unsigned kMaxNafAdders = 3;
  unsigned width = cst.getBitWidth();
  SmallVector<NafDigit, 5> digits;
  APInt v = cst.sext(width + 2);
  for (unsigned i = 0; !v.isZero(); ++i, v.ashrInPlace(1)) {
    if (!v[0])
      continue;
    bool neg = v[1]; // v = 3 mod 4 takes digit -1 so the next digit is zero
    if (i < width)
      digits.push_back({i, neg});
    if (neg)
      v += 1;
    else
      v -= 1;
  }
  if (digits.empty())
    return std::nullopt; // a zero factor is the folder's, not a network
  bool positive = llvm::any_of(digits, [](NafDigit d) { return !d.negative; });
  if (digits.size() - 1 + (positive ? 0 : 1) > kMaxNafAdders)
    return std::nullopt;
  return digits;
}

/// The shift-add network of \p digits over \p x, at \p ty. Positive digits
/// first, so the accumulator starts without a negate; a factor with none
/// starts the subtractions from zero.
static Value nafBuild(PatternRewriter &rewriter, Location loc, Value x,
                      ArrayRef<NafDigit> digits, Type ty) {
  auto term = [&](NafDigit d) -> Value {
    if (d.shift == 0)
      return x;
    return arith::ShLIOp::create(rewriter, loc, x,
                                 konstOf(rewriter, loc, ty, d.shift));
  };
  Value acc;
  for (NafDigit d : digits)
    if (!d.negative)
      acc = acc ? Value(arith::AddIOp::create(rewriter, loc, acc, term(d)))
                : term(d);
  if (!acc)
    acc = konstOf(rewriter, loc, ty, 0);
  for (NafDigit d : digits)
    if (d.negative)
      acc = arith::SubIOp::create(rewriter, loc, acc, term(d));
  return acc;
}

/// `x * d` at \p ty: the shift-add network where it is small, else a multiply
/// left to operator selection.
static Value mulByConst(PatternRewriter &rewriter, Location loc, Value x,
                        uint64_t d, Type ty) {
  unsigned w = ty.isIndex() ? 64 : cast<IntegerType>(ty).getWidth();
  if (auto digits = nafDigits(APInt(w, d)))
    return nafBuild(rewriter, loc, x, *digits, ty);
  return arith::MulIOp::create(rewriter, loc, x,
                               konstOf(rewriter, loc, ty, d));
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

/// `n - q * d` in the plan's width, \p q already there: the quotient times the
/// divisor never exceeds the dividend.
static Value remainderFrom(PatternRewriter &rewriter, Location loc, Value n,
                           Value q, const MagicPlan &p, Type type) {
  Value nn = resizeUInt(rewriter, loc, n, p.width);
  Value qd = mulByConst(rewriter, loc, q, p.divisor, q.getType());
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

//===----------------------------------------------------------------------===//
// The classic mulh forms: the product built at `2w`, which a pipelined
// multiplier row carries registered, so the clock never sees it whole. The
// magic data comes from LLVM's division-by-constant machinery; the shapes are
// Hacker's Delight 10-3 / 10-8.
//===----------------------------------------------------------------------===//

/// The unsigned quotient at width `p.width`, returned in that width.
static Value ipMagicQuotientU(PatternRewriter &rewriter, Location loc,
                              Value n0, const MagicPlan &p) {
  unsigned w = p.width;
  auto info = llvm::UnsignedDivisionByConstantInfo::get(APInt(w, p.divisor));
  Type nty = rewriter.getIntegerType(w);
  Type wide = rewriter.getIntegerType(2 * w);
  Value n = resizeUInt(rewriter, loc, n0, w);
  if (info.PreShift)
    n = arith::ShRUIOp::create(rewriter, loc, n,
                               konstOf(rewriter, loc, nty, info.PreShift));
  Value prod = arith::MulIOp::create(
      rewriter, loc, arith::ExtUIOp::create(rewriter, loc, wide, n),
      konstOf(rewriter, loc, wide, info.Magic.getZExtValue()));
  if (!info.IsAdd)
    return arith::TruncIOp::create(
        rewriter, loc, nty,
        arith::ShRUIOp::create(
            rewriter, loc, prod,
            konstOf(rewriter, loc, wide, w + info.PostShift)));
  // The magic overflowed one bit: q = ((n - t)/2 + t) >> s, where LLVM's
  // PostShift already carries the halving's one.
  Value t = arith::TruncIOp::create(
      rewriter, loc, nty,
      arith::ShRUIOp::create(rewriter, loc, prod,
                             konstOf(rewriter, loc, wide, w)));
  Value diff = arith::SubIOp::create(rewriter, loc, n, t);
  Value sum = arith::AddIOp::create(
      rewriter, loc,
      arith::ShRUIOp::create(rewriter, loc, diff,
                             konstOf(rewriter, loc, nty, 1)),
      t);
  return arith::ShRUIOp::create(
      rewriter, loc, sum, konstOf(rewriter, loc, nty, info.PostShift));
}

/// The signed quotient at the op's own integer type, `p.width` wide: mulhs,
/// the add-back where the magic is negative, and the final round toward zero
/// off the quotient's own sign. `INT_MIN` needs no carve-out here.
static Value ipMagicQuotientS(PatternRewriter &rewriter, Location loc,
                              Value n, const MagicPlan &p) {
  unsigned w = p.width;
  assert(n.getType() == rewriter.getIntegerType(w) &&
         "the signed form runs at the type width; index went through i32");
  auto info = llvm::SignedDivisionByConstantInfo::get(APInt(w, p.divisor));
  Type nty = n.getType();
  Type wide = rewriter.getIntegerType(2 * w);
  Value prod = arith::MulIOp::create(
      rewriter, loc, arith::ExtSIOp::create(rewriter, loc, wide, n),
      konstOf(rewriter, loc, wide, (uint64_t)info.Magic.getSExtValue()));
  Value q = arith::TruncIOp::create(
      rewriter, loc, nty,
      arith::ShRSIOp::create(rewriter, loc, prod,
                             konstOf(rewriter, loc, wide, w)));
  if (info.Magic.isNegative())
    q = arith::AddIOp::create(rewriter, loc, q, n);
  if (info.ShiftAmount)
    q = arith::ShRSIOp::create(
        rewriter, loc, q, konstOf(rewriter, loc, nty, info.ShiftAmount));
  return arith::AddIOp::create(
      rewriter, loc, q,
      arith::ShRUIOp::create(rewriter, loc, q,
                             konstOf(rewriter, loc, nty, w - 1)));
}

template <typename OpT> struct MagicBase : OpRewritePattern<OpT> {
  MagicBase(MLIRContext *ctx, unsigned maxMulWidth, unsigned maxIpMulWidth)
      : OpRewritePattern<OpT>(ctx), maxMulWidth(maxMulWidth),
        maxIpMulWidth(maxIpMulWidth) {}
  unsigned maxMulWidth;
  unsigned maxIpMulWidth;

  /// The unsigned quotient in the plan's width, whichever form the plan asks.
  Value quotientU(PatternRewriter &rewriter, Location loc, Value n,
                  const MagicPlan &p) const {
    if (p.viaIp)
      return ipMagicQuotientU(rewriter, loc, n, p);
    return arith::TruncIOp::create(
        rewriter, loc, rewriter.getIntegerType(p.width),
        magicQuotient(rewriter, loc, n, p.divisor, p.width));
  }
};

struct MagicDivUI : MagicBase<arith::DivUIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p =
        magicPlan(op, /*isSigned=*/false, maxMulWidth, maxIpMulWidth);
    if (!p)
      return failure();
    if (p->folds()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, op.getType(), rewriter.getZeroAttr(op.getType()));
      return success();
    }
    Value q = quotientU(rewriter, op.getLoc(), op.getLhs(), *p);
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
        magicPlan(op, /*isSigned=*/false, maxMulWidth, maxIpMulWidth);
    if (!p)
      return failure();
    if (p->folds()) {
      rewriter.replaceOp(op, op.getLhs()); // n < d, so the remainder is n
      return success();
    }
    Location loc = op.getLoc();
    Value q = quotientU(rewriter, loc, op.getLhs(), *p);
    rewriter.replaceOp(
        op, remainderFrom(rewriter, loc, op.getLhs(), q, *p, op.getType()));
    return success();
  }
};

struct MagicDivSI : MagicBase<arith::DivSIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p =
        magicPlan(op, /*isSigned=*/true, maxMulWidth, maxIpMulWidth);
    if (!p)
      return failure();
    Location loc = op.getLoc();
    if (p->folds()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, op.getType(), rewriter.getZeroAttr(op.getType()));
      return success();
    }
    Value q;
    if (p->nonneg)
      q = resizeToType(rewriter, loc,
                       quotientU(rewriter, loc, op.getLhs(), *p),
                       op.getType());
    else if (p->viaIp)
      q = ipMagicQuotientS(rewriter, loc, op.getLhs(), *p);
    else
      q = signedMagicQuotient(rewriter, loc, op.getLhs(), *p, op.getType());
    rewriter.replaceOp(op, q);
    return success();
  }
};

struct MagicRemSI : MagicBase<arith::RemSIOp> {
  using MagicBase::MagicBase;
  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<MagicPlan> p =
        magicPlan(op, /*isSigned=*/true, maxMulWidth, maxIpMulWidth);
    if (!p)
      return failure();
    Location loc = op.getLoc();
    if (p->folds()) {
      rewriter.replaceOp(op, op.getLhs()); // |n| < d, so the remainder is n
      return success();
    }
    if (p->nonneg) {
      Value q = quotientU(rewriter, loc, op.getLhs(), *p);
      rewriter.replaceOp(
          op, remainderFrom(rewriter, loc, op.getLhs(), q, *p, op.getType()));
      return success();
    }
    // n - (n divsi d) * d at the full width: the sign follows the dividend,
    // and the product never exceeds it, so nothing wraps.
    Value q = p->viaIp
                  ? ipMagicQuotientS(rewriter, loc, op.getLhs(), *p)
                  : signedMagicQuotient(rewriter, loc, op.getLhs(), *p,
                                        op.getType());
    Value qd = mulByConst(rewriter, loc, q, p->divisor, op.getType());
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getLhs(), qd);
    return success();
  }
};

// An `index` division no other pattern takes is rebuilt at `kIndexWidth`, the
// width the datapath gives `index` anyway. At a concrete type it binds a
// pipelined divider core; left at `index` it has no row to match and becomes
// a full-width combinational divider that derates the whole module's clock.
struct IndexDivToTyped : RewritePattern {
  IndexDivToTyped(MLIRContext *ctx, bool expand, unsigned maxMulWidth,
                  unsigned maxIpMulWidth)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        expand(expand), maxMulWidth(maxMulWidth),
        maxIpMulWidth(maxIpMulWidth) {}
  bool expand;
  unsigned maxMulWidth;
  unsigned maxIpMulWidth;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp, arith::RemSIOp>(
            op) ||
        !op->getResult(0).getType().isIndex())
      return failure();
    bool isSigned = isa<arith::DivSIOp, arith::RemSIOp>(op);
    // A power-of-two divisor is a shift and a planned constant one a
    // reciprocal; both lower without any divider.
    if (powerOfTwoDivisor(op->getOperand(1)) ||
        (expand && magicPlan(op, isSigned, maxMulWidth, maxIpMulWidth)))
      return failure();
    Location loc = op->getLoc();
    Type ty = rewriter.getIntegerType(kIndexWidth);
    auto shrink = [&](Value v) -> Value {
      if (isSigned)
        return arith::IndexCastOp::create(rewriter, loc, ty, v);
      return arith::IndexCastUIOp::create(rewriter, loc, ty, v);
    };
    OperationState state(loc, op->getName());
    state.addOperands({shrink(op->getOperand(0)), shrink(op->getOperand(1))});
    state.addTypes(ty);
    Value r = rewriter.create(state)->getResult(0);
    Type ity = op->getResult(0).getType();
    rewriter.replaceOp(
        op, isSigned
                ? arith::IndexCastOp::create(rewriter, loc, ity, r).getResult()
                : arith::IndexCastUIOp::create(rewriter, loc, ity, r)
                      .getResult());
    return success();
  }
};

/// Whether \p op sits at an integer width no divider row declares while a
/// wider row exists: the widths a mixed-signedness promotion mints (i33) and
/// any other off-row width a kernel spells.
static bool widensToDivRow(Operation *op, const OperatorLibrary &lib) {
  auto ity = dyn_cast<IntegerType>(op->getResult(0).getType());
  if (!ity)
    return false;
  unsigned row = lib.smallestAdvancedRowWidth(op->getName().stripDialect(),
                                              ity.getWidth());
  return row != 0 && row != ity.getWidth();
}

// A division at an off-row width no other pattern takes prices as a
// full-width combinational divider and derates the module, like the `index`
// case above. Widened to the narrowest declared row it binds that pipelined
// core; the extension keeps quotient and remainder exact and the truncation
// back is free.
struct WidenDivToRow : RewritePattern {
  WidenDivToRow(MLIRContext *ctx, bool expand, unsigned maxMulWidth,
                unsigned maxIpMulWidth, const OperatorLibrary &lib)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        expand(expand), maxMulWidth(maxMulWidth),
        maxIpMulWidth(maxIpMulWidth), lib(lib) {}
  bool expand;
  unsigned maxMulWidth;
  unsigned maxIpMulWidth;
  const OperatorLibrary &lib;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp, arith::RemSIOp>(
            op) ||
        !widensToDivRow(op, lib))
      return failure();
    bool isSigned = isa<arith::DivSIOp, arith::RemSIOp>(op);
    if (powerOfTwoDivisor(op->getOperand(1)) ||
        (expand && magicPlan(op, isSigned, maxMulWidth, maxIpMulWidth)))
      return failure();
    Location loc = op->getLoc();
    unsigned row = lib.smallestAdvancedRowWidth(
        op->getName().stripDialect(),
        cast<IntegerType>(op->getResult(0).getType()).getWidth());
    Type wide = rewriter.getIntegerType(row);
    auto grow = [&](Value v) -> Value {
      if (isSigned)
        return arith::ExtSIOp::create(rewriter, loc, wide, v);
      return arith::ExtUIOp::create(rewriter, loc, wide, v);
    };
    OperationState state(loc, op->getName());
    state.addOperands({grow(op->getOperand(0)), grow(op->getOperand(1))});
    state.addTypes(wide);
    Value r = rewriter.create(state)->getResult(0);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(
        op, op->getResult(0).getType(), r);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant tables and constant multiplies -> wiring and shift-adds.
//===----------------------------------------------------------------------===//

/// The element a constant-table read at literal indices names, or nullopt
/// where the read stays a real access (a variable index, an uninitialized or
/// written array, an out-of-range literal).
static std::optional<TypedAttr> constantTableElement(affine::AffineLoadOp op) {
  std::optional<Attribute> init = globalInitOf(op.getMemRef());
  if (!init || !isConstantTable(op.getMemRef()))
    return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(*init);
  if (!dense)
    return std::nullopt;
  SmallVector<Attribute> operands;
  for (Value idx : op.getMapOperands()) {
    APInt cst;
    if (!matchPattern(idx, m_ConstantInt(&cst)))
      return std::nullopt;
    operands.push_back(IntegerAttr::get(IndexType::get(op.getContext()),
                                        cst.getSExtValue()));
  }
  SmallVector<Attribute> indices;
  if (failed(op.getAffineMap().constantFold(operands, indices)))
    return std::nullopt;
  int64_t flat = 0;
  for (auto [attr, dim] :
       llvm::zip(indices, op.getMemRefType().getShape())) {
    int64_t i = cast<IntegerAttr>(attr).getInt();
    if (i < 0 || i >= dim)
      return std::nullopt;
    flat = flat * dim + i;
  }
  Type et = dense.getElementType();
  if (isa<IntegerType>(et))
    return IntegerAttr::get(et, dense.getValues<APInt>()[flat]);
  if (isa<FloatType>(et))
    return FloatAttr::get(et, dense.getValues<APFloat>()[flat]);
  return std::nullopt;
}

struct FoldConstantTableRead : OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<TypedAttr> elem = constantTableElement(op);
    if (!elem)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, *elem);
    return success();
  }
};

struct NafConstMul : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<IntegerType, IndexType>(op.getType()))
      return failure();
    APInt cst;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&cst)))
      return failure();
    std::optional<SmallVector<NafDigit, 5>> digits = nafDigits(cst);
    if (!digits)
      return failure();
    rewriter.replaceOp(
        op, nafBuild(rewriter, op.getLoc(), op.getLhs(), *digits,
                     op.getType()));
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

    // The widest multiply the clock takes whole, register floor included, and
    // the widest one a pipelined multiplier row carries registered: a typed
    // division expands where either form of the reciprocal's product fits,
    // and keeps its divider IP only past both. With no period stated every
    // typed division stays on its IP.
    unsigned maxMulWidth = 0;
    if (expandConstArith && periodNs > 0.0)
      for (unsigned w = 2; w <= 129; ++w)
        if (std::optional<double> delay = lib.measuredCombDelay(OpKind::Mul, w))
          if (*delay <= periodNs)
            maxMulWidth = w;
    unsigned maxIpMulWidth = expandConstArith ? lib.maxPipelinedMulWidth() : 0;

    // Table reads at literal indices fold to a fixpoint first, so a multiply
    // sees its literal factor before it is judged: the conversion below visits
    // each op once, which is too early for an operand another rewrite
    // constant-folds.
    RewritePatternSet folds(&getContext());
    folds.add<FoldConstantTableRead>(&getContext());
    if (expandConstArith)
      folds.add<NafConstMul>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(folds))))
      return signalPassFailure();

    // Reuse the upstream expansion patterns
    RewritePatternSet patterns(&getContext());
    arith::populateArithExpandOpsPatterns(patterns);
    patterns.add<LowerBitGetSlice, LowerBitSetSlice, ReduceDivUI, ReduceRemUI,
                 ReduceDivSI, ReduceRemSI>(&getContext());
    patterns.add<IndexDivToTyped>(&getContext(), expandConstArith, maxMulWidth,
                                  maxIpMulWidth);
    patterns.add<WidenDivToRow>(&getContext(), expandConstArith, maxMulWidth,
                                maxIpMulWidth, lib);
    if (expandConstArith)
      patterns.add<MagicDivUI, MagicRemUI, MagicDivSI, MagicRemSI>(
          &getContext(), maxMulWidth, maxIpMulWidth);

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
    // when the flag asks; anything still at `index` moves to the typed width,
    // and an off-row width to the narrowest divider row above it. Every other
    // divisor stays and is bound to a device core.
    bool expand = expandConstArith;
    target.addDynamicallyLegalOp<arith::DivUIOp, arith::RemUIOp>(
        [expand, maxMulWidth, maxIpMulWidth, &lib](Operation *op) {
          if (powerOfTwoDivisor(op->getOperand(1)) ||
              op->getResult(0).getType().isIndex() ||
              widensToDivRow(op, lib))
            return false;
          return !(expand && magicPlan(op, /*isSigned=*/false, maxMulWidth,
                                       maxIpMulWidth));
        });
    target.addDynamicallyLegalOp<arith::DivSIOp, arith::RemSIOp>(
        [expand, maxMulWidth, maxIpMulWidth, &lib](Operation *op) {
          if (powerOfTwoDivisor(op->getOperand(1)) ||
              op->getResult(0).getType().isIndex() ||
              widensToDivRow(op, lib))
            return false;
          return !(expand && magicPlan(op, /*isSigned=*/true, maxMulWidth,
                                       maxIpMulWidth));
        });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
