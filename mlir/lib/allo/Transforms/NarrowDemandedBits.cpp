/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h" // kIndexWidth
#include "allo/Support/BitAnalysis.h"    // knownBits
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

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

//===----------------------------------------------------------------------===//
// Forward value hulls
//===----------------------------------------------------------------------===//

/// A signed hull [lo, hi] of the values an SSA value can carry.
using Hull = std::pair<int64_t, int64_t>;

/// The hull, when it fits int64; unknown on overflow.
std::optional<Hull> mkHull(__int128 lo, __int128 hi) {
  assert(lo <= hi && "a hull is ordered");
  if (lo < std::numeric_limits<int64_t>::min() ||
      hi > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return Hull{(int64_t)lo, (int64_t)hi};
}

/// Significant bits of a hull, the signed convention the datapath sizes by.
unsigned bitsOfHull(Hull h) {
  auto bits = [](int64_t v) {
    return (unsigned)APInt(64, (uint64_t)v, /*isSigned=*/true)
        .getSignificantBits();
  };
  return std::max(bits(h.first), bits(h.second));
}

/// The walk is a recursion over a DAG, so a modest cap bounds the blowup.
constexpr unsigned kHullDepth = 8;

std::optional<Hull> hullOf(Value v, unsigned depth);

/// Interval-evaluate an affine expr; dims and symbols read \p operands.
std::optional<Hull> hullOfExpr(AffineExpr e, unsigned numDims,
                               ValueRange operands, unsigned depth) {
  auto operand = [&](unsigned pos) -> std::optional<Hull> {
    return pos < operands.size() ? hullOf(operands[pos], depth) : std::nullopt;
  };
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return Hull{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return operand(d.getPosition());
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return operand(numDims + s.getPosition());
  auto bin = cast<AffineBinaryOpExpr>(e);
  auto lhs = hullOfExpr(bin.getLHS(), numDims, operands, depth);
  auto rhs = hullOfExpr(bin.getRHS(), numDims, operands, depth);
  if (!lhs || !rhs)
    return std::nullopt;
  auto [a, b] = *lhs;
  auto [c, d] = *rhs;
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return mkHull((__int128)a + c, (__int128)b + d);
  case AffineExprKind::Mul: {
    __int128 p[] = {(__int128)a * c, (__int128)a * d, (__int128)b * c,
                    (__int128)b * d};
    return mkHull(*std::min_element(p, p + 4), *std::max_element(p, p + 4));
  }
  case AffineExprKind::FloorDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideFloorSigned(a, c), llvm::divideFloorSigned(b, c)};
  case AffineExprKind::CeilDiv:
    if (c != d || c <= 0)
      return std::nullopt;
    return Hull{llvm::divideCeilSigned(a, c), llvm::divideCeilSigned(b, c)};
  case AffineExprKind::Mod:
    if (c != d || c <= 0)
      return std::nullopt;
    return a >= 0 && b < c ? lhs : std::optional<Hull>(Hull{0, c - 1});
  default:
    return std::nullopt;
  }
}

/// The hull of the value \p v carries: a forward interval walk over constants,
/// constant loop bounds and the monotone arith transfers. Unknown is always
/// sound. A hull the value's own carrier could wrap is refused, so a returned
/// hull is the value's range, never its residue mod 2^width.
std::optional<Hull> hullOf(Value v, unsigned depth) {
  if (!depth--)
    return std::nullopt;
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.getSignificantBits() <= 64
               ? std::optional<Hull>(
                     Hull{cst.getSExtValue(), cst.getSExtValue()})
               : std::nullopt;
  if (isa<BlockArgument>(v)) {
    affine::AffineForOp fo = affine::getForInductionVarOwner(v);
    if (!fo || !fo.hasConstantLowerBound() || !fo.hasConstantUpperBound() ||
        fo.getConstantLowerBound() >= fo.getConstantUpperBound())
      return std::nullopt;
    return Hull{fo.getConstantLowerBound(), fo.getConstantUpperBound() - 1};
  }
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto in = [&](unsigned k) { return hullOf(op->getOperand(k), depth); };
  auto rhsConst = [&]() -> std::optional<int64_t> {
    APInt c;
    if (matchPattern(op->getOperand(1), m_ConstantInt(&c)) &&
        c.getSignificantBits() <= 64)
      return c.getSExtValue();
    return std::nullopt;
  };
  auto binary = [&](auto f) -> std::optional<Hull> {
    auto x = in(0), y = in(1);
    if (!x || !y)
      return std::nullopt;
    return f(x->first, x->second, y->first, y->second);
  };
  std::optional<Hull> h =
      llvm::TypeSwitch<Operation *, std::optional<Hull>>(op)
          .Case<affine::AffineApplyOp>([&](affine::AffineApplyOp ap) {
            AffineMap m = ap.getAffineMap();
            return hullOfExpr(m.getResult(0), m.getNumDims(), ap.getOperands(),
                              depth);
          })
          .Case<arith::AddIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              return mkHull((__int128)a + c, (__int128)b + d);
            });
          })
          .Case<arith::SubIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              return mkHull((__int128)a - d, (__int128)b - c);
            });
          })
          .Case<arith::MulIOp>([&](auto) {
            return binary([](int64_t a, int64_t b, int64_t c, int64_t d) {
              __int128 p[] = {(__int128)a * c, (__int128)a * d,
                              (__int128)b * c, (__int128)b * d};
              return mkHull(*std::min_element(p, p + 4),
                            *std::max_element(p, p + 4));
            });
          })
          .Case<arith::AndIOp>([&](auto) -> std::optional<Hull> {
            // AND with a non-negative mask lands in [0, mask] whatever the
            // other side holds.
            auto c = rhsConst();
            if (!c || *c < 0)
              return std::nullopt;
            return Hull{0, *c};
          })
          .Case<arith::OrIOp, arith::XOrIOp>([&](auto) -> std::optional<Hull> {
            auto x = in(0), y = in(1);
            if (!x || !y || x->first < 0 || y->first < 0)
              return std::nullopt;
            unsigned k = std::max(APInt(64, x->second).getActiveBits(),
                                  APInt(64, y->second).getActiveBits());
            return k > 62 ? std::nullopt
                          : std::optional<Hull>(
                                Hull{0, (int64_t(1) << k) - 1});
          })
          .Case<arith::RemUIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            if (!c || *c <= 0)
              return std::nullopt;
            return Hull{0, *c - 1};
          })
          .Case<arith::RemSIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            if (!c || *c <= 0)
              return std::nullopt;
            auto x = in(0);
            if (x && x->first >= 0)
              return Hull{0, std::min(x->second, *c - 1)};
            return Hull{-(*c - 1), *c - 1};
          })
          .Case<arith::DivSIOp, arith::DivUIOp>(
              [&](auto) -> std::optional<Hull> {
                auto c = rhsConst();
                auto x = in(0);
                if (!c || *c <= 0 || !x)
                  return std::nullopt;
                if (isa<arith::DivUIOp>(op) && x->first < 0)
                  return std::nullopt;
                return Hull{x->first / *c, x->second / *c};
              })
          .Case<arith::ShLIOp>([&](auto) -> std::optional<Hull> {
            auto c = rhsConst();
            auto x = in(0);
            if (!c || *c < 0 || *c > 62 || !x)
              return std::nullopt;
            __int128 p = __int128(1) << *c;
            return mkHull(x->first * p, x->second * p);
          })
          .Case<arith::ShRUIOp, arith::ShRSIOp>(
              [&](auto) -> std::optional<Hull> {
                auto c = rhsConst();
                auto x = in(0);
                if (!c || *c < 0 || *c > 62 || !x)
                  return std::nullopt;
                if (isa<arith::ShRUIOp>(op) && x->first < 0)
                  return std::nullopt;
                int64_t p = int64_t(1) << *c;
                return Hull{llvm::divideFloorSigned(x->first, p),
                            llvm::divideFloorSigned(x->second, p)};
              })
          .Case<arith::SelectOp>([&](auto) -> std::optional<Hull> {
            auto x = in(1), y = in(2);
            if (!x || !y)
              return std::nullopt;
            return Hull{std::min(x->first, y->first),
                        std::max(x->second, y->second)};
          })
          .Case<arith::MinSIOp, arith::MaxSIOp>(
              [&](auto) -> std::optional<Hull> {
                return binary([&](int64_t a, int64_t b, int64_t c, int64_t d) {
                  return isa<arith::MinSIOp>(op)
                             ? std::optional<Hull>(
                                   Hull{std::min(a, c), std::min(b, d)})
                             : std::optional<Hull>(
                                   Hull{std::max(a, c), std::max(b, d)});
                });
              })
          .Case<arith::MinUIOp, arith::MaxUIOp>(
              [&](auto) -> std::optional<Hull> {
                return binary([&](int64_t a, int64_t b, int64_t c,
                                  int64_t d) -> std::optional<Hull> {
                  if (a < 0 || c < 0)
                    return std::nullopt;
                  return isa<arith::MinUIOp>(op)
                             ? Hull{std::min(a, c), std::min(b, d)}
                             : Hull{std::max(a, c), std::max(b, d)};
                });
              })
          .Case<arith::ExtSIOp, arith::IndexCastOp, arith::TruncIOp>(
              [&](auto) { return in(0); })
          .Case<arith::ExtUIOp, arith::IndexCastUIOp>(
              [&](auto) -> std::optional<Hull> {
                // Reinterprets the bits unsigned: exact only proven >= 0.
                auto x = in(0);
                if (!x || x->first < 0)
                  return std::nullopt;
                return x;
              })
          .Default([](auto) { return std::nullopt; });
  // The wrap refusal: a truncating cast or a ring op computing mod 2^width
  // holds the transfer's hull only when that hull fits the carrier.
  if (h)
    if (auto ity = dyn_cast<IntegerType>(v.getType()))
      if (bitsOfHull(*h) > ity.getWidth())
        return std::nullopt;
  return h;
}

/// The width the datapath builds this carrier at; the width narrowing must
/// beat to be a narrowing at all.
unsigned carrierWidth(Type t) {
  return isa<IndexType>(t) ? kIndexWidth : cast<IntegerType>(t).getWidth();
}

//===----------------------------------------------------------------------===//
// Rewrites
//===----------------------------------------------------------------------===//

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

// Rebuilds a ring or bitwise op whose value hull needs fewer bits than its
// carrier at the hull's width, with resize casts at the seams. The casts are
// wiring; the operator is built and priced at the width the value spans. An
// `index` operand enters the integer domain here, which is what lets a
// truncation reach it at all.
struct NarrowFromHull : RewritePattern {
  NarrowFromHull(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isRingOp(op) &&
        !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op))
      return failure();
    Type ty = op->getResult(0).getType();
    unsigned width = carrierWidth(ty);
    std::optional<Hull> h = hullOf(op->getResult(0), kHullDepth);
    if (!h)
      return failure();
    unsigned w = bitsOfHull(*h);
    if (w >= width)
      return failure();
    bool index = isa<IndexType>(ty);
    Type nty = rewriter.getIntegerType(w);
    Location loc = op->getLoc();
    auto shrink = [&](Value x) -> Value {
      if (index)
        return arith::IndexCastOp::create(rewriter, loc, nty, x);
      return arith::TruncIOp::create(rewriter, loc, nty, x);
    };
    // Rebuilt without the original's attributes, like the sink above: keeping
    // per-site ids apart would stop CSE from merging equal rebuilt cones.
    OperationState state(loc, op->getName());
    state.addOperands({shrink(op->getOperand(0)), shrink(op->getOperand(1))});
    state.addTypes(nty);
    Value narrow = rewriter.create(state)->getResult(0);
    rewriter.replaceOp(
        op, index
                ? arith::IndexCastOp::create(rewriter, loc, ty, narrow)
                      .getResult()
                : arith::ExtSIOp::create(rewriter, loc, ty, narrow)
                      .getResult());
    return success();
  }
};

// `x & (2^k - 1)` is a zero-extended truncation spelled as a mask. The cast
// form makes the low-bit demand explicit so the truncation can sink into the
// producer; the casts themselves are wiring.
struct MaskToTrunc : OpRewritePattern<arith::AndIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AndIOp op,
                                PatternRewriter &rewriter) const override {
    APInt mask;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&mask)) || !mask.isMask())
      return failure();
    unsigned k = mask.popcount();
    if (k == 0 || k >= carrierWidth(op.getType()))
      return failure();
    Type nty = rewriter.getIntegerType(k);
    Location loc = op.getLoc();
    if (isa<IndexType>(op.getType())) {
      Value t = arith::IndexCastOp::create(rewriter, loc, nty, op.getLhs());
      rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(op, op.getType(), t);
    } else {
      Value t = arith::TruncIOp::create(rewriter, loc, nty, op.getLhs());
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(), t);
    }
    return success();
  }
};

// A resize that hops through `index` is a resize: fold the pair to one direct
// cast so a truncation keeps moving toward its producer.
//   cast(cast(x: iA -> index) -> iB)  =>  trunci/ext(x -> iB)
//   trunci(cast(x: index -> iA) -> iB)  =>  cast(x -> iB)
struct FoldCastThroughIndex : RewritePattern {
  FoldCastThroughIndex(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<arith::TruncIOp>(op)) {
      Operation *inner = op->getOperand(0).getDefiningOp();
      if (!inner || !isa<arith::IndexCastOp, arith::IndexCastUIOp>(inner) ||
          !isa<IndexType>(inner->getOperand(0).getType()))
        return failure();
      // Both steps keep the low bits, so one truncating cast does.
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
          op, op->getResult(0).getType(), inner->getOperand(0));
      return success();
    }
    if (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(op) ||
        !isa<IntegerType>(op->getResult(0).getType()))
      return failure();
    Operation *inner = op->getOperand(0).getDefiningOp();
    if (!inner || !isa<arith::IndexCastOp, arith::IndexCastUIOp>(inner))
      return failure();
    auto ity = dyn_cast<IntegerType>(inner->getOperand(0).getType());
    if (!ity)
      return failure();
    Value x = inner->getOperand(0);
    unsigned a = ity.getWidth();
    unsigned b = cast<IntegerType>(op->getResult(0).getType()).getWidth();
    Type bty = op->getResult(0).getType();
    if (b == a) {
      rewriter.replaceOp(op, x);
    } else if (b < a) {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, bty, x);
    } else if (isa<arith::IndexCastOp>(inner)) {
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, bty, x);
    } else {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, bty, x);
    }
    return success();
  }
};

struct NarrowDemandedBitsPass
    : public allo::impl::NarrowDemandedBitsPassBase<NarrowDemandedBitsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SinkTruncThroughRingOp, DropRedundantMask, NarrowFromHull,
                 MaskToTrunc, FoldCastThroughIndex>(ctx);
    // The cast folds are what make the rewrite chain: without them a sunk
    // truncation stops on top of an extend instead of collapsing into it.
    arith::TruncIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtSIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::ExtUIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(patterns, ctx);
    arith::IndexCastUIOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
