#include "allo/TransformOps/AlloTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::allo;

namespace {

Value stripCast(Value value) {
  while (true) {
    auto defOp = value.getDefiningOp();
    if (!defOp)
      break;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::TruncIOp>(defOp))
      value = defOp->getOperand(0);
    else
      break;
  }
  return value;
}

/// Try to interpret select(cmp, a, b) as min/max(a,b).
/// If successful, returns {isMax, x, y} where result == (isMax? max(x,y) :
/// min(x,y)). Supports lhs/rhs swapping and true/false swapping for
/// sge/sgt/sle/slt (also u* variants).
/// Case 1:
/// %cmp = cmpi sge/uge/sgt/ugt %a, %b
/// $res = select %cmp, %a, %b => max(%a, %b)
/// Case 2:
/// %cmp = cmpi sge/uge/sgt/ugt %a, %b
/// $res = select %cmp, %b, %a => min(%a, %b)
/// Case 3:
/// %cmp = cmpi sle/ule/slt/ult %a, %b
/// $res = select %cmp, %a, %b => min(%a, %b)
/// Case 4:
/// %cmp = cmpi sle/ule/slt/ult %a, %b
/// $res = select %cmp, %b, %a => max(%a, %b)
FailureOr<std::tuple<bool, Value, Value>>
matchSelectAsMinMax(arith::SelectOp sel) {
  auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return failure();

  auto lhs = cmp.getLhs();
  auto rhs = cmp.getRhs();
  auto t = sel.getTrueValue();
  auto f = sel.getFalseValue();
  auto pred = cmp.getPredicate();

  auto isGE = [](arith::CmpIPredicate p) {
    using P = arith::CmpIPredicate;
    return p == P::sge || p == P::sgt || p == P::uge || p == P::ugt;
  };

  auto isLE = [](arith::CmpIPredicate p) {
    using P = arith::CmpIPredicate;
    return p == P::sle || p == P::slt || p == P::ule || p == P::ult;
  };

  auto checkSwapped = [&](Value a, Value b) -> std::optional<bool> {
    if (t == a && f == b)
      return false; // not swapped
    if (t == b && f == a)
      return true; // swapped
    return std::nullopt;
  };

  auto swappedOr = checkSwapped(lhs, rhs);
  if (!swappedOr) {
    // Not a min/max pattern if true/false doesn't match (lhs,rhs) or (rhs,lhs).
    return failure();
  }
  bool swapped = *swappedOr;

  bool predIsGE = isGE(pred);
  bool predIsLE = isLE(pred);
  if (!predIsGE && !predIsLE) {
    // eq/ne predicates don't match min/max patterns.
    return failure();
  }

  if (!swapped) {
    if (predIsGE)
      return std::make_tuple(true, lhs, rhs); // max
    else
      return std::make_tuple(false, lhs, rhs); // min
  } else {
    if (predIsGE)
      return std::make_tuple(false, lhs, rhs); // min
    else
      return std::make_tuple(true, lhs, rhs); // max
  }
}

struct AffineBound {
  AffineMap map;
  SmallVector<Value, 4> operands;
};

struct AffineParallelBoundSet {
  SmallVector<AffineMap, 4> maps;
  SmallVector<Value, 8> operands;
};

struct AffineExprBuilder {
  MLIRContext *ctx;
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> syms;
  llvm::SmallDenseMap<Value, unsigned> dimPos;
  llvm::SmallDenseMap<Value, unsigned> symPos;
  llvm::SmallDenseMap<Value, AffineExpr> exprCache;
  llvm::SmallDenseSet<Value, 4> exprFailureCache;
  SmallVector<AffineExpr, 4> results;

  explicit AffineExprBuilder(MLIRContext *ctx) : ctx(ctx) {}

  AffineExpr addDim(Value v) {
    auto it = dimPos.find(v);
    if (it != dimPos.end()) {
      return getAffineDimExpr(it->second, ctx);
    }
    unsigned pos = dims.size();
    dims.push_back(v);
    dimPos[v] = pos;
    return getAffineDimExpr(pos, ctx);
  }

  AffineExpr addSym(Value v) {
    auto it = symPos.find(v);
    if (it != symPos.end()) {
      return getAffineSymbolExpr(it->second, ctx);
    }
    unsigned pos = syms.size();
    syms.push_back(v);
    symPos[v] = pos;
    return getAffineSymbolExpr(pos, ctx);
  }

  FailureOr<AffineExpr> importValueAsExpr(Value v) {
    using namespace matchers;
    v = stripCast(v);
    if (auto it = exprCache.find(v); it != exprCache.end())
      return it->second;
    if (exprFailureCache.contains(v))
      return failure();

    IntegerAttr::ValueType cst;
    if (matchPattern(v, m_ConstantInt(&cst))) {
      auto expr = getAffineConstantExpr(cst.getSExtValue(), ctx);
      exprCache[v] = expr;
      return expr;
    }

    auto cacheFailure = [&]() -> FailureOr<AffineExpr> {
      exprFailureCache.insert(v);
      return failure();
    };

    // Memoize affine expression import so shared index DAGs are expanded once.
    if (affine::isValidDim(v))
      return exprCache.try_emplace(v, addDim(v)).first->second;
    if (affine::isValidSymbol(v))
      return exprCache.try_emplace(v, addSym(v)).first->second;

    if (auto applyOp = v.getDefiningOp<affine::AffineApplyOp>()) {
      auto map = applyOp.getAffineMap();
      // only support single result maps
      if (map.getNumResults() != 1)
        return failure();
      auto operands = applyOp.getOperands();
      SmallVector<AffineExpr, 4> dimExprs;
      SmallVector<AffineExpr, 4> symExprs;

      for (unsigned i = 0; i < map.getNumDims(); ++i) {
        auto dimExpr = importValueAsExpr(operands[i]);
        if (failed(dimExpr))
          return cacheFailure();
        dimExprs.push_back(*dimExpr);
      }
      for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
        auto symExpr = importValueAsExpr(operands[map.getNumDims() + i]);
        if (failed(symExpr))
          return cacheFailure();
        symExprs.push_back(*symExpr);
      }
      AffineExpr result =
          map.getResult(0).replaceDimsAndSymbols(dimExprs, symExprs);
      exprCache[v] = result;
      return result;
    }
    if (auto addi = v.getDefiningOp<arith::AddIOp>()) {
      auto lhs = importValueAsExpr(addi.getLhs());
      auto rhs = importValueAsExpr(addi.getRhs());
      if (failed(lhs) || failed(rhs))
        return cacheFailure();
      auto expr = *lhs + *rhs;
      exprCache[v] = expr;
      return expr;
    }
    if (auto subi = v.getDefiningOp<arith::SubIOp>()) {
      auto lhs = importValueAsExpr(subi.getLhs());
      auto rhs = importValueAsExpr(subi.getRhs());
      if (failed(lhs) || failed(rhs))
        return cacheFailure();
      auto expr = *lhs - *rhs;
      exprCache[v] = expr;
      return expr;
    }
    if (auto muli = v.getDefiningOp<arith::MulIOp>()) {
      auto lhs = importValueAsExpr(muli.getLhs());
      auto rhs = importValueAsExpr(muli.getRhs());
      if (failed(lhs) || failed(rhs))
        return cacheFailure();
      if (!lhs->isSymbolicOrConstant() && !rhs->isSymbolicOrConstant())
        return cacheFailure();
      auto expr = *lhs * *rhs;
      exprCache[v] = expr;
      return expr;
    }
    // TODO: support more arith operations that can be represented in affine
    // expressions.
    return cacheFailure();
  }

  AffineBound composeResults() {
    SmallVector<Value, 4> operands;
    operands.append(dims);
    operands.append(syms);

    auto map = AffineMap::get(dims.size(), syms.size(), results, ctx);
    affine::fullyComposeAffineMapAndOperands(&map, &operands,
                                             /*composeAffineMin=*/true);
    affine::canonicalizeMapAndOperands(&map, &operands);
    MutableAffineMap mMap(map);
    mMap.simplify();
    return {mMap.getAffineMap(), operands};
  }
};

LogicalResult importAffineMapResults(AffineExprBuilder &builder, AffineMap map,
                                     ValueRange operands) {
  SmallVector<AffineExpr, 4> dimExprs;
  SmallVector<AffineExpr, 4> symExprs;

  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    auto dimExpr = builder.importValueAsExpr(operands[i]);
    if (failed(dimExpr))
      return failure();
    dimExprs.push_back(*dimExpr);
  }
  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
    auto symExpr = builder.importValueAsExpr(operands[map.getNumDims() + i]);
    if (failed(symExpr))
      return failure();
    symExprs.push_back(*symExpr);
  }
  for (AffineExpr result : map.getResults()) {
    // affine.min/max may carry multi-result maps; each result is one candidate
    // bound expression in the final min/max set.
    builder.results.push_back(result.replaceDimsAndSymbols(dimExprs, symExprs));
  }
  return success();
}

LogicalResult collectBoundExprs(Value value, bool isLowerBound,
                                AffineExprBuilder &builder);

LogicalResult collectBinaryBoundExprs(Value lhs, Value rhs, bool isLowerBound,
                                      AffineExprBuilder &builder) {
  if (failed(collectBoundExprs(lhs, isLowerBound, builder)))
    return failure();
  if (failed(collectBoundExprs(rhs, isLowerBound, builder)))
    return failure();
  return success();
}

LogicalResult collectBoundExprs(Value value, bool isLowerBound,
                                AffineExprBuilder &builder) {
  Value v = stripCast(value);
  Operation *defOp = v.getDefiningOp();

  // Expand arithmetic max for lower bounds and min for upper bounds.
  if (isLowerBound) {
    if (auto maxOp = dyn_cast_or_null<arith::MaxSIOp>(defOp)) {
      return collectBinaryBoundExprs(maxOp.getLhs(), maxOp.getRhs(),
                                     isLowerBound, builder);
    }
    if (auto maxOp = dyn_cast_or_null<arith::MaxUIOp>(defOp)) {
      return collectBinaryBoundExprs(maxOp.getLhs(), maxOp.getRhs(),
                                     isLowerBound, builder);
    }
  } else {
    if (auto minOp = dyn_cast_or_null<arith::MinSIOp>(defOp)) {
      return collectBinaryBoundExprs(minOp.getLhs(), minOp.getRhs(),
                                     isLowerBound, builder);
    }
    if (auto minOp = dyn_cast_or_null<arith::MinUIOp>(defOp)) {
      return collectBinaryBoundExprs(minOp.getLhs(), minOp.getRhs(),
                                     isLowerBound, builder);
    }
  }

  // Expand select-as-min/max if it matches the bound direction.
  if (auto sel = dyn_cast_or_null<arith::SelectOp>(defOp)) {
    auto match = matchSelectAsMinMax(sel);
    if (succeeded(match)) {
      auto [isMax, x, y] = *match;
      if ((isLowerBound && isMax) || (!isLowerBound && !isMax)) {
        return collectBinaryBoundExprs(x, y, isLowerBound, builder);
      }
    }
  }

  // Expand affine.max/affine.min by importing all map results.
  if (auto maxOp = dyn_cast_or_null<affine::AffineMaxOp>(defOp)) {
    if (!isLowerBound)
      return failure();
    return importAffineMapResults(builder, maxOp.getAffineMap(),
                                  maxOp.getOperands());
  }
  if (auto minOp = dyn_cast_or_null<affine::AffineMinOp>(defOp)) {
    if (isLowerBound)
      return failure();
    return importAffineMapResults(builder, minOp.getAffineMap(),
                                  minOp.getOperands());
  }

  // Leaf case: import as a single affine expression.
  // This covers plain dim/symbol/constant expressions and affine.apply chains.
  auto expr = builder.importValueAsExpr(v);
  if (failed(expr))
    return failure();
  builder.results.push_back(*expr);
  return success();
}

FailureOr<AffineBound> matchAffineBound(Value root, bool isLowerBound) {
  AffineExprBuilder builder(root.getContext());
  if (failed(collectBoundExprs(root, isLowerBound, builder)))
    return failure();
  if (builder.results.empty())
    return failure();
  return builder.composeResults();
}

FailureOr<AffineBound> computeAffineMapAndArgs(MLIRContext *ctx,
                                               ValueRange indices) {
  if (indices.empty())
    return AffineBound{AffineMap::get(0, 0, {}, ctx), {}};

  AffineExprBuilder builder(ctx);
  for (Value idx : indices) {
    // Build one affine map result per memref index expression.
    auto expr = builder.importValueAsExpr(idx);
    if (failed(expr))
      return failure();
    builder.results.push_back(*expr);
  }
  return builder.composeResults();
}

template <typename OpTy>
void updateIndicesToAffineApply(RewriterBase &b, OpTy op) {
  OpBuilder::InsertionGuard g(b);
  MLIRContext *ctx = b.getContext();
  auto indices = op.getIndices();
  auto mapOr = computeAffineMapAndArgs(ctx, indices);
  if (failed(mapOr)) {
    return;
  }
  b.setInsertionPoint(op);
  if constexpr (std::is_same_v<OpTy, memref::LoadOp>) {
    b.replaceOpWithNewOp<affine::AffineLoadOp>(op, op.getMemRef(), mapOr->map,
                                               mapOr->operands);
  } else if constexpr (std::is_same_v<OpTy, memref::StoreOp>) {
    b.replaceOpWithNewOp<affine::AffineStoreOp>(
        op, op.getValueToStore(), op.getMemRef(), mapOr->map, mapOr->operands);
  } else {
    // replace indices with the result of affine.apply
    SmallVector<Value, 4> newIndices;
    for (unsigned i = 0; i < indices.size(); ++i) {
      auto subMap = mapOr->map.getSubMap(i);
      auto apply = affine::AffineApplyOp::create(
          b, op.getLoc(), indices[i].getType(), subMap, mapOr->operands);
      newIndices.push_back(apply);
    }
    b.modifyOpInPlace(op, [&]() { op.getIndicesMutable().assign(newIndices); });
  }
}

void raiseAccesses(transform::TransformRewriter &rewriter, Operation *root) {
  root->walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      updateIndicesToAffineApply(rewriter, load);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      updateIndicesToAffineApply(rewriter, store);
    } else if (auto chanOp = dyn_cast<ChannelAccessOpInterface>(op)) {
      updateIndicesToAffineApply(rewriter, chanOp);
    }
  });
}

FailureOr<AffineParallelBoundSet>
normalizeParallelBounds(ArrayRef<AffineBound> bounds) {
  if (bounds.empty())
    return failure();

  MLIRContext *ctx = bounds.front().map.getContext();
  SmallVector<Value, 8> dims;
  SmallVector<Value, 8> syms;
  llvm::SmallDenseMap<Value, unsigned> dimPos;
  llvm::SmallDenseMap<Value, unsigned> symPos;

  auto registerOperand = [&](Value v) -> LogicalResult {
    v = stripCast(v);
    auto addDim = [&]() {
      if (!dimPos.count(v)) {
        dimPos[v] = dims.size();
        dims.push_back(v);
      }
      return success();
    };
    auto addSym = [&]() {
      if (!symPos.count(v)) {
        symPos[v] = syms.size();
        syms.push_back(v);
      }
      return success();
    };
    // Prefer dim if both are legal.
    if (affine::isValidDim(v))
      return addDim();
    if (affine::isValidSymbol(v))
      return addSym();
    return failure();
  };

  for (const AffineBound &bound : bounds) {
    for (Value operand : bound.operands) {
      if (failed(registerOperand(operand)))
        return failure();
    }
  }

  SmallVector<AffineMap, 4> normalizedMaps;
  normalizedMaps.reserve(bounds.size());

  auto getGlobalExpr = [&](Value v) -> FailureOr<AffineExpr> {
    v = stripCast(v);
    if (auto it = dimPos.find(v); it != dimPos.end())
      return getAffineDimExpr(it->second, ctx);
    if (auto it = symPos.find(v); it != symPos.end())
      return getAffineSymbolExpr(it->second, ctx);
    return failure();
  };

  for (const AffineBound &bound : bounds) {
    SmallVector<AffineExpr, 4> dimExprs;
    SmallVector<AffineExpr, 4> symExprs;
    dimExprs.reserve(bound.map.getNumDims());
    symExprs.reserve(bound.map.getNumSymbols());

    unsigned numDims = bound.map.getNumDims();
    unsigned numSyms = bound.map.getNumSymbols();
    if (bound.operands.size() != numDims + numSyms)
      return failure();

    for (unsigned i = 0; i < numDims; ++i) {
      auto expr = getGlobalExpr(bound.operands[i]);
      if (failed(expr))
        return failure();
      dimExprs.push_back(*expr);
    }
    for (unsigned i = 0; i < numSyms; ++i) {
      auto expr = getGlobalExpr(bound.operands[numDims + i]);
      if (failed(expr))
        return failure();
      symExprs.push_back(*expr);
    }

    SmallVector<AffineExpr, 4> results;
    results.reserve(bound.map.getNumResults());
    for (AffineExpr expr : bound.map.getResults())
      results.push_back(expr.replaceDimsAndSymbols(dimExprs, symExprs));

    // Rebuild each bound map in a shared input space required by
    // affine.parallel.
    normalizedMaps.push_back(
        AffineMap::get(dims.size(), syms.size(), results, ctx));
  }

  SmallVector<Value, 8> operands;
  operands.append(dims);
  operands.append(syms);
  return AffineParallelBoundSet{normalizedMaps, operands};
}

std::optional<int64_t> getConstPositiveStep(Value step) {
  auto cstOp = step.getDefiningOp<arith::ConstantIndexOp>();
  if (!cstOp)
    return std::nullopt;
  int64_t stepVal = cstOp.value();
  if (stepVal <= 0)
    return std::nullopt;
  return stepVal;
}

/// raise `scf.for` to `affine.for` if the bounds and step are affine.
DiagnosedSilenceableFailure
raiseForOp(transform::TransformRewriter &rewriter, scf::ForOp forOp,
           transform::ApplyToEachResultList &results,
           transform::TransformState &state) {

  rewriter.setInsertionPoint(forOp);
  auto lbOr = matchAffineBound(forOp.getLowerBound(), /*isLowerBound=*/true);
  if (failed(lbOr)) {
    return emitSilenceableFailure(forOp)
           << "lower bound does not match affine bound pattern";
  }
  auto ubOr = matchAffineBound(forOp.getUpperBound(), /*isLowerBound=*/false);
  if (failed(ubOr)) {
    return emitSilenceableFailure(forOp)
           << "upper bound does not match affine bound pattern";
  }

  auto lb = *lbOr;
  auto ub = *ubOr;
  Value step = stripCast(forOp.getStep());

  // Fast path: constant positive step
  if (auto stepInt = getConstPositiveStep(step)) {
    auto affineLoop = affine::AffineForOp::create(
        rewriter, forOp->getLoc(), lb.operands, lb.map, ub.operands, ub.map,
        *stepInt, forOp.getInitArgs());
    // merge loop body into the new loop and update induction variable uses.
    auto affineBody = affineLoop.getBody();
    // delete the implicit yield
    if (affineLoop->getNumResults() == 0) {
      affineBody->getTerminator()->erase();
    }
    // manually create affine.yield
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(yield);
    affine::AffineYieldOp::create(rewriter, yield->getLoc(),
                                  yield->getOperands());
    yield->erase();
    // Merge the rest of the body into the new loop. The source block has one
    // IV argument plus all loop-carried region arguments, so provide a full
    // replacement list in that order.
    SmallVector<Value, 4> argRepls;
    argRepls.push_back(affineLoop.getInductionVar());
    for (Value iterArg : affineLoop.getRegionIterArgs())
      argRepls.push_back(iterArg);
    rewriter.mergeBlocks(forOp.getBody(), affineLoop.getBody(), argRepls);
    // After raising loops, try to raise in-body memref accesses to affine ops.
    raiseAccesses(rewriter, affineLoop);
    if (forOp->hasAttr(OpIdentifier))
      affineLoop->setAttr(OpIdentifier, forOp->getAttr(OpIdentifier));
    rewriter.replaceOp(forOp, affineLoop);
    results.push_back(affineLoop);
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(forOp)
         << "step is not a constant positive integer";
}

DiagnosedSilenceableFailure
raiseParallelOp(transform::TransformRewriter &rewriter,
                scf::ParallelOp parallelOp,
                transform::ApplyToEachResultList &results,
                transform::TransformState &state) {
  if (parallelOp.getNumReductions() != 0) {
    // Keep this path conservative for now; only non-reduction parallel loops
    // are raised in this transform.
    return emitSilenceableFailure(parallelOp)
           << "parallel reduction is not supported yet";
  }

  unsigned numLoops = parallelOp.getNumLoops();
  auto lowerBounds = parallelOp.getLowerBound();
  auto upperBounds = parallelOp.getUpperBound();
  auto stepValues = parallelOp.getStep();
  SmallVector<AffineBound, 4> lbs;
  SmallVector<AffineBound, 4> ubs;
  SmallVector<int64_t, 4> steps;
  lbs.reserve(numLoops);
  ubs.reserve(numLoops);
  steps.reserve(numLoops);

  for (unsigned i = 0; i < numLoops; ++i) {
    // Cache ranges above: getLower/Upper/Step can materialize temporary ranges.
    auto lbOr = matchAffineBound(lowerBounds[i], /*isLowerBound=*/true);
    if (failed(lbOr)) {
      return emitSilenceableFailure(parallelOp)
             << "lower bound of loop dim " << i
             << " does not match affine bound pattern";
    }
    lbs.push_back(*lbOr);

    auto ubOr = matchAffineBound(upperBounds[i], /*isLowerBound=*/false);
    if (failed(ubOr)) {
      return emitSilenceableFailure(parallelOp)
             << "upper bound of loop dim " << i
             << " does not match affine bound pattern";
    }
    ubs.push_back(*ubOr);

    Value step = stripCast(stepValues[i]);
    auto stepInt = getConstPositiveStep(step);
    if (!stepInt) {
      return emitSilenceableFailure(parallelOp)
             << "step of loop dim " << i
             << " is not a constant positive integer";
    }
    steps.push_back(*stepInt);
  }

  auto normLBs = normalizeParallelBounds(lbs);
  if (failed(normLBs)) {
    return emitSilenceableFailure(parallelOp)
           << "failed to normalize lower bounds to affine.parallel input space";
  }
  auto normUBs = normalizeParallelBounds(ubs);
  if (failed(normUBs)) {
    return emitSilenceableFailure(parallelOp)
           << "failed to normalize upper bounds to affine.parallel input space";
  }

  rewriter.setInsertionPoint(parallelOp);
  SmallVector<arith::AtomicRMWKind> reductions;
  // Construct affine.parallel with normalized per-dimension min/max bounds.
  auto affineParallel = affine::AffineParallelOp::create(
      rewriter, parallelOp.getLoc(), TypeRange{}, reductions, normLBs->maps,
      normLBs->operands, normUBs->maps, normUBs->operands, steps);

  auto affineBody = affineParallel.getBody();
  if (affineParallel->getNumResults() == 0)
    affineBody->getTerminator()->erase();

  auto reduce = cast<scf::ReduceOp>(parallelOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(reduce);
  affine::AffineYieldOp::create(rewriter, reduce.getLoc(),
                                reduce.getOperands());
  reduce->erase();

  ValueRange affineIvs(affineParallel.getIVs());
  rewriter.mergeBlocks(parallelOp.getBody(), affineParallel.getBody(),
                       affineIvs);
  raiseAccesses(rewriter, affineParallel);
  if (parallelOp->hasAttr(OpIdentifier))
    affineParallel->setAttr(OpIdentifier, parallelOp->getAttr(OpIdentifier));
  rewriter.replaceOp(parallelOp, affineParallel->getResults());
  results.push_back(affineParallel);
  return DiagnosedSilenceableFailure::success();
}
} // namespace

DiagnosedSilenceableFailure transform::RaiseToAffineOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    return raiseForOp(rewriter, forOp, results, state);
  }
  if (auto parallelOp = dyn_cast<scf::ParallelOp>(target)) {
    return raiseParallelOp(rewriter, parallelOp, results, state);
  }
  if (isa<affine::AffineForOp, affine::AffineParallelOp>(target)) {
    // Already affine loops; just try to raise in-body memref accesses to affine
    // ops.
    raiseAccesses(rewriter, target);
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected scf.for or scf.parallel, but got " << target->getName();
}
