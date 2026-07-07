/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AffineValueMapBuilder.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::allo;

Value mlir::allo::stripCast(Value value) {
  while (true) {
    auto *defOp = value.getDefiningOp();
    if (!defOp)
      return value;
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::TruncIOp>(defOp)) {
      value = defOp->getOperand(0);
    } else {
      return value;
    }
  }
}

AffineExpr AffineValueMapBuilder::addDim(Value v) {
  auto *it = llvm::find(dims, v);
  // if found, return its existing position
  if (it != dims.end()) {
    return getAffineDimExpr(std::distance(dims.begin(), it), ctx);
  }
  // otherwise, register it
  unsigned pos = dims.size();
  dims.push_back(v);
  return getAffineDimExpr(pos, ctx);
}

AffineExpr AffineValueMapBuilder::addSym(Value v) {
  // same as adding a dim
  auto *it = llvm::find(syms, v);
  if (it != syms.end()) {
    return getAffineSymbolExpr(std::distance(syms.begin(), it), ctx);
  }
  unsigned pos = syms.size();
  syms.push_back(v);
  return getAffineSymbolExpr(pos, ctx);
}

FailureOr<AffineExpr> AffineValueMapBuilder::importValueInternal(Value v) {
  v = stripCast(v);
  // Case 0: fast rejection
  if (exprFailureCache.contains(v))
    return failure();
  // Case 1: constant integer
  IntegerAttr::ValueType cst;
  if (matchPattern(v, m_ConstantInt(&cst))) {
    return getAffineConstantExpr(cst.getSExtValue(), ctx);
  }
  // Allo worker-id / worker-count queries are invariant within a kernel and can
  // be treated as affine symbols (e.g. `fifo[i, j]` where `i = get_wid(0)`).
  if (isa_and_present<GetWorkerIdOp, GetNumWorkersOp>(v.getDefiningOp())) {
    return addSym(v);
  }
  // Case 2: affine dim or symbol, use upstream utils
  if (affine::isValidDim(v)) {
    return addDim(v);
  }
  if (affine::isValidSymbol(v)) {
    return addSym(v);
  }
  auto *defOp = v.getDefiningOp();
  // Case 3: affine.apply, recursively import its map results
  if (auto applyOp = dyn_cast_if_present<affine::AffineApplyOp>(defOp)) {
    if (failed(importMapAndOperands(
            applyOp.getAffineMap(), applyOp.getDimOperands(),
            applyOp.getSymbolOperands(), /*allowMultiResults=*/false)))
      return cacheFailure(v);
    return exprs.back();
  }
  // Case 4: arithmetic operations
  if (auto addOp = dyn_cast_if_present<arith::AddIOp>(defOp)) {
    auto lhs = importValueInternal(addOp.getLhs());
    auto rhs = importValueInternal(addOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    return *lhs + *rhs;
  }
  if (auto subOp = dyn_cast_if_present<arith::SubIOp>(defOp)) {
    auto lhs = importValueInternal(subOp.getLhs());
    auto rhs = importValueInternal(subOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    return *lhs - *rhs;
  }
  if (auto mulOp = dyn_cast_if_present<arith::MulIOp>(defOp)) {
    auto lhs = importValueInternal(mulOp.getLhs());
    auto rhs = importValueInternal(mulOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // multiplication of affine exprs is not affine unless one of them is a
    // symbol d0 * d1 is not affine, but s0 * d0 or s0 * s1 is affine.
    auto result = *lhs * *rhs;
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  if (isa_and_present<arith::DivSIOp, arith::DivUIOp>(defOp)) {
    auto lhs = importValueInternal(defOp->getOperand(0));
    auto rhs = importValueInternal(defOp->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // division of affine exprs is not affine unless the divisor is a symbol or
    // constant
    auto result = lhs->floorDiv(*rhs);
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  if (isa_and_present<arith::RemSIOp, arith::RemUIOp>(defOp)) {
    auto lhs = importValueInternal(defOp->getOperand(0));
    auto rhs = importValueInternal(defOp->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return cacheFailure(v);
    // remainder of affine exprs is not affine unless the divisor is a symbol or
    // constant
    auto result = *lhs % *rhs;
    if (!result.isPureAffine())
      return cacheFailure(v);
    return result;
  }
  // AffineExpr can only be +, -, *, //, %, while max/min is not an AffineExpr.
  return cacheFailure(v);
}

affine::AffineValueMap AffineValueMapBuilder::compose() const {
  SmallVector<Value, 4> operands(dims);
  llvm::append_range(operands, syms);
  auto map = AffineMap::get(dims.size(), syms.size(), exprs, ctx);
  affine::AffineValueMap vMap(map, operands, results);
  vMap.composeSimplifyAndCanonicalize();
  return vMap;
}

void AffineValueMapBuilder::reset() {
  dims.clear();
  syms.clear();
  results.clear();
  // exprFailureCache.clear();
  exprs.clear();
}

LogicalResult AffineValueMapBuilder::importMapAndOperands(
    AffineMap map, ValueRange dims, ValueRange syms, bool allowMultiResults) {
  if (map.getNumResults() > 1 && !allowMultiResults)
    return failure();
  SmallVector<AffineExpr, 4> dimExprs;
  SmallVector<AffineExpr, 4> symExprs;

  for (auto dim : dims) {
    auto dimExpr = importValueInternal(dim);
    if (failed(dimExpr))
      return failure();
    dimExprs.push_back(*dimExpr);
  }
  for (auto sym : syms) {
    auto symExpr = importValueInternal(sym);
    if (failed(symExpr))
      return failure();
    symExprs.push_back(*symExpr);
  }
  for (auto result : map.getResults()) {
    // affine.min/max may carry multi-result maps; each result is one
    // candidate bound expression in the final min/max set.
    exprs.push_back(result.replaceDimsAndSymbols(dimExprs, symExprs));
  }
  return success();
}
