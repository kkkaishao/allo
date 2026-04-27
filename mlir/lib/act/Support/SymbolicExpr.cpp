#include "act/Support/SymbolicExpr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <utility>

#define DEBUG_TYPE "symbolic-expr"

using namespace mlir;
using namespace mlir::act;

int64_t SymExpr::evaluate(ArrayRef<int64_t> paramValues) const {
  switch (kind) {
  case Kind::Constant:
    return value;
  case Kind::Param:
    assert(paramIdx < paramValues.size() && "param index out of bounds");
    return paramValues[paramIdx];
  case Kind::Add:
    assert(lhs && rhs && "expected binary expression operands");
    return lhs->evaluate(paramValues) + rhs->evaluate(paramValues);
  case Kind::Mul:
    assert(lhs && rhs && "expected binary expression operands");
    return lhs->evaluate(paramValues) * rhs->evaluate(paramValues);
  }
  llvm_unreachable("unknown symbolic expression kind");
}

void SymExpr::collectParams(DenseSet<unsigned> &params) const {
  switch (kind) {
  case Kind::Constant:
    return;
  case Kind::Param:
    params.insert(paramIdx);
    return;
  case Kind::Add:
  case Kind::Mul:
    assert(lhs && rhs && "expected binary expression operands");
    lhs->collectParams(params);
    rhs->collectParams(params);
    return;
  }
  llvm_unreachable("unknown symbolic expression kind");
}

std::optional<int64_t> SymExpr::getConstantValue() const {
  if (isConstant())
    return value;
  return std::nullopt;
}

std::optional<unsigned> SymExpr::getParamIdx() const {
  if (isParam())
    return paramIdx;
  return std::nullopt;
}

std::string SymExpr::toString() const {
  switch (kind) {
  case Kind::Constant:
    return "Const(" + std::to_string(value) + ")";
  case Kind::Param:
    return "Param<" + std::to_string(paramIdx) + ">";
  case Kind::Add:
    assert(lhs && rhs && "expected binary expression operands");
    return "(" + lhs->toString() + " + " + rhs->toString() + ")";
  case Kind::Mul:
    assert(lhs && rhs && "expected binary expression operands");
    return "(" + lhs->toString() + " * " + rhs->toString() + ")";
  }
  llvm_unreachable("unknown symbolic expression kind");
}

FailureOr<SymExpr> mlir::act::buildSymExpr(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return SymExpr::param(arg.getArgNumber());

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return failure();

  if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
    auto intAttr = dyn_cast<IntegerAttr>(constant.getValue());
    if (!intAttr)
      return failure();
    return SymExpr::constant(intAttr.getInt());
  }

  if (auto add = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = buildSymExpr(add.getLhs());
    auto rhs = buildSymExpr(add.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return *lhs + *rhs;
  }

  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhs = buildSymExpr(mul.getLhs());
    auto rhs = buildSymExpr(mul.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return *lhs * *rhs;
  }

  LLVM_DEBUG(llvm::dbgs() << "unsupported symbolic expr op: "
                          << defOp->getName() << "\n");
  return failure();
}

FailureOr<SymExpr> mlir::act::buildSymExpr(OpFoldResult value) {
  if (auto attr = dyn_cast<Attribute>(value)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return failure();
    return SymExpr::constant(intAttr.getInt());
  }
  return buildSymExpr(cast<Value>(value));
}

static FailureOr<SymShape> generateStridedShape(StridedOp op,
                                                BufferTypeInterface bufferTy) {
  MLIRContext *ctx = op.getContext();
  auto counts = getMixedValues(op.getStaticCounts(), op.getCounts(), ctx);

  SymShape shape;
  for (OpFoldResult count : counts) {
    auto expr = buildSymExpr(count);
    if (failed(expr))
      return op.emitError() << "failed to build symbolic count expression";
    shape.push_back(*expr);
  }

  if (!isa<HBMBufferType>(bufferTy)) {
    for (int64_t dim : bufferTy.getShape())
      shape.push_back(SymExpr::constant(dim));
  }

  if (shape.size() > 1 && shape.front().getConstantValue() == 1)
    shape.erase(shape.begin());

  return shape;
}

static FailureOr<SymShape> generateExpandShape(ExpandShapeOp op,
                                               BufferTypeInterface bufferTy) {
  Operation *sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source access pattern has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufferTy);
  if (failed(sourceShape))
    return failure();

  auto outputShape = getMixedValues(op.getStaticOutputShape(),
                                    op.getOutputShape(), op.getContext());

  SymShape shape;
  for (OpFoldResult dim : outputShape) {
    auto expr = buildSymExpr(dim);
    if (failed(expr))
      return op.emitError() << "failed to build symbolic output shape";
    shape.push_back(*expr);
  }
  return shape;
}

static FailureOr<SymShape> generateCollapseShape(CollapseShapeOp op,
                                                 BufferTypeInterface bufferTy) {
  Operation *sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source access pattern has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufferTy);
  if (failed(sourceShape))
    return failure();

  auto reassociation = op.getReassociationIndices();
  SymShape shape;
  for (ArrayRef<int64_t> group : reassociation) {
    assert(!group.empty() && "expected non-empty reassociation group");
    assert(group.front() < sourceShape->size() &&
           "reassociation index out of source shape bounds");
    SymExpr dim = (*sourceShape)[group.front()];
    for (int64_t idx : group.drop_front()) {
      assert(idx < static_cast<int64_t>(sourceShape->size()) &&
             "reassociation index out of source shape bounds");
      dim = dim * (*sourceShape)[idx];
    }
    shape.push_back(std::move(dim));
  }
  return shape;
}

static FailureOr<SymShape>
generateTransposeShape(TransposeOp op, BufferTypeInterface bufferTy) {
  Operation *sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source access pattern has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufferTy);
  if (failed(sourceShape))
    return failure();

  auto permutation = op.getPermutation();
  SymShape shape;
  shape.reserve(permutation.size());
  for (int64_t dim : permutation) {
    assert(dim >= 0 && dim < static_cast<int64_t>(sourceShape->size()) &&
           "transpose permutation index out of source shape bounds");
    shape.push_back((*sourceShape)[dim]);
  }
  return shape;
}

FailureOr<SymShape>
mlir::act::generateShapeExpr(Operation *accessOp,
                             BufferTypeInterface bufferType) {
  assert(accessOp && "expected access pattern op");

  if (auto strided = dyn_cast<StridedOp>(accessOp))
    return generateStridedShape(strided, bufferType);
  if (auto expand = dyn_cast<ExpandShapeOp>(accessOp))
    return generateExpandShape(expand, bufferType);
  if (auto collapse = dyn_cast<CollapseShapeOp>(accessOp))
    return generateCollapseShape(collapse, bufferType);
  if (auto transpose = dyn_cast<TransposeOp>(accessOp))
    return generateTransposeShape(transpose, bufferType);
  if (isa<TiledOp>(accessOp))
    return accessOp->emitError()
           << "symbolic shape extraction for act.tiled is not supported yet";

  return accessOp->emitError()
         << "unsupported access pattern op for symbolic shape extraction";
}

std::string mlir::act::symShapeToString(const SymShape &shape) {
  std::string result = "[";
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    if (idx != 0)
      result += ", ";
    result += dim.toString();
  }
  result += "]";
  return result;
}
