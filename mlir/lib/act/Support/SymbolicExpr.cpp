#include "act/Support/SymbolicExpr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "symbolic-expr"

using namespace mlir;
using namespace mlir::act;

//===----------------------------------------------------------------------===//
// buildSymExpr
//===----------------------------------------------------------------------===//

FailureOr<SymExpr> mlir::act::buildSymExpr(Value v) {
  // Block argument → Param
  if (auto arg = dyn_cast<BlockArgument>(v))
    return SymExpr::param(arg.getArgNumber());

  auto *defOp = v.getDefiningOp();
  if (!defOp)
    return failure();

  // arith.constant → Constant
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return SymExpr::constant(intAttr.getInt());
    return failure();
  }

  // arith.muli → Mul
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhs = buildSymExpr(mulOp.getLhs());
    auto rhs = buildSymExpr(mulOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return SymExpr::mul(*lhs, *rhs);
  }

  // arith.addi → Add
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = buildSymExpr(addOp.getLhs());
    auto rhs = buildSymExpr(addOp.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    return SymExpr::add(*lhs, *rhs);
  }

  LLVM_DEBUG(llvm::dbgs() << "  [buildSymExpr] unsupported op: "
                          << defOp->getName() << "\n");
  return failure();
}

FailureOr<SymExpr> mlir::act::buildSymExpr(OpFoldResult ofr) {
  if (auto attr = dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return SymExpr::constant(intAttr.getInt());
    return failure();
  }
  return buildSymExpr(cast<Value>(ofr));
}

//===----------------------------------------------------------------------===//
// generateShapeExpr — per-op implementations
//===----------------------------------------------------------------------===//

/// StridedOp: shape = [counts...] ++ bufferType.getShape() (non-HBM only)
/// With rank reduction: drop leading count dim if it's Constant(1) and there
/// are trailing element dims.
static FailureOr<SymShape> generateShapeExprStrided(StridedOp op,
                                                    BufferTypeInterface bufTy) {
  auto ctx = op.getContext();
  auto mixedCounts = getMixedValues(op.getStaticCounts(), op.getCounts(), ctx);

  SymShape shape;
  for (auto &count : mixedCounts) {
    auto expr = buildSymExpr(count);
    if (failed(expr))
      return op.emitError() << "failed to build symbolic expr for count";
    shape.push_back(*expr);
  }

  // Append buffer element dimensions for non-HBM buffers
  if (!isa<HBMBufferType>(bufTy)) {
    for (int64_t dim : bufTy.getShape())
      shape.push_back(SymExpr::constant(dim));
  }

  // Rank reduction: StridedOp::materialize calls
  // inferCanonicalRankReducedResultType(shaped.getRank()-1, ...) which drops
  // the leading count dim when it's statically 1 (one slot = the element
  // itself)
  if (shape.size() > 1 && shape[0].isConstant() && shape[0].value == 1)
    shape.erase(shape.begin());

  return shape;
}

/// ExpandShapeOp: replace source dims according to reassociation with
/// output_shape dims.
static FailureOr<SymShape> generateShapeExprExpand(ExpandShapeOp op,
                                                   BufferTypeInterface bufTy) {
  auto sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufTy);
  if (failed(sourceShape))
    return failure();

  auto reassoc = op.getReassociationIndices();
  auto mixedOutputShape = getMixedValues(op.getStaticOutputShape(),
                                         op.getOutputShape(), op.getContext());

  // Build result shape: for each reassociation group, replace source dim(s)
  // with the corresponding output dims
  SymShape result;
  unsigned outputIdx = 0;
  for (auto &group : reassoc) {
    // Each group maps one or more source dims to multiple output dims
    unsigned numOutputDims = group.size();
    // The output dims for this group come from mixedOutputShape
    // For expand_shape: the number of output dims per group = group.size()
    // But wait — reassociation in expand_shape means source has fewer dims.
    // reassociation[i] lists which OUTPUT dims correspond to source dim i.
    // So reassoc.size() == sourceShape.size(), and each group lists output dim
    // indices.
    for (unsigned idx : group) {
      if (idx >= mixedOutputShape.size())
        return op.emitError() << "output shape index out of bounds";
      auto expr = buildSymExpr(mixedOutputShape[idx]);
      if (failed(expr))
        return op.emitError() << "failed to build symbolic expr for output dim";
      result.push_back(*expr);
    }
  }

  return result;
}

/// CollapseShapeOp: merge source dims according to reassociation with Mul.
static FailureOr<SymShape>
generateShapeExprCollapse(CollapseShapeOp op, BufferTypeInterface bufTy) {
  auto sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufTy);
  if (failed(sourceShape))
    return failure();

  auto reassoc = op.getReassociationIndices();
  SymShape result;
  for (auto &group : reassoc) {
    // Merge all source dims in this group with multiplication
    SymExpr merged = (*sourceShape)[group[0]];
    for (unsigned i = 1; i < group.size(); ++i)
      merged = SymExpr::mul(std::move(merged), (*sourceShape)[group[i]]);
    result.push_back(std::move(merged));
  }

  return result;
}

/// TransposeOp: permute source dims.
static FailureOr<SymShape>
generateShapeExprTranspose(TransposeOp op, BufferTypeInterface bufTy) {
  auto sourceOp = op.getSource().getDefiningOp();
  if (!sourceOp)
    return op.emitError() << "source has no defining op";

  auto sourceShape = generateShapeExpr(sourceOp, bufTy);
  if (failed(sourceShape))
    return failure();

  auto perm = op.getPermutation();
  SymShape result(perm.size());
  for (unsigned i = 0; i < perm.size(); ++i)
    result[i] = (*sourceShape)[perm[i]];

  return result;
}

//===----------------------------------------------------------------------===//
// generateShapeExpr — top-level dispatch
//===----------------------------------------------------------------------===//

FailureOr<SymShape> mlir::act::generateShapeExpr(Operation *accessOp,
                                                 BufferTypeInterface bufTy) {
  if (auto strided = dyn_cast<StridedOp>(accessOp))
    return generateShapeExprStrided(strided, bufTy);
  if (auto expand = dyn_cast<ExpandShapeOp>(accessOp))
    return generateShapeExprExpand(expand, bufTy);
  if (auto collapse = dyn_cast<CollapseShapeOp>(accessOp))
    return generateShapeExprCollapse(collapse, bufTy);
  if (auto transpose = dyn_cast<TransposeOp>(accessOp))
    return generateShapeExprTranspose(transpose, bufTy);

  return accessOp->emitError() << "unsupported access op for generateShapeExpr";
}
