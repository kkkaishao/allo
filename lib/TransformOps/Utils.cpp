#include "allo/TransformOps/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace {

bool mapOperandDependsOnValue(mlir::Value operand, mlir::Value needle) {
  if (operand == needle)
    return true;

  auto arithDependsOnNeedle = [&](mlir::Operation *defOp) {
    if (!defOp || defOp->getNumRegions() != 0 || defOp->getNumResults() != 1)
      return false;
    if (auto *dialect = defOp->getDialect()) {
      if (dialect->getNamespace() != "arith")
        return false;
    } else {
      return false;
    }
    return llvm::any_of(defOp->getOperands(), [&](mlir::Value in) {
      return mapOperandDependsOnValue(in, needle);
    });
  };

  auto applyOp = operand.getDefiningOp<mlir::affine::AffineApplyOp>();
  if (applyOp) {
    mlir::AffineMap map = applyOp.getAffineMap();
    for (mlir::AffineExpr resultExpr : map.getResults()) {
      if (mlir::allo::affineExprUsesValue(resultExpr, applyOp.getMapOperands(),
                                          map.getNumDims(), needle)) {
        return true;
      }
    }
    return false;
  }

  return arithDependsOnNeedle(operand.getDefiningOp());
}

} // namespace

namespace mlir::allo {

allo::CallOp convertCallToAlloCall(RewriterBase &b, func::CallOp call) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(call);
  auto newCall =
      allo::CallOp::create(b, call->getLoc(), call.getCallee(),
                           call.getResultTypes(), call.getOperands());
  b.replaceAllOpUsesWith(call, newCall);
  b.eraseOp(call);
  return newCall;
}

allo::KernelOp convertFuncToKernel(RewriterBase &b, func::FuncOp func) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(func);
  auto kernel = allo::KernelOp::create(
      b, func.getLoc(), func.getName(), func.getFunctionType(),
      func.getSymVisibilityAttr(), func.getArgAttrsAttr(),
      func.getResAttrsAttr(), /*virtual_mapping=*/b.getI64ArrayAttr(1));
  Region &kernelRegion = kernel.getRegion();
  kernelRegion.takeBody(func.getBody());
  b.eraseOp(func);
  // replace func.return to allo.return
  auto returnOps = kernelRegion.getOps<func::ReturnOp>();
  for (auto returnOp : llvm::make_early_inc_range(returnOps)) {
    b.setInsertionPoint(returnOp);
    allo::ReturnOp::create(b, returnOp.getLoc(), returnOp.getOperands());
    b.eraseOp(returnOp);
  }
  return kernel;
}

bool affineExprUsesValue(AffineExpr expr, ValueRange mapOperands,
                         unsigned numDims, Value needle) {
  bool used = false;
  expr.walk([&](AffineExpr inner) {
    if (used)
      return;
    if (auto dim = dyn_cast<AffineDimExpr>(inner)) {
      unsigned pos = dim.getPosition();
      if (pos < mapOperands.size() &&
          mapOperandDependsOnValue(mapOperands[pos], needle)) {
        used = true;
      }
      return;
    }
    auto sym = dyn_cast<AffineSymbolExpr>(inner);
    if (!sym)
      return;
    unsigned pos = numDims + sym.getPosition();
    if (pos < mapOperands.size() &&
        mapOperandDependsOnValue(mapOperands[pos], needle)) {
      used = true;
    }
  });
  return used;
}

int findMemRefAxisFromIV(affine::AffineStoreOp storeOp, Value iv) {
  AffineMap map = storeOp.getAffineMap();
  auto operands = storeOp.getMapOperands();
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (affineExprUsesValue(map.getResult(i), operands, map.getNumDims(), iv))
      return static_cast<int>(i);
  }
  return -1;
}

// Follow view-like aliases and resolve to a root buffer value.
Value resolveMemRefValueRoot(Value value) {
  SmallPtrSet<Value, 8> visited;
  while (value && visited.insert(value).second) {
    if (isa<BlockArgument>(value))
      return value;

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return value;

    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      value = subview.getSource();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(defOp)) {
      value = view.getSource();
      continue;
    }
    if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      value = reinterpretCast.getSource();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(defOp)) {
      value = castOp.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(defOp)) {
      value = transpose.getIn();
      continue;
    }
    return value;
  }
  return value;
}
} // namespace mlir::allo
