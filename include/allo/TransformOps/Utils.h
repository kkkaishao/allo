#ifndef ALLO_TRANSFORM_OPS_UTILS_H
#define ALLO_TRANSFORM_OPS_UTILS_H

#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::allo {
allo::CallOp convertCallToAlloCall(RewriterBase &b, func::CallOp call);
allo::KernelOp convertFuncToKernel(RewriterBase &b, func::FuncOp func);
bool affineExprUsesValue(AffineExpr expr, ValueRange mapOperands,
                         unsigned numDims, Value needle);
int findMemRefAxisFromIV(affine::AffineStoreOp storeOp, Value iv);
Value resolveMemRefValueRoot(Value value);
bool isTrivialMapping(ArrayRef<int32_t> mapping);
// strip away index casts, extension/truncation ops,
// which do not affect the value as an affine expression
Value stripCast(Value value);
} // namespace mlir::allo

#endif // ALLO_TRANSFORM_OPS_UTILS_H
