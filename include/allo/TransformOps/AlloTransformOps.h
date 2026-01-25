#ifndef ALLO_TRANSFORM_OPS_H
#define ALLO_TRANSFORM_OPS_H

#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "allo/TransformOps/AlloTransformOps.h.inc"

namespace mlir::allo {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace mlir::allo

#endif // ALLO_TRANSFORM_OPS_H
