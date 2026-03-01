#ifndef ALLO_INIT_ALL_DIALECTS_H
#define ALLO_INIT_ALL_DIALECTS_H

#include "allo/TransformOps/AlloTransformOps.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/Extensions/ShardingExtensions.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/InitAllDialects.h"

namespace mlir::allo {
inline void registerAllDialects(DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<allo::AlloDialect>();
  linalg::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  affine::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
  allo::registerTransformDialectExtension(registry);

  tensor::registerShardingInterfaceExternalModels(registry);
}
} // namespace mlir::allo

#endif // ALLO_INIT_ALL_DIALECTS_H
