#include "allo/IR/AlloOps.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "InitAllDialects.h"
#include "InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  allo::registerAllDialects(registry);
  allo::registerAllPasses();
  allo::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  return asMainReturnCode(MlirOptMain(argc, argv, "Allo optimization driver\n", registry));
}