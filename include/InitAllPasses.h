#ifndef ALLO_INIT_ALL_PASSES_H
#define ALLO_INIT_ALL_PASSES_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Transform/Passes.h"

namespace mlir::allo {
inline void registerAllPasses() {
  // general passes
  registerTransformsPasses();

  registerConversionPasses();

  // dialect passes
  affine::registerAffinePasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  registerConvertControlFlowToLLVMPass();
  func::registerFuncPasses();
  registerLinalgPasses();
  LLVM::registerLLVMPasses();
  LLVM::registerTargetLLVMIRTransformsPasses();
  math::registerMathPasses();
  memref::registerMemRefPasses();
  omp::registerOpenMPPasses();
  registerSCFPasses();
  registerConvertIndexToLLVMPass();
  registerShapePasses();
  tensor::registerTensorPasses();
  tosa::registerTosaOptPasses();
  transform::registerTransformPasses();
  vector::registerVectorPasses();

  // pass pipelines
  bufferization::registerBufferizationPipelines();
  tosa::registerTosaToLinalgPipelines();

  // allo passes
  allo::registerApplyVirtualMapping();
}
} // namespace mlir::allo
#endif // ALLO_INIT_ALL_PASSES_H