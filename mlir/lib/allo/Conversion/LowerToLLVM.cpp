#include "allo/Conversion/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

void allo::populateLowerToLLVMPipeline(OpPassManager &pm, bool enableTensor) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLowerDataflowPass());

  if (enableTensor) {
    pm.addPass(createConvertTensorToLinalgPass());
    bufferization::OneShotBufferizePassOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.bufferAlignment = 64;
    options.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    options.allowReturnAllocsFromLoops = false;
    pm.addPass(bufferization::createOneShotBufferizePass(options));
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    bufferization::buildBufferDeallocationPipeline(pm);
    pm.addPass(createConvertBufferizationToMemRefPass());
  }

  pm.addPass(createConvertAlloToFuncPass());
  auto &nestedPM = pm.nest<func::FuncOp>();
  nestedPM.addPass(LLVM::createLLVMRequestCWrappersPass());
  nestedPM.addPass(createConvertLinalgToAffineLoopsPass());
  nestedPM.addPass(affine::createAffineScalarReplacementPass());
  nestedPM.addPass(createLoopInvariantCodeMotionPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertOpenMPToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

namespace {
struct AlloLowerToLLVMPipelineOptions
    : public PassPipelineOptions<AlloLowerToLLVMPipelineOptions> {
  Option<bool> enableTensor{
      *this, "enable-tensor",
      llvm::cl::desc(
          "Run tensor->linalg + one-shot bufferization before lowering"),
      llvm::cl::init(true)};
};
} // namespace

void allo::registerAlloLLVMLoweringPipeline() {
  PassPipelineRegistration<AlloLowerToLLVMPipelineOptions>(
      "allo-lower-to-llvm", "Lower allo/canonical-form IR to the LLVM dialect",
      [](OpPassManager &pm, const AlloLowerToLLVMPipelineOptions &opts) {
        populateLowerToLLVMPipeline(pm, opts.enableTensor);
      });
}
