#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "allo/Conversion/Passes.h"

using namespace mlir;

static void assembleLLVMConversionPipeline(OpPassManager &pm) {
  pm.addPass(allo::createConvertInstructionToCanonicalFormPass({true}));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createConvertTensorToLinalgPass());
  bufferization::OneShotBufferizePassOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.bufferAlignment = 64;
  options.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  options.allowReturnAllocsFromLoops = false;
  pm.addPass(bufferization::createOneShotBufferizePass(options));
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  bufferization::buildBufferDeallocationPipeline(pm);
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(LLVM::createLLVMRequestCWrappersPass());

  pm.addPass(createConvertLinalgToAffineLoopsPass());
  pm.addPass(affine::createAffineScalarReplacementPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void allo::registerLLVMLoweringPipeline() {
  PassPipelineRegistration<>("lower-to-llvm", "Lower to LLVM IR",
                             assembleLLVMConversionPipeline);
}