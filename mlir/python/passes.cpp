#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"

#include "allo/Translation/VivadoHLSEmitter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

void bindPasses(nb::module_ &m) {

  m.def("run", [](std::string_view pipeline, Operation *op) {
    std::string error;
    llvm::raw_string_ostream os(error);
    auto pmOr = parsePassPipeline(pipeline, os);
    if (failed(pmOr))
      throw std::runtime_error("Failed to parse pass pipeline: " + os.str());
    PassManager pm(op->getContext(), pmOr->getOpAnchorName());
    pm.enableVerifier();

    static_cast<OpPassManager &>(pm) = std::move(*pmOr);
    if (failed(pm.run(op)))
      throw std::runtime_error("Failed to run pass pipeline: " + os.str());
  });

  m.def("run_canonicalize", [](Operation *op) {
    PassManager pm(op->getContext());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(op)))
      throw std::runtime_error("Failed to run canonicalizer pass");
  });

  m.def("lower_to_llvm", [](ModuleOp mod, bool enableTensor = false) {
    PassManager pm(mod->getContext());
    pm.enableVerifier();
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

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
    pm.addPass(LLVM::createLLVMRequestCWrappersPass());

    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(affine::createAffineScalarReplacementPass());
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(mod)))
      throw std::runtime_error("Failed to run LLVM lowering pipeline");
  });

  m.def(
      "emit_vivado_hls",
      [](ModuleOp mod, unsigned indexWidth = 32, unsigned indentSize = 2,
         bool withLocation = false) -> std::optional<std::string> {
        std::string code;
        llvm::raw_string_ostream os(code);
        auto result =
            emitVivadoHLS(mod, os, indexWidth, indentSize, withLocation);
        if (failed(result))
          return std::nullopt;
        return os.str();
      },
      nb::arg("mod"), nb::arg("index_width") = 32, nb::arg("indent_size") = 2,
      nb::arg("with_location") = false);
}
