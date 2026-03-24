#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"

#include "allo/Translation/VivadoHLSEmitter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

void bindPasses(nb::module_ &m) {
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

  m.def("cse_and_canonicalize", [](Operation *op) {
    PassManager pm(op->getContext());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(op)))
      throw std::runtime_error("CSE pass failed");
  });

  m.def("lower_to_llvm", [](ModuleOp mod) {
    PassManager pm(mod.getContext());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    // pm.addPass(createConvertTensorToLinalgPass());
    // pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
    // bufferization::OneShotBufferizePassOptions options;
    // options.bufferizeFunctionBoundaries = true;
    // options.bufferAlignment = 64;
    // options.functionBoundaryTypeConversion =
    //     mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
    // pm.addPass(bufferization::createOneShotBufferizePass(options));
    // pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(pm.run(mod)))
      throw std::runtime_error("Lowering to LLVM failed");
  });
}
