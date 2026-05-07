#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string_view.h"

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
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(op)))
      throw std::runtime_error("Failed to run canonicalizer pass");
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
