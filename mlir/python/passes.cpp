#include "ir.h"

#include "allo/Translation/VivadoHLSEmitter.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"

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
}
