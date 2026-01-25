#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "mlir/InitAllDialects.h"
#include "allo/Dialect/AlloDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<allo::AlloDialect>();
  registerAllDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}