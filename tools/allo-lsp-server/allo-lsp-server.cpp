#include "allo/IR/AlloOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<allo::AlloDialect>();
  registerAllExtensions(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}