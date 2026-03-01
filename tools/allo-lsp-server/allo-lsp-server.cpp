#include "InitAllDialects.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  allo::registerAllDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}