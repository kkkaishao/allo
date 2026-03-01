#include "allo/IR/AlloOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<allo::AlloDialect>();
  registerAllExtensions(registry);
  registerAllPasses();
  return asMainReturnCode(
      MlirOptMain(argc, argv, "Allo optimization driver\n", registry));
}