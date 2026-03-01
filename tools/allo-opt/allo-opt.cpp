#include "InitAllDialects.h"
#include "allo/IR/AlloOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  allo::registerAllDialects(registry);
  registerAllPasses();
  return asMainReturnCode(
      MlirOptMain(argc, argv, "Allo optimization driver\n", registry));
}