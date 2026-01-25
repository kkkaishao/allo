#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<allo::AlloDialect>();
  registerAllPasses();
  allo::registerAlloPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "Allo optimization driver\n", registry));
}