
#include "act/Conversion/Passes.h"
#include "act/IR/ActOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<act::ActDialect>();
  registerAllExtensions(registry);
  act::registerActConversionPasses();
  registerAllPasses();
  return asMainReturnCode(MlirOptMain(
      argc, argv, "MLIR optimizer driver for the ACT dialect", registry));
}