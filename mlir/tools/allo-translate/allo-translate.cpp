#include "allo/Translation/VivadoHLSEmitter.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  allo::registerVivadoHLSTranslation();
  return failed(mlirTranslateMain(argc, argv, "Allo IR translator"));
}