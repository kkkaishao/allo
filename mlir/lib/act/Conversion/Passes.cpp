#include "act/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::act;

void act::registerActConversionPasses() {
  registerConvertActToCanonicalFormPass();
  registerConvertCanonicalFormToActPass();
  registerLLVMLoweringPipeline();
}
