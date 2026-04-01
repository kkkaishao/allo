#include "allo/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::allo;

void allo::registerAlloConversions() {
  registerConvertInstructionToCanonicalFormPass();
  registerLLVMLoweringPipeline();
}