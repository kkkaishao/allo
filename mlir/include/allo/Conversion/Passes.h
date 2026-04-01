#ifndef ALLO_CONVERSION_PASSES_H
#define ALLO_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::allo {
#define GEN_PASS_DECL
#include "allo/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "allo/Conversion/Passes.h.inc"

void registerAlloConversions();
void registerLLVMLoweringPipeline();
} // namespace mlir::allo

#endif //ALLO_CONVERSION_PASSES_H
