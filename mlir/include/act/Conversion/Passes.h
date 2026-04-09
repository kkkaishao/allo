#ifndef ACT_CONVERSION_PASSES_H
#define ACT_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::act {
#define GEN_PASS_DECL
#include "act/Conversion/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "act/Conversion/Passes.h.inc"

void registerActConversionPasses();

void registerLLVMLoweringPipeline();
} // namespace mlir::act

#endif // ACT_CONVERSION_PASSES_H
