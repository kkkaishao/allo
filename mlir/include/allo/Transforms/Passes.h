#ifndef ALLO_TRANSFORMS_PASSES_H
#define ALLO_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::allo {
#define GEN_PASS_DECL
#include "allo/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "allo/Transforms/Passes.h.inc"

void registerAlloTransforms();
} // namespace mlir::allo

#endif // ALLO_TRANSFORMS_PASSES_H
