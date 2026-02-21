#ifndef ALLO_TRANSFORM_PASSES_H
#define ALLO_TRANSFORM_PASSES_H

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "allo/IR/AlloOps.h"

namespace mlir::allo {
#define GEN_PASS_DECL
#include "allo/Transform/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "allo/Transform/Passes.h.inc"
} // namespace mlir::allo

#endif // ALLO_TRANSFORM_PASSES_H