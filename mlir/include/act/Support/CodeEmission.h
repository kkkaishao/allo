#ifndef ACT_SUPPORT_CODE_EMISSION_H
#define ACT_SUPPORT_CODE_EMISSION_H

#include "act/Support/SemanticMatching.h"
#include "act/Support/TilingAnalysis.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::act {

/// Run Stage 3 code emission: coverage resolution, memory layout, and
/// code generation (scf.for loops + act.emit calls).
LogicalResult runCodeEmission(ModuleOp module,
                              ArrayRef<TiledMatchCandidate> tiledMatches,
                              ArrayRef<EdgeLayoutAnnotation> layoutAnnotations);

} // namespace mlir::act

#endif // ACT_SUPPORT_CODE_EMISSION_H
