#ifndef ACT_SUPPORT_TILING_ANALYSIS_H
#define ACT_SUPPORT_TILING_ANALYSIS_H

#include "act/Support/SemanticMatching.h"
#include "act/Support/SymbolicExpr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

namespace mlir::act {

/// Classification of addr parameters.
enum class AddrParamKind {
  Shape,  // appears in counts/output_shape — controls computation size
  Offset, // appears in basis — controls data position
  Mixed,  // appears in both
};

/// Tiling scheme for one match: solved params + per-dim tiling info.
struct TilingScheme {
  /// Solved shape params: addr block arg index → concrete value.
  DenseMap<unsigned, int64_t> solvedParams;

  /// Per iteration dimension tiling info.
  struct DimTiling {
    int64_t sourceBound; // source iteration domain bound
    SymExpr nativeBound; // symbolic instruction bound
    int64_t nativeValue; // evaluated native bound (after solving params)
    int64_t tileFactor;  // ceil(sourceBound / nativeValue)
    bool needsPadding;   // sourceBound % nativeValue != 0
    utils::IteratorType iterType; // parallel or reduction
  };
  SmallVector<DimTiling> dims;
};

/// A match candidate annotated with tiling information.
struct TiledMatchCandidate {
  MatchCandidate base;                          // from Stage 1
  TilingScheme tiling;                          // from Phase 2a
  DenseMap<unsigned, AddrParamKind> paramKinds; // from Phase 2b
  bool isValid = false;      // false if constraints are inconsistent
  unsigned numOuterDims = 0; // leading batch dims (from rank mismatch)

  /// Does this match require tiling loops?
  bool needsTiling() const {
    return llvm::any_of(tiling.dims,
                        [](const auto &d) { return d.tileFactor > 1; });
  }
};

/// Top-level: run tiling analysis on all match candidates from Stage 1.
/// For each match, extracts symbolic shapes from the instruction's addr region,
/// maps them to iteration domain bounds, compares against the source op's
/// concrete iteration domain, and solves for shape params + tiling factors.
LogicalResult runTilingAnalysis(ModuleOp module,
                                ArrayRef<MatchCandidate> matches,
                                SmallVectorImpl<TiledMatchCandidate> &results);

} // namespace mlir::act

#endif // ACT_SUPPORT_TILING_ANALYSIS_H
