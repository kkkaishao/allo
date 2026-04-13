#ifndef ACT_SUPPORT_SEMANTIC_MATCHING_H
#define ACT_SUPPORT_SEMANTIC_MATCHING_H

#include "act/IR/ActOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::act {

struct StaticSliceSpec {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
};

/// Semantic fingerprint of an instruction's core computation.
struct SemanticFingerprint {
  enum Kind { Named, Generic, Identity };
  Kind kind;

  /// For Named kind: the op name (e.g., "linalg.matmul").
  StringRef opName;

  /// For Generic kind: structural description of the linalg.generic body.
  SmallVector<AffineMap> indexingMaps;
  SmallVector<utils::IteratorType> iteratorTypes;
  Region *bodyRegion = nullptr; // non-owning, for detailed comparison
  unsigned numInputs = 0;
  unsigned numOutputs = 0;

  llvm::hash_code hash() const;
  bool matches(const SemanticFingerprint &other) const;
};

/// A successful match between a source op and an instruction.
struct MatchCandidate {
  Operation *sourceOp;
  DefineOp instruction;
  unsigned numOuterDims = 0; // >0 for structural suffix matches (rank mismatch)
};

enum class EdgeLayoutDirection {
  Input,
  Output,
};

enum class EdgeLayoutTransformKind {
  Transpose,
  ExtractSlice,
  InsertSlice,
};

/// Boundary transform annotation on a logical edge adjacent to a compute op.
struct EdgeLayoutAnnotation {
  EdgeLayoutDirection direction;
  EdgeLayoutTransformKind transformKind;
  Operation *layoutOp;     // linalg.transpose, tensor.extract_slice, etc.
  Operation *computeOp;    // the adjacent compute op
  unsigned edgeIdx;        // input operand idx or output result idx
  unsigned transformOrder; // 0 = closest to logical value / writeback target
  SmallVector<int64_t> permutation;
  StaticSliceSpec sliceSpec;
};

/// Catalog of instruction fingerprints for fast lookup.
class InstructionCatalog {
public:
  /// Build catalog from all act.define ops in the module.
  static InstructionCatalog build(ModuleOp module);

  /// Find matching instructions for a source op.
  SmallVector<MatchCandidate> match(Operation *sourceOp) const;

  /// Debug: dump the catalog contents.
  void dump();

private:
  struct Entry {
    DefineOp defineOp;
    SemanticFingerprint fingerprint;
  };
  DenseMap<llvm::hash_code, SmallVector<Entry>> index;
};

/// Top-level: match all source compute ops against the instruction catalog.
/// Also collects layout ops as edge annotations.
LogicalResult
runSemanticMatching(ModuleOp module, SmallVectorImpl<MatchCandidate> &results,
                    SmallVectorImpl<EdgeLayoutAnnotation> &layoutAnnotations);

/// Structural matching: for source ops not matched by semantic fingerprinting,
/// try suffix matching on iteration types + indexing map compatibility + body
/// equivalence against all DefineOps in the module.
LogicalResult runStructuralMatching(ModuleOp module,
                                    ArrayRef<Operation *> unmatchedOps,
                                    SmallVectorImpl<MatchCandidate> &results);

} // namespace mlir::act

#endif // ACT_SUPPORT_SEMANTIC_MATCHING_H
