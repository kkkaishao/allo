#ifndef ACT_SUPPORT_SEMANTIC_MATCHING_H
#define ACT_SUPPORT_SEMANTIC_MATCHING_H

#include "act/IR/ActOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::act {

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
LogicalResult runSemanticMatching(ModuleOp module,
                                  SmallVectorImpl<MatchCandidate> &results);

} // namespace mlir::act

#endif // ACT_SUPPORT_SEMANTIC_MATCHING_H
