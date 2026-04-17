#ifndef ACT_SUPPORT_SEMANTIC_MATCHING_H
#define ACT_SUPPORT_SEMANTIC_MATCHING_H

#include "act/IR/ActOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/Hashing.h"

#include <optional>
#include <vector>

namespace mlir::act {

struct StaticSliceSpec {
  SmallVector<int64_t, 4> offsets;
  SmallVector<int64_t, 4> sizes;
  SmallVector<int64_t, 4> strides;

  bool operator==(const StaticSliceSpec &other) const {
    return offsets == other.offsets && sizes == other.sizes &&
           strides == other.strides;
  }
};

/// Semantic fingerprint of an instruction's core computation.
struct SemanticFingerprint {
  enum Kind { Named, Generic, Identity, Opaque };
  Kind kind = Opaque;

  /// For Named kind: the op name (e.g., "linalg.matmul").
  StringRef opName;

  /// For Generic kind: structural description of the linalg.generic body.
  SmallVector<AffineMap> indexingMaps;
  SmallVector<utils::IteratorType> iteratorTypes;
  Region *bodyRegion = nullptr; // non-owning, for detailed comparison
  Block *body = nullptr;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;

  llvm::hash_code hash() const;
  bool matches(const SemanticFingerprint &other) const;
  explicit SemanticFingerprint(Operation *op);
};

enum class SemanticEdgeTransformKind {
  Transpose,
  ExtractSlice,
  InsertSlice,
};

struct SemanticEdgeTransform {
  SemanticEdgeTransformKind kind = SemanticEdgeTransformKind::Transpose;
  SmallVector<int64_t, 4> permutation;
  std::optional<StaticSliceSpec> sliceSpec;

  bool operator==(const SemanticEdgeTransform &other) const {
    return kind == other.kind && permutation == other.permutation &&
           sliceSpec == other.sliceSpec;
  }
};

using SemanticEdgeTransformChain = SmallVector<SemanticEdgeTransform, 2>;

struct GraphBoundaryPort {
  unsigned nodeIdx = 0;
  unsigned portIdx = 0;
  unsigned boundaryOperandIdx = std::numeric_limits<unsigned>::max();
  Value value;
  SemanticEdgeTransformChain transforms;
};

struct GraphNode {
  Operation *op = nullptr;
  SemanticFingerprint fp;
  SmallVector<unsigned, 4> inEdgeIdxs;
  SmallVector<unsigned, 4> outEdgeIdxs;
};

struct GraphEdge {
  unsigned producerNodeIdx = 0;
  unsigned consumerNodeIdx = 0;
  unsigned producerResultIdx = 0;
  unsigned consumerOperandIdx = 0;
  SemanticEdgeTransformChain transforms;
  static constexpr unsigned kNullIdx = std::numeric_limits<unsigned>::max();
};

struct GraphBase {
  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;
  SmallVector<GraphBoundaryPort, 4> boundaryInputs;
  SmallVector<GraphBoundaryPort, 4> boundaryOutputs;

  LogicalResult init(Block &b);
  const GraphEdge *findEdge(unsigned producerNodeIdx, unsigned consumerNodeIdx,
                            unsigned producerResultIdx,
                            unsigned consumerOperandIdx) const;
  bool hasAnyEdgeBetween(unsigned lhsNodeIdx, unsigned rhsNodeIdx) const;
  std::optional<unsigned> getNodeIndex(Operation *op) const;
  void dump(raw_ostream &os, StringRef title) const;
};

struct InstructionGraph : GraphBase {
  DefineOp instruction;
  unsigned anchorNodeIdx = 0; // search anchor for matching only
  unsigned domainNodeIdx = 0; // semantic carrier of the full iteration domain
  static FailureOr<InstructionGraph> build(DefineOp defineOp);
  void dump() const;
};

struct ProgramGraph : GraphBase {
  func::FuncOp funcOp;
  static FailureOr<ProgramGraph> build(func::FuncOp funcOp);
  void dump();
};

struct SubgraphBoundaryInputBinding {
  unsigned patternNodeIdx = 0;
  unsigned patternPortIdx = 0;
  unsigned boundaryOperandIdx = GraphEdge::kNullIdx;
  unsigned sourceNodeIdx = GraphEdge::kNullIdx;
  unsigned sourcePortIdx = GraphEdge::kNullIdx;
  Value value;
  SemanticEdgeTransformChain transforms;
};

/// One selected node output at the compute boundary plus its optional
/// materialized external value after output-side layout transforms.
struct SubgraphBoundaryOutputBinding {
  unsigned patternNodeIdx = 0;
  unsigned patternPortIdx = 0;
  unsigned sourceNodeIdx = GraphEdge::kNullIdx;
  unsigned sourcePortIdx = GraphEdge::kNullIdx;
  Value producedValue;
  Value materializedValue;
  SemanticEdgeTransformChain transforms;
};

struct SubgraphMatchCandidate {
  const InstructionGraph *pattern = nullptr;
  SmallVector<Operation *, 4> sourceOps;
  SmallVector<unsigned, 4> patternToSourceNodeIdx;
  SmallVector<SubgraphBoundaryInputBinding, 4> boundaryInputs;
  SmallVector<SubgraphBoundaryOutputBinding, 2> boundaryOutputs;
  unsigned priority = 0;
  unsigned sourceAnchorNodeIdx = 0;

  DefineOp getInstruction() const {
    return pattern ? pattern->instruction : nullptr;
  }
  void dump() const;
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

struct SemanticsGraphNode {
  SmallVector<Operation *, 4> sourceOps;
  unsigned anchorNodeIdx = GraphEdge::kNullIdx; // search-only metadata
  unsigned domainNodeIdx = GraphEdge::kNullIdx;
  DefineOp instruction;
  Operation *domainComputeOp = nullptr;
  Operation *domainSourceOp = nullptr;
  SmallVector<SubgraphBoundaryInputBinding, 4> boundaryInputs;
  SmallVector<SubgraphBoundaryOutputBinding, 2> boundaryOutputs;
};

struct SemanticsGraphEdge {
  unsigned producerNodeIdx = 0;
  unsigned consumerNodeIdx = 0;
  unsigned prodIdx = 0; // boundary output index in producer node
  unsigned consIdx = 0; // consumer operand index at the compute boundary
  SemanticEdgeTransformChain transforms;
};

struct SemanticsGraph {
  func::FuncOp funcOp;
  std::vector<SemanticsGraphNode> nodes;
  std::vector<SemanticsGraphEdge> edges;

  void dump() const;
};

struct InstructionCatalog {
  static FailureOr<InstructionCatalog> build(ModuleOp module);
  SmallVector<const InstructionGraph *, 4>
  lookup(const SemanticFingerprint &anchorFp) const;
  void dump() const;

  SmallVector<InstructionGraph, 8> graphs;
  /// Indices into `graphs`, not pointers — safe across moves.
  DenseMap<llvm::hash_code, SmallVector<unsigned, 4>> byAnchor;
};

using SemanticsGraphs = SmallVector<SemanticsGraph, 2>;

FailureOr<SemanticsGraphs> runSemanticMatching(ModuleOp module);
FailureOr<SemanticsGraph> runSemanticMatching(func::FuncOp func,
                                              InstructionCatalog &catalog);

} // namespace mlir::act

#endif // ACT_SUPPORT_SEMANTIC_MATCHING_H
