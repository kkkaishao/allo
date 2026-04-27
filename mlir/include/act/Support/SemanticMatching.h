#ifndef ACT_SUPPORT_SEMANTICMATCHING_H
#define ACT_SUPPORT_SEMANTICMATCHING_H

#include "act/IR/ActOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>

namespace mlir::act {
constexpr unsigned kNullId = std::numeric_limits<unsigned>::max();

struct SemanticIdentity {
  Operation *op;
  StringRef opName;

  // precomputed hash for quick lookup/pruning
  llvm::hash_code hashValue;

  bool semanticallyMatches(const SemanticIdentity &other) const;

  explicit SemanticIdentity(Operation *op);
};

struct ProgramGraphNode {
  SemanticIdentity identity;
  SmallVector<unsigned, 2> inEdgeIds;
  SmallVector<unsigned, 2> outEdgeIds;
  explicit ProgramGraphNode(Operation *op) : identity(op) {}
};

struct StaticSlice {
  SmallVector<int64_t, 4> offsets;
  SmallVector<int64_t, 4> sizes;
  SmallVector<int64_t, 4> strides;
};

struct LayoutChain {
  Value source;
  std::optional<StaticSlice> slice;
  std::optional<StaticSlice> targetSlice;
  SmallVector<Operation *, 4> layoutOps; // e.g., transpose, extract_slice
};

struct ProgramGraphEdge {
  unsigned producerNodeId = kNullId;
  unsigned consumerNodeId = kNullId;
  unsigned producerResultId = kNullId;
  unsigned consumerOperandId = kNullId;
  Value value;
  LayoutChain layoutChain;

  bool isExternalInput() const { return producerNodeId == kNullId; }
  bool isExternalOutput() const { return consumerNodeId == kNullId; }
  bool isExternal() const { return isExternalInput() || isExternalOutput(); }
};

struct ProgramGraph {
  std::vector<ProgramGraphNode> nodes;
  std::vector<ProgramGraphEdge> edges;

  func::FuncOp func;
  static FailureOr<ProgramGraph> build(func::FuncOp func);
  void dump(llvm::raw_ostream &os) const;
  const ProgramGraphEdge *findEdge(unsigned producerNodeId,
                                   unsigned consumerNodeId,
                                   unsigned producerResultId,
                                   unsigned consumerOperandId) const;
};

using InstructionGraphNode = ProgramGraphNode; // same structure for now
using InstructionGraphEdge = ProgramGraphEdge; // same structure for now

struct InstructionGraph {
  DefineOp instruction;
  std::vector<InstructionGraphNode> nodes;
  std::vector<InstructionGraphEdge> edges;

  unsigned anchorNodeId; // search anchor for matching only

  static FailureOr<InstructionGraph> build(DefineOp defineOp);
  void dump(llvm::raw_ostream &os) const;
};

struct InstructionCollection {
  static FailureOr<InstructionCollection> build(ModuleOp module);
  SmallVector<const InstructionGraph *, 2>
  lookup(SemanticIdentity &anchor) const;

  std::vector<InstructionGraph> instructions;
  llvm::DenseMap<llvm::hash_code, SmallVector<unsigned, 2>> byAnchorHash;
};

struct MatchCandidate {
  const InstructionGraph *instruction;
  SmallVector<SemanticIdentity, 4> sourceOps;
  // unsigned priority = 0; // TODO: add priority logic
  unsigned programAnchorNodeId = 0; // for debugging

  // flat mapping between pattern node idx and program node idx
  SmallVector<unsigned, 4> patternToProgramNodeId;
};

using MatchCandidates = SmallVector<MatchCandidate, 4>;

struct SemanticInputBinding {
  unsigned accessOperandIdx = 0;
  unsigned patternNodeId = 0;
  unsigned patternOperandId = 0;
  unsigned sourceNodeId = kNullId;
  unsigned sourcePatternNodeId = kNullId;
  unsigned sourceResultId = kNullId;
  Value value;
  LayoutChain layout;
};

struct SemanticOutputBinding {
  unsigned outputIdx = 0;
  unsigned accessOperandIdx = 0;
  unsigned patternNodeId = 0;
  unsigned patternResultId = 0;
  unsigned consumerNodeId = kNullId;
  unsigned consumerPatternNodeId = kNullId;
  unsigned consumerOperandId = kNullId;
  Value value;
  LayoutChain layout;
};

struct SemanticGraphNode {
  SmallVector<SemanticIdentity, 4> sourceOps;
  const InstructionGraph &pattern;
  unsigned anchorNodeId = kNullId; // search-only metadata
  SmallVector<unsigned, 2> inEdgeIds;
  SmallVector<unsigned, 2> outEdgeIds;
  SmallVector<SemanticInputBinding, 4> inputBindings;
  SmallVector<SemanticOutputBinding, 2> outputBindings;

  explicit SemanticGraphNode(const InstructionGraph &pattern)
      : pattern(pattern) {}
};

struct SemanticGraphEdge {
  unsigned producerNodeId = 0;
  unsigned consumerNodeId = 0;
  unsigned producerResultId = 0;
  unsigned consumerOperandId = 0;
  unsigned producerPatternNodeId = 0;
  unsigned consumerPatternNodeId = 0;
  LayoutChain transforms;
};

struct SemanticGraph {
  func::FuncOp func;
  std::vector<SemanticGraphNode> nodes;
  std::vector<SemanticGraphEdge> edges;

  void dump(llvm::raw_ostream &os) const;
};

FailureOr<SemanticGraph> runSemanticMatching(func::FuncOp func,
                                             InstructionCollection &collection);
} // namespace mlir::act

#endif // ACT_SUPPORT_SEMANTICMATCHING_H
