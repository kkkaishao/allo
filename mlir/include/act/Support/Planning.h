#ifndef ACT_SUPPORT_PLANNING_H
#define ACT_SUPPORT_PLANNING_H

#include "act/Support/ParamSolving.h"
#include "act/Support/SemanticMatching.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::act {

enum class LogicalTransformKind {
  Transpose,
  ExtractSlice,
  InsertSlice,
};

struct LogicalTransform {
  LogicalTransformKind kind = LogicalTransformKind::Transpose;
  SmallVector<int64_t> permutation;
  std::optional<StaticSliceSpec> sliceSpec;

  static LogicalTransform transpose(ArrayRef<int64_t> perm) {
    LogicalTransform layout;
    layout.kind = LogicalTransformKind::Transpose;
    layout.permutation.assign(perm.begin(), perm.end());
    return layout;
  }

  static LogicalTransform extractSlice(const StaticSliceSpec &slice) {
    LogicalTransform layout;
    layout.kind = LogicalTransformKind::ExtractSlice;
    layout.sliceSpec = slice;
    return layout;
  }

  static LogicalTransform insertSlice(const StaticSliceSpec &slice) {
    LogicalTransform layout;
    layout.kind = LogicalTransformKind::InsertSlice;
    layout.sliceSpec = slice;
    return layout;
  }

  bool isTranspose() const { return kind == LogicalTransformKind::Transpose; }
  bool isExtractSlice() const {
    return kind == LogicalTransformKind::ExtractSlice;
  }
  bool isInsertSlice() const {
    return kind == LogicalTransformKind::InsertSlice;
  }
};

using LogicalTransformChain = SmallVector<LogicalTransform, 2>;

bool isIdentityTransformChain(ArrayRef<LogicalTransform> transforms);

enum class LogicalPlanValueKind {
  FunctionInput,
  Produced,
  MaterializedOutput,
};

/// One consumer use of a logical value.
struct LogicalPlanInputUse {
  unsigned consumerNodeIdx;
  unsigned consumerOperandIdx;
  LogicalTransformChain requiredTransforms;
};

struct LogicalPlanNodeInput {
  unsigned valueId = GraphEdge::kNullIdx;
  LogicalTransformChain requiredTransforms;
};

struct LogicalPlanNodeOutput {
  unsigned valueId = GraphEdge::kNullIdx;
  std::optional<unsigned> writebackTargetValueId;
  LogicalTransformChain writebackTransforms;
};

/// Logical tensor value flowing between compute nodes.
struct LogicalPlanValue {
  LogicalPlanValueKind kind = LogicalPlanValueKind::Produced;
  Value sourceValue;
  RankedTensorType type;
  std::optional<unsigned> definingNodeIdx;
  SmallVector<LogicalPlanInputUse, 1> uses;
};

/// One selected compute node in the logical plan.
struct LogicalPlanNode {
  SmallVector<Operation *, 4> sourceOps;
  DefineOp instruction;
  DenseMap<unsigned, int64_t> solvedParams;
  DenseMap<unsigned, AddrParamKind> paramKinds;
  SmallVector<LogicalPlanNodeInput, 2> inputs;
  SmallVector<LogicalPlanNodeOutput, 1> outputs;
};

/// Selected-only compute/dataflow plan used by Stage 3.
struct LogicalPlan {
  SmallVector<LogicalPlanNode, 4> nodes;
  SmallVector<LogicalPlanValue, 8> values;
  DenseMap<Value, unsigned> externalValueIds;

  void dump() const;
};

/// Layout of a single logical value in the shared buffer.
struct TensorLayout {
  int64_t baseOffset;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
};

/// Planned residence of one instruction operand.
struct OperandResidence {
  StringAttr bufferName;
  int64_t offset;
  int64_t size;
  unsigned operandIdx;
};

/// One planned data movement step.
struct MovementStep {
  DefineOp instruction;
  StringAttr srcBuffer;
  int64_t srcOffset;
  StringAttr dstBuffer;
  int64_t dstOffset;
  int64_t size;
};

/// Layout signature extracted from an identity instruction's addr region.
struct LayoutSignature {
  bool hasTranspose = false;
  SmallVector<int64_t> permutation;
  bool matches(const LayoutSignature &required) const;
};

/// Data movement catalog: maps (srcBuffer, dstBuffer) -> identity DefineOps
/// with layout signatures, enabling layout-aware instruction selection.
struct DataMovementCatalog {
  DenseMap<std::pair<StringAttr, StringAttr>,
           SmallVector<std::pair<LayoutSignature, DefineOp>>>
      entries;

  std::optional<DefineOp> lookup(StringAttr src, StringAttr dst,
                                 const LayoutSignature &required = {}) const;
};

/// Scratchpad forwarding: skip HBM round-trip between consecutive nodes.
struct ForwardingEdge {
  unsigned producerNodeIdx;
  unsigned producerDstOperandIdx;
  unsigned consumerNodeIdx;
  unsigned consumerSrcOperandIdx;
  StringAttr bufferName;
};

struct InputMovementPlan {
  unsigned nodeIdx;
  unsigned srcOperandIdx;
  unsigned hbmValueId;
  LogicalTransformChain hbmTransforms;
  SmallVector<MovementStep> steps;
};

struct AccumulatorInitPlan {
  unsigned nodeIdx;
  unsigned dstOperandIdx;
  unsigned hbmValueId;
  LogicalTransformChain hbmTransforms;
  SmallVector<MovementStep> steps;
};

struct OutputMovementPlan {
  unsigned nodeIdx;
  unsigned dstOperandIdx;
  unsigned hbmValueId;
  LogicalTransformChain hbmTransforms;
  SmallVector<MovementStep> steps;
};

/// Resource annotations derived from a logical plan.
struct ResourcePlan {
  StringAttr bufferName;
  int64_t bufferSize = 0;
  DenseMap<unsigned, TensorLayout> layouts;
  int64_t totalAllocated = 0;

  bool needsDataMovement = false;
  SmallVector<SmallVector<OperandResidence>> operandResidences;
  DataMovementCatalog dmCatalog;
  SmallVector<ForwardingEdge> forwardingEdges;
  SmallVector<InputMovementPlan, 2> inputMovements;
  SmallVector<AccumulatorInitPlan, 1> accumulatorInits;
  SmallVector<OutputMovementPlan, 1> outputMovements;

  void dump() const;
};

FailureOr<LogicalPlan>
buildLogicalPlan(func::FuncOp funcOp, const SemanticsGraph &graph,
                 const GraphParamSolution &paramSolution);

FailureOr<ResourcePlan> buildResourcePlan(func::FuncOp funcOp,
                                          const LogicalPlan &plan,
                                          ModuleOp module);

LogicalTransformChain getRequiredTransforms(const LogicalPlan &plan,
                                            unsigned valueId,
                                            unsigned consumerNodeIdx,
                                            unsigned consumerOperandIdx);

} // namespace mlir::act

#endif // ACT_SUPPORT_PLANNING_H
