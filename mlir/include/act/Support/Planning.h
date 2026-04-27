#ifndef ACT_SUPPORT_PLANNING_H
#define ACT_SUPPORT_PLANNING_H

#include "act/Support/ParamSolving.h"
#include "act/Support/SemanticMatching.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

namespace mlir::act {

enum class PlanValueKind {
  HBMInput,
  Placeholder,
  Produced,
};

enum class PlanActionKind {
  Compute,
  Writeback,
};

enum class PlanScheduleKind {
  Compute,
  Move,
};

struct FlatStridedRegion {
  int64_t base = 0;
  SmallVector<int64_t, 4> sizes;
  SmallVector<int64_t, 4> strides;
};

struct PhysicalRegion {
  StringAttr bufferName;
  int64_t offset = 0;
  int64_t size = 0;
};

struct ScratchResource {
  StringAttr bufferName;
  BufferTypeInterface bufferType;
  int64_t capacity = 0;
};

struct HBMAllocation {
  Value value;
  RankedTensorType type;
  unsigned boundaryIdx = kNullId;
  int64_t base = 0;
  SmallVector<int64_t, 4> shape;
  SmallVector<int64_t, 4> strides;
  bool isResult = false;
};

struct ValueLifetime {
  unsigned firstAction = kNullId;
  unsigned lastAction = kNullId;

  bool isValid() { return firstAction != kNullId && lastAction != kNullId; }
};

struct ValuePlacement {
  unsigned valueId = kNullId;
  PhysicalRegion region;
  unsigned firstAction = kNullId;
  unsigned lastAction = kNullId;
  bool overCapacity = false;
};

struct PlanValueUse {
  unsigned nodeIdx = kNullId;
  unsigned accessOperandIdx = kNullId;
};

struct PlanValueWriteback {
  unsigned nodeIdx = kNullId;
  unsigned outputIdx = kNullId;
  unsigned funcResultIdx = kNullId;
  unsigned actionId = kNullId;
  Value value;
  LayoutChain layout;
  std::optional<FlatStridedRegion> hbmRegion;
};

struct LayoutSignature {
  bool hasTranspose = false;
  SmallVector<int64_t, 4> permutation;

  bool matches(LayoutSignature &required);
};

struct PlanValue {
  PlanValueKind kind = PlanValueKind::HBMInput;
  Value value;
  Value baseValue;
  RankedTensorType type;
  std::optional<unsigned> definingNodeIdx;
  std::optional<unsigned> outputIdx;
  ValueLifetime lifetime;
  bool requiresMovement = false;
  SmallVector<unsigned, 2> placementIds;
  SmallVector<PlanValueUse, 4> uses;
  SmallVector<PlanValueWriteback, 2> writebacks;
};

struct PlanOperandAccess {
  SymbolicAccess access;
  AccessRole role = AccessRole::Read;
  std::optional<unsigned> inputValueId;
  Value value;
  LayoutChain layout;
  unsigned sourceNodeId = kNullId;
  unsigned sourcePatternNodeId = kNullId;
  unsigned sourceResultId = kNullId;
  std::optional<unsigned> placementId;
  std::optional<FlatStridedRegion> hbmRegion;
  std::optional<PhysicalRegion> scratchRegion;
  bool forwarded = false;
  bool requiresMovement = false;
  bool requiresInit = false;
  bool overCapacity = false;
};

struct PlanNode {
  SemanticGraphNode *semanticNode = nullptr;
  DefineOp instruction;
  unsigned actionId = kNullId;
  DenseMap<unsigned, int64_t> paramBindings;
  SmallVector<PlanOperandAccess, 4> operands;
  SmallVector<unsigned, 2> outputValueIds;
};

struct PlanActionAccess {
  AccessRole role = AccessRole::Read;
  unsigned nodeIdx = kNullId;
  unsigned accessOperandIdx = kNullId;
  unsigned valueId = kNullId;
  std::optional<unsigned> placementId;
  std::optional<PhysicalRegion> scratchRegion;
  std::optional<FlatStridedRegion> hbmRegion;
};

struct PlanAction {
  PlanActionKind kind = PlanActionKind::Compute;
  unsigned nodeIdx = kNullId;
  unsigned outputIdx = kNullId;
  unsigned valueId = kNullId;
  unsigned writebackIdx = kNullId;
  SmallVector<PlanActionAccess, 4> accesses;
};

struct PlanMoveEndpoint {
  bool isHBM = false;
  bool isResult = false;
  unsigned boundaryIdx = kNullId;
  unsigned valueId = kNullId;
  std::optional<FlatStridedRegion> hbmRegion;
  std::optional<StaticSlice> hbmSlice;
  std::optional<PhysicalRegion> scratchRegion;
};

struct PlanMoveNode {
  DefineOp instruction;
  DenseMap<unsigned, int64_t> paramBindings;
  PlanMoveEndpoint src;
  PlanMoveEndpoint dst;
  LayoutSignature layout;
  unsigned anchorActionId = kNullId;
};

struct PlanScheduleStep {
  PlanScheduleKind kind = PlanScheduleKind::Compute;
  unsigned nodeIdx = kNullId;
};

struct DataMovementCatalog {
  DenseMap<std::pair<StringAttr, StringAttr>,
           SmallVector<std::pair<LayoutSignature, DefineOp>, 2>>
      entries;

  std::optional<DefineOp> lookup(StringAttr src, StringAttr dst,
                                 LayoutSignature required = {});
};

struct ExecutionPlan {
  func::FuncOp func;
  SmallVector<PlanNode, 4> nodes;
  SmallVector<PlanMoveNode, 8> moveNodes;
  SmallVector<PlanScheduleStep, 16> schedule;
  SmallVector<PlanValue, 8> values;
  SmallVector<PlanAction, 8> actions;
  SmallVector<ValuePlacement, 8> placements;
  DenseMap<Value, unsigned> externalValueIds;
  StringAttr hbmBufferName;
  int64_t hbmCapacity = 0;
  int64_t hbmUsed = 0;
  SmallVector<HBMAllocation, 4> hbmInputs;
  SmallVector<HBMAllocation, 2> hbmResults;
  DenseMap<Value, unsigned> hbmInputIds;
  SmallVector<ScratchResource, 4> scratchResources;

  void dump(llvm::raw_ostream &os);
};

FailureOr<ExecutionPlan> buildExecutionPlan(SemanticGraph &graph,
                                            GraphParamSolution &solutions);

} // namespace mlir::act

#endif // ACT_SUPPORT_PLANNING_H
