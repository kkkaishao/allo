#include "act/Support/Planning.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <string>
#include <utility>

#define DEBUG_TYPE "planning"

using namespace mlir;
using namespace mlir::act;

using llvm::dbgs;

static Operation *getErrorOp(SemanticGraphNode &node) {
  assert(!node.sourceOps.empty() && "expected matched source op");
  return node.sourceOps.front().op;
}

static std::string getOperationName(Operation *op) {
  assert(op && "expected non-null operation");
  return op->getName().getStringRef().str();
}

static std::string formatValue(Value value) {
  if (!value)
    return "n/a";

  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return getOperationName(blockArg.getOwner()->getParentOp()) + ":%arg" +
           std::to_string(blockArg.getArgNumber());

  return getOperationName(value.getDefiningOp()) + ":%" +
         std::to_string(cast<OpResult>(value).getResultNumber());
}

static void printIntArray(llvm::raw_ostream &os, ArrayRef<int64_t> values) {
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
}

static void printLayoutChain(llvm::raw_ostream &os, Value value,
                             LayoutChain &layout) {
  os << " value=" << formatValue(value)
     << " base=" << formatValue(layout.source);
  if (layout.slice) {
    os << " slice=(";
    printIntArray(os, layout.slice->offsets);
    os << ", ";
    printIntArray(os, layout.slice->sizes);
    os << ", ";
    printIntArray(os, layout.slice->strides);
    os << ")";
  }
  if (!layout.layoutOps.empty()) {
    os << " ops=[";
    llvm::interleaveComma(layout.layoutOps, os,
                          [&](Operation *op) { os << op->getName(); });
    os << "]";
  }
}

static void printFlatRegion(llvm::raw_ostream &os, FlatStridedRegion &region) {
  os << "base=" << region.base << " sizes=";
  printIntArray(os, region.sizes);
  os << " strides=";
  printIntArray(os, region.strides);
}

static void printPhysicalRegion(llvm::raw_ostream &os, PhysicalRegion &region) {
  os << "@" << region.bufferName.getValue() << " offset=" << region.offset
     << " size=" << region.size;
}

static StringRef getValueKindName(PlanValueKind kind) {
  switch (kind) {
  case PlanValueKind::HBMInput:
    return "hbm-input";
  case PlanValueKind::Placeholder:
    return "placeholder";
  case PlanValueKind::Produced:
    return "produced";
  }
  llvm_unreachable("unknown plan value kind");
}

static StringRef getActionKindName(PlanActionKind kind) {
  switch (kind) {
  case PlanActionKind::Compute:
    return "compute";
  case PlanActionKind::Writeback:
    return "writeback";
  }
  llvm_unreachable("unknown plan action kind");
}

static StringRef getScheduleKindName(PlanScheduleKind kind) {
  switch (kind) {
  case PlanScheduleKind::Compute:
    return "compute";
  case PlanScheduleKind::Move:
    return "move";
  }
  llvm_unreachable("unknown plan schedule kind");
}

static StringRef accessRoleToString(AccessRole role) {
  switch (role) {
  case AccessRole::Read:
    return "read";
  case AccessRole::Write:
    return "write";
  case AccessRole::ReadWrite:
    return "readwrite";
  }
  llvm_unreachable("unknown access role");
}

static SmallVector<int64_t, 4> getTensorShape(RankedTensorType type) {
  SmallVector<int64_t, 4> shape;
  llvm::append_range(shape, type.getShape());
  return shape;
}

static int64_t getNumElements(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t dim : shape)
    result *= dim;
  return result;
}

bool LayoutSignature::matches(LayoutSignature &required) {
  if (hasTranspose != required.hasTranspose)
    return false;
  if (required.hasTranspose && permutation != required.permutation)
    return false;
  return true;
}

std::optional<DefineOp> DataMovementCatalog::lookup(StringAttr src,
                                                    StringAttr dst,
                                                    LayoutSignature required) {
  auto it = entries.find({src, dst});
  if (it == entries.end())
    return std::nullopt;
  for (auto &[signature, instruction] : it->second)
    if (signature.matches(required))
      return instruction;
  return std::nullopt;
}

static bool isIdentityInstruction(DefineOp defineOp) {
  Block &computeBlock = defineOp.getSemanticsBlock();
  if (!computeBlock.without_terminator().empty())
    return false;

  Operation *yieldOp = computeBlock.getTerminator();
  return llvm::all_of(yieldOp->getOperands(), [](Value operand) {
    return isa<BlockArgument>(operand);
  });
}

static LayoutSignature extractLayoutSignature(DefineOp defineOp) {
  LayoutSignature signature;
  Block &addrBlock = defineOp.getAccessBlock();
  Operation *yieldOp = addrBlock.getTerminator();
  Operation *op = yieldOp->getOperand(0).getDefiningOp();
  while (op && !isa<StridedOp>(op)) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      signature.hasTranspose = true;
      auto permutation = transpose.getPermutation();
      signature.permutation.assign(permutation.begin(), permutation.end());
      op = transpose.getSource().getDefiningOp();
      continue;
    }
    if (auto expand = dyn_cast<ExpandShapeOp>(op)) {
      op = expand.getSource().getDefiningOp();
      continue;
    }
    if (auto collapse = dyn_cast<CollapseShapeOp>(op)) {
      op = collapse.getSource().getDefiningOp();
      continue;
    }
    break;
  }
  return signature;
}

static DataMovementCatalog buildDataMovementCatalog(ModuleOp module) {
  DataMovementCatalog catalog;
  module.walk([&](DefineOp defineOp) {
    if (!isIdentityInstruction(defineOp))
      return;
    if (defineOp.getSources().size() != 1 ||
        defineOp.getDestinations().size() != 1)
      return;

    auto sources = defineOp.getSources().getAsRange<FlatSymbolRefAttr>();
    auto destinations =
        defineOp.getDestinations().getAsRange<FlatSymbolRefAttr>();
    StringAttr srcBuffer = (*sources.begin()).getAttr();
    StringAttr dstBuffer = (*destinations.begin()).getAttr();
    catalog.entries[{srcBuffer, dstBuffer}].push_back(
        {extractLayoutSignature(defineOp), defineOp});
  });
  return catalog;
}

static SmallVector<int64_t, 4> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, 4> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

static int64_t getBufferElementCapacity(DeclareBufferOp buffer) {
  BufferTypeInterface bufferType = buffer.getBufferType();
  if (auto hbm = dyn_cast<HBMBufferType>(bufferType))
    return buffer.getSize() * getNumElements(hbm.getShape());
  return buffer.getSize();
}

static FailureOr<DeclareBufferOp> identifyHBMBuffer(ModuleOp module) {
  DeclareBufferOp found = nullptr;
  module.walk([&](DeclareBufferOp buffer) {
    if (isa<HBMBufferType>(buffer.getBufferType())) {
      assert(!found && "expected at most one HBM buffer for Round3");
      found = buffer;
    }
  });
  if (found)
    return found;
  return module.emitError() << "no HBM buffer found for execution planning";
}

static LogicalResult initializeScratchResources(ExecutionPlan &plan) {
  ModuleOp module = plan.func->getParentOfType<ModuleOp>();
  assert(module && "planned function should belong to a module");

  module.walk([&](DeclareBufferOp buffer) {
    BufferTypeInterface bufferType = buffer.getBufferType();
    if (isa<HBMBufferType>(bufferType))
      return;

    ScratchResource resource;
    resource.bufferName = buffer.getSymNameAttr();
    resource.bufferType = bufferType;
    resource.capacity = getBufferElementCapacity(buffer);
    plan.scratchResources.push_back(std::move(resource));
  });

  return success();
}

static LogicalResult appendHBMAllocation(ExecutionPlan &plan, Value value,
                                         RankedTensorType type,
                                         unsigned boundaryIdx, bool isResult,
                                         Operation *errorOp) {
  if (!type.hasStaticShape())
    return errorOp->emitError() << "dynamic HBM boundary shape is unsupported";

  HBMAllocation allocation;
  allocation.value = value;
  allocation.type = type;
  allocation.boundaryIdx = boundaryIdx;
  allocation.base = plan.hbmUsed;
  allocation.shape = getTensorShape(type);
  allocation.strides = computeRowMajorStrides(allocation.shape);
  allocation.isResult = isResult;

  int64_t numElements = getNumElements(allocation.shape);
  if (plan.hbmUsed + numElements > plan.hbmCapacity)
    return errorOp->emitError() << "HBM allocation exceeds capacity of @"
                                << plan.hbmBufferName.getValue();

  plan.hbmUsed += numElements;
  if (isResult) {
    plan.hbmResults.push_back(std::move(allocation));
    return success();
  }

  assert(value && "expected HBM input value");
  plan.hbmInputIds[value] = plan.hbmInputs.size();
  plan.hbmInputs.push_back(std::move(allocation));
  return success();
}

static LogicalResult initializeHBMMapping(ExecutionPlan &plan) {
  ModuleOp module = plan.func->getParentOfType<ModuleOp>();
  assert(module && "planned function should belong to a module");
  auto hbmOr = identifyHBMBuffer(module);
  if (failed(hbmOr))
    return failure();

  plan.hbmBufferName = hbmOr->getSymNameAttr();
  plan.hbmCapacity = getBufferElementCapacity(*hbmOr);

  for (BlockArgument arg : plan.func.getArguments()) {
    auto type = dyn_cast<RankedTensorType>(arg.getType());
    if (!type)
      continue;
    if (failed(appendHBMAllocation(plan, arg, type, arg.getArgNumber(),
                                   /*isResult=*/false, plan.func)))
      return failure();
  }

  for (auto [idx, type] : llvm::enumerate(plan.func.getResultTypes())) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType)
      continue;
    if (failed(appendHBMAllocation(plan, Value(), tensorType, idx,
                                   /*isResult=*/true, plan.func)))
      return failure();
  }

  return success();
}

static LogicalResult getFlatRegion(HBMAllocation &allocation,
                                   std::optional<StaticSlice> slice,
                                   FlatStridedRegion &region,
                                   Operation *errorOp) {
  region.base = allocation.base;
  if (!slice) {
    region.sizes = allocation.shape;
    region.strides = allocation.strides;
    return success();
  }

  if (slice->offsets.size() != allocation.shape.size() ||
      slice->sizes.size() != allocation.shape.size() ||
      slice->strides.size() != allocation.shape.size())
    return errorOp->emitError() << "HBM slice rank does not match allocation";

  for (unsigned dim = 0; dim < allocation.shape.size(); ++dim) {
    if (slice->offsets[dim] < 0 || slice->sizes[dim] <= 0 ||
        slice->strides[dim] <= 0)
      return errorOp->emitError() << "invalid HBM slice";
    int64_t end =
        slice->offsets[dim] + (slice->sizes[dim] - 1) * slice->strides[dim];
    if (end >= allocation.shape[dim])
      return errorOp->emitError() << "HBM slice exceeds allocation shape";
    region.base += slice->offsets[dim] * allocation.strides[dim];
    region.sizes.push_back(slice->sizes[dim]);
    region.strides.push_back(slice->strides[dim] * allocation.strides[dim]);
  }

  return success();
}

static HBMAllocation *findInputHBMAllocation(ExecutionPlan &plan, Value value) {
  auto it = plan.hbmInputIds.find(value);
  if (it == plan.hbmInputIds.end())
    return nullptr;
  assert(it->second < plan.hbmInputs.size() && "invalid HBM input id");
  return &plan.hbmInputs[it->second];
}

static FailureOr<RankedTensorType> getTensorType(Value value,
                                                 Operation *errorOp) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return errorOp->emitError() << "expected ranked tensor plan value";
  return type;
}

static Value getLayoutBase(LayoutChain &layout) {
  assert(layout.source && "expected layout chain base value");
  return layout.source;
}

static FailureOr<unsigned> getOrCreateExternalValue(ExecutionPlan &plan,
                                                    Value baseValue,
                                                    Operation *errorOp) {
  auto it = plan.externalValueIds.find(baseValue);
  if (it != plan.externalValueIds.end())
    return it->second;

  auto type = getTensorType(baseValue, errorOp);
  if (failed(type))
    return failure();

  unsigned valueId = plan.values.size();
  PlanValue value;
  value.kind = findInputHBMAllocation(plan, baseValue)
                   ? PlanValueKind::HBMInput
                   : PlanValueKind::Placeholder;
  value.value = baseValue;
  value.baseValue = baseValue;
  value.type = *type;
  plan.externalValueIds[baseValue] = valueId;
  plan.values.push_back(std::move(value));
  return valueId;
}

static FailureOr<unsigned> createProducedValue(ExecutionPlan &plan,
                                               SemanticOutputBinding &binding,
                                               unsigned nodeIdx,
                                               Operation *errorOp) {
  Value baseValue = getLayoutBase(binding.layout);
  auto type = getTensorType(baseValue, errorOp);
  if (failed(type))
    return failure();

  unsigned valueId = plan.values.size();
  PlanValue value;
  value.kind = PlanValueKind::Produced;
  value.value = baseValue;
  value.baseValue = baseValue;
  value.type = *type;
  value.definingNodeIdx = nodeIdx;
  value.outputIdx = binding.outputIdx;
  plan.values.push_back(std::move(value));
  return valueId;
}

static SemanticOutputBinding *findOutputBinding(SemanticGraphNode &node,
                                                unsigned outputIdx) {
  for (SemanticOutputBinding &binding : node.outputBindings)
    if (binding.outputIdx == outputIdx)
      return &binding;
  return nullptr;
}

static PlanOperandAccess *findPlanOperand(PlanNode &node,
                                          unsigned accessOperandIdx) {
  for (PlanOperandAccess &operand : node.operands)
    if (operand.access.operandIdx == accessOperandIdx)
      return &operand;
  return nullptr;
}

static FailureOr<unsigned> findProducedValue(ExecutionPlan &plan,
                                             SemanticGraph &graph,
                                             SemanticInputBinding &binding,
                                             Operation *errorOp) {
  assert(binding.sourceNodeId != kNullId && "expected internal producer");
  assert(binding.sourceNodeId < graph.nodes.size() &&
         "producer node id should be valid");
  assert(binding.sourceNodeId < plan.nodes.size() &&
         "producer plan node id should be valid");

  SemanticGraphNode &producerGraphNode = graph.nodes[binding.sourceNodeId];
  PlanNode &producerPlanNode = plan.nodes[binding.sourceNodeId];
  for (SemanticOutputBinding &output : producerGraphNode.outputBindings) {
    if (output.patternNodeId != binding.sourcePatternNodeId ||
        output.patternResultId != binding.sourceResultId)
      continue;
    assert(output.outputIdx < producerPlanNode.outputValueIds.size() &&
           "output index should be valid");
    unsigned valueId = producerPlanNode.outputValueIds[output.outputIdx];
    assert(valueId != kNullId && "producer output should be created first");
    return valueId;
  }

  return errorOp->emitError()
         << "failed to find producer output for plan input";
}

static LogicalResult addPlanUse(ExecutionPlan &plan, SemanticGraph &graph,
                                unsigned nodeIdx,
                                SemanticInputBinding &binding) {
  assert(nodeIdx < plan.nodes.size() && "plan node id should be valid");
  PlanNode &node = plan.nodes[nodeIdx];
  PlanOperandAccess *operand = findPlanOperand(node, binding.accessOperandIdx);
  if (!operand)
    return getErrorOp(*node.semanticNode)->emitError()
           << "input binding references missing access operand "
           << binding.accessOperandIdx;
  if (operand->inputValueId)
    return getErrorOp(*node.semanticNode)->emitError()
           << "access operand " << binding.accessOperandIdx
           << " assigned more than once in execution plan";

  FailureOr<unsigned> valueId;
  if (binding.sourceNodeId == kNullId) {
    Value baseValue = getLayoutBase(binding.layout);
    valueId = getOrCreateExternalValue(plan, baseValue,
                                       getErrorOp(*node.semanticNode));
  } else {
    valueId =
        findProducedValue(plan, graph, binding, getErrorOp(*node.semanticNode));
  }
  if (failed(valueId))
    return failure();

  operand->inputValueId = *valueId;
  if (operand->role == AccessRole::Write)
    operand->role = AccessRole::ReadWrite;
  operand->value = binding.value;
  operand->layout = binding.layout;
  operand->sourceNodeId = binding.sourceNodeId;
  operand->sourcePatternNodeId = binding.sourcePatternNodeId;
  operand->sourceResultId = binding.sourceResultId;

  PlanValueUse use;
  use.nodeIdx = nodeIdx;
  use.accessOperandIdx = binding.accessOperandIdx;
  plan.values[*valueId].uses.push_back(std::move(use));
  return success();
}

static LogicalResult addPlanWriteback(ExecutionPlan &plan, unsigned nodeIdx,
                                      SemanticOutputBinding &binding) {
  if (binding.consumerNodeId != kNullId)
    return success();

  assert(nodeIdx < plan.nodes.size() && "plan node id should be valid");
  PlanNode &node = plan.nodes[nodeIdx];
  assert(binding.outputIdx < node.outputValueIds.size() &&
         "output index should be valid");
  unsigned valueId = node.outputValueIds[binding.outputIdx];
  assert(valueId != kNullId && "output value should be created first");

  PlanValueWriteback writeback;
  writeback.nodeIdx = nodeIdx;
  writeback.outputIdx = binding.outputIdx;
  writeback.funcResultIdx = binding.consumerOperandId;
  writeback.value = binding.value;
  writeback.layout = binding.layout;
  plan.values[valueId].writebacks.push_back(std::move(writeback));
  return success();
}

static LogicalResult validatePlanNodeInputs(PlanNode &node) {
  for (PlanOperandAccess &operand : node.operands) {
    if (!accessReads(operand.role) || operand.inputValueId)
      continue;
    return getErrorOp(*node.semanticNode)->emitError()
           << "execution plan access operand " << operand.access.operandIdx
           << " was not assigned by boundary binding";
  }
  return success();
}

static HBMAllocation *findResultHBMAllocation(ExecutionPlan &plan,
                                              unsigned resultIdx) {
  for (HBMAllocation &allocation : plan.hbmResults)
    if (allocation.boundaryIdx == resultIdx)
      return &allocation;
  return nullptr;
}

static LogicalResult annotateHBMRegions(ExecutionPlan &plan) {
  for (PlanNode &node : plan.nodes) {
    Operation *errorOp = getErrorOp(*node.semanticNode);
    for (PlanOperandAccess &operand : node.operands) {
      if (!operand.inputValueId)
        continue;
      if (operand.sourceNodeId != kNullId)
        continue;

      HBMAllocation *allocation =
          findInputHBMAllocation(plan, operand.layout.source);
      if (!allocation)
        continue;

      FlatStridedRegion region;
      if (failed(getFlatRegion(*allocation, operand.layout.slice, region,
                               errorOp)))
        return failure();
      operand.hbmRegion = std::move(region);
    }
  }

  for (PlanValue &value : plan.values) {
    for (PlanValueWriteback &writeback : value.writebacks) {
      if (writeback.funcResultIdx == kNullId)
        return plan.func.emitError()
               << "writeback is missing function result index";
      HBMAllocation *allocation =
          findResultHBMAllocation(plan, writeback.funcResultIdx);
      if (!allocation)
        return plan.func.emitError()
               << "writeback references non-HBM function result "
               << writeback.funcResultIdx;

      FlatStridedRegion region;
      if (failed(getFlatRegion(*allocation, writeback.layout.targetSlice,
                               region, plan.func)))
        return failure();
      writeback.hbmRegion = std::move(region);
    }
  }

  return success();
}

static ScratchResource *findScratchResource(ExecutionPlan &plan,
                                            StringAttr bufferName) {
  for (ScratchResource &resource : plan.scratchResources)
    if (resource.bufferName == bufferName)
      return &resource;
  return nullptr;
}

static std::optional<int64_t>
evaluateExpr(SymExpr &expr, DenseMap<unsigned, int64_t> &paramBindings) {
  switch (expr.kind) {
  case SymExpr::Kind::Constant:
    return expr.value;
  case SymExpr::Kind::Param: {
    auto it = paramBindings.find(expr.paramIdx);
    if (it == paramBindings.end())
      return std::nullopt;
    return it->second;
  }
  case SymExpr::Kind::Add: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = evaluateExpr(*expr.lhs, paramBindings);
    auto rhs = evaluateExpr(*expr.rhs, paramBindings);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  case SymExpr::Kind::Mul: {
    assert(expr.lhs && expr.rhs && "expected binary expression operands");
    auto lhs = evaluateExpr(*expr.lhs, paramBindings);
    auto rhs = evaluateExpr(*expr.rhs, paramBindings);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs * *rhs;
  }
  }
  llvm_unreachable("unknown symbolic expression kind");
}

static FailureOr<int64_t> getStorageDim(SymShape &dims, PlanNode &node,
                                        Operation *errorOp, StringRef name) {
  if (dims.size() != 1)
    return errorOp->emitError()
           << "Round4 scratch allocation supports only 1D on-chip " << name
           << " expressions";
  auto value = evaluateExpr(dims.front(), node.paramBindings);
  if (!value)
    return errorOp->emitError()
           << "failed to resolve on-chip " << name << " expression";
  return *value;
}

static FailureOr<int64_t> getScratchAccessSize(PlanNode &node,
                                               PlanOperandAccess &operand,
                                               Operation *errorOp) {
  auto count =
      getStorageDim(operand.access.storage.counts, node, errorOp, "count");
  auto stride =
      getStorageDim(operand.access.storage.strides, node, errorOp, "stride");
  if (failed(count) || failed(stride))
    return failure();
  if (*count <= 0)
    return errorOp->emitError() << "on-chip access count must be positive";
  if (*stride != 1)
    return errorOp->emitError()
           << "Round4 scratch allocation supports only unit-stride on-chip "
              "accesses";
  return *count;
}

static bool overlaps(PhysicalRegion &lhs, PhysicalRegion &rhs) {
  assert(lhs.bufferName == rhs.bufferName && "expected same-buffer regions");
  int64_t lhsEnd = lhs.offset + lhs.size;
  int64_t rhsEnd = rhs.offset + rhs.size;
  return lhs.offset < rhsEnd && rhs.offset < lhsEnd;
}

static bool lifetimesOverlap(unsigned lhsFirst, unsigned lhsLast,
                             unsigned rhsFirst, unsigned rhsLast) {
  return lhsFirst <= rhsLast && rhsFirst <= lhsLast;
}

static bool placementConflicts(ValuePlacement &placement,
                               PhysicalRegion &region, unsigned firstAction,
                               unsigned lastAction) {
  if (placement.region.bufferName != region.bufferName)
    return false;
  if (!lifetimesOverlap(placement.firstAction, placement.lastAction,
                        firstAction, lastAction))
    return false;
  return overlaps(placement.region, region);
}

static FailureOr<PhysicalRegion> allocateRegion(
    ExecutionPlan &plan,
    DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> &localRegions,
    StringAttr bufferName, int64_t size, unsigned firstAction,
    unsigned lastAction, Operation *errorOp, bool &overCapacity) {
  ScratchResource *resource = findScratchResource(plan, bufferName);
  if (!resource)
    return errorOp->emitError()
           << "unknown scratch buffer @" << bufferName.getValue();
  if (size <= 0)
    return errorOp->emitError() << "scratch allocation size must be positive";

  overCapacity = false;
  if (size > resource->capacity) {
    overCapacity = true;
    return PhysicalRegion{bufferName, 0, size};
  }

  int64_t offset = 0;
  while (offset + size <= resource->capacity) {
    PhysicalRegion candidate{bufferName, offset, size};
    int64_t nextOffset = offset;

    for (PhysicalRegion &region : localRegions[bufferName])
      if (overlaps(candidate, region))
        nextOffset = std::max(nextOffset, region.offset + region.size);

    for (ValuePlacement &placement : plan.placements)
      if (placementConflicts(placement, candidate, firstAction, lastAction))
        nextOffset = std::max(nextOffset,
                              placement.region.offset + placement.region.size);

    if (nextOffset == offset)
      return candidate;
    offset = nextOffset;
  }

  overCapacity = true;
  return PhysicalRegion{bufferName, 0, size};
}

static LogicalResult bindBasisToOffset(PlanNode &node,
                                       PlanOperandAccess &operand,
                                       int64_t offset, Operation *errorOp) {
  SymShape &basis = operand.access.storage.basis;
  if (basis.size() != 1)
    return errorOp->emitError()
           << "Round4 scratch allocation supports only 1D on-chip basis "
              "expressions";

  SymExpr &expr = basis.front();
  auto known = evaluateExpr(expr, node.paramBindings);
  if (known) {
    if (*known == offset)
      return success();
    return errorOp->emitError()
           << "scratch basis expression is already bound to " << *known
           << " but allocator selected " << offset;
  }

  if (auto paramIdx = expr.getParamIdx()) {
    node.paramBindings[*paramIdx] = offset;
    return success();
  }

  return errorOp->emitError()
         << "cannot bind non-param scratch basis expression";
}

static unsigned createPlacement(ExecutionPlan &plan, unsigned valueId,
                                PhysicalRegion region, unsigned firstAction,
                                unsigned lastAction, bool overCapacity) {
  assert(valueId < plan.values.size() && "invalid value id for placement");
  unsigned placementId = plan.placements.size();
  ValuePlacement placement;
  placement.valueId = valueId;
  placement.region = region;
  placement.firstAction = firstAction;
  placement.lastAction = lastAction;
  placement.overCapacity = overCapacity;
  plan.placements.push_back(std::move(placement));
  plan.values[valueId].placementIds.push_back(placementId);
  if (overCapacity)
    plan.values[valueId].requiresMovement = true;
  return placementId;
}

static ValuePlacement *findLivePlacement(ExecutionPlan &plan, unsigned valueId,
                                         StringAttr bufferName, int64_t size,
                                         unsigned actionId) {
  assert(valueId < plan.values.size() && "invalid value id");
  PlanValue &value = plan.values[valueId];
  for (unsigned placementId : value.placementIds) {
    assert(placementId < plan.placements.size() && "invalid placement id");
    ValuePlacement &placement = plan.placements[placementId];
    if (placement.region.bufferName != bufferName ||
        placement.region.size != size)
      continue;
    if (placement.firstAction <= actionId && actionId <= placement.lastAction)
      return &placement;
  }
  return nullptr;
}

static void addLocalRegion(
    DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> &localRegions,
    PhysicalRegion &region) {
  localRegions[region.bufferName].push_back(region);
}

static LogicalResult setOperandRegion(PlanNode &node,
                                      PlanOperandAccess &operand,
                                      PhysicalRegion region,
                                      Operation *errorOp) {
  if (failed(bindBasisToOffset(node, operand, region.offset, errorOp)))
    return failure();
  operand.scratchRegion = region;
  return success();
}

static LogicalResult assignInputRegion(
    ExecutionPlan &plan, PlanNode &node, PlanOperandAccess &operand,
    DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> &localRegions) {
  if (isa<HBMBufferType>(operand.access.bufferType) ||
      !accessReads(operand.role))
    return success();

  Operation *errorOp = getErrorOp(*node.semanticNode);
  unsigned actionId = node.actionId;
  assert(actionId < plan.actions.size() && "node action should be built");
  auto size = getScratchAccessSize(node, operand, errorOp);
  if (failed(size))
    return failure();

  assert(operand.inputValueId &&
         "read operand should be wired before planning");
  PlanValue &input = plan.values[*operand.inputValueId];
  if (ValuePlacement *placement =
          findLivePlacement(plan, *operand.inputValueId,
                            operand.access.bufferName, *size, actionId)) {
    operand.placementId =
        static_cast<unsigned>(placement - plan.placements.data());
    operand.forwarded = true;
    operand.overCapacity = placement->overCapacity;
    return setOperandRegion(node, operand, placement->region, errorOp);
  }

  if (input.kind == PlanValueKind::Placeholder)
    operand.requiresInit = true;
  else {
    operand.requiresMovement = true;
    if (input.kind == PlanValueKind::Produced)
      input.requiresMovement = true;
  }

  bool overCapacity = false;
  auto region =
      allocateRegion(plan, localRegions, operand.access.bufferName, *size,
                     actionId, actionId, errorOp, overCapacity);
  if (failed(region))
    return failure();
  operand.overCapacity = overCapacity;
  if (overCapacity)
    input.requiresMovement = true;
  addLocalRegion(localRegions, *region);
  return setOperandRegion(node, operand, *region, errorOp);
}

static bool canAliasInputForOutput(ExecutionPlan &plan,
                                   PlanOperandAccess &operand, int64_t size,
                                   unsigned actionId,
                                   ValuePlacement *&placement) {
  if (!operand.inputValueId)
    return false;
  PlanValue &input = plan.values[*operand.inputValueId];
  placement = findLivePlacement(plan, *operand.inputValueId,
                                operand.access.bufferName, size, actionId);
  if (!placement)
    return false;
  return input.lifetime.isValid() && input.lifetime.lastAction <= actionId;
}

static LogicalResult assignOutputRegion(
    ExecutionPlan &plan, PlanNode &node, PlanOperandAccess &operand,
    DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> &localRegions) {
  if (isa<HBMBufferType>(operand.access.bufferType) ||
      !accessWrites(operand.role))
    return success();

  Operation *errorOp = getErrorOp(*node.semanticNode);
  unsigned numSources = node.instruction.getSources().size();
  if (operand.access.operandIdx < numSources)
    return errorOp->emitError()
           << "writeable source operands are not supported in Round4 planning";

  unsigned outputIdx = operand.access.operandIdx - numSources;
  assert(outputIdx < node.outputValueIds.size() &&
         "destination access should map to an output value");
  unsigned valueId = node.outputValueIds[outputIdx];
  assert(valueId < plan.values.size() && "invalid output value id");

  unsigned actionId = node.actionId;
  assert(actionId < plan.actions.size() && "node action should be built");
  auto size = getScratchAccessSize(node, operand, errorOp);
  if (failed(size))
    return failure();

  PlanValue &value = plan.values[valueId];
  unsigned firstAction =
      value.lifetime.isValid() ? value.lifetime.firstAction : actionId;
  unsigned lastAction =
      value.lifetime.isValid() ? value.lifetime.lastAction : actionId;

  ValuePlacement *inputPlacement = nullptr;
  if (canAliasInputForOutput(plan, operand, *size, actionId, inputPlacement)) {
    assert(inputPlacement && "expected input placement for alias");
    PhysicalRegion aliasRegion = inputPlacement->region;
    bool aliasOverCapacity = inputPlacement->overCapacity;
    unsigned placementId = createPlacement(
        plan, valueId, aliasRegion, firstAction, lastAction, aliasOverCapacity);
    operand.placementId = placementId;
    operand.forwarded = true;
    operand.overCapacity = aliasOverCapacity;
    return setOperandRegion(node, operand, aliasRegion, errorOp);
  }

  if (operand.inputValueId) {
    PlanValue &input = plan.values[*operand.inputValueId];
    if (input.kind == PlanValueKind::Placeholder)
      operand.requiresInit = true;
    else {
      operand.requiresMovement = true;
      if (input.kind == PlanValueKind::Produced)
        input.requiresMovement = true;
    }
  }

  bool overCapacity = false;
  auto region =
      allocateRegion(plan, localRegions, operand.access.bufferName, *size,
                     firstAction, lastAction, errorOp, overCapacity);
  if (failed(region))
    return failure();
  addLocalRegion(localRegions, *region);
  unsigned placementId = createPlacement(plan, valueId, *region, firstAction,
                                         lastAction, overCapacity);
  operand.placementId = placementId;
  operand.overCapacity = overCapacity;
  if (failed(setOperandRegion(node, operand, *region, errorOp)))
    return failure();
  return success();
}

static LogicalResult buildActions(ExecutionPlan &plan) {
  for (auto [nodeIdx, node] : llvm::enumerate(plan.nodes)) {
    PlanAction action;
    action.kind = PlanActionKind::Compute;
    action.nodeIdx = nodeIdx;
    node.actionId = plan.actions.size();
    plan.actions.push_back(std::move(action));

    for (auto [outputIdx, valueId] : llvm::enumerate(node.outputValueIds)) {
      assert(valueId < plan.values.size() && "invalid output value id");
      PlanValue &value = plan.values[valueId];
      for (auto [writebackIdx, writeback] : llvm::enumerate(value.writebacks)) {
        PlanAction writebackAction;
        writebackAction.kind = PlanActionKind::Writeback;
        writebackAction.nodeIdx = nodeIdx;
        writebackAction.outputIdx = outputIdx;
        writebackAction.valueId = valueId;
        writebackAction.writebackIdx = writebackIdx;
        writeback.actionId = plan.actions.size();
        plan.actions.push_back(std::move(writebackAction));
      }
    }
  }
  return success();
}

static void extendLifetime(ValueLifetime &lifetime, unsigned actionId) {
  assert(actionId != kNullId && "expected valid action id");
  if (!lifetime.isValid()) {
    lifetime.firstAction = actionId;
    lifetime.lastAction = actionId;
    return;
  }
  lifetime.firstAction = std::min(lifetime.firstAction, actionId);
  lifetime.lastAction = std::max(lifetime.lastAction, actionId);
}

static LogicalResult computeValueLifetimes(ExecutionPlan &plan) {
  for (auto [valueId, value] : llvm::enumerate(plan.values)) {
    if (value.kind == PlanValueKind::Produced) {
      assert(value.definingNodeIdx && "produced value should have a producer");
      unsigned nodeIdx = *value.definingNodeIdx;
      assert(nodeIdx < plan.nodes.size() && "invalid producer node");
      extendLifetime(value.lifetime, plan.nodes[nodeIdx].actionId);
    }

    for (PlanValueUse &use : value.uses) {
      assert(use.nodeIdx < plan.nodes.size() && "invalid value use node");
      extendLifetime(value.lifetime, plan.nodes[use.nodeIdx].actionId);
    }

    for (PlanValueWriteback &writeback : value.writebacks) {
      if (writeback.actionId == kNullId)
        return plan.func.emitError() << "writeback action was not built";
      extendLifetime(value.lifetime, writeback.actionId);
    }
  }
  return success();
}

static LogicalResult allocateScratchPlacements(ExecutionPlan &plan) {
  for (PlanNode &node : plan.nodes) {
    DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> localRegions;
    unsigned numSources = node.instruction.getSources().size();
    for (PlanOperandAccess &operand : node.operands) {
      if (accessWrites(operand.role) && operand.access.operandIdx >= numSources)
        continue;
      if (failed(assignInputRegion(plan, node, operand, localRegions)))
        return failure();
    }
    for (PlanOperandAccess &operand : node.operands)
      if (failed(assignOutputRegion(plan, node, operand, localRegions)))
        return failure();
  }
  return success();
}

static unsigned getWrittenValueId(PlanNode &node, PlanOperandAccess &operand) {
  unsigned numSources = node.instruction.getSources().size();
  if (operand.access.operandIdx < numSources)
    return kNullId;

  unsigned outputIdx = operand.access.operandIdx - numSources;
  assert(outputIdx < node.outputValueIds.size() &&
         "destination access should map to an output value");
  return node.outputValueIds[outputIdx];
}

static void populateActionAccesses(ExecutionPlan &plan) {
  for (PlanAction &action : plan.actions) {
    action.accesses.clear();
    if (action.kind == PlanActionKind::Compute) {
      assert(action.nodeIdx < plan.nodes.size() && "invalid action node id");
      PlanNode &node = plan.nodes[action.nodeIdx];
      for (PlanOperandAccess &operand : node.operands) {
        PlanActionAccess access;
        access.role = operand.role;
        access.nodeIdx = action.nodeIdx;
        access.accessOperandIdx = operand.access.operandIdx;
        if (accessWrites(operand.role))
          access.valueId = getWrittenValueId(node, operand);
        if (access.valueId == kNullId)
          access.valueId = operand.inputValueId.value_or(kNullId);
        access.placementId = operand.placementId;
        access.scratchRegion = operand.scratchRegion;
        access.hbmRegion = operand.hbmRegion;
        action.accesses.push_back(std::move(access));
      }
      continue;
    }

    assert(action.valueId < plan.values.size() && "invalid writeback value");
    PlanValue &value = plan.values[action.valueId];
    assert(action.writebackIdx < value.writebacks.size() &&
           "invalid writeback action index");
    PlanValueWriteback &writeback = value.writebacks[action.writebackIdx];
    for (unsigned placementId : value.placementIds) {
      ValuePlacement &placement = plan.placements[placementId];
      if (placement.firstAction <= writeback.actionId &&
          writeback.actionId <= placement.lastAction) {
        PlanActionAccess read;
        read.role = AccessRole::Read;
        read.nodeIdx = action.nodeIdx;
        read.valueId = action.valueId;
        read.placementId = placementId;
        read.scratchRegion = placement.region;
        action.accesses.push_back(std::move(read));
        break;
      }
    }
    if (action.accesses.empty())
      value.requiresMovement = true;

    PlanActionAccess write;
    write.role = AccessRole::Write;
    write.nodeIdx = action.nodeIdx;
    write.valueId = action.valueId;
    write.hbmRegion = writeback.hbmRegion;
    action.accesses.push_back(std::move(write));
  }
}

static LayoutSignature getRequiredLayoutSignature(LayoutChain &layout) {
  LayoutSignature signature;
  for (Operation *op : layout.layoutOps) {
    auto transpose = dyn_cast<linalg::TransposeOp>(op);
    if (!transpose)
      continue;
    signature.hasTranspose = true;
    auto permutation = transpose.getPermutation();
    signature.permutation.assign(permutation.begin(), permutation.end());
  }
  return signature;
}

static FailureOr<DefineOp>
lookupMovementInstruction(DataMovementCatalog &catalog, StringAttr src,
                          StringAttr dst, LayoutSignature required,
                          Operation *errorOp) {
  auto instruction = catalog.lookup(src, dst, required);
  if (instruction)
    return *instruction;

  InFlightDiagnostic diag = errorOp->emitError()
                            << "no data movement instruction for @"
                            << src.getValue() << " -> @" << dst.getValue();
  if (required.hasTranspose) {
    diag << " with transpose";
  }
  return failure();
}

static ValuePlacement *findPlacementAtAction(ExecutionPlan &plan,
                                             unsigned valueId,
                                             unsigned actionId) {
  assert(valueId < plan.values.size() && "invalid value id");
  for (unsigned placementId : plan.values[valueId].placementIds) {
    assert(placementId < plan.placements.size() && "invalid placement id");
    ValuePlacement &placement = plan.placements[placementId];
    if (placement.firstAction <= actionId && actionId <= placement.lastAction)
      return &placement;
  }
  return nullptr;
}

static PlanMoveEndpoint makeScratchEndpoint(unsigned valueId,
                                            PhysicalRegion region) {
  PlanMoveEndpoint endpoint;
  endpoint.valueId = valueId;
  endpoint.scratchRegion = region;
  return endpoint;
}

static PlanMoveEndpoint makeHBMInputEndpoint(ExecutionPlan &plan,
                                             unsigned valueId,
                                             FlatStridedRegion region,
                                             std::optional<StaticSlice> slice) {
  assert(valueId < plan.values.size() && "invalid HBM input value id");
  PlanValue &value = plan.values[valueId];
  assert(value.kind == PlanValueKind::HBMInput && "expected HBM input value");
  HBMAllocation *allocation = findInputHBMAllocation(plan, value.baseValue);
  assert(allocation && "expected HBM input allocation");

  PlanMoveEndpoint endpoint;
  endpoint.isHBM = true;
  endpoint.isResult = false;
  endpoint.boundaryIdx = allocation->boundaryIdx;
  endpoint.valueId = valueId;
  endpoint.hbmRegion = std::move(region);
  endpoint.hbmSlice = std::move(slice);
  return endpoint;
}

static PlanMoveEndpoint
makeHBMResultEndpoint(unsigned valueId, unsigned resultIdx,
                      FlatStridedRegion region,
                      std::optional<StaticSlice> slice) {
  PlanMoveEndpoint endpoint;
  endpoint.isHBM = true;
  endpoint.isResult = true;
  endpoint.boundaryIdx = resultIdx;
  endpoint.valueId = valueId;
  endpoint.hbmRegion = std::move(region);
  endpoint.hbmSlice = std::move(slice);
  return endpoint;
}

static int64_t getEndpointOffset(PlanMoveEndpoint &endpoint) {
  if (endpoint.isHBM) {
    assert(endpoint.hbmRegion && "expected HBM move endpoint region");
    return endpoint.hbmRegion->base;
  }
  assert(endpoint.scratchRegion && "expected scratch move endpoint region");
  return endpoint.scratchRegion->offset;
}

static int64_t getMoveSize(PlanMoveEndpoint &src, PlanMoveEndpoint &dst) {
  if (src.scratchRegion)
    return src.scratchRegion->size;
  if (dst.scratchRegion)
    return dst.scratchRegion->size;
  llvm_unreachable("move should touch at least one scratch region");
}

static LogicalResult appendMoveNode(ExecutionPlan &plan, DefineOp instruction,
                                    PlanMoveEndpoint src, PlanMoveEndpoint dst,
                                    LayoutSignature layout,
                                    unsigned anchorActionId,
                                    Operation *errorOp) {
  if (instruction.getAccessBlock().getNumArguments() != 3)
    return errorOp->emitError()
           << "Round5 movement instruction expects three addr params";

  PlanMoveNode move;
  move.instruction = instruction;
  move.src = std::move(src);
  move.dst = std::move(dst);
  move.layout = std::move(layout);
  move.anchorActionId = anchorActionId;
  move.paramBindings[0] = getEndpointOffset(move.src);
  move.paramBindings[1] = getEndpointOffset(move.dst);
  move.paramBindings[2] = getMoveSize(move.src, move.dst);

  PlanScheduleStep step;
  step.kind = PlanScheduleKind::Move;
  step.nodeIdx = plan.moveNodes.size();
  plan.schedule.push_back(step);
  plan.moveNodes.push_back(std::move(move));
  return success();
}

static LogicalResult appendInputMove(ExecutionPlan &plan, PlanNode &node,
                                     PlanOperandAccess &operand,
                                     DataMovementCatalog &catalog) {
  assert(operand.requiresMovement && "expected movement operand");
  assert(operand.scratchRegion && "movement target should be in scratch");
  assert(operand.inputValueId && "movement operand should have an input value");

  Operation *errorOp = getErrorOp(*node.semanticNode);
  unsigned actionId = node.actionId;
  PhysicalRegion dstRegion = *operand.scratchRegion;

  if (operand.hbmRegion) {
    LayoutSignature required = getRequiredLayoutSignature(operand.layout);
    auto instruction = lookupMovementInstruction(
        catalog, plan.hbmBufferName, dstRegion.bufferName, required, errorOp);
    if (failed(instruction))
      return failure();

    PlanMoveEndpoint src = makeHBMInputEndpoint(
        plan, *operand.inputValueId, *operand.hbmRegion, operand.layout.slice);
    PlanMoveEndpoint dst =
        makeScratchEndpoint(*operand.inputValueId, dstRegion);
    return appendMoveNode(plan, *instruction, std::move(src), std::move(dst),
                          std::move(required), actionId, errorOp);
  }

  ValuePlacement *placement =
      findPlacementAtAction(plan, *operand.inputValueId, actionId);
  if (!placement)
    return errorOp->emitError()
           << "no live scratch placement found for movement operand";
  if (placement->region.bufferName == dstRegion.bufferName)
    return errorOp->emitError()
           << "movement requested between identical scratch buffers";
  if (placement->region.size != dstRegion.size)
    return errorOp->emitError()
           << "scratch movement source and destination sizes differ";

  LayoutSignature required;
  auto instruction =
      lookupMovementInstruction(catalog, placement->region.bufferName,
                                dstRegion.bufferName, required, errorOp);
  if (failed(instruction))
    return failure();

  PlanMoveEndpoint src =
      makeScratchEndpoint(*operand.inputValueId, placement->region);
  PlanMoveEndpoint dst = makeScratchEndpoint(*operand.inputValueId, dstRegion);
  return appendMoveNode(plan, *instruction, std::move(src), std::move(dst),
                        std::move(required), actionId, errorOp);
}

static FailureOr<PhysicalRegion>
allocateMoveTempRegion(ExecutionPlan &plan, StringAttr bufferName, int64_t size,
                       unsigned actionId, Operation *errorOp) {
  DenseMap<StringAttr, SmallVector<PhysicalRegion, 4>> localRegions;
  bool overCapacity = false;
  auto region = allocateRegion(plan, localRegions, bufferName, size, actionId,
                               actionId, errorOp, overCapacity);
  if (failed(region))
    return failure();
  if (overCapacity)
    return errorOp->emitError()
           << "no scratch space for intermediate movement through @"
           << bufferName.getValue();
  return *region;
}

static LogicalResult
appendDirectWritebackMove(ExecutionPlan &plan, DefineOp instruction,
                          unsigned valueId, PhysicalRegion srcRegion,
                          PlanValueWriteback &writeback, unsigned actionId,
                          Operation *errorOp) {
  assert(writeback.hbmRegion && "writeback should have an HBM region");
  LayoutSignature required;
  PlanMoveEndpoint src = makeScratchEndpoint(valueId, srcRegion);
  PlanMoveEndpoint dst =
      makeHBMResultEndpoint(valueId, writeback.funcResultIdx,
                            *writeback.hbmRegion, writeback.layout.targetSlice);
  return appendMoveNode(plan, instruction, std::move(src), std::move(dst),
                        std::move(required), actionId, errorOp);
}

static LogicalResult appendWritebackMoves(ExecutionPlan &plan,
                                          PlanAction &action,
                                          DataMovementCatalog &catalog) {
  assert(action.kind == PlanActionKind::Writeback &&
         "expected writeback action");
  assert(action.valueId < plan.values.size() && "invalid writeback value");
  Operation *errorOp = getErrorOp(*plan.nodes[action.nodeIdx].semanticNode);

  PlanValue &value = plan.values[action.valueId];
  assert(action.writebackIdx < value.writebacks.size() &&
         "invalid writeback id");
  PlanValueWriteback &writeback = value.writebacks[action.writebackIdx];
  ValuePlacement *placement =
      findPlacementAtAction(plan, action.valueId, writeback.actionId);
  if (!placement)
    return errorOp->emitError()
           << "no live scratch placement found for writeback";

  LayoutSignature required;
  if (auto directStore = catalog.lookup(placement->region.bufferName,
                                        plan.hbmBufferName, required)) {
    return appendDirectWritebackMove(plan, *directStore, action.valueId,
                                     placement->region, writeback,
                                     writeback.actionId, errorOp);
  }

  for (ScratchResource &resource : plan.scratchResources) {
    if (resource.bufferName == placement->region.bufferName)
      continue;
    auto mov = catalog.lookup(placement->region.bufferName, resource.bufferName,
                              required);
    if (!mov)
      continue;
    auto store =
        catalog.lookup(resource.bufferName, plan.hbmBufferName, required);
    if (!store)
      continue;

    auto tempRegion = allocateMoveTempRegion(plan, resource.bufferName,
                                             placement->region.size,
                                             writeback.actionId, errorOp);
    if (failed(tempRegion))
      return failure();

    PlanMoveEndpoint movSrc =
        makeScratchEndpoint(action.valueId, placement->region);
    PlanMoveEndpoint movDst = makeScratchEndpoint(action.valueId, *tempRegion);
    if (failed(appendMoveNode(plan, *mov, std::move(movSrc), std::move(movDst),
                              LayoutSignature{}, writeback.actionId, errorOp)))
      return failure();

    return appendDirectWritebackMove(plan, *store, action.valueId, *tempRegion,
                                     writeback, writeback.actionId, errorOp);
  }

  return errorOp->emitError() << "no data movement path from @"
                              << placement->region.bufferName.getValue()
                              << " to HBM @" << plan.hbmBufferName.getValue();
}

static void appendComputeScheduleStep(ExecutionPlan &plan, unsigned nodeIdx) {
  PlanScheduleStep step;
  step.kind = PlanScheduleKind::Compute;
  step.nodeIdx = nodeIdx;
  plan.schedule.push_back(step);
}

static LogicalResult buildMovementPlan(ExecutionPlan &plan) {
  ModuleOp module = plan.func->getParentOfType<ModuleOp>();
  assert(module && "planned function should belong to a module");
  DataMovementCatalog catalog = buildDataMovementCatalog(module);

  for (PlanAction &action : plan.actions) {
    if (action.kind == PlanActionKind::Compute) {
      assert(action.nodeIdx < plan.nodes.size() && "invalid compute node");
      PlanNode &node = plan.nodes[action.nodeIdx];
      for (PlanOperandAccess &operand : node.operands)
        if (operand.requiresMovement &&
            failed(appendInputMove(plan, node, operand, catalog)))
          return failure();
      appendComputeScheduleStep(plan, action.nodeIdx);
      continue;
    }

    if (failed(appendWritebackMoves(plan, action, catalog)))
      return failure();
  }

  return success();
}

static void printAddrParams(llvm::raw_ostream &os, DefineOp instruction,
                            DenseMap<unsigned, int64_t> &paramBindings) {
  os << " addr(";
  for (unsigned idx = 0; idx < instruction.getAccessBlock().getNumArguments();
       ++idx) {
    if (idx)
      os << ", ";
    auto it = paramBindings.find(idx);
    assert(it != paramBindings.end() && "addr param should be bound");
    os << it->second;
  }
  os << ")";
}

static void printMoveEndpoint(llvm::raw_ostream &os,
                              PlanMoveEndpoint &endpoint) {
  if (!endpoint.isHBM) {
    assert(endpoint.scratchRegion && "expected scratch endpoint");
    os << "@" << endpoint.scratchRegion->bufferName.getValue() << "["
       << endpoint.scratchRegion->offset << "]";
    return;
  }

  os << "hbm " << (endpoint.isResult ? "result" : "arg")
     << endpoint.boundaryIdx;
  if (endpoint.hbmRegion)
    os << "[base=" << endpoint.hbmRegion->base << "]";
  if (endpoint.hbmSlice) {
    os << " slice=";
    printIntArray(os, endpoint.hbmSlice->offsets);
  }
}

static void printMoveScheduleStep(llvm::raw_ostream &os, unsigned stepIdx,
                                  PlanMoveNode &move) {
  os << "  " << getScheduleKindName(PlanScheduleKind::Move) << stepIdx << " @"
     << move.instruction.getSymName();
  if (move.layout.hasTranspose)
    os << " transpose";
  printAddrParams(os, move.instruction, move.paramBindings);
  os << " ";
  printMoveEndpoint(os, move.src);
  os << " -> ";
  printMoveEndpoint(os, move.dst);
  os << " size=" << move.paramBindings.lookup(2);
  os << "\n";
}

static void printOperandLocation(llvm::raw_ostream &os,
                                 PlanOperandAccess &operand) {
  if (operand.scratchRegion) {
    os << "@" << operand.scratchRegion->bufferName.getValue() << "["
       << operand.scratchRegion->offset << "]";
    return;
  }
  if (operand.hbmRegion) {
    os << "hbm[base=" << operand.hbmRegion->base << "]";
    return;
  }
  os << "<unplaced>";
}

static void printComputeScheduleStep(llvm::raw_ostream &os, unsigned stepIdx,
                                     PlanNode &node) {
  os << "  " << getScheduleKindName(PlanScheduleKind::Compute) << stepIdx
     << " @" << node.instruction.getSymName();
  printAddrParams(os, node.instruction, node.paramBindings);
  os << " ";

  unsigned numSources = node.instruction.getSources().size();
  bool first = true;
  for (PlanOperandAccess &operand : node.operands) {
    if (operand.access.operandIdx >= numSources)
      continue;
    if (!first)
      os << ", ";
    printOperandLocation(os, operand);
    first = false;
  }

  os << " -> ";
  first = true;
  for (PlanOperandAccess &operand : node.operands) {
    if (operand.access.operandIdx < numSources)
      continue;
    if (!first)
      os << ", ";
    printOperandLocation(os, operand);
    first = false;
  }

  bool hasOverCapacity = llvm::any_of(
      node.operands, [](PlanOperandAccess &op) { return op.overCapacity; });
  if (hasOverCapacity)
    os << " over-capacity";
  os << "\n";
}

static void printIssueLocation(llvm::raw_ostream &os,
                               PlanOperandAccess &operand) {
  if (operand.scratchRegion) {
    printPhysicalRegion(os, *operand.scratchRegion);
    return;
  }
  if (operand.hbmRegion) {
    os << "hbm[base=" << operand.hbmRegion->base << "]";
    return;
  }
  os << "<unplaced>";
}

static void printIssues(llvm::raw_ostream &os, ExecutionPlan &plan) {
  os << "issues:\n";
  bool hasIssue = false;

  for (auto [placementId, placement] : llvm::enumerate(plan.placements)) {
    if (!placement.overCapacity)
      continue;
    os << "  over-capacity p" << placementId << " v" << placement.valueId
       << " ";
    printPhysicalRegion(os, placement.region);
    os << " live=[" << placement.firstAction << ", " << placement.lastAction
       << "]\n";
    hasIssue = true;
  }

  for (auto [nodeIdx, node] : llvm::enumerate(plan.nodes)) {
    for (PlanOperandAccess &operand : node.operands) {
      if (!operand.requiresInit)
        continue;
      os << "  needs-init node" << nodeIdx << " @"
         << node.instruction.getSymName()
         << " access=" << operand.access.operandIdx << " ";
      printIssueLocation(os, operand);
      os << "\n";
      hasIssue = true;
    }
  }

  if (!hasIssue)
    os << "  none\n";
}

void ExecutionPlan::dump(llvm::raw_ostream &os) {
  os << "=== Execution Plan @" << func.getSymName() << " ===\n";

  os << "hbm @" << (hbmBufferName ? hbmBufferName.getValue() : "<none>")
     << " capacity=" << hbmCapacity << " used=" << hbmUsed << "\n";
  for (HBMAllocation &allocation : hbmInputs) {
    os << "  arg" << allocation.boundaryIdx << " base=" << allocation.base
       << " shape=";
    printIntArray(os, allocation.shape);
    os << "\n";
  }
  for (HBMAllocation &allocation : hbmResults) {
    os << "  result" << allocation.boundaryIdx << " base=" << allocation.base
       << " shape=";
    printIntArray(os, allocation.shape);
    os << "\n";
  }

  os << "scratch resources:\n";
  for (ScratchResource &resource : scratchResources) {
    os << "  @" << resource.bufferName.getValue()
       << " capacity=" << resource.capacity << "\n";
  }

  printIssues(os, *this);

  os << "schedule:\n";
  for (auto [stepIdx, step] : llvm::enumerate(schedule)) {
    if (step.kind == PlanScheduleKind::Move) {
      assert(step.nodeIdx < moveNodes.size() && "invalid move node id");
      printMoveScheduleStep(os, stepIdx, moveNodes[step.nodeIdx]);
      continue;
    }
    assert(step.nodeIdx < nodes.size() && "invalid compute node id");
    printComputeScheduleStep(os, stepIdx, nodes[step.nodeIdx]);
  }
}

FailureOr<ExecutionPlan>
act::buildExecutionPlan(SemanticGraph &graph, GraphParamSolution &solutions) {
  assert(solutions.size() == graph.nodes.size() &&
         "param solution count should match semantic graph node count");

  ExecutionPlan plan;
  plan.func = graph.func;
  if (failed(initializeHBMMapping(plan)))
    return failure();
  if (failed(initializeScratchResources(plan)))
    return failure();
  plan.nodes.reserve(graph.nodes.size());

  for (auto [nodeIdx, solution] : llvm::enumerate(solutions)) {
    assert(&solution.node == &graph.nodes[nodeIdx] &&
           "param solution should reference matching semantic node");
    if (!solution.isValid)
      return getErrorOp(solution.node)->emitError()
             << "parameter solving failed for execution plan node";

    PlanNode node;
    node.semanticNode = &solution.node;
    node.instruction = solution.model.defineOp;
    node.paramBindings = solution.solvedParams;
    node.outputValueIds.assign(node.instruction.getDestinations().size(),
                               kNullId);
    for (SymbolicAccess &access : solution.model.accesses) {
      PlanOperandAccess operand;
      operand.access = access;
      operand.role = access.role;
      node.operands.push_back(std::move(operand));
    }
    plan.nodes.push_back(std::move(node));

    PlanNode &planNode = plan.nodes.back();
    for (unsigned outputIdx = 0; outputIdx < planNode.outputValueIds.size();
         ++outputIdx) {
      SemanticOutputBinding *binding =
          findOutputBinding(solution.node, outputIdx);
      if (!binding)
        return getErrorOp(solution.node)->emitError()
               << "execution plan output " << outputIdx
               << " has no boundary binding";
      auto valueId = createProducedValue(plan, *binding, nodeIdx,
                                         getErrorOp(solution.node));
      if (failed(valueId))
        return failure();
      planNode.outputValueIds[outputIdx] = *valueId;
    }
  }

  for (auto [nodeIdx, graphNode] : llvm::enumerate(graph.nodes)) {
    for (SemanticInputBinding &binding : graphNode.inputBindings)
      if (failed(addPlanUse(plan, graph, nodeIdx, binding)))
        return failure();
    for (SemanticOutputBinding &binding : graphNode.outputBindings)
      if (failed(addPlanWriteback(plan, nodeIdx, binding)))
        return failure();
    if (failed(validatePlanNodeInputs(plan.nodes[nodeIdx])))
      return failure();
  }

  if (failed(annotateHBMRegions(plan)))
    return failure();
  if (failed(buildActions(plan)))
    return failure();
  if (failed(computeValueLifetimes(plan)))
    return failure();
  if (failed(allocateScratchPlacements(plan)))
    return failure();
  populateActionAccesses(plan);
  if (failed(buildMovementPlan(plan)))
    return failure();

  LLVM_DEBUG(plan.dump(dbgs()));
  return std::move(plan);
}
