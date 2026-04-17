#include "act/Support/Planning.h"

#include "act/Support/SymbolicExpr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

#include <limits>

#define DEBUG_TYPE "planning"

using namespace mlir;
using namespace mlir::act;

static void printSliceSpec(raw_ostream &os, const StaticSliceSpec &sliceSpec) {
  os << " offsets=[";
  for (unsigned i = 0; i < sliceSpec.offsets.size(); ++i) {
    if (i)
      os << ",";
    os << sliceSpec.offsets[i];
  }
  os << "] sizes=[";
  for (unsigned i = 0; i < sliceSpec.sizes.size(); ++i) {
    if (i)
      os << ",";
    os << sliceSpec.sizes[i];
  }
  os << "] strides=[";
  for (unsigned i = 0; i < sliceSpec.strides.size(); ++i) {
    if (i)
      os << ",";
    os << sliceSpec.strides[i];
  }
  os << "]";
}

static void printLogicalTransform(raw_ostream &os,
                                  const LogicalTransform &transform) {
  switch (transform.kind) {
  case LogicalTransformKind::Transpose:
    os << "/transpose[";
    for (unsigned i = 0; i < transform.permutation.size(); ++i) {
      if (i)
        os << ",";
      os << transform.permutation[i];
    }
    os << "]";
    return;
  case LogicalTransformKind::ExtractSlice:
    os << "/extract_slice";
    assert(transform.sliceSpec && "extract_slice requires slice spec");
    printSliceSpec(os, *transform.sliceSpec);
    return;
  case LogicalTransformKind::InsertSlice:
    os << "/insert_slice";
    assert(transform.sliceSpec && "insert_slice requires slice spec");
    printSliceSpec(os, *transform.sliceSpec);
    return;
  }
  llvm_unreachable("unknown logical transform");
}

static void printLogicalTransformChain(raw_ostream &os,
                                       ArrayRef<LogicalTransform> transforms) {
  for (auto &transform : transforms)
    printLogicalTransform(os, transform);
}

static std::optional<StaticSliceSpec>
getStaticSliceSpec(tensor::ExtractSliceOp sliceOp) {
  StaticSliceSpec spec;
  auto staticOffsets = sliceOp.getStaticOffsets();
  auto staticSizes = sliceOp.getStaticSizes();
  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::any_of(staticOffsets,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticSizes,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticStrides,
                   [](int64_t v) { return v == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  spec.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  spec.sizes.assign(staticSizes.begin(), staticSizes.end());
  spec.strides.assign(staticStrides.begin(), staticStrides.end());
  return spec;
}

static std::optional<StaticSliceSpec>
getStaticSliceSpec(tensor::InsertSliceOp sliceOp) {
  StaticSliceSpec spec;
  auto staticOffsets = sliceOp.getStaticOffsets();
  auto staticSizes = sliceOp.getStaticSizes();
  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::any_of(staticOffsets,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticSizes,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticStrides,
                   [](int64_t v) { return v == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  spec.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  spec.sizes.assign(staticSizes.begin(), staticSizes.end());
  spec.strides.assign(staticStrides.begin(), staticStrides.end());
  return spec;
}

/// Create a planner logical value for a selected node output at the compute
/// boundary. Produced values are always unique, even if they share an IR Value
/// with an external materialized form.
static unsigned createProducedLogicalValue(LogicalPlan &plan, Value value,
                                           unsigned definingNodeIdx) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  assert(tensorType && "logical plan values must be ranked tensors");

  unsigned valueId = plan.values.size();
  plan.values.push_back(
      {LogicalPlanValueKind::Produced, value, tensorType, definingNodeIdx, {}});
  return valueId;
}

static Operation *getPlanningErrorOp(const SemanticsGraphNode &node) {
  if (node.domainSourceOp)
    return node.domainSourceOp;
  if (!node.sourceOps.empty())
    return node.sourceOps.front();
  return const_cast<DefineOp &>(node.instruction).getOperation();
}

/// Intern external logical values by their materialized SSA value.
static unsigned getOrCreateExternalLogicalValue(LogicalPlan &plan, Value value,
                                                LogicalPlanValueKind kind) {
  auto it = plan.externalValueIds.find(value);
  if (it != plan.externalValueIds.end()) {
    assert(plan.values[it->second].kind == kind &&
           "external logical value reused with inconsistent kind");
    return it->second;
  }

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  assert(tensorType && "logical plan values must be ranked tensors");

  unsigned valueId = plan.values.size();
  plan.externalValueIds[value] = valueId;
  plan.values.push_back({kind, value, tensorType, std::nullopt, {}});
  return valueId;
}

static StringRef getLogicalValueKindName(LogicalPlanValueKind kind) {
  switch (kind) {
  case LogicalPlanValueKind::FunctionInput:
    return "function-input";
  case LogicalPlanValueKind::Produced:
    return "produced";
  case LogicalPlanValueKind::MaterializedOutput:
    return "materialized-output";
  }
  llvm_unreachable("unknown logical plan value kind");
}

static bool isIdentityInstruction(DefineOp defineOp) {
  Block &compBlock = defineOp.getSemanticsBlock();
  unsigned numOps = 0;
  for (auto &op : compBlock.without_terminator())
    ++numOps;
  if (numOps != 0)
    return false;

  auto *yieldOp = compBlock.getTerminator();
  return llvm::all_of(yieldOp->getOperands(), [](Value operand) {
    return isa<BlockArgument>(operand);
  });
  return true;
}

static LayoutSignature extractLayoutSignature(DefineOp defineOp) {
  LayoutSignature sig;
  Block &addrBlock = defineOp.getAccessBlock();
  auto *yieldOp = addrBlock.getTerminator();
  Value token = yieldOp->getOperand(0);
  Operation *op = token.getDefiningOp();
  while (op && !isa<StridedOp>(op)) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      sig.hasTranspose = true;
      auto perm = transpose.getPermutation();
      sig.permutation.assign(perm.begin(), perm.end());
      op = transpose.getSource().getDefiningOp();
    } else if (auto expand = dyn_cast<ExpandShapeOp>(op)) {
      op = expand.getSource().getDefiningOp();
    } else if (auto collapse = dyn_cast<CollapseShapeOp>(op)) {
      op = collapse.getSource().getDefiningOp();
    } else {
      break;
    }
  }
  return sig;
}

static DataMovementCatalog buildDataMovementCatalog(ModuleOp module) {
  DataMovementCatalog catalog;
  module.walk([&](DefineOp defineOp) {
    if (!isIdentityInstruction(defineOp))
      return;
    auto sources = defineOp.getSources().getAsRange<FlatSymbolRefAttr>();
    auto dests = defineOp.getDestinations().getAsRange<FlatSymbolRefAttr>();
    StringAttr srcBuf = (*sources.begin()).getAttr();
    StringAttr dstBuf = (*dests.begin()).getAttr();
    auto key = std::make_pair(srcBuf, dstBuf);
    catalog.entries[key].push_back(
        {extractLayoutSignature(defineOp), defineOp});
  });
  return catalog;
}

static int64_t getBufferElementCapacity(DeclareBufferOp bufOp) {
  auto bufType = bufOp.getBufferType();
  if (auto hbmTy = dyn_cast<HBMBufferType>(bufType)) {
    int64_t cap = 1;
    for (int64_t d : hbmTy.getShape())
      cap *= d;
    return bufOp.getSize() * cap;
  }
  if (isa<ScalarBufferType>(bufType))
    return bufOp.getSize();
  return bufOp.getSize();
}

static FailureOr<DeclareBufferOp> identifyHBMBuffer(ModuleOp module) {
  DeclareBufferOp hbmBuf = nullptr;
  module.walk([&](DeclareBufferOp bufOp) {
    if (isa<HBMBufferType>(bufOp.getBufferType()))
      hbmBuf = bufOp;
  });
  if (hbmBuf)
    return hbmBuf;
  return module.emitError() << "no HBM buffer (type !act.hbm<...>) found";
}

static StridedOp findStridedOp(Value accessToken) {
  Operation *op = accessToken.getDefiningOp();
  while (op && !isa<StridedOp>(op)) {
    if (auto expand = dyn_cast<ExpandShapeOp>(op))
      op = expand.getSource().getDefiningOp();
    else if (auto collapse = dyn_cast<CollapseShapeOp>(op))
      op = collapse.getSource().getDefiningOp();
    else if (auto transpose = dyn_cast<TransposeOp>(op))
      op = transpose.getSource().getDefiningOp();
    else
      break;
  }
  return dyn_cast_or_null<StridedOp>(op);
}

static FailureOr<int64_t>
evaluateOperandSlotCount(DefineOp defineOp, unsigned operandIdx,
                         const DenseMap<unsigned, int64_t> &solvedParams) {
  Block &addrBlock = defineOp.getAccessBlock();
  auto *yieldOp = addrBlock.getTerminator();
  if (operandIdx >= yieldOp->getNumOperands())
    return defineOp.emitError()
           << "operand index " << operandIdx << " out of range";

  auto strided = findStridedOp(yieldOp->getOperand(operandIdx));
  if (!strided)
    return defineOp.emitError()
           << "could not find StridedOp for operand " << operandIdx;

  auto mixedCounts = getMixedValues(strided.getStaticCounts(),
                                    strided.getCounts(), strided.getContext());

  unsigned numParams = addrBlock.getNumArguments();
  SmallVector<int64_t> paramValues(numParams, 0);
  for (auto &[idx, val] : solvedParams) {
    if (idx < numParams)
      paramValues[idx] = val;
  }

  int64_t totalSlots = 1;
  for (auto &count : mixedCounts) {
    auto expr = buildSymExpr(count);
    if (failed(expr))
      return strided.emitError() << "failed to build symbolic expr for count";
    totalSlots *= expr->evaluate(paramValues);
  }

  return totalSlots;
}

static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

static bool isSupportedStaticContiguousSlice(ArrayRef<int64_t> tensorShape,
                                             const StaticSliceSpec &sliceSpec) {
  if (tensorShape.size() != sliceSpec.offsets.size() ||
      tensorShape.size() != sliceSpec.sizes.size() ||
      tensorShape.size() != sliceSpec.strides.size())
    return false;

  int partialDim = -1;
  for (unsigned i = 0; i < tensorShape.size(); ++i) {
    if (sliceSpec.strides[i] != 1)
      return false;
    if (sliceSpec.offsets[i] < 0 || sliceSpec.sizes[i] <= 0)
      return false;
    if (sliceSpec.offsets[i] + sliceSpec.sizes[i] > tensorShape[i])
      return false;
    if (sliceSpec.sizes[i] != tensorShape[i]) {
      if (partialDim != -1)
        return false;
      partialDim = static_cast<int>(i);
    }
  }

  if (partialDim < 0)
    return true;

  for (int i = 0; i < partialDim; ++i) {
    if (sliceSpec.sizes[i] != 1)
      return false;
  }
  for (unsigned i = partialDim + 1; i < tensorShape.size(); ++i) {
    if (sliceSpec.sizes[i] != tensorShape[i])
      return false;
  }
  return true;
}

static FailureOr<TensorLayout>
applyLogicalTransforms(const TensorLayout &baseLayout,
                       ArrayRef<LogicalTransform> transforms) {
  TensorLayout current = baseLayout;
  for (auto &transform : transforms) {
    switch (transform.kind) {
    case LogicalTransformKind::Transpose: {
      if (current.shape.size() != transform.permutation.size())
        return failure();
      SmallVector<int64_t> newShape;
      SmallVector<int64_t> newStrides;
      newShape.reserve(transform.permutation.size());
      newStrides.reserve(transform.permutation.size());
      for (int64_t idx : transform.permutation) {
        if (idx < 0 || static_cast<size_t>(idx) >= current.shape.size())
          return failure();
        newShape.push_back(current.shape[idx]);
        newStrides.push_back(current.strides[idx]);
      }
      current.shape = std::move(newShape);
      current.strides = std::move(newStrides);
      break;
    }
    case LogicalTransformKind::ExtractSlice:
    case LogicalTransformKind::InsertSlice: {
      if (!transform.sliceSpec)
        return failure();
      if (!isSupportedStaticContiguousSlice(current.shape,
                                            *transform.sliceSpec))
        return failure();
      int64_t delta = 0;
      for (unsigned i = 0; i < current.shape.size(); ++i)
        delta += transform.sliceSpec->offsets[i] * current.strides[i];
      current.baseOffset += delta;
      current.shape.assign(transform.sliceSpec->sizes.begin(),
                           transform.sliceSpec->sizes.end());
      break;
    }
    }
  }
  return current;
}

static bool isSingleBufferInstruction(DefineOp defineOp) {
  StringAttr first;
  for (auto src : defineOp.getSources().getAsRange<FlatSymbolRefAttr>()) {
    if (!first)
      first = src.getAttr();
    else if (first != src.getAttr())
      return false;
  }
  for (auto dst : defineOp.getDestinations().getAsRange<FlatSymbolRefAttr>()) {
    if (!first)
      first = dst.getAttr();
    else if (first != dst.getAttr())
      return false;
  }
  return true;
}

static StringAttr getOperandBuffer(DefineOp defineOp, unsigned operandIdx) {
  unsigned numSrc = defineOp.getSources().size();
  if (operandIdx < numSrc)
    return cast<FlatSymbolRefAttr>(defineOp.getSources()[operandIdx]).getAttr();
  return cast<FlatSymbolRefAttr>(
             defineOp.getDestinations()[operandIdx - numSrc])
      .getAttr();
}

namespace {
struct LiveRange {
  unsigned valueId;
  int64_t numElements;
  int liveStart;
  int liveEnd;
};
} // namespace

static SmallVector<LiveRange>
computeLiveRanges(func::FuncOp funcOp, const LogicalPlan &plan,
                  const DenseMap<unsigned, unsigned> &layoutAliases) {
  assert(funcOp.getFunctionBody().getBlocks().size() == 1 &&
         "expected flat single-block function");
  DenseMap<unsigned, LiveRange> ranges;

  for (auto [valueId, value] : llvm::enumerate(plan.values)) {
    auto aliasIt = layoutAliases.find(static_cast<unsigned>(valueId));
    if (aliasIt != layoutAliases.end())
      continue;
    int64_t numElts = 1;
    for (int64_t s : value.type.getShape())
      numElts *= s;
    int liveStart =
        value.definingNodeIdx ? static_cast<int>(*value.definingNodeIdx) : -1;
    ranges[valueId] = {static_cast<unsigned>(valueId), numElts, liveStart,
                       liveStart};
    for (auto &use : value.uses) {
      ranges[valueId].liveEnd = std::max(ranges[valueId].liveEnd,
                                         static_cast<int>(use.consumerNodeIdx));
    }
  }

  auto &bodyBlock = funcOp.getFunctionBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(bodyBlock.getTerminator());
  assert(returnOp && "expected func.return terminator");
  for (Value val : returnOp.getOperands()) {
    auto it = plan.externalValueIds.find(val);
    if (it == plan.externalValueIds.end())
      continue;
    unsigned valueId = it->second;
    if (auto aliasIt = layoutAliases.find(valueId);
        aliasIt != layoutAliases.end())
      valueId = aliasIt->second;
    auto rangeIt = ranges.find(valueId);
    if (rangeIt == ranges.end())
      continue;
    rangeIt->second.liveEnd =
        std::max(rangeIt->second.liveEnd, static_cast<int>(plan.nodes.size()));
  }

  SmallVector<LiveRange> result;
  for (auto &[_, lr] : ranges)
    result.push_back(lr);
  llvm::sort(result, [](const LiveRange &a, const LiveRange &b) {
    return a.liveStart < b.liveStart;
  });
  return result;
}

/// Reuse HBM storage when a produced value and its materialized writeback
/// target are physically identical.
static DenseMap<unsigned, unsigned>
computeLayoutAliases(const LogicalPlan &plan) {
  DenseMap<unsigned, unsigned> aliases;
  for (auto &node : plan.nodes) {
    for (auto &output : node.outputs) {
      if (!output.writebackTargetValueId ||
          !isIdentityTransformChain(output.writebackTransforms))
        continue;
      unsigned producedValueId = output.valueId;
      unsigned materializedValueId = *output.writebackTargetValueId;
      if (producedValueId >= plan.values.size() ||
          materializedValueId >= plan.values.size())
        continue;
      if (plan.values[producedValueId].sourceValue !=
          plan.values[materializedValueId].sourceValue)
        continue;
      aliases[materializedValueId] = producedValueId;
    }
  }
  return aliases;
}

static DenseMap<unsigned, int64_t> greedyAllocate(ArrayRef<LiveRange> ranges,
                                                  int64_t &totalAllocated) {
  struct AllocRegion {
    int64_t offset;
    int64_t size;
    int liveEnd;
  };

  SmallVector<AllocRegion> allocated;
  DenseMap<unsigned, int64_t> offsets;
  int64_t nextFreshOffset = 0;

  for (auto &lr : ranges) {
    int bestIdx = -1;
    int64_t bestWaste = std::numeric_limits<int64_t>::max();
    for (unsigned i = 0; i < allocated.size(); ++i) {
      if (allocated[i].liveEnd < lr.liveStart &&
          allocated[i].size >= lr.numElements) {
        int64_t waste = allocated[i].size - lr.numElements;
        if (waste < bestWaste) {
          bestWaste = waste;
          bestIdx = static_cast<int>(i);
        }
      }
    }

    if (bestIdx >= 0) {
      offsets[lr.valueId] = allocated[bestIdx].offset;
      allocated[bestIdx].liveEnd = lr.liveEnd;
    } else {
      offsets[lr.valueId] = nextFreshOffset;
      allocated.push_back({nextFreshOffset, lr.numElements, lr.liveEnd});
      nextFreshOffset += lr.numElements;
    }
  }

  totalAllocated = nextFreshOffset;
  return offsets;
}

static SmallVector<ForwardingEdge>
detectForwardingOpportunities(const LogicalPlan &plan, StringAttr hbmName) {
  SmallVector<ForwardingEdge> edges;
  for (unsigned i = 0; i + 1 < plan.nodes.size(); ++i) {
    DefineOp producerDef = plan.nodes[i].instruction;
    DefineOp consumerDef = plan.nodes[i + 1].instruction;

    unsigned pNumSrc = producerDef.getSources().size();
    unsigned pNumDst = producerDef.getDestinations().size();
    unsigned cNumSrc = consumerDef.getSources().size();

    for (unsigned di = 0; di < pNumDst; ++di) {
      unsigned producerValueId = plan.nodes[i].outputs[di].valueId;
      StringAttr pDstBuf = getOperandBuffer(producerDef, pNumSrc + di);
      if (pDstBuf == hbmName)
        continue;

      for (unsigned si = 0; si < cNumSrc; ++si) {
        if (plan.nodes[i + 1].inputs[si].valueId != producerValueId)
          continue;
        if (!isIdentityTransformChain(
                plan.nodes[i + 1].inputs[si].requiredTransforms))
          continue;
        StringAttr cSrcBuf = getOperandBuffer(consumerDef, si);
        if (pDstBuf == cSrcBuf)
          edges.push_back({i, di, i + 1, si, pDstBuf});
      }
    }
  }
  return edges;
}

static int64_t getHBMBaseOffset(const ResourcePlan &plan, unsigned valueId) {
  auto it = plan.layouts.find(valueId);
  assert(it != plan.layouts.end() && "missing HBM layout for logical value");
  return it->second.baseOffset;
}

static FailureOr<int64_t>
getTransformedHBMBaseOffset(const ResourcePlan &plan, unsigned valueId,
                            ArrayRef<LogicalTransform> transforms) {
  auto it = plan.layouts.find(valueId);
  if (it == plan.layouts.end())
    return failure();
  auto transformedLayout = applyLogicalTransforms(it->second, transforms);
  if (failed(transformedLayout))
    return failure();
  return transformedLayout->baseOffset;
}

static int64_t findIntermediateSlot(const ResourcePlan &layout,
                                    unsigned nodeIdx, unsigned numSrc,
                                    StringAttr bufferName) {
  auto &residences = layout.operandResidences[nodeIdx];
  for (unsigned srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
    if (residences[srcIdx].bufferName == bufferName)
      return residences[srcIdx].offset;
  }
  return 0;
}

static FailureOr<InputMovementPlan>
buildInputMovementPlan(func::FuncOp funcOp, const LogicalPlan &plan,
                       const ResourcePlan &layout, unsigned nodeIdx,
                       unsigned srcOperandIdx) {
  auto &node = plan.nodes[nodeIdx];
  DefineOp defineOp = node.instruction;
  StringAttr srcBuf = getOperandBuffer(defineOp, srcOperandIdx);
  unsigned hbmValueId = node.inputs[srcOperandIdx].valueId;
  auto &residence = layout.operandResidences[nodeIdx][srcOperandIdx];

  LayoutSignature requiredLayout;
  for (auto &transform : node.inputs[srcOperandIdx].requiredTransforms) {
    if (!transform.isTranspose())
      continue;
    requiredLayout.hasTranspose = true;
    requiredLayout.permutation = transform.permutation;
  }

  auto loadInstr =
      layout.dmCatalog.lookup(layout.bufferName, srcBuf, requiredLayout);
  if (!loadInstr) {
    return funcOp.emitError()
           << "no data movement instruction for ("
           << layout.bufferName.getValue() << " -> " << srcBuf.getValue() << ")"
           << (requiredLayout.hasTranspose ? " with transpose" : "");
  }

  auto hbmBaseOffset = getTransformedHBMBaseOffset(
      layout, hbmValueId, node.inputs[srcOperandIdx].requiredTransforms);
  if (failed(hbmBaseOffset))
    return funcOp.emitError()
           << "failed to compute HBM base offset for input movement";

  InputMovementPlan movement{nodeIdx,
                             srcOperandIdx,
                             hbmValueId,
                             node.inputs[srcOperandIdx].requiredTransforms,
                             {}};
  movement.steps.push_back({*loadInstr, layout.bufferName, *hbmBaseOffset,
                            srcBuf, residence.offset, residence.size});
  return movement;
}

static FailureOr<AccumulatorInitPlan>
buildAccumulatorInitPlan(func::FuncOp funcOp, const LogicalPlan &plan,
                         const ResourcePlan &layout, unsigned nodeIdx,
                         unsigned dstOperandIdx) {
  auto &node = plan.nodes[nodeIdx];
  DefineOp defineOp = node.instruction;
  unsigned numSrc = defineOp.getSources().size();
  unsigned operandIdx = numSrc + dstOperandIdx;
  StringAttr dstBuf = getOperandBuffer(defineOp, operandIdx);
  unsigned hbmValueId =
      node.outputs[dstOperandIdx].writebackTargetValueId
          ? *node.outputs[dstOperandIdx].writebackTargetValueId
          : node.outputs[dstOperandIdx].valueId;
  auto &residence = layout.operandResidences[nodeIdx][operandIdx];

  auto hbmBaseOffset = getTransformedHBMBaseOffset(
      layout, hbmValueId, node.outputs[dstOperandIdx].writebackTransforms);
  if (failed(hbmBaseOffset))
    return funcOp.emitError()
           << "failed to compute HBM base offset for accumulator init";

  AccumulatorInitPlan init{nodeIdx,
                           dstOperandIdx,
                           hbmValueId,
                           node.outputs[dstOperandIdx].writebackTransforms,
                           {}};
  auto directLoad = layout.dmCatalog.lookup(layout.bufferName, dstBuf);
  if (directLoad) {
    init.steps.push_back({*directLoad, layout.bufferName, *hbmBaseOffset,
                          dstBuf, residence.offset, residence.size});
    return init;
  }

  for (auto &[key, defs] : layout.dmCatalog.entries) {
    if (key.first != layout.bufferName || key.second == dstBuf)
      continue;
    StringAttr intermBuf = key.second;
    auto loadInstr = layout.dmCatalog.lookup(layout.bufferName, intermBuf);
    if (!loadInstr)
      continue;
    auto movInstr = layout.dmCatalog.lookup(intermBuf, dstBuf);
    if (!movInstr)
      continue;

    int64_t intermSlot =
        findIntermediateSlot(layout, nodeIdx, numSrc, intermBuf);
    init.steps.push_back({*loadInstr, layout.bufferName, *hbmBaseOffset,
                          intermBuf, intermSlot, residence.size});
    init.steps.push_back({*movInstr, intermBuf, intermSlot, dstBuf,
                          residence.offset, residence.size});
    return init;
  }

  return funcOp.emitError() << "no data movement path from HBM to @"
                            << dstBuf.getValue() << " for accumulator init";
}

static FailureOr<OutputMovementPlan>
buildOutputMovementPlan(func::FuncOp funcOp, const LogicalPlan &plan,
                        const ResourcePlan &layout, unsigned nodeIdx,
                        unsigned dstOperandIdx) {
  auto &node = plan.nodes[nodeIdx];
  DefineOp defineOp = node.instruction;
  unsigned numSrc = defineOp.getSources().size();
  unsigned operandIdx = numSrc + dstOperandIdx;
  StringAttr dstBuf = getOperandBuffer(defineOp, operandIdx);
  unsigned hbmValueId =
      node.outputs[dstOperandIdx].writebackTargetValueId
          ? *node.outputs[dstOperandIdx].writebackTargetValueId
          : node.outputs[dstOperandIdx].valueId;
  auto &residence = layout.operandResidences[nodeIdx][operandIdx];

  auto hbmBaseOffset = getTransformedHBMBaseOffset(
      layout, hbmValueId, node.outputs[dstOperandIdx].writebackTransforms);
  if (failed(hbmBaseOffset))
    return funcOp.emitError()
           << "failed to compute HBM base offset for output movement";

  OutputMovementPlan movement{nodeIdx,
                              dstOperandIdx,
                              hbmValueId,
                              node.outputs[dstOperandIdx].writebackTransforms,
                              {}};
  auto storeInstr = layout.dmCatalog.lookup(dstBuf, layout.bufferName);
  if (storeInstr) {
    movement.steps.push_back({*storeInstr, dstBuf, residence.offset,
                              layout.bufferName, *hbmBaseOffset,
                              residence.size});
    return movement;
  }

  for (auto &[key, defs] : layout.dmCatalog.entries) {
    if (key.first != dstBuf || key.second == layout.bufferName)
      continue;
    StringAttr intermBuf = key.second;
    auto movInstr = layout.dmCatalog.lookup(dstBuf, intermBuf);
    if (!movInstr)
      continue;
    auto storeViaIntermediate =
        layout.dmCatalog.lookup(intermBuf, layout.bufferName);
    if (!storeViaIntermediate)
      continue;

    int64_t intermSlot =
        findIntermediateSlot(layout, nodeIdx, numSrc, intermBuf);
    movement.steps.push_back({*movInstr, dstBuf, residence.offset, intermBuf,
                              intermSlot, residence.size});
    movement.steps.push_back({*storeViaIntermediate, intermBuf, intermSlot,
                              layout.bufferName, *hbmBaseOffset,
                              residence.size});
    return movement;
  }

  return funcOp.emitError()
         << "no data movement path from @" << dstBuf.getValue() << " to HBM @"
         << layout.bufferName.getValue();
}

bool mlir::act::isIdentityTransformChain(
    ArrayRef<LogicalTransform> transforms) {
  return transforms.empty();
}

void LogicalPlan::dump() const {
  llvm::dbgs() << "\n=== Logical Plan ===\n";
  llvm::dbgs() << "Values:\n";
  for (auto [idx, value] : llvm::enumerate(values)) {
    llvm::dbgs() << "  v" << idx << " (" << getLogicalValueKindName(value.kind)
                 << "): ";
    if (value.sourceValue)
      llvm::dbgs() << value.sourceValue;
    else
      llvm::dbgs() << "<null>";
    llvm::dbgs() << " type=" << value.type;
    if (value.definingNodeIdx)
      llvm::dbgs() << " def=n" << *value.definingNodeIdx;
    else
      llvm::dbgs() << " def=<input>";
    llvm::dbgs() << " uses=[";
    for (unsigned i = 0; i < value.uses.size(); ++i) {
      auto &use = value.uses[i];
      if (i)
        llvm::dbgs() << ", ";
      llvm::dbgs() << "n" << use.consumerNodeIdx << ":op"
                   << use.consumerOperandIdx;
      printLogicalTransformChain(llvm::dbgs(), use.requiredTransforms);
    }
    llvm::dbgs() << "]\n";
  }

  llvm::dbgs() << "Nodes:\n";
  for (auto [idx, node] : llvm::enumerate(nodes)) {
    llvm::dbgs() << "  n" << idx << ": ";
    for (unsigned i = 0; i < node.sourceOps.size(); ++i) {
      if (i)
        llvm::dbgs() << "+";
      llvm::dbgs() << node.sourceOps[i]->getName();
    }
    llvm::dbgs() << " -> @"
                 << const_cast<DefineOp &>(node.instruction).getSymName()
                 << " in=[";
    for (unsigned i = 0; i < node.inputs.size(); ++i) {
      if (i)
        llvm::dbgs() << ", ";
      llvm::dbgs() << "v" << node.inputs[i].valueId;
      printLogicalTransformChain(llvm::dbgs(),
                                 node.inputs[i].requiredTransforms);
    }
    llvm::dbgs() << "] out=[";
    for (unsigned i = 0; i < node.outputs.size(); ++i) {
      if (i)
        llvm::dbgs() << ", ";
      llvm::dbgs() << "v" << node.outputs[i].valueId;
      if (node.outputs[i].writebackTargetValueId) {
        llvm::dbgs() << "->v" << *node.outputs[i].writebackTargetValueId;
        printLogicalTransformChain(llvm::dbgs(),
                                   node.outputs[i].writebackTransforms);
      }
    }
    llvm::dbgs() << "]\n";
  }
}

void ResourcePlan::dump() const {
  llvm::dbgs() << "\n=== Resource Plan ===\n";
  llvm::dbgs() << "HBM @" << (bufferName ? bufferName.getValue() : "<none>")
               << " total=" << totalAllocated << " / " << bufferSize << "\n";
  for (auto &[valueId, tl] : layouts) {
    llvm::dbgs() << "  v" << valueId << " -> offset=" << tl.baseOffset
                 << " shape=[";
    for (unsigned i = 0; i < tl.shape.size(); ++i) {
      if (i)
        llvm::dbgs() << ",";
      llvm::dbgs() << tl.shape[i];
    }
    llvm::dbgs() << "]\n";
  }

  llvm::dbgs() << "Operand residences:\n";
  for (auto [nodeIdx, residences] : llvm::enumerate(operandResidences)) {
    llvm::dbgs() << "  n" << nodeIdx << ": ";
    for (unsigned i = 0; i < residences.size(); ++i) {
      if (i)
        llvm::dbgs() << ", ";
      auto &res = residences[i];
      llvm::dbgs() << "op" << res.operandIdx << "=@"
                   << res.bufferName.getValue() << "[" << res.offset << ".."
                   << (res.offset + res.size - 1) << "]";
    }
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Forwarding:\n";
  for (auto &edge : forwardingEdges) {
    llvm::dbgs() << "  n" << edge.producerNodeIdx << ":dst"
                 << edge.producerDstOperandIdx << " -> n"
                 << edge.consumerNodeIdx << ":src" << edge.consumerSrcOperandIdx
                 << " via @" << edge.bufferName.getValue() << "\n";
  }

  llvm::dbgs() << "Input movements:\n";
  for (auto &plan : inputMovements) {
    llvm::dbgs() << "  n" << plan.nodeIdx << ":src" << plan.srcOperandIdx
                 << " v" << plan.hbmValueId;
    printLogicalTransformChain(llvm::dbgs(), plan.hbmTransforms);
    llvm::dbgs() << "\n";
    for (auto &step : plan.steps) {
      llvm::dbgs() << "    @"
                   << const_cast<DefineOp &>(step.instruction).getSymName()
                   << " " << step.srcBuffer.getValue() << "[" << step.srcOffset
                   << "] -> " << step.dstBuffer.getValue() << "["
                   << step.dstOffset << "] size=" << step.size << "\n";
    }
  }

  llvm::dbgs() << "Accumulator inits:\n";
  for (auto &plan : accumulatorInits) {
    llvm::dbgs() << "  n" << plan.nodeIdx << ":dst" << plan.dstOperandIdx
                 << " v" << plan.hbmValueId;
    printLogicalTransformChain(llvm::dbgs(), plan.hbmTransforms);
    llvm::dbgs() << "\n";
    for (auto &step : plan.steps) {
      llvm::dbgs() << "    @"
                   << const_cast<DefineOp &>(step.instruction).getSymName()
                   << " " << step.srcBuffer.getValue() << "[" << step.srcOffset
                   << "] -> " << step.dstBuffer.getValue() << "["
                   << step.dstOffset << "] size=" << step.size << "\n";
    }
  }

  llvm::dbgs() << "Output movements:\n";
  for (auto &plan : outputMovements) {
    llvm::dbgs() << "  n" << plan.nodeIdx << ":dst" << plan.dstOperandIdx
                 << " v" << plan.hbmValueId;
    printLogicalTransformChain(llvm::dbgs(), plan.hbmTransforms);
    llvm::dbgs() << "\n";
    for (auto &step : plan.steps) {
      llvm::dbgs() << "    @"
                   << const_cast<DefineOp &>(step.instruction).getSymName()
                   << " " << step.srcBuffer.getValue() << "[" << step.srcOffset
                   << "] -> " << step.dstBuffer.getValue() << "["
                   << step.dstOffset << "] size=" << step.size << "\n";
    }
  }
}

bool LayoutSignature::matches(const LayoutSignature &required) const {
  if (required.hasTranspose != hasTranspose)
    return false;
  if (required.hasTranspose && permutation != required.permutation)
    return false;
  return true;
}

std::optional<DefineOp>
DataMovementCatalog::lookup(StringAttr src, StringAttr dst,
                            const LayoutSignature &required) const {
  auto it = entries.find({src, dst});
  if (it == entries.end())
    return std::nullopt;
  for (auto &[sig, def] : it->second) {
    if (sig.matches(required))
      return def;
  }
  return std::nullopt;
}

/// Convert a SemanticEdgeTransformChain to a LogicalTransformChain.
static LogicalTransformChain
toLogicalTransformChain(const SemanticEdgeTransformChain &chain) {
  LogicalTransformChain result;
  for (auto &t : chain) {
    LogicalTransform lt;
    switch (t.kind) {
    case SemanticEdgeTransformKind::Transpose:
      lt.kind = LogicalTransformKind::Transpose;
      lt.permutation.assign(t.permutation.begin(), t.permutation.end());
      break;
    case SemanticEdgeTransformKind::ExtractSlice:
      lt.kind = LogicalTransformKind::ExtractSlice;
      lt.sliceSpec = t.sliceSpec;
      break;
    case SemanticEdgeTransformKind::InsertSlice:
      lt.kind = LogicalTransformKind::InsertSlice;
      lt.sliceSpec = t.sliceSpec;
      break;
    }
    result.push_back(std::move(lt));
  }
  return result;
}

/// Assign one planner node input slot exactly once from either an internal
/// graph edge or a true external boundary input.
static LogicalResult assignNodeInput(LogicalPlanNode &node, unsigned operandIdx,
                                     unsigned valueId,
                                     const LogicalTransformChain &transforms,
                                     Operation *errorOp) {
  if (operandIdx >= node.inputs.size())
    return errorOp->emitError()
           << "logical plan input slot " << operandIdx << " out of range";
  if (node.inputs[operandIdx].valueId != GraphEdge::kNullIdx)
    return errorOp->emitError() << "logical plan input slot " << operandIdx
                                << " assigned more than once";
  node.inputs[operandIdx] = {valueId, transforms};
  return success();
}

FailureOr<LogicalPlan>
mlir::act::buildLogicalPlan(func::FuncOp funcOp, const SemanticsGraph &graph,
                            const GraphParamSolution &paramSolution) {
  LogicalPlan plan;

  // Register function arguments as logical values.
  for (auto arg : funcOp.getArguments()) {
    if (!isa<RankedTensorType>(arg.getType()))
      continue;
    (void)getOrCreateExternalLogicalValue(plan, arg,
                                          LogicalPlanValueKind::FunctionInput);
  }

  // Create planner nodes and their produced output values first.
  for (auto [graphNodeIdx, graphNode] : llvm::enumerate(graph.nodes)) {
    auto &ps = paramSolution[graphNodeIdx];
    if (!ps.isValid) {
      getPlanningErrorOp(graphNode)->emitError(
          "parameter solving failed for this instruction match");
      return failure();
    }

    unsigned nodeIdx = plan.nodes.size();
    LogicalPlanNode node;
    node.sourceOps = graphNode.sourceOps;
    DefineOp instruction = const_cast<DefineOp &>(graphNode.instruction);
    node.instruction = instruction;
    node.solvedParams = ps.solvedParams;
    node.paramKinds = ps.paramKinds;

    unsigned numTensorOperands =
        instruction.getSources().size() + instruction.getDestinations().size();
    node.inputs.resize(numTensorOperands);

    unsigned numOutputs = instruction.getDestinations().size();
    if (graphNode.boundaryOutputs.size() != numOutputs) {
      return getPlanningErrorOp(graphNode)->emitError()
             << "selected node @" << instruction.getSymName() << " has "
             << graphNode.boundaryOutputs.size()
             << " boundary outputs, expected " << numOutputs;
    }

    node.outputs.resize(numOutputs);
    for (auto [outputIdx, binding] :
         llvm::enumerate(graphNode.boundaryOutputs)) {
      unsigned producedValueId =
          createProducedLogicalValue(plan, binding.producedValue, nodeIdx);
      node.outputs[outputIdx].valueId = producedValueId;

      if (!binding.materializedValue)
        continue;

      unsigned materializedValueId = getOrCreateExternalLogicalValue(
          plan, binding.materializedValue,
          LogicalPlanValueKind::MaterializedOutput);
      node.outputs[outputIdx].writebackTargetValueId = materializedValueId;
      node.outputs[outputIdx].writebackTransforms =
          toLogicalTransformChain(binding.transforms);
    }

    plan.nodes.push_back(std::move(node));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Planner input wiring from SemanticsGraph edges:\n";
    for (auto [edgeIdx, edge] : llvm::enumerate(graph.edges)) {
      llvm::dbgs() << "  edge[" << edgeIdx << "] n" << edge.producerNodeIdx
                   << ":out" << edge.prodIdx << " -> n" << edge.consumerNodeIdx
                   << ":op" << edge.consIdx << "\n";
    }
  });

  // Wire inter-node uses from the selected graph edges.
  for (auto [edgeIdx, edge] : llvm::enumerate(graph.edges)) {
    if (edge.producerNodeIdx >= plan.nodes.size() ||
        edge.consumerNodeIdx >= plan.nodes.size()) {
      return funcOp.emitError() << "selected graph edge " << edgeIdx
                                << " references an out-of-range planner node";
    }

    LogicalPlanNode &producer = plan.nodes[edge.producerNodeIdx];
    LogicalPlanNode &consumer = plan.nodes[edge.consumerNodeIdx];
    if (edge.prodIdx >= producer.outputs.size()) {
      return producer.sourceOps.front()->emitError()
             << "selected graph edge " << edgeIdx
             << " references a missing producer output slot";
    }

    unsigned producedValueId = producer.outputs[edge.prodIdx].valueId;
    LogicalTransformChain transforms = toLogicalTransformChain(edge.transforms);
    if (failed(assignNodeInput(consumer, edge.consIdx, producedValueId,
                               transforms, consumer.sourceOps.front())))
      return failure();
    plan.values[producedValueId].uses.push_back(
        {edge.consumerNodeIdx, edge.consIdx, std::move(transforms)});
  }

  // Wire true external inputs after internal graph edges are fixed in place.
  for (auto [graphNodeIdx, graphNode] : llvm::enumerate(graph.nodes)) {
    LogicalPlanNode &node = plan.nodes[graphNodeIdx];
    for (auto &binding : graphNode.boundaryInputs) {
      unsigned valueId = getOrCreateExternalLogicalValue(
          plan, binding.value, LogicalPlanValueKind::FunctionInput);
      LogicalTransformChain transforms =
          toLogicalTransformChain(binding.transforms);
      if (failed(assignNodeInput(node, binding.boundaryOperandIdx, valueId,
                                 transforms, getPlanningErrorOp(graphNode))))
        return failure();
      plan.values[valueId].uses.push_back(
          {static_cast<unsigned>(graphNodeIdx),
           static_cast<unsigned>(binding.boundaryOperandIdx),
           std::move(transforms)});
    }
  }

  // Validate that every tensor operand at the compute boundary is satisfied.
  for (auto [nodeIdx, node] : llvm::enumerate(plan.nodes)) {
    for (auto [operandIdx, input] : llvm::enumerate(node.inputs)) {
      if (input.valueId != GraphEdge::kNullIdx)
        continue;
      return node.sourceOps.front()->emitError()
             << "logical plan input slot " << operandIdx
             << " for selected node " << nodeIdx
             << " was not assigned by either a graph edge or an external input";
    }
  }

  LLVM_DEBUG(plan.dump());
  return plan;
}

FailureOr<ResourcePlan> mlir::act::buildResourcePlan(func::FuncOp funcOp,
                                                     const LogicalPlan &plan,
                                                     ModuleOp module) {
  if (plan.nodes.empty()) {
    ResourcePlan empty;
    return empty;
  }

  DefineOp defineOp = plan.nodes[0].instruction;
  bool singleBuffer = isSingleBufferInstruction(defineOp);

  if (singleBuffer) {
    StringAttr bufferName =
        cast<FlatSymbolRefAttr>(defineOp.getSources()[0]).getAttr();
    auto bufOp = SymbolTable::lookupNearestSymbolFrom<DeclareBufferOp>(
        module, bufferName);
    if (!bufOp)
      return defineOp.emitError() << "buffer @" << bufferName << " not found";

    ResourcePlan layout;
    layout.bufferName = bufferName;
    layout.bufferSize = getBufferElementCapacity(bufOp);
    layout.operandResidences.resize(plan.nodes.size());

    auto layoutAliases = computeLayoutAliases(plan);
    auto liveRanges = computeLiveRanges(funcOp, plan, layoutAliases);
    int64_t peakAlloc = 0;
    auto allocatedOffsets = greedyAllocate(liveRanges, peakAlloc);
    for (auto &lr : liveRanges) {
      auto shape = plan.values[lr.valueId].type.getShape();
      layout.layouts[lr.valueId] = {
          allocatedOffsets[lr.valueId],
          SmallVector<int64_t>(shape.begin(), shape.end()),
          computeRowMajorStrides(shape)};
    }
    for (auto &[aliasValueId, baseValueId] : layoutAliases) {
      auto it = layout.layouts.find(baseValueId);
      if (it == layout.layouts.end())
        continue;
      layout.layouts[aliasValueId] = it->second;
    }
    layout.totalAllocated = peakAlloc;

    for (unsigned nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
      DefineOp instrDef = plan.nodes[nodeIdx].instruction;
      unsigned numOperands =
          instrDef.getSources().size() + instrDef.getDestinations().size();
      auto &residences = layout.operandResidences[nodeIdx];
      for (unsigned operandIdx = 0; operandIdx < numOperands; ++operandIdx) {
        residences.push_back({bufferName, 0, 0, operandIdx});
      }
    }

    if (layout.totalAllocated > layout.bufferSize) {
      return funcOp.emitError()
             << "total allocation (" << layout.totalAllocated
             << ") exceeds buffer capacity (" << layout.bufferSize << ") of @"
             << bufferName.getValue();
    }

    LLVM_DEBUG(layout.dump());
    return layout;
  }

  auto hbmBufOrErr = identifyHBMBuffer(module);
  if (failed(hbmBufOrErr))
    return failure();
  DeclareBufferOp hbmBuf = *hbmBufOrErr;

  ResourcePlan layout;
  layout.bufferName = hbmBuf.getSymNameAttr();
  layout.bufferSize = getBufferElementCapacity(hbmBuf);
  layout.needsDataMovement = true;
  layout.dmCatalog = buildDataMovementCatalog(module);
  layout.operandResidences.resize(plan.nodes.size());

  auto layoutAliases = computeLayoutAliases(plan);
  auto liveRanges = computeLiveRanges(funcOp, plan, layoutAliases);
  int64_t peakAlloc = 0;
  auto allocatedOffsets = greedyAllocate(liveRanges, peakAlloc);
  for (auto &lr : liveRanges) {
    auto shape = plan.values[lr.valueId].type.getShape();
    layout.layouts[lr.valueId] = {
        allocatedOffsets[lr.valueId],
        SmallVector<int64_t>(shape.begin(), shape.end()),
        computeRowMajorStrides(shape)};
  }
  for (auto &[aliasValueId, baseValueId] : layoutAliases) {
    auto it = layout.layouts.find(baseValueId);
    if (it == layout.layouts.end())
      continue;
    layout.layouts[aliasValueId] = it->second;
  }
  layout.totalAllocated = peakAlloc;
  if (layout.totalAllocated > layout.bufferSize) {
    return funcOp.emitError()
           << "total HBM allocation (" << layout.totalAllocated
           << ") exceeds capacity (" << layout.bufferSize << ") of @"
           << layout.bufferName.getValue();
  }

  layout.forwardingEdges =
      detectForwardingOpportunities(plan, layout.bufferName);

  DenseMap<std::pair<unsigned, unsigned>, std::pair<unsigned, unsigned>>
      forwardMap;
  for (auto &edge : layout.forwardingEdges) {
    DefineOp producerDef = plan.nodes[edge.producerNodeIdx].instruction;
    unsigned pNumSrc = producerDef.getSources().size();
    forwardMap[{edge.consumerNodeIdx, edge.consumerSrcOperandIdx}] = {
        edge.producerNodeIdx, pNumSrc + edge.producerDstOperandIdx};
  }

  DenseMap<StringAttr, int64_t> maxPerBuffer;
  for (unsigned nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
    auto &node = plan.nodes[nodeIdx];
    DefineOp instrDef = node.instruction;
    unsigned numOperands =
        instrDef.getSources().size() + instrDef.getDestinations().size();
    DenseMap<StringAttr, int64_t> entryNextSlot;

    for (auto &edge : layout.forwardingEdges) {
      if (edge.consumerNodeIdx != nodeIdx)
        continue;
      auto [prodNode, prodOperandIdx] =
          forwardMap[{nodeIdx, edge.consumerSrcOperandIdx}];
      auto &prodResidence = layout.operandResidences[prodNode][prodOperandIdx];
      int64_t fwdEnd = prodResidence.offset + prodResidence.size;
      entryNextSlot[edge.bufferName] =
          std::max(entryNextSlot[edge.bufferName], fwdEnd);
    }

    auto &residences = layout.operandResidences[nodeIdx];
    for (unsigned operandIdx = 0; operandIdx < numOperands; ++operandIdx) {
      StringAttr buf = getOperandBuffer(instrDef, operandIdx);
      if (buf == layout.bufferName) {
        residences.push_back({buf, 0, 0, operandIdx});
        continue;
      }

      auto fwdIt = forwardMap.find({nodeIdx, operandIdx});
      if (fwdIt != forwardMap.end()) {
        auto [prodNode, prodOperandIdx] = fwdIt->second;
        auto &prodResidence =
            layout.operandResidences[prodNode][prodOperandIdx];
        residences.push_back({prodResidence.bufferName, prodResidence.offset,
                              prodResidence.size, operandIdx});
        continue;
      }

      auto slotCount =
          evaluateOperandSlotCount(instrDef, operandIdx, node.solvedParams);
      if (failed(slotCount))
        return failure();

      int64_t slotOffset = entryNextSlot[buf];
      entryNextSlot[buf] += *slotCount;
      residences.push_back({buf, slotOffset, *slotCount, operandIdx});
    }

    for (auto &[buf, total] : entryNextSlot)
      maxPerBuffer[buf] = std::max(maxPerBuffer[buf], total);
  }

  for (auto &[bufName, totalSlots] : maxPerBuffer) {
    auto bufOp =
        SymbolTable::lookupNearestSymbolFrom<DeclareBufferOp>(module, bufName);
    if (!bufOp)
      return funcOp.emitError() << "buffer @" << bufName << " not found";
    if (totalSlots > bufOp.getSize()) {
      return funcOp.emitError()
             << "scratchpad allocation (" << totalSlots
             << " slots) exceeds capacity (" << bufOp.getSize()
             << " slots) of @" << bufName.getValue();
    }
  }

  for (unsigned nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
    auto &node = plan.nodes[nodeIdx];
    DefineOp instrDef = node.instruction;
    unsigned numSrc = instrDef.getSources().size();
    unsigned numDst = instrDef.getDestinations().size();

    for (unsigned srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
      StringAttr srcBuf = getOperandBuffer(instrDef, srcIdx);
      if (srcBuf == layout.bufferName)
        continue;
      if (forwardMap.contains({nodeIdx, srcIdx}))
        continue;
      auto inputPlan =
          buildInputMovementPlan(funcOp, plan, layout, nodeIdx, srcIdx);
      if (failed(inputPlan))
        return failure();
      layout.inputMovements.push_back(std::move(*inputPlan));
    }

    for (unsigned dstIdx = 0; dstIdx < numDst; ++dstIdx) {
      unsigned operandIdx = numSrc + dstIdx;
      StringAttr dstBuf = getOperandBuffer(instrDef, operandIdx);
      if (dstBuf == layout.bufferName)
        continue;

      bool isForwarded = llvm::any_of(layout.forwardingEdges, [&](auto &edge) {
        return edge.producerNodeIdx == nodeIdx &&
               edge.producerDstOperandIdx == dstIdx;
      });
      if (isForwarded)
        continue;

      auto outputPlan =
          buildOutputMovementPlan(funcOp, plan, layout, nodeIdx, dstIdx);
      if (failed(outputPlan))
        return failure();
      layout.outputMovements.push_back(std::move(*outputPlan));
    }
  }

  LLVM_DEBUG(layout.dump());
  return layout;
}

LogicalTransformChain act::getRequiredTransforms(const LogicalPlan &plan,
                                                 unsigned valueId,
                                                 unsigned consumerNodeIdx,
                                                 unsigned consumerOperandIdx) {
  if (consumerNodeIdx < plan.nodes.size()) {
    auto &node = plan.nodes[consumerNodeIdx];
    if (consumerOperandIdx < node.inputs.size() &&
        node.inputs[consumerOperandIdx].valueId == valueId) {
      return node.inputs[consumerOperandIdx].requiredTransforms;
    }
  }

  auto &value = plan.values[valueId];
  for (auto &use : value.uses) {
    if (use.consumerNodeIdx == consumerNodeIdx &&
        use.consumerOperandIdx == consumerOperandIdx)
      return use.requiredTransforms;
  }
  return {};
}
