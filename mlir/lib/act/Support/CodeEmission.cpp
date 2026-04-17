#include "act/Support/CodeEmission.h"
#include "act/Support/Planning.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "code-emission"

using namespace mlir;
using namespace mlir::act;

//===----------------------------------------------------------------------===//
// Phase 3c helpers
//===----------------------------------------------------------------------===//

/// For each instruction operand (indexed by yield position), find the addr
/// block arg used as the base offset (the "basis" of the underlying StridedOp).
/// Returns a map: operand index -> addr block arg index.
static DenseMap<unsigned, unsigned>
mapOperandsToOffsetParams(DefineOp defineOp) {
  Block &addrBlock = defineOp.getAccessBlock();
  auto *yieldOp = addrBlock.getTerminator();
  DenseMap<unsigned, unsigned> result;

  for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
    Operation *op = yieldOp->getOperand(i).getDefiningOp();
    if (!op)
      continue;
    // Trace back through relayout ops to find the StridedOp
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
    auto strided = dyn_cast_or_null<StridedOp>(op);
    if (!strided)
      continue;

    auto mixedBasis = getMixedValues(strided.getStaticBasis(),
                                     strided.getBasis(), strided.getContext());
    if (mixedBasis.size() != 1)
      continue;
    auto v = dyn_cast<Value>(mixedBasis[0]);
    if (!v)
      continue;
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg || blockArg.getOwner() != &addrBlock)
      continue;
    result[i] = blockArg.getArgNumber();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Operand->OffsetParam map: {";
    bool first = true;
    for (auto &[opIdx, paramIdx] : result) {
      if (!first)
        llvm::dbgs() << ", ";
      llvm::dbgs() << "op" << opIdx << "->p" << paramIdx;
      first = false;
    }
    llvm::dbgs() << "}\n";
  });

  return result;
}

namespace {
struct HBMAccess {
  unsigned valueId;
  ArrayRef<LogicalTransform> transforms;
};
} // namespace

static HBMAccess getHBMAccessForOperand(const LogicalPlanNode &node,
                                        DefineOp defineOp,
                                        unsigned operandIdx) {
  unsigned numSrc = defineOp.getSources().size();
  if (operandIdx < numSrc) {
    auto &input = node.inputs[operandIdx];
    return {input.valueId, input.requiredTransforms};
  }

  auto &output = node.outputs[operandIdx - numSrc];
  if (output.writebackTargetValueId)
    return {*output.writebackTargetValueId, output.writebackTransforms};
  static const LogicalTransformChain emptyTransforms;
  return {output.valueId, emptyTransforms};
}

static const OperandResidence *getOperandResidence(const ResourcePlan &layout,
                                                   unsigned nodeIdx,
                                                   unsigned operandIdx) {
  if (nodeIdx >= layout.operandResidences.size())
    return nullptr;
  auto &residences = layout.operandResidences[nodeIdx];
  if (operandIdx >= residences.size())
    return nullptr;
  return &residences[operandIdx];
}

static const InputMovementPlan *
findInputMovementPlan(const ResourcePlan &layout, unsigned nodeIdx,
                      unsigned srcOperandIdx) {
  for (auto &plan : layout.inputMovements) {
    if (plan.nodeIdx == nodeIdx && plan.srcOperandIdx == srcOperandIdx)
      return &plan;
  }
  return nullptr;
}

static const OutputMovementPlan *
findOutputMovementPlan(const ResourcePlan &layout, unsigned nodeIdx,
                       unsigned dstOperandIdx) {
  for (auto &plan : layout.outputMovements) {
    if (plan.nodeIdx == nodeIdx && plan.dstOperandIdx == dstOperandIdx)
      return &plan;
  }
  return nullptr;
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
      if (!transform.sliceSpec || !isSupportedStaticContiguousSlice(
                                      current.shape, *transform.sliceSpec))
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

//===----------------------------------------------------------------------===//
// Phase 3c: Data Movement Helpers
//===----------------------------------------------------------------------===//

/// Emit an act.emit for a data movement instruction with all-static params.
/// dmDefine: the identity DefineOp (e.g., @load_rm)
/// srcOffset: offset in the source buffer
/// dstOffset: offset in the destination buffer
/// slotCount: number of slots to move
static void emitDataMovement(RewriterBase &rewriter, Location loc,
                             DefineOp dmDefine, int64_t srcOffset,
                             int64_t dstOffset, int64_t slotCount) {
  MLIRContext *ctx = rewriter.getContext();
  auto operandToParam = mapOperandsToOffsetParams(dmDefine);

  unsigned numParams = dmDefine.getAccessBlock().getNumArguments();
  SmallVector<int64_t> staticParams(numParams, 0);

  // Set offset params from operand mapping
  for (auto &[opIdx, paramIdx] : operandToParam) {
    unsigned numSrc = dmDefine.getSources().size();
    // opIdx < numSrc means it's a src operand
    staticParams[paramIdx] = (opIdx < numSrc) ? srcOffset : dstOffset;
  }

  // Set remaining params (not mapped to operands) to slotCount
  DenseSet<unsigned> offsetParams;
  for (auto &[_, paramIdx] : operandToParam)
    offsetParams.insert(paramIdx);
  for (unsigned p = 0; p < numParams; ++p) {
    if (!offsetParams.contains(p))
      staticParams[p] = slotCount;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    act.emit @" << dmDefine.getSymName() << " addr(";
    for (unsigned i = 0; i < staticParams.size(); ++i) {
      if (i)
        llvm::dbgs() << ", ";
      llvm::dbgs() << staticParams[i];
    }
    llvm::dbgs() << ") compute()\n";
  });

  EmitOp::create(
      rewriter, loc, FlatSymbolRefAttr::get(ctx, dmDefine.getSymName()),
      ValueRange{}, ValueRange{}, DenseI64ArrayAttr::get(ctx, staticParams),
      DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
}

static LogicalResult emitRegionLowering(RewriterBase &rewriter,
                                        LogicalPlan &plan,
                                        ResourcePlan &layout) {

  if (plan.nodes.empty())
    return success();

  /// Helper: compute static HBM offset for a tensor operand.
  auto computeHBMOffset =
      [&](const LogicalPlanNode &node, DefineOp defineOp, unsigned operandIdx,
          unsigned hbmValueId,
          ArrayRef<LogicalTransform> transforms) -> FailureOr<int64_t> {
    auto layoutIt = layout.layouts.find(hbmValueId);
    if (layoutIt == layout.layouts.end())
      return int64_t{0};
    auto transformedLayout =
        applyLogicalTransforms(layoutIt->second, transforms);
    if (failed(transformedLayout)) {
      node.sourceOps[0]->emitError()
          << "failed to compute transformed HBM layout for operand "
          << operandIdx;
      return failure();
    }
    return transformedLayout->baseOffset;
  };

  for (unsigned nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
    const LogicalPlanNode &node = plan.nodes[nodeIdx];
    Operation *sourceOp = node.sourceOps.front();
    DefineOp defineOp = node.instruction;
    Location loc = sourceOp->getLoc();

    LLVM_DEBUG(llvm::dbgs() << "\n  Emitting code for " << sourceOp->getName()
                            << " -> @" << defineOp.getSymName() << "\n");

    auto emitMovementSteps =
        [&](ArrayRef<MovementStep> steps, unsigned operandIdx,
            unsigned hbmValueId,
            ArrayRef<LogicalTransform> hbmTransforms) -> LogicalResult {
      for (const MovementStep &step : steps) {
        bool srcIsHBM = step.srcBuffer == layout.bufferName;
        bool dstIsHBM = step.dstBuffer == layout.bufferName;
        if (srcIsHBM && dstIsHBM)
          return sourceOp->emitError()
                 << "movement step cannot use HBM for both src and dst";

        if (srcIsHBM || dstIsHBM) {
          auto hbmOff = computeHBMOffset(node, defineOp, operandIdx, hbmValueId,
                                         hbmTransforms);
          if (failed(hbmOff))
            return failure();
          if (srcIsHBM) {
            emitDataMovement(rewriter, loc, step.instruction, *hbmOff,
                             step.dstOffset, step.size);
          } else {
            emitDataMovement(rewriter, loc, step.instruction, step.srcOffset,
                             *hbmOff, step.size);
          }
          continue;
        }

        emitDataMovement(rewriter, loc, step.instruction, step.srcOffset,
                         step.dstOffset, step.size);
      }
      return success();
    };

    auto operandToParam = mapOperandsToOffsetParams(defineOp);
    DenseMap<unsigned, unsigned> paramToOperand;
    for (auto &[opIdx, paramIdx] : operandToParam)
      paramToOperand[paramIdx] = opIdx;

    auto buildStaticAddrParams =
        [&](bool useScratchpadOffsets) -> FailureOr<SmallVector<int64_t>> {
      unsigned numAddrParams = defineOp.getAccessBlock().getNumArguments();
      SmallVector<int64_t> staticAddrParams(numAddrParams, 0);

      for (unsigned p = 0; p < numAddrParams; ++p) {
        auto kindIt = node.paramKinds.find(p);
        AddrParamKind kind = (kindIt != node.paramKinds.end())
                                 ? kindIt->second
                                 : AddrParamKind::Offset;

        if (kind == AddrParamKind::Shape) {
          auto solvedIt = node.solvedParams.find(p);
          if (solvedIt == node.solvedParams.end()) {
            sourceOp->emitError() << "shape param p" << p << " not solved";
            return failure();
          }
          staticAddrParams[p] = solvedIt->second;
          continue;
        }

        auto opIt = paramToOperand.find(p);
        if (opIt == paramToOperand.end()) {
          sourceOp->emitError()
              << "offset param p" << p << " not mapped to any operand";
          return failure();
        }
        unsigned operandIdx = opIt->second;

        if (useScratchpadOffsets) {
          auto *residence = getOperandResidence(layout, nodeIdx, operandIdx);
          if (!residence) {
            sourceOp->emitError()
                << "missing planned residence for operand " << operandIdx;
            return failure();
          }
          if (residence->bufferName == layout.bufferName) {
            auto hbmAccess = getHBMAccessForOperand(node, defineOp, operandIdx);
            auto hbmOff =
                computeHBMOffset(node, defineOp, operandIdx, hbmAccess.valueId,
                                 hbmAccess.transforms);
            if (failed(hbmOff))
              return failure();
            staticAddrParams[p] = *hbmOff;
          } else {
            staticAddrParams[p] = residence->offset;
          }
          continue;
        }

        auto hbmAccess = getHBMAccessForOperand(node, defineOp, operandIdx);
        auto hbmOff = computeHBMOffset(node, defineOp, operandIdx,
                                       hbmAccess.valueId, hbmAccess.transforms);
        if (failed(hbmOff))
          return failure();
        staticAddrParams[p] = *hbmOff;
      }

      return staticAddrParams;
    };

    if (layout.needsDataMovement) {
      unsigned numSrc = defineOp.getSources().size();
      unsigned numDst = defineOp.getDestinations().size();

      for (unsigned srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
        auto *movementPlan = findInputMovementPlan(layout, nodeIdx, srcIdx);
        if (!movementPlan)
          continue;
        if (failed(emitMovementSteps(movementPlan->steps, srcIdx,
                                     movementPlan->hbmValueId,
                                     movementPlan->hbmTransforms)))
          return failure();
      }

      auto staticAddrParams =
          buildStaticAddrParams(/*useScratchpadOffsets=*/true);
      if (failed(staticAddrParams))
        return failure();
      EmitOp::create(
          rewriter, loc,
          FlatSymbolRefAttr::get(rewriter.getContext(), defineOp.getSymName()),
          ValueRange{}, ValueRange{},
          DenseI64ArrayAttr::get(rewriter.getContext(), *staticAddrParams),
          DenseI64ArrayAttr::get(rewriter.getContext(),
                                 SmallVector<int64_t>{}));

      for (unsigned dstIdx = 0; dstIdx < numDst; ++dstIdx) {
        auto *movementPlan = findOutputMovementPlan(layout, nodeIdx, dstIdx);
        if (!movementPlan)
          continue;
        if (failed(emitMovementSteps(movementPlan->steps, numSrc + dstIdx,
                                     movementPlan->hbmValueId,
                                     movementPlan->hbmTransforms)))
          return failure();
      }
      continue;
    }

    auto staticAddrParams =
        buildStaticAddrParams(/*useScratchpadOffsets=*/false);
    if (failed(staticAddrParams))
      return failure();
    EmitOp::create(
        rewriter, loc,
        FlatSymbolRefAttr::get(rewriter.getContext(), defineOp.getSymName()),
        ValueRange{}, ValueRange{},
        DenseI64ArrayAttr::get(rewriter.getContext(), *staticAddrParams),
        DenseI64ArrayAttr::get(rewriter.getContext(), SmallVector<int64_t>{}));
  }

  return success();
}

LogicalResult act::emitInstructionSequence(RewriterBase &rewriter,
                                           FunctionLoweringPlan &plan) {
  assert(plan.isComplete);

  func::FuncOp func = plan.func;
  auto module = func->getParentOfType<ModuleOp>();
  assert(module && "expected function to be nested in a module");

  rewriter.setInsertionPointAfter(func);
  auto sequence =
      SequenceOp::create(rewriter, func.getLoc(), func.getSymNameAttr());
  sequence->setDiscardableAttrs(func->getDiscardableAttrDictionary());
  Block *entryBlock = sequence.addEntryBlock();

  rewriter.setInsertionPointToStart(entryBlock);
  if (failed(
          emitRegionLowering(rewriter, plan.logicalPlan, plan.resourcePlan))) {
    rewriter.eraseOp(sequence);
    return failure();
  }

  rewriter.eraseOp(func);
  return success();
}
