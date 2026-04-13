#include "act/Support/CodeEmission.h"
#include "act/Support/Planning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "code-emission"

using namespace mlir;
using namespace mlir::act;

//===----------------------------------------------------------------------===//
// Compute op classification
//===----------------------------------------------------------------------===//

static bool isComputeOp(Operation *op) {
  // Skip non-linalg
  if (!isa<linalg::LinalgOp>(op))
    return false;
  // linalg.fill is infrastructure, linalg.transpose is layout annotation
  if (isa<linalg::FillOp, linalg::TransposeOp>(op))
    return false;
  return true;
}

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

struct HBMAccess {
  unsigned valueId;
  ArrayRef<LogicalTransform> transforms;
};

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

static const AccumulatorInitPlan *
findAccumulatorInitPlan(const ResourcePlan &layout, unsigned nodeIdx,
                        unsigned dstOperandIdx) {
  for (auto &plan : layout.accumulatorInits) {
    if (plan.nodeIdx == nodeIdx && plan.dstOperandIdx == dstOperandIdx)
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
static void emitDataMovement(OpBuilder &builder, Location loc,
                             DefineOp dmDefine, int64_t srcOffset,
                             int64_t dstOffset, int64_t slotCount) {
  MLIRContext *ctx = builder.getContext();
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
      builder, loc, FlatSymbolRefAttr::get(ctx, dmDefine.getSymName()),
      ValueRange{}, ValueRange{}, DenseI64ArrayAttr::get(ctx, staticParams),
      DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
}

/// Emit an act.emit for a data movement instruction with dynamic src offset.
static void emitDataMovementDynamic(OpBuilder &builder, Location loc,
                                    DefineOp dmDefine, Value srcOffset,
                                    int64_t dstOffset, int64_t slotCount) {
  MLIRContext *ctx = builder.getContext();
  auto operandToParam = mapOperandsToOffsetParams(dmDefine);

  unsigned numParams = dmDefine.getAccessBlock().getNumArguments();
  SmallVector<int64_t> staticParams(numParams, 0);
  SmallVector<Value> dynamicParams;

  // Identify which param index corresponds to src offset
  unsigned numSrc = dmDefine.getSources().size();
  DenseSet<unsigned> offsetParams;
  for (auto &[opIdx, paramIdx] : operandToParam) {
    offsetParams.insert(paramIdx);
    if (opIdx < numSrc) {
      // src offset is dynamic
      staticParams[paramIdx] = ShapedType::kDynamic;
      dynamicParams.push_back(srcOffset);
    } else {
      staticParams[paramIdx] = dstOffset;
    }
  }

  // Remaining params = slotCount
  for (unsigned p = 0; p < numParams; ++p) {
    if (!offsetParams.contains(p))
      staticParams[p] = slotCount;
  }

  EmitOp::create(
      builder, loc, FlatSymbolRefAttr::get(ctx, dmDefine.getSymName()),
      dynamicParams, ValueRange{}, DenseI64ArrayAttr::get(ctx, staticParams),
      DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
}

/// Emit a data movement for store: dynamic dst offset (HBM), static src.
static void emitStoreDynamic(OpBuilder &builder, Location loc,
                             DefineOp dmDefine, int64_t srcOffset,
                             Value dstOffset, int64_t slotCount) {
  MLIRContext *ctx = builder.getContext();
  auto operandToParam = mapOperandsToOffsetParams(dmDefine);

  unsigned numParams = dmDefine.getAccessBlock().getNumArguments();
  SmallVector<int64_t> staticParams(numParams, 0);
  SmallVector<Value> dynamicParams;

  unsigned numSrc = dmDefine.getSources().size();
  DenseSet<unsigned> offsetParams;
  for (auto &[opIdx, paramIdx] : operandToParam) {
    offsetParams.insert(paramIdx);
    if (opIdx < numSrc) {
      staticParams[paramIdx] = srcOffset;
    } else {
      // dst offset is dynamic
      staticParams[paramIdx] = ShapedType::kDynamic;
      dynamicParams.push_back(dstOffset);
    }
  }

  for (unsigned p = 0; p < numParams; ++p) {
    if (!offsetParams.contains(p))
      staticParams[p] = slotCount;
  }

  EmitOp::create(
      builder, loc, FlatSymbolRefAttr::get(ctx, dmDefine.getSymName()),
      dynamicParams, ValueRange{}, DenseI64ArrayAttr::get(ctx, staticParams),
      DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
}

//===----------------------------------------------------------------------===//
// Phase 3c: Code Emission
//===----------------------------------------------------------------------===//

static LogicalResult emitCode(func::FuncOp funcOp, const LogicalPlan &plan,
                              const ResourcePlan &layout) {
  MLIRContext *ctx = funcOp.getContext();
  Location loc = funcOp.getLoc();
  OpBuilder builder(ctx);

  /// Helper: compute flat HBM offset for a tensor operand, possibly dynamic.
  /// Returns {isStatic, staticValue, dynamicValue}.
  struct HBMOffset {
    bool isStatic;
    int64_t staticVal;
    Value dynamicVal;
  };
  auto computeHBMOffset =
      [&](const LogicalPlanNode &node, linalg::LinalgOp sourceLinalgOp,
          DefineOp defineOp, unsigned operandIdx, unsigned hbmValueId,
          ArrayRef<LogicalTransform> transforms, const TiledMatchCandidate &tm,
          const DenseMap<unsigned, Value> &loopIVs) -> FailureOr<HBMOffset> {
    auto layoutIt = layout.layouts.find(hbmValueId);
    if (layoutIt == layout.layouts.end())
      return HBMOffset{true, 0, Value{}}; // fallback
    auto transformedLayout =
        applyLogicalTransforms(layoutIt->second, transforms);
    if (failed(transformedLayout)) {
      node.sourceOp->emitError()
          << "failed to compute transformed HBM layout for operand "
          << operandIdx;
      return failure();
    }
    const TensorLayout &tl = *transformedLayout;

    auto indexingMaps = sourceLinalgOp.getIndexingMapsArray();
    if (operandIdx >= indexingMaps.size())
      return HBMOffset{true, tl.baseOffset, Value{}};
    AffineMap map = indexingMaps[operandIdx];

    // Check if any loop IV affects this operand's offset
    bool allStatic = true;
    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(j));
      if (dimExpr && loopIVs.count(dimExpr.getPosition())) {
        allStatic = false;
        break;
      }
    }

    if (allStatic)
      return HBMOffset{true, tl.baseOffset, Value{}};

    // Dynamic: compute flat offset from loop IVs
    Value offset = arith::ConstantIndexOp::create(builder, loc, tl.baseOffset);
    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(j));
      if (!dimExpr)
        continue;
      unsigned iterDim = dimExpr.getPosition();
      int64_t stride = tl.strides[j];
      int64_t nativeValue = tm.tiling.dims[iterDim].nativeValue;
      auto ivIt = loopIVs.find(iterDim);
      if (ivIt == loopIVs.end())
        continue;
      int64_t stepSize = nativeValue * stride;
      Value stepVal = arith::ConstantIndexOp::create(builder, loc, stepSize);
      Value contrib =
          arith::MulIOp::create(builder, loc, ivIt->second, stepVal);
      offset = arith::AddIOp::create(builder, loc, offset, contrib);
    }
    return HBMOffset{false, 0, offset};
  };

  struct LoopDim {
    unsigned dimIdx;
    int64_t tileFactor;
    int64_t nativeValue;
    utils::IteratorType iterType;
  };

  for (unsigned nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
    auto &node = plan.nodes[nodeIdx];
    auto *sourceOp = node.sourceOp;
    auto sourceLinalgOp = cast<linalg::LinalgOp>(sourceOp);
    const TiledMatchCandidate &tm = *node.match;
    DefineOp defineOp = tm.base.instruction;

    builder.setInsertionPoint(sourceOp);

    LLVM_DEBUG(llvm::dbgs() << "\n  Emitting code for " << sourceOp->getName()
                            << " -> @" << defineOp.getSymName() << "\n");

    // --- Identify tiling dims that need loops ---
    SmallVector<LoopDim> parallelLoops, reductionLoops;
    for (unsigned d = 0; d < tm.tiling.dims.size(); ++d) {
      auto &dim = tm.tiling.dims[d];
      if (dim.tileFactor <= 1)
        continue;
      LoopDim ld{d, dim.tileFactor, dim.nativeValue, dim.iterType};
      if (dim.iterType == utils::IteratorType::parallel)
        parallelLoops.push_back(ld);
      else
        reductionLoops.push_back(ld);
    }

    SmallVector<LoopDim> allLoops;
    allLoops.append(parallelLoops.begin(), parallelLoops.end());
    allLoops.append(reductionLoops.begin(), reductionLoops.end());

    LLVM_DEBUG({
      llvm::dbgs() << "  Loop dims: ";
      for (auto &ld : allLoops)
        llvm::dbgs() << "d" << ld.dimIdx << "(tf=" << ld.tileFactor
                     << ",nat=" << ld.nativeValue << ","
                     << (ld.iterType == utils::IteratorType::parallel ? "par"
                                                                      : "red")
                     << ") ";
      if (allLoops.empty())
        llvm::dbgs() << "(none)";
      llvm::dbgs() << "\n";
    });

    // --- Emit nested scf.for loops (parallel first, then reduction) ---
    DenseMap<unsigned, Value> loopIVs;
    SmallVector<scf::ForOp> forOps;

    // Phase 1: Create parallel loops
    for (auto &ld : parallelLoops) {
      Value lb = arith::ConstantIndexOp::create(builder, loc, 0);
      Value ub = arith::ConstantIndexOp::create(builder, loc, ld.tileFactor);
      Value step = arith::ConstantIndexOp::create(builder, loc, 1);
      auto forOp = scf::ForOp::create(builder, loc, lb, ub, step);
      forOps.push_back(forOp);
      loopIVs[ld.dimIdx] = forOp.getInductionVar();
      builder.setInsertionPointToStart(forOp.getBody());
    }

    auto emitMovementSteps =
        [&](ArrayRef<MovementStep> steps, unsigned operandIdx,
            unsigned hbmValueId,
            ArrayRef<LogicalTransform> hbmTransforms) -> LogicalResult {
      for (auto &step : steps) {
        bool srcIsHBM = step.srcBuffer == layout.bufferName;
        bool dstIsHBM = step.dstBuffer == layout.bufferName;
        if (srcIsHBM && dstIsHBM)
          return sourceOp->emitError()
                 << "movement step cannot use HBM for both src and dst";

        if (srcIsHBM || dstIsHBM) {
          auto hbmOff =
              computeHBMOffset(node, sourceLinalgOp, defineOp, operandIdx,
                               hbmValueId, hbmTransforms, tm, loopIVs);
          if (failed(hbmOff))
            return failure();
          if (srcIsHBM) {
            if (hbmOff->isStatic) {
              emitDataMovement(builder, loc, step.instruction,
                               hbmOff->staticVal, step.dstOffset, step.size);
            } else {
              emitDataMovementDynamic(builder, loc, step.instruction,
                                      hbmOff->dynamicVal, step.dstOffset,
                                      step.size);
            }
          } else {
            if (hbmOff->isStatic) {
              emitDataMovement(builder, loc, step.instruction, step.srcOffset,
                               hbmOff->staticVal, step.size);
            } else {
              emitStoreDynamic(builder, loc, step.instruction, step.srcOffset,
                               hbmOff->dynamicVal, step.size);
            }
          }
          continue;
        }

        emitDataMovement(builder, loc, step.instruction, step.srcOffset,
                         step.dstOffset, step.size);
      }
      return success();
    };

    // Phase 2: For multi-buffer + reduction, replay accumulator init plans.
    if (layout.needsDataMovement && !reductionLoops.empty()) {
      unsigned numDst = defineOp.getDestinations().size();
      unsigned numSrc = defineOp.getSources().size();
      for (unsigned dstIdx = 0; dstIdx < numDst; ++dstIdx) {
        auto *initPlan = findAccumulatorInitPlan(layout, nodeIdx, dstIdx);
        if (!initPlan)
          continue;
        if (failed(emitMovementSteps(initPlan->steps, numSrc + dstIdx,
                                     initPlan->hbmValueId,
                                     initPlan->hbmTransforms)))
          return failure();
      }
    }

    // Phase 3: Create reduction loops
    for (auto &ld : reductionLoops) {
      Value lb = arith::ConstantIndexOp::create(builder, loc, 0);
      Value ub = arith::ConstantIndexOp::create(builder, loc, ld.tileFactor);
      Value step = arith::ConstantIndexOp::create(builder, loc, 1);
      auto forOp = scf::ForOp::create(builder, loc, lb, ub, step);
      forOps.push_back(forOp);
      loopIVs[ld.dimIdx] = forOp.getInductionVar();
      builder.setInsertionPointToStart(forOp.getBody());
    }

    // ================================================================
    // Iteration 2: Data movement path
    // ================================================================
    if (layout.needsDataMovement) {
      unsigned numSrc = defineOp.getSources().size();
      unsigned numDst = defineOp.getDestinations().size();

      // --- Replay planned input movements ---
      for (unsigned srcIdx = 0; srcIdx < numSrc; ++srcIdx) {
        auto *movementPlan = findInputMovementPlan(layout, nodeIdx, srcIdx);
        if (!movementPlan)
          continue;
        if (failed(emitMovementSteps(movementPlan->steps, srcIdx,
                                     movementPlan->hbmValueId,
                                     movementPlan->hbmTransforms)))
          return failure();
      }

      // --- Emit compute instruction with planned operand residences ---
      {
        auto operandToParam = mapOperandsToOffsetParams(defineOp);
        DenseMap<unsigned, unsigned> paramToOperand;
        for (auto &[opIdx, paramIdx] : operandToParam)
          paramToOperand[paramIdx] = opIdx;

        unsigned numAddrParams = defineOp.getAccessBlock().getNumArguments();
        SmallVector<int64_t> staticAddrParams(numAddrParams, 0);
        SmallVector<Value> dynamicAddrParams;

        for (unsigned p = 0; p < numAddrParams; ++p) {
          auto kindIt = tm.paramKinds.find(p);
          AddrParamKind kind = (kindIt != tm.paramKinds.end())
                                   ? kindIt->second
                                   : AddrParamKind::Offset;

          if (kind == AddrParamKind::Shape) {
            auto solvedIt = tm.tiling.solvedParams.find(p);
            if (solvedIt != tm.tiling.solvedParams.end()) {
              staticAddrParams[p] = solvedIt->second;
            } else {
              return sourceOp->emitError()
                     << "shape param p" << p << " not solved";
            }
            continue;
          }

          // Offset param: use scratchpad slot offset
          auto opIt = paramToOperand.find(p);
          if (opIt == paramToOperand.end())
            return sourceOp->emitError()
                   << "offset param p" << p << " not mapped to any operand";
          unsigned operandIdx = opIt->second;
          auto *residence = getOperandResidence(layout, nodeIdx, operandIdx);
          if (!residence)
            return sourceOp->emitError()
                   << "missing planned residence for operand " << operandIdx;

          if (residence->bufferName == layout.bufferName) {
            auto hbmAccess = getHBMAccessForOperand(node, defineOp, operandIdx);
            auto hbmOff = computeHBMOffset(node, sourceLinalgOp, defineOp,
                                           operandIdx, hbmAccess.valueId,
                                           hbmAccess.transforms, tm, loopIVs);
            if (failed(hbmOff))
              return failure();
            if (hbmOff->isStatic) {
              staticAddrParams[p] = hbmOff->staticVal;
            } else {
              staticAddrParams[p] = ShapedType::kDynamic;
              dynamicAddrParams.push_back(hbmOff->dynamicVal);
            }
          } else {
            staticAddrParams[p] = residence->offset;
          }
        }

        EmitOp::create(builder, loc,
                       FlatSymbolRefAttr::get(ctx, defineOp.getSymName()),
                       dynamicAddrParams, ValueRange{},
                       DenseI64ArrayAttr::get(ctx, staticAddrParams),
                       DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
        LLVM_DEBUG(llvm::dbgs()
                   << "  Emitted compute @" << defineOp.getSymName() << "\n");
      }

      // --- Emit stores: for each dst operand not in HBM ---
      // Stores go outside reduction loops but inside parallel loops.
      // If we have reduction loops, move insertion point after them.
      if (!reductionLoops.empty()) {
        // Move past the outermost reduction loop
        auto &outermostReduction = forOps[parallelLoops.size()];
        builder.setInsertionPointAfter(outermostReduction);
      }

      for (unsigned dstIdx = 0; dstIdx < numDst; ++dstIdx) {
        auto *movementPlan = findOutputMovementPlan(layout, nodeIdx, dstIdx);
        if (!movementPlan)
          continue;
        if (failed(emitMovementSteps(movementPlan->steps, numSrc + dstIdx,
                                     movementPlan->hbmValueId,
                                     movementPlan->hbmTransforms)))
          return failure();
      }

      LLVM_DEBUG(llvm::dbgs() << "  Done with data movement emission\n");
      continue; // skip Iteration 1 path below
    }

    // ================================================================
    // Iteration 1: Direct-buffer path (all operands in same buffer)
    // ================================================================
    {
      auto operandToParam = mapOperandsToOffsetParams(defineOp);
      DenseMap<unsigned, unsigned> paramToOperand;
      for (auto &[opIdx, paramIdx] : operandToParam)
        paramToOperand[paramIdx] = opIdx;

      unsigned numAddrParams = defineOp.getAccessBlock().getNumArguments();
      SmallVector<int64_t> staticAddrParams(numAddrParams, 0);
      SmallVector<Value> dynamicAddrParams;

      for (unsigned p = 0; p < numAddrParams; ++p) {
        auto kindIt = tm.paramKinds.find(p);
        AddrParamKind kind = (kindIt != tm.paramKinds.end())
                                 ? kindIt->second
                                 : AddrParamKind::Offset;

        if (kind == AddrParamKind::Shape) {
          auto solvedIt = tm.tiling.solvedParams.find(p);
          if (solvedIt != tm.tiling.solvedParams.end()) {
            staticAddrParams[p] = solvedIt->second;
            LLVM_DEBUG(llvm::dbgs() << "    p" << p << " = " << solvedIt->second
                                    << " (shape)\n");
          } else {
            return sourceOp->emitError()
                   << "shape param p" << p << " not solved";
          }
          continue;
        }

        // Offset param: compute flat buffer offset from loop IVs
        auto opIt = paramToOperand.find(p);
        if (opIt == paramToOperand.end())
          return sourceOp->emitError()
                 << "offset param p" << p
                 << " not mapped to any instruction operand";
        unsigned operandIdx = opIt->second;

        auto hbmAccess = getHBMAccessForOperand(node, defineOp, operandIdx);
        auto hbmOff = computeHBMOffset(node, sourceLinalgOp, defineOp,
                                       operandIdx, hbmAccess.valueId,
                                       hbmAccess.transforms, tm, loopIVs);
        if (failed(hbmOff))
          return failure();
        if (hbmOff->isStatic) {
          staticAddrParams[p] = hbmOff->staticVal;
          LLVM_DEBUG(llvm::dbgs()
                     << "    p" << p << " = " << hbmOff->staticVal
                     << " (offset, static, operand " << operandIdx << ")\n");
        } else {
          staticAddrParams[p] = ShapedType::kDynamic;
          dynamicAddrParams.push_back(hbmOff->dynamicVal);
          LLVM_DEBUG(llvm::dbgs()
                     << "    p" << p << " = <dynamic> (offset, operand "
                     << operandIdx << ")\n");
        }
      }

      EmitOp::create(builder, loc,
                     FlatSymbolRefAttr::get(ctx, defineOp.getSymName()),
                     dynamicAddrParams, ValueRange{},
                     DenseI64ArrayAttr::get(ctx, staticAddrParams),
                     DenseI64ArrayAttr::get(ctx, SmallVector<int64_t>{}));
      LLVM_DEBUG(llvm::dbgs()
                 << "  Emitted act.emit @" << defineOp.getSymName() << "\n");
    }
  }

  // --- Cleanup: erase source ops and transform function signature ---
  // First, replace the return op with a void return
  func::ReturnOp oldReturn = nullptr;
  funcOp.walk([&](func::ReturnOp ret) { oldReturn = ret; });

  if (oldReturn) {
    builder.setInsertionPoint(oldReturn);
    func::ReturnOp::create(builder, oldReturn.getLoc());
    oldReturn.erase();
  }

  // Erase compute ops (in reverse order to handle dependencies)
  for (auto it = plan.nodes.rbegin(); it != plan.nodes.rend(); ++it) {
    it->sourceOp->dropAllUses();
    it->sourceOp->erase();
  }

  // Iteratively erase dead infrastructure ops until nothing more can be removed
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
    funcOp.walk([&](Operation *op) {
      if (op->getParentOfType<DefineOp>())
        return;
      if (op->use_empty() &&
          isa<linalg::FillOp, tensor::EmptyOp, arith::ConstantOp,
              linalg::TransposeOp, tensor::ExpandShapeOp,
              tensor::CollapseShapeOp, tensor::ExtractSliceOp,
              tensor::InsertSliceOp>(op)) {
        // Don't erase index constants we just created
        if (auto cst = dyn_cast<arith::ConstantOp>(op))
          if (cst.getType().isIndex())
            return;
        toErase.push_back(op);
      }
    });
    for (auto *op : toErase) {
      op->erase();
      changed = true;
    }
  }

  // Update function signature: remove tensor args and returns
  Block &entryBlock = funcOp.getBody().front();
  // Drop uses of block args (should be dead by now)
  for (auto arg : entryBlock.getArguments())
    arg.dropAllUses();
  // Erase block args in reverse order
  for (unsigned i = entryBlock.getNumArguments(); i > 0; --i)
    entryBlock.eraseArgument(i - 1);
  // Update function type
  funcOp.setType(FunctionType::get(ctx, {}, {}));

  return success();
}

//===----------------------------------------------------------------------===//
// Top-level: runCodeEmission
//===----------------------------------------------------------------------===//

LogicalResult
act::runCodeEmission(ModuleOp module,
                     ArrayRef<TiledMatchCandidate> tiledMatches,
                     ArrayRef<EdgeLayoutAnnotation> layoutAnnotations) {
  LLVM_DEBUG(llvm::dbgs() << "\n=== Code Emission (Stage 3) ===\n");

  auto result = success();
  module.walk([&](func::FuncOp funcOp) {
    // Skip functions inside act.define
    if (funcOp->getParentOfType<DefineOp>())
      return;
    // Skip functions that have no compute ops
    bool hasCompute = false;
    funcOp.walk([&](Operation *op) {
      if (isComputeOp(op) && !op->getParentOfType<DefineOp>())
        hasCompute = true;
    });
    if (!hasCompute)
      return;

    LLVM_DEBUG(llvm::dbgs()
               << "\nProcessing function: " << funcOp.getSymName() << "\n");

    // Phase 3a: Build selected logical plan
    auto plan = buildLogicalPlan(funcOp, tiledMatches, layoutAnnotations);
    if (failed(plan)) {
      result = failure();
      return;
    }

    // Phase 3b: Resource planning
    auto layout = buildResourcePlan(funcOp, *plan, module);
    if (failed(layout)) {
      result = failure();
      return;
    }

    // Phase 3c: Code emission
    if (failed(emitCode(funcOp, *plan, *layout))) {
      result = failure();
      return;
    }
  });

  return result;
}
