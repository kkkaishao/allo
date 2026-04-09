#include "act/Conversion/Passes.h"
#include "act/IR/ActOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "convert-act-to-canonical-form"

namespace mlir::act {
#define GEN_PASS_DEF_CONVERTACTTOCANONICALFORMPASS
#include "act/Conversion/Passes.h.inc"
} // namespace mlir::act

using namespace mlir;
using namespace mlir::act;

namespace {
struct ActConversionContext {
  SymbolTable &symbolTable;
  SmallVector<memref::GlobalOp> globalBuffers;
  DenseMap<func::FuncOp, DenseMap<StringAttr, Value>> funcToBufferMap;
  bool enableTensor = false;
  explicit ActConversionContext(SymbolTable &symbolTable)
      : symbolTable(symbolTable) {}

  void dumpMap() const {
    llvm::dbgs() << "Function to buffer map:\n";
    for (auto &[func, bufferMap] : funcToBufferMap) {
      llvm::dbgs() << "Function: " << "\n";
      for (const auto &[symName, handle] : bufferMap) {
        llvm::dbgs() << "  " << symName.getValue() << " -> " << handle << "\n";
      }
    }
  }
};
} // namespace

namespace {
struct ConvertDeclareBufferOpPattern
    : public OpConversionPattern<DeclareBufferOp> {
  ConvertDeclareBufferOpPattern(MLIRContext *ctx, ActConversionContext &context)
      : OpConversionPattern(ctx), context(context) {}

  LogicalResult
  matchAndRewrite(DeclareBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    BufferTypeInterface bufferTy = op.getBufferType();
    SmallVector<int64_t, 4> shape;
    if (op.getSize() != 1)
      shape.push_back(op.getSize());
    llvm::append_range(shape, bufferTy.getShape());
    auto memrefTy = MemRefType::get(shape, bufferTy.getElementType());
    Location loc = op.getLoc();
    StringAttr symName = op.getSymNameAttr();
    auto global = memref::GlobalOp::create(
        rewriter, loc, symName, StringAttr(), memrefTy,
        /*initValue*/ Attribute(),
        /*constant*/ false, /*alignment*/ IntegerAttr());
    context.globalBuffers.push_back(global);
    rewriter.eraseOp(op);
    return success();
  }

private:
  ActConversionContext &context;
};
} // namespace

static void createLocalHandles(RewriterBase &rewriter, func::FuncOp func,
                               ActConversionContext &context) {
  rewriter.setInsertionPointToStart(&func.getBody().front());
  auto &bufferMap = context.funcToBufferMap[func];
  for (auto global : context.globalBuffers) {
    Value handle = memref::GetGlobalOp::create(
        rewriter, func.getLoc(), global.getType(), global.getSymName());
    if (context.enableTensor) {
      auto memrefTy = global.getType();
      auto tensorTy =
          RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
      handle = bufferization::ToTensorOp::create(
          rewriter, func.getLoc(), tensorTy, handle, /*restrict=*/true,
          /*writable*/ true);
    }
    bufferMap[global.getSymNameAttr()] = handle;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Created local handles for function " << func.getName()
                 << ":\n";
    for (auto &[symName, handle] : bufferMap) {
      llvm::dbgs() << "  " << symName.getValue() << " -> " << handle << "\n";
    }
  });
}

static bool isCastLikeOp(Operation *op) {
  return isa<arith::TruncFOp, arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
             arith::ExtUIOp, arith::UIToFPOp, arith::FPToSIOp, arith::SIToFPOp,
             arith::FPToUIOp, arith::IndexCastOp, arith::BitcastOp>(op);
}

namespace {
struct SemanticsBuilder {
  SemanticsBuilder(RewriterBase &b, Location loc, IRMapping &mapping)
      : b(b), loc(loc), mapping(mapping) {}

  LogicalResult build(Block &block) const {
    for (Operation &op : block.without_terminator()) {
      SmallVector<Value, 4> newOperands;
      for (Value operand : op.getOperands())
        newOperands.push_back(mapping.lookupOrDefault(operand));
      if (isa<linalg::LinalgOp, linalg::SoftmaxOp, tensor::ExpandShapeOp>(op)) {
        if (failed(buildReifiedOp(cast<ReifyRankedShapedTypeOpInterface>(op),
                                  newOperands)))
          return failure();
        continue;
      }
      if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        if (failed(buildCollapseShapeOp(collapseOp, newOperands)))
          return failure();
        continue;
      }
      if (isa<arith::ConstantOp>(op)) {
        b.clone(op, mapping);
        continue;
      }
      if (isa<arith::CmpFOp, arith::CmpIOp>(op)) {
        Type operandTy = op.getOperand(0).getType();
        Type resTy;
        if (auto shaped = dyn_cast<ShapedType>(operandTy))
          resTy = static_cast<Type>(shaped.clone(b.getI1Type()));
        else
          resTy = b.getI1Type();
        buildTrivialOp(&op, newOperands, resTy, op.getAttrs());
        continue;
      }
      if (isa<arith::ArithDialect, math::MathDialect>(op.getDialect())) {
        if (failed(buildArithLikeOp(&op, newOperands)))
          return failure();
        continue;
      }
      return op.emitError()
             << "unsupported operation in semantics block " << op.getName();
    }
    return success();
  }

private:
  RewriterBase &b;
  Location loc;
  IRMapping &mapping;

  void buildTrivialOp(Operation *op, ArrayRef<Value> newOperands,
                      ArrayRef<Type> newTypes,
                      ArrayRef<NamedAttribute> newAttrs) const {
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(newTypes);
    state.addAttributes(newAttrs);
    Operation *newOp = b.create(state);
    mapping.map(op, newOp);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
  }

  LogicalResult buildReifiedOp(ReifyRankedShapedTypeOpInterface op,
                               ArrayRef<Value> newOperands) const {
    // check if rank is matched
    for (auto [oldTy, newOperand] :
         llvm::zip(op->getOperandTypes(), newOperands)) {
      auto oldShaped = dyn_cast<ShapedType>(oldTy);
      auto newShaped = dyn_cast<ShapedType>(newOperand.getType());
      if (oldShaped && newShaped && oldShaped.getRank() != newShaped.getRank())
        return op.emitError()
               << "rank mismatch between original operand and new operand. "
               << "original: " << oldShaped << ", inferred: " << newShaped;
    }
    // clone a dummy op to analyze the result types
    auto reifyOp = cast<ReifyRankedShapedTypeOpInterface>(b.clone(*op));
    b.modifyOpInPlace(reifyOp, [&]() { reifyOp->setOperands(newOperands); });
    ReifiedRankedShapedTypeDims shapes;
    if (failed(reifyOp.reifyResultShapes(b, shapes))) {
      b.eraseOp(reifyOp);
      return op.emitError() << "failed to reify result shapes";
    }
    b.eraseOp(reifyOp);

    SmallVector<Type, 4> resultTys;
    for (auto [shape, oldType] : llvm::zip(shapes, op->getResultTypes())) {
      auto oldTensor = dyn_cast<RankedTensorType>(oldType);
      if (!oldTensor)
        return op.emitError()
               << "expected ranked tensor result type, got " << oldType;
      SmallVector<int64_t, 4> dims;
      for (OpFoldResult dim : shape) {
        if (auto attr = dyn_cast<Attribute>(dim)) {
          dims.push_back(cast<IntegerAttr>(attr).getInt());
          continue;
        }
        dims.push_back(ShapedType::kDynamic);
      }
      resultTys.push_back(oldTensor.clone(dims));
    }

    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(resultTys);
    state.addAttributes(op->getAttrs());
    if (op->getNumRegions() > 0)
      state.addRegion();
    Operation *newOp = b.create(state);
    mapping.map(op.getOperation(), newOp);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    // clone region
    if (op->getNumRegions() == 0)
      return success();
    if (op->getNumRegions() > 1)
      return op.emitError()
             << "unexpected number of regions, expected at most 1";
    Region &newRegion = newOp->getRegion(0);
    Region &oldRegion = op->getRegion(0);
    b.cloneRegionBefore(oldRegion, newRegion, newRegion.end(), mapping);
    return success();
  }

  LogicalResult buildCollapseShapeOp(tensor::CollapseShapeOp op,
                                     ArrayRef<Value> newOperands) const {
    if (newOperands.size() != 1)
      return op.emitError()
             << "unexpected number of operands for tensor.collapse_shape";
    auto srcTy = dyn_cast<RankedTensorType>(newOperands.front().getType());
    if (!srcTy)
      return op.emitError() << "expected ranked tensor source type, got "
                            << newOperands.front().getType();

    RankedTensorType resultTy = tensor::CollapseShapeOp::inferCollapsedType(
        srcTy, op.getReassociationIndices());
    if (!resultTy)
      return op.emitError() << "failed to infer collapsed result type";

    buildTrivialOp(op, newOperands, resultTy, op->getAttrs());
    return success();
  }

  LogicalResult buildArithLikeOp(Operation *op,
                                 ArrayRef<Value> newOperands) const {
    if (op->getNumResults() != 1 ||
        (op->getNumOperands() != 1 && op->getNumOperands() != 2))
      return op->emitError()
             << "unexpected number of operands/results for arithmetic op";
    Type operandTy = newOperands.front().getType();
    Type resultTy = op->getResult(0).getType();
    if (isa<ShapedType>(operandTy)) {
      return op->emitError()
             << "use linalg operations for shaped types, arith operations "
                "should only be used for scalar types, got "
             << operandTy;
    }
    Type newTy = isCastLikeOp(op) ? resultTy : operandTy;
    buildTrivialOp(op, newOperands, newTy, op->getAttrs());
    return success();
  }
};
} // namespace

namespace {
struct ConvertEmitOpPattern : public OpConversionPattern<EmitOp> {
  ConvertEmitOpPattern(MLIRContext *ctx, ActConversionContext &context)
      : OpConversionPattern(ctx), context(context) {}

  LogicalResult
  matchAndRewrite(EmitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto declOp = context.symbolTable.lookup<DefineOp>(op.getInstruction());
    if (!declOp)
      return op.emitError() << "referenced instruction '@"
                            << op.getInstruction() << "' not found";

    Block &accessBlock = declOp.getAccessBlock();
    Block &semBlock = declOp.getSemanticsBlock();
    auto &bufferMap =
        context.funcToBufferMap[op->getParentOfType<func::FuncOp>()];
    rewriter.setInsertionPoint(op);
    auto accessOps = getBufferAccessOps(rewriter, accessBlock, op, adaptor);
    auto slices = materializeSliceOps(rewriter, op.getLoc(), bufferMap, declOp,
                                      accessOps);
    auto computeArgs = generateExtraComputeArgs(rewriter, op, declOp);
    slices.append(computeArgs);
    // map block arguments to slices
    IRMapping mapping;
    for (auto [buffer, slice] :
         llvm::zip_equal(semBlock.getArguments(), slices))
      mapping.map(buffer, slice);

    // instantiate semantics block
    SemanticsBuilder builder(rewriter, op.getLoc(), mapping);
    if (failed(builder.build(semBlock)))
      return failure();
    // materialize write back ops
    if (context.enableTensor) {
      auto yieldOp = semBlock.getTerminator();
      SmallVector<Value, 4> valuesToWrite;
      for (Value operand : yieldOp->getOperands())
        valuesToWrite.push_back(mapping.lookupOrDefault(operand));
      materializeWriteBackOps(rewriter, op.getLoc(), bufferMap, declOp,
                              valuesToWrite, accessOps);
    }
    for (auto accessOp : accessOps)
      rewriter.eraseOp(accessOp);
    rewriter.eraseOp(op);
    return success();
  }

private:
  ActConversionContext &context;

  static SmallVector<BufferAccessOpInterface, 4>
  getBufferAccessOps(RewriterBase &b, Block &accessBlock, EmitOp op,
                     OpAdaptor adaptor) {
    SmallVector<BufferAccessOpInterface, 4> accessOps;
    // Step 1: build param mapping
    IRMapping mapping;
    unsigned dynamicIdx = 0;
    auto dynamicParams = adaptor.getAddrParams();
    for (auto [blockArg, staticParam] : llvm::zip_equal(
             accessBlock.getArguments(), op.getStaticAddrParams())) {
      if (ShapedType::isDynamic(staticParam))
        mapping.map(blockArg, dynamicParams[dynamicIdx++]);
      else {
        auto cst = arith::ConstantIndexOp::create(b, op.getLoc(), staticParam);
        mapping.map(blockArg, cst);
      }
    }
    // Step 2: clone access block, tracking the yield terminator
    Operation *yieldClone = nullptr;
    for (Operation &nested : accessBlock.getOperations()) {
      Operation *cloned = b.clone(nested, mapping);
      if (isa<YieldOp>(cloned))
        yieldClone = cloned;
    }
    // Step 3: collect buffer access ops from yield operands
    assert(yieldClone && "access block must have a yield terminator");
    for (auto operand : yieldClone->getOperands()) {
      auto accessOp = operand.getDefiningOp<BufferAccessOpInterface>();
      assert(accessOp &&
             "terminator operands should be defined by buffer access ops");
      accessOps.push_back(accessOp);
    }
    b.eraseOp(yieldClone);
    return accessOps;
  }

  static SmallVector<Value, 4>
  generateExtraComputeArgs(RewriterBase &b, EmitOp op, DefineOp declOp) {
    unsigned dynamicIdx = 0;
    SmallVector<Value, 4> computeArgs;
    auto dynamicParams = op.getComputeParams();
    for (auto [staticParam, blockArg] :
         llvm::zip(op.getStaticComputeParams(), declOp.getExtraComputeArgs())) {
      if (ShapedType::isDynamic(staticParam))
        computeArgs.push_back(dynamicParams[dynamicIdx++]);
      else {
        Value cst;
        if (isa<IntegerType>(blockArg.getType()))
          cst = arith::ConstantIntOp::create(b, op.getLoc(), blockArg.getType(),
                                             staticParam);
        else if (isa<IndexType>(blockArg.getType()))
          cst = arith::ConstantIndexOp::create(b, op.getLoc(), staticParam);
        else
          llvm_unreachable("unsupported compute parameter type");
        computeArgs.push_back(cst);
      }
    }
    return computeArgs;
  }

  SmallVector<Value, 4> materializeSliceOps(
      RewriterBase &b, Location loc,
      llvm::DenseMap<StringAttr, Value> &bufferMap, DefineOp declOp,
      SmallVectorImpl<BufferAccessOpInterface> &accessOps) const {
    SmallVector<StringAttr, 4> bufferNames;
    for (auto buffer : declOp.getSources().getAsRange<FlatSymbolRefAttr>())
      bufferNames.push_back(buffer.getAttr());
    for (auto buffer : declOp.getDestinations().getAsRange<FlatSymbolRefAttr>())
      bufferNames.push_back(buffer.getAttr());

    assert(accessOps.size() == bufferNames.size());
    SmallVector<Value, 4> slices;
    slices.reserve(accessOps.size());
    for (auto [bufferName, accessOp] :
         llvm::zip_equal(bufferNames, accessOps)) {
      auto it = bufferMap.find(bufferName);
      assert(it != bufferMap.end() && "buffer not found in local handle map");
      Value handle = it->second;
      Value slice = accessOp.materialize(b, loc, handle, context.enableTensor);
      slices.push_back(slice);
    }
    return slices;
  }

  void materializeWriteBackOps(
      RewriterBase &b, Location loc,
      llvm::DenseMap<StringAttr, Value> &bufferMap, DefineOp declOp,
      SmallVectorImpl<Value> &valuesToWrite,
      SmallVectorImpl<BufferAccessOpInterface> &accessOps) const {
    SmallVector<StringAttr, 4> bufferNames;
    for (auto buffer : declOp.getDestinations().getAsRange<FlatSymbolRefAttr>())
      bufferNames.push_back(buffer.getAttr());

    unsigned nSrc = declOp.getSources().size();
    for (unsigned i = 0; i < bufferNames.size(); ++i) {
      Value value = valuesToWrite[i];
      Value handle = bufferMap[bufferNames[i]];
      BufferAccessOpInterface accessOp = accessOps[i + nSrc];
      Value slice = accessOp.materialize(b, loc, value, handle);
      // update buffer map
      bufferMap[bufferNames[i]] = slice;
    }
  }
};
} // namespace

static void writeBackLocalHandles(RewriterBase &b, func::FuncOp func,
                                  ActConversionContext &context) {
  auto &bufferMap = context.funcToBufferMap[func];
  for (auto &[symName, handle] : bufferMap) {
    if (auto defOp = handle.getDefiningOp<bufferization::ToTensorOp>()) {
      if (handle.use_empty()) {
        defOp.erase();
        continue;
      }
    }
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        context.symbolTable.getOp(), symName);
    assert(global && "global should exist in symbol table");
    b.setInsertionPoint(func.getBody().front().getTerminator());
    auto get = memref::GetGlobalOp::create(b, func.getLoc(), global.getType(),
                                           global.getSymName());
    bufferization::MaterializeInDestinationOp::create(
        b, func.getLoc(), TypeRange(), handle, get, /*restrict=*/true,
        /*writable=*/true);
  }
}

namespace {
struct ConvertActToCanonicalFormPass
    : public act::impl::ConvertActToCanonicalFormPassBase<
          ConvertActToCanonicalFormPass> {

  ConvertActToCanonicalFormPass() = default;
  explicit ConvertActToCanonicalFormPass(
      const ConvertActToCanonicalFormPassOptions &options) {
    enableTensor = options.enableTensor;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    SymbolTable symbolTable(getOperation());
    ActConversionContext context(symbolTable);
    context.enableTensor = enableTensor;

    // Step 1: convert resources to global buffers
    RewritePatternSet patterns(ctx);
    target.addIllegalOp<DeclareBufferOp>();
    target.addLegalOp<memref::GlobalOp>();
    patterns.add<ConvertDeclareBufferOpPattern>(ctx, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
    // Step 2.1: create local handles for global buffers
    IRRewriter rewriter(ctx);
    getOperation()->walk([&](func::FuncOp func) {
      createLocalHandles(rewriter, func, context);
    });
    // Step 2.2: instantiate EmitOp
    RewritePatternSet emitPatterns(ctx);
    //===== primitive operation set starts ====//
    target.addLegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp,
                      tensor::ExpandShapeOp, tensor::CollapseShapeOp,
                      tensor::DimOp, memref::SubViewOp>();
    target.addLegalOp<math::Exp2Op, math::Log2Op, math::ExpOp, math::LogOp,
                      math::AbsFOp, math::AbsIOp, math::FloorOp, math::SqrtOp,
                      math::RsqrtOp, math::CeilOp, math::TruncOp>();
    target.addLegalOp<linalg::GenericOp, linalg::YieldOp, linalg::MapOp,
                      linalg::ReduceOp, linalg::TransposeOp, linalg::FillOp,
                      linalg::ContractOp, linalg::SoftmaxOp>();
    target.addLegalDialect<arith::ArithDialect>();
    //===== primitive operation set ends ====//
    target.addIllegalOp<EmitOp>();
    emitPatterns.add<ConvertEmitOpPattern>(ctx, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(emitPatterns))))
      signalPassFailure();

    // Step 3: write back to global buffers
    if (enableTensor) {
      getOperation()->walk([&](func::FuncOp func) {
        writeBackLocalHandles(rewriter, func, context);
      });
    }
    // Step 4: clean up instruction definitions
    SmallVector<DefineOp> defineOps;
    getOperation()->walk([&](DefineOp op) { defineOps.push_back(op); });
    for (auto op : defineOps)
      op.erase();
    LLVM_DEBUG(getOperation()->dumpPretty());
  }

private:
  bool enableTensor = true;
};
} // namespace
