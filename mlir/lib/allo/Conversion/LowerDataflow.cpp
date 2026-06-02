#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LOWERDATAFLOWPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {
constexpr StringLiteral kCreate = "allo_sim_stream_create";
constexpr StringLiteral kWrite = "allo_sim_stream_write";
constexpr StringLiteral kRead = "allo_sim_stream_read";
constexpr StringLiteral kWriteMem = "allo_sim_stream_write_mem";
constexpr StringLiteral kReadMem = "allo_sim_stream_read_mem";
constexpr StringLiteral kDestroy = "allo_sim_stream_destroy";
constexpr StringLiteral kCreateAttr = "allo.dataflow.stream_create";
} // namespace

static bool isStreamType(Type type) { return isa<StreamType>(type); }

static bool hasStreamType(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), isStreamType) ||
      llvm::any_of(op->getResultTypes(), isStreamType))
    return true;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (llvm::any_of(block.getArgumentTypes(), isStreamType))
        return true;
    }
  }
  return false;
}

static bool isScalarPayload(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() <= 64;
  if (isa<IndexType>(type))
    return true;
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() <= 64;
  return false;
}

static unsigned payloadWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (isa<IndexType>(type))
    return 64;
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  llvm_unreachable("expected scalar stream payload");
}

static bool isBlockElementPayload(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

static int64_t product(ArrayRef<int64_t> shape) {
  int64_t lanes = 1;
  for (int64_t dim : shape) {
    assert(dim >= 0 && "stream shape dimensions are verified non-negative");
    lanes *= dim;
  }
  return lanes;
}

static void declareRuntimeFunc(ModuleOp module, StringRef name,
                               FunctionType type) {
  if (module.lookupSymbol<func::FuncOp>(name))
    return;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto func = func::FuncOp::create(builder, module.getLoc(), name, type);
  func.setPrivate();
}

static void declareRuntimeFuncs(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  declareRuntimeFunc(module, kCreate,
                     FunctionType::get(ctx, {i64, i64, i64}, {i64}));
  declareRuntimeFunc(module, kWrite,
                     FunctionType::get(ctx, {i64, i64, i64}, {}));
  declareRuntimeFunc(module, kRead, FunctionType::get(ctx, {i64, i64}, {i64}));
  declareRuntimeFunc(module, kWriteMem,
                     FunctionType::get(ctx, {i64, i64, i64}, {}));
  declareRuntimeFunc(module, kReadMem,
                     FunctionType::get(ctx, {i64, i64, i64}, {}));
  declareRuntimeFunc(module, kDestroy, FunctionType::get(ctx, {i64}, {}));
}

static Value makeI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 64);
}

static FailureOr<int64_t> memrefPayloadBytes(MemRefType type) {
  if (!type.hasStaticShape() || !isBlockElementPayload(type.getElementType()))
    return failure();

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0)
    return failure();

  int64_t expectedStride = 1;
  for (int64_t i = static_cast<int64_t>(type.getRank()) - 1; i >= 0; --i) {
    if (strides[i] != expectedStride)
      return failure();
    expectedStride *= type.getDimSize(i);
  }
  return type.getNumElements() *
         ((payloadWidth(type.getElementType()) + 7) / 8);
}

static FailureOr<int64_t> streamPayloadBytes(Type type) {
  if (isScalarPayload(type))
    return (payloadWidth(type) + 7) / 8;
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefPayloadBytes(memrefType);
  return failure();
}

static Value alignedPointerAsI64(OpBuilder &builder, Location loc,
                                 Value memref) {
  auto i64 = IntegerType::get(builder.getContext(), 64);
  Value ptr =
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, memref);
  return arith::IndexCastOp::create(builder, loc, i64, ptr);
}

static Value linearizeLane(OpBuilder &builder, Location loc,
                           StreamType streamType, ValueRange indices) {
  auto i64 = IntegerType::get(builder.getContext(), 64);
  if (indices.empty())
    return makeI64Constant(builder, loc, 0);

  Value flat = indices.front();
  ArrayRef<int64_t> shape = streamType.getShape();
  assert(shape.size() == indices.size());
  for (size_t i = 1; i < indices.size(); ++i) {
    auto dim = arith::ConstantIndexOp::create(builder, loc, shape[i]);
    flat = arith::MulIOp::create(builder, loc, flat, dim);
    flat = arith::AddIOp::create(builder, loc, flat, indices[i]);
  }
  return arith::IndexCastOp::create(builder, loc, i64, flat);
}

static FailureOr<Value> packScalar(OpBuilder &builder, Location loc,
                                   Value value) {
  auto i64 = IntegerType::get(builder.getContext(), 64);
  Type type = value.getType();
  if (!isScalarPayload(type))
    return failure();

  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 64)
      return value;
    return arith::ExtUIOp::create(builder, loc, i64, value).getResult();
  }
  if (isa<IndexType>(type))
    return arith::IndexCastOp::create(builder, loc, i64, value).getResult();

  auto floatType = cast<FloatType>(type);
  auto intType = IntegerType::get(builder.getContext(), floatType.getWidth());
  Value bits = arith::BitcastOp::create(builder, loc, intType, value);
  if (floatType.getWidth() == 64)
    return bits;
  return arith::ExtUIOp::create(builder, loc, i64, bits).getResult();
}

static FailureOr<Value> unpackScalar(OpBuilder &builder, Location loc,
                                     Type type, Value bits) {
  if (!isScalarPayload(type))
    return failure();

  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 64)
      return bits;
    return arith::TruncIOp::create(builder, loc, intType, bits).getResult();
  }
  if (isa<IndexType>(type))
    return arith::IndexCastOp::create(builder, loc, type, bits).getResult();

  auto floatType = cast<FloatType>(type);
  auto intType = IntegerType::get(builder.getContext(), floatType.getWidth());
  Value narrowed = bits;
  if (floatType.getWidth() != 64)
    narrowed = arith::TruncIOp::create(builder, loc, intType, bits);
  return arith::BitcastOp::create(builder, loc, type, narrowed).getResult();
}

static bool isStreamInvoke(InvokeOp op) {
  return llvm::any_of(op.getOperands(), [](Value operand) {
    return isa<StreamType>(operand.getType());
  });
}

static LogicalResult wrapDataflowInvokes(OpBuilder &builder, KernelOp kernel,
                                         unsigned numThreads) {
  if (kernel.getBody().empty())
    return success();
  if (!llvm::hasSingleElement(kernel.getBody()))
    return kernel.emitError("lower-dataflow expects single-block kernels");

  Block &block = kernel.getBody().front();
  SmallVector<InvokeOp> invokes;
  for (Operation &op : block) {
    if (auto invoke = dyn_cast<InvokeOp>(op);
        invoke && isStreamInvoke(invoke)) {
      // TODO: can this limitation be lifted?
      if (invoke.getNumResults() != 0)
        return invoke.emitError(
            "stream-connected dataflow invokes must not return values");
      invokes.push_back(invoke);
    }
  }
  if (invokes.size() < 2)
    return success();

  InvokeOp first = invokes.front();
  InvokeOp last = invokes.back();
  bool inGroup = false;
  for (Operation &op : block) {
    if (&op == first)
      inGroup = true;
    if (inGroup && &op != last && !isa<InvokeOp>(op))
      return op.emitError(
          "lower-dataflow expects stream-connected invokes to be contiguous");
    if (inGroup && isa<InvokeOp>(op) && !isStreamInvoke(cast<InvokeOp>(op)))
      return op.emitError("lower-dataflow does not mix stream and non-stream "
                          "invokes in one group");
    if (&op == last)
      break;
  }

  builder.setInsertionPoint(first);
  Location loc = first.getLoc();
  Value numThreadsCst =
      arith::ConstantIntOp::create(builder, loc, invokes.size(), 64);
  auto parallel = omp::ParallelOp::create(
      builder, loc, ValueRange{}, ValueRange{}, Value{}, numThreadsCst,
      ValueRange{}, ArrayAttr(), UnitAttr(), omp::ClauseProcBindKindAttr(),
      omp::ReductionModifierAttr(), ValueRange{}, DenseBoolArrayAttr(),
      ArrayAttr());
  Block &parallelBlock = parallel.getRegion().emplaceBlock();

  builder.setInsertionPointToEnd(&parallelBlock);
  auto sections = omp::SectionsOp::create(
      builder, loc, ValueRange{}, ValueRange{}, UnitAttr(), ValueRange{},
      ArrayAttr(), UnitAttr(), omp::ReductionModifierAttr(), ValueRange{},
      DenseBoolArrayAttr(), ArrayAttr());
  Block &sectionsBlock = sections.getRegion().emplaceBlock();

  builder.setInsertionPointToEnd(&sectionsBlock);
  for (InvokeOp invoke : invokes) {
    OpBuilder::InsertionGuard guard(builder);
    auto section = omp::SectionOp::create(builder, invoke.getLoc());
    Block &sectionBlock = section.getRegion().emplaceBlock();
    builder.setInsertionPointToEnd(&sectionBlock);
    auto terminator = omp::TerminatorOp::create(builder, invoke.getLoc());
    invoke->moveBefore(terminator);
  }
  omp::TerminatorOp::create(builder, loc);
  builder.setInsertionPointAfter(sections);
  omp::TerminatorOp::create(builder, loc);
  return success();
}

static LogicalResult wrapDataflowInvokes(ModuleOp module, unsigned numThreads) {
  OpBuilder builder(module);
  for (auto kernel : module.getOps<KernelOp>()) {
    if (failed(wrapDataflowInvokes(builder, kernel, numThreads)))
      return failure();
  }
  return success();
}

namespace {
struct StreamCreateLowering : public OpConversionPattern<StreamCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StreamCreateOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto streamType = op.getStream().getType();
    Type baseType = streamType.getBaseType();
    FailureOr<int64_t> itemBytes = streamPayloadBytes(baseType);
    if (failed(itemBytes))
      return op.emitOpError(
          "lower-dataflow only supports scalar or static contiguous memref "
          "stream payloads");

    Location loc = op.getLoc();
    auto i64 = IntegerType::get(rewriter.getContext(), 64);
    SmallVector<Value> operands = {
        makeI64Constant(rewriter, loc, product(streamType.getShape())),
        makeI64Constant(rewriter, loc, streamType.getDepth()),
        makeI64Constant(rewriter, loc, *itemBytes),
    };
    auto call =
        func::CallOp::create(rewriter, loc, rewriter.getStringAttr(kCreate),
                             TypeRange{i64}, operands);
    call->setAttr(kCreateAttr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct StreamPutLowering : public OpConversionPattern<StreamPutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StreamPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto streamType = op.getStream().getType();
    Location loc = op.getLoc();
    Value lane = linearizeLane(rewriter, loc, streamType, adaptor.getIndices());

    if (isa<MemRefType>(streamType.getBaseType())) {
      Value ptr = alignedPointerAsI64(rewriter, loc, adaptor.getValue());
      func::CallOp::create(rewriter, loc, rewriter.getStringAttr(kWriteMem),
                           TypeRange{},
                           ValueRange{adaptor.getStream(), lane, ptr});
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<Value> bits = packScalar(rewriter, loc, adaptor.getValue());
    if (failed(bits))
      return op.emitOpError(
          "lower-dataflow only supports scalar or static contiguous memref "
          "stream payloads");
    func::CallOp::create(rewriter, loc, rewriter.getStringAttr(kWrite),
                         TypeRange{},
                         ValueRange{adaptor.getStream(), lane, *bits});
    rewriter.eraseOp(op);
    return success();
  }
};

struct StreamGetLowering : public OpConversionPattern<StreamGetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StreamGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i64 = IntegerType::get(rewriter.getContext(), 64);
    auto streamType = op.getStream().getType();
    Location loc = op.getLoc();
    Value lane = linearizeLane(rewriter, loc, streamType, adaptor.getIndices());

    if (auto memrefType = dyn_cast<MemRefType>(streamType.getBaseType())) {
      auto alloc = memref::AllocOp::create(rewriter, loc, memrefType);
      Value ptr = alignedPointerAsI64(rewriter, loc, alloc.getResult());
      func::CallOp::create(rewriter, loc, rewriter.getStringAttr(kReadMem),
                           TypeRange{},
                           ValueRange{adaptor.getStream(), lane, ptr});
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }

    auto call = func::CallOp::create(
        rewriter, loc, rewriter.getStringAttr(kRead), TypeRange{i64},
        ValueRange{adaptor.getStream(), lane});
    FailureOr<Value> value =
        unpackScalar(rewriter, loc, op.getValue().getType(), call.getResult(0));
    if (failed(value))
      return op.emitOpError(
          "lower-dataflow only supports scalar or static contiguous memref "
          "stream payloads");
    rewriter.replaceOp(op, *value);
    return success();
  }
};

struct InvokeLowering : public OpConversionPattern<InvokeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InvokeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto newOp =
        InvokeOp::create(rewriter, op.getLoc(), resultTypes, op.getCalleeAttr(),
                         adaptor.getOperands(), op.getArgAttrsAttr(),
                         op.getResAttrsAttr(), op.getAsyncAttr());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};
} // namespace

static void insertStreamDestroys(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  module.walk([&](KernelOp kernel) {
    SmallVector<Value> handles;
    kernel.walk([&](func::CallOp call) {
      if (call->hasAttr(kCreateAttr) && call.getNumResults() == 1) {
        handles.push_back(call.getResult(0));
        call->removeAttr(kCreateAttr);
      }
    });
    if (handles.empty())
      return;
    kernel.walk([&](ReturnOp ret) {
      OpBuilder builder(ret);
      for (Value handle : handles)
        func::CallOp::create(builder, ret.getLoc(),
                             builder.getStringAttr(kDestroy), TypeRange{},
                             ValueRange{handle});
    });
  });
}

namespace {
struct LowerDataflowPass
    : public allo::impl::LowerDataflowPassBase<LowerDataflowPass> {
  LowerDataflowPass() = default;
  LowerDataflowPass(const LowerDataflowPassOptions &options)
      : numThreads(options.numThreads) {}
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(wrapDataflowInvokes(module, numThreads))) {
      signalPassFailure();
      return;
    }

    // inject runtime helper functions
    declareRuntimeFuncs(module);

    MLIRContext *ctx = module.getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    // convert stream type to a runtime handle/pointer represented as i64
    auto handleTy = IntegerType::get(ctx, 64);
    converter.addConversion(
        [handleTy](StreamType) -> Type { return handleTy; });

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<KernelOp>(patterns,
                                                               converter);
    patterns.add<StreamCreateLowering, StreamPutLowering, StreamGetLowering,
                 InvokeLowering>(converter, ctx);

    ConversionTarget target(*ctx);
    target.addIllegalOp<StreamCreateOp, StreamPutOp, StreamGetOp>();
    target.addDynamicallyLegalOp<KernelOp>([&](KernelOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<InvokeOp>(
        [&](InvokeOp op) { return converter.isLegal(op.getOperation()); });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return !hasStreamType(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    insertStreamDestroys(module);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, omp::OpenMPDialect>();
  }

private:
  unsigned numThreads = 8;
};
} // namespace
