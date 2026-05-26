#include "allo/Transforms/Passes.h"

#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::allo {
#define GEN_PASS_DEF_MATERIALIZETOPOLOGYPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

static bool isRankedStream(Type type) {
  auto stream = dyn_cast<StreamType>(type);
  return stream && !stream.getShape().empty();
}

namespace {
struct PortInfo {
  unsigned sourceArg;
  SmallVector<int64_t> lane;
  bool scalarPort;
};

struct KernelInfo {
  SmallVector<PortInfo> inputs;
};

using KernelInfoMap = DenseMap<StringAttr, KernelInfo>;
} // namespace

static FailureOr<SmallVector<int64_t>> getStaticLane(Operation *op,
                                                     ValueRange indices) {
  SmallVector<int64_t> lane;
  for (Value index : indices) {
    IntegerAttr::ValueType cst;
    if (!matchPattern(index, m_ConstantInt(&cst)))
      return op->emitError("stream lane index must be static after SPMD "
                           "specialization");
    lane.push_back(cst.getSExtValue());
  }
  return lane;
}

namespace {
struct ScalarizeStreamPorts : public OpRewritePattern<KernelOp> {
  ScalarizeStreamPorts(MLIRContext *ctx, KernelInfoMap &kernelInfos)
      : OpRewritePattern(ctx), kernelInfos(kernelInfos) {}

  LogicalResult matchAndRewrite(KernelOp kernel,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(kernel.getMapping(), [](int64_t x) { return x == 1; }))
      return kernel.emitError(
          "materialize-topology only supports kernels with identity mapping");

    DenseMap<std::pair<BlockArgument, SmallVector<int64_t>>, BlockArgument>
        portMap;
    DenseMap<BlockArgument, PortInfo> portInfo;
    Block &entry = kernel.getBody().front();
    Location loc = kernel.getLoc();

    bool hasRankedStreamArgs = false;
    for (auto arg : entry.getArgumentTypes()) {
      if (auto streamType = dyn_cast<StreamType>(arg);
          streamType && !streamType.getShape().empty()) {
        hasRankedStreamArgs = true;
        break;
      }
    }
    if (!hasRankedStreamArgs)
      return failure(); // nothing to do

    WalkResult walkResult = kernel->walk([&](Operation *op) {
      BlockArgument arg;
      SmallVector<Value, 4> indices;
      if (auto get = dyn_cast<StreamGetOp>(op)) {
        arg = dyn_cast<BlockArgument>(get.getStream());
        llvm::append_range(indices, get.getIndices());
      } else if (auto put = dyn_cast<StreamPutOp>(op)) {
        arg = dyn_cast<BlockArgument>(put.getStream());
        llvm::append_range(indices, put.getIndices());
      } else
        return WalkResult::advance();

      if (!arg || arg.getOwner()->getParentOp() != kernel)
        return WalkResult::advance();

      auto streamType = cast<StreamType>(arg.getType());
      // skip unranked streams, we don't need to materialize ports for them.
      if (streamType.getShape().empty())
        return WalkResult::advance();
      // analyze the lane indices to determine which lane ports we need to
      // materialize.
      auto laneOr = getStaticLane(op, indices);
      if (failed(laneOr))
        return WalkResult::interrupt();
      // check if we already have a port for this lane
      auto key = std::make_pair(arg, std::move(*laneOr));
      auto it = portMap.find(key);
      BlockArgument portArg;
      if (it != portMap.end())
        portArg = it->second;
      else {
        // add a new port for this lane
        auto scalarStream =
            StreamType::get(kernel.getContext(), streamType.getBaseType(),
                            streamType.getDepth(), {});
        portArg = entry.addArgument(scalarStream, loc);
        portMap.insert({key, portArg});
        portInfo.insert(
            {portArg, {arg.getArgNumber(), std::move(key.second), true}});
      }

      // rewrite the op to use the port argument and no lane indices
      rewriter.setInsertionPoint(op);
      if (auto get = dyn_cast<StreamGetOp>(op)) {
        auto newGet = StreamGetOp::create(rewriter, op->getLoc(), portArg,
                                          ArrayRef<Value>{});
        rewriter.replaceOp(get, newGet);
      } else {
        auto put = cast<StreamPutOp>(op);
        StreamPutOp::create(rewriter, op->getLoc(), portArg, ValueRange{},
                            put.getValue());
        rewriter.eraseOp(put);
      }
    });
    if (walkResult.wasInterrupted())
      return failure();

    // rewrite the function type to eliminate the ranked stream arguments
    ArrayAttr oldArgAttrs = kernel.getArgAttrsAttr();
    BitVector toErase(entry.getNumArguments());
    SmallVector<Type, 8> newInputs;
    SmallVector<Attribute> newArgAttrs;
    KernelInfo info;
    for (auto arg : entry.getArguments()) {
      if (auto it = portInfo.find(arg); it != portInfo.end()) {
        newInputs.push_back(arg.getType());
        info.inputs.push_back(it->second);
        if (oldArgAttrs)
          newArgAttrs.push_back(rewriter.getDictionaryAttr({}));
        continue;
      }

      if (arg.use_empty()) {
        toErase.set(arg.getArgNumber());
        continue;
      }
      if (isRankedStream(arg.getType())) {
        if (!arg.use_empty())
          return kernel.emitError("failed to eliminate ranked stream argument");
        toErase.set(arg.getArgNumber());
      } else {
        newInputs.push_back(arg.getType());
        info.inputs.push_back({arg.getArgNumber(), {}, false});
        if (oldArgAttrs) {
          assert(arg.getArgNumber() < oldArgAttrs.size() &&
                 "arg_attrs must match the old function type");
          newArgAttrs.push_back(oldArgAttrs[arg.getArgNumber()]);
        }
      }
    }
    entry.eraseArguments(toErase);
    auto newType = FunctionType::get(kernel.getContext(), newInputs,
                                     kernel.getFunctionType().getResults());
    kernel.setFunctionType(newType);
    if (oldArgAttrs)
      kernel->setAttr(kernel.getArgAttrsAttrName(),
                      rewriter.getArrayAttr(newArgAttrs));
    kernelInfos[kernel.getSymNameAttr()] = std::move(info);
    return success();
  }

private:
  KernelInfoMap &kernelInfos;
};
} // namespace

static std::string getLaneKey(ArrayRef<int64_t> lane) {
  std::string key;
  for (int64_t index : lane) {
    if (!key.empty())
      key += ".";
    key += std::to_string(index);
  }
  return key;
}

static Value getOrCreateScalarStream(
    Value rankedStream, ArrayRef<int64_t> lane, InvokeOp invoke,
    IRRewriter &rewriter,
    DenseMap<Value, llvm::StringMap<Value>> &scalarStreams) {
  auto &streamsByLane = scalarStreams[rankedStream];
  std::string key = getLaneKey(lane);
  if (auto it = streamsByLane.find(key); it != streamsByLane.end())
    return it->second;

  auto rankedType = cast<StreamType>(rankedStream.getType());
  auto scalarType = StreamType::get(
      invoke.getContext(), rankedType.getBaseType(), rankedType.getDepth(), {});

  OpBuilder::InsertionGuard guard(rewriter);
  if (auto create = rankedStream.getDefiningOp<StreamCreateOp>())
    rewriter.setInsertionPointAfter(create);
  else
    rewriter.setInsertionPoint(invoke);
  Value scalarStream =
      StreamCreateOp::create(rewriter, invoke.getLoc(), scalarType);
  streamsByLane.insert({key, scalarStream});
  return scalarStream;
}

static bool isEmptyKernel(KernelOp kernel) {
  if (kernel.getFunctionType().getNumInputs() != 0 ||
      kernel.getFunctionType().getNumResults() != 0)
    return false;
  if (!llvm::hasSingleElement(kernel.getBody()))
    return false;
  Block &entry = kernel.getBody().front();
  return llvm::hasSingleElement(entry) && isa<ReturnOp>(entry.front());
}

static LogicalResult rewriteInvokes(ModuleOp module, KernelInfoMap &kernelInfos,
                                    IRRewriter &rewriter) {
  DenseMap<Value, llvm::StringMap<Value>> scalarStreams;
  DenseSet<StringAttr> emptyKernels;
  for (auto kernel : module.getOps<KernelOp>())
    if (isEmptyKernel(kernel))
      emptyKernels.insert(kernel.getSymNameAttr());

  SmallVector<InvokeOp> invokes;
  module.walk([&](InvokeOp invoke) {
    if (kernelInfos.contains(invoke.getCalleeAttr().getAttr()))
      invokes.push_back(invoke);
  });

  for (InvokeOp invoke : invokes) {
    StringAttr callee = invoke.getCalleeAttr().getAttr();
    if (emptyKernels.contains(callee)) {
      assert(invoke->getNumResults() == 0 &&
             "empty kernels cannot produce results");
      rewriter.eraseOp(invoke);
      continue;
    }

    auto it = kernelInfos.find(callee);
    assert(it != kernelInfos.end() && "invoke must reference rewritten kernel");

    ArrayAttr oldArgAttrs = invoke.getArgAttrsAttr();
    SmallVector<Value> newOperands;
    SmallVector<Attribute> newArgAttrs;
    for (auto &port : it->second.inputs) {
      assert(port.sourceArg < invoke->getNumOperands() &&
             "rewritten port must reference an old invoke operand");
      Value source = invoke->getOperand(port.sourceArg);
      if (!port.scalarPort) {
        newOperands.push_back(source);
        if (oldArgAttrs) {
          assert(port.sourceArg < oldArgAttrs.size() &&
                 "arg_attrs must match the old invoke operands");
          newArgAttrs.push_back(oldArgAttrs[port.sourceArg]);
        }
        continue;
      }

      if (!isRankedStream(source.getType()))
        return invoke.emitError("scalarized stream port source must be a "
                                "ranked stream operand");
      newOperands.push_back(getOrCreateScalarStream(source, port.lane, invoke,
                                                    rewriter, scalarStreams));
      if (oldArgAttrs)
        newArgAttrs.push_back(rewriter.getDictionaryAttr({}));
    }

    rewriter.modifyOpInPlace(invoke, [&]() {
      invoke->setOperands(newOperands);
      if (oldArgAttrs)
        invoke->setAttr(invoke.getArgAttrsAttrName(),
                        rewriter.getArrayAttr(newArgAttrs));
    });
  }

  SmallVector<KernelOp> kernelsToErase;
  for (auto kernel : module.getOps<KernelOp>())
    if (isEmptyKernel(kernel) && kernel.symbolKnownUseEmpty(module))
      kernelsToErase.push_back(kernel);
  for (KernelOp kernel : kernelsToErase)
    rewriter.eraseOp(kernel);

  SmallVector<StreamCreateOp> streamsToErase;
  module.walk([&](StreamCreateOp create) {
    if (isRankedStream(create.getStream().getType()) &&
        create.getStream().use_empty())
      streamsToErase.push_back(create);
  });
  for (StreamCreateOp create : streamsToErase)
    rewriter.eraseOp(create);

  return success();
}

namespace {
struct MaterializeTopologyPass
    : public allo::impl::MaterializeTopologyPassBase<MaterializeTopologyPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    KernelInfoMap kernelInfos;
    RewritePatternSet patterns(context);
    patterns.add<ScalarizeStreamPorts>(context, kernelInfos);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    IRRewriter rewriter(context);
    if (failed(rewriteInvokes(getOperation(), kernelInfos, rewriter)))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, allo::AlloDialect>();
  }
};
} // namespace
