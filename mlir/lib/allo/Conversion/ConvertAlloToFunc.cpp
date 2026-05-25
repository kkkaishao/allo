#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "allo/Conversion/Passes.h"

namespace mlir::allo {
#define GEN_PASS_DEF_CONVERTALLOTOFUNCPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

static std::string makeInstanceName(StringRef base, ArrayRef<int32_t> grid) {
  std::string name = base.str();
  for (int32_t coord : grid) {
    name += ".";
    name += std::to_string(coord);
  }
  return name;
}

using GridCoord = SmallVector<int32_t, 4>;
using Grid = SmallVector<GridCoord>;
using LoweredKernelMap =
    DenseMap<StringAttr, SmallVector<FlatSymbolRefAttr, 4>>;

static Grid generateGrid(ArrayRef<int32_t> mapping) {
  Grid grid(1, GridCoord(mapping.size(), 0));
  for (size_t axis = 0; axis < mapping.size(); ++axis) {
    assert(mapping[axis] > 0 && "worker counts must be positive");
    Grid next;
    for (GridCoord coord : grid) {
      for (int32_t wid = 0; wid < mapping[axis]; ++wid) {
        coord[axis] = wid;
        next.push_back(coord);
      }
    }
    grid = std::move(next);
  }
  return grid;
}

static bool isIdentityMapping(ArrayRef<int32_t> mapping) {
  return llvm::all_of(mapping, [](int32_t x) { return x == 1; });
}

static void copyInvokeAttrs(InvokeOp invoke, func::CallOp call) {
  if (auto argAttrs = invoke.getArgAttrsAttr())
    call.setArgAttrsAttr(argAttrs);
  if (auto resAttrs = invoke.getResAttrsAttr())
    call.setResAttrsAttr(resAttrs);
}

static LogicalResult lowerInvokeToFuncCalls(InvokeOp invoke,
                                            ValueRange operands,
                                            TypeRange results,
                                            ArrayRef<FlatSymbolRefAttr> callees,
                                            PatternRewriter &rewriter) {
  assert(!callees.empty() && "expected at least one lowered callee");
  rewriter.setInsertionPoint(invoke);
  if (callees.size() == 1) {
    auto call = func::CallOp::create(rewriter, invoke->getLoc(),
                                     callees.front(), results, operands);
    copyInvokeAttrs(invoke, call);
    rewriter.replaceOp(invoke, call.getResults());
    return success();
  }

  if (invoke->getNumResults() != 0)
    return invoke->emitError()
           << "Cannot convert non-void invokes to multiple func calls";

  for (auto callee : callees) {
    auto call = func::CallOp::create(rewriter, invoke->getLoc(), callee,
                                     TypeRange{}, operands);
    copyInvokeAttrs(invoke, call);
  }
  rewriter.eraseOp(invoke);
  return success();
}

static LogicalResult lowerKnownInvokes(Operation *op,
                                       LoweredKernelMap &loweredKernels,
                                       PatternRewriter &rewriter) {
  SmallVector<InvokeOp> invokes;
  op->walk([&](InvokeOp invoke) {
    if (loweredKernels.contains(invoke.getCalleeAttr().getAttr()))
      invokes.push_back(invoke);
  });

  for (InvokeOp invoke : invokes) {
    auto it = loweredKernels.find(invoke.getCalleeAttr().getAttr());
    assert(it != loweredKernels.end() && "known invoke must have a callee");
    if (failed(lowerInvokeToFuncCalls(invoke, invoke.getOperands(),
                                      invoke->getResultTypes(), it->second,
                                      rewriter)))
      return failure();
  }
  return success();
}

namespace {
struct ConvertKernelToFunc : OpRewritePattern<KernelOp> {
  ConvertKernelToFunc(MLIRContext *ctx, Operation *symbolTableOp,
                      LoweredKernelMap &loweredKernels)
      : OpRewritePattern(ctx), symbolTableOp(symbolTableOp),
        loweredKernels(loweredKernels) {}

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const override {
    auto mapping = op.getMapping();
    bool identityMapping = isIdentityMapping(mapping);
    if (op->getNumResults() != 0 && !identityMapping)
      return op->emitError() << "Cannot convert non-void kernels without an "
                                "identical mapping to func";

    Grid grid = generateGrid(mapping);
    SmallVector<func::FuncOp> fns;
    SmallVector<FlatSymbolRefAttr, 4> loweredCallees;
    for (auto &coord : grid) {
      std::string instName = identityMapping
                                 ? op.getSymName().str()
                                 : makeInstanceName(op.getSymName(), coord);
      rewriter.setInsertionPoint(op);
      auto fn =
          func::FuncOp::create(rewriter, op->getLoc(), instName,
                               op.getFunctionType(), op.getSymVisibilityAttr(),
                               op.getArgAttrsAttr(), op.getResAttrsAttr());
      fn->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      fns.push_back(fn);
      loweredCallees.push_back(FlatSymbolRefAttr::get(fn));
      rewriter.cloneRegionBefore(op.getRegion(), fn.getBody(),
                                 fn.getBody().begin());

      fn.walk([&](Operation *nested) {
        if (auto wid = dyn_cast<GetWorkerIdOp>(nested)) {
          rewriter.setInsertionPoint(wid);
          auto cst = arith::ConstantIndexOp::create(rewriter, wid->getLoc(),
                                                    coord[wid.getAxis()]);
          rewriter.replaceOp(wid, cst);
        } else if (auto num = dyn_cast<GetNumWorkersOp>(nested)) {
          rewriter.setInsertionPoint(num);
          auto cst = arith::ConstantIndexOp::create(rewriter, num->getLoc(),
                                                    mapping[num.getAxis()]);
          rewriter.replaceOp(num, cst);
        }
      });
    }

    assert(!fns.empty() && "kernel conversion must create at least one func");
    assert(!loweredKernels.contains(op.getSymNameAttr()) &&
           "kernel must be converted only once");
    loweredKernels.insert({op.getSymNameAttr(), loweredCallees});
    for (auto fn : fns)
      if (failed(lowerKnownInvokes(fn, loweredKernels, rewriter)))
        return failure();

    SmallVector<InvokeOp> invokes;
    auto uses = SymbolTable::getSymbolUses(op.getSymNameAttr(), symbolTableOp);
    if (!uses)
      return op->emitError()
             << "Cannot collect symbol uses of kernel @" << op.getSymName();
    for (auto use : *uses) {
      Operation *owner = use.getUser();
      auto invoke = dyn_cast<InvokeOp>(owner);
      if (!invoke)
        return op->emitError() << "Unidentified user of kernel @"
                               << op.getSymName() << ", should be allo.invoke";
      if (invoke->getParentOfType<KernelOp>())
        continue;
      invokes.push_back(invoke);
    }

    for (InvokeOp invoke : invokes)
      if (failed(lowerInvokeToFuncCalls(invoke, invoke.getOperands(),
                                        invoke->getResultTypes(),
                                        loweredCallees, rewriter)))
        return failure();
    rewriter.eraseOp(op);
    return success();
  }

private:
  Operation *symbolTableOp;
  LoweredKernelMap &loweredKernels;
};
} // namespace

namespace {
struct ConvertInvokeToFunc : OpConversionPattern<InvokeOp> {
  ConvertInvokeToFunc(MLIRContext *ctx, LoweredKernelMap &loweredKernels)
      : OpConversionPattern(ctx), loweredKernels(loweredKernels) {}

  LogicalResult
  matchAndRewrite(InvokeOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    auto it = loweredKernels.find(op.getCalleeAttr().getAttr());
    if (it == loweredKernels.end())
      return failure();
    return lowerInvokeToFuncCalls(op, adapter.getOperands(),
                                  op->getResultTypes(), it->second, rewriter);
  }

private:
  LoweredKernelMap &loweredKernels;
};
} // namespace

namespace {
struct ConvertReturnToFunc : OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->getParentOfType<func::FuncOp>())
      return failure();
    rewriter.setInsertionPoint(op);
    auto ret =
        func::ReturnOp::create(rewriter, op->getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, ret);
    return success();
  }
};
} // namespace

namespace {
struct ConvertAlloToFuncPass
    : public allo::impl::ConvertAlloToFuncPassBase<ConvertAlloToFuncPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    LoweredKernelMap loweredKernels;
    RewritePatternSet patterns(context);
    patterns.add<ConvertKernelToFunc>(context, getOperation(), loweredKernels);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    ConversionTarget target(*context);
    patterns.clear();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           omp::OpenMPDialect>();
    target.addIllegalOp<KernelOp, ReturnOp, GetWorkerIdOp, GetNumWorkersOp>();
    patterns.add<ConvertInvokeToFunc>(context, loweredKernels);
    patterns.add<ConvertReturnToFunc>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);
    (void)applyPatternsGreedily(getOperation(), std::move(owningPatterns));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, omp::OpenMPDialect>();
  }
};
} // namespace
