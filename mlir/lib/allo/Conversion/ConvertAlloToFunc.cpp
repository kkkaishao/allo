#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

namespace {
struct ConvertKernelToFunc : OpRewritePattern<KernelOp> {
  ConvertKernelToFunc(MLIRContext *ctx, Operation *symbolTableOp)
      : OpRewritePattern(ctx), symbolTableOp(symbolTableOp) {}

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const override {
    auto mapping = op.getMapping();
    bool identityMapping = isIdentityMapping(mapping);
    if (op->getNumResults() != 0 && !identityMapping)
      return op->emitError() << "Cannot convert non-void kernels without an "
                                "identical mapping to func";

    Grid grid = generateGrid(mapping);
    rewriter.setInsertionPoint(op);
    SmallVector<func::FuncOp> fns;
    for (auto &coord : grid) {
      std::string instName = identityMapping
                                 ? op.getSymName().str()
                                 : makeInstanceName(op.getSymName(), coord);
      auto fn =
          func::FuncOp::create(rewriter, op->getLoc(), instName,
                               op.getFunctionType(), op.getSymVisibilityAttr(),
                               op.getArgAttrsAttr(), op.getResAttrsAttr());
      fn->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      fns.push_back(fn);
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

    for (InvokeOp invoke : invokes) {
      rewriter.setInsertionPoint(invoke);
      if (identityMapping) {
        auto call = func::CallOp::create(rewriter, invoke->getLoc(),
                                         fns.front(), invoke.getOperands());
        copyInvokeAttrs(invoke, call);
        rewriter.replaceOp(invoke, call);
      }
      assert(invoke->getNumResults() == 0 &&
             "non-identical kernel invokes must be void");
      for (auto fn : fns) {
        auto call = func::CallOp::create(rewriter, invoke->getLoc(), fn,
                                         invoke.getOperands());
        copyInvokeAttrs(invoke, call);
      }
      rewriter.eraseOp(invoke);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  Operation *symbolTableOp;
};
} // namespace

namespace {
struct ConvertInvokeToFunc : OpConversionPattern<InvokeOp> {
  using OpConversionPattern<InvokeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InvokeOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    if (SymbolTable::lookupNearestSymbolFrom<KernelOp>(op, op.getCalleeAttr()))
      return failure();
    rewriter.setInsertionPoint(op);
    auto call =
        func::CallOp::create(rewriter, op->getLoc(), op.getCalleeAttr(),
                             op->getResultTypes(), adapter.getOperands());
    copyInvokeAttrs(op, call);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
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
    RewritePatternSet patterns(context);
    patterns.add<ConvertKernelToFunc>(context, getOperation());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    ConversionTarget target(*context);
    patterns.clear();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<KernelOp, ReturnOp, GetWorkerIdOp, GetNumWorkersOp>();
    patterns.add<ConvertInvokeToFunc, ConvertReturnToFunc>(context);
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
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }
};
} // namespace
