/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Support/TopologyGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "allo/Transforms/Passes.h"

namespace mlir::allo {
#define GEN_PASS_DEF_GRIDMAPPINGPASS
#include "allo/Transforms/Passes.h.inc"
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

namespace {
struct SpecializeKernelsPattern : OpRewritePattern<KernelOp> {
  SpecializeKernelsPattern(MLIRContext *ctx, Operation *symbolTableOp)
      : OpRewritePattern(ctx), symbolTableOp(symbolTableOp) {}

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const override {
    auto mapping = op.getMapping();
    bool identityMapping = isIdentityMapping(mapping);
    if (identityMapping)
      return failure(); // no need to specialize identity mapping kernels

    if (op->getNumResults() != 0)
      return op->emitError() << "Cannot specialize non-identity mapping "
                                "kernels with non-void results";

    Grid grid = generateGrid(mapping);
    SmallVector<KernelOp> kernels;
    for (auto &coord : grid) {
      std::string instName = makeInstanceName(op.getSymName(), coord);
      rewriter.setInsertionPoint(op);
      auto kernel = cast<KernelOp>(rewriter.clone(*op));
      kernel.setSymName(instName);
      kernel.setMapping({1});
      kernel->setAttr(kGridAttrName, rewriter.getDenseI32ArrayAttr(mapping));
      kernel->setAttr(kCoordAttrName, rewriter.getDenseI32ArrayAttr(coord));
      kernels.push_back(kernel);

      kernel.walk([&](Operation *nested) {
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
    // rewrite the calls to the original kernel
    auto uses = SymbolTable::getSymbolUses(op.getSymNameAttr(), symbolTableOp);
    if (!uses)
      return op.emitError()
             << "Cannot collect symbol uses of kernel @" << op.getSymName();
    for (auto use : *uses) {
      auto invoke = cast<InvokeOp>(use.getUser());
      rewriter.setInsertionPoint(invoke);
      for (auto kernel : kernels) {
        cast<InvokeOp>(rewriter.clone(*invoke))
            .setCalleeAttr(FlatSymbolRefAttr::get(kernel));
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
struct GridMappingPass
    : public allo::impl::GridMappingPassBase<GridMappingPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SpecializeKernelsPattern>(context, getOperation());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);
    (void)applyPatternsGreedily(getOperation(), std::move(owningPatterns));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
};
} // namespace
