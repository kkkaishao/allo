#include "act/Conversion/Passes.h"
#include "act/IR/ActOps.h"
#include "act/Support/CodeEmission.h"
#include "act/Support/ParamSolving.h"
#include "act/Support/Planning.h"
#include "act/Support/SemanticMatching.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "convert-canonical-to-act"

namespace mlir::act {
#define GEN_PASS_DEF_CONVERTCANONICALFORMTOACTPASS
#include "act/Conversion/Passes.h.inc"
} // namespace mlir::act

using namespace mlir;
using namespace mlir::act;

static bool isControlFlowOp(Operation *op) {
  return isa<LoopLikeOpInterface, BranchOpInterface, RegionBranchOpInterface>(
      op);
}

static LogicalResult validateInputProgram(func::FuncOp func, ModuleOp module) {
  if (!func.symbolKnownUseEmpty(module))
    return func.emitError()
           << "cannot lower function @" << func.getSymName()
           << " to act.sequence while it still has symbol users";

  if (func.getFunctionBody().getBlocks().size() != 1)
    return func.emitError()
           << "non-flat function structure is not supported; expected exactly "
              "one block in function body";

  WalkResult walkResult = func.walk([&](Operation *op) {
    if (isControlFlowOp(op)) {
      op->emitError()
          << "input control flow is not supported in instruction selection";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success(!walkResult.wasInterrupted());
}

namespace {
struct LowerFunctionToActPattern : OpConversionPattern<func::FuncOp> {
  LowerFunctionToActPattern(MLIRContext *ctx, ModuleOp module,
                            InstructionCollection &collection)
      : OpConversionPattern(ctx), module(module), collection(collection) {}

  LogicalResult
  matchAndRewrite(func::FuncOp func, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(validateInputProgram(func, module)))
      return failure();

    auto graphOr = runSemanticMatching(func, collection);
    if (failed(graphOr))
      return failure();
    auto solutionsOr = runParamSolving(*graphOr);
    if (failed(solutionsOr))
      return failure();
    auto planOr = buildExecutionPlan(*graphOr, *solutionsOr);
    if (failed(planOr))
      return failure();
    if (failed(emitInstructionSequence(rewriter, *planOr)))
      return failure();
    return success();
    // DenseSet<Operation *> semanticOps;
    // for (GraphNode &node : programOr->nodes)
    //   semanticOps.insert(node.op);
    // if (semanticOps.empty()) {
    //   plan.graph = SemanticsGraph{func, {}, {}};
    //   plan.isComplete = true;
    // } else {
    //   auto graphOr = runSemanticMatching(func, catalog);
    //   if (failed(graphOr))
    //     return failure();
    //   plan.graph = std::move(*graphOr);
    //
    //   for (SemanticsGraphNode &node : plan.graph.nodes)
    //     for (Operation *op : node.sourceOps)
    //       plan.coveredSemanticOps.insert(op);
    //
    //   if (llvm::any_of(semanticOps, [&](Operation *op) {
    //         return !plan.coveredSemanticOps.contains(op);
    //       })) {
    //     return func.emitError()
    //            << "partial instruction lowering is not supported for function
    //            @"
    //            << func.getSymName();
    //   }
    //   auto paramOr = runParamSolving(plan.graph, module);
    //   if (failed(paramOr))
    //     return failure();
    //   plan.solution = std::move(*paramOr);
    //
    //   auto logicalOr = buildLogicalPlan(func, plan.graph, plan.solution);
    //   if (failed(logicalOr))
    //     return failure();
    //   plan.logicalPlan = std::move(*logicalOr);
    //
    //   auto resourceOr = buildResourcePlan(func, plan.logicalPlan, module);
    //   if (failed(resourceOr))
    //     return failure();
    //   plan.resourcePlan = std::move(*resourceOr);
    //   plan.isComplete = true;
    // }
    //
    // if (!plan.isComplete)
    //   return func.emitError()
    //          << "failed to build a complete lowering plan for function @"
    //          << func.getSymName();
    //
    // LLVM_DEBUG({
    //   llvm::dbgs() << "Built FunctionLoweringPlan for @" << func.getSymName()
    //                << " with " << plan.logicalPlan.nodes.size()
    //                << " selected node(s)\n";
    // });
    //
    // LLVM_DEBUG(llvm::dbgs() << "\n=== Lowering function @" <<
    // func.getSymName()
    //                         << " ===\n";);
    // return emitInstructionSequence(rewriter, plan);
  }

private:
  ModuleOp module;
  InstructionCollection &collection;
};
} // namespace

namespace {
struct ConvertCanonicalToActPass
    : public act::impl::ConvertCanonicalFormToActPassBase<
          ConvertCanonicalToActPass> {
  ConvertCanonicalToActPass() = default;
  explicit ConvertCanonicalToActPass(
      const ConvertCanonicalFormToActPassOptions &options) {
    isaPath = options.isaPath;
  }

  void runOnOperation() override {
    auto module = cast<ModuleOp>(getOperation());

    // Step 0: Load ISA definitions if external file provided
    if (isaPath.has_value() && failed(inlineISA()))
      return signalPassFailure();

    // Step 1: Pre-pass pipeline (specialize generic → named ops, canonicalize)
    {
      OpPassManager pm(getOperation()->getName());
      LinalgMorphOpsPassOptions morphOpts;
      morphOpts.genericToNamed = true;
      pm.addPass(createLinalgMorphOpsPass(morphOpts));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    auto collectionOr = InstructionCollection::build(module);
    if (failed(collectionOr)) {
      module.emitError()
          << "failed to build instruction collection for semantic "
             "matching";
      return signalPassFailure();
    }

    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<act::ActDialect, affine::AffineDialect,
                           arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<func::FuncOp>();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFunctionToActPattern>(ctx, module, *collectionOr);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<act::ActDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

private:
  LogicalResult inlineISA() {
    std::string errMsg;
    auto buffer = openInputFile(*isaPath, &errMsg);
    if (!errMsg.empty())
      return getOperation()->emitError()
             << "Error opening ISA file: " << errMsg;
    ParserConfig config(&getContext());
    auto module = parseSourceString<ModuleOp>(buffer->getBuffer(), config);
    if (!module)
      return getOperation()->emitError()
             << "Error parsing ISA file: " << *isaPath;
    IRRewriter rewriter(&getContext());
    auto currModule = dyn_cast<ModuleOp>(getOperation());
    if (!currModule)
      return getOperation()->emitError()
             << "Expected a module operation as the top-level operation.";
    rewriter.inlineBlockBefore(module->getBody(), currModule.getBody(),
                               currModule.getBody()->begin());
    // Release ownership before erasing to avoid double-free
    Operation *parsedOp = module.release();
    rewriter.eraseOp(parsedOp);
    if (failed(currModule.verify()))
      return getOperation()->emitError()
             << "Inlined module from ISA file is not well-formed.";
    return success();
  }
};
} // namespace
