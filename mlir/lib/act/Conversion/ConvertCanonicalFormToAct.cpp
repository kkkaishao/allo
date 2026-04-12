#include "act/Conversion/Passes.h"
#include "act/IR/ActOps.h"
#include "act/Support/SemanticMatching.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
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
      OpPassManager pm("builtin.module");
      LinalgMorphOpsPassOptions morphOpts;
      morphOpts.genericToNamed = true;
      pm.addPass(createLinalgMorphOpsPass(morphOpts));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      if (failed(runPipeline(pm, getOperation())))
        return signalPassFailure();
    }

    // Step 2: Semantic matching
    SmallVector<MatchCandidate> matches;
    if (failed(runSemanticMatching(module, matches)))
      return signalPassFailure();

    // Step 3: Report results
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Semantic Matching Results ===\n";
      for (auto &m : matches)
        llvm::dbgs() << "  " << m.sourceOp->getName() << " -> @"
                     << m.instruction.getSymName() << "\n";
      llvm::dbgs() << "=== " << matches.size() << " match(es) found ===\n";
    });
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<act::ActDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
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
