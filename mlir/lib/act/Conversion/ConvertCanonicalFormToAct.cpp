#include "act/Conversion/Passes.h"
#include "act/IR/ActOps.h"
#include "act/Support/CodeEmission.h"
#include "act/Support/SemanticMatching.h"
#include "act/Support/TilingAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
      OpPassManager pm(getOperation()->getName());
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
    SmallVector<EdgeLayoutAnnotation, 4> layoutAnnotations;
    if (failed(runSemanticMatching(module, matches, layoutAnnotations)))
      return signalPassFailure();

    // Step 2.5: Structural matching for rank-mismatched ops
    {
      DenseSet<Operation *> matchedOps;
      for (auto &m : matches)
        matchedOps.insert(m.sourceOp);

      SmallVector<Operation *> unmatchedOps;
      module.walk([&](func::FuncOp funcOp) {
        funcOp.walk([&](Operation *op) {
          if (!isa<linalg::LinalgOp>(op))
            return;
          // Skip layout ops (they are edge annotations, not compute)
          if (isa<linalg::TransposeOp>(op))
            return;
          if (op->getParentOfType<DefineOp>())
            return;
          if (!matchedOps.contains(op))
            unmatchedOps.push_back(op);
        });
      });

      SmallVector<MatchCandidate> structuralMatches;
      if (failed(
              runStructuralMatching(module, unmatchedOps, structuralMatches)))
        return signalPassFailure();

      matches.append(structuralMatches.begin(), structuralMatches.end());
    }

    // Step 3: Report matching results
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Matching Results ===\n";
      for (auto &m : matches) {
        llvm::dbgs() << "  " << m.sourceOp->getName() << " -> @"
                     << m.instruction.getSymName();
        if (m.numOuterDims > 0)
          llvm::dbgs() << " [structural, outer=" << m.numOuterDims << "]";
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "=== " << matches.size() << " match(es) found ===\n";
    });

    // Step 4: Tiling analysis (Stage 2)
    SmallVector<TiledMatchCandidate> tiledMatches;
    if (failed(runTilingAnalysis(module, matches, tiledMatches)))
      return signalPassFailure();

    // Step 5: Report Stage 2 results
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Tiling Analysis Results ===\n";
      for (auto &tm : tiledMatches) {
        llvm::dbgs() << "  " << tm.base.sourceOp->getName() << " -> @"
                     << tm.base.instruction.getSymName();
        if (tm.isValid) {
          llvm::dbgs() << " [valid]";
          if (tm.needsTiling())
            llvm::dbgs() << " needs-tiling";
          else
            llvm::dbgs() << " single-invocation";
          // Print solved params
          if (!tm.tiling.solvedParams.empty()) {
            llvm::dbgs() << " params={";
            bool first = true;
            for (auto &[idx, val] : tm.tiling.solvedParams) {
              if (!first)
                llvm::dbgs() << ",";
              llvm::dbgs() << "p" << idx << "=" << val;
              first = false;
            }
            llvm::dbgs() << "}";
          }
          // Print outer dims info
          if (tm.numOuterDims > 0)
            llvm::dbgs() << " outer-dims=" << tm.numOuterDims;
          // Print tiling factors if tiling needed
          if (tm.needsTiling()) {
            llvm::dbgs() << " tiles=[";
            for (unsigned i = 0; i < tm.tiling.dims.size(); ++i) {
              if (i > 0)
                llvm::dbgs() << ",";
              llvm::dbgs() << tm.tiling.dims[i].tileFactor;
            }
            llvm::dbgs() << "]";
          }
        } else {
          llvm::dbgs() << " [INFEASIBLE]";
        }
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "=== " << tiledMatches.size() << " tiled match(es) ===\n";
    });

    // Step 6: Code emission (Stage 3)
    if (failed(runCodeEmission(module, tiledMatches, layoutAnnotations)))
      return signalPassFailure();
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
