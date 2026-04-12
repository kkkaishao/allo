#include "act/Support/SemanticMatching.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "semantic-matching"

using namespace mlir;
using namespace mlir::act;

//===----------------------------------------------------------------------===//
// Layout op classification
//===----------------------------------------------------------------------===//

static bool isLayoutOp(Operation *op) {
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp,
             linalg::TransposeOp>(op);
}

//===----------------------------------------------------------------------===//
// Compute region validation
//===----------------------------------------------------------------------===//

/// Validate that a compute region contains no layout ops. After the
/// compute/access separation, all layout transforms belong in the addr
/// region. If layout ops are found, they are classified as boundary
/// transforms (pre/post) and reported as errors — the ISA author should
/// move them to the addr region.

/// Identify pre-transform ops: layout ops whose operands are all block args
/// or other pre-transforms. Fixed-point iteration.
static DenseSet<Operation *> findPreTransforms(Block &block) {
  DenseSet<Operation *> preTransforms;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : block) {
      if (preTransforms.contains(&op) || !isLayoutOp(&op))
        continue;
      bool allOperandsFromBoundary =
          llvm::all_of(op.getOperands(), [&](Value v) {
            if (isa<BlockArgument>(v))
              return true;
            auto *defOp = v.getDefiningOp();
            return defOp && (preTransforms.contains(defOp) ||
                             isa<arith::ConstantOp>(defOp));
          });
      if (allOperandsFromBoundary) {
        preTransforms.insert(&op);
        changed = true;
      }
    }
  }
  return preTransforms;
}

/// Identify post-transform ops: layout ops whose results feed only into
/// yield or other post-transforms. Fixed-point iteration.
static DenseSet<Operation *> findPostTransforms(Block &block) {
  DenseSet<Operation *> postTransforms;
  auto *yieldOp = block.getTerminator();

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : block) {
      if (postTransforms.contains(&op) || !isLayoutOp(&op))
        continue;
      bool allUsersAtBoundary =
          llvm::all_of(op.getResults(), [&](Value result) {
            return llvm::all_of(result.getUsers(), [&](Operation *user) {
              return user == yieldOp || postTransforms.contains(user);
            });
          });
      if (allUsersAtBoundary) {
        postTransforms.insert(&op);
        changed = true;
      }
    }
  }
  return postTransforms;
}

/// Check that a compute block has no boundary layout ops. Returns failure
/// and emits diagnostics if any are found.
static LogicalResult validateComputeRegion(DefineOp defineOp) {
  Block &block = defineOp.getSemanticsBlock();
  auto preTransforms = findPreTransforms(block);
  auto postTransforms = findPostTransforms(block);

  bool hasErrors = false;
  for (auto *op : preTransforms) {
    defineOp.emitWarning()
        << "compute region of @" << defineOp.getSymName()
        << " contains boundary layout op '" << op->getName()
        << "' that should be in the addr region (pre-transform)";
    hasErrors = true;
  }
  for (auto *op : postTransforms) {
    defineOp.emitWarning()
        << "compute region of @" << defineOp.getSymName()
        << " contains boundary layout op '" << op->getName()
        << "' that should be in the addr region (post-transform)";
    hasErrors = true;
  }

  if (hasErrors) {
    LLVM_DEBUG(llvm::dbgs() << "  [warning] @" << defineOp.getSymName()
                            << " has " << preTransforms.size()
                            << " pre-transform(s) and " << postTransforms.size()
                            << " post-transform(s) in compute region\n");
  }
  return success(!hasErrors);
}

/// Collect core compute ops from a compute block, filtering out yield,
/// constants, and tensor.empty (allocation, not compute). After the
/// compute/access separation, all remaining ops should be compute ops.
static SmallVector<Operation *> collectCoreOps(Block &block) {
  SmallVector<Operation *> coreOps;
  for (Operation &op : block) {
    if (isa<YieldOp>(&op) || isa<arith::ConstantOp>(&op) ||
        isa<tensor::EmptyOp>(&op))
      continue;
    coreOps.push_back(&op);
  }
  return coreOps;
}

//===----------------------------------------------------------------------===//
// Fingerprint extraction
//===----------------------------------------------------------------------===//

/// Extract a semantic fingerprint from an act.define's compute region.
static SemanticFingerprint extractFingerprint(DefineOp defineOp) {
  Block &block = defineOp.getSemanticsBlock();
  auto coreOps = collectCoreOps(block);

  // Identity: no core compute ops
  if (coreOps.empty())
    return {SemanticFingerprint::Identity, {}, {}, {}, nullptr, 0, 0};

  // Single core op
  if (coreOps.size() == 1) {
    Operation *op = coreOps[0];
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Named linalg op (not generic)
      if (!isa<linalg::GenericOp>(op)) {
        return {SemanticFingerprint::Named,
                op->getName().getStringRef(),
                {},
                {},
                nullptr,
                0,
                0};
      }
      // linalg.generic — extract structural info
      return {SemanticFingerprint::Generic,
              {},
              linalgOp.getIndexingMapsArray(),
              linalgOp.getIteratorTypesArray(),
              &op->getRegion(0),
              static_cast<unsigned>(linalgOp.getNumDpsInputs()),
              static_cast<unsigned>(linalgOp.getNumDpsInits())};
    }
    // Non-linalg single op (shouldn't normally happen)
    return {SemanticFingerprint::Named,
            op->getName().getStringRef(),
            {},
            {},
            nullptr,
            0,
            0};
  }

  // Multi-op: not yet supported, log and return identity as fallback
  LLVM_DEBUG(llvm::dbgs() << "  [skip] multi-op pattern in @"
                          << defineOp.getSymName() << " (not yet supported)\n");
  return {SemanticFingerprint::Identity, {}, {}, {}, nullptr, 0, 0};
}

/// Compute a semantic fingerprint for a source op.
static SemanticFingerprint computeSourceFingerprint(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (!isa<linalg::GenericOp>(op)) {
      return {SemanticFingerprint::Named,
              op->getName().getStringRef(),
              {},
              {},
              nullptr,
              0,
              0};
    }
    return {SemanticFingerprint::Generic,
            {},
            linalgOp.getIndexingMapsArray(),
            linalgOp.getIteratorTypesArray(),
            &op->getRegion(0),
            static_cast<unsigned>(linalgOp.getNumDpsInputs()),
            static_cast<unsigned>(linalgOp.getNumDpsInits())};
  }
  // Non-linalg op — use op name
  return {SemanticFingerprint::Named,
          op->getName().getStringRef(),
          {},
          {},
          nullptr,
          0,
          0};
}

//===----------------------------------------------------------------------===//
// SemanticFingerprint hashing and comparison
//===----------------------------------------------------------------------===//

llvm::hash_code SemanticFingerprint::hash() const {
  switch (kind) {
  case Named:
    return llvm::hash_combine(kind, opName);
  case Generic: {
    // Hash iterator types + input/output counts (not body — used for lookup)
    llvm::hash_code h = llvm::hash_combine(kind, numInputs, numOutputs);
    for (auto it : iteratorTypes)
      h = llvm::hash_combine(h, static_cast<int>(it));
    return h;
  }
  case Identity:
    return llvm::hash_combine(kind);
  }
  llvm_unreachable("unknown fingerprint kind");
}

bool SemanticFingerprint::matches(const SemanticFingerprint &other) const {
  if (kind != other.kind)
    return false;

  switch (kind) {
  case Named:
    return opName == other.opName;

  case Generic: {
    // Compare indexing maps structurally
    if (indexingMaps.size() != other.indexingMaps.size())
      return false;
    for (size_t i = 0; i < indexingMaps.size(); ++i)
      if (indexingMaps[i] != other.indexingMaps[i])
        return false;
    // Compare iterator types
    if (iteratorTypes != other.iteratorTypes)
      return false;
    // Compare body regions
    if (!bodyRegion || !other.bodyRegion)
      return false;
    // Use OperationEquivalence for body comparison.
    // Block args match by position, ignoring types (shape-agnostic).
    return OperationEquivalence::isRegionEquivalentTo(
        bodyRegion, other.bodyRegion,
        [](Value lhs, Value rhs) -> LogicalResult {
          auto lhsArg = dyn_cast<BlockArgument>(lhs);
          auto rhsArg = dyn_cast<BlockArgument>(rhs);
          if (lhsArg && rhsArg)
            return success(lhsArg.getArgNumber() == rhsArg.getArgNumber());
          return failure();
        },
        /*markEquivalent=*/nullptr,
        OperationEquivalence::Flags::IgnoreLocations);
  }

  case Identity:
    return true;
  }
  llvm_unreachable("unknown fingerprint kind");
}

//===----------------------------------------------------------------------===//
// InstructionCatalog
//===----------------------------------------------------------------------===//

InstructionCatalog InstructionCatalog::build(ModuleOp module) {
  InstructionCatalog catalog;
  module.walk([&](DefineOp defineOp) {
    // Validate that compute regions are free of boundary layout ops.
    (void)validateComputeRegion(defineOp);

    auto fp = extractFingerprint(defineOp);
    LLVM_DEBUG({
      llvm::dbgs() << "  Instruction @" << defineOp.getSymName() << ": ";
      switch (fp.kind) {
      case SemanticFingerprint::Named:
        llvm::dbgs() << "Named(" << fp.opName << ")";
        break;
      case SemanticFingerprint::Generic:
        llvm::dbgs() << "Generic(inputs=" << fp.numInputs
                     << ", outputs=" << fp.numOutputs << ")";
        break;
      case SemanticFingerprint::Identity:
        llvm::dbgs() << "Identity";
        break;
      }
      llvm::dbgs() << "\n";
    });
    auto h = fp.hash();
    catalog.index[h].push_back({defineOp, std::move(fp)});
  });
  return catalog;
}

SmallVector<MatchCandidate>
InstructionCatalog::match(Operation *sourceOp) const {
  auto sourceFP = computeSourceFingerprint(sourceOp);
  auto h = sourceFP.hash();

  SmallVector<MatchCandidate> results;
  auto it = index.find(h);
  if (it == index.end())
    return results;

  for (auto &entry : it->second) {
    if (entry.fingerprint.matches(sourceFP))
      results.push_back({sourceOp, entry.defineOp});
  }
  return results;
}

void InstructionCatalog::dump() {
  llvm::dbgs() << "InstructionCatalog (" << index.size() << " hash buckets):\n";
  for (auto &[h, entries] : index) {
    for (auto &entry : entries) {
      llvm::dbgs() << "  @" << entry.defineOp.getSymName() << " -> ";
      switch (entry.fingerprint.kind) {
      case SemanticFingerprint::Named:
        llvm::dbgs() << "Named(" << entry.fingerprint.opName << ")";
        break;
      case SemanticFingerprint::Generic:
        llvm::dbgs() << "Generic";
        break;
      case SemanticFingerprint::Identity:
        llvm::dbgs() << "Identity";
        break;
      }
      llvm::dbgs() << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Top-level semantic matching
//===----------------------------------------------------------------------===//

LogicalResult
mlir::act::runSemanticMatching(ModuleOp module,
                               SmallVectorImpl<MatchCandidate> &results) {
  LLVM_DEBUG(llvm::dbgs() << "Building instruction catalog...\n");
  auto catalog = InstructionCatalog::build(module);
  LLVM_DEBUG(catalog.dump());

  LLVM_DEBUG(llvm::dbgs() << "\nMatching source ops...\n");
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Only match linalg ops (the compute ops)
      if (!isa<linalg::LinalgOp>(op))
        return;
      // Skip linalg ops inside act.define regions
      if (op->getParentOfType<DefineOp>())
        return;

      auto matches = catalog.match(op);
      if (matches.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  [unmatched] " << op->getName() << " at "
                                << op->getLoc() << "\n");
      } else {
        for (auto &m : matches) {
          LLVM_DEBUG(llvm::dbgs() << "  [matched] " << op->getName() << " -> @"
                                  << m.instruction.getSymName() << "\n");
          results.push_back(std::move(m));
        }
      }
    });
  });

  return success();
}
