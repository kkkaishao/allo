#include "act/Support/SemanticMatching.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
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

template <typename SliceOp>
static std::optional<StaticSliceSpec> getStaticSliceSpec(SliceOp op) {
  StaticSliceSpec spec;
  auto staticOffsets = op.getStaticOffsets();
  auto staticSizes = op.getStaticSizes();
  auto staticStrides = op.getStaticStrides();
  if (llvm::any_of(staticOffsets,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticSizes,
                   [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(staticStrides,
                   [](int64_t v) { return v == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  spec.offsets.assign(staticOffsets.begin(), staticOffsets.end());
  spec.sizes.assign(staticSizes.begin(), staticSizes.end());
  spec.strides.assign(staticStrides.begin(), staticStrides.end());
  return spec;
}

static void collectInputBoundaryAnnotations(
    Value value, Operation *computeOp, unsigned operandIdx,
    SmallVectorImpl<EdgeLayoutAnnotation> &layoutAnnotations) {
  SmallVector<EdgeLayoutAnnotation, 2> reversedAnnotations;
  Value current = value;

  while (Operation *defOp = current.getDefiningOp()) {
    if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp)) {
      EdgeLayoutAnnotation ann;
      ann.direction = EdgeLayoutDirection::Input;
      ann.transformKind = EdgeLayoutTransformKind::Transpose;
      ann.layoutOp = transposeOp;
      ann.computeOp = computeOp;
      ann.edgeIdx = operandIdx;
      auto perm = transposeOp.getPermutation();
      ann.permutation.assign(perm.begin(), perm.end());
      reversedAnnotations.push_back(std::move(ann));
      current = transposeOp.getInput();
      continue;
    }

    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      if (auto sliceSpec = getStaticSliceSpec(extractSliceOp)) {
        EdgeLayoutAnnotation ann;
        ann.direction = EdgeLayoutDirection::Input;
        ann.transformKind = EdgeLayoutTransformKind::ExtractSlice;
        ann.layoutOp = extractSliceOp;
        ann.computeOp = computeOp;
        ann.edgeIdx = operandIdx;
        ann.sliceSpec = *sliceSpec;
        reversedAnnotations.push_back(std::move(ann));
      }
      current = extractSliceOp.getSource();
      continue;
    }

    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
      current = expandOp.getSrc();
      continue;
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
      current = collapseOp.getSrc();
      continue;
    }
    break;
  }

  for (unsigned i = 0; i < reversedAnnotations.size(); ++i) {
    auto ann =
        std::move(reversedAnnotations[reversedAnnotations.size() - 1 - i]);
    ann.transformOrder = i;
    layoutAnnotations.push_back(std::move(ann));
  }
}

static void collectOutputBoundaryAnnotations(
    Operation *computeOp,
    SmallVectorImpl<EdgeLayoutAnnotation> &layoutAnnotations) {
  for (auto [resultIdx, result] : llvm::enumerate(computeOp->getResults())) {
    for (Operation *user : result.getUsers()) {
      auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(user);
      if (!insertSliceOp || insertSliceOp.getSource() != result)
        continue;
      if (auto sliceSpec = getStaticSliceSpec(insertSliceOp)) {
        EdgeLayoutAnnotation ann;
        ann.direction = EdgeLayoutDirection::Output;
        ann.transformKind = EdgeLayoutTransformKind::InsertSlice;
        ann.layoutOp = insertSliceOp;
        ann.computeOp = computeOp;
        ann.edgeIdx = resultIdx;
        ann.transformOrder = 0;
        ann.sliceSpec = *sliceSpec;
        layoutAnnotations.push_back(std::move(ann));
      }
    }
  }
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
// Compute region linalg op extraction
//===----------------------------------------------------------------------===//

/// Find the single linalg op in a DefineOp's compute region, or nullptr.
static linalg::LinalgOp findComputeLinalgOp(DefineOp defineOp) {
  Block &block = defineOp.getSemanticsBlock();
  auto coreOps = collectCoreOps(block);
  if (coreOps.size() != 1)
    return nullptr;
  return dyn_cast<linalg::LinalgOp>(coreOps[0]);
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
    // Identity instructions (data movement) never match source compute ops.
    if (fp.kind == SemanticFingerprint::Identity)
      return;
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

LogicalResult mlir::act::runSemanticMatching(
    ModuleOp module, SmallVectorImpl<MatchCandidate> &results,
    SmallVectorImpl<EdgeLayoutAnnotation> &layoutAnnotations) {
  LLVM_DEBUG(llvm::dbgs() << "Building instruction catalog...\n");
  auto catalog = InstructionCatalog::build(module);
  LLVM_DEBUG(catalog.dump());

  LLVM_DEBUG(llvm::dbgs() << "\nMatching source ops...\n");
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Only match linalg compute ops (skip layout ops like transpose)
      if (!isa<linalg::LinalgOp>(op))
        return;
      if (isLayoutOp(op))
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

  // Collect layout ops as edge annotations
  LLVM_DEBUG(llvm::dbgs() << "\nCollecting layout edge annotations...\n");
  module.walk([&](func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      if (op->getParentOfType<DefineOp>() || !isa<linalg::LinalgOp>(op) ||
          isLayoutOp(op))
        return;

      auto linalgOp = cast<linalg::LinalgOp>(op);
      for (auto [idx, input] : llvm::enumerate(linalgOp.getDpsInputs()))
        collectInputBoundaryAnnotations(input, op, idx, layoutAnnotations);
      collectOutputBoundaryAnnotations(op, layoutAnnotations);
    });
  });

  LLVM_DEBUG({
    for (auto &ann : layoutAnnotations) {
      llvm::dbgs() << "  [edge] "
                   << (ann.direction == EdgeLayoutDirection::Input ? "input "
                                                                   : "output ")
                   << ann.computeOp->getName() << " edge " << ann.edgeIdx
                   << " order " << ann.transformOrder << " ";
      switch (ann.transformKind) {
      case EdgeLayoutTransformKind::Transpose:
        llvm::dbgs() << "transpose[";
        for (unsigned i = 0; i < ann.permutation.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << ann.permutation[i];
        }
        llvm::dbgs() << "]";
        break;
      case EdgeLayoutTransformKind::ExtractSlice:
      case EdgeLayoutTransformKind::InsertSlice:
        llvm::dbgs() << (ann.transformKind ==
                                 EdgeLayoutTransformKind::ExtractSlice
                             ? "extract_slice"
                             : "insert_slice")
                     << " offsets=[";
        for (unsigned i = 0; i < ann.sliceSpec.offsets.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << ann.sliceSpec.offsets[i];
        }
        llvm::dbgs() << "] sizes=[";
        for (unsigned i = 0; i < ann.sliceSpec.sizes.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << ann.sliceSpec.sizes[i];
        }
        llvm::dbgs() << "] strides=[";
        for (unsigned i = 0; i < ann.sliceSpec.strides.size(); ++i) {
          if (i)
            llvm::dbgs() << ",";
          llvm::dbgs() << ann.sliceSpec.strides[i];
        }
        llvm::dbgs() << "]";
        break;
      }
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Structural suffix matching (rank mismatch)
//===----------------------------------------------------------------------===//

/// Check if instrTypes is a suffix of sourceTypes with all-parallel prefix.
/// Returns the offset (number of outer dims) or std::nullopt on failure.
static std::optional<unsigned>
checkIteratorTypeSuffix(ArrayRef<utils::IteratorType> sourceTypes,
                        ArrayRef<utils::IteratorType> instrTypes) {
  if (sourceTypes.size() <= instrTypes.size())
    return std::nullopt;
  unsigned offset = sourceTypes.size() - instrTypes.size();
  // Check suffix match
  for (unsigned i = 0; i < instrTypes.size(); ++i) {
    if (sourceTypes[offset + i] != instrTypes[i])
      return std::nullopt;
  }
  // Check prefix is all parallel
  for (unsigned i = 0; i < offset; ++i) {
    if (sourceTypes[i] != utils::IteratorType::parallel)
      return std::nullopt;
  }
  return offset;
}

/// Check indexing map compatibility: strip results referencing batch dims
/// from source maps, reindex remaining dims, and compare with instruction maps.
static bool checkIndexingMapCompatibility(linalg::LinalgOp sourceOp,
                                          linalg::LinalgOp instrOp,
                                          unsigned offset) {
  auto sourceMaps = sourceOp.getIndexingMapsArray();
  auto instrMaps = instrOp.getIndexingMapsArray();
  if (sourceMaps.size() != instrMaps.size())
    return false;

  MLIRContext *ctx = sourceOp.getContext();
  unsigned sourceNumDims = sourceMaps[0].getNumDims();
  unsigned instrNumDims = instrMaps[0].getNumDims();
  if (sourceNumDims != instrNumDims + offset)
    return false;

  // Build replacement: d_i -> d_{i-offset} for i >= offset
  SmallVector<AffineExpr> dimReplacements(sourceNumDims);
  for (unsigned i = 0; i < offset; ++i)
    dimReplacements[i] = getAffineConstantExpr(0, ctx);
  for (unsigned i = offset; i < sourceNumDims; ++i)
    dimReplacements[i] = getAffineDimExpr(i - offset, ctx);

  for (unsigned opIdx = 0; opIdx < sourceMaps.size(); ++opIdx) {
    AffineMap srcMap = sourceMaps[opIdx];
    AffineMap instrMap = instrMaps[opIdx];

    // Strip results that reference only batch dims (d0..d_{offset-1})
    SmallVector<AffineExpr> strippedResults;
    for (unsigned r = 0; r < srcMap.getNumResults(); ++r) {
      AffineExpr expr = srcMap.getResult(r);
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (dimExpr && dimExpr.getPosition() < offset)
        continue; // batch dim result — strip it
      strippedResults.push_back(expr);
    }

    if (strippedResults.size() != instrMap.getNumResults())
      return false;

    // Reindex: replace d_i with d_{i-offset}
    AffineMap strippedMap =
        AffineMap::get(sourceNumDims, 0, strippedResults, ctx)
            .replaceDimsAndSymbols(dimReplacements, {}, instrNumDims, 0);

    if (strippedMap != instrMap)
      return false;
  }
  return true;
}

/// Check body region equivalence between two linalg ops.
/// Compares the scalar body operations structurally: same op names, same
/// operand patterns (by block arg index or intra-body def position).
static bool checkBodyEquivalence(linalg::LinalgOp sourceOp,
                                 linalg::LinalgOp instrOp) {
  Region &sourceRegion = sourceOp->getRegion(0);
  Region &instrRegion = instrOp->getRegion(0);

  if (sourceRegion.empty() || instrRegion.empty())
    return false;

  Block &srcBlock = sourceRegion.front();
  Block &instrBlock = instrRegion.front();

  if (srcBlock.getNumArguments() != instrBlock.getNumArguments())
    return false;

  // Check block argument types match
  for (unsigned i = 0; i < srcBlock.getNumArguments(); ++i) {
    if (srcBlock.getArgument(i).getType() !=
        instrBlock.getArgument(i).getType())
      return false;
  }

  // Check same number of operations
  if (std::distance(srcBlock.begin(), srcBlock.end()) !=
      std::distance(instrBlock.begin(), instrBlock.end()))
    return false;

  // Build a value mapping: block args map by index, SSA results map by
  // matching definition order
  DenseMap<Value, unsigned> srcValueId;
  DenseMap<Value, unsigned> instrValueId;
  unsigned nextId = 0;

  // Map block args
  for (unsigned i = 0; i < srcBlock.getNumArguments(); ++i) {
    srcValueId[srcBlock.getArgument(i)] = nextId;
    instrValueId[instrBlock.getArgument(i)] = nextId;
    ++nextId;
  }

  // Walk ops in lockstep
  auto srcIt = srcBlock.begin();
  auto instrIt = instrBlock.begin();
  for (; srcIt != srcBlock.end(); ++srcIt, ++instrIt) {
    Operation &srcOp = *srcIt;
    Operation &instrOp = *instrIt;

    // Same op name
    if (srcOp.getName() != instrOp.getName())
      return false;

    // Same number of operands and results
    if (srcOp.getNumOperands() != instrOp.getNumOperands())
      return false;
    if (srcOp.getNumResults() != instrOp.getNumResults())
      return false;

    // Same attributes (ignoring location)
    if (srcOp.getAttrDictionary() != instrOp.getAttrDictionary())
      return false;

    // Check operands match by value ID
    for (unsigned i = 0; i < srcOp.getNumOperands(); ++i) {
      auto srcIdIt = srcValueId.find(srcOp.getOperand(i));
      auto instrIdIt = instrValueId.find(instrOp.getOperand(i));
      if (srcIdIt == srcValueId.end() || instrIdIt == instrValueId.end())
        return false;
      if (srcIdIt->second != instrIdIt->second)
        return false;
    }

    // Map results
    for (unsigned i = 0; i < srcOp.getNumResults(); ++i) {
      srcValueId[srcOp.getResult(i)] = nextId;
      instrValueId[instrOp.getResult(i)] = nextId;
      ++nextId;
    }
  }

  return true;
}

LogicalResult
mlir::act::runStructuralMatching(ModuleOp module,
                                 ArrayRef<Operation *> unmatchedOps,
                                 SmallVectorImpl<MatchCandidate> &results) {
  LLVM_DEBUG(llvm::dbgs() << "\n=== Structural Matching (Rank Mismatch) ===\n");

  if (unmatchedOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No unmatched ops to try.\n");
    return success();
  }

  // Collect all DefineOps
  SmallVector<DefineOp> defineOps;
  module.walk([&](DefineOp op) { defineOps.push_back(op); });

  for (Operation *sourceOp : unmatchedOps) {
    auto sourceLinalgOp = dyn_cast<linalg::LinalgOp>(sourceOp);
    if (!sourceLinalgOp)
      continue;

    auto sourceIterTypes = sourceLinalgOp.getIteratorTypesArray();

    for (DefineOp defineOp : defineOps) {
      linalg::LinalgOp instrLinalgOp = findComputeLinalgOp(defineOp);
      if (!instrLinalgOp)
        continue;

      auto instrIterTypes = instrLinalgOp.getIteratorTypesArray();

      // Check iterator type suffix match + all-parallel prefix
      auto offset = checkIteratorTypeSuffix(sourceIterTypes, instrIterTypes);
      if (!offset)
        continue;

      // Check body equivalence
      if (!checkBodyEquivalence(sourceLinalgOp, instrLinalgOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [skip] " << sourceOp->getName() << " vs @"
                   << defineOp.getSymName() << ": body mismatch\n");
        continue;
      }

      // Check indexing map compatibility
      if (!checkIndexingMapCompatibility(sourceLinalgOp, instrLinalgOp,
                                         *offset)) {
        LLVM_DEBUG(llvm::dbgs() << "  [skip] " << sourceOp->getName() << " vs @"
                                << defineOp.getSymName()
                                << ": indexing map incompatibility\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "  [structural-match] " << sourceOp->getName()
                              << " -> @" << defineOp.getSymName()
                              << " (outer dims=" << *offset << ")\n");

      MatchCandidate mc;
      mc.sourceOp = sourceOp;
      mc.instruction = defineOp;
      mc.numOuterDims = *offset;
      results.push_back(std::move(mc));
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "=== " << results.size()
                          << " structural match(es) found ===\n");
  return success();
}
