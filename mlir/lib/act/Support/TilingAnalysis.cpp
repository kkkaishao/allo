#include "act/Support/TilingAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tiling-analysis"

using namespace mlir;
using namespace mlir::act;

//===----------------------------------------------------------------------===//
// Buffer type lookup
//===----------------------------------------------------------------------===//

/// Look up the buffer types for all src/dst operands of a DefineOp.
static SmallVector<BufferTypeInterface> getBufferTypes(DefineOp defineOp,
                                                       ModuleOp module) {
  SmallVector<BufferTypeInterface> types;
  auto lookup = [&](FlatSymbolRefAttr ref) {
    auto bufOp =
        SymbolTable::lookupNearestSymbolFrom<DeclareBufferOp>(module, ref);
    assert(bufOp && "buffer not found — should have been caught by verifier");
    types.push_back(bufOp.getBufferType());
  };
  for (auto src : defineOp.getSources().getAsRange<FlatSymbolRefAttr>())
    lookup(src);
  for (auto dst : defineOp.getDestinations().getAsRange<FlatSymbolRefAttr>())
    lookup(dst);
  return types;
}

//===----------------------------------------------------------------------===//
// Phase 2a: Symbolic iteration domain extraction
//===----------------------------------------------------------------------===//

/// Extract symbolic shapes for all operands of a DefineOp by walking
/// the addr region's access chains.
static FailureOr<SmallVector<SymShape>> extractSymbolicShapes(DefineOp defineOp,
                                                              ModuleOp module) {
  auto bufferTypes = getBufferTypes(defineOp, module);
  Block &addrBlock = defineOp.getAccessBlock();
  auto *yieldOp = addrBlock.getTerminator();

  SmallVector<SymShape> shapes;
  for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
    Operation *accessOp = yieldOp->getOperand(i).getDefiningOp();
    if (!accessOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] yield operand " << i << " has no defining op\n");
      return failure();
    }
    auto shape = generateShapeExpr(accessOp, bufferTypes[i]);
    if (failed(shape)) {
      LLVM_DEBUG(llvm::dbgs() << "  [error] failed to generate shape expr for "
                                 "operand "
                              << i << "\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "  operand " << i
                            << " shape: " << symShapeToString(*shape) << "\n");
    shapes.push_back(std::move(*shape));
  }
  return shapes;
}

/// Map symbolic shapes to iteration domain bounds using the compute region's
/// linalg op indexing maps.
///
/// For each operand dimension, the indexing map tells which iteration dim it
/// corresponds to. We build iterDomain[d] = symExpr from the first operand
/// that constrains it.
static FailureOr<SmallVector<SymExpr>>
extractSymbolicIterationDomain(DefineOp defineOp,
                               ArrayRef<SymShape> symShapes) {
  // Find the linalg op in the compute region
  Block &computeBlock = defineOp.getSemanticsBlock();
  linalg::LinalgOp linalgOp = nullptr;
  for (Operation &op : computeBlock) {
    if (auto lOp = dyn_cast<linalg::LinalgOp>(&op)) {
      linalgOp = lOp;
      break;
    }
  }
  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs() << "  [skip] no linalg op in compute region of @"
                            << defineOp.getSymName() << "\n");
    return failure();
  }

  auto indexingMaps = linalgOp.getIndexingMapsArray();
  unsigned numIterDims = indexingMaps[0].getNumDims();

  // Initialize iteration domain with "unset" markers
  SmallVector<std::optional<SymExpr>> iterDomain(numIterDims, std::nullopt);

  unsigned numOperands = symShapes.size();
  for (unsigned i = 0; i < numOperands; ++i) {
    if (i >= indexingMaps.size()) {
      LLVM_DEBUG(llvm::dbgs() << "  [warning] more sym shapes than indexing "
                                 "maps, skipping operand "
                              << i << "\n");
      continue;
    }
    AffineMap map = indexingMaps[i];
    const SymShape &shape = symShapes[i];

    // Verify rank consistency
    if (map.getNumResults() != shape.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] operand " << i << " indexing map has "
                 << map.getNumResults() << " results but shape has "
                 << shape.size() << " dims\n");
      return failure();
    }

    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      auto expr = map.getResult(j);
      // Only handle simple dimension projections: d_k
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(llvm::dbgs() << "  [skip] non-dim-projection in operand "
                                << i << " dim " << j << "\n");
        continue;
      }
      unsigned iterDimIdx = dimExpr.getPosition();
      if (!iterDomain[iterDimIdx]) {
        iterDomain[iterDimIdx] = shape[j];
      }
      // Could check consistency here — if already set, verify equivalence
    }
  }

  // Convert to non-optional
  SmallVector<SymExpr> result;
  for (unsigned d = 0; d < numIterDims; ++d) {
    if (!iterDomain[d]) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] iteration dim " << d << " unconstrained\n");
      return failure();
    }
    result.push_back(std::move(*iterDomain[d]));
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Phase 2a: Source iteration domain extraction
//===----------------------------------------------------------------------===//

/// Get concrete source iteration domain bounds from a linalg op.
/// Uses the built-in getStaticLoopRanges() from IndexingMapOpInterface.
static FailureOr<SmallVector<int64_t>>
getSourceIterationDomain(linalg::LinalgOp sourceOp) {
  SmallVector<int64_t> loopRanges = sourceOp.getStaticLoopRanges();
  // MVP: require all static dimensions
  for (int64_t r : loopRanges) {
    if (ShapedType::isDynamic(r))
      return failure();
  }
  return loopRanges;
}

/// Get iterator types (parallel/reduction) from a linalg op.
static SmallVector<utils::IteratorType>
getIteratorTypes(linalg::LinalgOp linalgOp) {
  return linalgOp.getIteratorTypesArray();
}

//===----------------------------------------------------------------------===//
// Phase 2a: Constraint solving + tiling factor computation
//===----------------------------------------------------------------------===//

/// Solve the constraint system: for each iteration dim, equate the symbolic
/// instruction bound with the concrete source bound.
///
/// MVP solver handles:
/// - Constant bounds: direct comparison, compute tile factor
/// - Single Param bounds: assign param = source_bound
/// - Compound expressions: evaluate after params are solved
static FailureOr<TilingScheme>
computeTilingScheme(ArrayRef<utils::IteratorType> iterTypes,
                    ArrayRef<SymExpr> symbolicIterDomain,
                    ArrayRef<int64_t> sourceIterDomain) {
  assert(symbolicIterDomain.size() == sourceIterDomain.size());
  unsigned numDims = symbolicIterDomain.size();
  assert(iterTypes.size() == numDims);

  TilingScheme scheme;

  // Pass 1: Solve simple constraints (Constant, single Param)
  for (unsigned d = 0; d < numDims; ++d) {
    const SymExpr &instrBound = symbolicIterDomain[d];
    int64_t sourceBound = sourceIterDomain[d];

    if (auto paramIdx = instrBound.getParamIdx()) {
      // Param constraint: param should be set to source bound
      auto it = scheme.solvedParams.find(*paramIdx);
      if (it != scheme.solvedParams.end()) {
        // Already constrained — check consistency
        if (it->second != sourceBound) {
          LLVM_DEBUG(llvm::dbgs() << "  [infeasible] param " << *paramIdx
                                  << " constrained to " << it->second << " and "
                                  << sourceBound << "\n");
          return failure(); // inconsistent constraints
        }
      } else {
        scheme.solvedParams[*paramIdx] = sourceBound;
      }
    }
  }

  // Pass 2: Evaluate all bounds with solved params, compute tiling factors
  // Build param values array for evaluation
  // Find max param index to size the array
  unsigned maxParamIdx = 0;
  for (auto &[idx, _] : scheme.solvedParams)
    maxParamIdx = std::max(maxParamIdx, idx);
  SmallVector<int64_t> paramValues(maxParamIdx + 1, 0);
  for (auto &[idx, val] : scheme.solvedParams)
    paramValues[idx] = val;

  for (unsigned d = 0; d < numDims; ++d) {
    const SymExpr &instrBound = symbolicIterDomain[d];
    int64_t sourceBound = sourceIterDomain[d];

    TilingScheme::DimTiling dim;
    dim.sourceBound = sourceBound;
    dim.nativeBound = instrBound;
    dim.iterType = iterTypes[d];

    // Evaluate the native bound
    if (instrBound.isConstant()) {
      dim.nativeValue = instrBound.value;
    } else {
      // Evaluate with solved params — may fail if params are missing
      DenseSet<unsigned> neededParams;
      instrBound.collectParams(neededParams);
      bool allSolved = true;
      for (unsigned p : neededParams) {
        if (!scheme.solvedParams.count(p)) {
          allSolved = false;
          break;
        }
      }
      if (!allSolved) {
        LLVM_DEBUG(llvm::dbgs() << "  [infeasible] dim " << d
                                << " has unsolved params in bound: "
                                << instrBound.toString() << "\n");
        return failure();
      }
      dim.nativeValue = instrBound.evaluate(paramValues);
    }

    // Compute tiling factor
    if (dim.nativeValue <= 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] dim " << d << " has non-positive native bound: "
                 << dim.nativeValue << "\n");
      return failure();
    }
    if (sourceBound > dim.nativeValue) {
      dim.tileFactor = (sourceBound + dim.nativeValue - 1) / dim.nativeValue;
      dim.needsPadding = (sourceBound % dim.nativeValue) != 0;
    } else {
      dim.tileFactor = 1;
      dim.needsPadding = false;
    }

    scheme.dims.push_back(std::move(dim));
  }

  return scheme;
}

//===----------------------------------------------------------------------===//
// Phase 2b: Addr param classification
//===----------------------------------------------------------------------===//

/// Classify addr params as Shape or Offset by examining where they appear
/// in the access chains.
static DenseMap<unsigned, AddrParamKind> classifyAddrParams(DefineOp defineOp) {
  DenseMap<unsigned, AddrParamKind> kinds;
  Block &addrBlock = defineOp.getAccessBlock();

  auto markParam = [&](unsigned idx, AddrParamKind newKind) {
    auto it = kinds.find(idx);
    if (it == kinds.end()) {
      kinds[idx] = newKind;
    } else if (it->second != newKind) {
      kinds[idx] = AddrParamKind::Mixed;
    }
  };

  // Walk all access ops in the addr block
  for (Operation &op : addrBlock) {
    if (auto strided = dyn_cast<StridedOp>(&op)) {
      MLIRContext *ctx = strided.getContext();
      // Basis params → Offset
      auto mixedBasis =
          getMixedValues(strided.getStaticBasis(), strided.getBasis(), ctx);
      for (auto &ofr : mixedBasis) {
        if (auto v = dyn_cast<Value>(ofr)) {
          DenseSet<unsigned> params;
          auto expr = buildSymExpr(v);
          if (succeeded(expr))
            expr->collectParams(params);
          for (unsigned p : params)
            markParam(p, AddrParamKind::Offset);
        }
      }
      // Counts params → Shape
      auto mixedCounts =
          getMixedValues(strided.getStaticCounts(), strided.getCounts(), ctx);
      for (auto &ofr : mixedCounts) {
        if (auto v = dyn_cast<Value>(ofr)) {
          DenseSet<unsigned> params;
          auto expr = buildSymExpr(v);
          if (succeeded(expr))
            expr->collectParams(params);
          for (unsigned p : params)
            markParam(p, AddrParamKind::Shape);
        }
      }
    } else if (auto expand = dyn_cast<ExpandShapeOp>(&op)) {
      // Output shape params → Shape
      auto mixedOutput =
          getMixedValues(expand.getStaticOutputShape(), expand.getOutputShape(),
                         expand.getContext());
      for (auto &ofr : mixedOutput) {
        if (auto v = dyn_cast<Value>(ofr)) {
          DenseSet<unsigned> params;
          auto expr = buildSymExpr(v);
          if (succeeded(expr))
            expr->collectParams(params);
          for (unsigned p : params)
            markParam(p, AddrParamKind::Shape);
        }
      }
    }
    // CollapseShapeOp and TransposeOp introduce no new params
  }

  return kinds;
}

//===----------------------------------------------------------------------===//
// Contiguity validation
//===----------------------------------------------------------------------===//

/// Check that every operand's tile is contiguous in row-major memory layout.
/// A tile is contiguous iff, scanning from innermost dimension outward, once
/// a tiled dimension is found, no outer dimension may also be tiled.
static bool areTilesContiguous(linalg::LinalgOp sourceOp,
                               const TilingScheme &tiling) {
  auto indexingMaps = sourceOp.getIndexingMapsArray();
  for (unsigned opIdx = 0; opIdx < sourceOp->getNumOperands(); ++opIdx) {
    auto tensorType =
        dyn_cast<RankedTensorType>(sourceOp->getOperand(opIdx).getType());
    if (!tensorType)
      continue;
    AffineMap map = indexingMaps[opIdx];
    auto shape = tensorType.getShape();
    unsigned rank = tensorType.getRank();

    // Compute effective tile size per operand dimension.
    SmallVector<int64_t> tileSize(rank);
    for (unsigned j = 0; j < rank; ++j) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(j));
      if (!dimExpr) {
        // Not a simple dim projection — conservatively treat as tiled.
        tileSize[j] = 1;
        continue;
      }
      unsigned iterDim = dimExpr.getPosition();
      if (iterDim < tiling.dims.size() && tiling.dims[iterDim].tileFactor > 1)
        tileSize[j] = tiling.dims[iterDim].nativeValue;
      else
        tileSize[j] = shape[j]; // untiled
    }

    // Scan innermost to outermost: once a tiled dim is found, no outer dim
    // may also be tiled.
    bool foundTiled = false;
    for (int j = rank - 1; j >= 0; --j) {
      bool isTiled = (tileSize[j] != shape[j]);
      if (foundTiled && isTiled) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [non-contiguous] operand " << opIdx << " dim " << j
                   << " (tile=" << tileSize[j] << " shape=" << shape[j]
                   << ") conflicts with inner tiled dim\n");
        return false;
      }
      if (isTiled)
        foundTiled = true;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Top-level: runTilingAnalysis
//===----------------------------------------------------------------------===//

LogicalResult
act::runTilingAnalysis(ModuleOp module, ArrayRef<MatchCandidate> matches,
                       SmallVectorImpl<TiledMatchCandidate> &results) {
  LLVM_DEBUG(llvm::dbgs() << "\n=== Tiling Analysis (Stage 2) ===\n");

  for (auto &match : matches) {
    DefineOp defineOp = match.instruction;
    auto *sourceOp = match.sourceOp;
    auto sourceLinalgOp = dyn_cast<linalg::LinalgOp>(sourceOp);
    if (!sourceLinalgOp) {
      LLVM_DEBUG(llvm::dbgs() << "  [skip] non-linalg source op: "
                              << sourceOp->getName() << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "\nAnalyzing: " << sourceOp->getName() << " -> @"
                            << defineOp.getSymName() << "\n");

    // Phase 2a: Extract symbolic shapes
    LLVM_DEBUG(llvm::dbgs() << "  Extracting symbolic shapes...\n");
    auto symShapes = extractSymbolicShapes(defineOp, module);
    if (failed(symShapes)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [fail] could not extract symbolic shapes\n");
      {
        TiledMatchCandidate tm;
        tm.base = match;
        tm.isValid = false;
        results.push_back(std::move(tm));
      }
      continue;
    }

    // Phase 2a: Map to iteration domain
    LLVM_DEBUG(llvm::dbgs() << "  Mapping to iteration domain...\n");
    auto symbolicIterDomain =
        extractSymbolicIterationDomain(defineOp, *symShapes);
    if (failed(symbolicIterDomain)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [fail] could not extract symbolic iteration domain\n");
      {
        TiledMatchCandidate tm;
        tm.base = match;
        tm.isValid = false;
        results.push_back(std::move(tm));
      }
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Symbolic iteration domain: [";
      for (unsigned i = 0; i < symbolicIterDomain->size(); ++i) {
        if (i > 0)
          llvm::dbgs() << ", ";
        llvm::dbgs() << (*symbolicIterDomain)[i].toString();
      }
      llvm::dbgs() << "]\n";
    });

    // Phase 2a: Get source iteration domain
    auto sourceIterDomain = getSourceIterationDomain(sourceLinalgOp);
    if (failed(sourceIterDomain)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [fail] could not extract source iteration domain "
                    "(dynamic shapes not supported in MVP)\n");
      {
        TiledMatchCandidate tm;
        tm.base = match;
        tm.isValid = false;
        results.push_back(std::move(tm));
      }
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Source iteration domain: [";
      for (unsigned i = 0; i < sourceIterDomain->size(); ++i) {
        if (i > 0)
          llvm::dbgs() << ", ";
        llvm::dbgs() << (*sourceIterDomain)[i];
      }
      llvm::dbgs() << "]\n";
    });

    // Check rank match
    unsigned instrRank = symbolicIterDomain->size();
    unsigned sourceRank = sourceIterDomain->size();
    unsigned numOuterDims = match.numOuterDims;

    if (sourceRank != instrRank) {
      // Rank mismatch: handle if this is a structural match with outer dims
      if (numOuterDims == 0 || sourceRank != instrRank + numOuterDims) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [fail] rank mismatch: instruction has " << instrRank
                   << " iter dims, source has " << sourceRank
                   << " (numOuterDims=" << numOuterDims << ")\n");
        {
          TiledMatchCandidate tm;
          tm.base = match;
          tm.isValid = false;
          results.push_back(std::move(tm));
        }
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "  Handling rank mismatch: " << numOuterDims
                              << " outer dims\n");

      auto sourceIterTypes = getIteratorTypes(sourceLinalgOp);

      // Extract inner portions (suffix)
      ArrayRef<int64_t> innerSourceBounds(
          sourceIterDomain->data() + numOuterDims, instrRank);
      ArrayRef<utils::IteratorType> innerIterTypes(
          sourceIterTypes.data() + numOuterDims, instrRank);

      // Compute inner tiling scheme
      auto innerTilingScheme = computeTilingScheme(
          innerIterTypes, *symbolicIterDomain, innerSourceBounds);
      if (failed(innerTilingScheme)) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "  [infeasible] inner tiling constraint solving failed\n");
        {
          TiledMatchCandidate tm;
          tm.base = match;
          tm.isValid = false;
          results.push_back(std::move(tm));
        }
        continue;
      }

      // Build full tiling scheme: prepend outer dims
      TilingScheme fullScheme;
      fullScheme.solvedParams = innerTilingScheme->solvedParams;

      for (unsigned d = 0; d < numOuterDims; ++d) {
        TilingScheme::DimTiling outerDim;
        outerDim.sourceBound = (*sourceIterDomain)[d];
        outerDim.nativeBound = SymExpr::constant(1);
        outerDim.nativeValue = 1;
        outerDim.tileFactor = (*sourceIterDomain)[d];
        outerDim.needsPadding = false;
        outerDim.iterType = sourceIterTypes[d];
        fullScheme.dims.push_back(std::move(outerDim));
      }
      for (auto &dim : innerTilingScheme->dims)
        fullScheme.dims.push_back(std::move(dim));

      LLVM_DEBUG({
        llvm::dbgs() << "  Rank-mismatch tiling: outer=[";
        for (unsigned d = 0; d < numOuterDims; ++d) {
          if (d > 0)
            llvm::dbgs() << ",";
          llvm::dbgs() << fullScheme.dims[d].sourceBound;
        }
        llvm::dbgs() << "] inner=[";
        for (unsigned d = numOuterDims; d < fullScheme.dims.size(); ++d) {
          if (d > numOuterDims)
            llvm::dbgs() << ",";
          auto &dim = fullScheme.dims[d];
          llvm::dbgs() << dim.sourceBound << "/" << dim.nativeValue << "="
                       << dim.tileFactor;
        }
        llvm::dbgs() << "]\n";
      });

      // Validate tile contiguity
      if (!areTilesContiguous(sourceLinalgOp, fullScheme)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [invalid] non-contiguous tiles detected\n");
        TiledMatchCandidate tm;
        tm.base = match;
        tm.isValid = false;
        results.push_back(std::move(tm));
        continue;
      }

      auto paramKinds = classifyAddrParams(defineOp);
      {
        TiledMatchCandidate tm;
        tm.base = match;
        tm.tiling = std::move(fullScheme);
        tm.paramKinds = std::move(paramKinds);
        tm.isValid = true;
        tm.numOuterDims = numOuterDims;
        results.push_back(std::move(tm));
      }
      continue;
    }

    // Phase 2a: Solve constraints + compute tiling (same-rank path)
    auto tilingScheme =
        computeTilingScheme(getIteratorTypes(sourceLinalgOp),
                            *symbolicIterDomain, *sourceIterDomain);
    if (failed(tilingScheme)) {
      LLVM_DEBUG(llvm::dbgs() << "  [infeasible] constraint solving failed\n");
      {
        TiledMatchCandidate tm;
        tm.base = match;
        tm.isValid = false;
        results.push_back(std::move(tm));
      }
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Solved params: {";
      bool first = true;
      for (auto &[idx, val] : tilingScheme->solvedParams) {
        if (!first)
          llvm::dbgs() << ", ";
        llvm::dbgs() << "p" << idx << "=" << val;
        first = false;
      }
      llvm::dbgs() << "}\n";
      llvm::dbgs() << "  Tiling factors: [";
      for (unsigned i = 0; i < tilingScheme->dims.size(); ++i) {
        if (i > 0)
          llvm::dbgs() << ", ";
        auto &dim = tilingScheme->dims[i];
        llvm::dbgs() << dim.sourceBound << "/" << dim.nativeValue << "="
                     << dim.tileFactor;
        if (dim.needsPadding)
          llvm::dbgs() << "(pad)";
      }
      llvm::dbgs() << "]\n";
    });

    // Validate tile contiguity
    if (!areTilesContiguous(sourceLinalgOp, *tilingScheme)) {
      LLVM_DEBUG(llvm::dbgs() << "  [invalid] non-contiguous tiles detected\n");
      TiledMatchCandidate tm;
      tm.base = match;
      tm.isValid = false;
      results.push_back(std::move(tm));
      continue;
    }

    // Phase 2b: Classify addr params
    auto paramKinds = classifyAddrParams(defineOp);
    LLVM_DEBUG({
      llvm::dbgs() << "  Param kinds: {";
      bool first = true;
      for (auto &[idx, kind] : paramKinds) {
        if (!first)
          llvm::dbgs() << ", ";
        llvm::dbgs() << "p" << idx << "=";
        switch (kind) {
        case AddrParamKind::Shape:
          llvm::dbgs() << "shape";
          break;
        case AddrParamKind::Offset:
          llvm::dbgs() << "offset";
          break;
        case AddrParamKind::Mixed:
          llvm::dbgs() << "mixed";
          break;
        }
        first = false;
      }
      llvm::dbgs() << "}\n";
    });

    {
      TiledMatchCandidate tm;
      tm.base = match;
      tm.tiling = std::move(*tilingScheme);
      tm.paramKinds = std::move(paramKinds);
      tm.isValid = true;
      results.push_back(std::move(tm));
    }
  }

  return success();
}
