#include "act/Support/ParamSolving.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "param-solving"

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
// Symbolic iteration domain extraction
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

static FailureOr<SmallVector<AffineMap>>
getCarrierIndexingMaps(Operation *domainComputeOp) {
  MLIRContext *ctx = domainComputeOp->getContext();
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(domainComputeOp)) {
    auto indexingMaps = linalgOp.getIndexingMapsArray();
    return SmallVector<AffineMap>(indexingMaps.begin(), indexingMaps.end());
  }

  if (auto softmaxOp = dyn_cast<linalg::SoftmaxOp>(domainComputeOp)) {
    unsigned rank = softmaxOp.getInputOperandRank();
    SmallVector<AffineExpr> dims;
    dims.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dims.push_back(getAffineDimExpr(i, ctx));
    AffineMap identity = AffineMap::get(rank, 0, dims, ctx);
    return SmallVector<AffineMap>{identity, identity};
  }

  return domainComputeOp->emitError()
         << "domain carrier does not expose indexing maps";
}

/// Map symbolic shapes to iteration domain bounds using the compute region's
/// linalg op indexing maps.
static FailureOr<SmallVector<SymExpr>>
extractSymbolicIterationDomain(DefineOp defineOp, Operation *domainComputeOp,
                               ArrayRef<SymShape> symShapes) {
  auto indexingMaps = getCarrierIndexingMaps(domainComputeOp);
  if (failed(indexingMaps)) {
    LLVM_DEBUG(llvm::dbgs() << "  [skip] invalid domain carrier in compute "
                               "region of @"
                            << defineOp.getSymName() << "\n");
    return failure();
  }
  unsigned numIterDims = (*indexingMaps)[0].getNumDims();

  SmallVector<std::optional<SymExpr>> iterDomain(numIterDims, std::nullopt);

  unsigned numOperands = symShapes.size();
  for (unsigned i = 0; i < numOperands; ++i) {
    if (i >= indexingMaps->size()) {
      LLVM_DEBUG(llvm::dbgs() << "  [warning] more sym shapes than indexing "
                                 "maps, skipping operand "
                              << i << "\n");
      continue;
    }
    AffineMap map = (*indexingMaps)[i];
    const SymShape &shape = symShapes[i];

    if (map.getNumResults() != shape.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] operand " << i << " indexing map has "
                 << map.getNumResults() << " results but shape has "
                 << shape.size() << " dims\n");
      return failure();
    }

    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      auto expr = map.getResult(j);
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        LLVM_DEBUG(llvm::dbgs() << "  [skip] non-dim-projection in operand "
                                << i << " dim " << j << "\n");
        continue;
      }
      unsigned iterDimIdx = dimExpr.getPosition();
      if (!iterDomain[iterDimIdx])
        iterDomain[iterDimIdx] = shape[j];
    }
  }

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
// Source iteration domain extraction
//===----------------------------------------------------------------------===//

/// Get concrete source iteration domain bounds from a linalg op.
static FailureOr<SmallVector<int64_t>>
getSourceIterationDomain(Operation *sourceOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(sourceOp)) {
    SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
    for (int64_t r : loopRanges) {
      if (ShapedType::isDynamic(r))
        return failure();
    }
    return loopRanges;
  }

  if (auto softmaxOp = dyn_cast<linalg::SoftmaxOp>(sourceOp)) {
    auto inputType = dyn_cast<RankedTensorType>(softmaxOp.getInput().getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();
    return SmallVector<int64_t>(inputType.getShape().begin(),
                                inputType.getShape().end());
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Constraint solving (reject on mismatch — no tiling)
//===----------------------------------------------------------------------===//

/// Solve the constraint system: for each iteration dim, equate the symbolic
/// instruction bound with the concrete source bound.
///
/// Unlike the previous tiling analysis, this rejects when source shapes exceed
/// instruction capacity instead of computing tiling factors.
static FailureOr<DenseMap<unsigned, int64_t>>
solveParams(ArrayRef<SymExpr> symbolicIterDomain,
            ArrayRef<int64_t> sourceIterDomain) {
  assert(symbolicIterDomain.size() == sourceIterDomain.size());
  unsigned numDims = symbolicIterDomain.size();

  DenseMap<unsigned, int64_t> solvedParams;

  // Pass 1: Solve simple constraints (single Param)
  for (unsigned d = 0; d < numDims; ++d) {
    const SymExpr &instrBound = symbolicIterDomain[d];
    int64_t sourceBound = sourceIterDomain[d];

    if (auto paramIdx = instrBound.getParamIdx()) {
      auto it = solvedParams.find(*paramIdx);
      if (it != solvedParams.end()) {
        if (it->second != sourceBound) {
          LLVM_DEBUG(llvm::dbgs() << "  [infeasible] param " << *paramIdx
                                  << " constrained to " << it->second << " and "
                                  << sourceBound << "\n");
          return failure();
        }
      } else {
        solvedParams[*paramIdx] = sourceBound;
      }
    }
  }

  // Pass 2: Evaluate all bounds with solved params, reject on mismatch
  unsigned maxParamIdx = 0;
  for (auto &[idx, _] : solvedParams)
    maxParamIdx = std::max(maxParamIdx, idx);
  SmallVector<int64_t> paramValues(maxParamIdx + 1, 0);
  for (auto &[idx, val] : solvedParams)
    paramValues[idx] = val;

  for (unsigned d = 0; d < numDims; ++d) {
    const SymExpr &instrBound = symbolicIterDomain[d];
    int64_t sourceBound = sourceIterDomain[d];

    int64_t nativeValue;
    if (instrBound.isConstant()) {
      nativeValue = instrBound.value;
    } else {
      DenseSet<unsigned> neededParams;
      instrBound.collectParams(neededParams);
      for (unsigned p : neededParams) {
        if (!solvedParams.count(p)) {
          LLVM_DEBUG(llvm::dbgs() << "  [infeasible] dim " << d
                                  << " has unsolved params in bound: "
                                  << instrBound.toString() << "\n");
          return failure();
        }
      }
      nativeValue = instrBound.evaluate(paramValues);
    }

    if (nativeValue <= 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [error] dim " << d
                 << " has non-positive native bound: " << nativeValue << "\n");
      return failure();
    }

    if (sourceBound != nativeValue) {
      LLVM_DEBUG(llvm::dbgs() << "  [reject] shape mismatch: dim " << d
                              << " source=" << sourceBound
                              << " native=" << nativeValue << "\n");
      return failure();
    }
  }

  return solvedParams;
}

//===----------------------------------------------------------------------===//
// Addr param classification
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

  for (Operation &op : addrBlock) {
    if (auto strided = dyn_cast<StridedOp>(&op)) {
      MLIRContext *ctx = strided.getContext();
      // Basis params -> Offset
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
      // Counts params -> Shape
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
// Top-level: runParamSolving
//===----------------------------------------------------------------------===//

/// Analyze a single SemanticsGraphNode: extract symbolic shapes from the
/// instruction, get source iteration domain from the anchor op, solve
/// constraints. Reject if shapes don't fit (no tiling).
static ParamSolution solveNode(SemanticsGraphNode &node, ModuleOp module) {
  ParamSolution ps;
  ps.node = &node;

  DefineOp defineOp = node.instruction;
  auto *domainComputeOp = node.domainComputeOp;
  if (!isa_and_nonnull<linalg::LinalgOp, linalg::SoftmaxOp>(domainComputeOp)) {
    LLVM_DEBUG(llvm::dbgs() << "  [skip] invalid domain compute op for @"
                            << defineOp.getSymName() << "\n");
    return ps;
  }

  auto *domainSourceOp = node.domainSourceOp;
  if (!isa_and_nonnull<linalg::LinalgOp, linalg::SoftmaxOp>(domainSourceOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "  [skip] invalid domain source op";
      if (domainSourceOp)
        llvm::dbgs() << ": " << domainSourceOp->getName();
      llvm::dbgs() << "\n";
    });
    return ps;
  }

  LLVM_DEBUG(llvm::dbgs() << "\nSolving: " << domainSourceOp->getName()
                          << " -> @" << defineOp.getSymName() << "\n");

  // Extract symbolic shapes
  auto symShapes = extractSymbolicShapes(defineOp, module);
  if (failed(symShapes)) {
    LLVM_DEBUG(llvm::dbgs() << "  [fail] could not extract symbolic shapes\n");
    return ps;
  }

  // Map to iteration domain
  auto symbolicIterDomain =
      extractSymbolicIterationDomain(defineOp, domainComputeOp, *symShapes);
  if (failed(symbolicIterDomain)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [fail] could not extract symbolic iteration domain\n");
    return ps;
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

  // Get source iteration domain
  auto sourceIterDomain = getSourceIterationDomain(domainSourceOp);
  if (failed(sourceIterDomain)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [fail] could not extract source iteration domain "
                  "(dynamic shapes not supported)\n");
    return ps;
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

  // Check rank match — no suffix matching, require exact rank
  unsigned instrRank = symbolicIterDomain->size();
  unsigned sourceRank = sourceIterDomain->size();
  if (sourceRank != instrRank) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [reject] rank mismatch: instruction has " << instrRank
               << " iter dims, source has " << sourceRank << "\n");
    return ps;
  }

  // Solve constraints
  auto solved = solveParams(*symbolicIterDomain, *sourceIterDomain);
  if (failed(solved)) {
    LLVM_DEBUG(llvm::dbgs() << "  [reject] parameter solving failed\n");
    return ps;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  Solved params: {";
    bool first = true;
    for (auto &[idx, val] : *solved) {
      if (!first)
        llvm::dbgs() << ", ";
      llvm::dbgs() << "p" << idx << "=" << val;
      first = false;
    }
    llvm::dbgs() << "}\n";
  });

  // Classify addr params
  ps.solvedParams = std::move(*solved);
  ps.paramKinds = classifyAddrParams(defineOp);
  ps.isValid = true;

  LLVM_DEBUG({
    llvm::dbgs() << "  Param kinds: {";
    bool first = true;
    for (auto &[idx, kind] : ps.paramKinds) {
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

  return ps;
}

FailureOr<GraphParamSolution> act::runParamSolving(SemanticsGraph &graph,
                                                   ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "\n=== Parameter Solving (Stage 2) ===\n");

  GraphParamSolution solution;
  for (SemanticsGraphNode &node : graph.nodes)
    solution.push_back(solveNode(node, module));

  return std::move(solution);
}
