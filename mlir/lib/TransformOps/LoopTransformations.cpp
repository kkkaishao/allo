/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Dialect/AlloOps.h"
#include "allo/TransformOps/AlloTransformOps.h"
#include "allo/TransformOps/Utils.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::allo;

///===----------------------------------------------------------------------===//
/// OutlineOp implementation
///===----------------------------------------------------------------------===//
/// Wraps the given operation `op` into an `scf.execute_region` operation. Uses
/// the provided rewriter for all operations to remain compatible with the
/// rewriting infra, as opposed to just splicing the op in place.
/// Supports operations with either zero or one region.
static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() > 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      scf::ExecuteRegionOp::create(b, op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = nullptr;
    if (op->getNumRegions() == 0) {
      clonedOp = b.clone(*op);
    } else {
      clonedOp = b.cloneWithoutRegions(*op);
      Region &clonedRegion = clonedOp->getRegions().front();
      assert(clonedRegion.empty() && "expected empty region");
      b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                           clonedRegion.end());
    }
    scf::YieldOp::create(b, op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(RewriterBase &rewriter, Operation *op,
                                Region &region) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, /*argValues=*/{});
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Modified from
/// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/TransformOps/SCFTransformOps.cpp
/// to support outlining of arbitrary operations with at most one region
DiagnosedSilenceableFailure
transform::OutlineOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  SmallVector<Operation *, 4> kernels;
  SmallVector<Operation *, 4> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location loc = target->getLoc();
    if (target->getNumRegions() > 1) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    auto exec = wrapInExecuteRegion(rewriter, target);
    if (!exec) {
      return emitSilenceableFailure(target)
             << "expected target operation to have at most one region";
    }
    func::CallOp call;
    auto outlined = outlineSingleBlockRegion(rewriter, loc, exec.getRegion(),
                                             getKernelName(), &call);
    if (failed(outlined)) {
      return emitSilenceableFailure(target)
             << "failed to outline the target operation";
    }

    if (symbolTableOp) {
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(*outlined);
      call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
    }
    call->setAttr(OpIdentifier,
                  rewriter.getStringAttr(getKernelName() + "::call"));
    // `scf.execute_region` is only an outlining helper container. Inline it
    // back so the final IR directly contains `allo.call`.
    replaceOpWithRegion(rewriter, exec, exec.getRegion());
    kernels.push_back(*outlined);
    calls.push_back(call);
  }
  results.set(cast<OpResult>(getKernel()), kernels);
  results.set(cast<OpResult>(getCall()), calls);
  return DiagnosedSilenceableFailure::success();
}

///==--------------------------------------------------------------------===//
/// ReorderOp implementation
///===-------------------------------------------------------------------===//

/// Checks if the given loops are in the same perfectly nested loop band.
/// Return the outermost loop if true. Otherwise returns null.
/// The input loops do not need to be contiguous, or sorted by depth
static affine::AffineForOp
inSamePerfectlyNestedLoopBand(ArrayRef<affine::AffineForOp> loops) {
  if (loops.empty())
    return {};
  if (loops.size() == 1)
    return {};
  // create a temp copy and sort by depth
  auto tmp = llvm::to_vector(loops);
  DenseMap<affine::AffineForOp, unsigned> depthMap;
  llvm::for_each(tmp, [&depthMap](auto op) {
    unsigned depth = 0;
    Operation *curr = op;
    while ((curr = curr->getParentOp()))
      depth++;
    depthMap[op] = depth;
  });
  llvm::sort(tmp,
             [&depthMap](auto a, auto b) { return depthMap[a] < depthMap[b]; });

  // no need to be contiguous
  // check perfectly nested
  for (unsigned i = 0; i < tmp.size() - 1; ++i) {
    affine::AffineForOp currLoop = tmp[i];
    affine::AffineForOp nextLoop = tmp[i + 1];
    // check if they are in the same loop nest
    if (!currLoop->isProperAncestor(nextLoop)) {
      return {};
    }
    // check if perfectly nested between currLoop and nextLoop
    Operation *ptr = currLoop;
    while (ptr != nextLoop) {
      auto loop = dyn_cast<affine::AffineForOp>(ptr);
      if (!loop) {
        return {};
      }
      Block *body = loop.getBody();
      // the first one is affine.for
      // the second one is the terminator
      if (body->getOperations().size() != 2) {
        return {};
      }
      ptr = &body->getOperations().front();
    }
  }
  // return the top-level loop
  return tmp.front();
}

static std::string getBufferAtSourceIdentifier(Value buffer) {
  if (auto blockArg = dyn_cast<BlockArgument>(buffer)) {
    Block *owner = blockArg.getOwner();
    Operation *parentOp = owner ? owner->getParentOp() : nullptr;
    if (!parentOp)
      return "";
    auto identifier = parentOp->getAttrOfType<StringAttr>(OpIdentifier);
    if (!identifier)
      return "";
    return (identifier.getValue() + ":arg" +
            std::to_string(blockArg.getArgNumber()))
        .str();
  }

  auto opResult = dyn_cast<OpResult>(buffer);
  if (!opResult)
    return "";
  Operation *defOp = opResult.getOwner();
  auto identifier = defOp->getAttrOfType<StringAttr>(OpIdentifier);
  if (!identifier)
    return "";

  if (opResult.getResultNumber() == 0 &&
      isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(defOp)) {
    return identifier.str();
  }
  return (identifier.getValue() + ":res" +
          std::to_string(opResult.getResultNumber()))
      .str();
}

DiagnosedSilenceableFailure
transform::LoopReorderOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {

  SmallVector<affine::AffineForOp> loops;
  // validate input operation handles
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      std::string msg = "expected an affine.for operation.";
      if (isa<scf::ForOp>(payload)) {
        msg += " Try raise scf.for to affine.for before reordering.";
      }
      return emitSilenceableFailure(payload) << msg;
    }
  }
  if (loops.size() < 2) {
    return emitSilenceableError()
           << "at least two loops are required for reordering";
  }

  auto outermostLoop = inSamePerfectlyNestedLoopBand(loops);
  if (!outermostLoop) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, outermostLoop);

  // Validate permutation size against the number of selected loops.
  if (getPermutation().size() != loops.size()) {
    return emitSilenceableError()
           << "the size of permutation must match the number of loops";
  }

  // Construct the permutation vector over selected loops.
  auto permutation = getPermutation();

  // Map selected loops to their original positions in the full perfect band.
  SmallVector<unsigned, 4> selectedOrgIndices;
  for (auto l : loops) {
    auto *it = llvm::find(band, l);
    if (it == band.end())
      return emitSilenceableError() << "selected loop is not in the loop band";
    unsigned idx = std::distance(band.begin(), it);
    selectedOrgIndices.push_back(idx);
  }

  // Build a full-band permutation map. Unselected loops keep identity mapping.
  SmallVector<unsigned, 4> permMap(band.size());
  std::iota(permMap.begin(), permMap.end(), 0u);

  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned targetPos = selectedOrgIndices[i];
    unsigned srcPos = selectedOrgIndices[permutation[i]];
    permMap[targetPos] = srcPos;
  }

  // perform reordering
  if (!affine::isValidLoopInterchangePermutation(band, permMap)) {
    return emitSilenceableError() << "permutation violates legality "
                                     "constraints (e.g., data dependencies)";
  }
  affine::permuteLoops(band, permMap);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::LoopReorderOp::verify() {
  // we cannot know the number of loops at verification time
  // so we only check the validity of the permutation itself
  unsigned nPerm = getPermutation().size();
  auto permutation = getPermutation();
  for (unsigned i = 0; i < nPerm; ++i) {
    if (permutation[i] < 0 || permutation[i] >= static_cast<int32_t>(nPerm)) {
      return emitOpError("permutation index out of bounds: ") << permutation[i];
    }
    for (unsigned j = i + 1; j < nPerm; ++j) {
      if (permutation[i] == permutation[j]) {
        return emitOpError("permutation contains duplicate index: ")
               << permutation[i];
      }
    }
  }
  return success();
}

///===----------------------------------------------------------------------===//
/// SplitOp implementation
///===----------------------------------------------------------------------===//

/// Checks if the given split factor is valid for the given loop.
/// A valid split factor should be positive and smaller than the loop range.
/// only checks constant bounds loops
static bool checkSplitFactor(affine::AffineForOp loop, int64_t factor) {
  if (!loop.hasConstantBounds()) {
    return true;
  }
  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  int64_t range = ub - lb;
  return factor > 0 && factor < range;
}

static FailureOr<int64_t> stripCastInt(Value value) {
  Value current = value;
  while (true) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      return failure();
    }
    if (isa<arith::IndexCastOp, arith::TruncIOp, arith::ExtUIOp,
            arith::ExtSIOp>(defOp)) {
      current = defOp->getOperand(0);
      continue;
    }
    IntegerAttr::ValueType cst;
    if (matchPattern(current, m_ConstantInt(&cst))) {
      return cst.getSExtValue();
    }
    return failure();
  }
}

static bool checkSplitFactor(scf::ForOp loop, int64_t factor) {
  auto lbOr = stripCastInt(loop.getLowerBound());
  auto ubOr = stripCastInt(loop.getUpperBound());
  if (failed(lbOr) || failed(ubOr)) {
    return true;
  }
  int64_t lb = *lbOr;
  int64_t ub = *ubOr;
  int64_t range = ub - lb;
  return factor > 0 && factor < range;
}

DiagnosedSilenceableFailure
transform::LoopSplitOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  int64_t factor = getFactorAttr().getInt();
  if (factor <= 0) {
    return emitSilenceableFailure(getOperation())
           << "split factor must be positive";
  }
  // Case 1: affine.for loop
  if (auto forOp = dyn_cast<affine::AffineForOp>(target)) {
    auto symName = forOp->getAttrOfType<StringAttr>(OpIdentifier);
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop range";
    }

    // perform split
    // single loop is always perfectly nested
    SmallVector<affine::AffineForOp, 2> splitOps;
    if (failed(affine::tilePerfectlyNested(forOp, factor, &splitOps))) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    assert(splitOps.size() == 2 && "expected exactly two loops after tiling");

    // normalize loop
    auto outer = splitOps.front();
    auto inner = splitOps.back();
    if (failed(affine::normalizeAffineFor(outer)) ||
        failed(affine::normalizeAffineFor(inner))) {
      return emitSilenceableFailure(forOp) << "failed to normalize the loop";
    }

    AffineMap innerUb = inner.getUpperBoundMap();
    if (innerUb.isConstant() && innerUb.getNumInputs() != 0) {
      // simplify the upper bound if it's a constant map with unused symbols
      auto cstUb =
          dyn_cast<AffineConstantExpr>(innerUb.getResult(0)).getValue();
      rewriter.setInsertionPoint(inner);
      inner.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
    }

    // Sink affine.apply ops that are only used in the inner loop.
    for (auto applyOp :
         llvm::make_early_inc_range(outer.getOps<affine::AffineApplyOp>())) {
      bool allUsesInInner = llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
        return inner->isProperAncestor(u.getOwner());
      });
      if (allUsesInInner) {
        applyOp->moveBefore(&inner.getBody()->front());
      }
    }
    // set sym_name
    if (symName) {
      auto symStr = symName.getValue();
      inner->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + "::inner"));
      outer->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + "::outer"));
    }
    // record results
    results.push_back(outer);
    results.push_back(inner);
    return DiagnosedSilenceableFailure::success();
  }
  // Case 2: scf.for loop
  if (auto forOp = dyn_cast<scf::ForOp>(target)) {
    auto symName = forOp->getAttrOfType<StringAttr>(OpIdentifier);
    if (!checkSplitFactor(forOp, factor)) {
      return emitSilenceableFailure(forOp)
             << "split factor is larger than or equal to the loop range";
    }
    // perform split
    rewriter.setInsertionPoint(forOp);
    Value cst =
        arith::ConstantIndexOp::create(rewriter, forOp.getLoc(), factor);
    auto loops = tilePerfectlyNested(forOp, cst);
    if (loops.size() != 1) {
      return emitSilenceableFailure(forOp) << "failed to split the loop";
    }
    // set sym_name
    if (symName) {
      auto symStr = symName.getValue();
      forOp->setAttr(OpIdentifier, rewriter.getStringAttr(symStr + "::outer"));
      loops.back()->setAttr(OpIdentifier,
                            rewriter.getStringAttr(symStr + "::inner"));
    }
    // record results
    results.push_back(forOp);
    results.push_back(loops.front());
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableFailure(target)
         << "expected target operation to be an affine.for or scf.for loop";
}

///===----------------------------------------------------------------------===///
/// LoopTileOp implementation
///===----------------------------------------------------------------------===///
static unsigned getOperationDepth(Operation *op) {
  unsigned depth = 0;
  Operation *curr = op;
  while ((curr = curr->getParentOp()))
    depth++;
  return depth;
}

namespace {
template <typename ForOp> struct LoopWithFactor {
  ForOp loop;
  uint64_t factor;
};
} // namespace

/// Sort (loop, factor) pairs by loop depth and check they form a single
/// ancestor chain with unique loops.
template <typename ForOp>
static LogicalResult
sortAndCheckLoopFactorPairs(SmallVectorImpl<LoopWithFactor<ForOp>> &pairs) {
  DenseSet<Operation *> seenLoops;
  for (auto pair : pairs) {
    if (!seenLoops.insert(pair.loop).second)
      return failure();
  }

  llvm::sort(pairs, [](const LoopWithFactor<ForOp> &a,
                       const LoopWithFactor<ForOp> &b) {
    return getOperationDepth(a.loop) < getOperationDepth(b.loop);
  });

  // Check they belong to one loop nest.
  for (unsigned i = 0; i < pairs.size() - 1; ++i) {
    if (!pairs[i].loop->isProperAncestor(pairs[i + 1].loop))
      return failure();
  }
  return success();
}

template <typename ForOp>
static bool isContiguousPerfectBand(SmallVectorImpl<ForOp> &loops) {
  if (loops.size() <= 1)
    return true;
  for (unsigned i = 0; i < loops.size() - 1; ++i) {
    Block *body = loops[i].getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != loops[i + 1].getOperation())
      return false;
  }
  return true;
}

static FailureOr<SmallVector<uint64_t, 4>> parseTileFactors(ArrayAttr attr) {
  SmallVector<uint64_t, 4> factors;
  factors.reserve(attr.size());
  for (Attribute a : attr) {
    int64_t factor = cast<IntegerAttr>(a).getInt();
    if (factor <= 0)
      return failure();
    factors.push_back(static_cast<uint64_t>(factor));
  }
  return factors;
}

template <typename ForOp>
static SmallVector<StringAttr, 4> collectLoopSymNames(ArrayRef<ForOp> loops) {
  SmallVector<StringAttr, 4> symNames;
  symNames.reserve(loops.size());
  for (ForOp loop : loops)
    symNames.push_back(loop->template getAttrOfType<StringAttr>(OpIdentifier));
  return symNames;
}

static void annotateTiledLoopSymNames(RewriterBase &rewriter,
                                      ArrayRef<StringAttr> inputSymNames,
                                      ArrayRef<Operation *> tileLoops,
                                      ArrayRef<Operation *> pointLoops) {
  assert(inputSymNames.size() == tileLoops.size() &&
         "expected one sym_name per tile loop");
  assert(inputSymNames.size() == pointLoops.size() &&
         "expected one sym_name per point loop");

  for (auto [symName, tileLoop, pointLoop] :
       llvm::zip_equal(inputSymNames, tileLoops, pointLoops)) {
    if (!symName)
      continue;
    StringRef base = symName.getValue();
    tileLoop->setAttr(OpIdentifier, rewriter.getStringAttr(base + "::tile"));
    pointLoop->setAttr(OpIdentifier, rewriter.getStringAttr(base + "::point"));
  }
}

DiagnosedSilenceableFailure
transform::LoopTileOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> affineLoops;
  SmallVector<scf::ForOp, 4> scfLoops;

  // Collect payload loops in handle iteration order.
  // This order is the semantic order for mapping input factors to loops.
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(payload)) {
      affineLoops.push_back(affineFor);
    } else if (auto scfFor = dyn_cast<scf::ForOp>(payload)) {
      scfLoops.push_back(scfFor);
    } else {
      return emitSilenceableFailure(payload)
             << "expected an affine.for or scf.for operation";
    }
  }

  // A single tile op must target one loop dialect only.
  if (!affineLoops.empty() && !scfLoops.empty()) {
    return emitSilenceableError()
           << "cannot mix affine.for and scf.for loops in the same tiling";
  }
  if (affineLoops.empty() && scfLoops.empty()) {
    return emitSilenceableError() << "expected at least one loop to tile";
  }

  // Parse and validate tile factors once for both affine/scf paths.
  auto maybeFactors = parseTileFactors(getFactors());
  if (failed(maybeFactors))
    return emitSilenceableError() << "tile factors must be positive";
  SmallVector<uint64_t, 4> factors = *maybeFactors;

  if (!affineLoops.empty()) {
    // Factors must be provided one-for-one with input loop handles.
    if (factors.size() != affineLoops.size()) {
      return emitSilenceableError()
             << "number of tile factors must match the number of loops";
    }

    // Semantic rule:
    // bind factors to loops by input-handle order, then sort pairs by depth.
    SmallVector<LoopWithFactor<affine::AffineForOp>, 4> loopFactors;
    loopFactors.reserve(affineLoops.size());
    for (auto [loop, factor] : llvm::zip_equal(affineLoops, factors))
      loopFactors.push_back({loop, factor});

    // Validate selected loops form a single nest and are not duplicated.
    // Sorting pairs preserves loop-factor associations after reordering.
    if (failed(sortAndCheckLoopFactorPairs(loopFactors))) {
      return emitSilenceableError()
             << "affine loops must be unique and in the same loop nest";
    }

    // Materialize depth-sorted loops/factors for downstream tiling APIs.
    SmallVector<affine::AffineForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    sortedLoops.reserve(loopFactors.size());
    sortedFactors.reserve(loopFactors.size());
    for (const auto &it : loopFactors) {
      sortedLoops.push_back(it.loop);
      sortedFactors.push_back(it.factor);
    }
    SmallVector<StringAttr, 4> inputSymNames =
        collectLoopSymNames<affine::AffineForOp>(sortedLoops);

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Choose perfect vs imperfect tiling based on structural contiguity.
    bool perfect = isContiguousPerfectBand<affine::AffineForOp>(sortedLoops);

    if (perfect) {
      // Perfect affine tiling creates [outer tile loops..., inner point
      // loops...].
      SmallVector<unsigned, 4> uFactors;
      uFactors.reserve(sortedFactors.size());
      for (uint64_t factor : sortedFactors) {
        if (factor > std::numeric_limits<unsigned>::max()) {
          return emitSilenceableError()
                 << "tile factor exceeds supported affine tile size range";
        }
        uFactors.push_back(static_cast<unsigned>(factor));
      }
      SmallVector<affine::AffineForOp, 8> tiledNest;
      SmallVector<affine::AffineForOp, 4> band = sortedLoops;
      if (failed(affine::tilePerfectlyNested(band, uFactors, &tiledNest)))
        return emitSilenceableFailure(sortedLoops.front())
               << "failed to tile affine perfectly nested loops";
      if (tiledNest.size() < sortedLoops.size() * 2)
        return emitSilenceableError()
               << "unexpected number of loops created by affine tiling";

      unsigned nLoops = sortedLoops.size();
      for (auto loop : tiledNest) {
        if (failed(affine::normalizeAffineFor(loop))) {
          return emitSilenceableFailure(loop)
                 << "failed to normalize tiled affine loop";
        }
      }

      // Canonicalize point-loop upper bounds
      for (auto loop : llvm::drop_begin(tiledNest, nLoops)) {
        AffineMap ubMap = loop.getUpperBoundMap();
        if (ubMap.isConstant() && ubMap.getNumInputs() != 0) {
          auto cstUb = cast<AffineConstantExpr>(ubMap.getResult(0)).getValue();
          rewriter.setInsertionPoint(loop);
          loop.setUpperBound({}, rewriter.getConstantAffineMap(cstUb));
        } else if (!ubMap.isConstant()) {
          if (ubMap.getNumResults() == 2 && ubMap.getNumInputs() == 1) {
            auto addMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                         ubMap.getResult(1));
            auto applyOp = dyn_cast_or_null<affine::AffineApplyOp>(
                loop.getUpperBoundOperands().front().getDefiningOp());
            if (!applyOp)
              continue;
            auto outerIV = applyOp.getOperand(0);
            AffineMap composed = addMap.compose(applyOp.getAffineMap());
            SmallVector<AffineExpr, 2> exprs{ubMap.getResult(0),
                                             composed.getResult(0)};
            AffineMap finalMap = AffineMap::get(
                /*dimCount=*/1, /*symbolCount=*/0, exprs,
                rewriter.getContext());
            loop.setUpperBound(outerIV, finalMap);
          }
        }
      }
      // sink affine.apply into point loops when all uses are inside.
      for (unsigned i = 0; i < nLoops; ++i) {
        auto outer = tiledNest[i];
        auto point = tiledNest[i + nLoops];
        for (auto applyOp : llvm::make_early_inc_range(
                 outer.getOps<affine::AffineApplyOp>())) {
          bool allUsesInPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return point->isProperAncestor(u.getOwner());
              });
          if (allUsesInPoint)
            applyOp->moveBefore(&point.getBody()->front());
        }
        tileLoops.push_back(outer);
        pointLoops.push_back(point);
      }
      // sink affine.apply to innermost point loops to make perfectly nested
      for (unsigned i = nLoops; i < tiledNest.size() - 1; ++i) {
        auto point = tiledNest[i];
        auto nextPoint = tiledNest[i + 1];
        for (auto applyOp : llvm::make_early_inc_range(
                 point.getOps<affine::AffineApplyOp>())) {
          bool allUsesInNextPoint =
              llvm::all_of(applyOp->getUses(), [&](OpOperand &u) {
                return nextPoint->isProperAncestor(u.getOwner());
              });
          if (allUsesInNextPoint)
            applyOp->moveBefore(&nextPoint.getBody()->front());
        }
      }
    } else {
      // Imperfect affine tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = affine::tile(sortedLoops, sortedFactors, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile affine imperfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    annotateTiledLoopSymNames(rewriter, inputSymNames, tileLoops, pointLoops);

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  if (!scfLoops.empty()) {
    // Factors must be provided one-for-one with input loop handles.
    if (factors.size() != scfLoops.size()) {
      return emitSilenceableError()
             << "number of tile factors must match the number of loops";
    }

    // Semantic rule:
    // bind factors to loops by input-handle order, then sort pairs by depth.
    SmallVector<LoopWithFactor<scf::ForOp>, 4> loopFactors;
    loopFactors.reserve(scfLoops.size());
    for (auto [loop, factor] : llvm::zip_equal(scfLoops, factors))
      loopFactors.push_back({loop, factor});

    // Validate selected loops form a single nest and are not duplicated.
    // Sorting pairs preserves loop-factor associations after reordering.
    if (failed(sortAndCheckLoopFactorPairs(loopFactors))) {
      return emitSilenceableError()
             << "scf loops must be unique and in the same loop nest";
    }

    // Materialize depth-sorted loops/factors for downstream tiling APIs.
    SmallVector<scf::ForOp, 4> sortedLoops;
    SmallVector<uint64_t, 4> sortedFactors;
    sortedLoops.reserve(loopFactors.size());
    sortedFactors.reserve(loopFactors.size());
    for (const auto &it : loopFactors) {
      sortedLoops.push_back(it.loop);
      sortedFactors.push_back(it.factor);
    }
    SmallVector<StringAttr, 4> inputSymNames =
        collectLoopSymNames<scf::ForOp>(sortedLoops);

    SmallVector<Operation *, 4> tileLoops;
    SmallVector<Operation *, 4> pointLoops;

    // Build runtime tile-size SSA values from sorted factors.
    SmallVector<Value, 4> sizeVals;
    sizeVals.reserve(sortedFactors.size());
    rewriter.setInsertionPoint(sortedLoops.front());
    for (uint64_t factor : sortedFactors) {
      if (factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return emitSilenceableError()
               << "tile factor exceeds supported scf tile size range";
      sizeVals.push_back(
          arith::ConstantIndexOp::create(rewriter, sortedLoops.front().getLoc(),
                                         static_cast<int64_t>(factor)));
    }

    // Choose perfect vs imperfect tiling based on structural contiguity.
    bool perfect = isContiguousPerfectBand<scf::ForOp>(sortedLoops);
    if (perfect) {
      // Perfect scf tiling returns only point loops; outer loops are updated
      // in-place and remain represented by sortedLoops.
      SmallVector<scf::ForOp, 8> point =
          tilePerfectlyNested(sortedLoops.front(), sizeVals);
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf perfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    } else {
      // Imperfect scf tiling strip-mines selected loops and sinks them
      // under the chosen target while preserving sorted loop order.
      auto point = ::mlir::tile(sortedLoops, sizeVals, sortedLoops.back());
      if (point.size() != sortedLoops.size()) {
        return emitSilenceableError()
               << "failed to tile scf imperfectly nested loops";
      }
      for (auto loop : sortedLoops)
        tileLoops.push_back(loop);
      for (auto loop : point)
        pointLoops.push_back(loop);
    }

    annotateTiledLoopSymNames(rewriter, inputSymNames, tileLoops, pointLoops);

    // Output handles are always reported in depth order.
    results.set(cast<OpResult>(getTileLoops()), tileLoops);
    results.set(cast<OpResult>(getPointLoops()), pointLoops);
    return DiagnosedSilenceableFailure::success();
  }

  return emitSilenceableError() << "failed to tile loops";
}

///===----------------------------------------------------------------------===//
/// LoopFlattenOp implementation
///===----------------------------------------------------------------------===///

// modified from lib/Transforms/Utils/LoopUtils.cpp
static void coalesceLoops(MutableArrayRef<affine::AffineForOp> loops,
                          int64_t flattenedTripCount,
                          transform::TransformRewriter &rewriter) {
  // RAII helper to restore the insertion point.
  OpBuilder::InsertionGuard guard(rewriter);

  affine::AffineForOp innermost = loops.back();
  affine::AffineForOp outermost = loops.front();
  Location loc = outermost.getLoc();

  // 1. Store the upper bound of the outermost loop in a variable.
  SmallVector<int64_t, 4> ubs;
  for (auto loop : loops) {
    auto cstUb = loop.getConstantUpperBound();
    ubs.push_back(cstUb);
  }

  // 2. The flattened trip count is validated by the caller.
  outermost.setConstantUpperBound(flattenedTripCount);

  // 3. Remap induction variables. For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  rewriter.setInsertionPointToStart(outermost.getBody());
  Value previous = outermost.getInductionVar();
  SmallVector<Operation *> opToSink;
  for (unsigned idx = loops.size(); idx > 0; --idx) {
    int64_t currUb = ubs[idx - 1];
    if (idx != loops.size()) {
      auto quotientMap =
          AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(0).floorDiv(currUb));
      previous =
          affine::AffineApplyOp::create(rewriter, loc, quotientMap, previous);
      opToSink.push_back(previous.getDefiningOp());
    }
    // Modified value of the induction variables of the nested loops after
    // coalescing.
    Value inductionVariable;
    if (idx == 1) {
      inductionVariable = previous;
    } else {
      auto modMap = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                   rewriter.getAffineDimExpr(0) % currUb);
      inductionVariable =
          affine::AffineApplyOp::create(rewriter, loc, modMap, previous);
      opToSink.push_back(inductionVariable.getDefiningOp());
    }
    replaceAllUsesInRegionWith(loops[idx - 1].getInductionVar(),
                               inductionVariable, loops.back().getRegion());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  affine::AffineForOp secondOutermostLoop = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(secondOutermostLoop.getOperation()),
      innermost.getBody()->getOperations());
  for (auto [iter, init] :
       llvm::zip_equal(secondOutermostLoop.getRegionIterArgs(),
                       secondOutermostLoop.getInits())) {
    iter.replaceAllUsesWith(init);
    iter.dropAllUses();
  }
  secondOutermostLoop.erase();

  // 5. Sink affine.apply operations.
  std::reverse(opToSink.begin(), opToSink.end());
  outermost.walk([&](affine::AffineForOp nestedLoop) {
    if (nestedLoop == outermost)
      return;
    bool canSinkAll = true;
    for (Operation *op : opToSink) {
      for (Operation *user : op->getUsers()) {
        if (!nestedLoop->isAncestor(user)) {
          canSinkAll = false;
          break;
        }
      }
      if (!canSinkAll)
        break;
    }
    if (canSinkAll) {
      Block *body = nestedLoop.getBody();
      for (Operation *op : opToSink) {
        op->moveBefore(&body->front());
      }
    }
  });
}

DiagnosedSilenceableFailure
transform::LoopFlattenOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<affine::AffineForOp, 4> loops;
  // validate input operation handles
  for (Operation *payload : state.getPayloadOps(getLoops())) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(payload)) {
      loops.push_back(forOp);
    } else {
      if (isa<scf::ForOp>(payload)) {
        return emitSilenceableFailure(payload)
               << "Try raise scf.for to affine.for before flattening.";
      }
      return emitSilenceableFailure(payload)
             << "expected an affine.for operation";
    }
  }
  if (loops.size() < 2) {
    results.set(cast<OpResult>(getResult()), loops);
    return DiagnosedSilenceableFailure::success();
  }

  auto namePrefix = loops.front()->getAttrOfType<StringAttr>(OpIdentifier);

  // Flatten supports unordered loop handles; normalize to depth order first.
  llvm::sort(loops, [](auto a, auto b) {
    return getOperationDepth(a) < getOperationDepth(b);
  });

  auto selectedOutermost = inSamePerfectlyNestedLoopBand(loops);
  if (!selectedOutermost) {
    return emitSilenceableError()
           << "loops must be in the same perfectly nested loop band";
  }

  // Flatten a contiguous band from selected outermost to selected innermost.
  SmallVector<affine::AffineForOp, 4> perfectBand;
  affine::getPerfectlyNestedLoops(perfectBand, selectedOutermost);
  auto *endIt = llvm::find(perfectBand, loops.back());
  if (endIt == perfectBand.end()) {
    return emitSilenceableError()
           << "failed to find selected innermost loop in perfect loop band";
  }
  SmallVector<affine::AffineForOp, 4> flattenBand(perfectBand.begin(),
                                                  std::next(endIt));

  // Current coalescing logic assumes normalized affine loops with constant
  // trip counts.
  int64_t flattenedTripCount = 1;
  for (auto loop : flattenBand) {
    if (loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
        loop.getConstantLowerBound() != 0 || !loop.hasConstantUpperBound()) {
      return emitSilenceableError()
             << "flatten requires normalized affine.for loops with step=1, "
                "constant lower bound=0 and constant upper bound";
    }
    int64_t ub = loop.getConstantUpperBound();
    if (ub <= 0) {
      return emitSilenceableError()
             << "flatten requires positive constant upper bounds";
    }
    if (flattenedTripCount > std::numeric_limits<int64_t>::max() / ub) {
      return emitSilenceableError()
             << "flattened loop trip count overflows int64";
    }
    flattenedTripCount *= ub;
  }

  // perform flattening
  ::coalesceLoops(flattenBand, flattenedTripCount, rewriter);

  // set sym_name
  if (namePrefix) {
    flattenBand.front()->setAttr(
        OpIdentifier, rewriter.getStringAttr(namePrefix.getValue() + "::flat"));
  }
  // record results
  results.set(cast<OpResult>(getResult()), {flattenBand.front()});
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===//
/// ComputeAt implementation
///===----------------------------------------------------------------------===///

static std::optional<std::string>
tryAffineLoopFusion(affine::AffineForOp producer, affine::AffineForOp consumer,
                    unsigned targetDepth) {
  using affine::FusionResult;
  affine::ComputationSliceState sliceState;
  affine::FusionStrategy strategy(affine::FusionStrategy::ProducerConsumer);
  FusionResult test = affine::canFuseLoops(producer, consumer, targetDepth,
                                           &sliceState, strategy);
  if (test.value == FusionResult::Success) {
    affine::fuseLoops(producer, consumer, sliceState);
    producer.erase();
    return std::nullopt;
  }
  std::string reason;
  if (test.value == FusionResult::FailPrecondition) {
    reason = "failed precondition for fusion (e.g. same block)";
  } else if (test.value == FusionResult::FailBlockDependence) {
    reason = "fusion would violate another dependence in block";
  } else if (test.value == FusionResult::FailFusionDependence) {
    reason = "fusion would reverse dependences between loops";
  } else if (test.value == FusionResult::FailComputationSlice) {
    reason = "unable to compute src loop computation slice";
  } else if (test.value == FusionResult::FailIncorrectSlice) {
    reason = "slice is computed, but it is incorrect";
  }
  return reason;
}

namespace {
enum class DependenceType : uint8_t {
  NONE = 0,
  RAW = 1 << 1u,
  WAR = 1 << 2u,
  WAW = 1 << 3u,
};

DependenceType operator|(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) |
                                     static_cast<uint8_t>(b));
}

DependenceType operator&(DependenceType a, DependenceType b) {
  return static_cast<DependenceType>(static_cast<uint8_t>(a) &
                                     static_cast<uint8_t>(b));
}
} // namespace

// check dependencies between two affine.for loop nests up to a certain depth
// assume forOpA is source, forOpB is sink
// return a bitmask of DependenceType
static FailureOr<DependenceType> checkDependencies(affine::AffineForOp forOpA,
                                                   affine::AffineForOp forOpB,
                                                   unsigned depth) {
  SmallVector<affine::MemRefAccess, 4> accA;
  SmallVector<affine::MemRefAccess, 4> accB;
  bool hasUnsupportedAccess = false;
  // Collect only affine accesses; non-affine memref accesses are conservatively
  // treated as unsupported so we can fail instead of mis-transforming.
  forOpA.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accA.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  forOpB.walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      accB.emplace_back(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      hasUnsupportedAccess = true;
    }
  });
  if (hasUnsupportedAccess)
    return failure();

  // Build a coarse dependence summary (RAW/WAR/WAW) between source and sink
  // loop nests, using affine dependence checks per matching memref access pair.
  DependenceType ret = DependenceType::NONE;
  for (auto &a : accA) {
    for (auto &b : accB) {
      if (a.memref != b.memref) {
        continue; // different memrefs
      }
      if (!a.isStore() && !b.isStore()) {
        continue; // we don't care about rar
      }
      SmallVector<affine::DependenceComponent, 2> deps;
      auto depResult =
          affine::checkMemrefAccessDependence(a, b, depth, nullptr, &deps);
      if (depResult.value == affine::DependenceResult::Failure) {
        return failure();
      }
      if (depResult.value == affine::DependenceResult::HasDependence) {
        if (a.isStore() && !b.isStore()) {
          ret = ret | DependenceType::RAW;
        } else if (!a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAR;
        } else if (a.isStore() && b.isStore()) {
          ret = ret | DependenceType::WAW;
        }
      }
    }
  }
  return ret;
}

namespace {
struct ConstantBoundsPrefix {
  SmallVector<int64_t, 4> lowerBounds;
  SmallVector<int64_t, 4> upperBounds;
};
} // namespace

static SmallVector<affine::AffineForOp, 4>
collectAffineLoopChain(Operation *op) {
  SmallVector<affine::AffineForOp, 4> chain;
  for (Operation *curr = op; curr; curr = curr->getParentOp()) {
    if (auto loop = dyn_cast<affine::AffineForOp>(curr))
      chain.push_back(loop);
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

static FailureOr<ConstantBoundsPrefix>
getConstantBoundsPrefix(ArrayRef<affine::AffineForOp> chain, unsigned depth) {
  if (chain.size() < depth)
    return failure();
  ConstantBoundsPrefix bounds;
  bounds.lowerBounds.reserve(depth);
  bounds.upperBounds.reserve(depth);
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp loop = chain[i];
    if (!loop.hasConstantBounds())
      return failure();
    bounds.lowerBounds.push_back(loop.getConstantLowerBound());
    bounds.upperBounds.push_back(loop.getConstantUpperBound());
  }
  return bounds;
}

static bool hasSinglePathLoopPrefix(ArrayRef<affine::AffineForOp> chain,
                                    unsigned prefixDepth) {
  if (prefixDepth > chain.size())
    return false;
  for (unsigned i = 0; i + 1 < prefixDepth; ++i) {
    affine::AffineForOp current = chain[i];
    affine::AffineForOp next = chain[i + 1];
    Block *body = current.getBody();
    if (body->getOperations().size() != 2)
      return false;
    if (&body->front() != next.getOperation())
      return false;
  }
  return true;
}

static bool hasIdenticalBounds(const ConstantBoundsPrefix &a,
                               const ConstantBoundsPrefix &b) {
  return a.lowerBounds == b.lowerBounds && a.upperBounds == b.upperBounds;
}

static bool isSubsetBounds(const ConstantBoundsPrefix &producerBounds,
                           const ConstantBoundsPrefix &consumerBounds) {
  for (auto [prodLb, prodUb, consLb, consUb] : llvm::zip_equal(
           producerBounds.lowerBounds, producerBounds.upperBounds,
           consumerBounds.lowerBounds, consumerBounds.upperBounds)) {
    if (prodLb < consLb || prodUb > consUb)
      return false;
  }
  return true;
}

static IntegerSet buildSubsetGuardSet(OpBuilder &builder,
                                      const ConstantBoundsPrefix &bounds) {
  unsigned depth = bounds.lowerBounds.size();
  SmallVector<AffineExpr, 8> constraints;
  constraints.reserve(depth * 2);
  for (unsigned i = 0; i < depth; ++i) {
    AffineExpr dim = builder.getAffineDimExpr(i);
    constraints.push_back(dim - bounds.lowerBounds[i]);
    constraints.push_back((bounds.upperBounds[i] - 1) - dim);
  }
  SmallVector<bool, 8> isEq(constraints.size(), false);
  return IntegerSet::get(depth, /*symbolCount=*/0, constraints, isEq);
}

static void remapProducerIVPrefix(ArrayRef<affine::AffineForOp> producerChain,
                                  ArrayRef<affine::AffineForOp> consumerChain,
                                  unsigned depth, Region &region) {
  for (unsigned i = 0; i < depth; ++i) {
    affine::AffineForOp producer = producerChain[i];
    affine::AffineForOp consumer = consumerChain[i];
    replaceAllUsesInRegionWith(producer.getInductionVar(),
                               consumer.getInductionVar(), region);
  }
}

static bool hasBlockingSideEffectsBetween(Operation *before, Operation *after) {
  assert(before->getBlock() == after->getBlock() &&
         "expected operations in the same block");
  assert(before->isBeforeInBlock(after) &&
         "expected `before` to appear before `after`");

  for (Operation *curr = before->getNextNode(); curr && curr != after;
       curr = curr->getNextNode()) {
    if (!isMemoryEffectFree(curr))
      return true;
  }
  return false;
}

namespace {
struct ComputeAtAnalysis {
  Operation *producerOp = nullptr;
  affine::AffineForOp consumerLoop = nullptr;
  SmallVector<affine::AffineForOp, 4> producerChain;
  SmallVector<affine::AffineForOp, 4> consumerChain;
  affine::AffineForOp producerRoot = nullptr;
  affine::AffineForOp consumerRoot = nullptr;
  unsigned producerDepth = 0;
  unsigned consumerDepth = 0;
  SmallVector<Value, 4> consumerPrefixIVs;
};
} // namespace

static std::optional<std::string>
analyzeComputeAt(Operation *producerOp, affine::AffineForOp consumerLoop,
                 ComputeAtAnalysis &analysis) {
  // Normalize producer/consumer into loop-chain metadata once so execution
  // paths can focus on transformation mechanics.
  analysis.producerOp = producerOp;
  analysis.consumerLoop = consumerLoop;
  analysis.producerChain = collectAffineLoopChain(producerOp);
  if (analysis.producerChain.empty()) {
    return std::string("producer must be inside an affine.for loop nest");
  }

  analysis.consumerChain = collectAffineLoopChain(consumerLoop);
  if (analysis.consumerChain.empty()) {
    return std::string("expected consumer_loop to resolve to an affine.for");
  }

  analysis.producerDepth = analysis.producerChain.size();
  analysis.consumerDepth = analysis.consumerChain.size();
  analysis.producerRoot = analysis.producerChain.front();
  analysis.consumerRoot = analysis.consumerChain.front();

  analysis.consumerPrefixIVs.clear();
  analysis.consumerPrefixIVs.reserve(analysis.consumerDepth);
  for (affine::AffineForOp loop : analysis.consumerChain)
    analysis.consumerPrefixIVs.push_back(loop.getInductionVar());

  if (analysis.producerRoot == analysis.consumerRoot) {
    return std::string(
        "producer and consumer must belong to different root loop nests");
  }
  if (analysis.producerRoot->getBlock() != analysis.consumerRoot->getBlock()) {
    return std::string(
        "producer and consumer loop nests must be in the same block");
  }
  if (analysis.producerDepth < analysis.consumerDepth) {
    return std::string(
        "producer loop nest depth is shallower than consumer depth");
  }
  return std::nullopt;
}

static std::optional<std::string>
applyNoDependenceMove(transform::TransformRewriter &rewriter,
                      ComputeAtAnalysis &analysis) {
  unsigned consumerDepth = analysis.consumerDepth;
  unsigned producerDepth = analysis.producerDepth;

  // We only rewrite a single-path producer prefix; imperfect control flow in
  // this prefix would make region move/remap semantics ambiguous.
  unsigned prefixDepthToValidate =
      producerDepth == consumerDepth ? producerDepth : consumerDepth + 1;
  if (!hasSinglePathLoopPrefix(analysis.producerChain, prefixDepthToValidate)) {
    return std::string(
        "producer loop prefix to be rewritten must be perfectly nested");
  }

  // No-dependence move must preserve top-level order and cannot jump over
  // side-effecting operations between producer root and consumer root.
  if (!analysis.producerRoot->isBeforeInBlock(analysis.consumerRoot)) {
    return std::string(
        "producer root loop must appear before consumer root loop");
  }
  if (hasBlockingSideEffectsBetween(analysis.producerRoot,
                                    analysis.consumerRoot)) {
    return std::string("cannot move producer across side-effecting operations "
                       "between producer and consumer roots");
  }

  // No-dependence path currently supports constant-bound prefix reasoning only;
  // subset bounds are handled by generating an affine.if guard.
  FailureOr<ConstantBoundsPrefix> producerBounds =
      getConstantBoundsPrefix(analysis.producerChain, consumerDepth);
  FailureOr<ConstantBoundsPrefix> consumerBounds =
      getConstantBoundsPrefix(analysis.consumerChain, consumerDepth);
  if (failed(producerBounds) || failed(consumerBounds)) {
    return std::string("compute_at currently supports only constant-bounds "
                       "loops for no-dependence move");
  }

  bool identicalBounds = hasIdenticalBounds(*producerBounds, *consumerBounds);
  if (!identicalBounds && !isSubsetBounds(*producerBounds, *consumerBounds)) {
    return std::string("producer loop bounds must be identical to, or a "
                       "subset of, consumer bounds");
  }

  // Move producer body/subtree under consumer loop. If bounds are subset-only,
  // first materialize an affine.if so execution stays within producer domain.
  Block *destination = analysis.consumerLoop.getBody();
  Region *ivRemapRegion = &analysis.consumerLoop.getRegion();
  if (!identicalBounds) {
    rewriter.setInsertionPointToStart(analysis.consumerLoop.getBody());
    auto ifOp = affine::AffineIfOp::create(
        rewriter, analysis.consumerLoop.getLoc(),
        buildSubsetGuardSet(rewriter, *producerBounds),
        analysis.consumerPrefixIVs,
        /*withElseRegion=*/false);
    destination = ifOp.getThenBlock();
    ivRemapRegion = &ifOp.getThenRegion();
  }

  if (producerDepth == consumerDepth) {
    Block *producerInnermostBody = analysis.producerChain.back().getBody();
    Value consumerInnermostIV = analysis.consumerChain.back().getInductionVar();
    rewriter.eraseOp(producerInnermostBody->getTerminator());
    rewriter.inlineBlockBefore(producerInnermostBody, destination,
                               destination->begin(),
                               ValueRange{consumerInnermostIV});
  } else {
    Operation *producerSubtree =
        analysis.producerChain[consumerDepth].getOperation();
    rewriter.moveOpBefore(producerSubtree, destination, destination->begin());
  }

  unsigned remapDepth = consumerDepth;
  if (producerDepth == consumerDepth)
    remapDepth -= 1;
  // Rewrite producer IV uses to consumer IVs in the moved region prefix.
  remapProducerIVPrefix(analysis.producerChain, analysis.consumerChain,
                        remapDepth, *ivRemapRegion);
  analysis.producerRoot.erase();
  return std::nullopt;
}

static bool mayWriteAliasingMemref(Operation *op, Value memref,
                                   AliasAnalysis &aliasAnalysis) {
  if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op))
    return !aliasAnalysis.alias(writeOp.getMemRef(), memref).isNo();

  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    iface.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;
      Value effectValue = effect.getValue();
      if (!effectValue)
        return true;
      if (!aliasAnalysis.alias(effectValue, memref).isNo())
        return true;
    }
    return false;
  }

  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          if (mayWriteAliasingMemref(&nested, memref, aliasAnalysis))
            return true;
        }
      }
    }
    return false;
  }

  // Unknown ops are conservatively assumed to possibly write.
  return true;
}

static void runComputeAtPostCleanup(affine::AffineForOp consumerLoop) {
  // Run local store-to-load forwarding in the transformed loop nest only.
  // This avoids full-function affineScalarReplace on large kernels.
  Operation *scopeOp = nullptr;
  if (auto kernel = consumerLoop->getParentOfType<func::FuncOp>())
    scopeOp = kernel.getOperation();
  else
    scopeOp = consumerLoop->getParentOp();
  AliasAnalysis aliasAnalysis(scopeOp);

  SmallVector<affine::AffineReadOpInterface, 16> loads;
  consumerLoop.walk(
      [&](affine::AffineReadOpInterface loadOp) { loads.push_back(loadOp); });

  SmallVector<Operation *, 16> loadsToErase;
  for (affine::AffineReadOpInterface loadOp : loads) {
    Operation *load = loadOp.getOperation();
    if (!load || !load->getBlock())
      continue;

    Value loadMemref = loadOp.getMemRef();
    affine::MemRefAccess loadAccess(load);
    Operation *forwardingStore = nullptr;

    for (Operation *curr = load->getPrevNode(); curr;
         curr = curr->getPrevNode()) {
      if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(curr)) {
        // Non-aliasing stores cannot affect this load.
        if (aliasAnalysis.alias(storeOp.getMemRef(), loadMemref).isNo())
          continue;

        affine::MemRefAccess storeAccess(curr);
        if (storeAccess == loadAccess)
          forwardingStore = curr;
        // Any aliasing store blocks the search in this block.
        break;
      }

      if (mayWriteAliasingMemref(curr, loadMemref, aliasAnalysis))
        break;
    }

    if (!forwardingStore)
      continue;
    Value storeValue =
        cast<affine::AffineWriteOpInterface>(forwardingStore).getValueToStore();
    if (storeValue.getType() != loadOp.getValue().getType())
      continue;
    loadOp.getValue().replaceAllUsesWith(storeValue);
    loadsToErase.push_back(load);
  }

  for (Operation *load : loadsToErase)
    load->erase();
}

DiagnosedSilenceableFailure
transform::ComputeAtOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &states) {
  (void)results;

  auto consumerLoops = states.getPayloadOps(getConsumerLoop());
  auto producers = states.getPayloadOps(getProducer());
  if (!llvm::hasSingleElement(consumerLoops) ||
      !llvm::hasSingleElement(producers)) {
    return emitSilenceableError()
           << "expected exactly one producer and one consumer loop";
  }

  Operation *producerOp = *producers.begin();
  auto consumerLoop = dyn_cast<affine::AffineForOp>(*consumerLoops.begin());
  if (!consumerLoop) {
    return emitSilenceableError()
           << "expected consumer_loop to resolve to an affine.for";
  }

  ComputeAtAnalysis analysis;
  if (auto reason = analyzeComputeAt(producerOp, consumerLoop, analysis)) {
    return emitSilenceableError() << *reason;
  }

  // Classify producer->consumer dependence first; this decides whether to use
  // affine fusion or conservative manual move.
  auto depTypeOr = checkDependencies(
      analysis.producerRoot, analysis.consumerLoop, analysis.consumerDepth);
  if (failed(depTypeOr)) {
    return emitSilenceableError()
           << "dependence analysis failed; refusing compute_at";
  }
  DependenceType depType = *depTypeOr;

  if ((depType & DependenceType::RAW) != DependenceType::NONE) {
    // RAW dependence requires true producer-consumer fusion to preserve
    // semantics while changing loop placement.
    auto reason = tryAffineLoopFusion(
        analysis.producerRoot, analysis.consumerLoop, analysis.consumerDepth);
    if (reason.has_value()) {
      return emitSilenceableError()
             << "cannot fuse producer and consumer loop nests: "
             << reason.value();
    }
  } else if (depType == DependenceType::NONE) {
    // No dependence: perform structurally-checked move/inline + IV remap.
    if (auto reason = applyNoDependenceMove(rewriter, analysis)) {
      return emitSilenceableError() << *reason;
    }
  } else {
    return emitSilenceableError()
           << "compute_at does not support WAR/WAW-only dependences";
  }

  runComputeAtPostCleanup(analysis.consumerLoop);

  // results.set(cast<OpResult>(getResult()), {targetForOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::ComputeAtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerMutable(), effects);
  // consumer_loop is read-only and must remain reusable.
  onlyReadsHandle(getConsumerLoopMutable(), effects);
  modifiesPayload(effects);
}

///===----------------------------------------------------------------------===//
/// Shared buffer-access analysis helpers
///===----------------------------------------------------------------------===//
namespace {
struct ComposedBufferAccess {
  Operation *op = nullptr;
  AffineMap map;
  SmallVector<Value, 4> operands;
};
} // namespace

static ComposedBufferAccess composeBufferAccess(Operation *accessOp) {
  affine::MemRefAccess access(accessOp);
  affine::AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  accessValueMap.composeSimplifyAndCanonicalize();
  return {accessOp, accessValueMap.getAffineMap(),
          llvm::to_vector(accessValueMap.getOperands())};
}

static void
collectFootprintOperands(ArrayRef<ComposedBufferAccess> accesses,
                         ArrayRef<affine::AffineForOp> innerLoops,
                         ArrayRef<Value> excludedValues,
                         DenseMap<Value, unsigned> &prefixOperandPos,
                         SmallVectorImpl<Value> &prefixOperands) {
  DenseSet<Value> excluded;
  for (Value value : excludedValues)
    excluded.insert(stripCast(value));
  for (affine::AffineForOp loop : innerLoops)
    excluded.insert(stripCast(loop.getInductionVar()));

  for (const ComposedBufferAccess &access : accesses) {
    for (AffineExpr resultExpr : access.map.getResults()) {
      for (Value operand : access.operands) {
        operand = stripCast(operand);
        if (excluded.contains(operand))
          continue;
        if (!affineExprUsesValue(resultExpr, access.operands,
                                 access.map.getNumDims(), operand)) {
          continue;
        }
        if (prefixOperandPos.contains(operand))
          continue;
        prefixOperandPos[operand] = prefixOperands.size();
        prefixOperands.push_back(operand);
      }
    }
  }
}

static void populateExprReplacements(
    AffineMap accessMap, ValueRange accessOperands,
    DenseMap<Value, unsigned> &prefixOperandPos,
    ArrayRef<std::pair<Value, AffineExpr>> explicitReplacements,
    unsigned prefixDimOffset, SmallVectorImpl<AffineExpr> &dimReplacements,
    SmallVectorImpl<AffineExpr> &symReplacements) {
  dimReplacements.clear();
  symReplacements.clear();
  dimReplacements.reserve(accessMap.getNumDims());
  symReplacements.reserve(accessMap.getNumSymbols());

  auto getReplacement = [&](Value operand) {
    operand = stripCast(operand);
    for (const auto &[explicitOperand, replacement] : explicitReplacements) {
      if (operand == stripCast(explicitOperand))
        return replacement;
    }

    auto it = prefixOperandPos.find(operand);
    if (it == prefixOperandPos.end())
      return getAffineConstantExpr(0, accessMap.getContext());
    return getAffineDimExpr(prefixDimOffset + it->second,
                            accessMap.getContext());
  };

  for (unsigned i = 0; i < accessMap.getNumDims(); ++i)
    dimReplacements.push_back(getReplacement(accessOperands[i]));
  for (unsigned i = 0; i < accessMap.getNumSymbols(); ++i)
    symReplacements.push_back(
        getReplacement(accessOperands[accessMap.getNumDims() + i]));
}

static void populateExprReplacements(
    AffineMap accessMap, ArrayRef<Value> accessOperands,
    DenseMap<Value, unsigned> &prefixOperandPos,
    SmallVectorImpl<Value> & /*prefixOperands*/, Value targetLoopIV,
    std::optional<int64_t> targetLoopConstant,
    std::optional<unsigned> targetLoopDimPos, unsigned prefixDimOffset,
    SmallVectorImpl<AffineExpr> &dimReplacements,
    SmallVectorImpl<AffineExpr> &symReplacements) {
  SmallVector<std::pair<Value, AffineExpr>, 1> explicitReplacements;
  if (targetLoopIV) {
    assert(targetLoopConstant.has_value() != targetLoopDimPos.has_value() &&
           "expected exactly one loop replacement kind");
    AffineExpr replacement =
        targetLoopDimPos
            ? getAffineDimExpr(*targetLoopDimPos, accessMap.getContext())
            : getAffineConstantExpr(*targetLoopConstant,
                                    accessMap.getContext());
    explicitReplacements.emplace_back(targetLoopIV, replacement);
  }

  populateExprReplacements(accessMap, accessOperands, prefixOperandPos,
                           explicitReplacements, prefixDimOffset,
                           dimReplacements, symReplacements);
}

static FailureOr<int64_t> getLinearAffineDimCoefficient(AffineExpr expr,
                                                        unsigned numDims,
                                                        unsigned dimPos) {
  FlatLinearConstraints localVarCst(numDims, /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  if (failed(getFlattenedAffineExpr(expr, numDims, /*numSymbols=*/0,
                                    &flattenedExpr, &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 || flattenedExpr.size() != numDims + 1)
    return failure();
  return flattenedExpr[dimPos];
}

static FailureOr<int64_t> getConstantExprDelta(AffineExpr lhs, AffineExpr rhs,
                                               unsigned numDims) {
  FlatLinearConstraints localVarCst(numDims, /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  AffineExpr diff = simplifyAffineExpr(lhs - rhs, numDims, /*numSymbols=*/0);
  if (failed(getFlattenedAffineExpr(diff, numDims, /*numSymbols=*/0,
                                    &flattenedExpr, &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 || flattenedExpr.size() != numDims + 1)
    return failure();
  for (unsigned i = 0, e = flattenedExpr.size() - 1; i < e; ++i)
    if (flattenedExpr[i] != 0)
      return failure();
  return flattenedExpr.back();
}

///===----------------------------------------------------------------------===//
/// ReuseAt implementation
///===----------------------------------------------------------------------===///
namespace {
struct LoopRoleInfo {
  DenseSet<Value> spatialIVs;
  DenseSet<Value> reductionIVs;
  DenseMap<Value, int64_t> reductionUpperBounds;
};

struct ReuseDimPlan {
  AffineMap anchorMap;
  int64_t extent = 1;
  int64_t axisCoeff = 0;
  int64_t innerMinOffset = 0;
  int64_t innerMaxOffset = 0;
  bool isSliding = false;
};

struct ReuseStatePlan {
  SmallVector<ReuseDimPlan, 4> dims;
  SmallVector<Value, 4> prefixOperands;
  SmallVector<unsigned, 4> keptDims;
  SmallVector<int64_t, 4> shape;
  SmallVector<int, 4> resultToReusePos;
  unsigned slidingDim = 0;
  int64_t slidingDelta = 1;
  int64_t slidingStepAbs = 1;
};

enum class ReuseBufferStrategy {
  PhysicalShift,
  Ring,
};

struct ReuseExecutionPlan {
  explicit ReuseExecutionPlan(ReuseStatePlan statePlan, bool enableRing)
      : statePlan(std::move(statePlan)),
        slidingDelta(this->statePlan.slidingDelta),
        slidingStepAbs(this->statePlan.slidingStepAbs) {
    int slidingReusePos =
        this->statePlan.resultToReusePos[this->statePlan.slidingDim];
    int64_t slidingExtent =
        this->statePlan.dims[this->statePlan.slidingDim].extent;
    if (enableRing && slidingReusePos >= 0 && slidingExtent > 1 &&
        slidingStepAbs < slidingExtent) {
      strategy = ReuseBufferStrategy::Ring;
      ringIncrement =
          slidingDelta > 0 ? slidingStepAbs : slidingExtent - slidingStepAbs;
    }
  }

  ReuseStatePlan statePlan;
  ReuseBufferStrategy strategy = ReuseBufferStrategy::PhysicalShift;
  int64_t slidingDelta = 1;
  int64_t slidingStepAbs = 1;
  int64_t steadyStateStart = 1;
  int64_t ringIncrement = 0;
};

struct ReuseDimFootprint {
  AffineExpr lowerExpr;
  int64_t extent = 1;
  int64_t axisCoeff = 0;
  int64_t innerMinOffset = 0;
  int64_t innerMaxOffset = 0;
};

struct NoRingReuseWrapper {
  affine::AffineLoadOp thenLoad;
  affine::AffineStoreOp thenStore;
  affine::AffineLoadOp elseLoad;
};
} // namespace

static std::optional<NoRingReuseWrapper>
matchNoRingReuseWrapper(affine::AffineIfOp ifOp, Value target) {
  if (!ifOp.getElseBlock() || ifOp.getNumResults() != 1)
    return std::nullopt;

  auto *thenTerminator = ifOp.getThenBlock()->getTerminator();
  auto *elseTerminator = ifOp.getElseBlock()->getTerminator();
  auto thenYield = dyn_cast<affine::AffineYieldOp>(thenTerminator);
  auto elseYield = dyn_cast<affine::AffineYieldOp>(elseTerminator);
  if (!thenYield || !elseYield || thenYield.getNumOperands() != 1 ||
      elseYield.getNumOperands() != 1) {
    return std::nullopt;
  }

  auto thenOps = ifOp.getThenBlock()->without_terminator();
  auto elseOps = ifOp.getElseBlock()->without_terminator();
  auto thenIt = thenOps.begin();
  auto elseIt = elseOps.begin();
  if (thenIt == thenOps.end() || elseIt == elseOps.end())
    return std::nullopt;

  auto thenLoad = dyn_cast<affine::AffineLoadOp>(&*thenIt++);
  auto thenStore =
      thenIt == thenOps.end() ? affine::AffineStoreOp() :
                                dyn_cast<affine::AffineStoreOp>(&*thenIt++);
  auto elseLoad = dyn_cast<affine::AffineLoadOp>(&*elseIt++);
  if (!thenLoad || !thenStore || !elseLoad || thenIt != thenOps.end() ||
      elseIt != elseOps.end()) {
    return std::nullopt;
  }

  if (thenStore.getMemRef() != target || elseLoad.getMemRef() != target)
    return std::nullopt;
  if (thenStore.getValueToStore() != thenLoad.getResult())
    return std::nullopt;
  if (thenYield.getOperand(0) != thenLoad.getResult() ||
      elseYield.getOperand(0) != elseLoad.getResult()) {
    return std::nullopt;
  }
  if (thenStore.getAffineMap() != elseLoad.getAffineMap() ||
      !llvm::equal(thenStore.getMapOperands(), elseLoad.getMapOperands())) {
    return std::nullopt;
  }

  return NoRingReuseWrapper{thenLoad, thenStore, elseLoad};
}

static bool valueDependsOnTargetLoad(Value value, Value target,
                                     DenseMap<Value, bool> &cache,
                                     SmallPtrSetImpl<Value> &visiting) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (!visiting.insert(value).second)
    return false;

  bool depends = false;
  if (auto loadOp = value.getDefiningOp<affine::AffineLoadOp>()) {
    depends = loadOp.getMemRef() == target;
  } else if (auto result = dyn_cast<OpResult>(value)) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(result.getOwner())) {
      unsigned resultNumber = result.getResultNumber();
      auto yieldedDependsOnTarget = [&](Block *block) {
        if (!block)
          return false;
        auto yieldOp = dyn_cast<affine::AffineYieldOp>(block->getTerminator());
        if (!yieldOp || resultNumber >= yieldOp.getNumOperands())
          return false;
        return valueDependsOnTargetLoad(yieldOp.getOperand(resultNumber), target,
                                        cache, visiting);
      };
      depends = yieldedDependsOnTarget(ifOp.getThenBlock()) ||
                yieldedDependsOnTarget(ifOp.getElseBlock());
    } else {
      depends = llvm::any_of(result.getOwner()->getOperands(), [&](Value operand) {
        return valueDependsOnTargetLoad(operand, target, cache, visiting);
      });
    }
  } else if (Operation *defOp = value.getDefiningOp()) {
    depends = llvm::any_of(defOp->getOperands(), [&](Value operand) {
      return valueDependsOnTargetLoad(operand, target, cache, visiting);
    });
  }

  visiting.erase(value);
  cache[value] = depends;
  return depends;
}

// ReuseAt assumes canonical affine loops: lb=0, step=1, constant bounds.
static bool requireNormalizedLoop(affine::AffineForOp forOp) {
  return forOp.getStepAsInt() == 1 && forOp.hasConstantBounds() &&
         forOp.getConstantLowerBound() == 0;
}

// Walk parent loops to the outermost loop of the selected axis loop.
static affine::AffineForOp getRootLoop(affine::AffineForOp loop) {
  affine::AffineForOp root = loop;
  while (auto parent = root->getParentOfType<affine::AffineForOp>())
    root = parent;
  return root;
}

// Resolve the axis handle to exactly one affine.for loop.
static FailureOr<affine::AffineForOp>
resolveAxisLoop(transform::TransformState &state, transform::ReuseAtOp op) {
  auto payloadOps = state.getPayloadOps(op.getAxis());
  if (!llvm::hasSingleElement(payloadOps))
    return failure();
  auto axisLoop = dyn_cast<affine::AffineForOp>(*payloadOps.begin());
  if (!axisLoop)
    return failure();
  return axisLoop;
}

// Classify each loop IV in the root nest:
// - spatial: contributes to store indexing
// - reduction: contributes to target-load indexing but not store indexing
// Also cache reduction loop upper bounds for span/distance derivation.
static LogicalResult
classifyLoopRoles(affine::AffineForOp rootForOp, Value target,
                  LoopRoleInfo &roles,
                  SmallVectorImpl<affine::AffineForOp> &allLoops) {
  WalkResult walkResult = rootForOp.walk([&](affine::AffineForOp forOp) {
    if (!requireNormalizedLoop(forOp))
      return WalkResult::interrupt();
    allLoops.push_back(forOp);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  DenseSet<Value> spatialCandidates;
  DenseMap<Value, bool> loadDependenceCache;
  rootForOp.walk([&](affine::AffineStoreOp storeOp) {
    SmallPtrSet<Value, 16> visiting;
    if (!valueDependsOnTargetLoad(storeOp.getValueToStore(), target,
                                  loadDependenceCache, visiting)) {
      return WalkResult::advance();
    }
    AffineMap storeMap = storeOp.getAffineMap();
    auto storeOperands = storeOp.getMapOperands();
    for (AffineExpr resultExpr : storeMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, storeOperands,
                                storeMap.getNumDims(),
                                loop.getInductionVar())) {
          spatialCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  DenseSet<Value> loadCandidates;
  rootForOp.walk([&](affine::AffineLoadOp loadOp) {
    if (loadOp.getMemRef() != target)
      return WalkResult::advance();
    AffineMap loadMap = loadOp.getAffineMap();
    auto loadOperands = loadOp.getMapOperands();
    for (AffineExpr resultExpr : loadMap.getResults()) {
      for (affine::AffineForOp loop : allLoops) {
        if (affineExprUsesValue(resultExpr, loadOperands, loadMap.getNumDims(),
                                loop.getInductionVar())) {
          loadCandidates.insert(loop.getInductionVar());
        }
      }
    }
    return WalkResult::advance();
  });

  if (loadCandidates.empty())
    return failure();

  roles.spatialIVs = std::move(spatialCandidates);
  for (Value iv : loadCandidates) {
    if (!roles.spatialIVs.contains(iv))
      roles.reductionIVs.insert(iv);
  }

  for (affine::AffineForOp loop : allLoops) {
    Value iv = loop.getInductionVar();
    if (!roles.reductionIVs.contains(iv))
      continue;
    roles.reductionUpperBounds[iv] = loop.getConstantUpperBound();
  }
  return success();
}

// True if the loop IV is inferred as reduction-only in the current nest.
static bool isReductionLoop(affine::AffineForOp forOp,
                            const LoopRoleInfo &roles) {
  return roles.reductionIVs.contains(forOp.getInductionVar());
}

// True if the loop IV appears in spatial indexing (store side).
static bool isSpatialLoop(affine::AffineForOp forOp,
                          const LoopRoleInfo &roles) {
  return roles.spatialIVs.contains(forOp.getInductionVar());
}

static void
collectReuseInnerLoops(affine::AffineForOp axisLoop,
                       SmallVectorImpl<affine::AffineForOp> &innerLoops) {
  axisLoop.walk([&](affine::AffineForOp forOp) {
    if (forOp != axisLoop)
      innerLoops.push_back(forOp);
  });
}

static LogicalResult
collectReuseAccesses(affine::AffineForOp axisLoop, Value target,
                     SmallVectorImpl<ComposedBufferAccess> &accesses,
                     SmallVectorImpl<affine::AffineForOp> &innerLoops) {
  collectReuseInnerLoops(axisLoop, innerLoops);

  Value targetRoot = resolveMemRefValueRoot(target);
  WalkResult walk =
      axisLoop.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
          if (auto wrapper = matchNoRingReuseWrapper(ifOp, target)) {
            accesses.push_back(
                composeBufferAccess(wrapper->elseLoad.getOperation()));
            return WalkResult::skip();
          }
        }

        Value memref;
        if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
          memref = readOp.getMemRef();
          if (memref == target)
            accesses.push_back(composeBufferAccess(op));
        } else if (auto writeOp =
                       dyn_cast<affine::AffineWriteOpInterface>(op)) {
          memref = writeOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            auto diag = axisLoop.emitError()
                        << "reuse_at requires the target buffer to be read-only "
                           "within the selected axis loop";
            diag.attachNote(writeOp->getLoc()) << "see write op here";
            return WalkResult::interrupt();
          }
        } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          memref = loadOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            auto diag = emitError(targetRoot.getLoc())
                        << "reuse_at only supports affine.load accesses to the "
                           "target buffer";
            diag.attachNote(loadOp.getLoc()) << "see memref.load op here";
            return WalkResult::interrupt();
          }
        } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
          memref = storeOp.getMemRef();
          if (resolveMemRefValueRoot(memref) == targetRoot) {
            auto diag = axisLoop.emitError()
                        << "reuse_at requires the target buffer to be read-only "
                           "within the selected axis loop";
            diag.attachNote(storeOp.getLoc()) << "see memref.store op here";
            return WalkResult::interrupt();
          }
        } else if (isMemRefCastOrViewLike(op)) {
          bool aliasesTarget = llvm::any_of(op->getResults(), [&](Value result) {
            return isa<BaseMemRefType>(result.getType()) &&
                   resolveMemRefValueRoot(result) == targetRoot;
          });
          if (aliasesTarget) {
            auto diag = axisLoop.emitError()
                        << "reuse_at does not support aliasing/view accesses "
                           "to the "
                           "target buffer within the selected axis loop";
            diag.attachNote(op->getLoc()) << "see aliasing/view op here";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

  if (walk.wasInterrupted())
    return failure();
  if (accesses.empty()) {
    auto diag =
        axisLoop.emitError()
        << "no affine.load access to the target buffer found within the "
           "selected axis loop";
    return diag;
  }
  return success();
}

static FailureOr<ReuseDimFootprint>
computeReuseDimFootprint(const ComposedBufferAccess &access, unsigned resultPos,
                         ArrayRef<affine::AffineForOp> innerLoops, Value axisIV,
                         DenseMap<Value, unsigned> &prefixOperandPos,
                         ArrayRef<Value> prefixOperands) {
  AffineExpr accessExpr = access.map.getResult(resultPos);
  SmallVector<affine::AffineForOp, 4> dependentLoops;
  for (affine::AffineForOp loop : innerLoops) {
    if (affineExprUsesValue(accessExpr, access.operands,
                            access.map.getNumDims(), loop.getInductionVar())) {
      dependentLoops.push_back(loop);
    }
  }

  unsigned prefixDimCount = prefixOperands.size();
  SmallVector<AffineExpr, 8> dimReplacements, symReplacements;
  SmallVector<std::pair<Value, AffineExpr>, 4> lowerReplacements;
  lowerReplacements.emplace_back(axisIV,
                                 getAffineDimExpr(0, access.map.getContext()));
  populateExprReplacements(
      access.map, access.operands, prefixOperandPos, lowerReplacements,
      /*prefixDimOffset=*/1, dimReplacements, symReplacements);
  AffineExpr lowerExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      1 + prefixDimCount, /*numSymbols=*/0);
  FailureOr<int64_t> axisCoeffOr =
      getLinearAffineDimCoefficient(lowerExpr, 1 + prefixDimCount,
                                    /*dimPos=*/0);
  if (failed(axisCoeffOr))
    return failure();

  ReuseDimFootprint footprint;
  footprint.lowerExpr = lowerExpr;
  footprint.axisCoeff = *axisCoeffOr;

  if (dependentLoops.empty())
    return footprint;

  SmallVector<affine::AffineForOp, 4> activeLoops;
  SmallVector<int64_t, 4> activeLoopExtents;
  activeLoops.reserve(dependentLoops.size());
  activeLoopExtents.reserve(dependentLoops.size());
  for (affine::AffineForOp loop : dependentLoops) {
    if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1 ||
        loop.getConstantLowerBound() != 0)
      return failure();
    int64_t extent =
        loop.getConstantUpperBound() - loop.getConstantLowerBound();
    if (extent <= 0)
      return failure();
    activeLoops.push_back(loop);
    activeLoopExtents.push_back(extent);
  }

  SmallVector<std::pair<Value, AffineExpr>, 6> offsetReplacements;
  for (auto [idx, loop] : llvm::enumerate(activeLoops)) {
    offsetReplacements.emplace_back(
        loop.getInductionVar(), getAffineDimExpr(idx, access.map.getContext()));
  }
  populateExprReplacements(
      access.map, access.operands, prefixOperandPos, offsetReplacements,
      /*prefixDimOffset=*/activeLoops.size(), dimReplacements, symReplacements);
  AffineExpr offsetExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      activeLoops.size() + prefixDimCount, /*numSymbols=*/0);

  SmallVector<AffineExpr, 4> expandedLowerDims;
  expandedLowerDims.reserve(1 + prefixDimCount);
  expandedLowerDims.push_back(
      getAffineConstantExpr(0, access.map.getContext()));
  for (unsigned i = 0; i < prefixDimCount; ++i)
    expandedLowerDims.push_back(
        getAffineDimExpr(activeLoops.size() + i, access.map.getContext()));
  AffineExpr expandedLowerExpr =
      simplifyAffineExpr(lowerExpr.replaceDims(expandedLowerDims),
                         activeLoops.size() + prefixDimCount, /*numSymbols=*/0);
  AffineExpr innerOffsetExpr = simplifyAffineExpr(
      offsetExpr - expandedLowerExpr, activeLoops.size() + prefixDimCount,
      /*numSymbols=*/0);

  FailureOr<int64_t> innerConstOr = getConstantExprDelta(
      offsetExpr, expandedLowerExpr, activeLoops.size() + prefixDimCount);
  if (succeeded(innerConstOr) && *innerConstOr == 0)
    return footprint;

  FlatLinearConstraints localVarCst(activeLoops.size(), /*numSymbols=*/0);
  SmallVector<int64_t, 8> flattenedExpr;
  if (failed(getFlattenedAffineExpr(innerOffsetExpr, activeLoops.size(),
                                    /*numSymbols=*/0, &flattenedExpr,
                                    &localVarCst)))
    return failure();
  if (localVarCst.getNumLocalVars() != 0 ||
      flattenedExpr.size() != activeLoops.size() + 1)
    return failure();
  if (flattenedExpr.back() != 0)
    return failure();

  int64_t maxOffset = 0;
  for (auto [idx, coeff] :
       llvm::enumerate(ArrayRef<int64_t>(flattenedExpr).drop_back())) {
    if (coeff < 0 || coeff > 1)
      return failure();
    if (coeff == 0)
      continue;
    maxOffset += activeLoopExtents[idx] - 1;
  }

  footprint.extent = maxOffset + 1;
  footprint.innerMaxOffset = maxOffset;
  return footprint;
}

static FailureOr<ReuseStatePlan>
analyzeReuseStatePlan(ArrayRef<ComposedBufferAccess> accesses,
                      ArrayRef<affine::AffineForOp> innerLoops,
                      affine::AffineForOp axisLoop, unsigned bufferRank,
                      MLIRContext *ctx) {
  ReuseStatePlan plan;
  Value axisIV = axisLoop.getInductionVar();
  DenseMap<Value, unsigned> prefixOperandPos;
  collectFootprintOperands(accesses, innerLoops, ArrayRef<Value>{axisIV},
                           prefixOperandPos, plan.prefixOperands);
  unsigned prefixDimCount = plan.prefixOperands.size();

  SmallVector<bool, 4> stationarySeen(bufferRank, false);
  SmallVector<AffineExpr, 4> refLower(bufferRank);
  SmallVector<int64_t, 4> minOffset(bufferRank, 0);
  SmallVector<int64_t, 4> maxUpper(bufferRank, 0);
  SmallVector<int64_t, 4> windowMinOffset(bufferRank, 0);
  SmallVector<int64_t, 4> windowMaxOffset(bufferRank, 0);
  SmallVector<SmallVector<int64_t, 4>, 8> axisCoeffs(accesses.size());

  std::optional<unsigned> detectedSlidingDim;
  std::optional<int64_t> detectedSlidingDelta;

  for (auto [accessIdx, access] : llvm::enumerate(accesses)) {
    auto readOp = cast<affine::AffineReadOpInterface>(access.op);
    if (access.map.getNumResults() != bufferRank) {
      auto diag = readOp->emitError()
                  << "reuse_at requires candidate loads to match the target "
                     "buffer rank";
      diag.attachNote(readOp->getLoc())
          << "candidate affine.load has " << access.map.getNumResults()
          << " indices, but the target buffer rank is " << bufferRank;
      return diag;
    }

    axisCoeffs[accessIdx].resize(bufferRank, 0);
    for (unsigned d = 0; d < bufferRank; ++d) {
      FailureOr<ReuseDimFootprint> footprintOr = computeReuseDimFootprint(
          access, d, innerLoops, axisIV, prefixOperandPos, plan.prefixOperands);
      if (failed(footprintOr)) {
        auto diag = readOp->emitError()
                    << "reuse_at requires buffer dimensions to have bounded "
                       "contiguous affine footprints";
        diag.attachNote(readOp->getLoc())
            << "failed to derive a contiguous local footprint for buffer "
               "dimension "
            << d;
        return diag;
      }
      const ReuseDimFootprint &footprint = *footprintOr;
      int64_t delta = footprint.axisCoeff;
      axisCoeffs[accessIdx][d] = delta;

      if (delta != 0) {
        if (detectedSlidingDim && *detectedSlidingDim != d) {
          auto diag = readOp->emitError()
                      << "candidate loads do not share a common sliding "
                         "dimension";
          diag.attachNote(readOp->getLoc())
              << "candidate affine.load slides along buffer dimension " << d
              << ", but previous candidates slide along buffer dimension "
              << *detectedSlidingDim;
          return diag;
        }
        if (detectedSlidingDelta && *detectedSlidingDelta != delta) {
          auto diag = readOp->emitError()
                      << "candidate loads do not share a common sliding "
                         "direction";
          diag.attachNote(readOp->getLoc())
              << "candidate affine.load uses selected-axis coefficient "
              << delta << " in buffer dimension " << d
              << ", but previous candidates use coefficient "
              << *detectedSlidingDelta;
          return diag;
        }
        detectedSlidingDim = d;
        detectedSlidingDelta = delta;
      }

      if (!stationarySeen[d]) {
        stationarySeen[d] = true;
        refLower[d] = footprint.lowerExpr;
        minOffset[d] = 0;
        maxUpper[d] = footprint.extent;
        windowMinOffset[d] = footprint.innerMinOffset;
        windowMaxOffset[d] = footprint.innerMaxOffset;
        continue;
      }

      FailureOr<int64_t> offsetOr = getConstantExprDelta(
          footprint.lowerExpr, refLower[d], 1 + prefixDimCount);
      if (failed(offsetOr)) {
        auto diag =
            readOp->emitError()
            << (delta != 0 ? "candidate loads do not share a common sliding "
                             "coordinate system"
                           : "candidate loads do not share a common local "
                             "coordinate system");
        diag.attachNote(readOp->getLoc())
            << "candidate affine.load uses a non-constant local offset for "
               "buffer dimension "
            << d;
        return diag;
      }
      minOffset[d] = std::min(minOffset[d], *offsetOr);
      maxUpper[d] = std::max(maxUpper[d], *offsetOr + footprint.extent);
      windowMinOffset[d] =
          std::min(windowMinOffset[d], *offsetOr + footprint.innerMinOffset);
      windowMaxOffset[d] =
          std::max(windowMaxOffset[d], *offsetOr + footprint.innerMaxOffset);
    }
  }

  if (!detectedSlidingDim || !detectedSlidingDelta) {
    auto diag = axisLoop->emitError()
                << "cannot find a reusable sliding dimension for the selected "
                   "axis";
    diag.attachNote(axisLoop.getLoc())
        << "none of the candidate affine.load accesses varies with the "
           "selected axis";
    return diag;
  }

  for (auto [accessIdx, coeffs] : llvm::enumerate(axisCoeffs)) {
    if (coeffs[*detectedSlidingDim] == 0) {
      auto readOp = cast<affine::AffineReadOpInterface>(accesses[accessIdx].op);
      auto diag = readOp->emitError()
                  << "candidate loads do not all depend on the selected axis "
                     "through the same sliding dimension";
      diag.attachNote(readOp->getLoc())
          << "candidate affine.load does not depend on the selected axis "
             "through buffer dimension "
          << *detectedSlidingDim;
      return diag;
    }
    for (auto [dim, coeff] : llvm::enumerate(coeffs)) {
      if (dim != *detectedSlidingDim && coeff != 0) {
        auto readOp =
            cast<affine::AffineReadOpInterface>(accesses[accessIdx].op);
        auto diag = readOp->emitError()
                    << "candidate loads do not share a common sliding "
                       "dimension";
        diag.attachNote(readOp->getLoc())
            << "candidate affine.load also depends on the selected axis "
               "through buffer dimension "
            << dim;
        return diag;
      }
    }
  }

  plan.dims.resize(bufferRank);
  plan.resultToReusePos.assign(bufferRank, -1);
  plan.slidingDim = *detectedSlidingDim;
  plan.slidingDelta = *detectedSlidingDelta;
  plan.slidingStepAbs = std::abs(*detectedSlidingDelta);

  for (unsigned d = 0; d < bufferRank; ++d) {
    ReuseDimPlan &dimPlan = plan.dims[d];
    if (!stationarySeen[d]) {
      auto diag = axisLoop->emitError()
                  << "reuse_at failed to derive a local footprint for a "
                     "buffer dimension";
      diag.attachNote(axisLoop.getLoc())
          << "failed while materializing local state for buffer dimension "
          << d;
      return diag;
    }

    AffineExpr anchorExpr =
        simplifyAffineExpr(refLower[d] + minOffset[d], 1 + prefixDimCount,
                           /*numSymbols=*/0);
    dimPlan.anchorMap =
        AffineMap::get(1 + prefixDimCount, /*symbolCount=*/0, anchorExpr, ctx);
    dimPlan.extent = maxUpper[d] - minOffset[d];
    dimPlan.axisCoeff = d == *detectedSlidingDim ? *detectedSlidingDelta : 0;
    dimPlan.innerMinOffset = windowMinOffset[d] - minOffset[d];
    dimPlan.innerMaxOffset = windowMaxOffset[d] - minOffset[d];
    dimPlan.isSliding = d == *detectedSlidingDim;

    if (dimPlan.extent <= 0) {
      auto diag = axisLoop->emitError()
                  << "reuse_at derived a non-positive local state extent";
      diag.attachNote(axisLoop.getLoc()) << "derived extent " << dimPlan.extent
                                         << " for buffer dimension " << d;
      return diag;
    }

    if (dimPlan.isSliding || dimPlan.extent > 1) {
      plan.resultToReusePos[d] = static_cast<int>(plan.keptDims.size());
      plan.keptDims.push_back(d);
      plan.shape.push_back(dimPlan.extent);
    }
  }

  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  if (plan.slidingStepAbs >= slidingExtent) {
    auto diag = axisLoop->emitError()
                << "reuse_at requires cross-iteration overlap on the sliding "
                   "dimension";
    diag.attachNote(axisLoop.getLoc())
        << "sliding step " << plan.slidingStepAbs
        << " does not leave reusable overlap within extent " << slidingExtent;
    return diag;
  }

  return plan;
}

static SmallVector<Value, 4>
getReuseStateOperands(Value axisIV, ArrayRef<Value> prefixOperands) {
  SmallVector<Value, 4> operands;
  operands.push_back(axisIV);
  operands.append(prefixOperands.begin(), prefixOperands.end());
  return operands;
}

static Value materializeGlobalAccessIndex(OpBuilder &builder, Location loc,
                                          const ComposedBufferAccess &access,
                                          unsigned resultDim);

static affine::AffineForOp createConstantAffineFor(OpBuilder &builder,
                                                   Location loc, int64_t lb,
                                                   int64_t ub) {
  AffineMap lowerMap = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                                      builder.getAffineConstantExpr(lb));
  AffineMap upperMap = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                                      builder.getAffineConstantExpr(ub));
  return affine::createCanonicalizedAffineForOp(
      builder, loc, /*lbOperands=*/ValueRange{}, lowerMap,
      /*ubOperands=*/ValueRange{}, upperMap, /*step=*/1);
}

static Value buildAnchorValue(OpBuilder &builder, Location loc,
                              const ReuseDimPlan &dimPlan,
                              ValueRange stateOperands) {
  SmallVector<OpFoldResult, 4> ofrs;
  ofrs.reserve(stateOperands.size());
  for (Value operand : stateOperands)
    ofrs.push_back(operand);
  return affine::makeComposedAffineApply(builder, loc, dimPlan.anchorMap, ofrs);
}

static Value buildOffsetValue(OpBuilder &builder, Location loc, Value base,
                              Value offset) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  return affine::makeComposedAffineApply(
      builder, loc, AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, d0 + d1),
      {base, offset});
}

static Value buildOffsetValue(OpBuilder &builder, Location loc, Value base,
                              int64_t offset) {
  AffineExpr d0 = builder.getAffineDimExpr(0);
  return affine::makeComposedAffineApply(
      builder, loc,
      AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                     d0 + builder.getAffineConstantExpr(offset)),
      {base});
}

static void generateReuseStateShift(OpBuilder &builder, Location loc,
                                    Value reuseBuffer,
                                    const ReuseExecutionPlan &executionPlan) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  if (slidingReusePos < 0 || slidingExtent <= 1)
    return;

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> loopIvs(plan.keptDims.size());
  for (auto [reusePos, dim] : llvm::enumerate(plan.keptDims)) {
    int64_t lb = 0;
    int64_t ub = plan.shape[reusePos];
    if (static_cast<int>(reusePos) == slidingReusePos) {
      if (executionPlan.slidingDelta > 0) {
        ub = slidingExtent - executionPlan.slidingStepAbs;
      } else {
        lb = executionPlan.slidingStepAbs;
      }
    }
    affine::AffineForOp forOp =
        createConstantAffineFor(builder, loc, /*lb=*/lb, /*ub=*/ub);
    builder.setInsertionPoint(
        forOp.getBody(), Block::iterator(forOp.getBody()->getTerminator()));
    loopIvs[reusePos] = forOp.getInductionVar();
  }

  SmallVector<Value, 4> srcIndices = loopIvs;
  SmallVector<Value, 4> dstIndices = loopIvs;
  srcIndices[slidingReusePos] = buildOffsetValue(
      builder, loc, loopIvs[slidingReusePos], executionPlan.slidingDelta);

  Value shifted =
      affine::AffineLoadOp::create(builder, loc, reuseBuffer, srcIndices);
  affine::AffineStoreOp::create(builder, loc, shifted, reuseBuffer, dstIndices);
}

static Value buildModuloOffsetValue(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs, int64_t modulus) {
  Value sum = arith::AddIOp::create(builder, loc, lhs, rhs);
  Value modulusValue = arith::ConstantIndexOp::create(builder, loc, modulus);
  return arith::RemUIOp::create(builder, loc, sum, modulusValue);
}

static SmallVector<Value, 4> materializeLogicalReuseIndices(
    OpBuilder &builder, Location loc, const ReuseStatePlan &plan,
    const ComposedBufferAccess &access, ValueRange stateOperands) {
  SmallVector<Value, 4> logicalIndices;
  logicalIndices.reserve(plan.keptDims.size());
  for (unsigned resultDim : plan.keptDims) {
    Value globalIndex =
        materializeGlobalAccessIndex(builder, loc, access, resultDim);
    Value anchor =
        buildAnchorValue(builder, loc, plan.dims[resultDim], stateOperands);
    logicalIndices.push_back(affine::makeComposedAffineApply(
        builder, loc,
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       builder.getAffineDimExpr(0) -
                           builder.getAffineDimExpr(1)),
        {globalIndex, anchor}));
  }
  return logicalIndices;
}

static SmallVector<Value, 4>
materializePhysicalReuseIndices(OpBuilder &builder, Location loc,
                                const ReuseExecutionPlan &executionPlan,
                                ArrayRef<Value> logicalIndices, Value ringHead,
                                Value physicalSlidingIndex = Value()) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  SmallVector<Value, 4> physicalIndices(logicalIndices.begin(),
                                        logicalIndices.end());
  if (executionPlan.strategy != ReuseBufferStrategy::Ring)
    return physicalIndices;

  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  assert(slidingReusePos >= 0 && "expected sliding dimension to be kept");
  if (physicalSlidingIndex) {
    physicalIndices[slidingReusePos] = physicalSlidingIndex;
    return physicalIndices;
  }
  assert(ringHead && "expected ring-head value for ring strategy");
  physicalIndices[slidingReusePos] = buildModuloOffsetValue(
      builder, loc, ringHead, logicalIndices[slidingReusePos],
      plan.dims[plan.slidingDim].extent);
  return physicalIndices;
}

static void generateReuseStateRefill(OpBuilder &builder, Location loc,
                                     Value sourceBuffer, Value reuseBuffer,
                                     const ReuseExecutionPlan &executionPlan,
                                     ValueRange stateOperands,
                                     Value ringHead = Value()) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  int slidingReusePos = plan.resultToReusePos[plan.slidingDim];
  int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
  int64_t enteringBaseOffset =
      executionPlan.slidingDelta > 0
          ? slidingExtent - executionPlan.slidingStepAbs
          : 0;

  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value, 4> logicalIndices(plan.keptDims.size());
  Value physicalSlidingIndex;
  if (slidingReusePos >= 0) {
    affine::AffineForOp enteringFaceFor = createConstantAffineFor(
        builder, loc, /*lb=*/0, /*ub=*/executionPlan.slidingStepAbs);
    builder.setInsertionPoint(
        enteringFaceFor.getBody(),
        Block::iterator(enteringFaceFor.getBody()->getTerminator()));
    Value enteringFaceIV = enteringFaceFor.getInductionVar();
    logicalIndices[slidingReusePos] =
        executionPlan.slidingDelta > 0
            ? buildOffsetValue(builder, loc, enteringFaceIV, enteringBaseOffset)
            : enteringFaceIV;
    if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
      physicalSlidingIndex = buildModuloOffsetValue(
          builder, loc, ringHead, logicalIndices[slidingReusePos],
          slidingExtent);
    }
  }

  for (auto [reusePos, dim] : llvm::enumerate(plan.keptDims)) {
    if (static_cast<int>(reusePos) == slidingReusePos)
      continue;
    affine::AffineForOp forOp =
        createConstantAffineFor(builder, loc, /*lb=*/0,
                                /*ub=*/plan.shape[reusePos]);
    builder.setInsertionPoint(
        forOp.getBody(), Block::iterator(forOp.getBody()->getTerminator()));
    logicalIndices[reusePos] = forOp.getInductionVar();
  }

  SmallVector<Value, 4> globalIndices(plan.dims.size());
  for (auto [resultDim, dimPlan] : llvm::enumerate(plan.dims)) {
    Value anchor = buildAnchorValue(builder, loc, dimPlan, stateOperands);
    int reusePos = plan.resultToReusePos[resultDim];
    if (reusePos < 0) {
      globalIndices[resultDim] = anchor;
      continue;
    }
    globalIndices[resultDim] =
        buildOffsetValue(builder, loc, anchor, logicalIndices[reusePos]);
  }

  Value loaded =
      affine::AffineLoadOp::create(builder, loc, sourceBuffer, globalIndices);
  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    SmallVector<Value, 4> physicalIndices = materializePhysicalReuseIndices(
        builder, loc, executionPlan, logicalIndices, ringHead,
        physicalSlidingIndex);
    memref::StoreOp::create(builder, loc, loaded, reuseBuffer, physicalIndices);
    return;
  }
  affine::AffineStoreOp::create(builder, loc, loaded, reuseBuffer,
                                logicalIndices);
}

static Value createConditionalReuseLoad(OpBuilder &builder, Location loc,
                                        const ComposedBufferAccess &access,
                                        Value sourceBuffer, Value reuseBuffer,
                                        const ReuseExecutionPlan &executionPlan,
                                        ValueRange stateOperands, Value axisIV,
                                        Value currentIterationRingHead) {
  const ReuseStatePlan &plan = executionPlan.statePlan;
  SmallVector<Value, 4> logicalIndices =
      materializeLogicalReuseIndices(builder, loc, plan, access, stateOperands);

  IntegerSet warmupSet = IntegerSet::get(
      /*dimCount=*/1, /*symbolCount=*/0,
      ArrayRef<AffineExpr>{builder.getAffineDimExpr(0)}, ArrayRef<bool>{true});
  auto ifOp = affine::AffineIfOp::create(
      builder, loc, TypeRange{access.op->getResult(0).getType()}, warmupSet,
      ValueRange{axisIV}, /*withElseRegion=*/true);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(ifOp.getThenBlock());
    Value loaded = affine::AffineLoadOp::create(builder, loc, sourceBuffer,
                                                access.map, access.operands);
    affine::AffineStoreOp::create(builder, loc, loaded, reuseBuffer,
                                  logicalIndices);
    affine::AffineYieldOp::create(builder, loc, loaded);
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(ifOp.getElseBlock());
    SmallVector<Value, 4> physicalIndices = materializePhysicalReuseIndices(
        builder, loc, executionPlan, logicalIndices, currentIterationRingHead);
    Value reused;
    if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
      reused =
          memref::LoadOp::create(builder, loc, reuseBuffer, physicalIndices);
    } else {
      reused = affine::AffineLoadOp::create(builder, loc, reuseBuffer,
                                            physicalIndices);
    }
    affine::AffineYieldOp::create(builder, loc, reused);
  }

  return ifOp.getResult(0);
}

static Value materializeGlobalAccessIndex(OpBuilder &builder, Location loc,
                                          const ComposedBufferAccess &access,
                                          unsigned resultDim) {
  AffineMap singleResultMap = access.map.getSubMap(resultDim);
  SmallVector<OpFoldResult, 4> ofrs;
  for (Value operand : access.operands)
    ofrs.push_back(operand);
  return affine::makeComposedAffineApply(builder, loc, singleResultMap, ofrs);
}

DiagnosedSilenceableFailure
transform::ReuseAtOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  // preconditions: input handles must resolve to exactly one value/operation of
  // the expected type
  auto targets = llvm::to_vector(state.getPayloadValues(getTarget()));
  if (targets.size() != 1) {
    return emitSilenceableError()
           << "expected target handle to resolve to exactly one payload value";
  }
  Value target = targets.front();
  MemRefType targetType = dyn_cast<MemRefType>(target.getType());
  if (!targetType)
    return emitSilenceableError()
           << "expected target to resolve to a memref value";

  auto loops = llvm::to_vector(state.getPayloadOps(getAxis()));
  if (loops.size() != 1) {
    return emitSilenceableError() << "expected axis handle to resolve to "
                                     "exactly one payload operation";
  }
  auto axisLoop = dyn_cast<affine::AffineForOp>(loops.front());
  if (!axisLoop)
    return emitSilenceableError()
           << "expected axis to resolve to exactly one affine.for loop";

  // require normalized loops
  if (axisLoop.getStepAsInt() != 1 || !axisLoop.hasConstantBounds() ||
      axisLoop.getConstantLowerBound() != 0)
    return emitSilenceableError()
           << "reuse_at requires the selected axis loop to have lower bound 0, "
              "step 1, and constant bounds";

  // require the target buffer to be defined outside the axis loop
  Operation *targetDef = target.getDefiningOp();
  if (targetDef && axisLoop->isAncestor(targetDef))
    return emitSilenceableError()
           << "expected target buffer to be defined outside the selected axis "
              "loop";

  Value axisIV = axisLoop.getInductionVar();
  unsigned rank = targetType.getRank();

  affine::AffineForOp rootLoop = getRootLoop(axisLoop);
  LoopRoleInfo roles;
  SmallVector<affine::AffineForOp, 8> allLoops;
  if (failed(classifyLoopRoles(rootLoop, target, roles, allLoops))) {
    return emitSilenceableError()
           << "failed to classify loop roles; loops must be normalized and "
              "target must be loaded in the axis stage";
  }
  // The chosen axis must be spatial (store-indexing), not reduction-only.
  if (isReductionLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is classified as a reduction loop";
  if (!isSpatialLoop(axisLoop, roles))
    return emitSilenceableError()
           << "selected axis loop is not classified as a spatial loop";

  SmallVector<ComposedBufferAccess, 8> accesses;
  SmallVector<affine::AffineForOp, 8> innerLoops;
  if (failed(collectReuseAccesses(axisLoop, target, accesses, innerLoops)))
    return emitSilenceableError()
           << "failed to collect reuse candidate accesses";

  FailureOr<ReuseStatePlan> planOr = analyzeReuseStatePlan(
      accesses, innerLoops, axisLoop, rank, rewriter.getContext());
  if (failed(planOr))
    return emitSilenceableError() << "failed to analyze reuse state plan";
  ReuseExecutionPlan executionPlan(std::move(*planOr), getUseRingBuffer());
  const ReuseStatePlan &plan = executionPlan.statePlan;

  rewriter.setInsertionPoint(axisLoop);
  auto reuseBuffer = memref::AllocOp::create(
      rewriter, axisLoop.getLoc(),
      MemRefType::get(plan.shape, targetType.getElementType()));
  if (targetDef) {
    if (auto targetSymName = targetDef->getAttrOfType<StringAttr>(OpIdentifier))
      reuseBuffer->setAttr(
          OpIdentifier, StringAttr::get(reuseBuffer->getContext(),
                                        targetSymName.getValue() + "::reuse"));
  }
  Value ringHeadBuffer;
  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    ringHeadBuffer = memref::AllocOp::create(
        rewriter, axisLoop.getLoc(),
        MemRefType::get(/*shape=*/{}, rewriter.getIndexType()));
    if (targetDef) {
      if (auto targetSymName =
              targetDef->getAttrOfType<StringAttr>(OpIdentifier)) {
        cast<memref::AllocOp>(ringHeadBuffer.getDefiningOp())
            ->setAttr(OpIdentifier, StringAttr::get(rewriter.getContext(),
                                                    targetSymName.getValue() +
                                                        "::reuse_head"));
      }
    }
    Value zero = arith::ConstantIndexOp::create(rewriter, axisLoop.getLoc(), 0);
    memref::StoreOp::create(rewriter, axisLoop.getLoc(), zero, ringHeadBuffer,
                            ValueRange{});
  }

  SmallVector<Value, 4> stateOperands =
      getReuseStateOperands(axisIV, plan.prefixOperands);

  rewriter.setInsertionPointToStart(axisLoop.getBody());
  Value currentIterationRingHead;
  IntegerSet steadyStateSet = IntegerSet::get(
      /*dimCount=*/1, /*symbolCount=*/0,
      ArrayRef<AffineExpr>{rewriter.getAffineDimExpr(0) -
                           executionPlan.steadyStateStart},
      ArrayRef<bool>{false});
  if (executionPlan.strategy == ReuseBufferStrategy::Ring) {
    auto updateIf = affine::AffineIfOp::create(
        rewriter, axisLoop.getLoc(), TypeRange{rewriter.getIndexType()},
        steadyStateSet, ValueRange{axisIV}, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(updateIf.getThenBlock());
      int64_t slidingExtent = plan.dims[plan.slidingDim].extent;
      Value currentHead = memref::LoadOp::create(rewriter, axisLoop.getLoc(),
                                                 ringHeadBuffer, ValueRange{});
      Value increment = arith::ConstantIndexOp::create(
          rewriter, axisLoop.getLoc(), executionPlan.ringIncrement);
      Value nextHead = buildModuloOffsetValue(
          rewriter, axisLoop.getLoc(), currentHead, increment, slidingExtent);
      memref::StoreOp::create(rewriter, axisLoop.getLoc(), nextHead,
                              ringHeadBuffer, ValueRange{});
      generateReuseStateRefill(rewriter, axisLoop.getLoc(), target, reuseBuffer,
                               executionPlan, stateOperands, nextHead);
      affine::AffineYieldOp::create(rewriter, axisLoop.getLoc(), nextHead);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(updateIf.getElseBlock());
      Value zero =
          arith::ConstantIndexOp::create(rewriter, axisLoop.getLoc(), 0);
      affine::AffineYieldOp::create(rewriter, axisLoop.getLoc(), zero);
    }
    currentIterationRingHead = updateIf.getResult(0);
  } else {
    auto updateIf = affine::AffineIfOp::create(
        rewriter, axisLoop.getLoc(), steadyStateSet, ValueRange{axisIV},
        /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(
        updateIf.getThenBlock(),
        Block::iterator(updateIf.getThenBlock()->getTerminator()));
    generateReuseStateShift(rewriter, axisLoop.getLoc(), reuseBuffer,
                            executionPlan);
    generateReuseStateRefill(rewriter, axisLoop.getLoc(), target, reuseBuffer,
                             executionPlan, stateOperands);
  }

  for (const auto &access : accesses) {
    rewriter.setInsertionPoint(access.op);
    Location loadLoc = access.op->getLoc();
    Value rewritten = createConditionalReuseLoad(
        rewriter, loadLoc, access, target, reuseBuffer, executionPlan,
        stateOperands, axisIV, currentIterationRingHead);
    rewriter.replaceOp(access.op, rewritten);
  }

  results.setValues(cast<OpResult>(getResult()), {reuseBuffer});
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===///
/// LoopPipeline implementation
///===----------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::TagPipelineOp::applyToOne(transform::TransformRewriter &rewriter,
                                     Operation *target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  if (!target || !isa<LoopLikeOpInterface>(target)) {
    return emitSilenceableError()
           << "expected target to resolve to exactly one loop-like operation";
  }
  auto ii = getIiAttr().getInt();
  if (ii <= 0) {
    return emitSilenceableError() << "expected ii to be a positive integer";
  }
  target->setAttr("pipeline.ii", getIiAttr());
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===///
/// LoopUnroll implementation
///===----------------------------------------------------------------------===///
DiagnosedSilenceableFailure
transform::TagUnrollOp::applyToOne(transform::TransformRewriter &rewriter,
                                   Operation *target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  if (!target || !isa<LoopLikeOpInterface>(target)) {
    return emitSilenceableError()
           << "expected target to resolve to exactly one loop-like operation";
  }
  auto factor = getFactorAttr().getInt();
  if (factor < 0) {
    return emitSilenceableError()
           << "expected unroll factor to be a non-negative "
           << "integer (0 for full unroll)";
  }
  target->setAttr("unroll.f", getFactorAttr());
  return DiagnosedSilenceableFailure::success();
}

///===----------------------------------------------------------------------===///
/// BufferAt implementation
///===----------------------------------------------------------------------===///

namespace {
struct BufferAtFootprint {
  // Per-instance local buffer layout derived from all accesses under the chosen
  // axis. `symbols` are the outer operands needed to materialize bounds/remaps.
  SmallVector<int64_t, 4> shape;
  SmallVector<AffineMap, 4> lowerBounds;
  SmallVector<AffineMap, 4> upperBounds;
  SmallVector<Value, 4> symbols;
  AffineMap localIndexRemap;
};
} // namespace

static LogicalResult
collectBufferAtAccesses(affine::AffineForOp axisLoop, Value buffer,
                        SmallVectorImpl<Operation *> &accessOps, bool &hasLoads,
                        bool &hasStores) {
  // Gather the affine accesses that will be rewritten to the local buffer.
  // At the same time, reject non-affine direct accesses and alias/view-based
  // accesses because the later remap step only understands direct affine
  // accesses to the chosen memref value.
  Value bufferRoot = resolveMemRefValueRoot(buffer);

  auto walkResult = axisLoop.walk([&](Operation *op) {
    if (auto readOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      Value memref = readOp.getMemRef();
      if (memref == buffer) {
        accessOps.push_back(op);
        hasLoads = true;
      }
    } else if (auto writeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      Value memref = writeOp.getMemRef();
      if (memref == buffer) {
        accessOps.push_back(op);
        hasStores = true;
      }
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      Value memref = loadOp.getMemRef();
      if (resolveMemRefValueRoot(memref) == bufferRoot) {
        auto diag = axisLoop.emitError()
                    << "buffer_at only supports affine.load/store accesses "
                       "to the target buffer within the selected axis loop";
        diag.attachNote(loadOp->getLoc()) << "see memref.load op here";
        return WalkResult::interrupt();
      }
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      Value memref = storeOp.getMemRef();
      if (resolveMemRefValueRoot(memref) == bufferRoot) {
        auto diag = axisLoop.emitError()
                    << "buffer_at only supports affine.load/store accesses "
                       "to the target buffer within the selected axis loop";
        diag.attachNote(storeOp->getLoc()) << "see memref.store op here";
        return WalkResult::interrupt();
      }
    } else if (isMemRefCastOrViewLike(op)) {
      bool aliasesBuffer = llvm::any_of(op->getResults(), [&](Value result) {
        return isa<BaseMemRefType>(result.getType()) &&
               resolveMemRefValueRoot(result) == bufferRoot;
      });
      if (aliasesBuffer) {
        auto diag = axisLoop.emitError()
                    << "buffer_at does not support aliasing/view accesses to "
                       "the target buffer within the selected axis loop";
        diag.attachNote(op->getLoc()) << "see aliasing/view op here";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

static FailureOr<std::pair<AffineExpr, int64_t>>
computeFootprintDim(const ComposedBufferAccess &access, unsigned resultPos,
                    ArrayRef<affine::AffineForOp> innerLoops,
                    DenseMap<Value, unsigned> &prefixOperandPos,
                    SmallVectorImpl<Value> &prefixOperands) {
  // Infer one local-buffer dimension from one access result. We only accept a
  // singleton point or a unit-stride interval driven by exactly one inner loop,
  // which keeps the local allocation finite and easy to reindex.
  AffineExpr accessExpr = access.map.getResult(resultPos);
  SmallVector<affine::AffineForOp, 2> dependentLoops;
  for (affine::AffineForOp loop : innerLoops) {
    if (affineExprUsesValue(accessExpr, access.operands,
                            access.map.getNumDims(), loop.getInductionVar())) {
      dependentLoops.push_back(loop);
    }
  }

  SmallVector<AffineExpr, 8> dimReplacements, symReplacements;
  if (dependentLoops.empty()) {
    populateExprReplacements(access.map, access.operands, prefixOperandPos,
                             prefixOperands, Value{}, std::nullopt,
                             std::nullopt, /*prefixDimOffset=*/0,
                             dimReplacements, symReplacements);
    AffineExpr lowerExpr = simplifyAffineExpr(
        accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
        prefixOperands.size(), /*numSymbols=*/0);
    return std::make_pair(lowerExpr, static_cast<int64_t>(1));
  }

  if (dependentLoops.size() != 1)
    return failure();

  affine::AffineForOp loop = dependentLoops.front();
  if (!loop.hasConstantBounds() || loop.getStepAsInt() != 1)
    return failure();

  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  if (ub <= lb)
    return failure();

  populateExprReplacements(access.map, access.operands, prefixOperandPos,
                           prefixOperands, loop.getInductionVar(), lb,
                           std::nullopt, /*prefixDimOffset=*/0, dimReplacements,
                           symReplacements);
  AffineExpr lowerExpr = simplifyAffineExpr(
      accessExpr.replaceDimsAndSymbols(dimReplacements, symReplacements),
      prefixOperands.size(), /*numSymbols=*/0);

  SmallVector<AffineExpr, 8> diffDims, diffSyms;
  populateExprReplacements(access.map, access.operands, prefixOperandPos,
                           prefixOperands, loop.getInductionVar(), std::nullopt,
                           /*targetLoopDimPos=*/0, /*prefixDimOffset=*/1,
                           diffDims, diffSyms);
  // Check that varying the chosen loop produces a zero-based contiguous index:
  // `(iv -> expr)` must be equivalent to `(iv - lb)` after subtracting the
  // common lower bound. This rules out gaps, permutations, and nonlinear forms.
  AffineExpr shiftedExpr =
      simplifyAffineExpr(accessExpr.replaceDimsAndSymbols(diffDims, diffSyms),
                         1 + prefixOperands.size(), /*numSymbols=*/0);

  SmallVector<AffineExpr, 4> expandedPrefixDims;
  expandedPrefixDims.reserve(prefixOperands.size());
  for (unsigned i = 0; i < prefixOperands.size(); ++i)
    expandedPrefixDims.push_back(
        getAffineDimExpr(i + 1, access.map.getContext()));
  AffineExpr expandedLowerExpr =
      simplifyAffineExpr(lowerExpr.replaceDims(expandedPrefixDims),
                         1 + prefixOperands.size(), /*numSymbols=*/0);
  AffineExpr zeroBasedExpr = simplifyAffineExpr(shiftedExpr - expandedLowerExpr,
                                                1 + prefixOperands.size(),
                                                /*numSymbols=*/0);
  AffineExpr expectedExpr =
      simplifyAffineExpr(getAffineDimExpr(0, access.map.getContext()) - lb,
                         1 + prefixOperands.size(), /*numSymbols=*/0);
  if (zeroBasedExpr != expectedExpr)
    return failure();

  return std::make_pair(lowerExpr, ub - lb);
}

static FailureOr<BufferAtFootprint>
analyzeBufferAtFootprint(ArrayRef<Operation *> accessOps,
                         ArrayRef<affine::AffineForOp> innerLoops,
                         unsigned bufferRank, MLIRContext *ctx) {
  // All accesses inside one axis instance must fit into the same local layout.
  // We infer a footprint per access and require them to agree exactly before
  // building copy loops and the global-to-local remap.
  BufferAtFootprint footprint;
  footprint.shape.resize(bufferRank);
  footprint.lowerBounds.resize(bufferRank);
  footprint.upperBounds.resize(bufferRank);

  SmallVector<ComposedBufferAccess, 8> accesses;
  accesses.reserve(accessOps.size());
  for (Operation *accessOp : accessOps)
    accesses.push_back(composeBufferAccess(accessOp));

  DenseMap<Value, unsigned> prefixOperandPos;
  collectFootprintOperands(accesses, innerLoops, /*excludedValues=*/{},
                           prefixOperandPos, footprint.symbols);
  SmallVector<std::pair<AffineExpr, int64_t>, 4> commonFootprint;
  commonFootprint.reserve(bufferRank);

  for (const ComposedBufferAccess &access : accesses) {
    if (access.map.getNumResults() != bufferRank)
      return failure();

    SmallVector<std::pair<AffineExpr, int64_t>, 4> accessFootprint;
    accessFootprint.reserve(bufferRank);
    for (unsigned d = 0; d < bufferRank; ++d) {
      auto dimFootprint = computeFootprintDim(
          access, d, innerLoops, prefixOperandPos, footprint.symbols);
      if (failed(dimFootprint))
        return failure();
      accessFootprint.push_back(*dimFootprint);
    }

    if (commonFootprint.empty()) {
      commonFootprint = accessFootprint;
      continue;
    }
    if (commonFootprint != accessFootprint)
      return failure();
  }

  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(bufferRank);
  for (unsigned d = 0; d < bufferRank; ++d) {
    AffineExpr lowerExpr = commonFootprint[d].first;
    int64_t extent = commonFootprint[d].second;
    if (extent <= 0)
      return failure();

    footprint.shape[d] = extent;
    footprint.lowerBounds[d] =
        AffineMap::get(footprint.symbols.size(), /*symbolCount=*/0, lowerExpr);
    footprint.upperBounds[d] = AffineMap::get(
        footprint.symbols.size(), /*symbolCount=*/0, lowerExpr + extent);

    AffineExpr globalIndex =
        getAffineDimExpr(footprint.symbols.size() + d, ctx);
    remapExprs.push_back(simplifyAffineExpr(
        globalIndex - lowerExpr, footprint.symbols.size() + bufferRank,
        /*numSymbols=*/0));
  }
  footprint.localIndexRemap =
      AffineMap::get(footprint.symbols.size() + bufferRank, /*symbolCount=*/0,
                     remapExprs, ctx);
  return footprint;
}

static bool affineExprUsesDimPosition(AffineExpr expr, unsigned dimPos) {
  bool used = false;
  expr.walk([&](AffineExpr inner) {
    if (auto dim = dyn_cast<AffineDimExpr>(inner);
        dim && dim.getPosition() == dimPos)
      used = true;
  });
  return used;
}

static LogicalResult
checkBufferAtFootprintSeparability(const BufferAtFootprint &footprint,
                                   affine::AffineForOp axisLoop) {
  // A legal buffer_at needs the selected axis to separate per-instance regions.
  // We approximate this by checking whether the axis moves some footprint bound
  // far enough that adjacent iterations do not overlap on that dimension.
  Value axisIV = axisLoop.getInductionVar();
  auto it = llvm::find(footprint.symbols, axisIV);
  if (it == footprint.symbols.end()) {
    InFlightDiagnostic diag = axisLoop.emitError();
    diag << "cannot buffer_at on this axis because the target buffer cannot "
            "be made private to each iteration";
    diag.attachNote(axisLoop.getLoc())
        << "the target-buffer access pattern does not depend on the selected "
           "axis, so every iteration would use the same region";
    return failure();
  }

  unsigned axisPos = std::distance(footprint.symbols.begin(), it);
  uint64_t axisStep = axisLoop.getStepAsInt();
  bool foundAxisSensitiveDim = false;
  for (auto [shape, lbMap] :
       llvm::zip_equal(footprint.shape, footprint.lowerBounds)) {
    AffineExpr lowerExpr = lbMap.getResult(0);
    if (!affineExprUsesDimPosition(lowerExpr, axisPos))
      continue;

    FailureOr<int64_t> axisCoeff = getLinearAffineDimCoefficient(
        lowerExpr, footprint.symbols.size(), axisPos);
    if (failed(axisCoeff) || *axisCoeff == 0)
      continue;

    foundAxisSensitiveDim = true;
    uint64_t separatingStride =
        static_cast<uint64_t>(std::abs(*axisCoeff)) * axisStep;
    if (separatingStride >= static_cast<uint64_t>(shape))
      return success();
  }

  if (!foundAxisSensitiveDim) {
    InFlightDiagnostic diag = axisLoop.emitError();
    diag << "cannot buffer_at on this axis because the target buffer cannot "
            "be made private to each iteration";
    diag.attachNote(axisLoop.getLoc())
        << "the target-buffer access pattern does not depend on the selected "
           "axis, so every iteration would use the same region";
    return failure();
  }
  InFlightDiagnostic diag = axisLoop.emitError();
  diag << "cannot buffer_at on this axis because the target buffer cannot be "
          "made private to each iteration";
  diag.attachNote(axisLoop.getLoc())
      << "different iterations of the selected axis access overlapping "
         "regions of the target buffer";
  return failure();
}

static void generateBufferAtCopy(OpBuilder &builder, Location loc,
                                 Value globalBuffer, Value localBuffer,
                                 const BufferAtFootprint &footprint,
                                 bool isCopyOut) {
  // Materialize copy-in/copy-out from the derived footprint maps instead of
  // cloning original accesses. The generated loops enumerate the global region
  // and compute local indices by subtracting each dimension's lower bound.
  unsigned rank = cast<MemRefType>(globalBuffer.getType()).getRank();
  if (rank == 0) {
    if (!isCopyOut) {
      Value globalLoad = affine::AffineLoadOp::create(
          builder, loc, globalBuffer, ValueRange{});
      affine::AffineStoreOp::create(builder, loc, globalLoad, localBuffer,
                                    ValueRange{});
    } else {
      Value localLoad =
          affine::AffineLoadOp::create(builder, loc, localBuffer, ValueRange{});
      affine::AffineStoreOp::create(builder, loc, localLoad, globalBuffer,
                                    ValueRange{});
    }
    return;
  }

  SmallVector<Value, 4> globalIndices;
  SmallVector<AffineExpr, 4> localExprs;
  SmallVector<Value, 8> localOperands;
  SmallVector<affine::AffineApplyOp, 4> maybeDeadApplys;
  globalIndices.reserve(rank);
  localExprs.reserve(rank);
  localOperands.reserve(2 * rank);

  for (unsigned d = 0; d < rank; ++d) {
    auto forOp = affine::createCanonicalizedAffineForOp(
        builder, loc, footprint.symbols, footprint.lowerBounds[d],
        footprint.symbols, footprint.upperBounds[d], /*step=*/1);
    builder = OpBuilder::atBlockTerminator(forOp.getBody());

    auto offset = affine::AffineApplyOp::create(
        builder, loc, footprint.lowerBounds[d], footprint.symbols);
    maybeDeadApplys.push_back(offset);
    localOperands.push_back(offset);
    localOperands.push_back(forOp.getInductionVar());
    localExprs.push_back(builder.getAffineDimExpr(2 * d + 1) -
                         builder.getAffineDimExpr(2 * d));
    globalIndices.push_back(forOp.getInductionVar());
  }

  AffineMap localMap = AffineMap::get(2 * rank, /*symbolCount=*/0, localExprs,
                                      builder.getContext());
  affine::fullyComposeAffineMapAndOperands(&localMap, &localOperands);
  localMap = simplifyAffineMap(localMap);
  affine::canonicalizeMapAndOperands(&localMap, &localOperands);
  for (affine::AffineApplyOp applyOp : maybeDeadApplys)
    if (applyOp.use_empty())
      applyOp.erase();

  if (!isCopyOut) {
    Value globalLoad =
        affine::AffineLoadOp::create(builder, loc, globalBuffer, globalIndices);
    affine::AffineStoreOp::create(builder, loc, globalLoad, localBuffer,
                                  localMap, localOperands);
    return;
  }

  Value localLoad = affine::AffineLoadOp::create(builder, loc, localBuffer,
                                                 localMap, localOperands);
  affine::AffineStoreOp::create(builder, loc, localLoad, globalBuffer,
                                globalIndices);
}

DiagnosedSilenceableFailure
transform::BufferAtOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  // Precondition checks: one memref target, one affine axis, and the target
  // buffer must outlive the chosen loop instance so a local copy makes sense.
  auto buffers = llvm::to_vector(state.getPayloadValues(getTarget()));
  if (buffers.size() != 1) {
    return emitSilenceableError()
           << "expected target handle to resolve to exactly one payload value";
  }
  auto buffer = buffers.front();
  auto bufferType = dyn_cast<MemRefType>(buffer.getType());
  if (!bufferType) {
    return emitSilenceableError()
           << "expected target to resolve to a memref value";
  }
  auto loops = llvm::to_vector(state.getPayloadOps(getAxis()));
  if (loops.size() != 1) {
    return emitSilenceableError() << "expected axis handle to resolve to "
                                     "exactly one payload operation";
  }
  auto axisLoop = dyn_cast<affine::AffineForOp>(loops.front());
  if (!axisLoop) {
    return emitSilenceableError()
           << "expected axis to resolve to a affine.for loop";
  }
  // buffer should be defined outside the axis loop
  Operation *bufferDef = buffer.getDefiningOp();
  if (bufferDef && axisLoop->isAncestor(bufferDef)) {
    return emitSilenceableError() << "expected target buffer to be defined "
                                     "outside the selected axis loop";
  }

  // get the root loop of the selected axis
  affine::AffineForOp rootLoop = getRootLoop(axisLoop);

  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, rootLoop);
  if (band.empty())
    return emitSilenceableError()
           << "cannot find contiguous nested loops for buffer_at";

  auto axisIt = llvm::find(band, axisLoop);
  if (axisIt == band.end())
    return emitSilenceableError()
           << "selected axis is not in a contiguous loop band";
  unsigned axisIdx = std::distance(band.begin(), axisIt);

  if (axisIdx == band.size() - 1) {
    return emitSilenceableError() << "cannot buffer at innermost loop axis";
  }

  // First collect the direct accesses we will rewrite. This also filters out
  // unsupported aliasing cases early, before we spend work inferring a
  // footprint that we would not be able to remap safely.
  SmallVector<Operation *, 8> localAccessOps;
  bool hasLoads = false;
  bool hasStores = false;
  if (failed(collectBufferAtAccesses(axisLoop, buffer, localAccessOps, hasLoads,
                                     hasStores)))
    return emitSilenceableError() << "buffer_at failed";
  if (localAccessOps.empty()) {
    return emitSilenceableError()
           << "no load/store of the target buffer found within the selected "
              "axis loop";
  }

  SmallVector<affine::AffineForOp, 4> innerLoops(axisIt + 1, band.end());
  // Then synthesize a single per-instance footprint and verify that different
  // axis iterations stay separate enough to privatize. These two checks encode
  // the main legality test for buffer_at.
  FailureOr<BufferAtFootprint> footprintOr = analyzeBufferAtFootprint(
      localAccessOps, innerLoops, bufferType.getRank(), axisLoop.getContext());
  if (failed(footprintOr)) {
    return emitSilenceableError()
           << "buffer_at requires a bounded, realizable per-instance affine "
              "footprint for the target buffer";
  }
  BufferAtFootprint footprint = std::move(*footprintOr);
  if (failed(checkBufferAtFootprintSeparability(footprint, axisLoop)))
    return emitSilenceableError() << "buffer_at failed";

  // Finally allocate the local buffer, emit copy-in/copy-out around the axis
  // body, and rewrite each collected access through the synthesized local index
  // remap.
  rewriter.setInsertionPointToStart(axisLoop.getBody());
  Location loc = buffer.getLoc();
  auto localBuffer = memref::AllocOp::create(
      rewriter, loc,
      MemRefType::get(footprint.shape, bufferType.getElementType()));

  if (hasLoads)
    generateBufferAtCopy(rewriter, loc, buffer, localBuffer, footprint,
                         /*isCopyOut=*/false);

  if (hasStores) {
    rewriter.setInsertionPoint(axisLoop.getBody()->getTerminator());
    generateBufferAtCopy(rewriter, loc, buffer, localBuffer, footprint,
                         /*isCopyOut=*/true);
  }

  for (Operation *accessOp : localAccessOps) {
    if (failed(affine::replaceAllMemRefUsesWith(
            buffer, localBuffer, accessOp,
            /*extraIndices=*/{}, footprint.localIndexRemap,
            /*extraOperands=*/footprint.symbols,
            /*symbolOperands=*/{}))) {
      return emitSilenceableError()
             << "buffer_at failed to remap accesses into the local buffer";
    }
  }

  // Give the local alloc a stable identifier derived from the source buffer
  // value so front-end value proxies can be reconstructed after refresh.
  std::string sourceIdentifier = getBufferAtSourceIdentifier(buffer);
  if (!sourceIdentifier.empty()) {
    localBuffer->setAttr("allo.value_id",
                         rewriter.getStringAttr(sourceIdentifier + "::local"));
    localBuffer->setAttr(OpIdentifier,
                         rewriter.getStringAttr(sourceIdentifier + "::local"));
  }
  results.setValues(cast<OpResult>(getResult()), {localBuffer});

  return DiagnosedSilenceableFailure::success();
}
