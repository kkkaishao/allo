/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Memory + stream dependence analysis. Mirrors CIRCT's MemoryDependenceAnalysis
// for affine memref accesses and additionally understands Allo stream get/put
// operations (see checkStreamDependence). Lifted verbatim from the old
// convert-loop-to-schedule pass.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/DependenceAnalysis.h"

#include "allo/IR/AlloOps.h"
#include "allo/Support/AffineValueMapBuilder.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// Memref dependences
//===----------------------------------------------------------------------===//

static void checkMemrefDependence(SmallVectorImpl<Operation *> &memoryOps,
                                  unsigned depth,
                                  MemoryDependenceResult &results) {
  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      // Look for inter-iteration dependences on the same memory location.
      affine::MemRefAccess src(source);
      affine::MemRefAccess dst(destination);
      affine::FlatAffineValueConstraints dependenceConstraints;
      SmallVector<affine::DependenceComponent, 2> depComps;

      // Requested depth might not be a valid comparison if they do not belong
      // to the same loop nest
      if (depth > affine::getInnermostCommonLoopDepth({source, destination}))
        continue;

      auto result = affine::checkMemrefAccessDependence(
          src, dst, depth, &dependenceConstraints, &depComps, true);

      results[destination].emplace_back(source, result.value, depComps);

      // Also consider intra-iteration dependences on the same memory location.
      // This currently does not consider aliasing.
      if (src != dst)
        continue;

      // Collect surrounding loops to use in dependence components. Only proceed
      // if we are in the innermost loop.
      SmallVector<affine::AffineForOp> enclosingLoops;
      affine::getAffineForIVs(*destination, &enclosingLoops);
      if (enclosingLoops.size() != depth)
        continue;

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      affine::getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      affine::getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        continue;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          // Build dependence components for each loop depth.
          SmallVector<affine::DependenceComponent> intraDeps;
          for (size_t i = 0; i < depth; ++i) {
            affine::DependenceComponent depComp;
            depComp.op = enclosingLoops[i];
            depComp.lb = 0;
            depComp.ub = 0;
            intraDeps.push_back(depComp);
          }

          results[destination].emplace_back(
              source, affine::DependenceResult::HasDependence, intraDeps);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Stream dependences
//===----------------------------------------------------------------------===//

// Returns the base stream SSA value a stream get/put operates on. Two accesses
// on different bases are always independent (SSA identity is a precise
// disambiguation for streams, which are not reassigned through aliases).
static Value getStreamBase(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getStream();
  return cast<StreamPutOp>(op).getStream();
}

// Returns the FIFO-selecting indices of a stream get/put operation.
static OperandRange getStreamIndices(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getIndices();
  return cast<StreamPutOp>(op).getIndices();
}

// Nearest enclosing affine.for, skipping non-loop parents (e.g. affine.if).
static affine::AffineForOp getNearestAffineFor(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto forOp = dyn_cast<affine::AffineForOp>(parent))
      return forOp;
  return nullptr;
}

// Whether two same-base stream accesses may touch the same FIFO. A stream value
// is an array of FIFOs selected by its indices, so this is an affine
// disambiguation on the indices, analogous to array-subscript aliasing.
namespace {
enum class FifoAlias { Same, Distinct, Unknown };
} // namespace

static FifoAlias compareFifo(AffineValueMapBuilder &builder, Operation *a,
                             Operation *b) {
  builder.reset();
  for (Value idx : getStreamIndices(a))
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  affine::AffineValueMap ma = builder.compose();

  builder.reset();
  for (Value idx : getStreamIndices(b))
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  affine::AffineValueMap mb = builder.compose();

  if (ma.getNumResults() != mb.getNumResults())
    return FifoAlias::Unknown;

  affine::AffineValueMap diff;
  affine::AffineValueMap::difference(ma, mb, &diff);
  bool allZero = true;
  for (AffineExpr e : diff.getAffineMap().getResults()) {
    auto cst = dyn_cast<AffineConstantExpr>(e);
    if (!cst) {
      // Symbolic offset: cannot prove same or distinct FIFO.
      allZero = false;
      continue;
    }
    if (cst.getValue() != 0)
      return FifoAlias::Distinct; // some coordinate differs by a constant
  }
  return allZero ? FifoAlias::Same : FifoAlias::Unknown;
}

// Build dependence components mirroring the op's enclosing loop nest, placing
// `distance` on the innermost loop (the only component the scheduler reads).
static SmallVector<affine::DependenceComponent>
streamDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::AffineForOp> loops;
  affine::getAffineForIVs(*op, &loops);
  SmallVector<affine::DependenceComponent> comps;
  for (auto loop : loops) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  assert(!comps.empty() && "stream op must be enclosed by a loop");
  comps.back().lb = distance;
  return comps;
}

// Streams are FIFOs: every pair of accesses to the same FIFO must preserve its
// program+iteration order, regardless of direction (unlike memory, get-get is
// ordered and there is no RAW/WAR/WAW distinction). Each may-aliasing pair is
// serialized with a distance-0 intra-iteration edge plus a distance-1
// loop-carried back edge, closing the recurrence that bounds the II.
static void checkStreamDependence(SmallVectorImpl<Operation *> &streamOps,
                                  AffineValueMapBuilder &builder,
                                  MemoryDependenceResult &results) {
  for (unsigned i = 0, e = streamOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      // `earlier` precedes `later` in program order: `walk` is a pre-order
      // traversal, so a smaller index is never scheduled after a larger one.
      Operation *earlier = streamOps[i];
      Operation *later = streamOps[j];

      if (getStreamBase(earlier) != getStreamBase(later))
        continue;

      // Only serialize accesses sharing the same innermost loop, so both ends
      // of the edge land in a single scheduling problem.
      affine::AffineForOp loop = getNearestAffineFor(earlier);
      if (!loop || loop != getNearestAffineFor(later))
        continue;

      // Provably-distinct FIFOs are independent; same or unknown are ordered.
      if (compareFifo(builder, earlier, later) == FifoAlias::Distinct)
        continue;

      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  streamDepComponents(later, /*distance=*/0));
      results[earlier].emplace_back(
          later, affine::DependenceResult::HasDependence,
          streamDepComponents(earlier, /*distance=*/1));
    }
  }
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis
//===----------------------------------------------------------------------===//

namespace mlir::allo {

DependenceAnalysis::DependenceAnalysis(func::FuncOp funcOp) : func(funcOp) {
  std::vector<SmallVector<affine::AffineForOp, 2>> depthToLoops;
  affine::gatherLoops(funcOp, depthToLoops);

  SmallVector<Operation *> memoryOps;
  SmallVector<Operation *> streamOps;
  funcOp->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      memoryOps.push_back(op);
    } else if (isa<StreamGetOp, StreamPutOp>(op)) {
      streamOps.push_back(op);
    }
  });

  for (unsigned d = 0; d < depthToLoops.size(); ++d)
    checkMemrefDependence(memoryOps, d, results);

  AffineValueMapBuilder builder(funcOp.getContext());
  checkStreamDependence(streamOps, builder, results);
}

void DependenceAnalysis::replaceOp(Operation *oldOp, Operation *newOp) {
  // Move the dependence list keyed on oldOp over to newOp.
  auto it = results.find(oldOp);
  if (it != results.end()) {
    results[newOp] = std::move(it->second);
    results.erase(it);
  }

  // Redirect any dependences that originate from oldOp.
  for (auto &entry : results)
    for (auto &dep : entry.second)
      if (dep.source == oldOp)
        dep.source = newOp;
}

} // namespace mlir::allo
