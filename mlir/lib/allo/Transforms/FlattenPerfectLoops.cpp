/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (array subscripts)
#include "allo/Scheduling/MemoryModel.h"  // partitionOf (cyclic axes)
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_FLATTENPERFECTLOOPSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A loop may be coalesced iff it is normalized (lower bound 0, step 1) with a
// constant trip count -- the form that keeps the coalesced body pure-affine --
// and carries no iter_args: `coalesceLoops` merges the induction spaces but
// drops loop-carried values, so coalescing a nest with an inner accumulator
// silently rewrites it to the init constant (wrong code). Leave such nests be.
bool isFlattenable(affine::AffineForOp loop) {
  return loop.getInits().empty() && loop.hasConstantLowerBound() &&
         loop.getConstantLowerBound() == 0 && loop.getStepAsInt() == 1 &&
         affine::getConstantTripCount(loop).has_value();
}

// Whether the affine map's \p dim-th result (a subscript) is a function of the
// value \p v -- i.e. \p v feeds that subscript, over \p operands (dims then
// symbols, as an affine access carries them).
bool resultUsesValue(AffineMap map, ValueRange operands, unsigned dim,
                     Value v) {
  bool used = false;
  map.getResult(dim).walk([&](AffineExpr e) {
    unsigned pos;
    if (auto d = dyn_cast<AffineDimExpr>(e))
      pos = d.getPosition();
    else if (auto s = dyn_cast<AffineSymbolExpr>(e))
      pos = map.getNumDims() + s.getPosition();
    else
      return;
    if (pos < operands.size() && operands[pos] == v)
      used = true;
  });
  return used;
}

// Whether \p loop's induction variable indexes a *cyclic-partitioned* dimension
// of any array accessed in its body. Coalescing such a loop delinearizes the IV
// (`iv floordiv N`), which turns the partition-dim subscript into a
// non-iteration-invariant expression -- defeating `MemoryModel::staticBank`, so
// the array falls back to a runtime crossbar instead of a static per-bank
// split. Leave the loop uncoalesced so banking stays static.
bool carriesPartition(affine::AffineForOp loop) {
  Value iv = loop.getInductionVar();
  bool carries = false;
  loop.walk([&](Operation *op) {
    std::optional<MemAccess> a = asMemAccess(op);
    if (!a || a->kind != AccessKind::Array || !a->map)
      return;
    for (auto [dim, factor] : partitionOf(a->root).cyclicAxes)
      if (dim < a->map.getNumResults() &&
          resultUsesValue(a->map, a->indices, dim, iv))
        carries = true;
  });
  return carries;
}

// A loop is the root of a perfect nest unless its parent is an affine.for that
// perfectly nests it (parent body is exactly {loop, terminator}).
bool isPerfectNestRoot(affine::AffineForOp loop) {
  auto parent = dyn_cast<affine::AffineForOp>(loop->getParentOp());
  if (!parent)
    return true;
  Block &body = parent.getRegion().front();
  bool perfect = &body.front() == loop.getOperation() &&
                 loop->getNextNode() == body.getTerminator();
  return !perfect;
}

struct FlattenPerfectLoopsPass
    : public allo::impl::FlattenPerfectLoopsPassBase<FlattenPerfectLoopsPass> {
  void runOnOperation() override {
    // Collect the perfect-nest roots first: coalescing a root rewrites its own
    // band (and only its band, which holds no other root), so a pre-collected
    // list stays valid across the rewrites.
    SmallVector<affine::AffineForOp> roots;
    getOperation().walk([&](affine::AffineForOp loop) {
      if (isPerfectNestRoot(loop))
        roots.push_back(loop);
    });

    for (affine::AffineForOp root : roots) {
      SmallVector<affine::AffineForOp> nest;
      affine::getPerfectlyNestedLoops(nest, root);
      unsigned n = 0;
      // Stop the coalescable band at a partition-carrying loop, so its IV stays
      // a bare index and the array banks statically (banking-design.md).
      while (n < nest.size() && isFlattenable(nest[n]) &&
             !carriesPartition(nest[n]))
        ++n;
      if (n >= 2) {
        auto loops = MutableArrayRef<affine::AffineForOp>(nest).take_front(n);
        (void)affine::coalesceLoops(loops);
        info(Stage::Prep, loops.back())
            << "Flattening perfect nest of " << n << " loops";
      }
    }
  }
};

} // namespace
