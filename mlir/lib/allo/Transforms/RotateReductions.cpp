/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_ROTATEREDUCTIONSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// A simple reduction: exactly one iter_arg `acc`, and the yielded value is a
// reduction step combining `acc` with a leaf, with `acc` used only there.
struct Reduction {
  affine::AffineForOp loop;
  ReductionStep step; // the recurrence operator (float bare / integer idiom)
};

std::optional<Reduction> matchReduction(affine::AffineForOp loop) {
  if (loop.getNumRegionIterArgs() != 1)
    return std::nullopt;
  Value acc = loop.getRegionIterArgs()[0];
  auto yield = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
  Value yielded = yield.getOperand(0);
  ReductionStep step = matchReductionStep(yielded);
  if (!step || !yielded.hasOneUse() || !acc.hasOneUse())
    return std::nullopt;
  // `acc` must be exactly one of the combined operands (the other is the leaf).
  auto [lhs, rhs] = reductionOperands(step);
  if ((lhs == acc) == (rhs == acc))
    return std::nullopt;
  return Reduction{loop, step};
}

// The identity element of `step`'s operator (0 for add, 1 for mul) as a
// constant of the accumulator's (narrow) type.
Value identityFor(OpBuilder &b, Location loc, const ReductionStep &step) {
  Type ty = step.type();
  if (ty.isIntOrIndex())
    return arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(ty, step.isMul() ? 1 : 0));
  return arith::ConstantOp::create(
      b, loc, b.getFloatAttr(ty, step.isMul() ? 1.0 : 0.0));
}

void rotate(Reduction red, unsigned n) {
  affine::AffineForOp loop = red.loop;
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  Value oldIv = loop.getInductionVar();
  Value oldAcc = loop.getRegionIterArgs()[0];
  const ReductionStep &step = red.step;

  // N accumulators: slot 0 keeps the original init, the rest are the identity.
  Value identity = identityFor(b, loc, step);
  SmallVector<Value> inits{loop.getInits()[0]};
  inits.append(n - 1, identity);

  auto newLoop = affine::AffineForOp::create(
      b, loc, loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), inits,
      [&](OpBuilder &nb, Location nloc, Value niv, ValueRange slots) {
        IRMapping map;
        map.map(oldIv, niv);
        map.map(oldAcc, slots.back()); // the operator reads the last slot
        for (Operation &o : loop.getBody()->without_terminator())
          nb.clone(o, map);
        // Rotate: the new value enters slot 0 and the others shift down.
        SmallVector<Value> yields{map.lookup(step.result())};
        for (Value slot : slots.drop_back())
          yields.push_back(slot);
        affine::AffineYieldOp::create(nb, nloc, yields);
      });

  b.setInsertionPointAfter(newLoop);
  Value total = buildBalancedTree(b, step, newLoop.getResults());
  loop.getResult(0).replaceAllUsesWith(total);
  info(Stage::Prep, newLoop)
      << "Rotating reduction across " << n << " accumulators";
  loop.erase();
}

struct RotateReductionsPass
    : public allo::impl::RotateReductionsPassBase<RotateReductionsPass> {
  using RotateReductionsPassBase::RotateReductionsPassBase;

  void runOnOperation() override {
    if (accumulators < 2)
      return;
    SmallVector<Reduction> targets;
    getOperation().walk([&](affine::AffineForOp loop) {
      std::optional<Reduction> red = matchReduction(loop);
      if (!red)
        return;
      // Skip loops too short to fill the rotated pipeline.
      std::optional<uint64_t> trip = affine::getConstantTripCount(loop);
      if (trip && *trip < accumulators) {
        warn(Stage::Prep, loop)
            << "reduction not rotated because its trip count " << *trip
            << " is below the requested " << accumulators << " accumulators";
        return;
      }
      targets.push_back(*red);
    });
    for (Reduction red : targets)
      rotate(red, accumulators);
  }
};

} // namespace
