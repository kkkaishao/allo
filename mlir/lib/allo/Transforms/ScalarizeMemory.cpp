/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr, kBindStorageAttr
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"

namespace mlir::allo {
#define GEN_PASS_DEF_SCALARIZEMEMORYPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// One local array as it stands before the transform. `op` is only ever compared
// (the operation it names may be erased); the location and the type are uniqued
// attributes, so they stay valid and can still be reported afterwards.
struct LocalArray {
  Operation *op;
  Location loc;
  MemRefType type;
};

struct ScalarizeMemoryPass
    : public allo::impl::ScalarizeMemoryPassBase<ScalarizeMemoryPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<LocalArray> before;
    func.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        before.push_back(
            {op, op->getLoc(), cast<MemRefType>(op->getResult(0).getType())});
    });

    affine::affineScalarReplace(func, getAnalysis<DominanceInfo>(),
                                getAnalysis<PostDominanceInfo>(),
                                getAnalysis<AliasAnalysis>());

    if (before.empty())
      return;
    DenseSet<Operation *> survived;
    func.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        survived.insert(op);
    });
    // Report the arrays that are gone, not the loads that were forwarded:
    // deleting an array is a change to the storage decision, which the pipeline
    // announces, while forwarding a load is ordinary redundancy removal.
    for (const LocalArray &array : before)
      if (!survived.count(array.op))
        log(Level::Info, Stage::Prep, array.loc)
            << "Every read of the local array " << array.type
            << " is forwarded from the store that produced it, so it is "
               "dataflow and needs no storage";

    MLIRContext *ctx = &getContext();
    auto complete = PartitionAttr::get(
        ctx, {PartitionAxisAttr::get(ctx, PartitionKindEnum::CompletePartition,
                                     /*factor=*/0,
                                     /*dim=*/0)});

    for (Operation *op : survived) {
      // An explicit storage choice always wins over an automatic one.
      if (op->hasAttr(kPartitionAttr) || op->hasAttr(kBindStorageAttr))
        continue;
      auto type = cast<MemRefType>(op->getResult(0).getType());
      if (!type.hasStaticShape() || type.getNumElements() > maxElements)
        continue;
      // Every use a direct access, so the array's whole traffic is in view.
      // This is what excludes handing it to a sub-kernel, which masters ports
      // on storage the caller owns and has never been tried against a Complete
      // partition, and threading it out of a region as a result.
      if (llvm::any_of(op->getUsers(), [](Operation *user) {
            return !isa<affine::AffineLoadOp, affine::AffineStoreOp,
                        memref::LoadOp, memref::StoreOp>(user);
          }))
        continue;

      op->setAttr(kPartitionAttr, complete);
      info(Stage::Prep, op)
          << "Complete-partitioned the local array " << type << " ("
          << type.getNumElements() << " elements, within the " << maxElements
          << "-element threshold), so it lowers to registers rather than a "
             "memory whose ports and access latency bound the II";
    }
  }
};

} // namespace
