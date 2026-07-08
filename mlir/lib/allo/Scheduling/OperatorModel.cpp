/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorModel.h"

#include "allo/IR/AlloOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

#include <string>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::allo;
using namespace circt::scheduling;

namespace {
using HandleFn = llvm::function_ref<WalkResult(Operation *)>;

// Set up the minimal operator library on `problem`, then drive `walkFn` with a
// per-op classifier. Ultimately the library should come from a dialect
// interface. Unclassified ops default to zero-latency combinational.
LogicalResult populateImpl(SharedOperatorsProblem &problem,
                           llvm::function_ref<void(HandleFn)> walkFn) {
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);
  (void)seqOpr;

  // Assign a limited operator+resource keyed on a memory handle.
  auto setLimitedResource = [&](Operation *op, Value handle, StringRef prefix) {
    auto key = (prefix + std::to_string(hash_value(handle))).str();
    Problem::OperatorType opr = problem.getOrInsertOperatorType(key);
    problem.setLatency(opr, 1);
    problem.setLinkedOperatorType(op, opr);

    auto rsrc = problem.getOrInsertResourceType(key + "_rsrc");
    problem.setLimit(rsrc, 1);
    problem.setLinkedResourceTypes(op,
                                   SmallVector<Problem::ResourceType>{rsrc});
  };

  auto handle = [&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp, SubIOp, IfOp, AffineYieldOp, arith::ConstantOp, CmpIOp,
              IndexCastOp, ExtSIOp, ExtUIOp, TruncIOp, memref::AllocaOp,
              scf::YieldOp>([&](Operation *combOp) {
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *memOp) {
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<memref::StoreOp>(*memOp).getMemRef();
          setLimitedResource(memOp, memRef, "mem_");
          return WalkResult::advance();
        })
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *memOp) {
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<memref::LoadOp>(*memOp).getMemRef();
          setLimitedResource(memOp, memRef, "mem_");
          return WalkResult::advance();
        })
        .Case<StreamGetOp, StreamPutOp>([&](Operation *streamOp) {
          // A stream access takes one cycle. Same-FIFO accesses are already
          // serialized by the dependence recurrence built in the analysis, so
          // no shared port resource is modeled here (which would wrongly
          // serialize accesses to distinct FIFOs of the same stream array).
          Problem::OperatorType streamOpr =
              problem.getOrInsertOperatorType("stream");
          problem.setLatency(streamOpr, 1);
          problem.setLinkedOperatorType(streamOp, streamOpr);
          return WalkResult::advance();
        })
        .Case<MulIOp, DivSIOp, DivUIOp, RemSIOp, RemUIOp, AddFOp, SubFOp,
              MulFOp, DivFOp>([&](Operation *mcOp) {
          // Known multi-cycle ops (integer mul/div/rem, floating point).
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *other) {
          // Conservative default: treat unclassified ops as zero-latency
          // combinational so scheduling never aborts. S4 refines this with
          // multi-cycle costs and a dialect interface.
          problem.setLinkedOperatorType(other, combOpr);
          return WalkResult::advance();
        });
  };

  walkFn(handle);
  return success();
}
} // namespace

namespace mlir::allo {

LogicalResult populateOperatorTypes(Block &body,
                                    SharedOperatorsProblem &problem) {
  return populateImpl(problem, [&](HandleFn h) { body.walk(h); });
}

LogicalResult populateOperatorTypes(ArrayRef<Operation *> ops,
                                    SharedOperatorsProblem &problem) {
  return populateImpl(problem, [&](HandleFn h) {
    for (Operation *top : ops)
      top->walk(h);
  });
}

} // namespace mlir::allo
