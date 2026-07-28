/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryAccess.h"

#include "allo/IR/AlloOps.h" // StreamGetOp / StreamPutOp

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

using namespace mlir;
using namespace mlir::allo;

// The base stream SSA value a stream get/put operates on.
static Value streamBaseOf(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getStream();
  return cast<StreamPutOp>(op).getStream();
}

// Peels views, casts, and DCP region-forwarding to the single storage root a
// memref/stream value names. A buffer live across a region boundary cannot be
// named directly (SSA dominance), so the region threads it out through its
// terminator; the sequential/pipeline cases below follow that forwarding so
// producer and consumer key on the same root.
Value mlir::allo::resolveRoot(Value v) {
  while (true) {
    if (Operation *def = v.getDefiningOp()) {
      if (auto op = dyn_cast<memref::SubViewOp>(def)) {
        v = op.getSource();
        continue;
      }
      if (auto op = dyn_cast<memref::CastOp>(def)) {
        v = op.getSource();
        continue;
      }
      if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = op.getSource();
        continue;
      }
      if (auto op = dyn_cast<memref::ViewOp>(def)) {
        v = op.getSource();
        continue;
      }
      // Follow the terminator's forwarded operand to the same root.
      unsigned k = cast<OpResult>(v).getResultNumber();
      if (auto seq = dyn_cast<dcp::DCPathSequentialOp>(def)) {
        v = seq.getBody().front().getTerminator()->getOperand(k);
        continue;
      }
      if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(def)) {
        // Terminator-kind agnostic: `uncondition` operands for a counted loop,
        // `condition`'s carried operands for a while (whose leading `i1` would
        // otherwise shift the indexing by one).
        v = pipe.getCarriedValues()[k];
        continue;
      }
      // A guard yields from two arms, so a value crossing one has no single
      // definition to peel to. No frontend shape produces that; fail loudly
      // rather than silently splitting the buffer.
      assert(!isa<dcp::DCPathSelectOp>(def) &&
             "resolveRoot: a memref/stream yielded from a dcp.select has no "
             "single storage root");
      // Any other defining op defines a fresh, non-aliasing root. A
      // transpose/collapse_shape/expand_shape/reshape is really an aliasing
      // view; keying it as distinct would silently drop a real dependence.
      assert((!isa<memref::TransposeOp, memref::CollapseShapeOp,
                   memref::ExpandShapeOp, memref::ReshapeOp>(def)) &&
             "resolveRoot: aliasing view not peeled; the distinct-root "
             "assumption would drop a real dependence");
      return v;
    }
    // A pipeline iter-arg (block argument 0 is the counter) forwards its init.
    auto barg = dyn_cast<BlockArgument>(v);
    if (!barg)
      return v;
    auto pipe = dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
    if (!pipe || barg.getArgNumber() == 0)
      return v; // a func argument, or the counter: already a root
    v = pipe.getInits()[barg.getArgNumber() - 1];
  }
}

// Normalizes affine and non-affine (memref.load/store, stream get/put)
// accesses to one shape: a root value, an affine map, and index operands.
// Non-affine ops get the multi-dim identity map so address pricing and bank
// digit derivation downstream can treat both forms uniformly.
std::optional<MemAccess> mlir::allo::asMemAccess(Operation *op) {
  MemAccess a;
  a.op = op;
  if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
    a.root = resolveRoot(read.getMemRef());
    a.map = read.getAffineMap();
    llvm::append_range(a.indices, read.getMapOperands());
    return a;
  }
  if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
    a.isWrite = true;
    a.root = resolveRoot(write.getMemRef());
    a.map = write.getAffineMap();
    llvm::append_range(a.indices, write.getMapOperands());
    return a;
  }
  // Non-affine subscript: identity map over indices matches the affine
  // encoding.
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    a.root = resolveRoot(load.getMemRef());
    llvm::append_range(a.indices, load.getIndices());
    a.map =
        AffineMap::getMultiDimIdentityMap(a.indices.size(), op->getContext());
    return a;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    a.isWrite = true;
    a.root = resolveRoot(store.getMemRef());
    llvm::append_range(a.indices, store.getIndices());
    a.map =
        AffineMap::getMultiDimIdentityMap(a.indices.size(), op->getContext());
    return a;
  }
  if (isa<StreamGetOp, StreamPutOp>(op)) {
    a.kind = AccessKind::Stream;
    a.isWrite = isa<StreamPutOp>(op);
    a.root = resolveRoot(streamBaseOf(op));
    if (auto get = dyn_cast<StreamGetOp>(op))
      llvm::append_range(a.indices, get.getIndices());
    else
      llvm::append_range(a.indices, cast<StreamPutOp>(op).getIndices());
    return a;
  }
  return std::nullopt;
}
