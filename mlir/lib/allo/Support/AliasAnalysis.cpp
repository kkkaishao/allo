/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/AliasAnalysis.h"

#include "allo/IR/AlloOps.h" // the dcp region ops a buffer is forwarded through

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cassert>

using namespace mlir;
using namespace mlir::allo;

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

AliasResult mlir::allo::DistinctRootAliasAnalysis::alias(Value lhs, Value rhs) {
  if (!isa<BaseMemRefType>(lhs.getType()) ||
      !isa<BaseMemRefType>(rhs.getType()))
    return AliasResult::MayAlias;
  // Equal roots answer MayAlias rather than MustAlias: two views of one buffer
  // need not overlap, and the local analysis is better placed to say which.
  return resolveRoot(lhs) == resolveRoot(rhs) ? AliasResult::MayAlias
                                              : AliasResult::NoAlias;
}

AliasAnalysis mlir::allo::alloAliasAnalysis(Operation *scope) {
  AliasAnalysis aa(scope);
  aa.addAnalysisImplementation(DistinctRootAliasAnalysis{});
  return aa;
}
