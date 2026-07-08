/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/Footprint.h"

#include "allo/IR/AlloTypes.h"
#include "allo/Scheduling/MemoryAccess.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::allo;

void mlir::allo::summarizeOp(Operation *op, Summary &s) {
  // A recognized load/store/stream access (root resolved through views). A
  // stream touches a FIFO root; an array access records its direction, and
  // whether it is affine (polyhedral disjointness applies) or a non-affine
  // memref.* (which defeats the sub-range refinement).
  if (std::optional<MemAccess> a = asMemAccess(op)) {
    if (a->kind == AccessKind::Stream) {
      s.streams.insert(a->root);
      return;
    }
    Access &acc = s.mem[a->root];
    (a->isWrite ? acc.writes : acc.reads) = true;
    if (a->map)
      acc.affine.push_back(op);
    else
      acc.nonAffine = true;
    return;
  }
  // Any other op that is not provably side-effect-free -- an opaque call, a
  // `memref.copy`, an unregistered op -- may touch memory: conservatively
  // read+write every memref operand root and mark every stream operand. A
  // container with recursive effects (scf.if / a loop) adds nothing here (it
  // has no memref operands); its body accesses are summarized by the walk. Pure
  // ops (arith/math, constants) are skipped.
  if (isMemoryEffectFree(op))
    return;
  for (Value operand : op->getOperands()) {
    Type t = operand.getType();
    if (isa<MemRefType>(t)) {
      Access &a = s.mem[resolveRoot(operand)];
      a.reads = a.writes = a.nonAffine = true;
    } else if (isa<allo::StreamType>(t)) {
      s.streams.insert(resolveRoot(operand));
    }
  }
}

bool mlir::allo::footprintsDisjoint(const Access &ai, const Access &aj) {
  if (ai.nonAffine || aj.nonAffine)
    return false;
  for (Operation *a : ai.affine) {
    for (Operation *b : aj.affine) {
      if (!isa<affine::AffineWriteOpInterface>(a) &&
          !isa<affine::AffineWriteOpInterface>(b))
        continue; // read-read pairs never conflict
      affine::MemRefAccess accA(a), accB(b);
      if (accA.memref != accB.memref)
        return false; // different memref (e.g. subview) -- cannot prove
                      // disjoint
      // The two accesses may touch the same element if a dependence exists at
      // ANY depth: carried by a common enclosing loop (1..n), or loop-
      // independent (n+1, the same iteration of every common loop). Checking
      // only depth 1 misses a same-iteration conflict between accesses that
      // share enclosing loops (numCommonLoops >= 1); for accesses with no
      // common loop (sibling regions) it reduces to the single loop-independent
      // check.
      unsigned n = affine::getNumCommonSurroundingLoops(*a, *b);
      for (unsigned d = 1; d <= n + 1; ++d) {
        affine::FlatAffineValueConstraints cst;
        affine::DependenceResult r = affine::checkMemrefAccessDependence(
            accA, accB, d, &cst, /*dependenceComponents=*/nullptr,
            /*allowRAR=*/true);
        if (r.value != affine::DependenceResult::NoDependence)
          return false; // may access the same element
      }
    }
  }
  return true;
}

Conflict mlir::allo::footprintConflict(const Access &a, const Access &b) {
  bool wa = a.writes, wb = b.writes;
  bool ta = a.reads || wa, tb = b.reads || wb;
  if (!((wa && tb) || (ta && wb)))
    return Conflict::None; // both read-only: no ordering constraint
  if (footprintsDisjoint(a, b))
    return Conflict::None; // provably disjoint elements
  return (wa && wb) ? Conflict::WAW : wa ? Conflict::RAW : Conflict::WAR;
}
