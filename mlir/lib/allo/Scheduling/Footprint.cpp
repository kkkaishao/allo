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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::allo;

void mlir::allo::summarizeOp(Operation *op, Summary &s) {
  // A recognized load/store/stream access (root resolved through views): a
  // stream touches a FIFO root, while an array access records its direction
  // and whether it is affine (polyhedral disjointness applies) or non-affine
  // (which defeats sub-range refinement).
  if (auto a = asMemAccess(op)) {
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
  // Any op not provably side-effect-free (an opaque call, memref.copy, an
  // unregistered op) may touch memory: conservatively read+write every
  // memref operand root and mark every stream operand; pure ops (arith/math,
  // constants) are skipped.
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
        return false; // different memref (e.g. subview): cannot prove
                      // disjoint
      // The two accesses may touch the same element if a dependence exists
      // at ANY depth: carried by a common enclosing loop (1..n), or
      // loop-independent (n+1, same iteration of every common loop); checking
      // only depth 1 would miss same-iteration conflicts.
      unsigned n = affine::getNumCommonSurroundingLoops(*a, *b);
      for (unsigned d = 1; d <= n + 1; ++d) {
        affine::FlatAffineValueConstraints cst;
        auto r = affine::checkMemrefAccessDependence(
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

//===----------------------------------------------------------------------===//
// Interprocedural (per-argument) call footprint.
//===----------------------------------------------------------------------===//

// Fold the memory effects `fn` has on its PARAMETERS into `s`, keyed by the
// caller-side root each parameter is bound to (`actuals`, one per parameter;
// null for a parameter the caller cannot observe). A nested call recurses with
// the binding composed, so a container's footprint is its leaves'. `active`
// guards a call cycle (the scheduler's call graph is a DAG, but this analysis
// does not rely on that). Returns false when a construct defeats the summary.
static bool summarizeFuncInto(func::FuncOp fn, ArrayRef<Value> actuals,
                              Summary &s,
                              llvm::SmallPtrSetImpl<Operation *> &active) {
  if (!active.insert(fn).second)
    return false; // a call cycle has no finite footprint
  llvm::scope_exit pop([&] { active.erase(fn); });

  // The caller-side root a callee value denotes: a parameter maps to the
  // actual it is bound to; anything else (a callee-local `alloc`) is memory
  // the caller cannot observe and constrains nothing at the call site.
  auto mapRoot = [&](Value v) -> Value {
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner()->getParentOp() != fn)
      return Value();
    return actuals[arg.getArgNumber()];
  };

  bool ok = true;
  fn.walk([&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal()) {
        ok = false;
        return;
      }
      SmallVector<Value> sub;
      for (Value o : call.getArgOperands()) {
        // A view operand rebases the callee's index space away from the root's,
        // which the region comparison cannot express.
        if (isa<MemRefType>(o.getType()) && resolveRoot(o) != o) {
          ok = false;
          return;
        }
        sub.push_back(mapRoot(o));
      }
      ok &= summarizeFuncInto(callee, sub, s, active);
      return;
    }
    if (auto a = asMemAccess(op)) {
      Value root = mapRoot(a->root);
      if (!root)
        return; // callee-local memory
      if (a->kind == AccessKind::Stream) {
        s.streams.insert(root);
        return;
      }
      Access &acc = s.mem[root];
      (a->isWrite ? acc.writes : acc.reads) = true;
      // Only an affine access naming the parameter DIRECTLY has indices in
      // the array's own index space (the space the caller shares with it),
      // so only then is its region comparable across the call.
      if (a->map && affine::MemRefAccess(op).memref == a->root)
        acc.affine.push_back(op);
      else
        acc.nonAffine = true;
      return;
    }
    // Any other op that may touch memory: conservative on each mapped operand
    // (as `summarizeOp`, but in the caller's terms).
    if (isMemoryEffectFree(op))
      return;
    for (Value operand : op->getOperands()) {
      Type t = operand.getType();
      if (isa<MemRefType>(t)) {
        if (Value root = mapRoot(resolveRoot(operand))) {
          Access &a = s.mem[root];
          a.reads = a.writes = a.nonAffine = true;
        }
      } else if (isa<allo::StreamType>(t)) {
        if (Value root = mapRoot(resolveRoot(operand)))
          s.streams.insert(root);
      }
    }
  });
  return ok;
}

bool mlir::allo::summarizeCall(func::CallOp call, Summary &s) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!callee || callee.isExternal())
    return false;
  SmallVector<Value> actuals;
  for (Value o : call.getArgOperands()) {
    // Bail on a view operand rather than key by its root: the callee indexes
    // the view, whose index space is offset from the root's. Every other
    // operand is its own root, matching `summarizeOp`'s `resolveRoot`-keyed
    // accesses.
    if (isa<MemRefType>(o.getType()) && resolveRoot(o) != o)
      return false;
    actuals.push_back(o);
  }
  llvm::SmallPtrSet<Operation *, 8> active;
  return summarizeFuncInto(callee, actuals, s, active);
}

// Whether two accesses to one array provably touch DISJOINT elements when
// they may live in DIFFERENT functions: each naming its own parameter, so
// `footprintsDisjoint`'s `MemRefAccess` pair test (which demands one
// identical memref Value) does not apply. Compare polyhedral REGIONS
// instead: `MemRefRegion` at loop depth 0 is the set of indices an access
// touches across all its enclosing loops, with exactly `rank` dimensions and
// the IVs projected out. A `func.call` is type-checked, so parameters bound
// to one array share its shape: the two regions then align positionally and
// intersect meaningfully. Symbols keep their Value identity across the
// merge, so two callees' unrelated symbols stay distinct columns and the
// intersection stays non-empty: unproven disjointness conservatively reads
// as a conflict.
static bool regionsDisjoint(const Access &ai, const Access &aj) {
  if (ai.nonAffine || aj.nonAffine)
    return false;
  for (Operation *a : ai.affine) {
    for (Operation *b : aj.affine) {
      if (!isa<affine::AffineWriteOpInterface>(a) &&
          !isa<affine::AffineWriteOpInterface>(b))
        continue; // read-read pairs never conflict
      affine::MemRefRegion ra(a->getLoc()), rb(b->getLoc());
      if (failed(ra.compute(a, /*loopDepth=*/0)) ||
          failed(rb.compute(b, /*loopDepth=*/0)))
        return false; // e.g. a non-affine bound: no region to compare
      assert(ra.getRank() == rb.getRank() &&
             "accesses to one array must share its rank");
      auto ca = *ra.getConstraints();
      auto cb = *rb.getConstraints();
      ca.mergeSymbolVars(cb); // union of symbols, aligned by Value
      ca.append(cb);          // intersect the two index sets
      if (!ca.isEmpty())
        return false; // may touch a common element
    }
  }
  return true;
}

Conflict mlir::allo::callFootprintConflict(const Access &a, const Access &b) {
  bool wa = a.writes, wb = b.writes;
  bool ta = a.reads || wa, tb = b.reads || wb;
  if (!((wa && tb) || (ta && wb)))
    return Conflict::None; // both read-only: no ordering constraint
  if (regionsDisjoint(a, b))
    return Conflict::None; // provably disjoint elements
  return (wa && wb) ? Conflict::WAW : wa ? Conflict::RAW : Conflict::WAR;
}
