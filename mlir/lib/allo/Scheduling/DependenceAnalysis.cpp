/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/DependenceAnalysis.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryAccess.h"
#include "allo/Support/AffineValueMapBuilder.h"
#include "allo/Support/AliasAnalysis.h" // resolveRoot (storage identity)
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// assume.ssa value facts
//
// Parse an `allo.assume.ssa` predicate into constant ranges on the SSA values
// it constrains: each comparison of one value against a constant becomes a
// bound, AND-ed predicates contribute independently, and the tightest bound
// wins.
//===----------------------------------------------------------------------===//

namespace {
// A single-variable linear fact `c*v + k (>= | ==) 0` from one comparison.
struct Assumption {
  Value v;
  int64_t c, k;
  bool isEq; // true: == 0, false: >= 0
};
} // namespace

// Parse a comparison of one SSA value against a constant into `c*v + k
// (>=|==) 0`. Returns nullopt for shapes not modeled (a `ne`, or both
// operands constant or both symbolic).
static std::optional<Assumption> parseComparison(arith::CmpIOp cmp) {
  auto cL = getConstantIntValue(cmp.getLhs());
  auto cR = getConstantIntValue(cmp.getRhs());
  if (cL.has_value() == cR.has_value())
    return std::nullopt; // need exactly one constant operand

  bool isEq = false, swap = false;
  int strict = 0;
  using P = arith::CmpIPredicate;
  switch (cmp.getPredicate()) {
  case P::sge:
  case P::uge:
    break; // L - R >= 0
  case P::sgt:
  case P::ugt:
    strict = 1;
    break; // L - R - 1 >= 0
  case P::sle:
  case P::ule:
    swap = true;
    break; // R - L >= 0
  case P::slt:
  case P::ult:
    swap = true;
    strict = 1;
    break; // R - L - 1 >= 0
  case P::eq:
    isEq = true;
    break; // L - R == 0
  default:
    return std::nullopt; // ne
  }

  // Normalize to `x - y - strict`, where exactly one of x, y is the value.
  Value x = swap ? cmp.getRhs() : cmp.getLhs();
  Value y = swap ? cmp.getLhs() : cmp.getRhs();
  if (auto cx = swap ? cR : cL)
    return Assumption{y, -1, *cx - strict,
                      isEq}; // x constant: -y + (cx - strict)
  auto cy = swap ? cL : cR;
  return Assumption{x, 1, -*cy - strict,
                    isEq}; // y constant: x + (-cy - strict)
}

// Distill the parsed facts into a per-value constant range, keeping the
// tightest bound when a value is constrained more than once.
static void buildAssumedRanges(ArrayRef<Assumption> assumptions,
                               llvm::DenseMap<Value, AssumedRange> &ranges) {
  auto tighten = [&](Value v, std::optional<int64_t> lb,
                     std::optional<int64_t> ub) {
    AssumedRange &r = ranges[v];
    if (lb)
      r.lb = r.lb ? std::max(*r.lb, *lb) : lb;
    if (ub)
      r.ub = r.ub ? std::min(*r.ub, *ub) : ub;
  };
  for (const Assumption &as : assumptions) {
    // `c*v + k (>=|==) 0`  ==>  `c*v (>=|==) -k` (c is +/-1).
    if (as.isEq) {
      if ((-as.k) % as.c == 0) // exact integer solution, else vacuous
        tighten(as.v, (-as.k) / as.c, (-as.k) / as.c);
    } else if (as.c > 0) // v >= ceil(-k / c)
      tighten(as.v, llvm::divideCeilSigned(-as.k, as.c), std::nullopt);
    else // c < 0: v <= floor(-k / c)
      tighten(as.v, std::nullopt, llvm::divideFloorSigned(-as.k, as.c));
  }
}

// Collect the facts implied by an assume.ssa predicate (an `and`-tree of
// comparisons). Unrecognized shapes are not collected.
static void collectAssumptions(Value cond, SmallVectorImpl<Assumption> &out) {
  Operation *def = cond.getDefiningOp();
  if (!def)
    return;
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    collectAssumptions(andOp.getLhs(), out);
    collectAssumptions(andOp.getRhs(), out);
  } else if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    if (auto as = parseComparison(cmp))
      out.push_back(*as);
  }
}

//===----------------------------------------------------------------------===//
// Memref dependences
//===----------------------------------------------------------------------===//

// Rewrite the components of one polyhedral result into ITERATION units, the
// unit every consumer reads them in.
static void
rescaleOnLoopStep(SmallVectorImpl<affine::DependenceComponent> &comps) {
  for (affine::DependenceComponent &c : comps) {
    int64_t step = cast<affine::AffineForOp>(c.op).getStepAsInt();
    auto rescale = [&](std::optional<int64_t> &v, int64_t open, bool isLower) {
      if (!v || *v == open) {
        v = std::nullopt;
        return;
      }
      int64_t d = isLower ? llvm::divideFloorSigned(*v, step)
                          : llvm::divideCeilSigned(*v, step);
      v = isLower && *v > 0 ? std::max<int64_t>(d, 1) : d;
    };
    rescale(c.lb, std::numeric_limits<int64_t>::min(), /*isLower=*/true);
    rescale(c.ub, std::numeric_limits<int64_t>::max(), /*isLower=*/false);
  }
}

// Records the affine memref dependences of every ordered pair of accesses.
// `checkMemrefAccessDependence` is queried at each loop depth from 1 to
// numCommonLoops (a dependence carried by the d-th common surrounding loop)
// and at numCommonLoops + 1, the loop-independent (intra-iteration) case with
// all common loops pinned to the same iteration. At that top depth,
// `allowRAR = false` orients the otherwise-symmetric dist-0 dependence by
// program order (reported only when the source ancestor precedes the
// destination ancestor in their common block) and drops read-read pairs,
// which never conflict. This also catches same-iteration conflicts between
// different subscripts that can alias (e.g. `A[i][j]` vs `A[j][i]` on the
// diagonal, or `A[2*i]` vs `A[i]` at i == 0). Aliasing between distinct
// memrefs is not modeled; distinct SSA memrefs are assumed disjoint.
//
// A pair with either endpoint the test cannot model (`nonPolyhedral`) is
// skipped entirely and left to the conservative path, so each pair is owned
// by exactly one analysis.
static void
checkMemrefDependence(ArrayRef<Operation *> memoryOps,
                      const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
                      MemoryDependenceResult &results) {
  for (Operation *dst : memoryOps) {
    results.try_emplace(dst); // every access gets a (possibly empty) entry
    if (nonPolyhedral.contains(dst))
      continue;
    affine::MemRefAccess dstAccess(dst);
    for (Operation *src : memoryOps) {
      if (src == dst || nonPolyhedral.contains(src))
        continue;
      affine::MemRefAccess srcAccess(src);
      unsigned numCommon = affine::getInnermostCommonLoopDepth({src, dst});
      for (unsigned depth = 1; depth <= numCommon + 1; ++depth) {
        // Carried depths keep read-after-read reuse edges (allowRAR = true);
        // the loop-independent depth uses allowRAR = false so the dist-0 edge
        // is oriented by program order and read-read pairs get none.
        bool allowRAR = depth <= numCommon;
        affine::FlatAffineValueConstraints constraints;
        SmallVector<affine::DependenceComponent, 2> comps;
        auto result = affine::checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &constraints, &comps, allowRAR);
        if (hasDependence(result.value)) {
          rescaleOnLoopStep(comps);
          results[dst].emplace_back(src, result.value, comps);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Stream dependences
//===----------------------------------------------------------------------===//

// Nearest enclosing counted loop (affine.for or scf.for), skipping non-loop
// parents (e.g. affine.if / scf.if). Null if the op is not inside a loop.
static Operation *getNearestLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(parent))
      return parent;
  return nullptr;
}

// Enclosing counted loops (affine.for or scf.for) of `op`, ordered outermost ->
// innermost (matching getAffineForIVs), for building dependence components.
static SmallVector<Operation *> getEnclosingLoops(Operation *op) {
  SmallVector<Operation *> inner; // innermost -> outermost as collected
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(parent))
      inner.push_back(parent);
  return llvm::to_vector(llvm::reverse(inner));
}

// Whether two same-base stream accesses may touch the same FIFO. A stream value
// is an array of FIFOs selected by its indices, so this is an affine
// disambiguation on the indices, analogous to array-subscript aliasing.
namespace {
enum class FifoAlias { Same, Distinct, Unknown };
} // namespace

// Whether result `k` of `m` is a function of an enclosing loop IV. The
// builder classifies loop IVs as affine DIMS and loop-invariant values
// (function args, worker-ids) as SYMBOLS, so "uses a dim" is exactly "varies
// across loop iterations", which is what a constant inter-access offset on
// that coordinate needs to mean the two accesses sweep overlapping FIFO-index
// ranges (a fixed offset on a worker-id index instead selects a distinct FIFO).
static bool coordDependsOnIV(const affine::AffineValueMap &m, unsigned k) {
  bool usesDim = false;
  m.getAffineMap().getResult(k).walk([&](AffineExpr e) {
    if (isa<AffineDimExpr>(e))
      usesDim = true;
  });
  return usesDim;
}

static FifoAlias compareFifo(AffineValueMapBuilder &builder, Operation *a,
                             Operation *b) {
  builder.reset();
  for (Value idx : asMemAccess(a)->indices)
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  auto ma = builder.compose();

  builder.reset();
  for (Value idx : asMemAccess(b)->indices)
    if (failed(builder.importValue(idx)))
      return FifoAlias::Unknown;
  auto mb = builder.compose();

  if (ma.getNumResults() != mb.getNumResults())
    return FifoAlias::Unknown;

  affine::AffineValueMap diff;
  affine::AffineValueMap::difference(ma, mb, &diff);
  bool allZero = true;
  for (unsigned k = 0, e = diff.getAffineMap().getNumResults(); k < e; ++k) {
    auto cst = dyn_cast<AffineConstantExpr>(diff.getAffineMap().getResult(k));
    if (!cst) {
      // Symbolic offset: cannot prove same or distinct FIFO.
      allZero = false;
      continue;
    }
    if (cst.getValue() != 0) {
      // A nonzero constant offset on an IV-dependent coordinate (`put
      // fifo[i+1]` / `get fifo[i]`) overlaps FIFO ranges across iterations, so
      // serialize. An IV-independent offset selects a genuinely distinct FIFO.
      if (coordDependsOnIV(ma, k))
        return FifoAlias::Unknown;
      return FifoAlias::Distinct;
    }
  }
  return allZero ? FifoAlias::Same : FifoAlias::Unknown;
}

// Build dependence components mirroring the op's enclosing loop nest, placing
// `distance` on the innermost loop (the only component the scheduler reads).
// Empty when no loop encloses `op`, and so unable to carry any distance: a
// func-scope access runs once, in the straight-line span.
static SmallVector<affine::DependenceComponent>
streamDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::DependenceComponent> comps;
  for (Operation *loop : getEnclosingLoops(op)) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  if (!comps.empty())
    comps.back().lb = distance;
  return comps;
}

// Streams are FIFOs: every pair of accesses to the same channel must preserve
// program and iteration order regardless of direction (get-get is ordered
// too, unlike memory's RAW/WAR/WAW). Each may-aliasing pair gets a distance-0
// intra-iteration edge plus a distance-1 loop-carried back edge, closing the
// recurrence that bounds II exactly at the FIFO issue-order bound (II >= 1 +
// (t_later - t_earlier)), not merely conservatively.
//
// All pairs are serialized rather than chaining program order, because FIFO
// may-aliasing is non-transitive: a chain would under-constrain a
// mutually-aliasing group, while per-pair `Distinct` pruning still keeps
// provably-separate FIFOs independent.
static void checkStreamDependence(SmallVectorImpl<Operation *> &streamOps,
                                  AffineValueMapBuilder &builder,
                                  MemoryDependenceResult &results) {
  for (unsigned i = 0, e = streamOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      // `earlier` precedes `later` in program order: `walk` is a pre-order
      // traversal, so a smaller index is never scheduled after a larger one.
      Operation *earlier = streamOps[i];
      Operation *later = streamOps[j];

      // Different stream base SSA values (views peeled) are always
      // independent: SSA identity is a precise disambiguation for streams.
      if (asMemAccess(earlier)->root != asMemAccess(later)->root)
        continue;

      // Only serialize accesses sharing the same innermost loop, so both ends
      // land in one scheduling problem. Two accesses with no enclosing loop
      // share the function's straight-line span, which is one problem too.
      Operation *loop = getNearestLoop(earlier);
      if (loop != getNearestLoop(later))
        continue;

      // Provably-distinct FIFOs are independent; same or unknown are ordered.
      if (compareFifo(builder, earlier, later) == FifoAlias::Distinct)
        continue;

      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  streamDepComponents(later, /*distance=*/0));
      // The back edge needs a loop level to carry its distance. Outside any
      // loop there are no iterations to overtake, so the forward edge alone is
      // already exact.
      if (loop)
        results[earlier].emplace_back(
            later, affine::DependenceResult::HasDependence,
            streamDepComponents(earlier, /*distance=*/1));
    }
  }
}

//===----------------------------------------------------------------------===//
// Conservative memref dependences
//===----------------------------------------------------------------------===//

// Whether the polyhedral test can model where `op` sits: every loop
// enclosing it must be an affine.for. `getAffineForIVs` (and through it
// `getInnermostCommonLoopDepth`) collects affine.for ancestors while silently
// skipping every other loop form, so an affine access under an
// scf.for/scf.while is outside the test's domain even though `MemRefAccess`
// accepts it: the depth ladder never names that loop, so a dependence it
// carries is never queried and the pair is reported loop-independent, losing
// a memory-carried recurrence (e.g. `for j in range(n): out[i] += ...`).
// Such accesses go to the conservative path with the non-affine ones.
static bool inAffineNest(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p) && !isa<affine::AffineForOp>(p))
      return false;
  return true;
}

// Dependence components mirroring the op's enclosing loop nest, `distance`
// on the innermost loop. Empty (loop-independent, distance 0) when the op is
// not in any loop: unlike streamDepComponents, a non-affine access may be
// straight-line.
static SmallVector<affine::DependenceComponent>
memDepComponents(Operation *op, int64_t distance) {
  SmallVector<affine::DependenceComponent> comps;
  for (Operation *loop : getEnclosingLoops(op)) {
    affine::DependenceComponent comp;
    comp.op = loop;
    comp.lb = 0;
    comp.ub = 0;
    comps.push_back(comp);
  }
  if (!comps.empty())
    comps.back().lb = distance;
  return comps;
}

// Conservative memory dependences for pairs the polyhedral test cannot model
// (`nonPolyhedral`: a plain memref.load/store, e.g. indirect A[idx[i]],
// histogram/scatter, or scf-lowered tiles; or an affine access whose loop
// nest is not all-affine; see inAffineNest). Any two accesses to the same
// array with at least one write are serialized in program order (a
// distance-0 forward edge), plus a distance-1 loop-carried back edge when
// they share an innermost loop (closing the recurrence that bounds II).
// Read-read pairs commute and are left independent. An `allo.assume.nodep`
// hint can prune a proven-false edge to recover II.
static void checkConservativeDependence(
    ArrayRef<Operation *> accessOps,
    const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
    MemoryDependenceResult &results) {
  for (unsigned i = 0, e = accessOps.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      Operation *earlier = accessOps[i];
      Operation *later = accessOps[j];

      // Pairs the polyhedral test models are handled precisely there.
      if (!nonPolyhedral.contains(earlier) && !nonPolyhedral.contains(later))
        continue;
      // Different arrays never conflict (distinct roots are distinct arrays;
      // the Allo frontend has no pointers); read-read pairs commute.
      auto ea = asMemAccess(earlier);
      auto la = asMemAccess(later);
      if (ea->root != la->root)
        continue;
      if (!ea->isWrite && !la->isWrite)
        continue;

      // Forward intra-iteration edge (preserve program order).
      results[later].emplace_back(earlier,
                                  affine::DependenceResult::HasDependence,
                                  memDepComponents(later, /*distance=*/0));
      // Loop-carried back edge when both share an innermost loop, closing the
      // recurrence that bounds II. An `allo.assume.nodep` hint can prune this
      // edge to recover II.
      Operation *loop = getNearestLoop(earlier);
      if (loop && loop == getNearestLoop(later))
        results[earlier].emplace_back(
            later, affine::DependenceResult::HasDependence,
            memDepComponents(earlier, /*distance=*/1));
    }
  }
}

//===----------------------------------------------------------------------===//
// assume.nodep hint consumption
//===----------------------------------------------------------------------===//

// Direction of a dependence edge source -> dst by the read/write nature of its
// endpoints. In both the forward and back-edge orientations `source` is the
// producer (scheduled first) and `dst` the consumer, so this is orientation-
// independent: read-after-write is a write source + read dst, etc.
static AssumeDepDirEnum edgeDirection(Operation *source, Operation *dst) {
  bool sw = asMemAccess(source)->isWrite, dw = asMemAccess(dst)->isWrite;
  if (sw && dw)
    return AssumeDepDirEnum::WAW;
  return sw ? AssumeDepDirEnum::RAW : AssumeDepDirEnum::WAR;
}

// The body block of the counted loop (affine.for or scf.for) whose induction
// variable is `iv`, or null if `iv` is not a counted-loop induction variable.
static Block *loopBodyForIV(Value iv) {
  if (auto loop = affine::getForInductionVarOwner(iv))
    return loop.getBody();
  if (auto loop = scf::getForInductionVarOwner(iv))
    return loop.getBody();
  // `loop-canonicalization` rewrites each original iv to an `affine.apply`
  // (floordiv/mod) of the surviving iv. Trace back through it so a nodep scoped
  // to a pre-coalescing iv still resolves to the coalesced loop.
  if (auto apply = iv.getDefiningOp<affine::AffineApplyOp>())
    for (Value operand : apply.getOperands())
      if (Block *body = loopBodyForIV(operand))
        return body;
  return nullptr;
}

// Prune the conservative dependence edges that an `allo.assume.nodep`
// (dependent = false) declares absent, matching by array, enclosing loop,
// inter/intra class, and, when given, direction and distance. Only
// conservative edges are removed: a proven affine dependence is never
// dropped, so a hint that merely restates something the analysis already
// inferred (an affine-provable independence leaves no edge; an
// affine-provable dependence is not a conservative edge) is a no-op.
static void
applyNoDepHints(ArrayRef<AssumeNoDepOp> hints,
                const llvm::SmallDenseSet<Operation *> &nonPolyhedral,
                MemoryDependenceResult &results) {
  for (AssumeNoDepOp hint : hints) {
    if (hint.getDependent())
      // Only "no dependence" assertions prune; `dependent = true`
      // (assert-add) is a no-op since the analysis is already conservative
      // and never misses a real dependence to re-add.
      continue;
    // Resolve through views so it compares equal to the access roots.
    Value array = resolveRoot(hint.getVariable());
    Block *body = loopBodyForIV(hint.getIv());
    if (!body)
      continue;
    bool inter = hint.getDepType() == AssumeDepTypeEnum::Inter;
    auto dir = hint.getDirection();
    IntegerAttr distAttr = hint.getDistanceAttr();

    auto matches = [&](Operation *source, Operation *dst,
                       const MemoryDependence &dep) {
      // Same array, at least one endpoint outside the polyhedral test (so this
      // is a conservative edge), both accesses inside the hinted loop.
      if (asMemAccess(source)->root != array || asMemAccess(dst)->root != array)
        return false;
      if (!nonPolyhedral.contains(source) && !nonPolyhedral.contains(dst))
        return false;
      if (!body->findAncestorOpInBlock(*source) ||
          !body->findAncestorOpInBlock(*dst))
        return false;
      // inter- vs intra-iteration by the innermost distance component.
      int64_t d = dep.dependenceComponents.empty()
                      ? 0
                      : dep.dependenceComponents.back().lb.value_or(0);
      if (inter ? d < 1 : d != 0)
        return false;
      if (dir && edgeDirection(source, dst) != *dir)
        return false;
      if (distAttr && d != distAttr.getInt())
        return false;
      return true;
    };

    size_t pruned = 0;
    for (auto &entry : results)
      llvm::erase_if(entry.second, [&](const MemoryDependence &dep) {
        bool match = matches(dep.source, entry.first, dep);
        pruned += match;
        return match;
      });

    // Report the outcome as an info-level fact: how many conservative edges
    // this hint retired and the claim that authorized it. A zero count flags
    // a hint matching nothing, i.e. the dependence was already inferred absent.
    auto note = info(Stage::Sched, hint.getOperation());
    note << "Applied dependence hint: pruned " << pruned << " conservative "
         << (inter ? "loop-carried" : "intra-iteration") << " dependence edge"
         << (pruned == 1 ? "" : "s");
    if (dir)
      note << " direction="
           << (*dir == AssumeDepDirEnum::RAW   ? "RAW"
               : *dir == AssumeDepDirEnum::WAR ? "WAR"
                                               : "WAW");
    if (distAttr)
      note << " distance=" << distAttr.getInt();
  }
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis
//===----------------------------------------------------------------------===//

namespace mlir::allo {

int64_t carriedDistanceAtLevel(ArrayRef<affine::DependenceComponent> comps,
                               unsigned level, bool &drop, bool &valid) {
  drop = false;
  valid = true;
  if (comps.empty())
    return 0; // loop-independent: same iteration at every level
  if (comps.size() < level) {
    valid = false;
    return 0;
  }
  // A `*`-direction component (lb == nullopt) is an UNKNOWN distance, not 0.
  // An OUTER level drops the edge only on a PROVEN positive distance; THIS
  // level falls back to 1 so the modulo solver never under-bounds II.
  for (unsigned k = 0; k + 1 < level; ++k) // components outer to the level
    if (comps[k].lb.value_or(0) > 0) {
      drop = true;
      return 0;
    }
  auto d = comps[level - 1].lb;
  return d.has_value() ? *d : 1;
}

bool isUnmodeledMemoryAccess(Operation *op) {
  // The complement of the access kinds the constructor's walk below collects.
  if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface,
          memref::LoadOp, memref::StoreOp, StreamGetOp, StreamPutOp,
          AssumeNoDepOp, AssumeSSAOp>(op))
    return false;
  auto mem = dyn_cast<MemoryEffectOpInterface>(op);
  return mem && (mem.hasEffect<MemoryEffects::Read>() ||
                 mem.hasEffect<MemoryEffects::Write>());
}

DependenceAnalysis::DependenceAnalysis(func::FuncOp funcOp) : func(funcOp) {
  SmallVector<Operation *> memoryOps;
  SmallVector<Operation *> streamOps;
  // All memref accesses in program (walk) order, plus the subset outside the
  // polyhedral test: the op itself is non-affine, or its loop nest is not
  // all-affine. That subset takes the conservative fallback below.
  SmallVector<Operation *> accessOps;
  SmallVector<AssumeNoDepOp> noDepHints;
  SmallVector<Assumption> assumptions;
  funcOp->walk([&](Operation *op) {
    if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(
            op)) {
      memoryOps.push_back(op);
      accessOps.push_back(op);
      if (!inAffineNest(op))
        nonPolyhedral.insert(op);
    } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
      nonPolyhedral.insert(op);
      accessOps.push_back(op);
    } else if (isa<StreamGetOp, StreamPutOp>(op)) {
      streamOps.push_back(op);
    } else if (auto hint = dyn_cast<AssumeNoDepOp>(op)) {
      noDepHints.push_back(hint);
    } else if (auto hint = dyn_cast<AssumeSSAOp>(op)) {
      collectAssumptions(hint.getCondition(), assumptions);
    }
  });

  // Affine memref dependences: each ordered pair over all carried depths plus
  // the loop-independent (intra-iteration) depth (see checkMemrefDependence).
  checkMemrefDependence(memoryOps, nonPolyhedral, results);

  // Conservative ordering for the pairs the polyhedral test skips.
  checkConservativeDependence(accessOps, nonPolyhedral, results);

  AffineValueMapBuilder builder(funcOp.getContext());
  checkStreamDependence(streamOps, builder, results);

  // User hints: prune conservative edges the programmer proves absent. Applied
  // last, over the fully-built edge set, so pruning a non-existent edge (the
  // fact was already inferred) is naturally a no-op.
  applyNoDepHints(noDepHints, nonPolyhedral, results);

  // Distill the assume.ssa value facts into per-value constant ranges (the seed
  // a value-range consumer reads; does not affect dependence edges).
  buildAssumedRanges(assumptions, assumedRanges);

  // Surface the distilled ranges as an info-level analysis fact, one line per
  // constrained value, mirroring how the scheduler reports its other facts.
  if (!assumedRanges.empty()) {
    info(Stage::Sched) << "Applied value hints: distilled "
                       << assumedRanges.size() << " value range"
                       << (assumedRanges.size() == 1 ? "" : "s");
    for (const auto &[v, r] : assumedRanges) {
      auto lb = r.lb ? std::to_string(*r.lb) : "-inf";
      auto ub = r.ub ? std::to_string(*r.ub) : "+inf";
      info(Stage::Sched) << "  " << logging::detail::describe(v.getLoc())
                         << " in [" << lb << ", " << ub << "]";
    }
  }
}

// Retargets oldOp's dependence list and every edge sourced from it to newOp.
void DependenceAnalysis::replaceOp(Operation *oldOp, Operation *newOp) {
  auto it = results.find(oldOp);
  if (it != results.end()) {
    results[newOp] = std::move(it->second);
    results.erase(it);
  }

  for (auto &entry : results)
    for (auto &dep : entry.second)
      if (dep.source == oldOp)
        dep.source = newOp;
}

} // namespace mlir::allo
