/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/HierarchicalDependence.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/Scheduler.h"
#include "allo/Scheduling/Utils.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_SDCSCHEDULINGPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

// Erase a consumed hint op along with any operand-producing ops it leaves
// trivially dead
static void eraseHintAndDeadInputs(RewriterBase &b, Operation *op) {
  SmallVector<Value, 4> operands(op->getOperands());
  b.eraseOp(op);
  for (Value v : operands)
    if (Operation *def = v.getDefiningOp())
      if (isOpTriviallyDead(def))
        eraseHintAndDeadInputs(b, def);
}

// Whether \p loop's body contains a nested loop (affine.for or scf.for), i.e.
// it is not truly innermost. Detecting both kinds also stops an affine.for that
// encloses an scf.for from being silently flattened.
static bool hasNestedLoop(LoopLikeOpInterface loop) {
  bool found = false;
  for (Region *r : loop.getLoopRegions()) {
    r->walk([&](Operation *op) {
      if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (found)
      break;
  }
  return found;
}

// The maximal perfect band of counted loops (affine.for / scf.for) rooted at
// \p root: descend while a level's body is exactly { inner counted loop,
// terminator }. Returns [root, ..., innermost]. Generalizes affine's
// `getPerfectlyNestedLoops` over `LoopLikeOpInterface` so affine and scf.for
// nests share one perfect-nest walk (driver + latency folding).
static SmallVector<LoopLikeOpInterface> perfectNest(LoopLikeOpInterface root) {
  SmallVector<LoopLikeOpInterface> nest{root};
  while (true) {
    Block &body = nest.back().getLoopRegions().front()->front();
    Operation *first = &body.front();
    if (first->getNextNode() != body.getTerminator())
      break; // the body holds more than just the inner loop
    auto inner = dyn_cast<LoopLikeOpInterface>(first);
    if (!inner || !isa<AffineForOp, scf::ForOp>(first))
      break;
    nest.push_back(inner);
  }
  return nest;
}

// The schedule length (single-iteration pipeline depth) of a solved problem:
// the last op's start cycle plus one.
static int64_t scheduleDepth(circt::scheduling::Problem &problem) {
  int64_t maxStart = 0;
  for (Operation *op : problem.getOperations())
    if (std::optional<unsigned> start = problem.getStartTime(op))
      maxStart = std::max<int64_t>(maxStart, static_cast<int64_t>(*start));
  return maxStart + 1;
}

// Whether the problem carries a loop-carried recurrence -- a dependence
// spanning
// >= 1 iteration. Its presence is why a modulo II can exceed the pure resource
// bound; reported in the schedule narrative.
static bool hasCarriedRecurrence(circt::scheduling::CyclicProblem &problem) {
  for (Operation *op : problem.getOperations())
    for (auto dep : problem.getDependences(op))
      if (problem.getDistance(dep).value_or(0) > 0)
        return true;
  return false;
}

// An inclusive integer interval `[lo, hi]`; an open endpoint is unbounded.
using Interval = std::pair<std::optional<int64_t>, std::optional<int64_t>>;

// Bound an affine trip-count expression given each operand's known range. The
// divisor/multiplier of a mul/div/mod is always a constant in affine form, so
// each case is exact interval arithmetic; a missing operand bound propagates as
// an open endpoint.
static Interval evalInterval(AffineExpr e, ArrayRef<AssumedRange> operands,
                             unsigned numDims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return {c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return {operands[d.getPosition()].lb, operands[d.getPosition()].ub};
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
    const AssumedRange &r = operands[numDims + s.getPosition()];
    return {r.lb, r.ub};
  }
  auto bin = cast<AffineBinaryOpExpr>(e);
  Interval l = evalInterval(bin.getLHS(), operands, numDims);
  auto constRHS = dyn_cast<AffineConstantExpr>(bin.getRHS());
  auto apply = [](std::optional<int64_t> a, std::optional<int64_t> b,
                  auto op) -> std::optional<int64_t> {
    if (a && b)
      return op(*a, *b);
    return std::nullopt;
  };
  switch (bin.getKind()) {
  case AffineExprKind::Add: {
    Interval r = evalInterval(bin.getRHS(), operands, numDims);
    auto add = [](int64_t a, int64_t b) { return a + b; };
    return {apply(l.first, r.first, add), apply(l.second, r.second, add)};
  }
  case AffineExprKind::Mul: {
    int64_t c = constRHS.getValue(); // affine: one factor is constant
    auto mul = [&](std::optional<int64_t> x) {
      return apply(x, c, std::multiplies<int64_t>());
    };
    return c >= 0 ? Interval{mul(l.first), mul(l.second)}
                  : Interval{mul(l.second), mul(l.first)};
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    int64_t c = constRHS.getValue(); // affine, positive divisor
    bool ceil = bin.getKind() == AffineExprKind::CeilDiv;
    auto div = [&](std::optional<int64_t> x) -> std::optional<int64_t> {
      if (!x)
        return std::nullopt;
      return ceil ? llvm::divideCeilSigned(*x, c)
                  : llvm::divideFloorSigned(*x, c);
    };
    return {div(l.first), div(l.second)};
  }
  case AffineExprKind::Mod:
    return {int64_t{0}, constRHS.getValue() - 1};
  default:
    return {std::nullopt, std::nullopt};
  }
}

// The trip count of one loop: exact when it is a compile-time constant,
// otherwise a worst-case upper bound derived from the `allo.assume.ssa` ranges
// of its symbolic bounds (setting `isBound`), or nullopt if still unbounded.
static std::optional<int64_t>
loopTripCount(AffineForOp loop, DependenceAnalysis &deps, bool &isBound) {
  if (std::optional<uint64_t> c = getConstantTripCount(loop))
    return static_cast<int64_t>(*c);

  AffineMap map;
  SmallVector<Value> operands;
  getTripCountMapAndOperands(loop, &map, &operands);
  if (!map || map.getNumResults() != 1)
    return std::nullopt;

  SmallVector<AssumedRange> ranges;
  for (Value v : operands) {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      ranges.push_back({*c, *c});
    else if (std::optional<AssumedRange> r = deps.getAssumedRange(v))
      ranges.push_back(*r);
    else
      ranges.push_back({});
  }
  Interval iv = evalInterval(map.getResult(0), ranges, map.getNumDims());
  if (!iv.second)
    return std::nullopt;
  isBound = true;
  return std::max<int64_t>(0, *iv.second);
}

// The trip count of one counted scf.for: exact when lb/ub/step are all
// compile-time constants, otherwise a worst-case upper bound from the
// `allo.assume.ssa` ranges of its (dynamic) bound operands (setting `isBound`),
// or nullopt if still unbounded. Mirrors `loopTripCount` for affine loops.
static std::optional<int64_t>
scfForTripCount(scf::ForOp loop, DependenceAnalysis &deps, bool &isBound) {
  auto rangeOf = [&](Value v) -> AssumedRange {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return {*c, *c};
    if (std::optional<AssumedRange> r = deps.getAssumedRange(v))
      return *r;
    return {};
  };
  AssumedRange lb = rangeOf(loop.getLowerBound());
  AssumedRange ub = rangeOf(loop.getUpperBound());
  AssumedRange step = rangeOf(loop.getStep());
  auto isConst = [](const AssumedRange &r) {
    return r.lb && r.ub && *r.lb == *r.ub;
  };
  // Exact when every bound is a known constant.
  if (isConst(lb) && isConst(ub) && isConst(step)) {
    int64_t s = *step.lb;
    if (s <= 0)
      return std::nullopt; // non-positive step unsupported
    return std::max<int64_t>(0, llvm::divideCeilSigned(*ub.lb - *lb.lb, s));
  }
  // Worst case: ceil((max ub - min lb) / min step), needing a positive step.
  if (ub.ub && lb.lb && step.lb && *step.lb >= 1) {
    isBound = true;
    return std::max<int64_t>(0,
                             llvm::divideCeilSigned(*ub.ub - *lb.lb, *step.lb));
  }
  return std::nullopt;
}

// Trip count of any counted loop (affine.for or scf.for).
static std::optional<int64_t>
loopTrip(Operation *loop, DependenceAnalysis &deps, bool &isBound) {
  if (auto affineLoop = dyn_cast<AffineForOp>(loop))
    return loopTripCount(affineLoop, deps, isBound);
  return scfForTripCount(cast<scf::ForOp>(loop), deps, isBound);
}

// Product of the trip counts of all counted loops (affine.for or scf.for)
// enclosing `op`. Returns nullopt if any is unknown.
static std::optional<int64_t>
enclosingTripProduct(Operation *op, DependenceAnalysis &deps, bool &isBound) {
  int64_t product = 1;
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    // An enclosing uncounted `scf.while` runs a data-dependent number of times,
    // so a region nested in it has no statically-known execution count.
    if (isa<scf::WhileOp>(p))
      return std::nullopt;
    if (!isa<AffineForOp, scf::ForOp>(p))
      continue;
    std::optional<int64_t> trip = loopTrip(p, deps, isBound);
    if (!trip)
      return std::nullopt;
    product *= *trip;
  }
  return product;
}

// Whole-region latency in cycles: the innermost body is pipelined
// (`depth + (trip - 1) * II`), and every surrounding loop -- the perfect band
// plus any imperfectly-nested enclosing loops -- multiplies it by its trip
// count. Handles affine.for and scf.for uniformly (perfect band via
// `perfectNest`, trips via `loopTrip`). Returns nullopt if a trip is unknown.
static std::optional<int64_t> regionLatency(Operation *anchor, unsigned ii,
                                            int64_t depth,
                                            DependenceAnalysis &deps,
                                            bool &isBound) {
  SmallVector<LoopLikeOpInterface> nest =
      perfectNest(cast<LoopLikeOpInterface>(anchor));

  std::optional<int64_t> innerTrip =
      loopTrip(nest.back().getOperation(), deps, isBound);
  if (!innerTrip)
    return std::nullopt;
  int64_t latency = depth + (*innerTrip - 1) * static_cast<int64_t>(ii);

  for (LoopLikeOpInterface l : ArrayRef(nest).drop_back()) {
    std::optional<int64_t> trip = loopTrip(l.getOperation(), deps, isBound);
    if (!trip)
      return std::nullopt;
    latency *= *trip;
  }
  std::optional<int64_t> above = enclosingTripProduct(anchor, deps, isBound);
  if (!above)
    return std::nullopt;
  return latency * *above;
}

/// Write the solved schedule of one region onto the IR as attributes: each
/// registered op gets `allo.sched.t` (start time) and `allo.sched.region`, and
/// a per-region descriptor is appended to the func-level `allo.sched.regions`
/// array. This is the schedule "carrier"; nothing structural is materialized.
static void annotateRegion(circt::scheduling::Problem &problem,
                           func::FuncOp func, int64_t regionId, StringRef kind,
                           std::optional<unsigned> ii,
                           std::optional<int64_t> latency,
                           bool latencyIsBound) {
  Builder b(func.getContext());

  int64_t maxStart = 0;
  for (Operation *op : problem.getOperations()) {
    std::optional<unsigned> start = problem.getStartTime(op);
    if (!start)
      continue;
    maxStart = std::max<int64_t>(maxStart, static_cast<int64_t>(*start));
    // A child-loop node (Phase B level problem) is scheduled as its own region,
    // and a loop terminator carries no schedulable compute -- neither is tagged
    // here (they still count toward the region length).
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>())
      continue;
    op->setAttr(sched::kStartTimeAttr, b.getI64IntegerAttr(*start));
    op->setAttr(sched::kRegionIdAttr, b.getI64IntegerAttr(regionId));
  }
  int64_t length = maxStart + 1;

  // Build the per-region descriptor.
  SmallVector<NamedAttribute> fields;
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyId),
                      b.getI64IntegerAttr(regionId));
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyKind),
                      b.getStringAttr(kind));
  if (ii)
    fields.emplace_back(b.getStringAttr(sched::kRegionKeyII),
                        b.getI64IntegerAttr(*ii));
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyLength),
                      b.getI64IntegerAttr(length));
  if (latency) {
    fields.emplace_back(b.getStringAttr(sched::kRegionKeyLatency),
                        b.getI64IntegerAttr(*latency));
    if (latencyIsBound)
      fields.emplace_back(b.getStringAttr(sched::kRegionKeyLatencyBound),
                          b.getUnitAttr());
  }
  auto descriptor = b.getDictionaryAttr(fields);

  // Append to the func-level regions array.
  SmallVector<Attribute> regions;
  if (auto existing = func->getAttrOfType<ArrayAttr>(sched::kRegionsAttr))
    regions.append(existing.begin(), existing.end());
  regions.push_back(descriptor);
  func->setAttr(sched::kRegionsAttr, b.getArrayAttr(regions));
}

/// Additionally write each op's sub-cycle start time (`allo.sched.z`, ns) from
/// a solved chaining problem.
static void
annotateStartTimeInCycle(circt::scheduling::ChainingProblem &problem) {
  Builder b(problem.getContainingOp()->getContext());
  for (Operation *op : problem.getOperations()) {
    if (std::optional<float> z = problem.getStartTimeInCycle(op))
      op->setAttr(sched::kStartTimeInCycleAttr, b.getF32FloatAttr(*z));
  }
}

// The pipeline directive on the loop (or an enclosing loop up to the region
// anchor), from `s.pipeline(ii=N)` -> `allo.pipeline.ii`:
//   >= 1  requested target II: a lower bound on the achieved II
//    0    auto: minimize the II (same as no directive)
//   -1    pipelining disabled: schedule the loop non-pipelined
// Absent => 0 (auto). The directive may sit on any level of a perfect nest.
static int64_t pipelineDirective(Operation *loop, Operation *anchor) {
  for (Operation *op = loop;; op = op->getParentOp()) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kPipelineIIAttr))
      return attr.getInt();
    if (op == anchor || !op->getParentOp())
      return 0;
  }
}

// Schedule one counted loop body (affine.for or scf.for) as a
// `ChainingModuloProblem` (resource-aware, timing-aware) and annotate the
// result (start times, II, sub-cycle times). \p minII lower-bounds the II.
// When \p pipelined is false (`s.pipeline(ii=-1)`) iterations do not overlap:
// the II is reported as the body length, so the region latency folds to
// `trip * depth`. Both cases reify to a dcp.pipeline (a non-pipelined loop is
// just a pipeline whose II equals its own depth).
static LogicalResult scheduleCyclic(LoopLikeOpInterface body,
                                    DependenceAnalysis &deps,
                                    const OperatorLibrary &lib,
                                    func::FuncOp funcOp,
                                    const SchedRegion &region, float cycleTime,
                                    unsigned minII, bool pipelined) {
  auto problem = buildCyclicProblem<ChainingModuloProblem>(body, deps);
  Block *bodyBlock = &body.getLoopRegions().front()->front();
  if (failed(populateOperatorTypes(*bodyBlock, problem, lib)))
    return failure();
  if (failed(populateMemoryResources(*bodyBlock, problem, lib.memoryLibrary())))
    return failure();
  Operation *anchor = bodyBlock->getTerminator();
  if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII)))
    return failure();
  int64_t depth = scheduleDepth(problem);
  unsigned ii = pipelined ? problem.getInitiationInterval().value_or(depth)
                          : static_cast<unsigned>(depth);
  bool isBound = false;
  std::optional<int64_t> latency =
      regionLatency(region.anchor(), ii, depth, deps, isBound);

  {
    auto d = info(Stage::Sched, region.anchor());
    d << "Scheduled: II=" << ii;
    if (!pipelined)
      d << " (pipelining off, iterations run back-to-back)";
    else if (ii == 1)
      d << " (fully pipelined)";
    else if (hasCarriedRecurrence(problem))
      d << " (>1: a loop-carried recurrence and/or shared-resource limit)";
    else
      d << " (>1: a shared-resource limit, e.g. memory ports)";
    if (latency)
      d << ", latency = " << *latency
        << (isBound ? " (assume-bounded worst case)" : "");
    else
      d << ", latency dynamic (trip not statically known)";
  }

  // A non-pipelined multi-cycle operator holds its unit for its whole latency
  // (occupancy stamped as kResourceCyclesAttr), so it caps iteration overlap.
  // Name the dominant one as a QoR explanation for II > 1.
  if (pipelined && ii > 1) {
    Operation *blocking = nullptr;
    int64_t maxOcc = 1;
    bodyBlock->walk([&](Operation *op) {
      if (auto a = op->getAttrOfType<IntegerAttr>(sched::kResourceCyclesAttr))
        if (a.getInt() > maxOcc) {
          maxOcc = a.getInt();
          blocking = op;
        }
    });
    if (blocking)
      info(Stage::Sched, blocking)
          << "Operator " << blocking->getName().getStringRef()
          << " is non-pipelined and holds its unit for " << maxOcc
          << " cycle(s), limiting iteration overlap";
  }

  annotateRegion(problem, funcOp, region.id, "cyclic", ii, latency, isBound);
  annotateStartTimeInCycle(problem);
  return success();
}

// Schedule an uncounted `scf.while` (before + after as one iteration) as a
// `ChainingModuloProblem` -- the flushing-pipeline scheduling view. The trip
// count is data-dependent, so latency is omitted (like a dynamic counted loop).
static LogicalResult scheduleWhile(scf::WhileOp w, DependenceAnalysis &deps,
                                   const OperatorLibrary &lib,
                                   func::FuncOp funcOp,
                                   const SchedRegion &region, float cycleTime) {
  auto problem = buildWhileProblem<ChainingModuloProblem>(w, deps);
  // Operator types over both regions in one memory-bank analysis.
  if (failed(populateOperatorTypesImpl(
          problem,
          [&](auto handle) {
            w.getBefore().walk(handle);
            w.getAfter().walk(handle);
          },
          lib)))
    return failure();
  if (failed(populateMemoryResourcesImpl(
          problem,
          [&](auto handle) {
            w.getBefore().walk(handle);
            w.getAfter().walk(handle);
          },
          lib.memoryLibrary())))
    return failure();
  Operation *anchor = w.getYieldOp().getOperation();
  // A while pipelines as a flushing pipeline; honor a requested target II (>=1)
  // as a lower bound. `ii=-1` (pipelining off) is not modeled for while loops.
  int64_t dir = pipelineDirective(w, region.anchor());
  unsigned minII = dir >= 1 ? static_cast<unsigned>(dir) : 1;
  if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII)))
    return failure();
  std::optional<unsigned> ii = problem.getInitiationInterval();
  info(Stage::Sched, w.getOperation())
      << "  -> while loop scheduled as a flushing pipeline: II="
      << ii.value_or(0)
      << " (trip is data-dependent, so whole-loop latency is unknown)";
  annotateRegion(problem, funcOp, region.id, "cyclic", ii, std::nullopt,
                 /*latencyIsBound=*/false);
  annotateStartTimeInCycle(problem);
  return success();
}

// Schedule one straight-line region as a `ChainingSharedOperatorsProblem` (the
// acyclic twin: resource-aware, timing-aware) and annotate the result.
static LogicalResult
scheduleAcyclic(ArrayRef<Operation *> ops, DependenceAnalysis &deps,
                const OperatorLibrary &lib, func::FuncOp funcOp,
                const SchedRegion &region, float cycleTime) {
  ChainingSharedOperatorsProblem problem =
      buildAcyclicProblem<ChainingSharedOperatorsProblem>(ops, deps);
  if (failed(populateOperatorTypes(ops, problem, lib)))
    return failure();
  if (failed(populateMemoryResources(ops, problem, lib.memoryLibrary())))
    return failure();
  if (failed(solveSchedulingProblem(problem, ops.back(), cycleTime)))
    return failure();
  // A straight-line region runs once per enclosing-loop iteration.
  bool isBound = false;
  std::optional<int64_t> above =
      enclosingTripProduct(ops.front(), deps, isBound);
  int64_t depth = scheduleDepth(problem);
  std::optional<int64_t> latency =
      above ? std::optional<int64_t>(depth * *above) : std::nullopt;
  {
    auto d = info(Stage::Sched, ops.front());
    d << "Scheduled: depth = " << depth << " cycles";
    if (latency)
      d << ", latency = " << *latency << (isBound ? " (assume-bounded)" : "");
  }
  annotateRegion(problem, funcOp, region.id, "acyclic", std::nullopt, latency,
                 isBound);
  annotateStartTimeInCycle(problem);
  return success();
}

// Whole-kernel latency: sum the per-region latencies (regions compose by
// program order, no overlap). Set the func attribute only when every region has
// a known latency; flag it as a bound if any region's latency was assumed.
static void annotateKernelLatency(func::FuncOp funcOp) {
  auto regionsAttr = funcOp->getAttrOfType<ArrayAttr>(sched::kRegionsAttr);
  if (!regionsAttr)
    return;
  int64_t total = 0;
  bool isBound = false;
  for (Attribute a : regionsAttr) {
    auto d = cast<DictionaryAttr>(a);
    // A region absorbed into a Phase B level is folded into that level's
    // latency; counting it here would double-count.
    if (d.get(sched::kRegionKeyParent))
      continue;
    auto lat = d.getAs<IntegerAttr>(sched::kRegionKeyLatency);
    if (!lat)
      return; // an unknown region latency leaves the kernel total unknown
    total += lat.getInt();
    isBound |= d.get(sched::kRegionKeyLatencyBound) != nullptr;
  }
  Builder b(funcOp.getContext());
  funcOp->setAttr(sched::kLatencyAttr, b.getI64IntegerAttr(total));
  if (isBound)
    funcOp->setAttr(sched::kLatencyBoundAttr, b.getUnitAttr());
}

// Per-invocation latency of an already-scheduled child loop -- one run of the
// loop, the occupancy the level's modulo problem reserves for the loop node.
// Every scheduled region nested under the loop has a descriptor latency that
// already folds in all enclosing trips; their sum is the loop's whole
// contribution, so dividing by the loop's enclosing trips leaves one
// invocation. This handles both a perfect child (one region) and an imperfect
// one decomposed into several sub-regions (the recursive latency fold). nullopt
// if any nested region has no known latency, or a trip is unknown.
static std::optional<int64_t> childInvocationLatency(Operation *childLoop,
                                                     func::FuncOp funcOp,
                                                     DependenceAnalysis &deps) {
  llvm::SmallDenseSet<int64_t> ids;
  childLoop->walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(sched::kRegionIdAttr))
      ids.insert(attr.getInt());
  });
  auto regions = funcOp->getAttrOfType<ArrayAttr>(sched::kRegionsAttr);
  if (ids.empty() || !regions)
    return std::nullopt;
  int64_t sum = 0;
  unsigned found = 0;
  for (Attribute a : regions) {
    auto d = cast<DictionaryAttr>(a);
    if (!ids.count(d.getAs<IntegerAttr>(sched::kRegionKeyId).getInt()))
      continue;
    auto l = d.getAs<IntegerAttr>(sched::kRegionKeyLatency);
    if (!l)
      return std::nullopt; // an unknown region latency
    sum += l.getInt();
    ++found;
  }
  bool isBound = false;
  std::optional<int64_t> above = enclosingTripProduct(childLoop, deps, isBound);
  if (found != ids.size() || !above || *above == 0)
    return std::nullopt;
  return sum / *above;
}

// After a level is solved: mark every sub-region nested under `level` (a child
// loop's own region(s)) as absorbed into the level -- stamp `parent = levelId`
// on its descriptor so the whole-kernel / reify latency composition skips it
// (its latency is already folded into the level's). Additionally record each
// child-loop node's start within the level's II (`parent_start`) on that
// child's primary region, so the reify offsets the nested `dcp.pipeline`.
static void absorbLevelChildren(func::FuncOp funcOp, LoopLikeOpInterface level,
                                int64_t levelId,
                                circt::scheduling::Problem &problem,
                                const LevelAnalysis &a) {
  // A child-loop node's primary region id: the region tagged on its (innermost)
  // body ops. A perfect child has exactly one such id.
  llvm::DenseMap<int64_t, int64_t> parentStart; // regionId -> start-in-level
  for (const LevelNode &n : a.nodes) {
    if (!n.isLoop)
      continue;
    std::optional<unsigned> start = problem.getStartTime(n.anchor);
    int64_t rid = -1;
    n.anchor->walk([&](Operation *o) {
      if (auto attr = o->getAttrOfType<IntegerAttr>(sched::kRegionIdAttr))
        rid = attr.getInt();
    });
    if (rid >= 0 && start)
      parentStart[rid] = static_cast<int64_t>(*start);
  }
  // Absorbed = every region id tagged strictly inside the level (i.e. under a
  // child loop); the level's own leaf ops carry `levelId` and are excluded.
  llvm::DenseSet<int64_t> absorbed;
  level.getLoopRegions().front()->front().walk([&](Operation *o) {
    if (auto attr = o->getAttrOfType<IntegerAttr>(sched::kRegionIdAttr))
      if (attr.getInt() != levelId)
        absorbed.insert(attr.getInt());
  });

  Builder b(funcOp.getContext());
  auto regions = funcOp->getAttrOfType<ArrayAttr>(sched::kRegionsAttr);
  SmallVector<Attribute> updated;
  for (Attribute at : regions) {
    auto d = cast<DictionaryAttr>(at);
    int64_t id = d.getAs<IntegerAttr>(sched::kRegionKeyId).getInt();
    if (!absorbed.count(id)) {
      updated.push_back(d);
      continue;
    }
    SmallVector<NamedAttribute> fields(d.begin(), d.end());
    fields.emplace_back(b.getStringAttr(sched::kRegionKeyParent),
                        b.getI64IntegerAttr(levelId));
    if (auto it = parentStart.find(id); it != parentStart.end())
      fields.emplace_back(b.getStringAttr(sched::kRegionKeyParentStart),
                          b.getI64IntegerAttr(it->second));
    updated.push_back(b.getDictionaryAttr(fields));
  }
  funcOp->setAttr(sched::kRegionsAttr, b.getArrayAttr(updated));
}

// Build the level's modulo problem from `a`: leaf ops with their library
// operators, child loops as non-pipelined unit-limit-1 resources (occupancy =
// per-invocation latency), the `analyzeLevel` edges with their distances, all
// nodes ordered before the loop terminator. Returns false when a child's
// per-invocation latency is unknown; on success `tagged` holds the loop nodes
// carrying a transient occupancy attr the caller must remove after solving.
//
// The problem is timing-aware (`ChainingModuloProblem`): leaf ops carry their
// library in/out delays so a combinational chain among level ops is cut at the
// cycle boundary, exactly as in the leaf body. A child-loop node is a
// registered boundary -- its inputs are latched at entry and its result drained
// into a register -- so it terminates any combinational chain (zero in/out
// delay); no per-region delay characterization is needed.
static bool buildLevelProblem(ChainingModuloProblem &problem,
                              const LevelAnalysis &a, LoopLikeOpInterface level,
                              DependenceAnalysis &deps,
                              const OperatorLibrary &lib, func::FuncOp funcOp,
                              int64_t &resourceBound,
                              SmallVectorImpl<Operation *> &tagged) {
  using P = circt::scheduling::Problem;
  Builder b(level.getContext());
  Operation *anchor = level.getLoopRegions().front()->front().getTerminator();
  resourceBound = 0;
  for (const LevelNode &n : a.nodes)
    problem.insertOperation(n.anchor);
  problem.insertOperation(anchor);

  for (auto [idx, n] : llvm::enumerate(a.nodes)) {
    if (n.isLoop) {
      std::optional<int64_t> lat =
          childInvocationLatency(n.anchor, funcOp, deps);
      if (!lat) {
        for (Operation *op : tagged)
          op->removeAttr(sched::kResourceCyclesAttr);
        tagged.clear();
        return false;
      }
      resourceBound = std::max(resourceBound, *lat);
      P::OperatorType opr =
          problem.getOrInsertOperatorType("loop" + std::to_string(idx));
      problem.setLatency(opr, *lat);
      problem.setLinkedOperatorType(n.anchor, opr);
      // A registered boundary: terminates combinational chains on both ports.
      problem.setIncomingDelay(opr, 0.0);
      problem.setOutgoingDelay(opr, 0.0);
      P::ResourceType rsrc =
          problem.getOrInsertResourceType("unit" + std::to_string(idx));
      problem.setLimit(rsrc, 1);
      problem.setLinkedResourceTypes(n.anchor,
                                     SmallVector<P::ResourceType>{rsrc});
      n.anchor->setAttr(sched::kResourceCyclesAttr, b.getI64IntegerAttr(*lat));
      tagged.push_back(n.anchor);
    } else {
      OperatorChar c = lib.lookup(n.anchor);
      P::OperatorType opr = problem.getOrInsertOperatorType(c.typeName);
      problem.setLatency(opr, c.latency);
      problem.setLinkedOperatorType(n.anchor, opr);
      problem.setIncomingDelay(opr, c.inDelay);
      problem.setOutgoingDelay(opr, c.outDelay);
    }
  }
  P::OperatorType zero = problem.getOrInsertOperatorType("_anchor");
  problem.setLatency(zero, 0);
  problem.setLinkedOperatorType(anchor, zero);
  problem.setIncomingDelay(zero, 0.0);
  problem.setOutgoingDelay(zero, 0.0);
  for (const LevelEdge &e : a.edges) {
    P::Dependence dep(a.nodes[e.src].anchor, a.nodes[e.dst].anchor);
    if (succeeded(problem.insertDependence(dep)) && e.distance > 0)
      problem.setDistance(dep, e.distance);
  }
  for (const LevelNode &n : a.nodes)
    (void)problem.insertDependence(P::Dependence(n.anchor, anchor));
  return true;
}

// Phase B (analysis only): build+solve the level problem and DEBUG-log the
// achieved II. No annotate/reify; transient occupancy attrs are removed.
static void logLevelII(const LevelAnalysis &a, LoopLikeOpInterface level,
                       DependenceAnalysis &deps, const OperatorLibrary &lib,
                       func::FuncOp funcOp, float cycleTimeNs) {
  if (!logging::detail::enabled(logging::Level::Debug))
    return;
  ChainingModuloProblem problem(level.getOperation());
  int64_t resourceBound = 0;
  SmallVector<Operation *> tagged;
  if (!buildLevelProblem(problem, a, level, deps, lib, funcOp, resourceBound,
                         tagged)) {
    debug(Stage::Sched) << "  level II_outer = (child latency unknown)";
    return;
  }
  Operation *anchor = level.getLoopRegions().front()->front().getTerminator();
  if (succeeded(
          solveSchedulingProblem(problem, anchor, cycleTimeNs, /*minII=*/1)))
    debug(Stage::Sched) << "  level II_outer = "
                        << problem.getInitiationInterval().value_or(0)
                        << " (resource bound " << resourceBound << ")";
  else
    debug(Stage::Sched) << "  level II_outer = (solve failed)";
  for (Operation *op : tagged)
    op->removeAttr(sched::kResourceCyclesAttr);
}

// Whether a pipelined imperfect level can be fused into ONE outer pipeline and
// reified as Phase B. Otherwise the level falls back to Phase A (sequential
// sub-regions), which is always correct. Requires:
//   * no loop-carried iter_args on the level itself -- their cross-iteration
//     recurrence is not modeled by the level problem (Phase A runs the level as
//     a sequential loop, so its iter_args are correct);
//   * every child region that is a loop is a SINGLE counted loop (affine.for /
//     scf.for, no further nested loop) whose trip and enclosing trips are known
//     -- so it characterizes as one fixed-latency node `L_child` and
//     materializes to exactly one nested dcp.pipeline (a nested/while/perfect-
//     band child would leave a wrapper loop inside the level pipeline).
// This gate runs BEFORE any child is scheduled, so the fallback stays a clean
// Phase A schedule with no double-scheduling.
static bool levelIsPhaseBReifiable(LoopLikeOpInterface level,
                                   DependenceAnalysis &deps) {
  if (!level.getInits().empty())
    return false;
  Block &body = level.getLoopRegions().front()->front();
  // A non-affine access (memref.load/store, opaque effectful op) could hide a
  // level-carried recurrence whose carried distance the affine analysis cannot
  // bound -- but only on a WRITTEN root. A root that is read-only across the
  // level (e.g. a gather `W[idx[j]]` from a weight/lookup table) carries no
  // dependence at all (RAR only), so a non-affine read of it is safe. Only a
  // non-affine access on a written root forces the (always-correct) Phase A.
  Summary footprint;
  body.walk([&](Operation *o) { summarizeOp(o, footprint); });
  for (const auto &kv : footprint.mem)
    if (kv.second.nonAffine && kv.second.writes)
      return false;
  for (const SchedRegion &sub : enumerateRegions(body)) {
    if (sub.kind != allo::RegionKind::Loop)
      continue;
    Operation *child = sub.anchor();
    auto childLoop = dyn_cast<LoopLikeOpInterface>(child);
    if (!childLoop || !isa<AffineForOp, scf::ForOp>(child) ||
        hasNestedLoop(childLoop))
      return false;
    bool isBound = false;
    if (!loopTrip(child, deps, isBound) ||
        !enclosingTripProduct(child, deps, isBound))
      return false;
  }
  return true;
}

namespace {
struct SdcSchedulingPass
    : public allo::impl::SdcSchedulingPassBase<SdcSchedulingPass> {
  using SdcSchedulingPassBase::SdcSchedulingPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Timing characterization for every op (latency + delays): a YAML library
    // from the `operator-library` option, else the built-in default. Loaded
    // once and shared by scheduling and reification.
    OperatorLibrary loadedLib;
    if (!operatorLibrary.empty()) {
      auto parsed = OperatorLibrary::loadFile(operatorLibrary);
      if (!parsed) {
        error(Stage::Sched, module) << llvm::toString(parsed.takeError());
        return signalPassFailure();
      }
      loadedLib = std::move(*parsed);
    } else {
      loadedLib = OperatorLibrary::defaultLibrary();
    }

    // Fail loudly if an advanced (raw-name) row names an op we cannot express
    // (a typo, or a dialect this pass does not load) rather than silently
    // ignoring the row.
    if (std::vector<std::string> bad =
            loadedLib.unregisteredAdvancedOps(*module.getContext());
        !bad.empty()) {
      error(Stage::Sched, module) << "operator library names unregistered "
                                     "op(s): "
                                  << llvm::join(bad, ", ");
      return signalPassFailure();
    }

    // Target clock period: the option overrides the library, else 5.0 ns.
    float cycleTimeNs =
        cycleTime > 0.0f ? cycleTime : loadedLib.cycleTime().value_or(5.0f);

    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
    IRRewriter r(&getContext());
    for (auto funcOp : funcs)
      if (!funcOp.isExternal())
        if (failed(scheduleFunc(r, funcOp, loadedLib, cycleTimeNs)))
          return signalPassFailure();

    // The schedule is emitted as the `allo.sched.*` carrier only; the pipeline
    // chains `convert-schedule-to-dcp` to reify it into `allo.dcp.*` ops.
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<allo::AlloDialect, arith::ArithDialect, func::FuncDialect,
                    math::MathDialect, affine::AffineDialect, scf::SCFDialect,
                    memref::MemRefDialect>();
  }

  // Phase B: pipeline an imperfect level over its children. Schedule the child
  // loops (only -- the leaf ops go into the level problem), then solve the
  // level as one modulo problem (each child loop a non-pipelined resource node)
  // so the outer loop pipelines at II_outer = max(child occupancy, recurrence
  // II), overlapping the leaf ops and consecutive iterations. The leaf ops and
  // the level descriptor are annotated; the child loops keep their own regions
  // (materialized as nested dcp.pipelines). Precondition: the caller entered
  // only after `levelIsPhaseBReifiable`, so every child is a single counted
  // loop with a known latency and the level problem is guaranteed
  // constructible.
  LogicalResult scheduleLevelPipelined(LoopLikeOpInterface level,
                                       unsigned &nextId,
                                       DependenceAnalysis &deps,
                                       const OperatorLibrary &lib,
                                       func::FuncOp funcOp, float cycleTimeNs) {
    Block &body = level.getLoopRegions().front()->front();
    for (const SchedRegion &sub : enumerateRegions(body))
      if (sub.kind == allo::RegionKind::Loop)
        if (failed(scheduleRegion(sub, nextId, deps, lib, funcOp, cycleTimeNs)))
          return failure();

    LevelAnalysis a = analyzeLevel(level, deps);
    logLevelAnalysis(a, level);
    {
      unsigned loops = 0;
      for (const LevelNode &n : a.nodes)
        loops += n.isLoop;
      info(Stage::Sched, level.getOperation())
          << "  level analysis: " << a.nodes.size() << " node(s) ("
          << (a.nodes.size() - loops) << " leaf op(s) + " << loops
          << " inner loop(s)), " << a.edges.size()
          << " dependence edge(s) constrain their overlap";
    }
    ChainingModuloProblem problem(level.getOperation());
    int64_t resourceBound = 0;
    SmallVector<Operation *> tagged;
    if (!buildLevelProblem(problem, a, level, deps, lib, funcOp, resourceBound,
                           tagged)) {
      error(Stage::Sched, level.getOperation())
          << "Phase B level problem construction failed unexpectedly";
      signalPassFailure();
      return failure();
    }

    Operation *anchor = body.getTerminator();
    LogicalResult solved =
        solveSchedulingProblem(problem, anchor, cycleTimeNs, /*minII=*/1);
    if (succeeded(solved)) {
      unsigned ii = problem.getInitiationInterval().value_or(1);
      bool isBound = false;
      std::optional<int64_t> latency = regionLatency(
          level.getOperation(), ii, scheduleDepth(problem), deps, isBound);
      int64_t levelId = nextId++;
      // Tag the level loop so the reify materializes it into the outer
      // dcp.pipeline (its children are materialized first as nested pipelines).
      level->setAttr(sched::kLevelAttr,
                     Builder(level.getContext()).getI64IntegerAttr(levelId));
      annotateRegion(problem, funcOp, levelId, "cyclic", ii, latency, isBound);
      annotateStartTimeInCycle(problem);
      absorbLevelChildren(funcOp, level, levelId, problem, a);
      {
        auto d = info(Stage::Sched, level.getOperation());
        d << "Phase B: outer loop pipelined at II=" << ii << " (";
        if (ii == static_cast<unsigned>(resourceBound))
          d << "the busiest inner loop's occupancy, " << resourceBound
            << " cycles";
        else
          d << "above the " << resourceBound
            << "-cycle inner-loop occupancy, raised by a level-carried "
               "recurrence";
        d << "); surrounding ops and consecutive outer iterations overlap";
      }
    }
    for (Operation *op : tagged)
      op->removeAttr(sched::kResourceCyclesAttr);
    if (failed(solved)) {
      error(Stage::Sched, level.getOperation())
          << "pipelined nest not scheduled";
      signalPassFailure();
      return failure();
    }
    return success();
  }

  // Schedule one region: a straight-line span as an acyclic problem, a counted
  // loop as a cyclic problem. An imperfect counted nest -- whose innermost band
  // body still holds loops (sibling loops, or ops surrounding an inner loop) --
  // is either pipelined over its children (Phase B, when the outer loop carries
  // an explicit pipeline directive) or decomposed into per-body sub-regions
  // (Phase A), the band loops staying as wrapper loops whose trips fold into
  // each sub-region's latency via `enclosingTripProduct`. `nextId` hands out
  // region ids in program order so the reify's prefix-sum composition stays
  // correct. Cross-region composition is by program order / SSA only.
  LogicalResult scheduleRegion(SchedRegion region, unsigned &nextId,
                               DependenceAnalysis &deps,
                               const OperatorLibrary &lib, func::FuncOp funcOp,
                               float cycleTimeNs) {
    if (region.kind != allo::RegionKind::Loop) {
      // An all-constant span is a tie-off the reify leaves in place (no
      // latency, no materialized region); scheduling it would cost a spurious
      // region and desync the whole-kernel latency.
      if (llvm::all_of(region.ops, [](Operation *op) {
            return isa<arith::ConstantOp>(op);
          }))
        return success();
      region.id = nextId++;
      info(Stage::Sched, region.anchor())
          << "Region " << region.id << " is a straight-line span of "
          << region.ops.size() << " op(s), using acyclic scheduling";
      return scheduleAcyclic(region.ops, deps, lib, funcOp, region,
                             cycleTimeNs);
    }
    if (isa<AffineForOp, scf::ForOp>(region.anchor())) {
      LoopLikeOpInterface innermost =
          perfectNest(cast<LoopLikeOpInterface>(region.anchor())).back();
      int64_t dir =
          pipelineDirective(innermost.getOperation(), region.anchor());
      if (hasNestedLoop(innermost)) {
        if (dir >= 1 && levelIsPhaseBReifiable(innermost, deps)) {
          info(Stage::Sched, innermost.getOperation())
              << "Detected imperfect nest with pipeline directives, "
                 "fusing the outer loop over its inner loop(s) as one modulo "
                 "problem (each inner loop as a fixed-latency resource node)";
          return scheduleLevelPipelined(innermost, nextId, deps, lib, funcOp,
                                        cycleTimeNs);
        }
        if (dir >= 1)
          warn(Stage::Sched, innermost.getOperation())
              << "Pipelined imperfect nest not fused into one pipeline. "
                 "Requires single counted inner loops with known trips and no "
                 "outer iter_args); scheduling its body as sequential "
                 "sub-regions";
        info(Stage::Sched, innermost.getOperation())
            << "Detected imperfect nest, decomposing into sub-regions "
               "scheduled in program order.";
        Block &body = innermost.getLoopRegions().front()->front();
        if (failed(scheduleBlock(body, nextId, deps, lib, funcOp, cycleTimeNs)))
          return failure();
        if (logging::detail::enabled(logging::Level::Debug)) {
          LevelAnalysis la = analyzeLevel(innermost, deps);
          logLevelAnalysis(la, innermost);
          logLevelII(la, innermost, deps, lib, funcOp, cycleTimeNs);
        }
        return success();
      }
      region.id = nextId++;
      {
        auto d = info(Stage::Sched, innermost.getOperation());
        d << "Region " << region.id << " detected as a for-loop";
        if (unsigned band =
                perfectNest(cast<LoopLikeOpInterface>(region.anchor())).size();
            band > 1)
          d << " (perfect band of " << band << " levels)";
        if (dir == -1)
          d << ", pipelining disabled";
        else if (dir >= 1)
          d << ", target II=" << dir;
        d << ", using modulo-scheduling in the innermost body";
      }
      return scheduleCyclic(innermost, deps, lib, funcOp, region, cycleTimeNs,
                            dir >= 1 ? static_cast<unsigned>(dir) : 1,
                            /*pipelined=*/dir != -1);
    }
    // A while with an all-straight-line body schedules as a flushing pipeline
    // (before + after as one iteration). A while whose body contains a nested
    // loop cannot: a flat problem would flatten the inner loop's ops into the
    // outer iteration space. Instead decompose the body Phase-A-style -- the
    // nested loop schedules as its own region, the surrounding ops as acyclic
    // spans -- and leave the outer `scf.while` raw: it runs sequentially, which
    // is correct because its data-dependent trip already leaves the latency
    // unknown and the carried state typically threads the inner loop. (A
    // counted while was raised to scf.for by `raise-counted-while`.)
    if (auto whileOp = dyn_cast<scf::WhileOp>(region.anchor())) {
      if (hasNestedLoop(whileOp)) {
        info(Stage::Sched, whileOp.getOperation())
            << "Detected while loop with a nested loop; decomposing its body "
               "into sub-regions scheduled in program order (the outer while "
               "runs sequentially, latency data-dependent)";
        return scheduleBlock(whileOp.getAfter().front(), nextId, deps, lib,
                             funcOp, cycleTimeNs);
      }
      if (!whileHasIdentityForwarding(whileOp)) {
        error(Stage::Sched, whileOp.getOperation())
            << "scf.while not scheduled";
        signalPassFailure();
        return failure();
      }
      region.id = nextId++;
      info(Stage::Sched, whileOp.getOperation())
          << "Region " << region.id
          << " detected as a while-loop, using flushing-pipeline schedule";
      return scheduleWhile(whileOp, deps, lib, funcOp, region, cycleTimeNs);
    }
    // A surviving `if` -- one that `fold-if-statements` could not predicate
    // because a branch holds a loop / stream / call -- is kept as a control
    // construct: decompose each branch body Phase-A-style (a guarded loop
    // becomes its own region, surrounding ops acyclic spans) and leave the `if`
    // raw, wrapping the materialized branch regions. Same
    // decompose-and-leave-raw shape as the nested-loop while above; the reifier
    // leaves the untagged `if` in place around its materialized children.
    if (isa<AffineIfOp, scf::IfOp>(region.anchor())) {
      Operation *ifOp = region.anchor();
      info(Stage::Sched, ifOp)
          << "Detected a conditional left opaque by if-conversion; decomposing "
             "each branch into sub-regions and keeping the `if` as a guard";
      for (Region &branch : ifOp->getRegions())
        if (!branch.empty())
          if (failed(scheduleBlock(branch.front(), nextId, deps, lib, funcOp,
                                   cycleTimeNs)))
            return failure();
      return success();
    }
    error(Stage::Sched, region.anchor()) << "loop not scheduled";
    signalPassFailure();
    return failure();
  }

  LogicalResult scheduleBlock(Block &block, unsigned &nextId,
                              DependenceAnalysis &deps,
                              const OperatorLibrary &lib, func::FuncOp funcOp,
                              float cycleTimeNs) {
    for (const SchedRegion &region : enumerateRegions(block))
      if (failed(
              scheduleRegion(region, nextId, deps, lib, funcOp, cycleTimeNs)))
        return failure();
    return success();
  }

  // Solve and annotate the schedule of one function.
  LogicalResult scheduleFunc(RewriterBase &b, func::FuncOp funcOp,
                             const OperatorLibrary &lib, float cycleTimeNs) {
    std::string infoStr =
        "-- Start scheduling for " + funcOp.getSymName().str();
    info(Stage::Sched) << std::string(infoStr.size() * 2, '-');
    info(Stage::Sched) << infoStr;
    info(Stage::Sched) << std::string(infoStr.size() * 2, '-');

    // Whole-func memory + stream dependence analysis, refined by the
    // `allo.assume.*` hints.
    DependenceAnalysis deps(funcOp);

    // Erase the consumed hint ops: they carry no schedulable computation and
    // would otherwise perturb the problem.
    SmallVector<Operation *, 4> hints;
    funcOp.walk([&](Operation *op) {
      if (isa<AssumeNoDepOp, AssumeSSAOp>(op))
        hints.push_back(op);
    });
    for (Operation *op : hints)
      eraseHintAndDeadInputs(b, op);

    // Schedule the function body's regions, recursing into imperfect nests.
    // Region ids are handed out in program order for the reify's latency
    // prefix sum.
    unsigned nextId = 0;
    if (failed(scheduleBlock(funcOp.getBody().front(), nextId, deps, lib,
                             funcOp, cycleTimeNs)))
      return failure();

    annotateKernelLatency(funcOp);
    return success();
  }
};
} // namespace
