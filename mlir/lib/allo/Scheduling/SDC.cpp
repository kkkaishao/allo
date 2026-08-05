/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/LatencyModel.h"
#include "allo/Scheduling/MemoryModel.h" // kIndexWidth
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/ScheduleModel.h"
#include "allo/Scheduling/Scheduler.h"
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
#include "llvm/Support/MathExtras.h"

#include <chrono>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

// Erase a consumed hint op along with any operand-producing ops it leaves
// trivially dead. The assert guards against a freed op's address being reused
// by the next `create`, which would alias a stale key in \p ranges.
static void
eraseHintAndDeadInputs(RewriterBase &b, Operation *op,
                       const DenseMap<Value, AssumedRange> &ranges) {
  SmallVector<Value, 4> operands(op->getOperands());
  b.eraseOp(op);
  for (Value v : operands)
    if (Operation *def = v.getDefiningOp())
      if (isOpTriviallyDead(def)) {
        assert(llvm::none_of(def->getResults(),
                             [&](Value r) { return ranges.count(r); }) &&
               "erasing a value the assumed-range map is keyed by");
        eraseHintAndDeadInputs(b, def, ranges);
      }
}

// The maximal perfect band of counted loops (affine.for / scf.for) rooted at
// \p root: descend while a level's body is exactly { inner counted loop,
// terminator }. Returns [root, ..., innermost].
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

// The region's outputs, as the terms whose max is its terminal cycle: how long
// after its last issue pulse the deepest output commits. Left as separate terms
// so the exact scheduler can bound a variable by each one and minimize the
// charged quantity; `drainOf` takes the max, after the solve.
//
// Each output is charged at the cycle the emitter commits it: a store presents
// at its start and commits `writeLatency` cycles later, a sync sub-kernel call
// charges the same way (its `done` rises at its start plus its contract), a
// stream put commits at its stage, and a value handed onward is latched the
// cycle it lands, one cycle above a store presented at the same depth.
//
// \p results are the values escaping the region. One only forwarded (a block
// argument, an earlier region's survivor, or a declaration) charges nothing: it
// is settled before the region starts or binds no hardware to wait on.
static SmallVector<DrainTerm> drainTerms(OccupancyProblem &problem,
                                         ValueRange results) {
  SmallVector<DrainTerm> terms;
  for (Operation *op : problem.getOperations()) {
    if (isa<AffineStoreOp, memref::StoreOp>(op) || isSyncSubKernelCall(op))
      terms.push_back({op, problem.latencyOf(op) - 1});
    else if (isa<StreamPutOp>(op))
      terms.push_back({op, 0});
  }
  for (Value v : results) {
    Operation *def = v.getDefiningOp();
    // A call's result is the one escaping value not read through a capture
    // register of this region: the region's `done` is the child's, charged by
    // the loop above, and the consumer's own arming cycle pays the latch.
    if (!def || isDeclarationOp(def) || isSyncSubKernelCall(def) ||
        !problem.hasOperation(def))
      continue;
    terms.push_back({def, problem.latencyOf(def)});
  }
  return terms;
}

// The flip-flops one cycle of delay on \p type costs, or 0 for a value not
// carried in a register at all (a memref, a stream). An index is charged at
// `kIndexWidth`, an upper bound since the emitter may build that address
// register narrower; charging it zero would let the solver lengthen an address
// chain for free.
static int64_t registerWidth(Type type) {
  if (auto i = dyn_cast<IntegerType>(type))
    return i.getWidth();
  if (auto f = dyn_cast<FloatType>(type))
    return f.getWidth();
  if (isa<IndexType>(type))
    return kIndexWidth;
  return 0;
}

// The values a region spends a delay register on, and what each one charges:
// mirrors `DatapathBuilder::resolveOperand` + `insertRegister`, stated over
// the problem so a solve can minimize the same quantity the emitter spends. Two
// kinds are charged: a scheduled producer read in the same region (a def-use
// edge in the problem), and a loop-carried read of an iter_arg, the same edge
// `distance` iterations back. A value held longer than the region (a survivor,
// an IO port, a literal) is free and is defined by no op in the problem, so it
// falls through. An enclosing loop's counter and the activation-pulse chain are
// not charged here, both left to the objective's sum-of-starts tie-break.
//
// \p carried is the counted-loop body whose block arguments after the
// induction variable are its iter_args, or null where there is no such
// recurrence to price (a straight-line span, a `while`).
static SmallVector<RegisterTerm>
registerTerms(OccupancyProblem &problem, Block *carried) {
  SmallVector<RegisterTerm> terms;
  DenseMap<Value, unsigned> slotOf;
  auto readBy = [&](Value v, Operation *def, Operation *reader,
                    int64_t distance) {
    int64_t width = registerWidth(v.getType());
    if (width == 0)
      return;
    auto [slot, isNew] = slotOf.try_emplace(v, terms.size());
    if (isNew)
      terms.push_back({def, problem.latencyOf(def), width, {}});
    terms[slot->second].reads.push_back({reader, distance});
  };

  for (Operation *reader : problem.getOperations()) {
    // A terminator takes no input register: the values it hands on are latched
    // by the region's completion, not delayed into it.
    if (reader->hasTrait<OpTrait::IsTerminator>())
      continue;
    for (auto &dep : problem.getDependences(reader))
      if (dep.isDefUse())
        readBy(dep.getSource()->getResult(*dep.getSourceIndex()),
               dep.getSource(), reader, /*distance=*/0);
  }

  if (!carried)
    return terms;
  Operation *yield = carried->getTerminator();
  for (unsigned i = 0, n = yield->getNumOperands(); i < n; ++i) {
    auto [def, distance] = iterArgSource(carried, yield, i);
    if (!def || !problem.hasOperation(def))
      continue;
    for (Operation *reader : carried->getArgument(i + 1).getUsers())
      if (problem.hasOperation(reader))
        readBy(def->getResult(0), def, reader, distance);
  }
  return terms;
}

// Whether the problem carries a loop-carried recurrence (a dependence spanning
// >= 1 iteration), which can hold the modulo II above the resource bound.
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
// or nullopt if still unbounded.
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

// The trip a region's own solution records: that of the innermost loop of the
// band it anchors, the one its solved `length`/`ii` describe. Every loop above
// it drives its child as a container, composed in `buildSpanNode`.
static std::optional<int64_t>
regionTrip(Operation *anchor, DependenceAnalysis &deps, bool &isBound) {
  return loopTrip(perfectNest(cast<LoopLikeOpInterface>(anchor)).back(), deps,
                  isBound);
}

// The values a straight-line span hands to something outside itself. Must match
// what the reify treats as escaping, so the two agree on what the region's
// completion waits to capture.
static SmallVector<Value> spanEscapingValues(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 16> inSpan(ops.begin(), ops.end());
  SmallVector<Value> escaping;
  for (Operation *op : ops)
    for (Value res : op->getResults())
      if (llvm::any_of(res.getUsers(),
                       [&](Operation *user) { return !inSpan.contains(user); }))
        escaping.push_back(res);
  return escaping;
}

/// Record the solved schedule of one region in \p model: every registered op's
/// start cycle and sub-cycle start, plus the region's own solution keyed by
/// \p owner, the op both descents land on: a counted band's innermost loop, a
/// flushing `scf.while`, or a straight-line span's first op.
template <class ProblemT>
static void annotateRegion(ProblemT &problem,
                           ScheduleModel &model, Operation *owner,
                           std::optional<int64_t> ii,
                           std::optional<int64_t> trip, bool tripIsBound,
                           int64_t drain) {
  for (Operation *op : problem.getOperations()) {
    std::optional<unsigned> start = problem.getStartTime(op);
    if (!start)
      continue;
    // A child loop is scheduled as its own region and a terminator carries no
    // compute. Neither is recorded, though both count toward the length.
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>())
      continue;
    model.setStart(op, *start);
    if (std::optional<float> z = problem.getStartTimeInCycle(op))
      model.setStartInCycle(op, *z);
  }
  RegionSolution &r = model.addRegion(owner);
  r.ii = ii;
  // Both are per-invocation: no composed total is stored.
  r.length = problem.scheduleDepth();
  r.drain = drain;
  r.trip = trip;
  r.tripIsBound = tripIsBound;
}

// Publish the solved allocation into \p model: one entry per instance the
// region builds, and the instance each operation runs on. Every operation on an
// allocated resource carries one: `applyAllocation` derives them alongside the
// counts it sets, and `verifyAllocation` has already failed the solve where one
// is missing.
static void annotateAllocation(OccupancyProblem &problem, ScheduleModel &model,
                               const OperatorLibrary &lib) {
  for (circt::scheduling::Problem::ResourceType rsrc :
       problem.getResourceTypes()) {
    std::optional<unsigned> units = problem.getAllocation(rsrc);
    if (!units)
      continue;
    SmallVector<Operation *> users = problem.usersOf(rsrc);
    assert(!users.empty() && "an allocated resource nothing runs on");
    // One resource is one operator identity, so every operation on it names
    // the same `dcp.operator`.
    unsigned base =
        model.addUnits(lib.lookup(users.front()).identity.realization, *units);
    for (Operation *op : users)
      model.setUnit(op, base + *problem.getAssignedUnit(op));
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

// Reserve a limit-1 resource, held for `latency + 1` cycles, for every sync
// sub-kernel call in a loop body: it is one child instance re-fired per
// iteration, not a pipelined operator, and the loop controller starts the next
// invocation on the previous one's `done` plus the cycle it takes to re-arm.
// Keyed per callsite, since distinct calls are distinct instances.
static void populateCallOccupancy(Block &body, ChainingModuloProblem &problem) {
  using P = circt::scheduling::Problem;
  unsigned idx = 0;
  for (Operation &op : body) {
    std::optional<std::pair<int64_t, std::string>> cl =
        scheduledCallLatency(&op);
    if (!cl)
      continue;
    P::ResourceType rsrc =
        problem.getOrInsertResourceType(cl->second + "#" + std::to_string(idx));
    problem.setLimit(rsrc, 1);
    problem.setLinkedResourceTypes(&op, SmallVector<P::ResourceType>{rsrc});
    problem.setResourceCycles(&op, cl->first + 1);
    ++idx;
  }
}

// A steady-clock stopwatch for timing one solve.
using Stopwatch = std::chrono::steady_clock::time_point;
static Stopwatch now() { return std::chrono::steady_clock::now(); }

// Record what one region's solve cost, timed from \p since. Keyed by where the
// region is rather than by the op that owned it: the schedule report is built
// later off the reified dcp ops, by which time this problem's loop is gone.
//
// \p ii is what the solve decided, which for a non-pipelined loop is not the
// interval the region is reported to run at (that is `annotateRegion`'s).
static void recordSolve(ScheduleModel &model, OccupancyProblem &problem,
                        StringRef kind, std::optional<unsigned> ii,
                        Stopwatch since) {
  SolveReport s;
  Operation *containing = problem.getContainingOp();
  if (auto fn = containing->getParentOfType<func::FuncOp>())
    s.func = fn.getSymName().str();
  s.where = logging::detail::describe(containing);
  s.kind = kind.str();
  s.ops = (int64_t)problem.getOperations().size();
  for (Operation *op : problem.getOperations())
    if (problem.holdsLimitedUnit(op))
      ++s.limitedOps;
  // Present only when an exact solve decided an allocation. The ceiling is what
  // the trivial allocation would have built.
  for (circt::scheduling::Problem::ResourceType rsrc :
       problem.getResourceTypes())
    if (std::optional<unsigned> units = problem.getAllocation(rsrc)) {
      s.allocatedUnits += *units;
      s.allocatedOps += problem.getAllocatable(rsrc)->ceiling;
    }
  if (ii)
    s.ii = (int64_t)*ii;
  s.millis = std::chrono::duration<double, std::milli>(now() - since).count();
  model.solves.push_back(std::move(s));
}

// Schedule one counted loop body (affine.for or scf.for) as a
// `ChainingModuloProblem` and annotate the result (start times, II, sub-cycle
// times). \p minII lower-bounds the II. When \p pipelined is false iterations
// do not overlap: the II is reported as the body length, so the region latency
// folds to `trip * depth`, and it still reifies to a dcp.pipeline.
static LogicalResult
scheduleCyclic(LoopLikeOpInterface body, DependenceAnalysis &deps,
               const OperatorLibrary &lib, ScheduleModel &model,
               const SchedRegion &region, float cycleTime, unsigned minII,
               bool pipelined, const SchedulerOptions &opts) {
  auto problem = buildCyclicProblem<ChainingModuloProblem>(body, deps);
  Block *bodyBlock = &body.getLoopRegions().front()->front();
  if (failed(populateOperatorTypes(*bodyBlock, problem, lib)))
    return failure();
  reportOperatorClassSplit(problem, lib);
  if (failed(populateMemoryResources(*bodyBlock, problem)))
    return failure();
  if (opts.allocate &&
      failed(populateOperatorAllocation(*bodyBlock, problem, lib)))
    return failure();
  populateCallOccupancy(*bodyBlock, problem);
  Operation *anchor = bodyBlock->getTerminator();
  bool isBound = false;
  std::optional<int64_t> trip = regionTrip(region.anchor(), deps, isBound);
  // A counted loop hands its carried next-values on: the terminator's operands.
  SmallVector<DrainTerm> outputs = drainTerms(problem, anchor->getOperands());
  SmallVector<RegisterTerm> regs = registerTerms(problem, bodyBlock);
  // The trip is withheld where iterations do not overlap: `ii` is the body
  // depth there, so depth, not drain, is what the trip multiplies.
  SpanObjective span{outputs, regs, pipelined ? trip : std::nullopt, &lib};
  Stopwatch solveStart = now();
  if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII, opts,
                                    span)))
    return failure();
  recordSolve(model, problem, "cyclic", problem.getInitiationInterval(),
              solveStart);
  int64_t depth = problem.scheduleDepth();
  unsigned ii = pipelined ? problem.getInitiationInterval().value_or(depth)
                          : static_cast<unsigned>(depth);
  int64_t drain = drainOf(problem, outputs);
  // For the report only, through the arithmetic that composes it for real.
  SpanNode node;
  node.trip = trip;
  node.drain = drain;
  node.ii = ii;
  std::optional<int64_t> latency = composeSpan(node);

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

  // A non-pipelined multi-cycle operator holds its unit for its whole latency,
  // so it caps iteration overlap. Name the dominant one to explain II > 1.
  if (pipelined && ii > 1) {
    Operation *blocking = nullptr;
    unsigned maxOcc = 1;
    for (Operation *op : problem.getOperations())
      if (unsigned occ = problem.getResourceCycles(op); occ > maxOcc) {
        maxOcc = occ;
        blocking = op;
      }
    if (blocking)
      info(Stage::Sched, blocking)
          << "Operator " << blocking->getName().getStringRef()
          << " is non-pipelined and holds its unit for " << maxOcc
          << " cycle(s), limiting iteration overlap";
  }

  annotateRegion(problem, model, body.getOperation(), ii, trip, isBound, drain);
  annotateAllocation(problem, model, lib);
  return success();
}

// Schedule an uncounted `scf.while` (before + after as one iteration) as a
// `ChainingModuloProblem`, the flushing-pipeline scheduling view. Its trip
// count is data-dependent, so no latency is reported.
static LogicalResult scheduleWhile(scf::WhileOp w, DependenceAnalysis &deps,
                                   const OperatorLibrary &lib,
                                   ScheduleModel &model,
                                   const SchedRegion &region, float cycleTime,
                                   const SchedulerOptions &opts) {
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
  reportOperatorClassSplit(problem, lib);
  if (failed(populateMemoryResourcesImpl(problem, [&](auto handle) {
        w.getBefore().walk(handle);
        w.getAfter().walk(handle);
      })))
    return failure();
  Operation *anchor = w.getYieldOp().getOperation();
  // Honor a requested target II (>=1) as a lower bound. `ii=-1` (pipelining
  // off) is not modeled for while loops.
  int64_t dir = pipelineDirective(w, region.anchor());
  unsigned minII = dir >= 1 ? static_cast<unsigned>(dir) : 1;
  SmallVector<DrainTerm> outputs = drainTerms(problem, anchor->getOperands());
  // A while's state recurrence is a register this does not price: its carried
  // values are not a counted loop's iter_args, so `registerTerms` is handed no
  // body to read them off.
  SmallVector<RegisterTerm> regs = registerTerms(problem, /*carried=*/nullptr);
  Stopwatch solveStart = now();
  // No trip, so no span is minimized: the objective stays the anchor's start
  // time.
  if (failed(
          solveSchedulingProblem(problem, anchor, cycleTime, minII, opts,
                                 SpanObjective{outputs, regs, std::nullopt,
                                               &lib})))
    return failure();
  std::optional<unsigned> ii = problem.getInitiationInterval();
  recordSolve(model, problem, "while", ii, solveStart);
  info(Stage::Sched, w.getOperation())
      << "  -> While loop scheduled as a flushing pipeline: II="
      << ii.value_or(0)
      << " (trip is data-dependent, so whole-loop latency is unknown)";
  // The trip is data-dependent, so no span composes off this drain: it is
  // recorded, like `ii`, as what the solve decided.
  annotateRegion(problem, model, w.getOperation(),
                 ii ? std::optional<int64_t>(*ii) : std::nullopt,
                 /*trip=*/std::nullopt, /*tripIsBound=*/false,
                 drainOf(problem, outputs));
  return success();
}

// Schedule one straight-line region as a `ChainingSharedOperatorsProblem` and
// annotate the result.
static LogicalResult scheduleAcyclic(ArrayRef<Operation *> ops,
                                     DependenceAnalysis &deps,
                                     const OperatorLibrary &lib,
                                     ScheduleModel &model, float cycleTime,
                                     const SchedulerOptions &opts) {
  ChainingSharedOperatorsProblem problem =
      buildAcyclicProblem<ChainingSharedOperatorsProblem>(ops, deps);
  if (failed(populateOperatorTypes(ops, problem, lib)))
    return failure();
  reportOperatorClassSplit(problem, lib);
  if (failed(populateMemoryResources(ops, problem)))
    return failure();
  if (opts.allocate && failed(populateOperatorAllocation(ops, problem, lib)))
    return failure();
  // A straight-line region runs once, so its whole cost is its drain, and it
  // carries nothing between iterations it does not have.
  SmallVector<DrainTerm> outputs = drainTerms(problem, spanEscapingValues(ops));
  SmallVector<RegisterTerm> regs = registerTerms(problem, /*carried=*/nullptr);
  Stopwatch solveStart = now();
  if (failed(solveSchedulingProblem(problem, ops.back(), cycleTime, opts,
                                    SpanObjective{outputs, regs, /*trip=*/1,
                                                  &lib})))
    return failure();
  recordSolve(model, problem, "acyclic", /*ii=*/std::nullopt, solveStart);
  info(Stage::Sched, ops.front())
      << "Scheduled: depth = " << problem.scheduleDepth() << " cycles";
  // How often its enclosing loops re-run it is charged where they are composed.
  annotateRegion(problem, model, ops.front(), /*ii=*/std::nullopt,
                 /*trip=*/std::nullopt, /*tripIsBound=*/false,
                 drainOf(problem, outputs));
  annotateAllocation(problem, model, lib);
  return success();
}

static std::optional<SpanNode> buildSpanNode(const SchedRegion &region,
                                             ScheduleModel &model,
                                             DependenceAnalysis &deps,
                                             bool &isBound);

// The body elements of a container, in program order.
static std::vector<SpanNode> buildSpanNodes(Block &body, ScheduleModel &model,
                                            DependenceAnalysis &deps,
                                            bool &isBound) {
  std::vector<SpanNode> nodes;
  for (const SchedRegion &child : enumerateRegions(body))
    if (std::optional<SpanNode> n = buildSpanNode(child, model, deps, isBound))
      nodes.push_back(std::move(*n));
  return nodes;
}

// One scheduling region as the latency model sees it, walked over the
// affine/scf loops; `PostConversion.cpp` walks the dcp regions built from those
// same loops, and both feed `composeSpan`.
//
// Descends the loop nest, not the solution list: one solution covers a whole
// perfect band, while the emitter drives every loop above the innermost as a
// container with its own boundary cycles, which a flat walk of solutions has
// nowhere to charge.
//
// nullopt means the region occupies no cycles and forms no node (a
// straight-line span of nothing but declarations). A data-dependent region
// still forms a node, with the unknown left in its own fields.
static std::optional<SpanNode> buildSpanNode(const SchedRegion &region,
                                             ScheduleModel &model,
                                             DependenceAnalysis &deps,
                                             bool &isBound) {
  SpanNode n;
  // Driven by an enclosing region rather than by the func's own sequencer, the
  // same question the reify side asks of a dcp op's parents.
  n.nested = !isa<func::FuncOp>(region.anchor()->getParentOp());
  n.elastic =
      llvm::any_of(region.ops, [](Operation *o) { return isElastic(o); });
  if (region.kind == allo::RegionKind::StraightLine) {
    if (!spanFormsRegion(region.ops))
      return std::nullopt;
    n.acyclic = true;
    n.trip = 1;
    if (RegionSolution *sol = model.regionOf(region.ops.front()))
      n.drain = sol->drain;
    return n;
  }
  Operation *anchor = region.anchor();
  // An `if` if-conversion left opaque runs under a predicate, which becomes a
  // `dcp.select` the reify side reads back as a Guard.
  if (isa<AffineIfOp, scf::IfOp>(anchor)) {
    n.shape = RegionShape::Guard;
    return n;
  }
  if (!isa<AffineForOp, scf::ForOp>(anchor))
    return n; // a while: a data-dependent trip, so no static span
  auto loop = cast<LoopLikeOpInterface>(anchor);
  n.trip = loopTrip(anchor, deps, isBound);
  n.shape = countedLoopShape(loop);
  Block &body = loop.getLoopRegions().front()->front();

  if (n.shape == RegionShape::CallNode) {
    // The body is one instance the controller re-fires per iteration, so a pass
    // costs the callee's own start to done contract and nothing else.
    for (Operation &op : body) {
      if (!isSyncSubKernelCall(&op))
        continue;
      Operation *callee = calleeOf(&op);
      SpanNode child;
      child.instance = true;
      child.contract = callee ? calleeStaticLatency(callee) : std::nullopt;
      n.children.push_back(std::move(child));
    }
    return n;
  }
  // A container owns no solution: it sequences the regions its body decomposed
  // into, and its span is composed from theirs.
  if (n.shape == RegionShape::Container) {
    n.children = buildSpanNodes(body, model, deps, isBound);
    return n;
  }
  // A leaf nests no loop, so it is the op the solve was keyed by.
  if (RegionSolution *sol = model.regionOf(anchor)) {
    n.drain = sol->drain;
    n.ii = sol->ii;
  }
  return n;
}

// Record every counted loop whose iteration count only an `allo.assume.ssa`
// range bounds, for the reify to stamp as `trip_bound` and the emitter to size
// its counter by. This is the one fact reification cannot re-derive: the hint
// that bounded a symbolic trip is already consumed and erased by the time reify
// runs, unlike a loop's lb/step/constant trip, which stay on the loop.
static void recordTripBounds(func::FuncOp funcOp, ScheduleModel &model,
                             DependenceAnalysis &deps) {
  funcOp.walk([&](Operation *op) {
    if (!isa<AffineForOp, scf::ForOp>(op))
      return;
    bool isBound = false;
    std::optional<int64_t> trip = loopTrip(op, deps, isBound);
    if (isBound && trip)
      model.setTripBound(op, *trip);
  });
}

// Compose the solved region tree into one whole-kernel span, and publish it.
// The only thing the scheduler writes to the IR, and the only thing a caller of
// this kernel sees. Sets the attribute only when every region has a known span.
//
// The span is the top-level regions composed over their dependence DAG, and
// must equal what the reify's `setDcpLatencies` composes off the dcp regions
// built from these. Independent siblings overlap, so it is the longest path and
// not the sum.
static void publishKernelLatency(func::FuncOp funcOp, ScheduleModel &model,
                                 DependenceAnalysis &deps) {
  Builder b(funcOp.getContext());

  // A callee whose own length is data-dependent leaves this kernel's unknown.
  // Must be asked here: the operator library prices an uncharacterized call at
  // zero, so the composition alone would omit it.
  bool callsKnown = true;
  funcOp.walk([&](func::CallOp call) {
    if (call->hasAttr(kAlloAsyncAttr))
      return;
    Operation *callee = calleeOf(call);
    if (!callee || !calleeStaticLatency(callee))
      callsKnown = false;
  });
  if (!callsKnown)
    return;

  bool isBound = false;
  std::vector<SpanNode> top;
  SmallVector<SmallVector<Operation *>> topOps;
  for (const SchedRegion &region : enumerateRegions(funcOp))
    if (std::optional<SpanNode> n =
            buildSpanNode(region, model, deps, isBound)) {
      top.push_back(std::move(*n));
      topOps.emplace_back(region.ops.begin(), region.ops.end());
    }
  std::optional<int64_t> total = composeDag(top, siblingPredecessors(topOps));
  if (!total)
    return; // a data-dependent region leaves the kernel total unknown

  // Only the number is published, not whether it is a bound: a bound is an
  // upper one, so a caller placing consumers against it is safe either way.
  funcOp->setAttr(kLatencyAttr, b.getI64IntegerAttr(*total));
}

// forward declarations for the recursive scheduling functions
static LogicalResult scheduleBlock(Block &block, DependenceAnalysis &deps,
                                   const OperatorLibrary &lib,
                                   ScheduleModel &model, float cycleTimeNs,
                                   const SchedulerOptions &opts);

// Schedule one region: a straight-line span as an acyclic problem, a counted
// loop as a cyclic problem. An imperfect counted nest, whose innermost band
// body still holds loops, is decomposed into per-body sub-regions, the band
// loops staying as wrapper loops that drive those sub-regions as containers.
static LogicalResult scheduleRegion(const SchedRegion &region,
                                    DependenceAnalysis &deps,
                                    const OperatorLibrary &lib,
                                    ScheduleModel &model, float cycleTimeNs,
                                    const SchedulerOptions &opts) {
  if (region.kind != allo::RegionKind::Loop) {
    // A span of nothing but declarations is a tie-off the reify leaves in
    // place, so scheduling it costs a spurious region and lets a func with
    // nothing else publish a zero-cycle latency. THE predicate, shared with
    // the composition below and with the reify.
    if (!spanFormsRegion(region.ops))
      return success();
    info(Stage::Sched, region.anchor())
        << "A straight-line span of " << region.ops.size()
        << " op(s), using acyclic scheduling";
    return scheduleAcyclic(region.ops, deps, lib, model, cycleTimeNs, opts);
  }
  if (isa<AffineForOp, scf::ForOp>(region.anchor())) {
    LoopLikeOpInterface innermost =
        perfectNest(cast<LoopLikeOpInterface>(region.anchor())).back();
    int64_t dir = pipelineDirective(innermost.getOperation(), region.anchor());
    // The same shape query `buildSpanNode` composes through, so solving and
    // costing agree on which level drives children. Only a Container
    // decomposes; a CallNode and a Leaf run one flat cyclic problem.
    if (countedLoopShape(innermost) == RegionShape::Container) {
      // Fusing the level over its inner loops into one modulo problem is not
      // implemented: the container sequences its children and runs no schedule
      // of its own.
      if (dir >= 1)
        warn(Stage::Sched, innermost.getOperation())
            << "A pipeline directive on an imperfect nest is not honored yet; "
               "scheduling its body as sequential sub-regions. Leave "
               "`unroll_under_pipeline` at its default, which unrolls the "
               "inner loops into the pipelined level instead";
      info(Stage::Sched, innermost.getOperation())
          << "Detected imperfect nest, decomposing into sub-regions "
             "scheduled in program order.";
      Block &body = innermost.getLoopRegions().front()->front();
      return scheduleBlock(body, deps, lib, model, cycleTimeNs, opts);
    }
    {
      auto d = info(Stage::Sched, innermost.getOperation());
      d << "Detected as a for-loop";
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
    return scheduleCyclic(innermost, deps, lib, model, region, cycleTimeNs,
                          dir >= 1 ? static_cast<unsigned>(dir) : 1,
                          /*pipelined=*/dir != -1, opts);
  }
  // An uncounted while; counted ones are already scf.for.
  if (auto whileOp = dyn_cast<scf::WhileOp>(region.anchor())) {
    // A nested loop (data-dependent per-iteration length) or a condition not
    // settled at issue forces the sequential CHECK/RUN controller. The
    // reifier's routing shares `conditionIsCombinational`, so the two agree.
    if (!whileFlushingPipelines(whileOp, lib)) {
      info(Stage::Sched, whileOp)
          << "While loop cannot flushing-pipeline (nested loop, sub-kernel "
             "call, or non-combinational condition); decomposing its body "
             "into sub-regions scheduled in program order (the outer while "
             "runs sequentially, latency data-dependent)";
      return scheduleBlock(whileOp.getAfter().front(), deps, lib, model,
                           cycleTimeNs, opts);
    }
    // `verify-rtl-legality` rejects a flushing while that does not forward
    // 1:1, so `buildWhileProblem`'s slot alignment holds here.
    assert(whileHasIdentityForwarding(whileOp) &&
           "a flushing while reached scheduling without identity forwarding");
    info(Stage::Sched, whileOp.getOperation())
        << "Detected as a while-loop, using flushing-pipeline schedule";
    return scheduleWhile(whileOp, deps, lib, model, region, cycleTimeNs, opts);
  }
  // An `if` that `fold-if-statements` could not predicate stays a control
  // construct: decompose each branch into sub-regions and leave the `if` raw
  // around them.
  if (isa<AffineIfOp, scf::IfOp>(region.anchor())) {
    Operation *ifOp = region.anchor();
    info(Stage::Sched, ifOp)
        << "Detected a conditional left opaque by if-conversion; decomposing "
           "each branch into sub-regions and keeping the `if` as a guard";
    for (Region &branch : ifOp->getRegions())
      if (!branch.empty())
        if (failed(scheduleBlock(branch.front(), deps, lib, model, cycleTimeNs,
                                 opts)))
          return failure();
    return success();
  }
  error(Stage::Sched, region.anchor()) << "Loop not scheduled";
  return failure();
}

static LogicalResult scheduleBlock(Block &block, DependenceAnalysis &deps,
                                   const OperatorLibrary &lib,
                                   ScheduleModel &model, float cycleTimeNs,
                                   const SchedulerOptions &opts) {
  for (const SchedRegion &region : enumerateRegions(block))
    if (failed(scheduleRegion(region, deps, lib, model, cycleTimeNs, opts)))
      return failure();
  return success();
}

// Solve the schedule of one function into \p model, and nothing else: what the
// solved tree costs is composed by the driver's next step, off the model this
// one fills. \p deps outlives this call because that composition reads it too.
static LogicalResult scheduleFunc(func::FuncOp funcOp,
                                  const OperatorLibrary &lib,
                                  ScheduleModel &model,
                                  DependenceAnalysis &deps, float cycleTimeNs,
                                  const SchedulerOptions &opts) {
  std::string infoStr = "-- Start scheduling for " + funcOp.getSymName().str();
  info(Stage::Sched) << std::string(infoStr.size() * 2, '-');
  info(Stage::Sched) << infoStr;
  info(Stage::Sched) << std::string(infoStr.size() * 2, '-');

#ifndef NDEBUG
  // `verify-rtl-legality` rejects an access the analysis does not model, so
  // reaching here means the two disagree and this op's dependences were
  // dropped; scheduling would freely reorder it against what it aliases.
  funcOp.walk([](Operation *op) {
    assert(!isUnmodeledMemoryAccess(op) &&
           "an unmodeled memory access reached scheduling");
  });
#endif

  // Schedule the function body's regions, recursing into imperfect nests.
  return scheduleBlock(funcOp.getBody().front(), deps, lib, model, cycleTimeNs,
                       opts);
}

static void loadDependentDialects(MLIRContext &context) {
  context.getOrLoadDialect<allo::AlloDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
}

LogicalResult mlir::allo::runSDCScheduler(ModuleOp module, StringRef top,
                                          float cycleTime,
                                          const SchedulerOptions &opts,
                                          ScheduleModel &model) {
  // Fail before any work when the exact scheduler was asked for and this build
  // has none, rather than region by region: the fix is to the build.
  if (usesExactScheduler(opts.kind) && !hasExactScheduler()) {
    unsupported(Stage::Sched, module)
        << "An exact scheduler was requested but this build was configured "
           "without OR-Tools. Rebuild with -DALLO_ENABLE_ORTOOLS=ON, or use "
           "the default scheduler=\"heuristic\"";
    return failure();
  }
  loadDependentDialects(*module->getContext());
  // Timing characterization for every op (latency + delays), built from the
  // injected `dcp.device` + `dcp.operator` IR, once for scheduling and reify.
  auto loadedLib = OperatorLibrary::fromModule(module);

  // Callees before callers: a caller's own region partition asks whether each
  // call is indeterminate, which reads the callee's published latency.
  auto topFunc = module.lookupSymbol<func::FuncOp>(top);
  if (!topFunc) {
    error(Stage::Prep, module) << "Top function '" << top << "' not found";
    return failure();
  }
  auto orderOr = callGraphPostOrder(topFunc);
  if (failed(orderOr))
    return failure();

  IRRewriter rewriter(module.getContext());
  for (func::FuncOp fn : *orderOr) {
    // Whole-func memory + stream dependence analysis, refined by the
    // `allo.assume.*` hints. Built ahead of both steps below: the composition
    // reads its value ranges to bound a symbolic trip.
    DependenceAnalysis deps(fn);

    // Erase the hints the analysis has just consumed: they carry no schedulable
    // computation and would perturb the problem. Erasing them before the
    // construction above would drop every assumption.
    SmallVector<Operation *, 4> hints;
    fn.walk([&](Operation *op) {
      if (isa<AssumeNoDepOp, AssumeSSAOp>(op))
        hints.push_back(op);
    });
    for (Operation *op : hints)
      eraseHintAndDeadInputs(rewriter, op, deps.getAssumedRanges());

    size_t solvedBefore = model.regionCount();
    if (failed(scheduleFunc(fn, loadedLib, model, deps, cycleTime, opts)))
      return failure();
    recordTripBounds(fn, model, deps);
    // A func that solved no region (an empty body, or nothing but declarations)
    // publishes no latency: composing over no node reports zero, which a caller
    // would read as an exact zero-cycle contract.
    if (model.regionCount() > solvedBefore)
      publishKernelLatency(fn, model, deps);
  }
  return success();
}
