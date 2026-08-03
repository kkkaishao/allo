/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/LatencyModel.h" // the one latency composer
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (the one address map)
#include "allo/Scheduling/MemoryModel.h"  // kBankAttr
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/ScheduleModel.h"

#include "allo/IR/AlloOps.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

// An `i64` attribute for an optional value, or a null attribute (elided) when
// absent. Every optional dcp schedule attr (`ii`, `length`, `latency`,
// `start`, `trip`) is built with this shape.
static IntegerAttr optI64Attr(Builder &b, std::optional<int64_t> v) {
  return v ? b.getI64IntegerAttr(*v) : IntegerAttr();
}

// Erase \p op, having dropped the schedule of everything under it. The two go
// together: an erased op's address is handed back out by the next `create`, so
// an entry that outlives its op answers for whatever lands there next.
static void eraseScheduled(ScheduleModel &model, Operation *op) {
  op->walk([&](Operation *inner) { model.forget(inner); });
  op->erase();
}

// A `#allo.determinacy<...>` attribute: the declared controller-regime
// discriminant a region or kernel carries so consumers read it instead of
// re-deriving it.
static DeterminacyEnumAttr determinacyAttr(Builder &b, DeterminacyEnum d) {
  return DeterminacyEnumAttr::get(b.getContext(), d);
}

// Whether \p v is a *pure* combinational arith tree over block args (the region
// counter and iter-args) and constants, the shape that can be lifted into
// start-0 `dcp.compute`s. A memory load, an IP result, or an already-scheduled
// op makes the tree impure; such a condition is left raw, so the datapath
// builder derives no negative-depth edge and `validateDatapath` rejects it.
static bool isPureCombCondition(ScheduleModel &model, Value v) {
  if (isa<BlockArgument>(v))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ConstantOp>(def))
    return true;
  // A dcp region result is a settled survivor, stable at the guard's start, so
  // the DFS stops here: a valid tree leaf exactly like a block argument.
  if (isa<DCPathRegionOpInterface>(def))
    return true;
  if (!isa<arith::ArithDialect>(def->getDialect()) || model.scheduleOf(def))
    return false;
  return llvm::all_of(def->getOperands(),
                      [&](Value o) { return isPureCombCondition(model, o); });
}

// Schedule \p v's defining arith op (and its operands) at start 0.
// Precondition: isPureCombCondition(v). The already-scheduled guard dedups a
// shared subtree.
static void tagConditionStartZero(ScheduleModel &model, Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || isa<arith::ConstantOp>(def) ||
      isa<DCPathRegionOpInterface>(def) || model.scheduleOf(def))
    return; // stop at a settled survivor (a region result); do not tag it
  model.setStart(def, 0);
  for (Value o : def->getOperands())
    tagConditionStartZero(model, o);
}

// Lift the predicate or continue-condition \p cond into start-0 `dcp.compute`s
// so it becomes a combinational Source rather than a raw arith tree, if it is
// pure (an affine.if guard over the counter, or a sequential-wrapper while
// condition over the iter-args). A memory- or IP-dependent condition, or an
// already-scheduled leaf while's condition, is left as-is.
static void scheduleConditionTree(ScheduleModel &model, Value cond) {
  if (isPureCombCondition(model, cond))
    tagConditionStartZero(model, cond);
}

// The ASAP ready cycle of a while continue-condition \p v, the cycle its
// settled value is available, over the cone of loads and arith feeding it. A
// leaf (block arg, constant, or a settled dcp region result) is ready at 0; an
// already-scheduled op (a flushing leaf while's in-body condition) is trusted
// verbatim; a load's ready cycle is its indices' ready max plus the memref's
// read latency; an arith op's is its operands' ready max plus its op latency.
// Returns nullopt when the cone holds a shape the sequential CHECK region
// cannot emit (a store, a call, anything else), so the caller leaves the
// condition raw for a clean reject. This is a pure query and tags nothing.
static std::optional<int64_t> conditionReadyCycle(ScheduleModel &model, Value v,
                                                  const OperatorLibrary &lib) {
  if (isa<BlockArgument>(v))
    return 0;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isa<arith::ConstantOp>(def))
    return 0;
  if (isa<DCPathRegionOpInterface>(def))
    return 0; // a settled survivor (a preceding region result)
  if (const OpSchedule *at = model.scheduleOf(def))
    return at->start + static_cast<int64_t>(lib.lookup(def).latency);
  if (!isa<AffineLoadOp, memref::LoadOp>(def) &&
      !isa<arith::ArithDialect>(def->getDialect()))
    return std::nullopt; // store / call / other -> leave raw (clean reject)
  int64_t start = 0;
  for (Value o : def->getOperands()) {
    if (isa<MemRefType>(o.getType()))
      continue; // the memref declaration carries no schedule
    std::optional<int64_t> r = conditionReadyCycle(model, o, lib);
    if (!r)
      return std::nullopt;
    start = std::max(start, *r);
  }
  return start + static_cast<int64_t>(lib.lookup(def).latency);
}

// Schedule each op of the (schedulable) condition cone \p v at an ASAP start,
// the max of its operands' ready cycles, so the datapath derives a
// non-negative register depth for the load->compare edge. Precondition:
// `conditionReadyCycle(v)` is non-null. An already-scheduled op is untouched.
static int64_t tagConditionCone(ScheduleModel &model, Value v,
                                const OperatorLibrary &lib) {
  if (isa<BlockArgument>(v))
    return 0;
  Operation *def = v.getDefiningOp();
  if (isa<arith::ConstantOp>(def) || isa<DCPathRegionOpInterface>(def))
    return 0;
  if (const OpSchedule *at = model.scheduleOf(def))
    return at->start + static_cast<int64_t>(lib.lookup(def).latency);
  int64_t start = 0;
  for (Value o : def->getOperands())
    if (!isa<MemRefType>(o.getType()))
      start = std::max(start, tagConditionCone(model, o, lib));
  model.setStart(def, start);
  return start + static_cast<int64_t>(lib.lookup(def).latency);
}

// Schedule a while's continue-condition so it becomes a resolvable Source, by
// ASAP-scheduling its cone (conditionReadyCycle / tagConditionCone). A memory-
// or IP-dependent cone (`a - b > tol`) gets real per-op starts so the
// sequential CHECK/RUN controller can wait for it and the datapath derives
// non-negative register depths; an unschedulable cone stays raw for a clean
// reject.
static void scheduleWhileCondition(ScheduleModel &model, Value cond,
                                   const OperatorLibrary &lib) {
  if (conditionReadyCycle(model, cond, lib))
    tagConditionCone(model, cond, lib);
}

//===----------------------------------------------------------------------===//
// Per-op conversion. The `dcp.operator` symbols are already injected from the
// device model, so the reifier only *references* them (the compute IP path) or
// characterizes an op as combinational or as a memory access via the passed-in
// `OperatorLibrary`. It never materializes an operator.
//===----------------------------------------------------------------------===//

// Forward decl: `convertOp` reifies EVERY call to a `dcp.instance`; the invoke
// builder is defined further down with the rest of the call machinery.
static DCPathInstanceOp makeInvoke(OpBuilder &b, Location loc,
                                   TypeRange resultTypes, ValueRange operands,
                                   FlatSymbolRefAttr calleeAttr, Operation *at,
                                   int64_t start);

// Convert \p op (an op of the scheduled loop body) into its `dcp` equivalent in
// the pipeline block \p b is inserting into, mapping its results in \p map. Ops
// that are not compute/memory (constants, address arithmetic) are cloned as-is.
static void convertOp(Operation &op, OpBuilder &b, IRMapping &map,
                      ScheduleModel &model, const OperatorLibrary &lib) {
  Location loc = op.getLoc();
  // ONE lookup, for both the issue cycle and "did any phase schedule this".
  const OpSchedule *at = model.scheduleOf(&op);
  int64_t start = at ? at->start : 0;
  auto rm = [&](Value v) { return map.lookupOrDefault(v); };
  auto remap = [&](auto values) {
    SmallVector<Value> out;
    for (Value v : values)
      out.push_back(rm(v));
    return out;
  };
  // Carry the sub-cycle start time (from the chaining solve) onto the dcp op.
  auto setZ = [&](Operation *dst) {
    if (at && at->startInCycle)
      dst->setAttr("z", b.getF32FloatAttr(*at->startInCycle));
  };
  // Keep an op verbatim inside the region, preserving its scheduled start so
  // the schedule export can still report it (streams, constants, address
  // arithmetic).
  auto cloneKept = [&]() {
    Operation *c = b.clone(op, map);
    if (at) {
      c->setAttr("start", b.getI64IntegerAttr(start));
      setZ(c);
    }
  };

  // A memory access's latency is the accessed memref's read/write latency
  // (from the device memory model), resolved here and carried on the dcp op.
  auto memLatency = [&]() -> uint64_t { return lib.lookup(&op).latency; };
  // The bank `assign-banks` decided, moved off the discardable attribute onto
  // the dcp op's own, so no later rewrite can drop the decision the schedule
  // was already billed against. Absent means the access reaches every bank.
  IntegerAttr bank = op.getAttrOfType<IntegerAttr>(kBankAttr);
  // The address map of an array access, from `asMemAccess`, the one place that
  // decides it: an affine op's own map, and for a non-affine one the identity
  // map over its indices.
  auto addrMap = [&]() { return asMemAccess(&op)->map; };
  if (auto l = dyn_cast<AffineLoadOp>(&op)) {
    auto nw = DCPathLoadOp::create(
        b, loc, l.getType(), rm(l.getMemRef()), remap(l.getMapOperands()),
        addrMap(), (uint64_t)start, memLatency(), bank, IntegerAttr());
    setZ(nw);
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto l = dyn_cast<memref::LoadOp>(&op)) {
    auto nw = DCPathLoadOp::create(
        b, loc, l.getType(), rm(l.getMemRef()), remap(l.getIndices()),
        addrMap(), (uint64_t)start, memLatency(), bank, IntegerAttr());
    setZ(nw);
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto s = dyn_cast<AffineStoreOp>(&op)) {
    auto nw = DCPathStoreOp::create(
        b, loc, rm(s.getValueToStore()), rm(s.getMemRef()),
        remap(s.getMapOperands()), addrMap(), (uint64_t)start, memLatency(),
        bank, IntegerAttr());
    setZ(nw);
    return;
  }
  if (auto s = dyn_cast<memref::StoreOp>(&op)) {
    auto nw = DCPathStoreOp::create(b, loc, rm(s.getValueToStore()),
                                    rm(s.getMemRef()), remap(s.getIndices()),
                                    addrMap(), (uint64_t)start, memLatency(),
                                    bank, IntegerAttr());
    setZ(nw);
    return;
  }
  // Streams stay as FIFO ops, not compute; keep them verbatim with their start.
  if (isa<StreamGetOp, StreamPutOp>(&op)) {
    cloneKept();
    return;
  }
  // A buffer allocation is a declaration; a `memref.get_global` references a
  // module-level constant table (a ROM built from the global's initializer).
  // Both stay verbatim so loads that read them still resolve their memref.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
          StreamCreateOp>(&op)) {
    cloneKept();
    return;
  }
  // EVERY sub-kernel call reifies to a `dcp.instance`, the call node the
  // datapath models as a `CallUnit`, so a scalar-returning call is an invoke
  // too. An `await` spawn differs only in start policy and rides `allo.async`.
  if (auto call = dyn_cast<func::CallOp>(&op)) {
    auto inv =
        makeInvoke(b, loc, call.getResultTypes(), remap(call.getOperands()),
                   call.getCalleeAttr(), &op, start);
    if (call->hasAttr(kAlloAsyncAttr))
      inv->setAttr(kAlloAsyncAttr, b.getUnitAttr());
    for (auto [old, nw] : llvm::zip(call.getResults(), inv.getResults()))
      map.map(old, nw);
    return;
  }
  // A scheduled single-result op (not a constant) is a compute op, realized on
  // one of two exclusive paths: a combinational op carries a `comb_kind`; an IP
  // op references its injected `dcp.operator` via `op_type`.
  if (op.getNumResults() == 1 && at && !isa<arith::ConstantOp>(op)) {
    OperatorIdentity id = lib.lookup(&op).identity;
    assert(id.realized() && "a scheduled compute op with no realization");
    CombOpKindEnumAttr combKind;
    FlatSymbolRefAttr opType;
    if (id.comb)
      combKind = CombOpKindEnumAttr::get(
          b.getContext(), *symbolizeCombOpKindEnum(id.realization));
    else
      opType = FlatSymbolRefAttr::get(b.getContext(), id.realization);
    // The instance the allocation put it on, when a solve decided one.
    FlatSymbolRefAttr unit;
    if (at->unit)
      unit = FlatSymbolRefAttr::get(b.getContext(),
                                    model.allocatedUnits()[*at->unit].name);
    auto nw = DCPathComputeOp::create(b, loc, op.getResult(0).getType(),
                                      remap(op.getOperands()), combKind, opType,
                                      b.getI64IntegerAttr(start), unit);
    for (NamedAttribute attr : op.getAttrs())
      nw->setAttr(attr.getName(), attr.getValue());
    setZ(nw);
    map.map(op.getResult(0), nw.getResult());
    return;
  }
  // Constants / address arithmetic: keep verbatim inside the region.
  cloneKept();
}

//===----------------------------------------------------------------------===//
// The timing attributes a region op is built with.
//===----------------------------------------------------------------------===//

namespace {
// What a `dcp.pipeline` / `dcp.sequential` is CONSTRUCTED with, derived from
// the `RegionSolution` for it. Every field the solution holds is the region's
// own and per-invocation; the span below is COMPOSED here rather than carried.
//
// A null solution leaves every field empty, and that is a real answer rather
// than a default: an all-constant span the solver skipped, or a residual
// wrapper that owns no solve of its own.
struct RegionAttrs {
  std::optional<int64_t> ii;
  std::optional<int64_t> length; // schedule depth, a report
  std::optional<int64_t> drain;  // terminal cycle, what a span composes from
  std::optional<int64_t> latency;
  bool latencyBound = false;

  RegionAttrs() = default;
  explicit RegionAttrs(const RegionSolution *r) {
    if (!r)
      return;
    ii = r->ii;
    length = r->length;
    drain = r->drain;
    // The region as the latency model sees it, from its own solved numbers. A
    // container's span is recomposed from its children and overwrites this;
    // what only this can supply is an assume-bounded trip.
    SpanNode n;
    n.drain = r->drain;
    n.ii = r->ii;
    n.acyclic = !r->ii;
    n.trip = r->trip;
    if (!n.trip && n.acyclic)
      n.trip = 1; // a straight-line span runs once
    latency = composeSpan(n);
    latencyBound = r->tripIsBound;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Region materialization.
//===----------------------------------------------------------------------===//

// Whether \p loop's body holds a nested loop (affine.for / scf.for /
// scf.while), i.e. it is not truly innermost. Checked BEFORE the loop's body is
// materialized, so it sees the raw affine/scf children (a wrapper), not the
// dcp children they later become. Mirrors the scheduler's own predicate.
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

// Compile-time-constant trip count of a counted loop, else nullopt.
static std::optional<int64_t> constantTripOf(LoopLikeOpInterface loop) {
  if (auto affineLoop = dyn_cast<AffineForOp>(loop.getOperation())) {
    if (std::optional<uint64_t> t = getConstantTripCount(affineLoop))
      return static_cast<int64_t>(*t);
    return std::nullopt;
  }
  auto scfLoop = cast<scf::ForOp>(loop.getOperation());
  std::optional<int64_t> lb = getConstantIntValue(scfLoop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(scfLoop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(scfLoop.getStep());
  if (lb && ub && step && *step > 0)
    return std::max<int64_t>(0, llvm::divideCeilSigned(*ub - *lb, *step));
  return std::nullopt;
}

// The constant value of an index SSA value, seeing through a `dcp.sequential`
// result: the reifier hoists a loop's loop-invariant constant lb/step into a
// preceding prologue region *before* the loop is reified, so the loop's bound
// operand is that region's result (not a foldable constant). Peel it back to
// the yielded value (a `dcp.uncondition` operand) and re-fold.
static std::optional<int64_t> constantIndexThroughRegions(Value v) {
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return c;
  if (auto res = dyn_cast<OpResult>(v))
    if (auto seq = dyn_cast<DCPathSequentialOp>(res.getOwner()))
      return constantIndexThroughRegions(
          seq.getBody().front().getTerminator()->getOperand(
              res.getResultNumber()));
  return std::nullopt;
}

// The lower bound and step of a counted loop, carried onto the dcp.pipeline so
// the induction register holds the real IV (`lb`, `lb+step`, ...). Each is a
// compile-time constant when statically known (folded through any prologue
// region), else the runtime SSA `index` value of an scf.for data-dependent
// bound: a constant `lb` with a dynamic ub (`for i in range(1, n)`), or a
// genuinely runtime lb/step (`for i in range(m, n)` with `m` loaded). An
// affine.for with a non-constant (symbolic) lb defaults to 0 (unhandled).
namespace {
struct LoopBounds {
  int64_t lb = 0, step = 1; // used iff the matching Value is null (constant)
  Value lbVal, stepVal;     // runtime bound (an scf.for SSA lb/step)
};
} // namespace
static LoopBounds lbStepOf(LoopLikeOpInterface loop) {
  if (auto af = dyn_cast<AffineForOp>(loop.getOperation()))
    return {af.hasConstantLowerBound() ? af.getConstantLowerBound() : 0,
            af.getStepAsInt(), Value(), Value()};
  auto sf = cast<scf::ForOp>(loop.getOperation());
  LoopBounds r;
  if (std::optional<int64_t> c =
          constantIndexThroughRegions(sf.getLowerBound()))
    r.lb = *c;
  else
    r.lbVal = sf.getLowerBound();
  if (std::optional<int64_t> c = constantIndexThroughRegions(sf.getStep()))
    r.step = *c;
  else
    r.stepVal = sf.getStep();
  return r;
}

// The runtime upper bound of an scf.for whose trip is NOT a compile-time
// constant (a memory-loaded / non-affine ub), wired as the pipeline's real ub;
// the counter runs [lb, ub) and terminates on `iv+step >= ub`. An affine.for's
// symbolic bound goes through `materializeAffineBound` instead.
static Value dynamicTripBound(LoopLikeOpInterface loop) {
  auto scfLoop = dyn_cast<scf::ForOp>(loop.getOperation());
  return scfLoop ? scfLoop.getUpperBound() : Value();
}

// Materialize an affine.for bound as an `index` value at `b`'s insertion
// point (before the loop): the max of the lower-bound map's results, the min
// of the upper-bound map's. A non-trivial expression synthesizes arith ops
// tagged `start=0` so `convertOp` lifts them to combinational `dcp.compute`
// Sources; this is the affine counterpart of an scf.for's runtime bound
// operand, for a symbolic (IV-relative) triangular or tile bound (`for j in
// range(i+1, n)`).
static Value materializeAffineBound(OpBuilder &b, ScheduleModel &model,
                                    Location loc, AffineForOp af,
                                    bool isLower) {
  AffineMap map = isLower ? af.getLowerBoundMap() : af.getUpperBoundMap();
  ValueRange operands =
      isLower ? af.getLowerBoundOperands() : af.getUpperBoundOperands();
  ValueRange dims = operands.take_front(map.getNumDims());
  ValueRange syms = operands.drop_front(map.getNumDims());
  Operation *loopOp = af.getOperation();
  Operation *before = loopOp->getPrevNode();
  SmallVector<Value> parts;
  for (AffineExpr e : map.getResults())
    parts.push_back(affine::expandAffineExpr(b, loc, e, dims, syms));
  Value bound = parts.front();
  for (Value v : llvm::drop_begin(parts))
    bound = isLower ? arith::MaxSIOp::create(b, loc, bound, v).getResult()
                    : arith::MinSIOp::create(b, loc, bound, v).getResult();
  // Schedule every op synthesized just now, which the solver never saw, so
  // `convertOp` reifies it as a combinational unit.
  for (Operation *o = before ? before->getNextNode()
                             : &loopOp->getBlock()->front();
       o != loopOp; o = o->getNextNode())
    model.setStart(o, 0);
  return bound;
}

// Create a `dcp.pipeline`'s single block: an index counter (arg 0) followed by
// one arg per iter-arg init, matching the counter+iter-args block-arg contract
// shared by counted and while pipelines.
static Block *createCounterBlock(OpBuilder &b, DCPathPipelineOp pipe,
                                 ValueRange inits, Location loc) {
  SmallVector<Type> argTypes{b.getIndexType()};
  SmallVector<Location> argLocs{loc};
  for (Value in : inits) {
    argTypes.push_back(in.getType());
    argLocs.push_back(loc);
  }
  return b.createBlock(&pipe.getBody(), {}, argTypes, argLocs);
}

// Rewrite an `scf.while` into a while `dcp.pipeline`: `trip` unset, terminated
// by `dcp.condition` carrying the condition value plus the loop-carried
// next-values. A straight-line while flushing-pipelines with `ii` from its
// own solve; a nested-loop while runs sequentially (`ii` unset), its
// after-block already materialized so its dcp children clone in verbatim.
// Requires identity forwarding: both the before-arg and after-arg of a slot
// map to the same iter-arg.
static void materializeWhilePipeline(const RegionAttrs &r, scf::WhileOp w,
                                     ScheduleModel &model,
                                     const OperatorLibrary &lib) {
  OpBuilder b(w);
  Location loc = w.getLoc();

  ValueRange inits = w.getInits();
  auto pipe = DCPathPipelineOp::create(
      b, loc, w.getResultTypes(), /*lbBound=*/Value(), /*dynamicBound=*/Value(),
      /*stepBound=*/Value(), inits, /*trip=*/IntegerAttr(),
      /*trip_bound=*/IntegerAttr(), /*lb=*/IntegerAttr(),
      /*step=*/IntegerAttr(), optI64Attr(b, r.ii), optI64Attr(b, r.length),
      optI64Attr(b, r.drain), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr(), DeterminacyEnumAttr());
  Block *blk = createCounterBlock(b, pipe, inits, loc);

  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  IRMapping map;
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    map.map(before.getArgument(j), blk->getArgument(j + 1));
    map.map(after.getArgument(j), blk->getArgument(j + 1));
  }

  // A sequential-wrapper while's before-block condition is unscheduled, so
  // schedule its cone into a Source::Unit; a leaf while's before-block ops
  // already carry real starts and pass through unchanged.
  scheduleWhileCondition(model, w.getConditionOp().getCondition(), lib);

  b.setInsertionPointToEnd(blk);
  for (Operation &op : before.without_terminator())
    convertOp(op, b, map, model, lib);
  for (Operation &op : after.without_terminator())
    convertOp(op, b, map, model, lib);

  Value cond = map.lookupOrDefault(w.getConditionOp().getCondition());
  SmallVector<Value> carried;
  for (Value v : w.getYieldOp().getOperands())
    carried.push_back(map.lookupOrDefault(v));
  DCPathConditionOp::create(b, loc, cond, carried);

  for (auto [old, nw] : llvm::zip(w.getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  eraseScheduled(model, w);
}

// Rewrite one counted loop (affine.for or scf.for) into a dcp.pipeline by
// converting its body ops. An already-materialized child `dcp.pipeline` or
// `dcp.sequential`, from a co-scheduled level or an imperfect wrapper, is
// cloned verbatim. The trip count is recorded only when it is a compile-time
// constant.
static DCPathPipelineOp materializeLoopToPipeline(const RegionAttrs &r,
                                                  LoopLikeOpInterface loop,
                                                  ScheduleModel &model,
                                                  const OperatorLibrary &lib) {
  Operation *loopOp = loop.getOperation();
  OpBuilder b(loopOp);
  Location loc = loop.getLoc();

  ValueRange inits = loop.getInits();
  // A trip that is not a compile-time constant wires the loop's upper bound as
  // the `dynamicBound` operand. Only an scf.for has a runtime (memory-loaded,
  // non-affine) ub; affine bounds are constant or affine-symbol.
  Value dynamicBound;
  if (!constantTripOf(loop)) {
    if (auto af = dyn_cast<AffineForOp>(loopOp))
      dynamicBound = materializeAffineBound(b, model, loc, af,
                                            /*isLower=*/false);
    else
      dynamicBound = dynamicTripBound(loop);
  }
  // Carry the source loop's lb/step so the induction register runs the real IV,
  // correct for `lb != 0` or `step != 1` even under a runtime ub. Each rides an
  // attribute when compile-time (elided at the default 0/1), else an operand.
  LoopBounds bounds = lbStepOf(loop);
  // An affine.for with a symbolic (IV-relative) lower bound, e.g. `for j in
  // range(i+1, n)` after a guard folds into it, materializes as a runtime lb
  // operand; lbStepOf defaulted that symbolic lb to 0.
  if (auto af = dyn_cast<AffineForOp>(loopOp))
    if (!af.hasConstantLowerBound())
      bounds.lbVal = materializeAffineBound(b, model, loc, af,
                                            /*isLower=*/true);
  std::optional<int64_t> lbAttr, stepAttr;
  if (!bounds.lbVal && bounds.lb != 0)
    lbAttr = bounds.lb;
  if (!bounds.stepVal && bounds.step != 1)
    stepAttr = bounds.step;
  // The worst-case count of a loop with no static one
  std::optional<int64_t> trip = constantTripOf(loop);
  std::optional<int64_t> tripBound;
  if (!trip)
    tripBound = model.tripBoundOf(loopOp);
  auto pipe = DCPathPipelineOp::create(
      b, loc, loopOp->getResultTypes(), bounds.lbVal, dynamicBound,
      bounds.stepVal, inits, optI64Attr(b, trip), optI64Attr(b, tripBound),
      optI64Attr(b, lbAttr), optI64Attr(b, stepAttr), optI64Attr(b, r.ii),
      optI64Attr(b, r.length), optI64Attr(b, r.drain), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr(), DeterminacyEnumAttr());
  Block *blk = createCounterBlock(b, pipe, inits, loc);

  // The induction var is body block argument 0, iter-args follow (both
  // dialects).
  Block *body = &loop.getLoopRegions().front()->front();
  IRMapping map;
  map.map(body->getArgument(0), blk->getArgument(0));
  // Carry the source IV's NameLoc onto the counter block arg so the datapath
  // emitter can name the iteration-counter wire after the loop variable (i).
  blk->getArgument(0).setLoc(body->getArgument(0).getLoc());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs()))
    map.map(arg, blk->getArgument(i + 1));

  b.setInsertionPointToEnd(blk);
  for (Operation &op : body->without_terminator())
    convertOp(op, b, map, model, lib);

  Operation *term = body->getTerminator();
  SmallVector<Value> yields;
  for (Value v : term->getOperands())
    yields.push_back(map.lookupOrDefault(v));
  DCPathUnconditionOp::create(b, term->getLoc(), yields);

  for (auto [old, nw] : llvm::zip(loopOp->getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  eraseScheduled(model, loopOp);
  return pipe;
}

// Rewrite a straight-line (acyclic) region into a dcp.sequential. A region of
// only declarations is left in place, sourced directly by identity like a
// loop-invariant tie-off or func-arg memref, so it forms no region and threads
// no cross-region SSA result; anything else is wrapped, with values used after
// the region yielded as sequential results.
static void materializeSequential(const RegionAttrs &r,
                                  ArrayRef<Operation *> ops,
                                  ScheduleModel &model,
                                  const OperatorLibrary &lib, bool container) {
  SmallVector<Operation *> body;
  for (Operation *op : ops)
    if (!op->hasTrait<OpTrait::IsTerminator>())
      body.push_back(op);

  // In a container, a static `memref.alloc` that a later region reads must stay
  // at func level: a memref is not a datapath value to latch, and the CallUnit
  // path needs the shared buffer identity-sourced. Hoist it out of the wrap.
  llvm::SmallPtrSet<Operation *, 8> inBody(body.begin(), body.end());
  auto escapesBody = [&](Operation *op) {
    return llvm::any_of(op->getResults(), [&](Value res) {
      return llvm::any_of(res.getUsers(),
                          [&](Operation *u) { return !inBody.contains(u); });
    });
  };
  SmallVector<Operation *> work, hoisted;
  for (Operation *op : body) {
    if (container && isa<memref::AllocOp, memref::AllocaOp>(op) &&
        op->getNumOperands() == 0 && escapesBody(op))
      hoisted.push_back(op); // leave at func level, do not wrap or erase
    else
      work.push_back(op);
  }

  if (work.empty() || llvm::all_of(work, isDeclarationOp))
    return;

  // Move the hoisted allocs above the region so they dominate the wrapped uses.
  for (Operation *op : hoisted)
    op->moveBefore(work.front());

  llvm::SmallPtrSet<Operation *, 8> inRegion(work.begin(), work.end());
  SmallVector<Value> escaping;
  for (Operation *op : work)
    for (Value res : op->getResults())
      if (llvm::any_of(res.getUsers(),
                       [&](Operation *u) { return !inRegion.contains(u); }))
        escaping.push_back(res);

  OpBuilder b(work.front());
  Location loc = work.front()->getLoc();

  SmallVector<Type> resultTypes(
      llvm::map_range(escaping, [](Value v) { return v.getType(); }));
  auto seq = DCPathSequentialOp::create(
      b, loc, resultTypes, optI64Attr(b, r.length), optI64Attr(b, r.drain),
      optI64Attr(b, r.latency), r.latencyBound ? b.getUnitAttr() : UnitAttr(),
      DeterminacyEnumAttr());
  Block *blk = b.createBlock(&seq.getBody());

  IRMapping map;
  b.setInsertionPointToEnd(blk);
  for (Operation *op : work)
    convertOp(*op, b, map, model, lib);

  SmallVector<Value> yields(llvm::map_range(
      escaping, [&](Value v) { return map.lookupOrDefault(v); }));
  DCPathUnconditionOp::create(b, loc, yields);

  for (auto [orig, res] : llvm::zip(escaping, seq.getResults()))
    orig.replaceAllUsesWith(res);
  for (Operation *op : llvm::reverse(work))
    eraseScheduled(model, op);
}

// Whether anything in the module targets \p mod. Asked while \p mod is being
// closed, which by post-order is before any of its callers is reified, so a
// caller still spells the edge `func.call`.
static bool isCalled(DCPathModuleOp mod) {
  bool called = false;
  mod->getParentOfType<ModuleOp>().walk([&](func::CallOp c) {
    if (c.getCallee() != mod.getSymName())
      return WalkResult::advance();
    called = true;
    return WalkResult::interrupt();
  });
  return called;
}

// Hold the scheduler's `allo.sched.latency` to the span the reify just built.
// The two need not agree: the `dcp.module`'s own becomes the emitter's exact
// `staticLatency`, while the scheduler's number only PLACES a caller's
// consumers and may be a loose upper bound. So the invariant is one-sided:
// an UNDERCOUNT (scheduler < reify) is the miscompile, since a consumer
// placed against it would sample before the callee writes. Checked only at a
// CALL, since a callee is reified before its callers and their `func.call`s
// are still here to see.
static void checkLatencyBound(DCPathModuleOp mod, std::optional<int64_t> dcpLat,
                              bool concurrent) {
  auto sched = mod->getAttrOfType<IntegerAttr>(kLatencyAttr);
  if (!sched || !dcpLat)
    return; // either side unknown: the call composes on `done`, not on a time
  // A CONCURRENT container's span is a completion floor over processes paced by
  // back-pressure, not a schedule, so neither number times the other.
  if (concurrent || sched.getInt() == *dcpLat || !isCalled(mod))
    return;
  if (sched.getInt() > *dcpLat) {
    debug(Stage::Dcp, mod) << "Latency bound is loose for callee '"
                           << mod.getSymName() << "': scheduler "
                           << sched.getInt() << ", reify " << *dcpLat << " ("
                           << sched.getInt() - *dcpLat
                           << " cycle(s) a caller waits through)";
    return;
  }
  // The caller placed its consumers against `sched` and the hardware runs for
  // `dcpLat`, so the difference is a read before the write.
  assert(false && "the scheduler's callee latency undercuts what the reify "
                  "builds; a consumer placed against it samples early");
  error(Stage::Dcp, mod) << "Latency bound is UNSOUND for callee '"
                         << mod.getSymName() << "': scheduler "
                         << sched.getInt() << " undercuts reify " << *dcpLat
                         << " by " << *dcpLat - sched.getInt()
                         << " cycle(s); a consumer time-triggered off this "
                            "callee would sample before it writes";
}

// Stamp `latency` and `determinacy` on every reified region (from
// `dcpRegionTiming`, the report), then the whole-kernel contract on the
// `dcp.module` itself.
//
// The kernel's span composes the top-level regions over their dependence
// DAG: independent siblings overlap (both start at the kernel's own
// `start`), so the span is the longest path through them, not the sum.
//
// A container publishes a span only when BOTH `allKnown` (every child
// publishes a start->done contract) and `known` (every top-level region has
// a static span to place those children within) hold; a kernel whose calls
// sit inside a `while` or an unpredicated `if` fails the second. What makes a
// container CONCURRENT is asked of its invokes directly via
// `spawnsConcurrently`.
static void setDcpLatencies(DCPathModuleOp mod) {
  mod.walk([&](DCPathRegionOpInterface region) {
    RegionTiming t = dcpRegionTiming(region);
    if (t.staticLatency)
      region.setLatency(static_cast<uint64_t>(*t.staticLatency));
    region.setDeterminacy(t.determinacy);
  });

  // The top-level regions, and the ops each owns for the sibling DAG. Index
  // aligned with `dcpSpanNodes` below, which selects the same ops. A
  // `dcp.instance` is never one of them: the reify wraps every call into a
  // region, so a bare instance at kernel scope would be charged by nothing.
  SmallVector<SmallVector<Operation *>> topOps;
  bool bounded = false;
  for (Operation &op : mod.getBody().front()) {
    assert(!isa<DCPathInstanceOp>(op) &&
           "a call is reified inside a region, so nothing composes a bare "
           "instance at kernel scope");
    if (auto region = dyn_cast<DCPathRegionOpInterface>(&op)) {
      topOps.push_back({&op});
      bounded |= region.getLatencyBound();
    }
  }
  std::optional<int64_t> total =
      composeDag(dcpSpanNodes(mod.getBody().front(), /*topLevel=*/true),
                 siblingPredecessors(topOps));
  bool known = total.has_value();

  // What the children say about themselves, which is a different question from
  // what the regions holding them compose to.
  {
    bool container = false, allKnown = true, structural = false;
    mod.walk([&](DCPathInstanceOp inv) {
      container = true;
      structural |= spawnsConcurrently(inv);
      if (!inv.getLatency())
        allKnown = false;
    });
    if (container) {
      // Both sides, as above.
      bool staticSpan = known && allKnown && !bounded;
      if (staticSpan) {
        mod.setLatency(*total);
        checkLatencyBound(mod, *total, structural);
      }
      // A container holding an `await` spawn or a stream-wired child is
      // `concurrent` and gets a structural top; a purely scheduled composition
      // is `counted_static` or `indeterminate` and stays a leaf.
      mod.setDeterminacy(structural   ? DeterminacyEnum::Concurrent
                         : staticSpan ? DeterminacyEnum::CountedStatic
                                      : DeterminacyEnum::Indeterminate);
      // Only the KERNEL's class is stamped: it crosses a module boundary a
      // caller cannot see across. A REGION's own the emitter derives itself.
      return;
    }
  }

  if (known) {
    mod.setLatency(*total);
    mod.setLatencyBound(bounded);
  }
  checkLatencyBound(mod, total, /*concurrent=*/false);
  // Whole-kernel determinacy: an exact static latency is `counted_static`; a
  // bounded (dynamic-trip) or unknown-length kernel is `indeterminate`. That is
  // the (latency && !latency_bound) test the op's verifier holds it to.
  mod.setDeterminacy(known && !bounded ? DeterminacyEnum::CountedStatic
                                       : DeterminacyEnum::Indeterminate);
}

// Retire the scheduler's provisional whole-kernel latency once the reify has
// published its own (`setDcpLatencies`), which is the exact one. It is the last
// thing the schedule left on the IR; everything else travels in the
// `ScheduleModel`.
static void stripScheduleCarrier(DCPathModuleOp mod) {
  mod->removeAttr(kLatencyAttr);
}

namespace {
// Post-order lowering of one function's loop/region tree. Mirrors the
// scheduler's `scheduleBlock` / `scheduleRegion` descent, materializing each
// region bottom-up (a loop is wrapped only after its body is materialized, so
// deepest-first ordering falls out of the recursion). A counted for-loop always
// becomes a `dcp.pipeline`, whether leaf, co-scheduled pipelined level, or
// sequential wrapper: the three differ only in where the II comes from. A
// straight-line span becomes a `dcp.sequential`, and a while or opaque `if` is
// left raw wrapping its materialized children.
struct Reifier {
  func::FuncOp func;
  ScheduleModel &model;
  const OperatorLibrary &lib;
  // Set in run(): this func calls sub-kernels, so a shared `memref.alloc` an
  // acyclic span holds must be hoisted to func level rather than yielded as a
  // cross-region survivor (materializeSequential).
  bool container = false;

  void materializeBlock(Block &block) {
    for (const SchedRegion &region : enumerateRegions(block))
      materializeRegion(region);
  }

  // One region of a block, by anchor kind. The `while` case splits on whether
  // the loop can flushing-pipeline. It cannot when it nests a loop (the
  // per-iteration length is then data-dependent), when it calls a sub-kernel
  // (one child instance is fired and awaited per iteration, which no
  // iteration-per-cycle schedule can follow), or when its continue-condition is
  // not combinational (a memory read or a latency IP is not settled in-cycle).
  // `conditionIsCombinational` is the predicate the scheduler routes on, read
  // here so the two descents stay in lockstep. Closing a sequential while needs
  // identity forwarding; a non-identity one is rare and left raw.
  void materializeRegion(const SchedRegion &region) {
    if (region.kind == allo::RegionKind::StraightLine) {
      materializeSequential(RegionAttrs(model.regionOf(region.ops.front())),
                            region.ops, model, lib, container);
      return;
    }
    Operation *anchor = region.anchor();
    if (isa<AffineForOp, scf::ForOp>(anchor)) {
      materializeCountedLoop(cast<LoopLikeOpInterface>(anchor));
    } else if (auto w = dyn_cast<scf::WhileOp>(anchor)) {
      if (hasNestedLoop(w) || !conditionIsCombinational(w, lib) ||
          blockHasSyncCall(w.getAfter().front())) {
        // A while that cannot flush-pipeline takes the sequential CHECK/RUN
        // controller: materialize the body into dcp child regions, then close
        // the while into a sequential dcp.pipeline (`ii` unset).
        materializeBlock(w.getAfter().front());
        if (whileHasIdentityForwarding(w))
          materializeWhilePipeline(RegionAttrs(), w, model, lib);
      } else {
        // A straight-line while with a combinational condition
        // flushing-pipelines (ii from its own solve).
        materializeWhilePipeline(RegionAttrs(model.regionOf(anchor)), w, model,
                                 lib);
      }
    } else if (isa<scf::IfOp, AffineIfOp>(anchor)) {
      // An opaque guard left by if-conversion: materialize each branch, then
      // close the `if` into a dcp.select. An affine.if's IntegerSet becomes an
      // i1 via `affineIfCondition`; an scf.if already has one.
      for (Region &branch : anchor->getRegions())
        if (!branch.empty())
          materializeBlock(branch.front());
      OpBuilder b(anchor);
      Value cond = isa<AffineIfOp>(anchor)
                       ? affineIfCondition(b, cast<AffineIfOp>(anchor))
                       : cast<scf::IfOp>(anchor).getCondition();
      // Lift a raw predicate tree to start-0 computes so the guard's condition
      // is a Source::Unit; an scf.if condition that is already a scheduled
      // survivor is left untouched.
      scheduleConditionTree(model, cond);
      closeIntoDcpSelect(b, anchor, cond);
    }
  }

  // Materialize an affine.if's IntegerSet predicate into an i1: the conjunction
  // of its constraints (each `expr >= 0`, or `== 0` for an equality), built
  // with `expandAffineExpr` + `cmpi` + `andi`. Mirrors upstream
  // AffineIfLowering. The ops are inserted before `b`'s point, ahead of the
  // dcp.select, and reference the loop IVs, which the enclosing wrapper rewires
  // to its counter.
  Value affineIfCondition(OpBuilder &b, AffineIfOp ifOp) {
    Location loc = ifOp.getLoc();
    IntegerSet set = ifOp.getIntegerSet();
    SmallVector<Value> operands(ifOp.getOperands());
    ArrayRef<Value> ops(operands);
    unsigned numDims = set.getNumDims();
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value cond;
    for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
      Value aff = affine::expandAffineExpr(b, loc, set.getConstraint(i),
                                           ops.take_front(numDims),
                                           ops.drop_front(numDims));
      auto pred =
          set.isEq(i) ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value cmp = arith::CmpIOp::create(b, loc, pred, aff, zero);
      cond = cond ? arith::AndIOp::create(b, loc, cond, cmp).getResult() : cmp;
    }
    if (!cond) // an empty set is always true
      cond = arith::ConstantIntOp::create(b, loc, /*value=*/1, /*width=*/1);
    return cond;
  }

  // Close a scheduled if (scf.if or affine.if, branches already materialized
  // into dcp regions) into a dcp.select with condition \p cond: move each
  // branch body verbatim, rewrite its yield to a dcp.uncondition, and forward
  // the results. Latency is left unset, since a data-dependent guard has no
  // static count.
  void closeIntoDcpSelect(OpBuilder &b, Operation *ifOp, Value cond) {
    auto sel = DCPathSelectOp::create(
        b, ifOp->getLoc(), ifOp->getResultTypes(), cond,
        /*latency=*/IntegerAttr(),
        /*latency_bound=*/UnitAttr(), DeterminacyEnumAttr());
    sel.getThenRegion().takeBody(ifOp->getRegion(0));
    if (!ifOp->getRegion(1).empty())
      sel.getElseRegion().takeBody(ifOp->getRegion(1));
    for (Region *r : {&sel.getThenRegion(), &sel.getElseRegion()}) {
      if (r->empty())
        continue;
      Operation *term = r->front().getTerminator();
      OpBuilder yb(term);
      DCPathUnconditionOp::create(yb, term->getLoc(), term->getOperands());
      eraseScheduled(model, term);
    }
    for (auto [oldR, newR] : llvm::zip(ifOp->getResults(), sel.getResults()))
      oldR.replaceAllUsesWith(newR);
    eraseScheduled(model, ifOp);
  }

  // A counted for-loop -> dcp.pipeline. The two cases are distinguished BEFORE
  // the body is materialized (so nested loops are still raw affine/scf ops):
  //   * sequential wrapper (imperfect or non-flattened band): materialize every
  //     sub-region, then wrap with ii = Σ child invocation latency;
  //   * leaf innermost: wrap directly, ii from the solve keyed by it.
  void materializeCountedLoop(LoopLikeOpInterface loop) {
    Operation *op = loop.getOperation();
    Block &body = loop.getLoopRegions().front()->front();
    // The scheduler composed this loop's span off the same classification.
    RegionShape shape = countedLoopShape(loop);
    // A counted loop owns a solve exactly when it is a LEAF. A residual wrapper
    // owns none and synthesizes its own from the children it sequences.
    RegionSolution *sol = model.regionOf(op);
    [[maybe_unused]] DCPathPipelineOp pipe;
    if (shape == RegionShape::Container) {
      materializeBlock(body);
      pipe = materializeLoopToPipeline(sequentialWrapperAttrs(loop), loop,
                                       model, lib);
    } else {
      assert(sol && "a leaf counted loop owns the solve keyed by it");
      pipe = materializeLoopToPipeline(RegionAttrs(sol), loop, model, lib);
    }
    // A container whose child spans are all declarations-only builds no child
    // region, so it comes out a leaf; nothing else may move.
    assert((dcpRegionShape(pipe) == shape ||
            (shape == RegionShape::Container &&
             dcpRegionShape(pipe) == RegionShape::Leaf)) &&
           "the region built disagrees with the shape both composers read");
  }

  // The synthesized timing of a residual sequential wrapper, which owns no
  // solve of its own, derived from its materialized children. Iterations do not
  // overlap, so its II is one body pass, the SAME sum its own latency is
  // composed from.
  //
  // A body pass and the whole span go unknown separately. A dynamic INNER trip
  // leaves the pass itself data-dependent, so `ii` and `length` are unset and
  // the wrapper becomes a done-based sequential controller; a dynamic OUTER
  // trip keeps a concrete pass but no static total, so only `latency` is unset.
  RegionAttrs sequentialWrapperAttrs(LoopLikeOpInterface loop) {
    Block &body = loop.getLoopRegions().front()->front();
    RegionAttrs r;
    // The wrapper described before it exists: the same node `dcpSpanNode`
    // would report of it afterwards.
    SpanNode n;
    n.shape = RegionShape::Container;
    n.children = dcpSpanNodes(body, /*topLevel=*/false);
    std::optional<int64_t> pass = composeSequence(n.children);
    if (!pass)
      return r;
    r.ii = *pass;
    r.length = *pass;
    n.trip = constantTripOf(loop);
    if (n.trip) {
      r.latency = composeSpan(n);
      r.latencyBound = llvm::any_of(body, [](Operation &o) {
        auto region = dyn_cast<DCPathRegionOpInterface>(&o);
        return region && region.getLatencyBound();
      });
    }
    return r;
  }

  void run() {
    func.walk([&](func::CallOp) {
      container = true;
      return WalkResult::interrupt();
    });
    materializeBlock(func.getBody().front());
  }
};
} // namespace

// The reify's POST-CONDITION, asked of the whole module once: every kernel is a
// `dcp.module`, every loop and conditional a `dcp.*` region, and every call a
// `dcp.instance`, so nothing from func/affine/scf may survive. A survivor
// signals a reifier bug or the rare deliberately-unclosed fallback of a
// non-identity-forwarding while.
//
// Module-level rather than per-kernel: the interesting failure is a kernel that
// produced NO dcp region at all, which a per-kernel check gated on having one
// cannot see. Non-fatal by design.
static void verifyDcpClosed(ModuleOp module) {
  module.walk([&](Operation *op) {
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp, AffineIfOp, scf::IfOp,
            func::CallOp, func::FuncOp>(op))
      warn(Stage::Dcp, op)
          << "Op '" << op->getName().getStringRef()
          << "' survived reification; the post-schedule IR should hold only "
             "dcp.module kernels of dcp.* regions and instances, with every "
             "loop, conditional and call closed";
  });
}

//===----------------------------------------------------------------------===//
// Call machinery: rewrite a leaf-bound sync `func.call` into a `dcp.instance`,
// the call node the leaf datapath models as a CallUnit.
//===----------------------------------------------------------------------===//

// A dcp.instance referencing \p calleeAttr, copying the callee's timing
// contract verbatim (\p at anchors the symbol lookup).
//
// Reification is post-order over the call graph, so the callee is already a
// `dcp.module` and the contract copied here is the exact one it publishes,
// never the scheduler's provisional upper bound; the assert states that.
// `verifySymbolUses` holds the copy to the original from here on.
static DCPathInstanceOp makeInvoke(OpBuilder &b, Location loc,
                                   TypeRange resultTypes, ValueRange operands,
                                   FlatSymbolRefAttr calleeAttr, Operation *at,
                                   int64_t start) {
  auto callee = dyn_cast_or_null<DCPathModuleOp>(
      SymbolTable::lookupNearestSymbolFrom(at, calleeAttr));
  assert(callee && "a callee is reified before the caller that composes "
                   "against it, so it is already a dcp.module");
  return DCPathInstanceOp::create(b, loc, resultTypes, operands, calleeAttr,
                                  b.getI64IntegerAttr(start),
                                  optI64Attr(b, callee.getLatency()),
                                  determinacyAttr(b, callee.getDeterminacy()));
}

// Close one reified kernel over the dcp dialect: `func.func` becomes
// `dcp.module` and `func.return` becomes `dcp.output`. The point is the timing
// contract: as op arguments it is reached by a verifier, which a discardable
// `dcp.latency` never could, since the `dcp.` prefix names no registered
// dialect for a `verifyOperationAttribute` hook to dispatch on.
//
// Runs after this kernel's own body is reified, so it holds no `func.call` for
// its callee having converted first to invalidate.
static DCPathModuleOp toDcpModule(func::FuncOp func) {
  OpBuilder b(func);
  auto mod = DCPathModuleOp::create(b, func.getLoc(), func.getName(),
                                    func.getFunctionType(),
                                    DeterminacyEnum::Indeterminate);
  // Frontend provenance (`allo.signed`, the schedule keys) plus the scheduler's
  // provisional latency, which `setDcpLatencies` still holds itself to before
  // `stripScheduleCarrier` drops it.
  for (NamedAttribute a : func->getDiscardableAttrs())
    mod->setAttr(a.getName(), a.getValue());
  mod.setSymVisibilityAttr(func.getSymVisibilityAttr());
  mod.setArgAttrsAttr(func.getArgAttrsAttr());
  mod.setResAttrsAttr(func.getResAttrsAttr());

  mod.getBody().takeBody(func.getBody());
  Operation *ret = mod.getBody().front().getTerminator();
  b.setInsertionPoint(ret);
  DCPathOutputOp::create(b, ret->getLoc(), ret->getOperands());
  ret->erase();
  func.erase();
  return mod;
}

static void materializeFunc(func::FuncOp func, ScheduleModel &model,
                            const OperatorLibrary &lib) {
  Reifier{func, model, lib}.run();

  // A fully-deferred function (nothing materialized) still closes into a
  // `dcp.module` so the module is uniform, but publishes no contract: it never
  // went through scheduling.
  bool hasDCP = false;
  func.walk([&](Operation *op) {
    if (isa<DCPathPipelineOp, DCPathSequentialOp>(op)) {
      hasDCP = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  DCPathModuleOp mod = toDcpModule(func);
  if (hasDCP)
    setDcpLatencies(mod);
  stripScheduleCarrier(mod);
}

static void loadDependentDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<allo::AlloDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<AffineDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
}

// Post-order over the call graph: every callee is reified before the caller
// that composes against it, so `makeInvoke` reads the callee's exact
// `dcp.module` latency rather than the scheduler's provisional one.
//
// `done` keys on the ADDRESS of a func it has closed into a `dcp.module`, and
// so never dereferences one: only the funcs collected before any conversion
// are ever looked up, and those were all live at once, so no two share an
// address.
static void reifyCalleesFirst(func::FuncOp func, ScheduleModel &model,
                              const OperatorLibrary &lib,
                              llvm::DenseSet<Operation *> &done) {
  if (!done.insert(func.getOperation()).second)
    return;
  // An already-reified callee is a `dcp.module`, which this cast skips.
  SmallVector<func::FuncOp> callees;
  func.walk([&](func::CallOp call) {
    if (auto c = dyn_cast_or_null<func::FuncOp>(
            SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr())))
      callees.push_back(c);
  });
  for (func::FuncOp c : callees)
    reifyCalleesFirst(c, model, lib, done);
  materializeFunc(func, model, lib);
}

void mlir::allo::runPostScheduleConversion(ModuleOp module,
                                           ScheduleModel &model) {
  loadDependentDialects(*module->getContext());
  auto lib = OperatorLibrary::fromModule(module);
  // One `dcp.unit` per allocated instance, declared at the top of the module
  // so the symbols resolve whatever order the funcs below are reified in.
  OpBuilder b(module.getBody(), module.getBody()->begin());
  for (const ScheduleModel::AllocatedUnit &u : model.allocatedUnits())
    DCPathUnitOp::create(b, module.getLoc(), b.getStringAttr(u.name),
                         FlatSymbolRefAttr::get(b.getContext(), u.opType));
  SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
  llvm::DenseSet<Operation *> reified;
  for (func::FuncOp func : funcs)
    reifyCalleesFirst(func, model, lib, reified);
  verifyDcpClosed(module);
  model.record(module);
} // namespace mlir::allo
