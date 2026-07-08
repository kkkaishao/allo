/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The scheduler's reification step: lower the solved schedule -- carried
// transiently as `allo.sched.*` attributes -- into `allo.dcp.*` ops, so the
// post-schedule IR is closed over the dcp dialect (no affine/scf control flow
// survives; only `arith` constant/index glue stays as intended mixed-IR).
//
// `Reifier` walks a function's loop/region tree as ONE post-order recursion --
// the structural twin of the scheduler's `scheduleBlock` / `scheduleRegion`
// descent (`materializeBlock` shares the same `enumerateRegions` partitioner,
// and a region is materialized only after its body, so deepest-first ordering
// falls out for free). Every control construct becomes a dcp region:
//   * counted `for`         -> `dcp.pipeline` (`dcp.uncondition` terminator)
//   * `scf.while`           -> `dcp.pipeline` (`dcp.condition` terminator)
//   * `affine.if` / `scf.if`-> `dcp.select`
//   * straight-line span    -> `dcp.sequential`
// `ii` present vs absent distinguishes pipelined (leaf / Phase-B level / C2
// flushing while) from sequential (imperfect wrapper / data-dependent while).
// Registers / muxes / control are NOT ops -- they are derived at hw lowering.
//
// `materializeModuleToDCP` is the callable `allo-schedule` runs as its final
// phase (module scope: `dcp.operator` declarations are module-level symbols);
// the `allo-materialize-dcpath` pass is a thin standalone wrapper.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/MaterializeDCPath.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/Utils.h"

#include "allo/Conversion/Passes.h"
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
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_CONVERTSCHEDULETODCPPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

static int64_t i64Attr(Operation *op, StringRef name) {
  return cast<IntegerAttr>(op->getAttr(name)).getInt();
}

// An `i64` attribute for an optional value, or a null attribute (elided) when
// absent -- the shape every optional dcp schedule attr
// (`ii`/`length`/`latency`/ `start`/`trip`) is built with.
static IntegerAttr optI64Attr(Builder &b, std::optional<int64_t> v) {
  return v ? b.getI64IntegerAttr(*v) : IntegerAttr();
}

//===----------------------------------------------------------------------===//
// Operator declarations: one `dcp.operator` symbol per (kind, width), emitted
// at module top and referenced by the compute/memory ops.
//===----------------------------------------------------------------------===//
namespace {
class OperatorEmitter {
public:
  OperatorEmitter(ModuleOp module, const OperatorLibrary &lib)
      : module(module), lib(lib) {}

  // The symbol of the `dcp.operator` characterizing \p op, creating it on first
  // use. The kind is the op's mnemonic (e.g. "addf", "load"); the width is the
  // classified datapath width.
  FlatSymbolRefAttr get(Operation &op) {
    OperatorChar oc = lib.lookup(&op);
    unsigned width = classify(&op).width;
    std::string kind = op.getName().stripDialect().str();
    std::string name = kind + "_w" + std::to_string(width);

    if (auto it = cache.find(name); it != cache.end())
      return it->second;

    OpBuilder b(module.getContext());
    b.setInsertionPointToStart(module.getBody());
    FloatAttr inDelay =
        oc.inDelay != 0.0 ? b.getF32FloatAttr(oc.inDelay) : FloatAttr();
    FloatAttr outDelay =
        oc.outDelay != 0.0 ? b.getF32FloatAttr(oc.outDelay) : FloatAttr();
    StringAttr impl = oc.impl.empty() ? StringAttr() : b.getStringAttr(oc.impl);
    DCPathOperatorOp::create(b, module.getLoc(), name, kind, width, oc.latency,
                             oc.pipelined, inDelay, outDelay, impl);
    auto ref = FlatSymbolRefAttr::get(module.getContext(), name);
    cache[name] = ref;
    return ref;
  }

private:
  ModuleOp module;
  const OperatorLibrary &lib;
  llvm::StringMap<FlatSymbolRefAttr> cache;
};
} // namespace
//===----------------------------------------------------------------------===//
// Per-op conversion.
//===----------------------------------------------------------------------===//

// Convert \p op (an op of the scheduled loop body) into its `dcp` equivalent in
// the pipeline block \p b is inserting into, mapping its results in \p map. Ops
// that are not compute/memory (constants, address arithmetic) are cloned as-is.
static void convertOp(Operation &op, OpBuilder &b, IRMapping &map,
                      OperatorEmitter &ops) {
  Location loc = op.getLoc();
  int64_t start = op.hasAttr(sched::kStartTimeAttr)
                      ? i64Attr(&op, sched::kStartTimeAttr)
                      : 0;
  auto rm = [&](Value v) { return map.lookupOrDefault(v); };
  auto remap = [&](auto values) {
    SmallVector<Value> out;
    for (Value v : values)
      out.push_back(rm(v));
    return out;
  };
  // Carry the sub-cycle start time (from the chaining solve) onto the dcp op.
  auto setZ = [&](Operation *dst) {
    if (auto z = op.getAttrOfType<FloatAttr>(sched::kStartTimeInCycleAttr))
      dst->setAttr("z", z);
  };
  // Keep an op verbatim inside the region, preserving its scheduled start so
  // the schedule export can still report it (streams, constants, address
  // arithmetic).
  auto cloneKept = [&]() {
    Operation *c = b.clone(op, map);
    if (op.hasAttr(sched::kStartTimeAttr)) {
      c->setAttr("start", b.getI64IntegerAttr(start));
      setZ(c);
    }
  };

  if (auto l = dyn_cast<AffineLoadOp>(&op)) {
    auto nw = DCPathLoadOp::create(b, loc, l.getType(), rm(l.getMemRef()),
                                   remap(l.getMapOperands()), l.getAffineMap(),
                                   (uint64_t)start, ops.get(op), IntegerAttr(),
                                   IntegerAttr());
    setZ(nw);
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto l = dyn_cast<memref::LoadOp>(&op)) {
    AffineMap id = AffineMap::getMultiDimIdentityMap(l.getIndices().size(),
                                                     b.getContext());
    auto nw = DCPathLoadOp::create(b, loc, l.getType(), rm(l.getMemRef()),
                                   remap(l.getIndices()), id, (uint64_t)start,
                                   ops.get(op), IntegerAttr(), IntegerAttr());
    setZ(nw);
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto s = dyn_cast<AffineStoreOp>(&op)) {
    auto nw = DCPathStoreOp::create(
        b, loc, rm(s.getValueToStore()), rm(s.getMemRef()),
        remap(s.getMapOperands()), s.getAffineMap(), (uint64_t)start,
        ops.get(op), IntegerAttr(), IntegerAttr());
    setZ(nw);
    return;
  }
  if (auto s = dyn_cast<memref::StoreOp>(&op)) {
    AffineMap id = AffineMap::getMultiDimIdentityMap(s.getIndices().size(),
                                                     b.getContext());
    auto nw = DCPathStoreOp::create(b, loc, rm(s.getValueToStore()),
                                    rm(s.getMemRef()), remap(s.getIndices()),
                                    id, (uint64_t)start, ops.get(op),
                                    IntegerAttr(), IntegerAttr());
    setZ(nw);
    return;
  }
  // Streams stay as FIFO ops, not compute; keep them verbatim with their start.
  if (isa<StreamGetOp, StreamPutOp>(&op)) {
    cloneKept();
    return;
  }
  // An internal-buffer allocation is a declaration
  if (isa<memref::AllocOp, memref::AllocaOp, StreamCreateOp>(&op)) {
    cloneKept();
    return;
  }
  // A scheduled single-result op (not a constant) is a compute op.
  if (op.getNumResults() == 1 && op.hasAttr(sched::kStartTimeAttr) &&
      !isa<arith::ConstantOp>(op)) {
    auto nw = DCPathComputeOp::create(
        b, loc, op.getResult(0).getType(), remap(op.getOperands()), ops.get(op),
        b.getI64IntegerAttr(start), FlatSymbolRefAttr());
    // Carry the source op's own attributes onto the compute op (e.g.
    // arith.cmpi's `predicate`) as discardable attrs, so the emitter can
    // recover op-specific semantics the operator *kind* alone does not encode
    // -- this covers any such op, not just compare. Skip the scheduling carrier
    // (`allo.sched.*`), which the reifier reads explicitly (start / z above);
    // the compute op's own inherent attrs (op_type/start/unit) do not collide
    // with a source arith op.
    for (NamedAttribute attr : op.getAttrs())
      if (!attr.getName().getValue().starts_with("allo.sched."))
        nw->setAttr(attr.getName(), attr.getValue());
    setZ(nw);
    map.map(op.getResult(0), nw.getResult());
    return;
  }
  // Constants / address arithmetic: keep verbatim inside the region.
  cloneKept();
}

//===----------------------------------------------------------------------===//
// Region metadata read from `allo.sched.regions`.
//===----------------------------------------------------------------------===//

namespace {
// The schedule numbers of one region, read from its `allo.sched.regions`
// descriptor (all optional -- absent = "data-dependent / not applicable"). The
// reifier reads these onto the dcp op; the region *kind* is not stored (the
// recursion dispatches on the anchor op type, not on the descriptor).
struct RegionInfo {
  std::optional<int64_t> ii;
  std::optional<int64_t> length;
  std::optional<int64_t> latency;
  bool latencyBound = false;
  std::optional<int64_t> absStart;    // prefix-sum of prior-region latencies
  std::optional<int64_t> parent;      // absorbed into a Phase B level
  std::optional<int64_t> parentStart; // a child's start within the level's II
};
} // namespace

// Read the per-region descriptors and compute each region's absolute start
// cycle as the prefix sum of prior-region latencies (regions compose in program
// order); the sum goes unknown as soon as a region latency is unknown.
static llvm::DenseMap<int64_t, RegionInfo> readRegions(ArrayAttr regionsAttr) {
  llvm::DenseMap<int64_t, RegionInfo> byId;
  SmallVector<int64_t> ids;
  for (Attribute a : regionsAttr) {
    auto d = cast<DictionaryAttr>(a);
    int64_t id = d.getAs<IntegerAttr>(sched::kRegionKeyId).getInt();
    RegionInfo r;
    if (auto ii = d.getAs<IntegerAttr>(sched::kRegionKeyII))
      r.ii = ii.getInt();
    if (auto len = d.getAs<IntegerAttr>(sched::kRegionKeyLength))
      r.length = len.getInt();
    if (auto lat = d.getAs<IntegerAttr>(sched::kRegionKeyLatency))
      r.latency = lat.getInt();
    r.latencyBound = d.get(sched::kRegionKeyLatencyBound) != nullptr;
    if (auto p = d.getAs<IntegerAttr>(sched::kRegionKeyParent))
      r.parent = p.getInt();
    if (auto ps = d.getAs<IntegerAttr>(sched::kRegionKeyParentStart))
      r.parentStart = ps.getInt();
    byId[id] = r;
    ids.push_back(id);
  }

  llvm::sort(ids);
  int64_t acc = 0;
  bool known = true;
  for (int64_t id : ids) {
    // A region absorbed into a Phase B level does not occupy a top-level slot;
    // its start is derived within the level (`parentStart`), not the prefix
    // sum.
    if (byId[id].parent)
      continue;
    if (known)
      byId[id].absStart = acc;
    if (known && byId[id].latency)
      acc += *byId[id].latency;
    else
      known = false;
  }
  return byId;
}

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

// The `i64` value of an optional integer attribute, or nullopt.
static std::optional<int64_t> optI64(Operation *op, StringRef name) {
  if (auto a = op->getAttrOfType<IntegerAttr>(name))
    return a.getInt();
  return std::nullopt;
}

// One full run of an already-materialized child region (its per-invocation
// latency), used to size a sequential wrapper's II. A `dcp.sequential` runs
// once (its `length`); a counted `dcp.pipeline` runs `length + (trip-1)*ii`; a
// while pipeline or a `dcp.select` guard is data-dependent (nullopt -- so a
// wrapper enclosing one has no static II); anything else (arith glue) is
// latency 0. nullopt if a child's own trip/length is unknown.
static std::optional<int64_t> perInvocationLatency(Operation *op) {
  if (isa<DCPathSelectOp>(op))
    return std::nullopt;
  if (isa<DCPathSequentialOp>(op))
    return optI64(op, "length");
  if (auto pipe = dyn_cast<DCPathPipelineOp>(op)) {
    if (pipe.isWhileLoop())
      return std::nullopt;
    std::optional<int64_t> trip = optI64(op, "trip");
    std::optional<int64_t> len = optI64(op, "length");
    std::optional<int64_t> ii = optI64(op, "ii");
    if (!trip || !len || !ii)
      return std::nullopt;
    return *len + (*trip - 1) * *ii;
  }
  return 0;
}

// The first `allo.sched.region` id tagged anywhere under \p container (its body
// leaf ops). For a leaf loop / simple while, every body op carries the same id.
static std::optional<int64_t> firstRegionId(Operation *container) {
  std::optional<int64_t> id;
  container->walk([&](Operation *o) {
    if (auto a = o->getAttrOfType<IntegerAttr>(sched::kRegionIdAttr)) {
      id = a.getInt();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return id;
}

// The first `allo.sched.region` id among a straight-line span's (leaf) ops.
static std::optional<int64_t> firstRegionId(ArrayRef<Operation *> ops) {
  for (Operation *op : ops)
    if (auto a = op->getAttrOfType<IntegerAttr>(sched::kRegionIdAttr))
      return a.getInt();
  return std::nullopt;
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
// region), else the runtime SSA `index` value (an scf.for data-dependent
// bound). A constant `lb` with a *dynamic* ub (`for i in range(1, n)`) is the
// C1 case; a genuinely runtime lb/step (`for i in range(m, n)` with `m` loaded)
// is Phase 2. An affine.for with a non-constant (symbolic) lb defaults to 0 --
// unhandled, a separate cut.
struct LoopBounds {
  int64_t lb = 0, step = 1; // used iff the matching Value is null (constant)
  Value lbVal, stepVal;     // runtime bound (an scf.for SSA lb/step)
};
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

// The runtime upper bound of a counted loop whose trip is NOT a compile-time
// constant, as the 0-based iteration count the reifier's counter runs against.
// Only an scf.for carries a runtime bound (a memory-loaded / non-affine ub);
// the ub IS the trip count because the reifier's counter is 0-based and maps
// the source IV to it directly -- the same lb=0/step=1 normalization every
// counted loop here already relies on (see materializeLoopToPipeline's counter
// map). An affine.for with a symbolic bound would need its trip derived from
// the map; left unwired for a later cut (the emitter then rejects it, not
// misindexes).
static Value dynamicTripBound(LoopLikeOpInterface loop) {
  auto scfLoop = dyn_cast<scf::ForOp>(loop.getOperation());
  return scfLoop ? scfLoop.getUpperBound() : Value();
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

// Rewrite an `scf.while` into a while `dcp.pipeline` -- `trip` unset,
// terminated by `dcp.condition` (the condition value + the loop-carried
// next-values). A straight-line while flushing-pipelines (`ii` from its
// descriptor); a nested-loop while runs sequentially (empty descriptor -> `ii`
// unset), its after-block already materialized so its dcp children clone in
// verbatim. Both the before-arg and after-arg of a slot map to the same
// iter-arg (identity forwarding, required); the counter block-arg is a
// free-running index.
static void materializeWhilePipeline(const RegionInfo &r, scf::WhileOp w,
                                     OperatorEmitter &operators) {
  OpBuilder b(w);
  Location loc = w.getLoc();

  IntegerAttr startA;
  if (isa<func::FuncOp>(w->getParentOp()))
    startA = optI64Attr(b, r.absStart);
  ValueRange inits = w.getInits();
  auto pipe = DCPathPipelineOp::create(
      b, loc, w.getResultTypes(), /*lbBound=*/Value(), /*dynamicBound=*/Value(),
      /*stepBound=*/Value(), inits, /*trip=*/IntegerAttr(),
      /*lb=*/IntegerAttr(),
      /*step=*/IntegerAttr(), optI64Attr(b, r.ii), startA,
      optI64Attr(b, r.length), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr());
  Block *blk = createCounterBlock(b, pipe, inits, loc);

  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  IRMapping map;
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    map.map(before.getArgument(j), blk->getArgument(j + 1));
    map.map(after.getArgument(j), blk->getArgument(j + 1));
  }

  b.setInsertionPointToEnd(blk);
  for (Operation &op : before.without_terminator())
    convertOp(op, b, map, operators);
  for (Operation &op : after.without_terminator())
    convertOp(op, b, map, operators);

  Value cond = map.lookupOrDefault(w.getConditionOp().getCondition());
  SmallVector<Value> carried;
  for (Value v : w.getYieldOp().getOperands())
    carried.push_back(map.lookupOrDefault(v));
  DCPathConditionOp::create(b, loc, cond, carried);

  for (auto [old, nw] : llvm::zip(w.getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  w.erase();
}

// Rewrite one counted loop (affine.for or scf.for) into a dcp.pipeline: convert
// its body ops (an already-materialized child `dcp.pipeline`/`dcp.sequential`
// -- of a Phase B level, or of an imperfect wrapper -- is cloned verbatim). The
// trip count is recorded only when it is a compile-time constant.
static void materializeLoopToPipeline(const RegionInfo &r,
                                      LoopLikeOpInterface loop,
                                      OperatorEmitter &operators) {
  Operation *loopOp = loop.getOperation();
  OpBuilder b(loopOp);
  Location loc = loop.getLoc();

  // A child of a Phase B level starts at its scheduled offset within the
  // level's II (`parentStart`). Otherwise the absolute start cycle is
  // meaningful only at the top level, where regions compose in program order; a
  // plain loop-nested pipeline starts at a different cycle each outer iteration
  // (left unset).
  IntegerAttr startA = optI64Attr(b, r.parentStart);
  if (!startA && isa<func::FuncOp>(loopOp->getParentOp()))
    startA = optI64Attr(b, r.absStart);

  ValueRange inits = loop.getInits();
  // A runtime (data-dependent) trip: the count is not a compile-time constant,
  // so wire the loop's upper bound as the `dynamicBound` operand (the 0-based
  // iteration count for the reifier's `lb=0`/`step=1` counter). Only an scf.for
  // has a runtime bound (a memory-loaded / non-affine ub); affine bounds are
  // constant/affine-symbol. Restricted to the lb=0/step=1 form the 0-based
  // counter assumes; a general dynamic lb/step is left for the enclosing pass.
  Value dynamicBound;
  if (!constantTripOf(loop))
    dynamicBound = dynamicTripBound(loop);
  // Carry the source loop's lb/step so the emitter's induction register runs
  // the real IV -- correct for a `lb != 0` / `step != 1` loop even when the ub
  // is a runtime bound. Each rides an attribute (elided when the default 0/1)
  // if compile-time, else an operand (a runtime range start / stride).
  LoopBounds bounds = lbStepOf(loop);
  std::optional<int64_t> lbAttr, stepAttr;
  if (!bounds.lbVal && bounds.lb != 0)
    lbAttr = bounds.lb;
  if (!bounds.stepVal && bounds.step != 1)
    stepAttr = bounds.step;
  auto pipe = DCPathPipelineOp::create(
      b, loc, loopOp->getResultTypes(), bounds.lbVal, dynamicBound,
      bounds.stepVal, inits, optI64Attr(b, constantTripOf(loop)),
      optI64Attr(b, lbAttr), optI64Attr(b, stepAttr), optI64Attr(b, r.ii),
      startA, optI64Attr(b, r.length), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr());
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
    convertOp(op, b, map, operators);

  Operation *term = body->getTerminator();
  SmallVector<Value> yields;
  for (Value v : term->getOperands())
    yields.push_back(map.lookupOrDefault(v));
  DCPathUnconditionOp::create(b, term->getLoc(), yields);

  for (auto [old, nw] : llvm::zip(loopOp->getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  loopOp->erase();
}

// Rewrite a straight-line (acyclic) region into a dcp.sequential. A region of
// only declarations is left in place (sourced directly by identity, like a
// loop-invariant tie-off / func-arg memref), so it forms no region and threads
// no cross-region SSA result; anything else is wrapped, with values used after
// the region yielded as sequential results.
static void materializeSequential(const RegionInfo &r,
                                  ArrayRef<Operation *> ops,
                                  OperatorEmitter &operators) {
  SmallVector<Operation *> body;
  for (Operation *op : ops)
    if (!op->hasTrait<OpTrait::IsTerminator>())
      body.push_back(op);
  if (body.empty() || llvm::all_of(body, [](Operation *op) {
        return isa<arith::ConstantOp, memref::AllocOp, memref::AllocaOp,
                   StreamCreateOp>(op);
      }))
    return;

  llvm::SmallPtrSet<Operation *, 8> inRegion(body.begin(), body.end());
  SmallVector<Value> escaping;
  for (Operation *op : body)
    for (Value res : op->getResults())
      if (llvm::any_of(res.getUsers(),
                       [&](Operation *u) { return !inRegion.contains(u); }))
        escaping.push_back(res);

  OpBuilder b(body.front());
  Location loc = body.front()->getLoc();

  IntegerAttr startA;
  if (isa<func::FuncOp>(body.front()->getParentOp()))
    startA = optI64Attr(b, r.absStart);
  SmallVector<Type> resultTypes(
      llvm::map_range(escaping, [](Value v) { return v.getType(); }));
  auto seq = DCPathSequentialOp::create(
      b, loc, resultTypes, startA, optI64Attr(b, r.length),
      optI64Attr(b, r.latency), r.latencyBound ? b.getUnitAttr() : UnitAttr());
  Block *blk = b.createBlock(&seq.getBody());

  IRMapping map;
  b.setInsertionPointToEnd(blk);
  for (Operation *op : body)
    convertOp(*op, b, map, operators);

  SmallVector<Value> yields(llvm::map_range(
      escaping, [&](Value v) { return map.lookupOrDefault(v); }));
  DCPathUnconditionOp::create(b, loc, yields);

  for (auto [orig, res] : llvm::zip(escaping, seq.getResults()))
    orig.replaceAllUsesWith(res);
  for (Operation *op : llvm::reverse(body))
    op->erase();
}

// Rewrite each region's `latency` to its single-run form -- one full invocation
// of that region alone (a pipeline's `length + (trip-1)*ii`, a sequential's
// `length`) -- so a nested region reports its own run, not that run folded
// across all enclosing loop trips. The whole-kernel `dcp.latency` is then the
// sum of the top-level regions' single runs (they compose in program order), so
// the func total and the per-region latencies are consistent by construction.
// `perInvocationLatency` reads only `ii`/`length`/`trip` (already single-run
// for a synthesized wrapper), so this is idempotent for wrappers and un-folds
// only the leaf/level regions whose descriptor latency folded the enclosing
// trips.
static void setDcpLatencies(func::FuncOp func) {
  Builder b(func.getContext());
  func.walk([&](Operation *op) {
    if (isa<DCPathPipelineOp, DCPathSequentialOp>(op))
      if (std::optional<int64_t> sr = perInvocationLatency(op))
        op->setAttr("latency", b.getI64IntegerAttr(*sr));
  });
  int64_t total = 0;
  bool known = true, bounded = false;
  for (Operation &op : func.getBody().front()) {
    if (!isa<DCPathPipelineOp, DCPathSequentialOp, DCPathSelectOp>(op))
      continue;
    auto lat = op.getAttrOfType<IntegerAttr>("latency");
    if (!lat) {
      known = false; // a data-dependent region leaves the kernel total unknown
      break;
    }
    total += lat.getInt();
    bounded |= op.hasAttr("latency_bound");
  }
  if (known) {
    func->setAttr("dcp.latency", b.getI64IntegerAttr(total));
    if (bounded)
      func->setAttr("dcp.latency_bound", b.getUnitAttr());
  }
}

// Remove the `allo.sched.*` schedule carrier now that the schedule lives in the
// dcp ops. The derived whole-kernel latency is set separately by
// `setDcpLatencies` from the materialized region tree.
static void stripScheduleCarrier(func::FuncOp func) {
  auto stripFrom = [](Operation *op) {
    SmallVector<StringRef> names;
    for (NamedAttribute a : op->getAttrs())
      if (a.getName().getValue().starts_with("allo.sched."))
        names.push_back(a.getName().getValue());
    for (StringRef n : names)
      op->removeAttr(n);
  };
  stripFrom(func);
  func.walk(stripFrom);
}

namespace {
// Post-order lowering of one function's loop/region tree. Mirrors the
// scheduler's `scheduleBlock` / `scheduleRegion` descent, materializing each
// region bottom-up (a loop is wrapped only after its body is materialized, so
// deepest-first ordering falls out of the recursion). A counted for-loop always
// becomes a `dcp.pipeline` -- leaf, Phase-B pipelined level, or sequential
// wrapper, the three differing only in where the II comes from; a straight-line
// span becomes a `dcp.sequential`; a while / opaque `if` is left raw wrapping
// its materialized children.
struct Reifier {
  func::FuncOp func;
  const llvm::DenseMap<int64_t, RegionInfo> &regions;
  OperatorEmitter &operators;

  // The descriptor for a region id (empty when the id is unknown -- an
  // all-constant span, or a synthesized wrapper handled separately).
  const RegionInfo &infoFor(std::optional<int64_t> id) {
    static const RegionInfo empty{};
    if (id)
      if (auto it = regions.find(*id); it != regions.end())
        return it->second;
    return empty;
  }

  void materializeBlock(Block &block) {
    for (const SchedRegion &region : enumerateRegions(block))
      materializeRegion(region);
  }

  void materializeRegion(const SchedRegion &region) {
    if (region.kind == allo::RegionKind::StraightLine) {
      materializeSequential(infoFor(firstRegionId(region.ops)), region.ops,
                            operators);
      return;
    }
    Operation *anchor = region.anchor();
    if (isa<AffineForOp, scf::ForOp>(anchor)) {
      materializeCountedLoop(cast<LoopLikeOpInterface>(anchor));
    } else if (auto w = dyn_cast<scf::WhileOp>(anchor)) {
      if (hasNestedLoop(w)) {
        // A while with a nested loop cannot flush-pipeline (the recurrence
        // threads the inner loop, so its per-iteration length is
        // data-dependent): materialize the after-block into dcp regions, then
        // close the while into a SEQUENTIAL while dcp.pipeline -- the same
        // merged-body reify, with `ii` unset (no static II). The merge needs
        // identity forwarding; a non-identity while is left raw (rare).
        materializeBlock(w.getAfter().front());
        if (whileHasIdentityForwarding(w))
          materializeWhilePipeline(RegionInfo{}, w, operators);
      } else {
        // A straight-line while became a flushing pipeline (ii from
        // descriptor).
        materializeWhilePipeline(infoFor(firstRegionId(anchor)), w, operators);
      }
    } else if (isa<scf::IfOp, AffineIfOp>(anchor)) {
      // An opaque guard left by if-conversion (a branch holds a loop / stream /
      // call it could not speculate): materialize each branch, then close the
      // `if` into a dcp.select. An scf.if already has an i1 condition; an
      // affine.if's IntegerSet is materialized into one (`affineIfCondition`).
      for (Region &branch : anchor->getRegions())
        if (!branch.empty())
          materializeBlock(branch.front());
      OpBuilder b(anchor);
      Value cond = isa<AffineIfOp>(anchor)
                       ? affineIfCondition(b, cast<AffineIfOp>(anchor))
                       : cast<scf::IfOp>(anchor).getCondition();
      closeIntoDcpSelect(b, anchor, cond);
    }
  }

  // Materialize an affine.if's IntegerSet predicate into an i1: the conjunction
  // of its constraints (each `expr >= 0`, or `== 0` for an equality), built
  // with `expandAffineExpr` + `cmpi` + `andi`. Mirrors upstream
  // AffineIfLowering. The ops are inserted before `b`'s point (ahead of the
  // dcp.select) and reference the loop IVs -- which the enclosing wrapper later
  // rewires to its counter.
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

  // Close a scheduled if (scf.if / affine.if, branches already materialized
  // into dcp regions) into a dcp.select with condition \p cond: move each
  // branch body verbatim, rewrite its yield to a dcp.uncondition, and forward
  // the results. Latency is left unset (a data-dependent guard has no static
  // count -- an affine active-count could fill it later).
  void closeIntoDcpSelect(OpBuilder &b, Operation *ifOp, Value cond) {
    auto sel = DCPathSelectOp::create(b, ifOp->getLoc(), ifOp->getResultTypes(),
                                      cond, /*start=*/IntegerAttr(),
                                      /*latency=*/IntegerAttr(),
                                      /*latency_bound=*/UnitAttr());
    sel.getThenRegion().takeBody(ifOp->getRegion(0));
    if (!ifOp->getRegion(1).empty())
      sel.getElseRegion().takeBody(ifOp->getRegion(1));
    for (Region *r : {&sel.getThenRegion(), &sel.getElseRegion()}) {
      if (r->empty())
        continue;
      Operation *term = r->front().getTerminator();
      OpBuilder yb(term);
      DCPathUnconditionOp::create(yb, term->getLoc(), term->getOperands());
      term->erase();
    }
    for (auto [oldR, newR] : llvm::zip(ifOp->getResults(), sel.getResults()))
      oldR.replaceAllUsesWith(newR);
    ifOp->erase();
  }

  // A counted for-loop -> dcp.pipeline. The three cases are distinguished
  // BEFORE the body is materialized (so nested loops are still raw affine/scf
  // ops):
  //   * Phase-B level (co-scheduled, iterations overlap): materialize only the
  //     child loops, then wrap -- the level's loose leaf ops convert in place;
  //   * sequential wrapper (imperfect Phase-A / non-flattened band):
  //   materialize
  //     every sub-region, then wrap with ii = Σ child invocation latency;
  //   * leaf innermost: wrap directly, ii from the solved descriptor.
  void materializeCountedLoop(LoopLikeOpInterface loop) {
    Operation *op = loop.getOperation();
    Block &body = loop.getLoopRegions().front()->front();
    if (auto level = op->getAttrOfType<IntegerAttr>(sched::kLevelAttr)) {
      for (const SchedRegion &sub : enumerateRegions(body))
        if (sub.kind == allo::RegionKind::Loop &&
            isa<AffineForOp, scf::ForOp>(sub.anchor()))
          materializeCountedLoop(cast<LoopLikeOpInterface>(sub.anchor()));
      materializeLoopToPipeline(infoFor(level.getInt()), loop, operators);
    } else if (hasNestedLoop(loop)) {
      materializeBlock(body);
      materializeLoopToPipeline(sequentialWrapperInfo(loop), loop, operators);
    } else {
      materializeLoopToPipeline(infoFor(firstRegionId(op)), loop, operators);
    }
  }

  // The synthesized descriptor of a residual sequential wrapper, derived from
  // its now-materialized children. Iterations do not overlap, so its II =
  // single body length = Σ perInvocationLatency(child). If any child's
  // invocation latency is data-dependent (a dynamic inner trip), the body
  // length -- hence the II and latency -- has no static value: leave
  // `ii`/`length`/`latency` unset (the wrapper is a done-based sequential
  // controller, not pipelined).
  RegionInfo sequentialWrapperInfo(LoopLikeOpInterface loop) {
    int64_t bodyLen = 0;
    bool known = true, bounded = false;
    for (Operation &o : loop.getLoopRegions().front()->front()) {
      if (o.hasTrait<OpTrait::IsTerminator>())
        continue;
      std::optional<int64_t> l = perInvocationLatency(&o);
      if (!l)
        known = false;
      else
        bodyLen += *l;
      if (o.hasAttr("latency_bound"))
        bounded = true;
    }
    RegionInfo r;
    if (known) {
      r.ii = bodyLen;
      r.length = bodyLen;
      if (std::optional<int64_t> trip = constantTripOf(loop)) {
        r.latency = *trip * bodyLen;
        r.latencyBound = bounded;
      }
    }
    return r;
  }

  void run() { materializeBlock(func.getBody().front()); }
};
} // namespace

// Post-condition: after reification every loop and conditional is a `dcp.*`
// region, so no affine/scf control-flow op may survive. Walk the whole function
// and warn on any that did -- a reifier bug, or a rare deliberately-unclosed
// fallback (a non-identity-forwarding while, see `materializeRegion`). Silent
// means the IR is fully closed over the dcp dialect. Non-fatal by design.
static void verifyControlFlowEliminated(func::FuncOp func) {
  func.walk([&](Operation *op) {
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp, AffineIfOp, scf::IfOp>(op))
      warn(Stage::Dcp, op)
          << "control-flow op '" << op->getName().getStringRef()
          << "' survived reification -- the post-schedule IR should hold only "
             "dcp.* regions (every loop/conditional closed)";
  });
}

static void materializeFunc(func::FuncOp func, OperatorEmitter &operators) {
  auto regionsAttr = func->getAttrOfType<ArrayAttr>(sched::kRegionsAttr);
  if (!regionsAttr)
    return;
  llvm::DenseMap<int64_t, RegionInfo> regions = readRegions(regionsAttr);

  Reifier{func, regions, operators}.run();

  // Retire the schedule carrier once the schedule is expressed as dcp ops. A
  // fully-deferred function (nothing materialized) keeps its attributes and is
  // not checked (it never went through scheduling).
  bool hasDCP = false;
  func.walk([&](Operation *op) {
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(op)) {
      hasDCP = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasDCP) {
    setDcpLatencies(func);
    stripScheduleCarrier(func);
    verifyControlFlowEliminated(func);
  }
}

namespace mlir::allo {

void materializeModuleToDCP(ModuleOp module, const OperatorLibrary &lib) {
  OperatorEmitter operators(module, lib);
  SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
  for (func::FuncOp func : funcs)
    materializeFunc(func, operators);
}

} // namespace mlir::allo

namespace {

struct ConvertScheduleToDCPPass
    : public allo::impl::ConvertScheduleToDCPPassBase<
          ConvertScheduleToDCPPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<allo::AlloDialect, arith::ArithDialect, func::FuncDialect,
                    affine::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    OperatorLibrary loaded;
    if (!operatorLibrary.empty()) {
      llvm::Expected<OperatorLibrary> parsed =
          OperatorLibrary::loadFile(operatorLibrary);
      if (!parsed) {
        error(Stage::Dcp, module) << llvm::toString(parsed.takeError());
        return signalPassFailure();
      }
      loaded = std::move(*parsed);
    } else {
      loaded = OperatorLibrary::defaultLibrary();
    }
    allo::materializeModuleToDCP(module, loaded);
  }
};

} // namespace
