/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// DatapathBuilder implementation. Step A allocates cells and applies the
// trivial binding; Step B derives the interconnect via the register-depth rule
// `d*II + (t_consumer - t_producer) - latency(producer)` (d = loop-carried
// distance). The induction variable is modelled as a region counter that
// "produces" the loop index at cycle 0, so a memory access at cycle t reads the
// index through a t-deep shift register -- address timing falls out of the same
// rule. See DatapathBuilder.h.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Reservation.h" // verifyBinding (MRT legality)

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h"   // characterize (storage shape)
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <deque>

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

namespace {

//===----------------------------------------------------------------------===//
// Pure DCP structural readers.
//===----------------------------------------------------------------------===//

// Start cycle (region-relative) carried by a dcp compute/load/store op.
int64_t dcpStart(Operation *op) {
  return cast<IntegerAttr>(op->getAttr("start")).getInt();
}

// The dcp.operator characterizing a compute/load op, or null.
dcp::DCPathOperatorOp dcpOperator(Operation *op) {
  FlatSymbolRefAttr sym;
  if (auto c = dyn_cast<dcp::DCPathComputeOp>(op))
    sym = c.getOpTypeAttr();
  else if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    sym = l.getOpTypeAttr();
  if (!sym)
    return {};
  return SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(op, sym);
}

// Result latency of a producing dcp op (0 if uncharacterized).
unsigned dcpLatency(Operation *op) {
  dcp::DCPathOperatorOp opr = dcpOperator(op);
  return opr ? static_cast<unsigned>(opr.getLatency()) : 0;
}

Value dcpMemref(Operation *op) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return l.getMemref();
  if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
    return s.getMemref();
  return nullptr;
}

// The addressing of a dcp memory access: its affine map plus index operands.
void dcpAddressing(Operation *op, AffineMap &map,
                   SmallVector<Value> &operands) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    map = l.getMap();
    operands.assign(l.getIndices().begin(), l.getIndices().end());
  } else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op)) {
    map = s.getMap();
    operands.assign(s.getIndices().begin(), s.getIndices().end());
  }
}

// The body block of a dcp region op. A guard (dcp.select) has no `else` here
// (result-mux guards are not yet lowered), so its body is the `then` branch --
// which holds the guarded sub-schedule (child regions), gated by the predicate.
Block *regionBody(Operation *regionOp) {
  if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp))
    return &pipe.getBody().front();
  if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
    return &sel.getThenRegion().front();
  return &cast<dcp::DCPathSequentialOp>(regionOp).getBody().front();
}

// Trace a pipeline iter-arg (0-based) back to the op defining its next value,
// counting one loop-carried distance per iter_arg-to-iter_arg shift. This is
// the recurrence distance the scheduler solved against. Reads the loop-carried
// next-values through `getCarriedValues()` (the `dcp.uncondition` operands of a
// counted loop, or the `dcp.condition`'s carried operands of a while -- which
// skip its leading condition), so iter-arg k always maps to carried[k].
std::pair<Operation *, unsigned> traceIterArgSource(dcp::DCPathPipelineOp pipe,
                                                    unsigned iterArg) {
  Block &body = pipe.getBody().front();
  OperandRange carried = pipe.getCarriedValues();
  Value v = carried[iterArg];
  unsigned distance = 0;
  llvm::SmallDenseSet<unsigned> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getOwner() != &body || arg.getArgNumber() == 0 ||
        !seen.insert(arg.getArgNumber()).second)
      return {nullptr, 0};
    ++distance;
    v = carried[arg.getArgNumber() - 1]; // block arg (k+1) -> carried[k]
  }
  Operation *def = v.getDefiningOp();
  return def ? std::make_pair(def, distance + 1)
             : std::make_pair<Operation *, unsigned>(nullptr, 0);
}

} // namespace

//===----------------------------------------------------------------------===//
// Step A: allocation & binding.
//===----------------------------------------------------------------------===//

void DatapathBuilder::collectConstants() {
  func.walk([&](arith::ConstantOp cst) {
    ConstCell c;
    c.id = dp.consts.size();
    c.value = cst.getValue();
    c.type = cst.getType();
    producerOf[cst] = Source{Source::Kind::Const, c.id, 0};
    dp.consts.push_back(c);
  });
}

MemId DatapathBuilder::getOrCreateMem(Value memref) {
  if (auto it = memOf.find(memref); it != memOf.end())
    return it->second;
  MemId id = dp.mems.size();
  MemUnit m;
  m.id = id;
  m.memref = memref;
  m.external = isa<BlockArgument>(memref);
  auto mt = cast<MemRefType>(memref.getType());
  m.width = mt.getElementTypeBitWidth();
  // Banking / ports from the same storage model the scheduler binds against
  // (allo.part / allo.bind.storage); depthWords is per-bank so that
  // numBanks * depthWords covers the array.
  allo::MemoryChar mc = allo::characterize(memref);
  m.numBanks = std::max(1u, mc.numBanks);
  // dcp-resolve-banking splits every *statically* banked internal array into
  // plain per-bank memrefs (numBanks == 1) before emit; a memref still banked
  // here has a data-dependent bank (internal -> crossbar, 2c) or is a
  // partitioned argument (external -> per-bank boundary interfaces, 2b).
  m.portsPerBank = mc.portsPerBank;
  m.impl = mc.impl;
  unsigned total = mt.hasStaticShape() ? mt.getNumElements() : 0;
  m.depthWords = total ? (total + m.numBanks - 1) / m.numBanks : 0;
  dp.mems.push_back(std::move(m));
  memOf[memref] = id;
  return id;
}

StreamId DatapathBuilder::getOrCreateStream(Value stream, bool isInput) {
  if (auto it = streamOf.find(stream); it != streamOf.end())
    return it->second;
  StreamId id = dp.streams.size();
  StreamChannel ch;
  ch.id = id;
  ch.stream = stream;
  auto st = cast<StreamType>(stream.getType());
  ch.payload = st.getBaseType();
  ch.depth = static_cast<unsigned>(st.getDepth());
  ch.isInput = isInput;
  dp.streams.push_back(std::move(ch));
  streamOf[stream] = id;
  return id;
}

RegionBlock DatapathBuilder::addRegion(Operation *regionOp, RegionId ridx) {
  regionIdxOf[regionOp] = ridx;

  RegionBlock rb;
  rb.id = ridx;
  // A container region nests another dcp region in its body (a loop wrapping an
  // inner loop). The nearest enclosing region op is the parent (already
  // processed -- pre-order walk), so it runs its children via hierarchical
  // control.
  Operation *p = regionOp->getParentOp();
  while (
      p &&
      !isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp, dcp::DCPathSelectOp>(
          p))
    p = p->getParentOp();
  if (p) {
    unsigned pidx = regionIdxOf.lookup(p);
    rb.parent = pidx;
    dp.regions[pidx].container = true;
    dp.regions[pidx].children.push_back(ridx);
  }

  if (isa<dcp::DCPathSelectOp>(regionOp)) {
    // A guard (dcp.select): a predicated container. It has no counter / trip of
    // its own -- it runs its children once iff the predicate holds -- so it
    // stays Acyclic; `container` is set when its children link below.
    rb.guard = true;
  } else if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
    rb.kind = RegionBlock::Kind::Cyclic;
    rb.conditional = pipe.isWhileLoop(); // dcp.condition terminator: flushing
    // The counter block arg keeps the source IV's NameLoc (preserved by the
    // reifier); carry its name so the emitter labels the iteration counter (i).
    if (auto n = nameFromLoc(pipe.getBody().front().getArgument(0).getLoc()))
      rb.counterName = sanitizeCppIdentifier(*n);
    // `ii` is absent for a data-dependent sequential wrapper; leave rb.ii unset
    // (downstream reg-depth uses `.value_or(1)`, EmitHW gates on II==1).
    if (std::optional<int64_t> ii = pipe.getIi())
      rb.ii = static_cast<unsigned>(*ii);
    if (IntegerAttr len = pipe.getLengthAttr())
      rb.length = len.getInt();
    if (IntegerAttr t = pipe.getTripAttr())
      rb.tripCount = t.getInt();
    rb.lb = pipe.getLb().value_or(0);
    rb.step = pipe.getStep().value_or(1);
  } else {
    rb.kind = RegionBlock::Kind::Acyclic;
    auto seq = cast<dcp::DCPathSequentialOp>(regionOp);
    if (IntegerAttr len = seq.getLengthAttr())
      rb.length = len.getInt();
  }
  return rb;
}

void DatapathBuilder::bindResource(Operation *op, RegionBlock &rb) {
  // A nested region op (a loop wrapper, or a dcp.select guard) is a child
  // region, walked in its own iteration; it binds no resource here.
  if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp, dcp::DCPathSelectOp>(
          op))
    return;

  // A stream access binds to a StreamChannel (a handshaked FIFO); a get
  // produces a token, a put consumes one (its data driver is resolved in
  // deriveInterconnect, like a store's).
  if (auto get = dyn_cast<StreamGetOp>(op)) {
    StreamId sid = getOrCreateStream(get.getStream(), /*isInput=*/true);
    unsigned aidx = dp.streams[sid].accesses.size();
    StreamChannel::Access acc;
    acc.op = op;
    acc.region = rb.id;
    acc.stage = static_cast<unsigned>(dcpStart(op));
    dp.streams[sid].accesses.push_back(acc);
    producerOf[op] = Source{Source::Kind::Stream, sid, aidx};
    return;
  }
  if (auto put = dyn_cast<StreamPutOp>(op)) {
    StreamId sid = getOrCreateStream(put.getStream(), /*isInput=*/false);
    StreamChannel::Access acc;
    acc.op = op;
    acc.isPut = true;
    acc.region = rb.id;
    acc.stage = static_cast<unsigned>(dcpStart(op));
    dp.streams[sid].accesses.push_back(acc);
    return;
  }

  // A memory access binds to a MemUnit port; a read produces a value.
  if (Value mr = dcpMemref(op)) {
    bool isWrite = isa<dcp::DCPathStoreOp>(op);
    MemId mid = getOrCreateMem(mr);
    unsigned aidx = dp.mems[mid].accesses.size();
    MemUnit::Access acc;
    acc.op = op;
    acc.isWrite = isWrite;
    acc.region = rb.id;
    dp.mems[mid].accesses.push_back(std::move(acc));
    if (!isWrite)
      producerOf[op] = Source{Source::Kind::Mem, mid, aidx};
    return;
  }

  // Literals are pre-registered as ConstCells (see collectConstants).
  if (isa<arith::ConstantOp>(op))
    return;

  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op)) {
    dcp::DCPathOperatorOp opr = dcpOperator(op);
    FuncUnit u;
    u.id = dp.units.size();
    u.opType = opr ? opr.getKind().str() : op->getName().stripDialect().str();
    if (opr && opr.getImpl())
      u.impl = opr.getImpl()->str();
    u.latency = opr ? static_cast<unsigned>(opr.getLatency()) : 0;
    u.pipelined = opr ? opr.getPipelined() : true;
    u.resultType = comp.getResult().getType();
    int64_t t = dcpStart(op);
    unsigned ii = rb.ii.value_or(1);
    unsigned residue = rb.kind == RegionBlock::Kind::Cyclic
                           ? static_cast<unsigned>(t % ii)
                           : static_cast<unsigned>(t);
    u.boundOps.push_back({op, residue});
    producerOf[op] = Source{Source::Kind::Unit, u.id, 0};
    dp.opToUnit[op] = u.id;
    rb.units.push_back(u.id);
    dp.units.push_back(std::move(u));
    return;
  }

  op->emitRemark("allo-datapath(dcp): unmodelled op in region");
}

void DatapathBuilder::recordRegionResult(const RegionBlock &rb,
                                         Operation *regionOp) {
  // A while (conditional) region's results are its frozen survivor registers,
  // recorded in `carryInfo` (recordCarryInfo) instead -- its terminator is a
  // dcp.condition whose leading operand is the condition, not a result.
  if (rb.conditional)
    return;
  Operation *term = regionBody(regionOp)->getTerminator();
  if (term->getNumOperands() == 0)
    return; // a result-less region yields nothing to a sibling
  // Each `uncondition` operand is a distinct survivor (indexed by result
  // number). An untracked result (a bare iter-arg, or a producer not modelled)
  // leaves a None placeholder so the vector stays aligned with the result
  // numbering a consumer resolves against.
  SmallVector<Source> results;
  for (Value res : term->getOperands()) {
    Operation *def = res.getDefiningOp();
    auto it = def ? producerOf.find(def) : producerOf.end();
    results.push_back(it != producerOf.end() ? it->second : Source{});
  }
  dp.regionResult[rb.id] = std::move(results);
}

void DatapathBuilder::recordCarryInfo(ArrayRef<Operation *> regionOps) {
  for (Operation *op : regionOps) {
    auto pipe = dyn_cast<dcp::DCPathPipelineOp>(op);
    if (!pipe || pipe.getInits().empty())
      continue;
    RegionBlock &rb = dp.regions[regionIdxOf.lookup(op)];
    // A leaf counted reduction carries its iter-arg in a fused accumulator (the
    // captureCountedResults path); only a container (children run per outer
    // iteration) or a while needs its iter-args latched into survivor
    // registers.
    if (!rb.conditional && !rb.container)
      continue;
    Datapath::CarryInfo wi;
    // A while's continue condition is a scheduled compute producer (a
    // cmpi/cmpf) for a leaf combinational while; a conditional container's body
    // is unscheduled, so its condition stays a raw arith tree (`producerOf` has
    // no entry) -- keep the Source None and record the root op for the emitter
    // to evaluate. A counted container has no condition.
    if (rb.conditional) {
      wi.condValue = pipe.getConditionValue();
      Operation *cdef = wi.condValue.getDefiningOp();
      wi.condition = cdef ? producerOf.lookup(cdef) : Source{};
    }
    // Per loop-carried value: its init (loaded at start) and its next-value
    // producer (advanced into the survivor when an outer iteration drains). A
    // next may be a nested region's result (a container's iter-arg is fed by a
    // child survivor) or an in-region compute producer. An untracked next (a
    // bare iter_arg / unmodelled producer) leaves a None placeholder, so no
    // survivor is built (asserts only if a sibling reads it).
    for (Value init : pipe.getInits())
      wi.inits.push_back(initSource(init));
    for (Value next : pipe.getCarriedValues())
      wi.nexts.push_back(boundSource(next)); // region result / compute / IO
    dp.carryInfo[regionIdxOf.lookup(op)] = std::move(wi);
  }
}

void DatapathBuilder::recordGuards(ArrayRef<Operation *> regionOps) {
  for (Operation *op : regionOps) {
    auto sel = dyn_cast<dcp::DCPathSelectOp>(op);
    if (!sel)
      continue;
    // A result-mux guard (both branches yield a value, muxed by the predicate)
    // needs the else branch + result-mux, not yet lowered -- a pure guard (its
    // stores live inside the then branch) has neither.
    assert(sel.getResults().empty() && sel.getElseRegion().empty() &&
           "result-mux dcp.select (else branch) not yet lowered");
    // The predicate is the select's i1 condition operand. If it is a scheduled
    // value (a preceding condition region's survivor) `boundSource` resolves
    // it; otherwise it is a raw arith tree over the enclosing counter,
    // evaluated at emit by evalRawArith (condition stays None). The reject gate
    // in emitModule checks the raw tree is combinational.
    Datapath::GuardInfo gi;
    gi.condValue = sel.getCondition();
    gi.condition = boundSource(gi.condValue);
    dp.guardCond[regionIdxOf.lookup(op)] = gi;
  }
}

Source DatapathBuilder::boundSource(Value v) {
  if (Operation *def = v.getDefiningOp()) {
    // A prologue region result: the runtime bound is one of its survivors
    // (result number selects which), the same channel a data survivor crosses.
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
    if (auto it = producerOf.find(def); it != producerOf.end())
      return it->second; // a hoisted producer (e.g. a constant bound)
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second; // a scalar-argument bound
  return {};
}

void DatapathBuilder::recordRegionBounds(ArrayRef<Operation *> regionOps) {
  // A runtime induction bound (ub / lb / step) resolves to the same F->G
  // channel a data survivor crosses (a prologue survivor or a scalar IO).
  auto record = [&](Value b, Source &into) {
    if (!b)
      return;
    into = boundSource(b);
    assert(into && "runtime induction bound with no resolvable Source");
  };
  for (Operation *op : regionOps)
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(op)) {
      RegionBlock &rb = dp.regions[regionIdxOf.lookup(op)];
      record(pipe.getDynamicBound(), rb.ubSource);
      record(pipe.getLbBound(), rb.lbSource);
      record(pipe.getStepBound(), rb.stepSource);
    }
}

void DatapathBuilder::bindIOArgs() {
  for (BlockArgument arg : func.getArguments()) {
    if (isa<MemRefType>(arg.getType()))
      continue;
    // A stream arg is a FIFO channel (a StreamChannel, created lazily by
    // bindResource on its first get/put), not a scalar port.
    if (isa<StreamType>(arg.getType()))
      continue;
    IOPort io;
    io.id = dp.ios.size();
    io.value = arg;
    io.type = arg.getType();
    io.isInput = true;
    ioOf[arg] = Source{Source::Kind::IO, io.id, 0};
    dp.ios.push_back(io);
  }
}

void DatapathBuilder::recordResults() {
  auto ret = cast<func::ReturnOp>(func.front().getTerminator());
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    assert(!isa<MemRefType>(v.getType()) &&
           "a memref result should be an out-param by emit "
           "(buffer-results-to-out-params)");
    Result r;
    r.source = boundSource(v); // survivor / passthrough IO / constant
    assert(r.source.kind != Source::Kind::None &&
           "function result with no resolvable Source");
    r.type = v.getType();
    r.name =
        ret.getNumOperands() == 1 ? "result" : ("result" + std::to_string(i));
    dp.results.push_back(std::move(r));
  }
}

//===----------------------------------------------------------------------===//
// Step B: interconnect derivation.
//===----------------------------------------------------------------------===//

Resolved DatapathBuilder::resolveOperand(Value v, Operation *consumer,
                                         unsigned ii) {
  int64_t tY = dcpStart(consumer);
  Operation *regionOp = consumer->getParentOp();

  auto edge = [&](Source base, Value key, int64_t tX, unsigned lat,
                  unsigned distance) -> Resolved {
    int64_t depth = static_cast<int64_t>(distance) * ii + (tY - tX) -
                    static_cast<int64_t>(lat);
    assert(depth >= 0 && "infeasible negative register depth");
    return {base, key, static_cast<unsigned>(depth), true};
  };

  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (auto it = ioOf.find(v); it != ioOf.end())
      return {it->second, Value(), 0, true};
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp())) {
      // The iteration counter produces the index at cycle 0. The counter
      // belongs to the region that *owns* this block argument, which for an
      // enclosing loop's index (consumed inside a nested region) is an outer
      // region -- not the consumer's. Attribute it to the owning region so the
      // emitter reads that region's counter.
      if (barg.getArgNumber() == 0)
        return edge(Source{Source::Kind::Counter, regionIdxOf.lookup(pipe), 0},
                    v, /*tX=*/0, /*lat=*/0, /*distance=*/0);
      // An iter_arg of an *enclosing* container (this consumer is nested inside
      // a sequential-wrapper while): the container's frozen survivor register,
      // read across the region boundary (depth 0, no chain). Symmetric to the
      // outer counter above -- attributed to the owning region. The consumer's
      // OWN iter_arg (pipe == its region) is the loop recurrence handled below.
      if (pipe != regionOp)
        return {Source{Source::Kind::Survivor, regionIdxOf.lookup(pipe),
                       barg.getArgNumber() - 1},
                Value(), 0, true};
      // An iter_arg carries a value from a previous iteration.
      unsigned iterArg = barg.getArgNumber() - 1;
      auto [def, distance] = traceIterArgSource(pipe, iterArg);
      if (!def || def->getParentOp() != regionOp)
        return {};
      auto it = producerOf.find(def);
      if (it == producerOf.end())
        return {};
      // This operand reads the loop-carried iter_arg: the emitter re-injects
      // its init (reduction identity) on THIS consumer input at the first
      // iteration, so a retriggered reduction restarts from the identity. Ride
      // it on the Resolved so deriveInterconnect tags the consuming unit's
      // input port -- the register carrying the recurrence may sit elsewhere in
      // the cycle (the widened idiom reads acc through a bare wire, not the
      // register).
      Resolved r = edge(it->second, def->getResult(0), dcpStart(def),
                        dcpLatency(def), distance);
      r.init = initSource(pipe.getInits()[iterArg]);
      return r;
    }
    return {};
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  // A value defined by a nested region op is one of that region's results -- a
  // cross-region survivor, held until the producing region completes (the
  // emitter latches it; see dp.regionResult). The result number selects which
  // survivor. No register chain: depth 0.
  if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(def))
    return {Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                   cast<OpResult>(v).getResultNumber()},
            Value(), 0, true};
  auto it = producerOf.find(def);
  if (it == producerOf.end())
    return {};
  if (it->second.kind == Source::Kind::Const)
    return {it->second, Value(), 0, true};
  if (def->getParentOp() != regionOp)
    return {}; // cross-region hand-off deferred
  return edge(it->second, v, dcpStart(def), dcpLatency(def), /*distance=*/0);
}

Source DatapathBuilder::initSource(Value v) {
  // A nested region's iter-arg init that reads an enclosing container's
  // iter-arg (a nested while's `%argN = %outerIterArg`): the container's frozen
  // survivor register. Safe to inject at the nested region's start -- the outer
  // register is stable for the whole nested run (unlike a sibling's data
  // survivor, which the region-result branch below re-injects only when it is a
  // constant).
  if (auto barg = dyn_cast<BlockArgument>(v))
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
        pipe && barg.getArgNumber() >= 1)
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(pipe),
                    barg.getArgNumber() - 1};
  if (Operation *def = v.getDefiningOp()) {
    if (auto it = producerOf.find(def); it != producerOf.end())
      return it->second; // typically a hoisted Const (the reduction identity)
    // An init fused into a prologue region (read as one of its `uncondition`
    // results): look through to that result's own Source, so a reduction
    // identity bundled into a multi-result survivor region still re-injects
    // (e.g. `acc = 0` alongside a loop-invariant load). Only a constant
    // identity is safe to re-inject -- it is available everywhere; a data
    // survivor lives in another region's port, so leave it to the reset value.
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(def))
      if (auto it = dp.regionResult.find(regionIdxOf.lookup(def));
          it != dp.regionResult.end()) {
        unsigned k = cast<OpResult>(v).getResultNumber();
        if (k < it->second.size() && it->second[k].kind == Source::Kind::Const)
          return it->second[k];
      }
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second;
  return {}; // None: an unmodelled init -- no re-injection (reset value stands)
}

RegId DatapathBuilder::insertRegister(Value key, ArrayRef<unsigned> depths,
                                      Source input, RegionId region) {
  unsigned maxDepth = *llvm::max_element(depths);
  SmallVector<unsigned> taps(depths.begin(), depths.end());
  llvm::sort(taps);
  taps.erase(std::unique(taps.begin(), taps.end()), taps.end());

  Register reg;
  reg.id = dp.regs.size();
  reg.value = key;
  reg.type = key.getType();
  reg.depth = maxDepth;
  reg.input = input;
  reg.taps.assign(taps.begin(), taps.end());
  reg.needsEnable = true;
  dp.valueToReg[key] = reg.id;
  dp.regions[region].regs.push_back(reg.id);
  dp.regs.push_back(std::move(reg));
  return reg.id;
}

void DatapathBuilder::deriveInterconnect() {
  for (FuncUnit &u : dp.units) {
    if (u.boundOps.empty())
      continue; // merged-away (dead) unit: dropped from its region
    unsigned n = u.boundOps.front().first->getNumOperands();
    u.inputs.assign(n, Source{});
    u.inputInits.assign(n,
                        Source{}); // parallel; set for recurrence inputs below
  }
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses) {
      SmallVector<Value> operands;
      dcpAddressing(acc.op, acc.addrMap, operands);
      acc.addr.assign(operands.size(), Source{});
    }

  llvm::MapVector<Value, SmallVector<unsigned>> depthsByKey;
  llvm::DenseMap<Value, Source> baseByKey;
  llvm::DenseMap<Value, unsigned> regionOfKey; // key -> region vector index
  struct Pending {
    Source *slot;
    Value key;
    unsigned depth;
  };
  SmallVector<Pending> pending;

  auto record = [&](Resolved r, Source &slot, unsigned regionIdx) {
    if (!r.ok)
      return;
    if (r.depth == 0) {
      slot = r.base;
      return;
    }
    depthsByKey[r.key].push_back(r.depth);
    baseByKey[r.key] = r.base;
    regionOfKey[r.key] = regionIdx;
    pending.push_back({&slot, r.key, r.depth});
  };

  // A shared unit's per-port drivers: one Source per bound op, resolved through
  // the same reg-depth path, then muxed once registers exist. A deque so
  // `record`'s pending slot pointers into `sources` survive later pushes.
  struct MuxBuild {
    UnitId unit;
    unsigned port;
    RegionId region;
    llvm::SmallVector<Operation *, 2> ops;
    llvm::SmallVector<Source, 2> sources; // parallel to ops; filled by record
  };
  std::deque<MuxBuild> muxBuilds;

  for (FuncUnit &u : dp.units) {
    if (u.boundOps.empty())
      continue;
    Operation *op0 = u.boundOps.front().first;
    unsigned ridx = regionIdxOf.lookup(op0->getParentOp());
    unsigned ii = dp.regions[ridx].ii.value_or(1);
    unsigned nPorts = op0->getNumOperands();
    if (u.boundOps.size() == 1) {
      for (unsigned k = 0; k < nPorts; ++k) {
        Resolved r = resolveOperand(op0->getOperand(k), op0, ii);
        record(r, u.inputs[k], ridx);
        u.inputInits[k] = r.init; // None unless k reads a loop-carried iter_arg
      }
      continue;
    }
    // Shared unit: resolve every bound op's port k independently (each may need
    // its own register depth), then a mux picks per op's issue cycle.
    for (unsigned k = 0; k < nPorts; ++k) {
      muxBuilds.push_back({u.id, k, ridx, {}, {}});
      MuxBuild &mb = muxBuilds.back();
      mb.sources.resize(u.boundOps.size());
      for (unsigned j = 0; j < u.boundOps.size(); ++j) {
        Operation *opj = u.boundOps[j].first;
        Resolved r = resolveOperand(opj->getOperand(k), opj, ii);
        assert(r.init.kind == Source::Kind::None &&
               "sharing a recurrence (reduction) unit is not modelled yet");
        mb.ops.push_back(opj);
        record(r, mb.sources[j], ridx);
      }
    }
  }

  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      SmallVector<Value> operands;
      AffineMap ignored;
      dcpAddressing(acc.op, ignored, operands);
      for (unsigned k = 0, e = operands.size(); k < e; ++k)
        record(resolveOperand(operands[k], acc.op, ii), acc.addr[k], ridx);
      if (acc.isWrite)
        record(resolveOperand(cast<dcp::DCPathStoreOp>(acc.op).getValue(),
                              acc.op, ii),
               acc.data, ridx);
    }

  // A stream put's data driver, resolved through the same reg-depth path as a
  // store's (the token value is presented at the put's stage).
  for (StreamChannel &s : dp.streams)
    for (StreamChannel::Access &acc : s.accesses)
      if (acc.isPut) {
        unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
        unsigned ii = dp.regions[ridx].ii.value_or(1);
        record(resolveOperand(cast<StreamPutOp>(acc.op).getValue(), acc.op, ii),
               acc.data, ridx);
      }

  for (auto &kv : depthsByKey)
    insertRegister(kv.first, kv.second, baseByKey[kv.first],
                   regionOfKey[kv.first]);

  for (const Pending &p : pending)
    *p.slot = Source{Source::Kind::Reg, dp.valueToReg[p.key], p.depth};

  // Materialize sharing muxes: sources are final now (registers built, pending
  // resolved). A port whose bound ops all read one driver needs no mux.
  auto sameSource = [](const Source &a, const Source &b) {
    return a.kind == b.kind && a.id == b.id && a.outPort == b.outPort;
  };
  for (MuxBuild &mb : muxBuilds) {
    Source &slot = dp.units[mb.unit].inputs[mb.port];
    if (llvm::all_of(mb.sources, [&](const Source &s) {
          return sameSource(s, mb.sources[0]);
        })) {
      slot = mb.sources[0];
      continue;
    }
    Mux mx;
    mx.id = dp.muxes.size();
    mx.type = mb.ops.front()->getOperand(mb.port).getType();
    mx.region = mb.region;
    mx.sources.assign(mb.sources.begin(), mb.sources.end());
    mx.selectOps.assign(mb.ops.begin(), mb.ops.end());
    dp.regions[mb.region].muxes.push_back(mx.id);
    slot = Source{Source::Kind::Mux, mx.id, 0};
    dp.muxes.push_back(std::move(mx));
  }
}

void DatapathBuilder::applyBinding(ArrayRef<SmallVector<UnitId, 2>> groups) {
  for (const SmallVector<UnitId, 2> &group : groups) {
    UnitId into = group.front();
    FuncUnit &su = dp.units[into];
    for (UnitId uid : ArrayRef<UnitId>(group).drop_front()) {
      for (const std::pair<Operation *, unsigned> &bo :
           dp.units[uid].boundOps) {
        su.boundOps.push_back(bo);
        dp.opToUnit[bo.first] = into;
        producerOf[bo.first] = Source{Source::Kind::Unit, into, 0};
      }
      dp.units[uid].boundOps.clear(); // dead: dropped from its region below
    }
  }
  // Drop the merged-away (empty) units from each region's membership; derive
  // and emit iterate region.units, so a dead unit is simply never visited.
  for (RegionBlock &rb : dp.regions) {
    SmallVector<UnitId> kept;
    for (UnitId uid : rb.units)
      if (!dp.units[uid].boundOps.empty())
        kept.push_back(uid);
    rb.units.assign(kept.begin(), kept.end());
  }
}

//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

void DatapathBuilder::build() {
  dp.func = func;

  collectConstants();

  // dcp region ops in program order. Pre-order so an enclosing container is
  // processed before its nested children (the parent/child linkage and the
  // outer-index counter attribution rely on parent-before-child).
  SmallVector<Operation *> regionOps;
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
            dcp::DCPathSelectOp>(op))
      regionOps.push_back(op);
  });

  for (unsigned ridx = 0, e = regionOps.size(); ridx < e; ++ridx) {
    Operation *regionOp = regionOps[ridx];
    RegionBlock rb = addRegion(regionOp, ridx);
    for (Operation &opRef : regionBody(regionOp)->without_terminator())
      bindResource(&opRef, rb);
    recordRegionResult(rb, regionOp);
    dp.regions.push_back(std::move(rb));
  }

  bindIOArgs();
  recordRegionBounds(
      regionOps); // dynamic-trip bounds (needs ioOf + regionIdxOf)
  recordCarryInfo(
      regionOps);          // container / while iter-arg recurrence (needs ioOf)
  recordGuards(regionOps); // guard (dcp.select) predicate Sources
  recordResults(); // scalar func-result output ports (needs ioOf + regionIdxOf)
  applyBinding(policy.plan(dp)); // trivial => no groups, no muxes
  deriveInterconnect();
  verifyBinding(dp); // MRT legality: no unit shared by conflicting ops
}

} // namespace mlir::allo::uarch
