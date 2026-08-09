/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h" // iface::ModuleInterface (CallUnit ports)
#include "allo/Microarch/Reservation.h" // verifyBinding (MRT legality)

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // splitAddress (strength reduction)
#include "allo/Scheduling/LatencyModel.h" // composeSpan (the one composer)
#include "allo/Scheduling/MemoryModel.h"  // characterize (storage shape)
#include "allo/Scheduling/OperatorLibrary.h" // operatorIdentity, characterize
#include "allo/Support/AliasAnalysis.h"      // resolveRoot (storage identity)
#include "allo/Support/Logging.h"            // unmodelled-op diagnostic
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GetGlobalOp/GlobalOp (ROM)
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <deque>
#include <numeric>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Pure DCP structural readers.
//===----------------------------------------------------------------------===//

static Value dcpMemref(Operation *op) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return l.getMemref();
  if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
    return s.getMemref();
  return nullptr;
}

// The addressing of a dcp memory access: its affine map plus index operands.
static void dcpAddressing(Operation *op, AffineMap &map,
                          SmallVector<Value> &operands) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    map = l.getMap();
    operands.assign(l.getIndices().begin(), l.getIndices().end());
  } else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op)) {
    map = s.getMap();
    operands.assign(s.getIndices().begin(), s.getIndices().end());
  }
}

// The body block of a dcp region op. A guard (dcp.select) reports its `then`
// branch; its else branch is walked separately by every caller that needs it.
static Block *regionBody(Operation *regionOp) {
  if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp))
    return &pipe.getBody().front();
  if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
    return &sel.getThenRegion().front();
  return &cast<dcp::DCPathSequentialOp>(regionOp).getBody().front();
}

// Trace a pipeline iter-arg (0-based) back to the value assigned to it each
// iteration, appending to \p chain each iter-arg the walk shifts through,
// itself first. The walk stops at the first value that is not such a shift,
// which is where the recurrence reads its datum from; `chain.size()` is then
// the recurrence distance the scheduler solved against. Returns null when the
// shifts form a cycle, so there is no such datum.
static Value traceIterArgSource(dcp::DCPathPipelineOp pipe, unsigned iterArg,
                                SmallVectorImpl<unsigned> &chain) {
  Block &body = pipe.getBody().front();
  auto carried = pipe.getCarriedValues();
  Value v = carried[iterArg];
  chain.push_back(iterArg);
  llvm::SmallDenseSet<unsigned> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    // Arg 0 is the counter and an arg of another block is an enclosing value;
    // neither shifts, so both end the walk.
    if (arg.getOwner() != &body || arg.getArgNumber() == 0)
      break;
    if (!seen.insert(arg.getArgNumber()).second)
      return {};
    chain.push_back(arg.getArgNumber() - 1);
    v = carried[chain.back()]; // block arg (k+1) -> carried[k]
  }
  return v;
}

// A source valid for the whole of a region's run, so a consumer ties straight
// in and needs no register: a literal, a boundary port, or a survivor an
// earlier region latched before this one started.
static bool isHeld(Source s) {
  return s.kind == Source::Kind::Const || s.kind == Source::Kind::IO ||
         s.kind == Source::Kind::Survivor;
}

// Is \p v a transient FIFO-din value, one that changes while the region is
// back-pressured (`valid & ~ready`), so it must be captured into a
// chain-enable-frozen register before it drives a FIFO write? True iff it is,
// or is a combinational function of, one of the two sources that move under
// back-pressure: a memory load (re-addressed as the counter advances/resets) or
// the loop counter (pipeline block arg 0, reset to `lb` in the drain).
// Everything else is frozen with the datapath while stalled.
static bool isTransientDin(Value v) {
  if (auto barg = dyn_cast<BlockArgument>(v))
    return isa_and_nonnull<dcp::DCPathPipelineOp>(
               barg.getOwner()->getParentOp()) &&
           barg.getArgNumber() == 0;
  auto *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<dcp::DCPathLoadOp>(def))
    return true;
  // Stable producers: a FIFO head, a region survivor, a call result, a literal.
  if (isa<StreamGetOp, dcp::DCPathRegionOpInterface, dcp::DCPathInstanceOp,
          arith::ConstantOp>(def))
    return false;
  if (dcpLatency(def) == 0)
    return llvm::any_of(def->getOperands(),
                        [](Value o) { return isTransientDin(o); });
  return false; // a registered (latency>=1) unit's output is frozen under stall
}

//===----------------------------------------------------------------------===//
// Allocation & binding.
//===----------------------------------------------------------------------===//

void DatapathBuilder::collectConstants() {
  func.walk([&](arith::ConstantOp cst) {
    ConstCell c;
    c.id = dp.consts.size();
    c.value = static_cast<Attribute>(cst.getValue());
    c.type = cst.getType();
    producerOf[cst.getResult()] = Source{Source::Kind::Const, c.id, 0};
    dp.consts.push_back(c);
  });
}

Source DatapathBuilder::constant(int64_t v, Type t) {
  ConstCell c;
  c.id = dp.consts.size();
  c.value = IntegerAttr::get(t, v);
  c.type = t;
  dp.consts.push_back(c);
  return Source{Source::Kind::Const, c.id, 0};
}

MemId DatapathBuilder::getOrCreateMem(Value memref) {
  // Key on the storage root, not the operand as written, so a buffer threaded
  // out of a region is the SAME memory to its producer and its consumer.
  memref = resolveRoot(memref);
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
  // (allo.part / allo.bind.storage): ONE characterization, so the ports billed
  // and the ports built cannot disagree.
  MemoryChar mc = allo::characterize(memref, dev.memory);
  // An initialized global the kernel stores to needs a real write port, so it
  // is a ROM only if nothing writes it: `MemoryChar::constantTable`, the same
  // predicate the scheduler's port model bills against.
  if (auto init = allo::globalInitOf(memref)) {
    m.romInit = *init;
    m.isRom = mc.constantTable;
  }
  m.layout = mc.layout;
  m.numBanks = m.layout.numBanks;
  // THE expression behind `scattered` (see its declaration for where the cells
  // live). A callee's array argument is the one place a complete partition
  // changes nothing: the storage is the parent's and the child masters an
  // addressed port on it.
  m.scattered = m.layout.registers && (!m.external || dp.atTop);
  m.storage = mc.storage;
  // Everything the device states about the resolved realization, from the same
  // rows the scheduler timed this memref's accesses against. The emitter builds
  // ports at these latencies; do not re-derive from the name.
  const StorageRealization *sr = dev.memory.row(m.storage);
  assert(sr && "`PreVerification` rejects an array whose storage realization "
               "the device does not declare");
  m.readLatency = sr->timing.latency.read;
  m.writeLatency = sr->timing.latency.write;
  m.ramStyle = sr->ramStyle;
  // Port budget from that same characterization, so the ports the scheduler
  // reserved and the ports `bindMemoryPorts` assigns are one number.
  m.ports = mc.ports;
  assert(mt.hasStaticShape() &&
         "datapath memory requires a static shape (a dynamic memref sizes to "
         "depthWords 0)");
  // Per-bank depth from the same element-space decomposition the emitter's
  // crossbar and the host-side layout use, so a bank's address space is exactly
  // the elements it holds (`ceil` per partitioned dim, not of the total).
  m.depthWords = static_cast<unsigned>(m.layout.bankWords());
  dp.mems.push_back(std::move(m));
  memOf[memref] = id;
  return id;
}

StreamId DatapathBuilder::getOrCreateStream(Value stream, bool isInput) {
  // Key on the storage root for the same reason a memref does: a channel
  // threaded out of a region names different Values at its two ends.
  stream = resolveRoot(stream);
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
  // A channel the kernel creates itself needs no port: both its ends are here.
  ch.internal = !isa<BlockArgument>(stream);
  // Initial tokens, when the declaration carries them: what breaks a feedback
  // cycle's start dependence (see `StreamChannel::init`).
  if (auto cr = stream.getDefiningOp<StreamCreateOp>())
    ch.init = cr.getInitAttr();
  dp.streams.push_back(std::move(ch));
  streamOf[stream] = id;
  return id;
}

RegionBlock DatapathBuilder::addRegion(Operation *regionOp, RegionId ridx) {
  regionIdxOf[regionOp] = ridx;

  RegionBlock rb;
  rb.id = ridx;
  rb.op = regionOp;
  // The nearest enclosing region op is the parent, already processed by this
  // pre-order walk; nesting a region makes that parent a container.
  Operation *p = regionOp->getParentOp();
  while (p && !isa<dcp::DCPathRegionOpInterface>(p))
    p = p->getParentOp();
  if (p) {
    unsigned pidx = regionIdxOf.lookup(p);
    rb.parent = pidx;
    dp.regions[pidx].container = true;
    // A guard (dcp.select) splits its children by branch: one nested in the
    // else body is an else-child, otherwise a then-child.
    bool isElse = false;
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(p)) {
      Operation *o = regionOp;
      while (o->getParentOp() != p)
        o = o->getParentOp();
      isElse = o->getParentRegion() == &sel.getElseRegion();
    }
    (isElse ? dp.regions[pidx].elseChildren : dp.regions[pidx].children)
        .push_back(ridx);
  }

  if (isa<dcp::DCPathSelectOp>(regionOp)) {
    // A predicated container: no counter or trip of its own, it runs its
    // children once iff the predicate holds, so it stays Acyclic.
    rb.guard = true;
  } else if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
    rb.kind = RegionBlock::Kind::Cyclic;
    rb.conditional = pipe.isWhileLoop(); // dcp.condition terminator: flushing
    // The counter block arg keeps the source IV's NameLoc (preserved by the
    // reifier); carry its name so the emitter labels the iteration counter (i).
    if (auto n = nameFromLoc(pipe.getBody().front().getArgument(0).getLoc()))
      rb.counterName = sanitizeCppIdentifier(*n);
    // `ii` is absent for a data-dependent sequential wrapper: such a region has
    // children, so `emitRegion` routes it to a container path that never reads
    // `ii`, and reg-depth paths default to 1.
    if (std::optional<int64_t> ii = pipe.getIi())
      rb.ii = static_cast<unsigned>(*ii);
    if (auto t = pipe.getTripAttr())
      rb.tripCount = t.getInt();
    if (auto t = pipe.getTripBoundAttr())
      rb.tripBound = t.getInt();
    assert(!(rb.tripCount && rb.tripBound) &&
           "an exact trip and a worst-case bound on the same loop");
    // The induction bounds are resolved by `recordRegionBounds`, once the
    // counter width is known and `resolveValue` sees the whole region model.
  } else {
    assert(isa<dcp::DCPathSequentialOp>(regionOp) &&
           "a RegionBlock is a dcp pipeline / sequential / select");
    rb.kind = RegionBlock::Kind::Acyclic;
  }

  // Composition class, DERIVED from the region rather than read back off the
  // attribute the reifier stamps.
  rb.determinacy = dcpRegionTiming(regionOp).determinacy;
  // The one number that IS read back, on purpose: the model's claim about this
  // region's terminal cycle, which `emitRegion` holds the datapath to.
  if (std::optional<uint64_t> d =
          cast<dcp::DCPathRegionOpInterface>(regionOp).getDrain())
    rb.modelledDrain = static_cast<int64_t>(*d);
  return rb;
}

// `dcpRegionShape` is the one answer the emitter's dispatch, the validator's
// legality rules and the latency composer all read. The BUILT model reaches it
// down a different path (linked parent/child edges and bound CallUnits), so the
// assert catches a region op and its built model describing different hardware.
void DatapathBuilder::deriveShapes() {
  for (RegionBlock &rb : dp.regions) {
    rb.shape = dcpRegionShape(rb.op);
    [[maybe_unused]] RegionBlock::Shape modelled =
        rb.guard               ? RegionBlock::Shape::Guard
        : !rb.children.empty() ? RegionBlock::Shape::Container
        : (rb.kind == RegionBlock::Kind::Cyclic && !rb.callUnits.empty())
            ? RegionBlock::Shape::CallNode
            : RegionBlock::Shape::Leaf;
    assert(rb.shape == modelled &&
           "the region op's shape disagrees with the built model's");

    assert((rb.shape != RegionBlock::Shape::Guard || !rb.children.empty()) &&
           "a guard region has no then-branch children to predicate");
    assert((rb.shape != RegionBlock::Shape::CallNode || rb.children.empty()) &&
           "a call-node region sequences an instance, not child regions");
    // The two axes must agree in the direction the composer relies on: a
    // flushing while is always DECLARED conditional. Not a biconditional, since
    // the reifier stamps a `dcp.select` `Conditional` with `conditional` false.
    assert(
        (!rb.conditional || rb.determinacy == DeterminacyEnum::Conditional) &&
        "a while region must be declared conditional");
  }
}

void DatapathBuilder::bindCall(dcp::DCPathInstanceOp inv, RegionBlock &rb) {
  assert(callees && "a dcp.instance in a leaf datapath needs callee context "
                    "(a rerouted container)");
  auto it = callees->ifaces.find(inv.getCallee());
  assert(it != callees->ifaces.end() &&
         "the callee interface must be registered (emitted bottom-up first)");
  const auto &mi = it->second;

  CallUnit cu;
  cu.id = dp.calls.size();
  cu.invoke = inv;
  cu.region = rb.id;
  cu.callee = inv.getCallee().str();
  cu.latency = inv.getLatency();
  cu.start = static_cast<unsigned>(dcpStart(inv));
  cu.async = inv->hasAttr(kAlloAsyncAttr);
  cu.determinate =
      !cu.async && inv.getDeterminacy() == DeterminacyEnum::CountedStatic;

  // Operands are in callee-argument order, so operand k is callee arg k. Each
  // memref operand contributes one MemArg per child port.
  for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
    if (isa<StreamType>(operand.getType())) {
      // A channel END: the child handshakes on three ports of its own, recorded
      // against the call/arg pair so the realization (one FIFO per consumer)
      // can wire them without scanning the calls.
      const iface::FIFO *f = mi.streamForArg(static_cast<int>(k));
      assert(f && "a stream operand with no matching callee stream port");
      StreamId sid = getOrCreateStream(operand, f->isInput);
      dp.streams[sid].callEnds.push_back(
          {cu.id, static_cast<unsigned>(cu.streamArgs.size())});
      // Buffering is a throughput hint and deeper is always KPN-safe, so a
      // channel takes the deepest request among its ends.
      dp.streams[sid].depth =
          std::max<unsigned>(dp.streams[sid].depth, f->depth);
      cu.streamArgs.push_back({static_cast<unsigned>(k), sid, f->isInput,
                               static_cast<unsigned>(f->depth), f->base,
                               f->data, f->valid, f->ready});
      continue;
    }
    if (!isa<MemRefType>(operand.getType())) {
      // A scalar operand feeds the child's scalar-input port; its driver is
      // resolved by recordCallScalars, once every region exists.
      const auto *sc = mi.scalarForArg(static_cast<int>(k));
      assert(sc && "a scalar operand with no matching callee scalar port");
      cu.scalarIns.push_back({Source{}, sc->name, sc->width});
      continue;
    }
    auto mem = getOrCreateMem(operand);
    bool isBoundary = isa<BlockArgument>(operand);
    for (const iface::Memory *m : mi.portsForArg(static_cast<int>(k))) {
      CallUnit::MemArg ma;
      ma.calleeArg = static_cast<unsigned>(k);
      ma.mem = mem;
      ma.isBoundary = isBoundary;
      ma.isWrite = m->write;
      // The bank this child port serves: a cyclically partitioned arg exposes
      // one port group per bank, carrying (bank, factor) at the boundary.
      ma.bank = static_cast<unsigned>(m->bank);
      ma.factor = static_cast<unsigned>(m->factor);
      ma.independent = m->independent;
      ma.addr = m->addr;
      ma.data = m->data;
      ma.we = m->we;
      // `ma.topBase` (the boundary port group) is assigned by
      // enumerateBoundaryPorts, once every access is bound.
      cu.memArgs.push_back(std::move(ma));
    }
  }
  // Each scalar result is a Source::Call this region yields: one producerOf
  // entry per result, so recordRegionResults latches each as its own survivor,
  // plus the child's result-output ports for emitCalls.
  for (const iface::Result &r : mi.results)
    cu.resultPorts.push_back(r.name);
  assert(inv.getNumResults() == cu.resultPorts.size() &&
         "an invoke's result count must match the callee's result ports");
  for (auto [k, res] : llvm::enumerate(inv->getResults()))
    producerOf[res] = Source{Source::Kind::Call, cu.id, unsigned(k)};

  rb.callUnits.push_back(cu.id);
  dp.calls.push_back(std::move(cu));
}

void DatapathBuilder::bindStream(Operation *op, RegionBlock &rb) {
  auto get = dyn_cast<StreamGetOp>(op);
  auto sid = getOrCreateStream(get ? get.getStream()
                                   : cast<StreamPutOp>(op).getStream(),
                               /*isInput=*/get != nullptr);
  unsigned aidx = dp.streams[sid].accesses.size();
  StreamChannel::Access acc;
  acc.op = op;
  acc.isPut = !get;
  acc.region = rb.id;
  acc.stage = static_cast<unsigned>(dcpStart(op));
  dp.streams[sid].accesses.push_back(acc);
  rb.streamAccesses.push_back({sid, aidx});
  // A get produces a token; a put consumes one, and its data driver is
  // resolved in deriveInterconnect like a store's.
  if (get)
    producerOf[get.getResult()] = Source{Source::Kind::Stream, sid, aidx};
}

void DatapathBuilder::bindMemory(Operation *op, Value memref, RegionBlock &rb) {
  bool isWrite = isa<dcp::DCPathStoreOp>(op);
  auto mid = getOrCreateMem(memref);
  // Fires only if `getOrCreateMem`'s ROM scan and this binding disagree.
  assert(!(isWrite && dp.mems[mid].isRom) &&
         "store bound to a memory classified read-only");
  // A mismatch would time a port against a cycle the consumer's register
  // depth was not solved for; both read the same device table.
  assert(dcpLatency(op) ==
             (isWrite ? dp.mems[mid].writeLatency : dp.mems[mid].readLatency) &&
         "scheduled access latency disagrees with the device memory model");
  unsigned aidx = dp.mems[mid].accesses.size();
  MemUnit::Access acc;
  acc.op = op;
  acc.isWrite = isWrite;
  acc.region = rb.id;
  dp.mems[mid].accesses.push_back(std::move(acc));
  rb.memAccesses.push_back({mid, aidx});
  if (!isWrite)
    producerOf[op->getResult(0)] = Source{Source::Kind::Mem, mid, aidx};
}

void DatapathBuilder::bindCompute(dcp::DCPathComputeOp comp, RegionBlock &rb) {
  FuncUnit u;
  u.id = dp.units.size();
  u.identity = operatorIdentity(comp);
  if (u.identity.comb) {
    // Combinational: emitted inline as a `comb` primitive (latency 0).
    u.latency = 0;
    u.pipelined = true;
  } else {
    // IP: the `dcp.operator` the identity names is the one copy of its timing
    // and stall contract.
    auto opr = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(
        comp, comp.getOpTypeAttr());
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    u.latency = static_cast<unsigned>(opr.getLatency());
    u.pipelined = opr.getPipelined();
    u.stall = opr.getStall();
  }
  // The unit's reservation slot: its issue cycle, taken modulo II in a cyclic
  // region since successive iterations overlap there.
  int64_t t = dcpStart(comp);
  unsigned ii = rb.ii.value_or(1);
  unsigned residue = rb.kind == RegionBlock::Kind::Cyclic
                         ? static_cast<unsigned>(t % ii)
                         : static_cast<unsigned>(t);
  u.boundOps.push_back({comp, residue});
  producerOf[comp.getResult()] = Source{Source::Kind::Unit, u.id, 0};
  dp.opToUnit[comp] = u.id;
  rb.units.push_back(u.id);
  dp.units.push_back(std::move(u));
}

void DatapathBuilder::bindResource(Operation *op, RegionBlock &rb) {
  if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op))
    return bindCall(inv, rb); // a sub-kernel call -> a CallUnit
  if (isa<StreamGetOp, StreamPutOp>(op))
    return bindStream(op, rb); // a handshaked FIFO access
  if (auto mr = dcpMemref(op))
    return bindMemory(op, mr, rb); // a MemUnit port
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return bindCompute(comp, rb); // a FuncUnit

  // A nested region op is a child region, walked in its own iteration.
  if (isa<dcp::DCPathRegionOpInterface>(op))
    return;
  // Literals are pre-registered as ConstCells (see collectConstants).
  if (isa<arith::ConstantOp>(op))
    return;
  // A declaration binds no resource: the memref / stream it defines is
  // materialized on first access.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
          StreamCreateOp>(op))
    return;

  unsupported(Stage::Emit, Code::OperationNotModelled, op)
      << "Operation '" << op->getName()
      << "' is not modelled by the datapath, so it would be dropped from the "
         "emitted hardware";
  dp.infeasible = true;
}

void DatapathBuilder::recordRegionResults(ArrayRef<Operation *> regionOps) {
  for (Operation *regionOp : regionOps) {
    RegionBlock &rb = dp.regions[regionIdxOf.lookup(regionOp)];

    // A guard yields from its two ARMS, not from one body terminator, and its
    // predicate is an explicit operand rather than a body value.
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp)) {
      rb.condition = resolveValue(sel.getCondition());
      auto arm = [&](Region &br) {
        SmallVector<Source> rs;
        if (!br.empty())
          for (Value v : br.front().getTerminator()->getOperands())
            rs.push_back(resolveValue(v));
        return rs;
      };
      SmallVector<Source> thenR = arm(sel.getThenRegion());
      SmallVector<Source> elseR = arm(sel.getElseRegion());
      assert((thenR.empty() || thenR.size() == elseR.size()) &&
             "a result-yielding dcp.select needs an else arm of equal arity");
      for (auto [k, then] : llvm::enumerate(thenR))
        rb.results.push_back({then, Source{}, elseR[k]});
      continue;
    }

    // A pipeline's results ARE its loop-carried recurrence: result k is the
    // final value of iter-arg k, and the verifier pairs each init with its
    // carried next 1:1. An unresolvable half stays None to keep the numbering.
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
      // A while's continue condition is a scheduled compute producer; a counted
      // loop has none.
      if (rb.conditional)
        rb.condition = resolveValue(pipe.getConditionValue());
      for (auto [init, next] :
           llvm::zip(pipe.getInits(), pipe.getCarriedValues()))
        rb.results.push_back(
            {resolveValue(next), resolveValue(init), Source{}});
      continue;
    }

    // A sequential region: each terminator operand lands exactly once, so there
    // is no recurrence to preload.
    for (Value res : regionBody(regionOp)->getTerminator()->getOperands())
      rb.results.push_back({resolveValue(res), Source{}, Source{}});
  }
}

void DatapathBuilder::recordCallScalars() {
  for (CallUnit &cu : dp.calls) {
    unsigned k = 0;
    for (Value operand : cast<dcp::DCPathInstanceOp>(cu.invoke).getInputs())
      if (!isa<MemRefType, StreamType>(operand.getType()))
        cu.scalarIns[k++].src = resolveValue(operand);
    assert(
        k == cu.scalarIns.size() &&
        "one scalar operand per scalar-input port (bindResource pairs them)");
  }
}

void DatapathBuilder::reclassifyRoms() {
  for (MemUnit &m : dp.mems) {
    if (!m.romInit || m.external)
      continue;
    bool written = llvm::any_of(
        m.accesses, [](const MemUnit::Access &a) { return a.isWrite; });
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        written |= ma.mem == m.id && ma.isWrite;
    m.isRom = !written;
  }
}

void DatapathBuilder::recordCallDeps() {
  // The MemIds a call touches, by role. Two calls share an array iff MemId
  // identity says so (`getOrCreateMem` keys on the storage root).
  auto memsOf = [&](const CallUnit &cu, std::optional<bool> write) {
    SmallVector<MemId, 4> ms;
    for (const CallUnit::MemArg &ma : cu.memArgs)
      if (!write || ma.isWrite == *write)
        ms.push_back(ma.mem);
    return ms;
  };
  auto shares = [](ArrayRef<MemId> a, ArrayRef<MemId> b) {
    return llvm::any_of(a, [&](MemId m) { return llvm::is_contained(b, m); });
  };
  // Two children joined by a CHANNEL, which back-pressure alone can order: they
  // are co-resident and the downstream one drains the queue the upstream one
  // fills, so waiting for the producer deadlocks on a queue shorter than run.
  auto channelled = [](const CallUnit &a, const CallUnit &b) {
    return llvm::any_of(a.streamArgs, [&](const CallUnit::StreamArg &x) {
      return llvm::any_of(b.streamArgs, [&](const CallUnit::StreamArg &y) {
        return x.chan == y.chan;
      });
    });
  };
  for (const RegionBlock &rb : dp.regions) {
    bool concurrent = rb.determinacy == DeterminacyEnum::Concurrent;
    for (auto [i, cid] : llvm::enumerate(rb.callUnits)) {
      CallUnit &cu = dp.calls[cid];
      auto add = [&](CallId p, bool viaResult) {
        for (CallUnit::Pred &e : cu.predecessors)
          if (e.call == p) {
            e.viaResult |= viaResult;
            return;
          }
        cu.predecessors.push_back({p, viaResult});
      };
      for (unsigned j = 0; j < i; ++j) {
        const CallUnit &p = dp.calls[rb.callUnits[j]];
        // A CONCURRENT container places every child at 0, so hazard DIRECTION
        // (RAW / WAW / WAR) is the whole ordering, between children the
        // channels do not order. A SCHEDULED one orders by `start`.
        bool hazard =
            concurrent
                ? !channelled(p, cu) &&
                      (shares(memsOf(p, true), memsOf(cu, std::nullopt)) ||
                       shares(memsOf(cu, true), memsOf(p, false)))
                : (p.start < cu.start || !p.latency) &&
                      shares(memsOf(p, std::nullopt), memsOf(cu, std::nullopt));
        if (hazard)
          add(p.id, /*viaResult=*/false);
      }
      // A child consuming an earlier child's scalar RESULT is ordered after it:
      // the result port only holds from the producer's `done`.
      for (const CallUnit::ScalarArg &sa : cu.scalarIns)
        if (sa.src.kind == Source::Kind::Call)
          add(sa.src.id, /*viaResult=*/true);
    }
  }
}

// Bind \p m: which port of its bank each access drives, and how many ports one
// bank is therefore built with.
//
// Two accesses share a port only where `portGraph` has no edge between them,
// which proves they never issue in the same cycle, so the port carries a select
// over them rather than an arbiter. Three shapes take a port of their own: an
// access with no bank of its own, which a crossbar routes to every bank; every
// access of an array with more of them than the relation is built for; and, on
// the write side, every write of an array whose splitting is not proven safe.
std::optional<DatapathBuilder::PortCounts>
DatapathBuilder::bindPorts(MemUnit &m, std::optional<bool> writes,
                           unsigned base) {
  llvm::SmallVector<Datapath::PortVertex> verts;
  llvm::SmallVector<uint64_t> adj = dp.portGraph(m.id, writes, verts);
  unsigned n = verts.size();
  llvm::SmallVector<unsigned> colour(n, 0);
  unsigned used = 0;
  // Whether the writes may go on separate ports, which are separate `always`
  // blocks with nothing between them to resolve a collision. Only a pair proven
  // to address different words may: two accesses of one region, which a memory
  // dependence made the scheduler separate, and two write ports of one child
  // that declared them independent. Two different children, or a child and a
  // local access, are related by nothing.
  bool split = true;

  if (adj.size() != n) {
    split = false; // too many accesses to relate: each keeps its own port
  } else {
    uint64_t all = n == 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].bank < 0) {
        adj[i] |= all & ~(uint64_t(1) << i);
        for (unsigned j = 0; j < n; ++j)
          if (j != i)
            adj[j] |= uint64_t(1) << i;
      }
    // What a pair on one port shares its address by decides how far apart they
    // may sit. Two writes need nothing, an address being a don't-care in any
    // cycle its enable is low, so stores share a port across regions. A read
    // and a write share by the write's own enable, likewise a signal in its own
    // right, so those share across regions too. Two reads share by a select
    // over their activation pulses held against a stall, and only accesses of
    // one region hold theirs against the same one, so two reads of different
    // regions and a child's read port keep ports of their own.
    //
    // A container's own reads share with nothing: they form its condition cone,
    // live on every cycle its children run rather than on a pulse of its own.
    auto containerRead = [&](unsigned i) {
      return !verts[i].write && verts[i].call < 0 &&
             dp.regions[m.accesses[verts[i].access].region].shape ==
                 RegionBlock::Shape::Container;
    };
    auto sharable = [&](unsigned i, unsigned j) {
      if (containerRead(i) || containerRead(j))
        return false;
      if (verts[i].write || verts[j].write)
        return true;
      if (verts[i].call >= 0 || verts[j].call >= 0)
        return false;
      return m.accesses[verts[i].access].region ==
             m.accesses[verts[j].access].region;
    };
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = i + 1; j < n; ++j) {
        if (!sharable(i, j)) {
          adj[i] |= uint64_t(1) << j;
          adj[j] |= uint64_t(1) << i;
        }
      }
    // Greedy in vertex order, taking the lowest port no neighbour holds.
    for (unsigned i = 0; i < n; ++i) {
      uint64_t taken = 0;
      for (unsigned j = 0; j < i; ++j)
        if ((adj[i] >> j) & 1)
          taken |= uint64_t(1) << colour[j];
      colour[i] = llvm::countr_one(taken);
      used = std::max(used, colour[i] + 1);
    }
    auto proven = [&](unsigned i, unsigned j) {
      if (verts[i].call < 0 && verts[j].call < 0)
        return m.accesses[verts[i].access].region ==
               m.accesses[verts[j].access].region;
      return verts[i].call >= 0 && verts[i].call == verts[j].call &&
             verts[i].independent;
    };
    for (unsigned i = 0; split && used > 1 && i < n; ++i)
      for (unsigned j = i + 1; j < n; ++j)
        if (verts[i].write && verts[j].write && ((adj[i] >> j) & 1) &&
            !proven(i, j)) {
          split = false;
          break;
        }
  }
  // An unsplittable set of writes stays on one `always` block, which arbitrates
  // the collision it might have. Each still keeps a port of its own so the
  // block holds one assignment per write: two writes to different words in one
  // cycle must both commit, and a select would drop one.
  //
  // That block is per direction, so a both-directions pass cannot express it
  // and declines; a binding giving every access its own port is never fewer
  // ports than the per-direction one it falls back to.
  if (!split) {
    if (!writes)
      return std::nullopt;
    for (unsigned i = 0; i < n; ++i)
      colour[i] = i;
    used = n;
  }
  // Only a colouring that included the writes has anything to say about them.
  if (!writes || *writes) {
    llvm::SmallDenseSet<unsigned> writeColours;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].write)
        writeColours.insert(colour[i]);
    m.writesIndependent = split && writeColours.size() > 1;
  }

  // The vertex order `portGraph` builds: writes before reads, and within each
  // this function's accesses before the ports its children master.
  unsigned v = 0;
  for (bool dir : {true, false}) {
    if (writes && *writes != dir)
      continue;
    for (MemUnit::Access &acc : m.accesses)
      if (acc.isWrite == dir)
        acc.port = base + colour[v++];
    for (CallUnit &cu : dp.calls)
      for (CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && ma.isWrite == dir)
          ma.port = base + colour[v++];
  }
  assert(v == n && "the port binding walks `portGraph`'s vertex order");

  // Ports one bank is built with: a bank is its own `seq.hlmem` and only the
  // accesses reaching it take its ports.
  PortCounts out;
  out.colours = used;
  for (unsigned k = 0; k < m.numBanks; ++k) {
    llvm::SmallDenseSet<unsigned> all, rd, wr;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].bank < 0 || verts[i].bank == int(k)) {
        all.insert(colour[i]);
        (verts[i].write ? wr : rd).insert(colour[i]);
      }
    out.total = std::max<unsigned>(out.total, all.size());
    out.reads = std::max<unsigned>(out.reads, rd.size());
    out.writes = std::max<unsigned>(out.writes, wr.size());
  }
  return out;
}

void DatapathBuilder::bindMemoryPorts() {
  for (MemUnit &m : dp.mems) {
    // Neither is addressed, so neither has a port to contend for: a scattered
    // array is one cell per element and a constant table is combinational.
    if (m.scattered || m.isRom)
      continue;
    // A skew answers this at the same granularity: a lane's accesses hold
    // distinct slots, so they reach distinct banks and share one port on each.
    if (m.skewed) {
      llvm::SmallDenseSet<unsigned> lanes[2];
      for (MemUnit::Access &acc : m.accesses) {
        acc.port = acc.lane;
        lanes[acc.isWrite].insert(acc.lane);
      }
      m.readPortsBuilt = lanes[0].size();
      m.writePortsBuilt = lanes[1].size();
      m.portsBuilt = m.readPortsBuilt + m.writePortsBuilt;
      continue;
    }
    // A direction at a time, reads numbered past the writes so no port carries
    // both. On a row whose directions are separate structures, merging them
    // buys an address multiplexer and nothing else.
    auto separate = [&] {
      PortCounts w = bindPorts(m, /*writes=*/true, /*base=*/0).value();
      PortCounts r = bindPorts(m, /*writes=*/false, /*base=*/w.colours).value();
      m.writePortsBuilt = w.writes;
      m.readPortsBuilt = r.reads;
      m.portsBuilt = w.writes + r.reads;
    };
    separate();
    // Where the row's ports are a pool, each serving either direction, a read
    // may ride a write's port and one address bus carries both. Worth the
    // multiplexer only where the array does not otherwise fit, and possible
    // only where the writes were split, an unsplittable set already being one
    // `always` block.
    if (m.ports.instPool && !m.external && !m.fitsStorage() &&
        (m.writePortsBuilt <= 1 || m.writesIndependent)) {
      // The shared bus carries the write's address on the cycle it commits, so
      // a write that presents its terms early would drive the read's cycle too.
      assert(m.writeLatency == 1 &&
             "a pooled row's write port realizes in one cycle");
      std::optional<PortCounts> u =
          bindPorts(m, /*writes=*/std::nullopt, /*base=*/0);
      if (u && u->total < m.portsBuilt) {
        m.readPortsBuilt = u->reads;
        m.writePortsBuilt = u->writes;
        m.portsBuilt = u->total;
      } else {
        separate();
      }
    }
  }
  // Instances of its row each bank is held in, decided here because the bound
  // ports are what it follows from. A skew binds its ports by lane and leaves
  // the loop early, so this runs over every memory rather than inside.
  for (MemUnit &m : dp.mems) {
    // Off `instReads`, what ONE instance serves, not off the array's own
    // allowance, which is already a multiple of it. Not bounded by the copies
    // budget either: the budget is what a CYCLE may issue, and a binding that
    // needs more address buses than that still builds one copy per bus.
    unsigned per = m.ports.instReads.value_or(0);
    // A bus carrying a write is on every copy, so a read riding one is served
    // wherever it lands and costs no port of its own.
    unsigned riding = 0;
    if (m.ports.instPool) {
      // A pooled port serves either direction, and the binding may already have
      // ridden a write on a read's where the two never issue together, so what
      // the bank was BUILT with is the question and not the two directions
      // separately. Within one instance's pool, one instance holds it.
      if (m.portsBuilt <= *m.ports.instPool)
        continue;
      riding = m.readPortsBuilt + m.writePortsBuilt - m.portsBuilt;
      // Past it every copy takes every write and the reads share what is left,
      // so a written block RAM serves one read a copy, which is what the part
      // does: 1024x32 measures one tile at one read and two at two. Nothing
      // left is an array this row cannot hold at all.
      per = std::min(per, *m.ports.instPool > m.writePortsBuilt
                              ? *m.ports.instPool - m.writePortsBuilt
                              : 0u);
    }
    unsigned own = m.readPortsBuilt - riding;
    if (!per || own <= per)
      continue;
    m.instances = (own + per - 1) / per;
    // Each bank ranks the read ports that reach it and hands them out a whole
    // instance at a time. Per bank, not over the memory: `readPortsBuilt` is
    // the largest any one bank holds, so ranking every colour together would
    // run past it wherever two banks hold different ones and would put more
    // reads on an instance than it has. A read on a write's bus goes to the
    // first instance, where the port it rides already exists.
    llvm::SmallDenseSet<unsigned> writePorts;
    for (const MemUnit::Access &acc : m.accesses)
      if (acc.isWrite)
        writePorts.insert(acc.port);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && ma.isWrite)
          writePorts.insert(ma.port);
    llvm::SmallVector<llvm::SmallVector<unsigned>> byBank(m.numBanks);
    auto reaches = [&](std::optional<unsigned> bank, unsigned port) {
      // A skew's `staticBank` is a slot, not a bank, and its accesses read
      // every bank through the crossbar, as an unassigned access does.
      if (bank && !m.skewed)
        byBank[*bank].push_back(port);
      else
        for (auto &ports : byBank)
          ports.push_back(port);
    };
    for (const MemUnit::Access &acc : m.accesses)
      if (!acc.isWrite)
        reaches(acc.staticBank, acc.port);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && !ma.isWrite)
          reaches(ma.bank, ma.port);
    for (auto [k, ports] : llvm::enumerate(byBank)) {
      llvm::sort(ports);
      ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
      unsigned rank = 0;
      for (unsigned port : ports) {
        unsigned inst = writePorts.contains(port) ? 0 : rank++ / per;
        assert(inst < m.instances && "a read ranked past the instances");
        m.readInstance[MemUnit::instanceKey(k, port)] = inst;
      }
    }
  }
}

void DatapathBuilder::classifyRealizations() {
  using R = MemUnit::Realization;
  for (MemUnit &m : dp.mems)
    // An argument is a boundary whatever shape its cells have upstream: this
    // module builds none of them and only holds ports on them.
    m.realization = m.external        ? R::Boundary
                    : m.isRom         ? R::Rom
                    : m.scattered     ? R::Scatter
                    : m.fitsStorage() ? R::Ram
                                      : R::RegisterFile;
}

void DatapathBuilder::enumerateBoundaryPorts() {
  llvm::SmallVector<Value> memRefs;
  for (const MemUnit &m : dp.mems)
    memRefs.push_back(m.memref);
  auto ownerOfMem = [&](MemId id) {
    return uniqueOwnerOf(dp.mems[id].memref, memRefs, memOwner(id));
  };
  auto key = [](MemId mem, bool write) {
    return (uint64_t(mem) << 1) | unsigned(write);
  };
  llvm::DenseMap<uint64_t, unsigned> group;

  for (MemUnit &m : dp.mems) {
    if (!m.external)
      continue;
    std::string owner = ownerOfMem(m.id);
    // A scattered argument's ports are per element, enumerated once for the
    // memory (not per access), since every access reads them all and selects.
    // Its accesses keep the default portIdx/portBase; nothing addresses them.
    if (m.scattered) {
      // The directions actually used decide the names: an argument used one way
      // takes the bare `A_k`, used both ways its two ports need telling apart.
      bool reads = false, writes = false;
      for (const MemUnit::Access &acc : m.accesses)
        (acc.isWrite ? writes : reads) = true;
      for (unsigned k = 0, e = m.depthWords; k < e; ++k) {
        MemUnit::ElemPort p;
        if (reads)
          p.in = elemBase(owner, k, writes ? ElemDir::In : ElemDir::Only);
        if (writes) {
          p.out = elemBase(owner, k, reads ? ElemDir::Out : ElemDir::Only);
          p.we = portWe(p.out);
        }
        m.elemPorts.push_back(std::move(p));
      }
      continue;
    }
    // One boundary port group per bound port: accesses that provably never
    // issue together share a port, and so share the interface the caller backs
    // the array with, driving it through a select on their own activation. A
    // group per access instead makes every caller provide that many interfaces
    // for bandwidth the schedule never asks for.
    //
    // Keyed by bank as well as port, since a port index is one per bank and two
    // accesses routed to different banks are different interfaces. A
    // data-dependent banked access spans every interface, and `bindMemoryPorts`
    // already gave it a port of its own. One map per direction, since the two
    // number their groups in their own port list.
    llvm::SmallDenseMap<std::pair<unsigned, unsigned>, unsigned> groupOfPort[2];
    for (auto [a, acc] : llvm::enumerate(m.accesses)) {
      auto &ports = acc.isWrite ? dp.writePorts : dp.readPorts;
      auto [it, isNew] = groupOfPort[acc.isWrite].try_emplace(
          {acc.staticBank.value_or(~0u), acc.port}, ports.size());
      if (!isNew) {
        acc.portIdx = it->second;
        acc.portBase = m.accesses[ports[acc.portIdx].idx].portBase;
        continue;
      }
      acc.portIdx = ports.size();
      acc.portBase =
          memBase(owner, acc.isWrite, group[key(m.id, acc.isWrite)]++);
      ports.push_back({m.id, unsigned(a)});
    }
  }
  // One port group per accessor of a (memory, role), concurrent and un-muxed: a
  // mux would time-share the port a second accessor exists to avoid.
  for (CallUnit &cu : dp.calls)
    for (CallUnit::MemArg &ma : cu.memArgs)
      if (ma.isBoundary)
        ma.topBase = memBase(ownerOfMem(ma.mem), ma.isWrite,
                             group[key(ma.mem, ma.isWrite)]++);
}

Source DatapathBuilder::resolveValue(Value v) {
  // A scheduled producer bound during the region walk: a compute unit, a
  // memory / stream read port, a call result, or a hoisted literal.
  if (auto it = producerOf.find(v); it != producerOf.end())
    return it->second;
  if (auto *def = v.getDefiningOp()) {
    // A nested region's result: the survivor register the producing region
    // latched it into, the ONLY channel a value leaves a region by.
    if (isa<dcp::DCPathRegionOpInterface>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
    return {}; // an unmodelled producer
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second; // a scalar function argument
  // A `dcp.pipeline` block argument. Arg 0 is the induction counter: its
  // region's counter register, held stable for the whole of a nested run.
  auto barg = cast<BlockArgument>(v);
  auto pipe = dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
  if (!pipe)
    return {};
  assert(regionIdxOf.count(pipe) &&
         "every dcp region op is registered by the region walk");
  RegionId rid = regionIdxOf.lookup(pipe);
  unsigned arg = barg.getArgNumber();
  if (arg == 0)
    return Source{Source::Kind::Counter, rid, 0};
  // The rest are the loop-carried values, readable only where the region
  // LATCHES them into a survivor. A childless counted reduction fuses its
  // accumulator in, so only `resolveOperand`'s recurrence edge reads it.
  const RegionBlock &owner = dp.regions[rid];
  if (!owner.container && !owner.conditional)
    return {};
  return Source{Source::Kind::Survivor, rid, arg - 1};
}

// The bits region \p rb's iteration counter needs. The register holds
// `lb, lb+step, ...` and the terminator compares `iv + step` against `ub` under
// a SIGNED predicate, so three values ride this width: `lb`, `step` (its own
// cell) and `lb + trip*step` (the one-past value, also `ub`). `step` must be
// counted even for an empty loop, whose `0 to 0` bounds alone would fit in a
// bit. A loop whose trip only an ASSUMPTION bounds uses the bound in place of
// the count.
static unsigned counterWidth(const RegionBlock &rb) {
  auto pipe = cast<dcp::DCPathPipelineOp>(rb.op);
  std::optional<int64_t> trip = rb.tripCount ? rb.tripCount : rb.tripBound;
  if (rb.conditional || !trip || pipe.getLbBound() || pipe.getStepBound())
    return kIndexWidth;
  int64_t lb = pipe.getLb().value_or(0), step = pipe.getStep().value_or(1);
  int64_t span, last;
  if (llvm::MulOverflow(*trip, step, span) || llvm::AddOverflow(lb, span, last))
    return kIndexWidth;
  auto bits = [](int64_t v) {
    return static_cast<unsigned>(
        APInt(64, static_cast<uint64_t>(v), /*isSigned=*/true)
            .getSignificantBits());
  };
  return std::min(kIndexWidth, std::max({bits(lb), bits(step), bits(last)}));
}

void DatapathBuilder::deriveCounterTypes() {
  for (RegionBlock &rb : dp.regions)
    if (rb.kind == RegionBlock::Kind::Cyclic)
      rb.counterType = IntegerType::get(func.getContext(), counterWidth(rb));
}

void DatapathBuilder::recordRegionBounds(ArrayRef<Operation *> regionOps) {
  // A runtime induction bound (ub / lb / step) crosses the same F->G channel a
  // data survivor does; an unresolvable one is reported, not silently run.
  auto recordBound = [&](Operation *pipe, Value b, Source &into) {
    if (!b)
      return;
    into = resolveValue(b);
    if (!into) {
      unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
          << "Loop bound is produced by a value this region cannot read; such "
             "a cross-region value hand-off is not lowered yet";
      dp.infeasible = true;
    }
  };
  for (Operation *op : regionOps)
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(op)) {
      RegionBlock &rb = dp.regions[regionIdxOf.lookup(op)];
      recordBound(op, pipe.getDynamicBound(), rb.ubSource);
      recordBound(op, pipe.getLbBound(), rb.lbSource);
      recordBound(op, pipe.getStepBound(), rb.stepSource);
      // A compile-time bound ties in as a literal cell. The ub is derivable
      // only when lb and step are literal too; otherwise `tripCount` carries
      // `lb + trip*step` to `terminatorOf`, since no cell can hold arithmetic.
      int64_t lb = pipe.getLb().value_or(0), step = pipe.getStep().value_or(1);
      if (!rb.lbSource)
        rb.lbSource = constant(lb, rb.counterType);
      if (!rb.stepSource)
        rb.stepSource = constant(step, rb.counterType);
      if (!rb.ubSource && rb.tripCount && !pipe.getLbBound() &&
          !pipe.getStepBound())
        rb.ubSource = constant(lb + *rb.tripCount * step, rb.counterType);
    }
}

void DatapathBuilder::bindIOArgs() {
  for (BlockArgument arg : func.getArguments()) {
    if (isa<MemRefType>(arg.getType()))
      continue;
    // A stream arg is a FIFO channel, created lazily on its first get/put.
    if (isa<StreamType>(arg.getType()))
      continue;
    IOPort io;
    io.id = dp.ios.size();
    io.value = arg;
    io.type = arg.getType();
    ioOf[arg] = Source{Source::Kind::IO, io.id, 0};
    dp.ios.push_back(io);
  }
}

void DatapathBuilder::recordResults() {
  auto ret = cast<dcp::DCPathOutputOp>(func.getBody().front().getTerminator());
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    assert(!isa<MemRefType>(v.getType()) &&
           "a memref result should be an out-param by emit "
           "(buffer-results-to-out-params)");
    Result r;
    // An unresolvable result Source is swept by `validateDatapath`, so the
    // build finishes and the diagnostic is raised once, in one place.
    r.source = resolveValue(v); // survivor / passthrough IO / constant
    r.type = v.getType();
    r.name = resultPortName(i, ret.getNumOperands());
    dp.results.push_back(std::move(r));
  }
}

//===----------------------------------------------------------------------===//
// Interconnect derivation.
//===----------------------------------------------------------------------===//

Resolved DatapathBuilder::resolveOperand(Value v, Operation *consumer,
                                         unsigned ii) {
  int64_t tY = dcpStart(consumer);
  Operation *regionOp = consumer->getParentOp();

  // Register depth for an edge whose producer's result is ready at `ready`
  // (cycles after its issuing pulse): distance-many II turns plus the
  // consumer's cycle, minus the ready cycle.
  auto edge = [&](Source base, Value key, unsigned ready,
                  unsigned distance) -> Resolved {
    int64_t depth =
        static_cast<int64_t>(distance) * ii + tY - static_cast<int64_t>(ready);
    // The scheduler must never place a consumer before its operand is ready.
    // Asserting alone is not enough: the `unsigned` cast below would wrap, so a
    // release build reports, clamps to 0, and fails in `validateDatapath`.
    if (depth < 0) {
      assert(false && "the scheduler placed a consumer before its operand is "
                      "ready; the register depth would wrap");
      error(Stage::Emit, Code::CompilerInconsistency, consumer)
          << "Infeasible schedule; the operand is not ready until cycle "
          << (static_cast<int64_t>(ready) - static_cast<int64_t>(distance) * ii)
          << " but its consumer is scheduled at cycle " << tY
          << " (producer ready " << ready << ", dependence distance "
          << distance << ", II " << ii << ")";
      dp.infeasible = true;
      depth = 0;
    }
    return {base, key, static_cast<unsigned>(depth), ready, true};
  };

  // The one operand that does not read `v` at all: an unlatched iter_arg of the
  // consumer's own region is the loop recurrence, so the edge runs back to
  // whatever the loop assigns it, `distance` iterations away.
  if (auto barg = dyn_cast<BlockArgument>(v))
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
        pipe == regionOp && barg.getArgNumber() >= 1 &&
        !dp.regions[regionIdxOf.lookup(pipe)].container) {
      unsigned iterArg = barg.getArgNumber() - 1;
      SmallVector<unsigned, 2> chain;
      Value next = traceIterArgSource(pipe, iterArg, chain);
      unsigned distance = chain.size();
      Source base = next ? resolveValue(next) : Source{};
      Operation *def = next ? next.getDefiningOp() : nullptr;
      Resolved r;
      // A held `next` never moves while this loop runs, so from iteration
      // `distance` on the recurrence reads it unchanged, off a wire. Anything
      // else must be produced by an op this region schedules, since only then
      // is there an iteration of it to reach back to.
      if (isHeld(base))
        r = {base, Value(), 0, 0, true};
      else if (base && def && def->getParentOp() == regionOp) {
        r = edge(base, next, readyCycleOf(def), distance);
        // The one edge whose delay counts cycles between two iterations, so
        // this region may not defer an issue on its own.
        dp.regions[regionIdxOf.lookup(regionOp)].cycleIndexedState = true;
      } else {
        // Anchored on the loop, where the faulty carried assignment is, rather
        // than on the consumer `validateDatapath` would anchor on.
        unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
            << "Value " << iterArg
            << " carried by this loop is assigned from a value the loop's own "
               "datapath cannot read; such a cross-region value hand-off is "
               "not lowered yet";
        dp.infeasible = true;
        return {};
      }
      // The emitter re-injects the identities on this consumer input, since the
      // recurrence register may sit elsewhere in the cycle. One per iteration
      // below `distance`: a chained carry shifts one iter_arg into the next, so
      // iteration n reads the init of the stage n steps down the chain, which
      // is what `chain` enumerates.
      for (unsigned stage : chain)
        r.inits.push_back(resolveValue(pipe.getInits()[stage]));
      // An unresolvable init leaves the accumulator to free-run from reset.
      // Only this site knows an init was expected; None is normal elsewhere.
      if (!llvm::all_of(r.inits, [](Source s) { return bool(s); })) {
        unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
            << "Loop-carried accumulator has an initial value this region "
               "cannot read; such a cross-region value hand-off is not "
               "lowered yet";
        dp.infeasible = true;
        r.inits.clear();
      }
      return r;
    }

  Source base = resolveValue(v);
  if (!base)
    return {};
  if (isHeld(base))
    return {base, Value(), 0, 0, true};
  // The counter presents its index at cycle 0 of ITS region, so a consumer
  // scheduled at tY delays it that far.
  if (base.kind == Source::Kind::Counter)
    return edge(base, v, /*ready=*/0, /*distance=*/0);
  // A scheduled producer: readable only from the region it issues in, and only
  // after it lands.
  Operation *def = v.getDefiningOp();
  assert(def && "a scheduled Source is produced by an op");
  if (def->getParentOp() != regionOp)
    return {}; // cross-region hand-off unsupported
  return edge(base, v, readyCycleOf(def), /*distance=*/0);
}

RegId DatapathBuilder::insertRegister(Value key, ArrayRef<unsigned> depths,
                                      RegHead head, RegionId region) {
  Register reg;
  reg.id = dp.regs.size();
  reg.value = key;
  reg.type = key.getType();
  // The chain is as long as its deepest consumer needs; the shallower ones read
  // their own tap off it (Source::Reg's `outPort`).
  reg.depth = *llvm::max_element(depths);
  reg.input = head.base;
  reg.ready = head.ready;
  dp.regions[region].regs.push_back(reg.id);
  dp.regs.push_back(reg);
  return reg.id;
}

void DatapathBuilder::deriveInterconnect() {
  allocateInputSlots();
  resolveUnitInputs();
  resolveAccessOperands();
  insertRegisters();
}

void DatapathBuilder::allocateInputSlots() {
  for (FuncUnit &u : dp.units) {
    unsigned n = u.repOp()->getNumOperands();
    u.inputs.assign(n, Source{});
    u.inputInits.resize(n); // parallel; filled for recurrence inputs below
  }
  for (MemUnit &m : dp.mems) {
    for (MemUnit::Access &acc : m.accesses) {
      SmallVector<Value> operands;
      dcpAddressing(acc.op, acc.addrMap, operands);
      acc.addr.assign(operands.size(), Source{});
      // Which bank this access reaches: the one `assign-banks` decided, or all
      // of them when it decided none. The one write of `staticBank`, so it
      // covers the unbanked memrefs too.
      if (m.numBanks == 1) {
        acc.staticBank = 0; // the one bank there is
        continue;
      }
      acc.staticBank = assignedBankOf(acc.op);
      // Under a skew the recorded index is a SLOT, which no derivation off the
      // map can confirm (the bank it names is only fixed at run time), so the
      // assert below would be comparing two different things.
      if (m.layout.skew())
        continue;
      // `bankAddress` builds the offset WITHIN this bank out of `addrMap`, so
      // where the map still resolves a bank on its own it has to be the decided
      // one. It often cannot: the decision read the loop steps too.
      std::optional<int64_t> derived =
          staticBankOf(m.layout, acc.addrMap,
                       cast<MemRefType>(m.memref.getType()).getShape());
      assert((!acc.staticBank || !derived ||
              *derived == static_cast<int64_t>(*acc.staticBank)) &&
             "the assigned bank is not the one this access's address map "
             "reaches");
      (void)derived;
    }
    assignLanes(m);
  }
}

// Group a skewed memory's accesses into LANES: within a lane the slots are
// distinct, so the accesses reach distinct banks and share one port on each.
// Same-slot accesses always collide, so each takes the next lane, the port the
// model billed it. Numbered per region and reads apart from writes, the
// granularity a port is contended at.
void DatapathBuilder::assignLanes(MemUnit &m) {
  // A constant table has no ports to share (it is combinational), and an
  // argument's ports are boundary interfaces the manifest already published,
  // one set per access, which is why `assign-banks` assigns it no slot either.
  if (!m.layout.skew() || m.external || m.isRom)
    return;
  // One access without a slot and the array is back to crossbarring: a lane
  // shares a port on the strength of every user holding a distinct slot.
  if (llvm::any_of(m.accesses,
                   [](const MemUnit::Access &a) { return !a.staticBank; }))
    return;
  m.skewed = true;
  llvm::DenseMap<std::tuple<unsigned, unsigned, unsigned>, unsigned> used;
  for (MemUnit::Access &acc : m.accesses) {
    assert(*acc.staticBank < m.numBanks && "a slot indexes the skew's banks");
    acc.lane = used[{acc.region, acc.isWrite, *acc.staticBank}]++;
  }
}

void DatapathBuilder::recordEdge(const Resolved &r, Source &slot,
                                 unsigned regionIdx) {
  if (!r.ok)
    return;
  if (r.depth == 0) {
    slot = r.base;
    return;
  }
  RegKey key{r.key, regionIdx};
  depthsByKey[key].push_back(r.depth);
  assert((!headByKey.count(key) || headByKey[key].ready == r.ready) &&
         "one value's edges disagree on when it lands");
  headByKey[key] = {r.base, r.ready};
  pending.push_back({&slot, key, r.depth});
}

void DatapathBuilder::recordCarriedEdge(const Resolved &r, Value operand,
                                        Operation *consumer, Source &slot,
                                        unsigned regionIdx) {
  if (r.inits.empty()) {
    recordEdge(r, slot, regionIdx);
    return;
  }
  // Arms of one issue pulse, split by iteration: identity n at iteration n and
  // the edge itself from `inits.size()` on, the same shape a shared unit port
  // carries. Every arm is sized before any is filled, since `recordEdge` takes
  // a pointer into `sources`.
  unsigned arms = r.inits.size() + 1;
  muxBuilds.push_back({&slot,
                       regionIdx,
                       operand.getType(),
                       SmallVector<Operation *, 2>(arms, consumer),
                       {},
                       {}});
  MuxBuild &mb = muxBuilds.back();
  for (unsigned n = 0; n + 1 < arms; ++n)
    mb.phases.push_back({Mux::Phase::At, n});
  mb.phases.push_back({Mux::Phase::From, arms - 1});
  mb.sources.assign(r.inits.begin(), r.inits.end()); // held: literal/port/...
  mb.sources.resize(arms);
  recordEdge(r, mb.sources.back(), regionIdx);
}

void DatapathBuilder::resolveUnitInputs() {
  for (FuncUnit &u : dp.units) {
    Operation *op0 = u.repOp();
    unsigned ridx = regionIdxOf.lookup(op0->getParentOp());
    unsigned ii = dp.regions[ridx].ii.value_or(1);
    unsigned nPorts = op0->getNumOperands();
    if (u.boundOps.size() == 1) {
      for (unsigned k = 0; k < nPorts; ++k) {
        auto r = resolveOperand(op0->getOperand(k), op0, ii);
        recordEdge(r, u.inputs[k], ridx);
        u.inputInits[k] =
            r.inits; // empty unless k reads a loop-carried iter_arg
      }
      continue;
    }
    // Shared unit: resolve every bound op's port k independently (each may need
    // its own register depth), then a mux picks per op's issue cycle. All
    // resolved up front because a recurrence operand takes an arm per identity
    // and `recordEdge` takes a pointer into `sources`, so the list must be
    // sized before any edge lands in it.
    for (unsigned k = 0; k < nPorts; ++k) {
      SmallVector<Resolved, 2> edges;
      unsigned arms = 0;
      for (const auto &bo : u.boundOps) {
        edges.push_back(resolveOperand(bo.first->getOperand(k), bo.first, ii));
        arms += 1 + edges.back().inits.size();
      }
      muxBuilds.push_back(
          {&u.inputs[k], ridx, op0->getOperand(k).getType(), {}, {}, {}});
      MuxBuild &mb = muxBuilds.back();
      mb.sources.resize(arms);
      unsigned arm = 0;
      for (auto [j, r] : llvm::enumerate(edges)) {
        Operation *opj = u.boundOps[j].first;
        // Each identity rides an arm of its own rather than a mux in front of
        // the port: a shared port carries a different op's operand each cycle,
        // leaving no cycle to time such a mux against. An unresolvable init is
        // reported by `resolveOperand` and takes no arm.
        for (auto [n, init] : llvm::enumerate(r.inits)) {
          mb.ops.push_back(opj);
          mb.phases.push_back({Mux::Phase::At, static_cast<unsigned>(n)});
          mb.sources[arm++] = init; // held: a literal, a port, a survivor
        }
        mb.ops.push_back(opj);
        mb.phases.push_back(r.inits.empty() ? Mux::Phase{}
                                            : Mux::Phase{Mux::Phase::From,
                                                         static_cast<unsigned>(
                                                             r.inits.size())});
        recordEdge(r, mb.sources[arm++], ridx);
      }
    }
  }
}

// Floor-based residue, which is what an affine `mod` means: non-negative for a
// positive divisor whatever the sign of \p a, so a digit register starts in
// range and its unsigned wrap compare is exact.
static int64_t mod(int64_t a, int64_t b) {
  return a - llvm::divideFloorSigned(a, b) * b;
}

// The width a stride register is built at: enough bits for every value it
// holds, and for the raw pre-wrap sum its update compares before fixing.
// A WRAPPING register lives in `[0, wrap)`; `raw = cur + step + bump` reaches
// `2*wrap - 1` going up (`step + bump <= wrap` by construction) or borrows
// from just below zero going down, same headroom either way under the
// unsigned compare. A PLAIN accumulator runs from `init` over the loop's
// advances, one past the last iteration since a counted controller still
// computes the step it does not take.
static unsigned strideWidth(const RegionBlock::AddrStride &s,
                            std::optional<int64_t> trip) {
  auto bits = [](uint64_t v) {
    return std::min(kIndexWidth, std::max(1u, APInt(64, v).getActiveBits()));
  };
  if (s.wrap) {
    assert(s.wrap > 0 && "a wrap is a modulus, and the update compares against "
                         "it unsigned");
    return bits(2 * static_cast<uint64_t>(s.wrap) - 1);
  }
  int64_t span, last;
  if (!trip || llvm::MulOverflow(s.step + s.bump, *trip, span) ||
      llvm::AddOverflow(s.init, span, last) || s.init < 0 || last < 0)
    return kIndexWidth;
  return bits(std::max(s.init, last));
}

// The slot in \p rb holding \p want, appended if no identical stride is there.
// The width is DERIVED from the rest, so it takes no part in the comparison.
static unsigned slotFor(RegionBlock &rb, RegionBlock::AddrStride want) {
  want.width = strideWidth(want, rb.tripCount ? rb.tripCount : rb.tripBound);
  auto *it =
      llvm::find_if(rb.addrStrides, [&](const RegionBlock::AddrStride &a) {
        return a.init == want.init && a.step == want.step &&
               a.bump == want.bump && a.wrap == want.wrap &&
               a.down == want.down && a.hasCarry == want.hasCarry &&
               (!a.hasCarry || a.carry == want.carry);
      });
  if (it == rb.addrStrides.end()) {
    rb.addrStrides.push_back(want);
    it = std::prev(rb.addrStrides.end());
  }
  return static_cast<unsigned>(it - rb.addrStrides.begin());
}

// The register holding `t.coeff * digit` over region \p rid's counter, plus the
// companion residue register a quotient digit carries off.
//
// \p base is absorbed by the first NON-WRAPPING register and zeroed, which
// avoids an extra adder on the port's setup path. A wrapping register cannot
// take it: it holds a residue whose wrap assumes it stays in range.
static MemUnit::Access::ScaledTerm strideFor(Datapath &dp, unsigned rid,
                                             const SplitAddress::Term &t,
                                             int64_t &base) {
  RegionBlock &rb = dp.regions[rid];
  // The digit's argument, `scale * counter + offset`: where it starts and what
  // it advances by, which is all the register needs. Running backwards, every
  // wrap becomes a borrow and every carry a decrement.
  int64_t start = t.scale * *dp.constantOf(rb.lbSource) + t.offset;
  int64_t advance = t.scale * *dp.constantOf(rb.stepSource);
  bool down = advance < 0;
  RegionBlock::AddrStride want;
  if (!t.isDigit()) {
    want = {t.coeff * start + base, t.coeff * advance};
    base = 0;
  } else if (t.divisor == 1) {
    // A pure residue accumulates and wraps on itself.
    want = {t.coeff * mod(start, t.modulus),
            t.coeff * advance,
            0,
            t.coeff * t.modulus,
            0,
            false,
            down};
  } else {
    // A quotient advances by one wherever its argument crosses a multiple of
    // the divisor, which is what the companion residue register says. Unscaled,
    // unreferenced by any access, and shared by every digit over that argument.
    unsigned carry = slotFor(
        rb, {mod(start, t.divisor), advance, 0, t.divisor, 0, false, down});
    int64_t q = llvm::divideFloorSigned(start, t.divisor);
    want = {t.coeff * (t.modulus ? mod(q, t.modulus) : q),
            0,
            down ? -t.coeff : t.coeff,
            t.modulus ? t.coeff * t.modulus : 0,
            carry,
            true,
            down};
  }
  return {rid, slotFor(rb, want)};
}

// Merge the terms landing on the same DIGIT of the same region, \p region
// giving each operand position the region whose counter it follows. A region
// has one counter, so those terms add their coefficients rather than taking a
// register and an adder each.
static SmallVector<SplitAddress::Term>
mergeTermsByDigit(ArrayRef<SplitAddress::Term> terms,
                  ArrayRef<std::optional<unsigned>> region) {
  using Digit = std::tuple<unsigned, int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<Digit, unsigned> group;
  SmallVector<SplitAddress::Term> merged;
  for (const SplitAddress::Term &t : terms) {
    Digit d{*region[t.operand], t.scale, t.offset, t.divisor, t.modulus};
    auto [it, isNew] = group.try_emplace(d, merged.size());
    if (isNew)
      merged.push_back(t);
    else
      merged[it->second].coeff += t.coeff;
  }
  llvm::erase_if(merged, [](const SplitAddress::Term &t) { return !t.coeff; });
  return merged;
}

// Reduce ONE cone of an address. The in-bank offset and the bank digit are the
// same kind of expression over the same operands: a bank under a cyclic
// partition is `counter mod F`, a wrap register like a delinearized subscript.
static MemUnit::Access::Reduced
reduceCone(Datapath &dp, AffineExpr e, AffineMap addrMap,
           ArrayRef<std::optional<unsigned>> region) {
  MemUnit::Access::Reduced out;
  if (!e)
    return out;
  SplitAddress sp =
      splitAddress(e, addrMap.getNumDims(), addrMap.getNumSymbols(),
                   [&](unsigned p) -> std::optional<int64_t> {
                     if (!region[p])
                       return std::nullopt;
                     return dp.constantOf(dp.regions[*region[p]].stepSource);
                   });
  int64_t base = sp.base;
  for (const SplitAddress::Term &t : mergeTermsByDigit(sp.terms, region))
    out.terms.push_back(strideFor(dp, *region[t.operand], t, base));
  // The digits the residual reads, IN ORDER and undeduplicated: it names them
  // by position, and the scheduler priced the same list from the same
  // `splitAddress`, so the two cannot disagree about which is which.
  for (const SplitAddress::Term &t : sp.reads)
    out.reads.push_back(strideFor(dp, *region[t.operand], t, base));
  out.base = base; // 0 unless no register took it
  out.residual = sp.residual;
  return out;
}

// Address strength reduction: decide which TERMS of each access's address can
// come from registers that advance with the loop counters, and record the
// scaled counters those registers need. A term that does not qualify stays in
// the residual, so this only ever removes arithmetic.
//
// Runs after `resolveAccessOperands` (a term has to resolve to a counter) and
// after `recordRegionBounds` (a stride is a constant only if the counter's
// bounds are). `splitAddress` is the same decomposition the scheduler priced
// the access with.
//
// An operand arriving through a delay chain is peeled to its HEAD: the scaled
// counter is delayed once for the whole sum rather than per operand, so
// counters wanted at different cycles cannot share that one delay; the first
// one's cycle decides and the rest stay in the residual.
//
// The split runs on the IN-BANK OFFSET (the flat index for an unbanked
// memref), which is what lets a banked access reduce at all: `buf[i, 4*j]`
// under a cyclic-4 last axis offsets by `i*extent + j`, as linear as any.
void DatapathBuilder::planAddressGenerators() {
  for (MemUnit &m : dp.mems) {
    auto shape = cast<MemRefType>(m.memref.getType()).getShape();
    for (MemUnit::Access &acc : m.accesses) {
      // Which operands a register can follow, decided up front so the predicate
      // handed to `splitAddress` is a pure one.
      SmallVector<std::optional<unsigned>> region(acc.addr.size());
      std::optional<unsigned> delay;
      for (unsigned p = 0, e = acc.addr.size(); p < e; ++p) {
        Source s = acc.addr[p];
        unsigned d = 0;
        if (s.kind == Source::Kind::Reg) {
          d = s.outPort;
          s = dp.regs[s.id].input;
        }
        if (s.kind != Source::Kind::Counter || (delay && *delay != d))
          continue;
        RegionBlock &rb = dp.regions[s.id];
        if (!dp.constantOf(rb.lbSource) || !dp.constantOf(rb.stepSource))
          continue;
        delay = d;
        region[p] = s.id;
      }
      AddressExprs e =
          addressExprsOf(m.layout, acc.addrMap, shape, acc.staticBank);
      acc.offset = reduceCone(dp, e.offset, acc.addrMap, region);
      acc.bank = reduceCone(dp, e.bank, acc.addrMap, region);
      // Both cones read the same operands, so one delay covers them, and a
      // digit the residual reads is a register like any other.
      bool anyRegister = !acc.offset.terms.empty() || !acc.bank.terms.empty() ||
                         !acc.offset.reads.empty() || !acc.bank.reads.empty();
      acc.addrDelay = anyRegister ? delay.value_or(0) : 0;
    }
  }
}

void DatapathBuilder::resolveAccessOperands() {
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      SmallVector<Value> operands;
      AffineMap ignored;
      dcpAddressing(acc.op, ignored, operands);
      for (unsigned k = 0, e = operands.size(); k < e; ++k)
        recordCarriedEdge(resolveOperand(operands[k], acc.op, ii), operands[k],
                          acc.op, acc.addr[k], ridx);
      if (acc.isWrite) {
        Value datum = cast<dcp::DCPathStoreOp>(acc.op).getValue();
        recordCarriedEdge(resolveOperand(datum, acc.op, ii), datum, acc.op,
                          acc.data, ridx);
      }
    }

  // Re-stamp an access's schedule cycle. `start` is the single source both the
  // datapath and `dcpStart` read, so the attribute and the cached stage move
  // together.
  auto restamp = [](StreamChannel::Access &acc, int64_t cycle) {
    acc.op->setAttr(
        "start",
        IntegerAttr::get(cast<IntegerAttr>(acc.op->getAttr("start")).getType(),
                         cycle));
    acc.stage = static_cast<unsigned>(cycle);
  };

  // A stream put's data driver resolves through the same reg-depth path as a
  // store's; a predicated get/put's i1 predicate is likewise delayed to the
  // access stage, so it gates the handshake in deriveStallShell.
  for (StreamChannel &s : dp.streams) {
    // Cycles the bump below has inserted into each region. The scheduler put a
    // channel's accesses on DISTINCT cycles, so a bump shifts every LATER
    // access too: moving one alone would land it on the next.
    llvm::DenseMap<unsigned, unsigned> inserted;
    for (StreamChannel::Access &acc : s.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      unsigned &shift = inserted[ridx];
      if (shift)
        restamp(acc, dcpStart(acc.op) + shift);
      if (acc.isPut) {
        auto token = cast<StreamPutOp>(acc.op).getValue();
        auto r = resolveOperand(token, acc.op, ii);
        // AXI-S data stability: a stage>=1 put's valid pulse persists into the
        // drain under back-pressure, so a transient din could commit stale
        // data. Bump its stage by one to route it through a frozen register.
        if (r.ok && r.depth == 0 && dcpStart(acc.op) >= 1 &&
            isTransientDin(token)) {
          restamp(acc, dcpStart(acc.op) + 1);
          ++shift;
          r = resolveOperand(token, acc.op, ii);
        }
        recordCarriedEdge(r, token, acc.op, acc.data, ridx);
      }
      auto pred = isa<StreamGetOp>(acc.op)
                      ? cast<StreamGetOp>(acc.op).getPred()
                      : cast<StreamPutOp>(acc.op).getPred();
      if (pred) {
        // Unlike `acc.data` (a None Source trips an assert), a None `acc.when`
        // reads as "unconditional", so an unresolved predicate would silently
        // turn a masked get/put into an every-cycle one.
        auto pr = resolveOperand(pred, acc.op, ii);
        if (!pr.ok) {
          unsupported(Stage::Emit, Code::CrossRegionHandOff, acc.op)
              << "Predicate of this stream access is produced by a value the "
                 "region cannot read; such a cross-region value hand-off is "
                 "not lowered yet, and the access would otherwise fire "
                 "unconditionally";
          dp.infeasible = true;
        }
        recordCarriedEdge(pr, pred, acc.op, acc.when, ridx);
      }
    }
  }
}

void DatapathBuilder::insertRegisters() {
  // One register per (value, region) key, keyed by its RegId, to patch the
  // pending slots that read it (each in the same region the register lives in).
  llvm::DenseMap<RegKey, RegId> keyToReg;
  for (auto &kv : depthsByKey)
    keyToReg[kv.first] = insertRegister(kv.first.first, kv.second,
                                        headByKey[kv.first], kv.first.second);

  for (const RegDepth &p : pending)
    *p.slot = Source{Source::Kind::Reg, keyToReg[p.key], p.depth};

  // Materialize sharing muxes: the sources are final once the registers are
  // built and the pending slots patched. One shared driver needs no mux.
  auto sameSource = [](const Source &a, const Source &b) {
    return a.kind == b.kind && a.id == b.id && a.outPort == b.outPort;
  };
  auto unphased = [](const Mux::Phase &p) {
    return p.kind == Mux::Phase::Always;
  };
  for (MuxBuild &mb : muxBuilds) {
    Source &slot = *mb.slot;
    // One driver across every arm is a wire. A phased arm never collapses: its
    // pair carries the same port in different iterations, not the same value.
    if (llvm::all_of(mb.phases, unphased) &&
        llvm::all_of(mb.sources, [&](const Source &s) {
          return sameSource(s, mb.sources[0]);
        })) {
      slot = mb.sources[0];
      continue;
    }
    Mux mx;
    mx.id = dp.muxes.size();
    mx.region = mb.region;
    mx.type = mb.type;
    mx.sources.assign(mb.sources.begin(), mb.sources.end());
    mx.selectOps.assign(mb.ops.begin(), mb.ops.end());
    mx.phases.assign(mb.phases.begin(), mb.phases.end());
    dp.regions[mb.region].muxes.push_back(mx.id);
    slot = Source{Source::Kind::Mux, mx.id, 0};
    dp.muxes.push_back(std::move(mx));
  }
}

void DatapathBuilder::allocateUnits(ArrayRef<SmallVector<UnitId, 2>> groups) {
  if (groups.empty())
    return; // the trivial allocation, which the walk already built

  // Where each unit folds: itself, unless the policy named it in a group.
  SmallVector<UnitId> leader(dp.units.size());
  std::iota(leader.begin(), leader.end(), 0);
  for (const SmallVector<UnitId, 2> &group : groups)
    for (UnitId uid : group) {
      assert(leader[uid] == uid &&
             "a policy named one unit in two groups; the second fold would "
             "silently win and its ops would issue on a unit nothing checked "
             "them against");
      leader[uid] = group.front();
    }

  // Rebuild rather than empty the folded-away entries: a `FuncUnit` with no
  // bound op has no `repOp()`, so a dense table keeps that an invariant instead
  // of a hazard every consumer has to remember to skip.
  SmallVector<UnitId> remap(dp.units.size(), 0);
  std::vector<FuncUnit> allocated;
  for (UnitId old = 0, e = dp.units.size(); old < e; ++old) {
    if (leader[old] != old)
      continue;
    remap[old] = allocated.size();
    allocated.push_back(std::move(dp.units[old]));
    allocated.back().id = remap[old];
  }
  // The leader keeps `boundOps.front()`, so `repOp()` and every name derived
  // from it are the ones the trivial allocation would have produced.
  for (UnitId old = 0, e = dp.units.size(); old < e; ++old)
    if (leader[old] != old)
      for (const std::pair<Operation *, unsigned> &bo : dp.units[old].boundOps)
        allocated[remap[leader[old]]].boundOps.push_back(bo);
  dp.units = std::move(allocated);

  // Region membership: the folded-away ids are gone, the survivors renumbered.
  for (RegionBlock &rb : dp.regions) {
    SmallVector<UnitId, 4> kept;
    for (UnitId uid : rb.units)
      if (leader[uid] == uid)
        kept.push_back(remap[uid]);
    rb.units = std::move(kept);
  }
  // The two provenance maps, rewritten FROM the table rather than alongside it,
  // so a Source's bound-op index cannot drift from the slot it names. They are
  // the whole of what holds a UnitId at this phase: no `record*` pass has run.
  for (const FuncUnit &u : dp.units)
    for (auto [slot, bo] : llvm::enumerate(u.boundOps)) {
      dp.opToUnit[bo.first] = u.id;
      producerOf[bo.first->getResult(0)] =
          Source{Source::Kind::Unit, u.id, static_cast<unsigned>(slot)};
    }
}

// Composition predecessors of each top-level region (`rb.predecessors`): the
// earlier top-level siblings it must start after, all attributed to the
// top-level ancestor. Per-region binding keeps functional units from
// conflicting across regions, so the signals are the (1) to (3) below: a shared
// memref, a shared channel, a cross-region SSA edge. The emitter starts a
// predecessor-free region concurrently with the kernel `start` and gates the
// rest on their producers' joined `done`.
//
// `siblingPredecessors` answers the same question off the IR, so it
// over-approximates; this works off the BUILT model. The model wants the
// superset: a spurious edge there only lengthens a span, while one HERE would
// serialize real hardware.
void DatapathBuilder::recordSiblingDeps(ArrayRef<Operation *> regionOps) {
  // Top-level ancestor of a region (walk the container chain to the root).
  auto topOf = [&](RegionId r) {
    while (dp.regions[r].parent)
      r = *dp.regions[r].parent;
    return r;
  };

  // Every op inside a top-level region maps to that region's id, a nested child
  // folding into it. A value defined outside any region has no entry.
  DenseMap<Operation *, RegionId> opTop;
  for (Operation *regionOp : regionOps) {
    RegionId rid = regionIdxOf.lookup(regionOp);
    if (dp.regions[rid].parent)
      continue; // walk only from a top-level root
    opTop[regionOp] = rid;
    regionOp->walk([&](Operation *o) { opTop[o] = rid; });
  }

  auto addPred = [&](RegionId producer, RegionId consumer) {
    assert(producer < consumer && "a predecessor must precede its consumer");
    auto &preds = dp.regions[consumer].predecessors;
    if (!llvm::is_contained(preds, producer))
      preds.push_back(producer);
  };

  // One shared resource orders the top-level regions that touch it: chain them
  // in program order, so the rest follows transitively.
  SmallVector<RegionId, 4> tops;
  auto addSharer = [&](RegionId region) {
    RegionId t = topOf(region);
    if (!llvm::is_contained(tops, t))
      tops.push_back(t);
  };
  auto chainSharers = [&]() {
    llvm::sort(tops);
    for (unsigned j = 1; j < tops.size(); ++j)
      addPred(tops[j - 1], tops[j]);
    tops.clear();
  };

  // (1) A shared memref. A CallUnit masters memref operands without a
  // MemUnit::Access, so it counts as a sharer too, or a child could read a
  // buffer concurrently with an earlier writer.
  for (const MemUnit &m : dp.mems) {
    for (const MemUnit::Access &a : m.accesses)
      addSharer(a.region);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id)
          addSharer(cu.region);
    chainSharers();
  }

  // (2) A shared channel: a FIFO is one port carrying the program's token
  // order, so two regions touching it must run in sequence, else they drive it
  // together and (for two gets) pop the same token twice.
  for (const StreamChannel &s : dp.streams) {
    for (const StreamChannel::Access &a : s.accesses)
      addSharer(a.region);
    chainSharers();
  }

  // (3) Cross-region SSA edges: an op in one top-level region uses a value
  // produced in an earlier one (a scalar survivor). SSA dominance guarantees
  // the producer precedes the consumer in program order.
  func.walk([&](Operation *o) {
    auto uit = opTop.find(o);
    if (uit == opTop.end())
      return;
    RegionId consumer = uit->second;
    for (Value v : o->getOperands()) {
      auto *def = v.getDefiningOp();
      if (!def)
        continue;
      if (auto dit = opTop.find(def); dit != opTop.end()) {
        if (dit->second != consumer)
          addPred(dit->second, consumer);
        continue;
      }
      // A def no region owns binds no hardware and so orders nothing:
      // `enumerateRegions` is total over a block, and the only ops the reify
      // leaves outside a region are declarations.
      assert(isDeclarationOp(def) &&
             "a computing op outside every region drives a region's input");
    }
  });

  // The two relations, diffed: each edge `siblingPredecessors` has beyond this
  // one is a pair the span serializes and the hardware overlaps.
  if (!logging::detail::enabled(Level::Debug))
    return;
  SmallVector<RegionId> topIds;
  SmallVector<SmallVector<Operation *>> nodeOps;
  for (Operation *regionOp : regionOps) {
    RegionId rid = regionIdxOf.lookup(regionOp);
    if (dp.regions[rid].parent)
      continue;
    topIds.push_back(rid);
    nodeOps.push_back({regionOp});
  }
  auto modelled = siblingPredecessors(nodeOps);
  for (auto [i, rid] : llvm::enumerate(topIds))
    for (unsigned p : modelled[i])
      if (!llvm::is_contained(dp.regions[rid].predecessors, topIds[p]))
        debug(Stage::Emit, dp.regions[rid].op)
            << "Latency model orders region " << topIds[p] << " before region "
            << rid
            << ", the built model does not: the composed span pays for "
               "a hand-off the hardware overlaps";
}

//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

void DatapathBuilder::build() {
  dp.func = func;

  collectConstants();

  // dcp region ops in program order. Pre-order so an enclosing container is
  // processed first: the parent/child linkage and the outer-index counter
  // attribution rely on parent-before-child.
  SmallVector<Operation *> regionOps;
  func.walk<WalkOrder::PreOrder>([&](dcp::DCPathRegionOpInterface region) {
    regionOps.push_back(region);
  });

  // Scalar-argument IO ports: one of the maps `resolveValue` reads, so every
  // pass below sees a scalar func arg as an IO source.
  bindIOArgs();

  for (unsigned ridx = 0, e = regionOps.size(); ridx < e; ++ridx) {
    Operation *regionOp = regionOps[ridx];
    auto rb = addRegion(regionOp, ridx);
    for (Operation &opRef : regionBody(regionOp)->without_terminator())
      bindResource(&opRef, rb);
    // A dual guard binds its else-branch loose ops too, since regionBody gives
    // only the then block. Nested regions are walked in their own iteration.
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
      if (!sel.getElseRegion().empty())
        for (Operation &opRef :
             sel.getElseRegion().front().without_terminator())
          bindResource(&opRef, rb);
    dp.regions.push_back(std::move(rb));
  }

  // The allocation, settled here and not later: every pass below resolves
  // Values to Sources against the unit table (see `allocateUnits`).
  allocateUnits(
      policy.plan(dp, {cycleTime, dev.operators})); // trivial => a no-op
  assert(llvm::all_of(dp.units,
                      [](const FuncUnit &u) { return !u.boundOps.empty(); }) &&
         "the unit table is the allocation: a unit exists because ops are "
         "bound to it");

  deriveShapes();       // controller discriminant (needs every child)
  deriveCounterTypes(); // counter width (each loop's own range)
  // Everything below resolves Values to Sources, and so runs here rather than
  // during the walk: `resolveValue` needs the complete region model.
  // Every op the reify leaves in the module body binds no hardware:
  // `enumerateRegions` is total over a block, so anything that computes is
  // inside a region.
#ifndef NDEBUG
  for (Operation &op : func.getBody().front())
    assert((isa<dcp::DCPathRegionOpInterface, dcp::DCPathOutputOp>(&op) ||
            isDeclarationOp(&op)) &&
           "an operation outside every region reached the datapath");
#endif
  recordRegionResults(regionOps); // per-region results/recurrence + predicate
  recordCallScalars();            // each dcp.instance's scalar operand drivers
  recordCallDeps();               // composition DAG on the instance substrate
  reclassifyRoms();               // read-only is a property of the USE
  recordRegionBounds(regionOps);  // induction bounds, at that width
  recordResults();                // scalar func-result output ports
  deriveInterconnect();
  planAddressGenerators(); // address strength reduction (needs resolved terms)
  // Ports last of all: an access holds one only once it knows which bank it
  // commits to and which skew lane it holds, both settled by the interconnect
  // pass above, and the boundary port list is one group per bound port.
  bindMemoryPorts();
  classifyRealizations();
  enumerateBoundaryPorts();
  recordSiblingDeps(regionOps); // top-level composition DAG (concurrency gates)
  verifyBinding(dp); // MRT legality: no unit shared by conflicting ops
}

} // namespace mlir::allo::uarch
