/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h" // iface::ModuleInterface (CallUnit ports)
#include "allo/Microarch/Reservation.h" // verifyBinding (MRT legality)

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h"   // characterize (storage shape)
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GetGlobalOp/GlobalOp (ROM)
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <deque>

using namespace mlir;
using namespace mlir::allo;

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

// The body block of a dcp region op. A guard (dcp.select) has no `else` here
// (result-mux guards are unsupported), so its body is the `then` branch,
// which holds the guarded sub-schedule (child regions), gated by the predicate.
static Block *regionBody(Operation *regionOp) {
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
// counted loop, or the `dcp.condition`'s carried operands of a while, which
// skip its leading condition), so iter-arg k always maps to carried[k].
static std::pair<Operation *, unsigned>
traceIterArgSource(dcp::DCPathPipelineOp pipe, unsigned iterArg) {
  Block &body = pipe.getBody().front();
  auto carried = pipe.getCarriedValues();
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
  auto *def = v.getDefiningOp();
  return def ? std::make_pair(def, distance + 1)
             : std::make_pair<Operation *, unsigned>(nullptr, 0);
}

// Is \p v a transient FIFO-din value, one that changes while the region is
// back-pressured (`valid & ~ready`), so it must be captured into a
// chain-enable-frozen register before it drives a FIFO write? It is transient
// iff it is, or is a purely combinational function of, one of the two sources
// that move under back-pressure:
//   * a memory load: a live counter-addressed read (an external port or an
//     always-enabled seq.read), re-addressed as the counter advances/resets;
//   * the loop counter (pipeline block arg 0), reset to `lb` in the drain.
// A value built only from FIFO heads (held while their get is not popped),
// survivors / call results (latched for the producing region's life),
// constants, io, or *registered* (latency>=1) units is frozen with the datapath
// while back-pressured, so it needs no extra register. Combinational
// (latency-0) ops propagate transient-ness from their operands; the SSA din
// tree is acyclic (iter-args are stable block args), so the recursion
// terminates.
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
  // Stable producers: a FIFO head, a nested region's survivor, a constant.
  if (isa<StreamGetOp, dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
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
  // A `memref.get_global` names a module-level constant table: resolve the
  // global's initializer so the emitter can build a ROM (read-only, no writable
  // hlmem). An uninitialized global stays a plain internal memory.
  if (auto gg = memref.getDefiningOp<memref::GetGlobalOp>()) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        gg, gg.getNameAttr());
    assert(global && "get_global references an undefined memref.global");
    if (auto init = global.getInitialValue()) {
      m.romInit = *init;
      m.isRom = true;
    }
  }
  // Banking / ports from the same storage model the scheduler binds against
  // (allo.part / allo.bind.storage); depthWords is per-bank so that
  // numBanks * depthWords covers the array.
  auto mc = allo::characterize(memref, memLib.defaultImpl);
  m.numBanks = std::max(1u, mc.numBanks);
  // dcp-resolve-banking splits every *statically* banked internal array into
  // plain per-bank memrefs before emit; a memref still banked here is either
  // data-dependent (internal -> crossbar) or a partitioned argument (external
  // -> per-bank).
  m.portsPerBank = mc.portsPerBank;
  m.impl = mc.impl;
  // Access latency of the resolved primitive, from the same device table the
  // scheduler timed this memref's accesses against (`MemoryLibrary::timing`).
  // The emitter builds ports at these latencies; do not re-derive from `impl`.
  auto mkt = memLib.timing(m.impl);
  m.readLatency = mkt.latency.read;
  m.writeLatency = mkt.latency.write;
  // A dynamic-shape memref would silently fall to total == 0 -> depthWords == 0
  // -> a zero-depth internal hlmem / zero-width external address interface,
  // with no diagnostic. Allo arrays are statically shaped by this stage.
  assert(mt.hasStaticShape() &&
         "datapath memory requires a static shape (a dynamic memref sizes to "
         "depthWords 0)");
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
  // A container region nests another dcp region in its body (a loop wrapping
  // an inner loop). The nearest enclosing region op is the parent, already
  // processed by this pre-order walk, so it runs its children hierarchically.
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
    // A guard (dcp.select) splits its children by branch: one nested in the
    // else body is an else-child (runs iff the predicate is false), otherwise
    // a then-child, found by walking up to the child whose parent is the
    // select.
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
    // A guard (dcp.select): a predicated container. It has no counter or trip
    // of its own; it runs its children once iff the predicate holds, so it
    // stays Acyclic. `container` is set when its children link below.
    rb.guard = true;
  } else if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
    rb.kind = RegionBlock::Kind::Cyclic;
    rb.conditional = pipe.isWhileLoop(); // dcp.condition terminator: flushing
    // The counter block arg keeps the source IV's NameLoc (preserved by the
    // reifier); carry its name so the emitter labels the iteration counter (i).
    if (auto n = nameFromLoc(pipe.getBody().front().getArgument(0).getLoc()))
      rb.counterName = sanitizeCppIdentifier(*n);
    // `ii` is absent for a data-dependent sequential wrapper (rb.ii stays
    // unset); such a region has children, so `emitRegion` routes it to a
    // container path that never reads `ii`; reg-depth paths use `.value_or(1)`.
    if (std::optional<int64_t> ii = pipe.getIi())
      rb.ii = static_cast<unsigned>(*ii);
    if (auto len = pipe.getLengthAttr())
      rb.length = len.getInt();
    if (auto t = pipe.getTripAttr())
      rb.tripCount = t.getInt();
    rb.lb = pipe.getLb().value_or(0);
    rb.step = pipe.getStep().value_or(1);
  } else {
    rb.kind = RegionBlock::Kind::Acyclic;
    auto seq = cast<dcp::DCPathSequentialOp>(regionOp);
    if (auto len = seq.getLengthAttr())
      rb.length = len.getInt();
  }

  // Declared composition class + single-run latency, so the composer
  // dispatches on a declared property rather than re-deriving region shape.
  // A `counted_static` guard (dcp.select) may carry no `latency`, handing off
  // via handshake instead of a static offset.
  if (auto d = regionOp->getAttrOfType<DeterminacyEnumAttr>("determinacy"))
    rb.determinacy = d.getValue();
  if (auto lat = regionOp->getAttrOfType<IntegerAttr>("latency"))
    rb.staticLatency = lat.getInt();
  assert((!rb.staticLatency.has_value() ||
          rb.determinacy == DeterminacyEnum::CountedStatic) &&
         "a region with a static latency must be declared counted_static");
  return rb;
}

// Every callee port interface for argument \p arg, reads before writes. A
// callee arg accessed at several points has several ports (read-twice -> two
// reads; an accumulator -> a read and a write), one per access GROUP; a
// cyclically partitioned access has one interface per BANK within its group: a
// static bank is one single-element group per bank, a data-dependent bank is
// one group spanning every bank (the child crossbars internally). Returning
// every per-bank interface of every group is what wires each child port.
static llvm::SmallVector<const iface::Memory *, 2>
ifaceMemsForArg(const iface::ModuleInterface &mi, int arg) {
  llvm::SmallVector<const iface::Memory *, 2> out;
  for (const std::vector<iface::Memory> &acc : mi.reads)
    for (const iface::Memory &m : acc)
      if (m.arg == arg)
        out.push_back(&m);
  for (const std::vector<iface::Memory> &acc : mi.writes)
    for (const iface::Memory &m : acc)
      if (m.arg == arg)
        out.push_back(&m);
  return out;
}

// The callee's scalar-input port for argument \p arg (a scalar operand the
// child consumes), or null if the arg is not a scalar input.
static const iface::Scalar *ifaceScalarForArg(const iface::ModuleInterface &mi,
                                              int arg) {
  for (const iface::Scalar &s : mi.scalars)
    if (s.arg == arg)
      return &s;
  return nullptr;
}

void DatapathBuilder::bindResource(Operation *op, RegionBlock &rb) {
  // A sub-kernel call: a CallUnit owned by this region. The child instance
  // masters its memref operands' memory ports; a scalar operand becomes a
  // Source input, a scalar result a survivor (checked in validateDatapath).
  if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op)) {
    assert(callees && "a dcp.instance in a leaf datapath needs callee context "
                      "(a rerouted container)");
    auto it = callees->ifaces.find(inv.getCallee());
    assert(it != callees->ifaces.end() &&
           "the callee interface must be registered (emitted bottom-up first)");
    const auto &mi = it->second;

    CallUnit cu;
    cu.id = dp.calls.size();
    cu.invoke = op;
    cu.region = rb.id;
    cu.callee = inv.getCallee().str();
    cu.latency = inv.getLatency();
    cu.determinacy = inv.getDeterminacy();
    cu.start = static_cast<unsigned>(dcpStart(op));

    // Operands are in callee-argument order, so operand k is callee arg k.
    // Each memref operand contributes one MemArg per child port; a boundary
    // port's top name is `<name>_<role>`, indexed per role, paired to the child
    // port by order.
    for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
      if (!isa<MemRefType>(operand.getType())) {
        // A scalar operand's driver feeds the child's scalar-input port. The
        // loop induction counter (pipeline block-arg 0) resolves to this
        // region's Counter directly; every other scalar resolves via
        // boundSource.
        const auto *sc = ifaceScalarForArg(mi, static_cast<int>(k));
        assert(sc && "a scalar operand with no matching callee scalar port");
        Source scalarSrc;
        if (auto barg = dyn_cast<BlockArgument>(operand))
          if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(
                  barg.getOwner()->getParentOp());
              pipe && barg.getArgNumber() == 0)
            scalarSrc =
                Source{Source::Kind::Counter, regionIdxOf.lookup(pipe), 0};
        if (!scalarSrc)
          scalarSrc = boundSource(operand);
        cu.scalarIns.push_back({scalarSrc, sc->name});
        continue;
      }
      auto mem = getOrCreateMem(operand);
      bool isBoundary = isa<BlockArgument>(operand);
      auto ports = ifaceMemsForArg(mi, static_cast<int>(k));
      for (const iface::Memory *m : ports) {
        CallUnit::MemArg ma;
        ma.calleeArg = static_cast<unsigned>(k);
        ma.mem = mem;
        ma.isBoundary = isBoundary;
        ma.isWrite = m->write;
        // The bank this child port serves: a cyclically partitioned arg
        // exposes one port group per bank (ifaceMemsForArg returns them all).
        // emitCalls routes an internal buffer via memBanks[mem][bank]; a
        // boundary group carries (bank, factor) for the cosim harness.
        ma.bank = static_cast<unsigned>(m->bank);
        ma.factor = static_cast<unsigned>(m->factor);
        ma.addr = m->addr;
        ma.data = m->data;
        ma.we = m->we;
        if (isBoundary) {
          // One boundary port group per accessor: a running index per
          // (argument, role) numbers the groups, concurrent and un-muxed,
          // matching the structural top's own numbering.
          auto key = std::to_string(mem) + (m->write ? "w" : "r");
          ma.topBase =
              memBoundaryPortBase(dp, mem, m->write, boundaryBaseSeq[key]++);
        }
        cu.memArgs.push_back(std::move(ma));
      }
    }
    // A scalar result is a Source::Call this region yields: register
    // producerOf so recordRegionResult latches it as a survivor at
    // start+latency, and record the child's result-output port for emitCalls.
    for (const iface::Result &r : mi.results)
      cu.resultPorts.push_back(r.name);
    if (inv.getNumResults() >= 1)
      producerOf[op] = Source{Source::Kind::Call, cu.id, 0};

    rb.callUnits.push_back(cu.id);
    dp.calls.push_back(std::move(cu));
    return;
  }
  // A nested region op (a loop wrapper, or a dcp.select guard) is a child
  // region, walked in its own iteration; it binds no resource here.
  if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp, dcp::DCPathSelectOp>(
          op))
    return;

  // A stream access binds to a StreamChannel (a handshaked FIFO); a get
  // produces a token, a put consumes one (its data driver is resolved in
  // deriveInterconnect, like a store's).
  if (auto get = dyn_cast<StreamGetOp>(op)) {
    auto sid = getOrCreateStream(get.getStream(), /*isInput=*/true);
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
    auto sid = getOrCreateStream(put.getStream(), /*isInput=*/false);
    StreamChannel::Access acc;
    acc.op = op;
    acc.isPut = true;
    acc.region = rb.id;
    acc.stage = static_cast<unsigned>(dcpStart(op));
    dp.streams[sid].accesses.push_back(acc);
    return;
  }

  // A memory access binds to a MemUnit port; a read produces a value.
  if (auto mr = dcpMemref(op)) {
    bool isWrite = isa<dcp::DCPathStoreOp>(op);
    auto mid = getOrCreateMem(mr);
    // A read-only ROM (a `memref.get_global` constant table) lowers to a
    // combinational `hw.aggregate_constant`, which has no write path; a store
    // bound to it would be silently ignored at emit (its data never lands).
    assert(
        !(isWrite && dp.mems[mid].isRom) &&
        "store to a read-only ROM (memref.get_global); the constant table has "
        "no write port");
    // The MemUnit's device-resolved latency IS the latency the scheduler
    // stamped on this access; a mismatch would build a port timed against a
    // cycle the consumer's register depth was not solved for. Both sides read
    // the same device table, so this is an equality, not an approximation.
    assert(dcpLatency(op) == (isWrite ? dp.mems[mid].writeLatency
                                      : dp.mems[mid].readLatency) &&
           "scheduled access latency disagrees with the device memory model");
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
    FuncUnit u;
    u.id = dp.units.size();
    if (auto ck = comp.getCombKind()) {
      // Combinational: emitted inline as a `comb` primitive (latency 0).
      u.opType = stringifyCombOpKindEnum(*ck).str();
      u.comb = true;
      u.latency = 0;
      u.pipelined = true;
    } else {
      // IP: `op_type` is the operator's sym_name = the RTL module name; its
      // timing + stall contract are stamped onto the compute at emit
      // (`stampOperatorTiming`).
      u.impl = comp.getOpTypeAttr().getValue().str();
      u.opType = u.impl;
      u.latency = dcpLatency(op);
      u.pipelined = comp->getAttrOfType<BoolAttr>("pipelined").getValue();
      u.stall = comp->getAttrOfType<StallContractEnumAttr>("stall").getValue();
    }
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
  // recorded in `carryInfo` (recordCarryInfo) instead, since its terminator is
  // a dcp.condition whose leading operand is the condition, not a result.
  if (rb.conditional)
    return;
  // A guard (dcp.select) yields from child regions, which are added AFTER this
  // (pre-order), so `regionIdxOf` is not yet complete here; its results are
  // resolved in recordGuards, which runs once every region exists.
  if (isa<dcp::DCPathSelectOp>(regionOp))
    return;
  auto *term = regionBody(regionOp)->getTerminator();
  if (term->getNumOperands() == 0)
    return; // a result-less region yields nothing to a sibling
  // Each `uncondition` operand is a distinct survivor, indexed by result
  // number. An untracked result (a bare iter-arg or unmodelled producer)
  // leaves a None placeholder so the vector stays aligned with that numbering.
  SmallVector<Source> results;
  for (Value res : term->getOperands()) {
    auto *def = res.getDefiningOp();
    auto it = def ? producerOf.find(def) : producerOf.end();
    results.push_back(it != producerOf.end() ? it->second : Source{});
  }
  dp.regionResult[rb.id] = std::move(results);

  // A counted loop yields one result per iter-arg (`uncondition` operands
  // align 1:1 with `inits`), so record each result's init: the emitter
  // preloads it into the survivor so a zero-trip run yields the identity, not a
  // stale accumulator; a sequential region has no iter-args, so none are
  // recorded.
  if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
    SmallVector<Source> inits;
    for (Value init : pipe.getInits())
      inits.push_back(initSource(init));
    dp.regionResultInit[rb.id] = std::move(inits);
  }
}

void DatapathBuilder::recordCarryInfo(ArrayRef<Operation *> regionOps) {
  for (Operation *op : regionOps) {
    auto pipe = dyn_cast<dcp::DCPathPipelineOp>(op);
    if (!pipe || pipe.getInits().empty())
      continue;
    RegionBlock &rb = dp.regions[regionIdxOf.lookup(op)];
    // A leaf counted reduction carries its iter-arg in a fused accumulator
    // (captureCountedResults); only a container (children run per outer
    // iteration) or a while needs iter-args latched into survivor registers.
    if (!rb.conditional && !rb.container)
      continue;
    Datapath::CarryInfo wi;
    // A while's continue condition is a scheduled compute producer (cmpi/cmpf,
    // landing in `producerOf` as a Source::Unit): solved in-body for a leaf
    // while, reified to a start-0 compute for a sequential wrapper; a counted
    // container has no condition.
    if (rb.conditional) {
      auto *cdef = pipe.getConditionValue().getDefiningOp();
      wi.condition = cdef ? producerOf.lookup(cdef) : Source{};
    }
    // Per loop-carried value: its init (loaded at start) and its next-value
    // producer (advanced into the survivor when an outer iteration drains,
    // either a nested region's result or an in-region compute). An untracked
    // next leaves a None placeholder, asserting only if a sibling reads it.
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
    // The predicate is the select's i1 condition operand, resolved to a
    // Source: a prologue region's survivor (data-dependent guard) or the
    // enclosing container's combinational predicate unit (an affine guard
    // reified to a start-0 compute).
    RegionId rid = regionIdxOf.lookup(op);
    Datapath::GuardInfo gi;
    gi.condition = boundSource(sel.getCondition());
    dp.guardCond[rid] = gi;

    // A result-mux guard resolves each branch's yielded Sources via
    // initSource (a child-region survivor or pass-through iter-arg): then
    // values go in `regionResult`, else in `selectElseResult`, muxed by the
    // predicate in emitGuard.
    auto resolveBranch = [&](Region &br) {
      SmallVector<Source> rs;
      if (!br.empty())
        for (Value v : br.front().getTerminator()->getOperands())
          rs.push_back(initSource(v));
      return rs;
    };
    SmallVector<Source> thenR = resolveBranch(sel.getThenRegion());
    if (!thenR.empty()) {
      dp.regionResult[rid] = std::move(thenR);
      dp.selectElseResult[rid] = resolveBranch(sel.getElseRegion());
      assert(dp.selectElseResult[rid].size() == dp.regionResult[rid].size() &&
             "a result-yielding dcp.select needs an else arm of equal arity");
    }
  }
}

Source DatapathBuilder::boundSource(Value v) {
  if (auto *def = v.getDefiningOp()) {
    // A prologue region result: the runtime bound is one of its survivors
    // (result number selects which), the same channel a data survivor crosses.
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
            dcp::DCPathSelectOp>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
    if (auto it = producerOf.find(def); it != producerOf.end())
      return it->second; // a hoisted producer (e.g. a constant bound)
  }
  // An enclosing loop's induction counter (arg 0 of a `dcp.pipeline`), e.g.
  // the `i` in `for ii in range(i, i_max)`: resolves to that region's counter
  // register, held stable for the nested run, the same channel a body address
  // reads it through (Source::Counter).
  if (auto barg = dyn_cast<BlockArgument>(v))
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
        pipe && barg.getArgNumber() == 0)
      return Source{Source::Kind::Counter, regionIdxOf.lookup(pipe), 0};
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second; // a scalar-argument bound
  return {};
}

void DatapathBuilder::recordRegionBounds(ArrayRef<Operation *> regionOps) {
  // A runtime induction bound (ub / lb / step) resolves to the same F->G
  // channel a data survivor crosses (a prologue survivor or a scalar IO).
  auto recordBound = [&](Value b, Source &into) {
    if (!b)
      return;
    into = boundSource(b);
    assert(into && "runtime induction bound with no resolvable Source");
  };
  for (Operation *op : regionOps)
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(op)) {
      RegionBlock &rb = dp.regions[regionIdxOf.lookup(op)];
      recordBound(pipe.getDynamicBound(), rb.ubSource);
      recordBound(pipe.getLbBound(), rb.lbSource);
      recordBound(pipe.getStepBound(), rb.stepSource);
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
  // consumer's cycle, minus the ready cycle (`readyCycleOf`, shared with the
  // emitter).
  auto edge = [&](Source base, Value key, unsigned ready,
                  unsigned distance) -> Resolved {
    int64_t depth =
        static_cast<int64_t>(distance) * ii + tY - static_cast<int64_t>(ready);
    // A negative depth means the schedule reads this operand before its
    // producer exists; report rather than assert, since the `unsigned` cast
    // would wrap to ~4e9 registers. Depth 0 keeps the build bounded;
    // `validateDatapath` fails on `infeasible` before emission.
    if (depth < 0) {
      consumer->emitError("allo-datapath-to-hw: infeasible schedule -- operand "
                          "is not ready until cycle ")
          << (static_cast<int64_t>(ready) - static_cast<int64_t>(distance) * ii)
          << " but its consumer is scheduled at cycle " << tY
          << " (producer ready " << ready << ", dependence distance "
          << distance << ", II " << ii << ")";
      dp.infeasible = true;
      depth = 0;
    }
    return {base, key, static_cast<unsigned>(depth), true};
  };

  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (auto it = ioOf.find(v); it != ioOf.end())
      return {it->second, Value(), 0, true};
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp())) {
      // The iteration counter produces the index at cycle 0, and belongs to
      // the region that *owns* this block argument, which for an enclosing
      // loop's index consumed inside a nested region is the outer region, not
      // the consumer's.
      if (barg.getArgNumber() == 0)
        return edge(Source{Source::Kind::Counter, regionIdxOf.lookup(pipe), 0},
                    v, /*ready=*/0, /*distance=*/0);
      // An iter_arg of an *enclosing* container (this consumer is nested in a
      // sequential-wrapper while): reads the container's frozen survivor
      // register directly (depth 0). The consumer's OWN iter_arg (pipe == its
      // region) is the loop recurrence, handled below.
      if (pipe != regionOp)
        return {Source{Source::Kind::Survivor, regionIdxOf.lookup(pipe),
                       barg.getArgNumber() - 1},
                Value(), 0, true};
      // A container's OWN in-body iter_arg (pipe == this region, with
      // children) is a latched survivor register (setupCarriedIterArgs), not
      // a leaf-reduction recurrence, read here at depth 0; the recurrence path
      // below is only for a childless leaf reduction.
      if (dp.regions[regionIdxOf.lookup(pipe)].container)
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
      // iteration. Ride it on the Resolved, since the recurrence register may
      // sit elsewhere in the cycle (the widened idiom reads acc through a bare
      // wire).
      auto r = edge(it->second, def->getResult(0), readyCycleOf(def), distance);
      r.init = initSource(pipe.getInits()[iterArg]);
      r.initDist = distance; // re-inject the init for the first `distance` runs
      // An unresolvable init would be silently dropped (the re-injection mux
      // is keyed on non-None), leaving the accumulator to free-run from its
      // reset value instead of the identity. Fail here as
      // HWEmitter::setupCarriedIterArgs does.
      assert(r.init && "a recurrence input has no resolvable init");
      return r;
    }
    return {};
  }

  auto *def = v.getDefiningOp();
  if (!def)
    return {};
  // A value defined by a nested region op is one of that region's results: a
  // cross-region survivor, held until the producing region completes and
  // latched by the emitter (dp.regionResult). No register chain: depth 0.
  if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp, dcp::DCPathSelectOp>(
          def))
    return {Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                   cast<OpResult>(v).getResultNumber()},
            Value(), 0, true};
  auto it = producerOf.find(def);
  if (it == producerOf.end())
    return {};
  if (it->second.kind == Source::Kind::Const)
    return {it->second, Value(), 0, true};
  if (def->getParentOp() != regionOp)
    return {}; // cross-region hand-off unsupported
  return edge(it->second, v, readyCycleOf(def), /*distance=*/0);
}

Source DatapathBuilder::initSource(Value v) {
  // A nested region's iter-arg init that reads an enclosing container's
  // iter-arg (`%argN = %outerIterArg`) resolves to that container's frozen
  // survivor register, safe to inject since it is stable for the whole nested
  // run.
  if (auto barg = dyn_cast<BlockArgument>(v))
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
        pipe && barg.getArgNumber() >= 1)
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(pipe),
                    barg.getArgNumber() - 1};
  if (auto *def = v.getDefiningOp()) {
    if (auto it = producerOf.find(def); it != producerOf.end())
      return it->second; // typically a hoisted Const (the reduction identity)
    // An init produced by a sibling region (e.g. `acc = x[i]` fused into a
    // prologue) resolves to its survivor register, the same way
    // `resolveOperand` resolves it as a data operand; the survivor stays stable
    // for the consuming run, unlike the producing Source's raw port.
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
            dcp::DCPathSelectOp>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second;
  return {}; // None: an unmodelled init (the callers reject it)
}

RegId DatapathBuilder::insertRegister(Value key, ArrayRef<unsigned> depths,
                                      Source input, RegionId region) {
  unsigned maxDepth = *llvm::max_element(depths);
  SmallVector<unsigned> taps(depths.begin(), depths.end());
  llvm::sort(taps);
  taps.erase(llvm::unique(taps), taps.end());

  Register reg;
  reg.id = dp.regs.size();
  reg.value = key;
  reg.type = key.getType();
  reg.depth = maxDepth;
  reg.input = input;
  reg.taps.assign(taps.begin(), taps.end());
  dp.regions[region].regs.push_back(reg.id);
  dp.regs.push_back(std::move(reg));
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
    if (u.boundOps.empty())
      continue; // merged-away (dead) unit: dropped from its region
    unsigned n = u.boundOps.front().first->getNumOperands();
    u.inputs.assign(n, Source{});
    u.inputInits.assign(n,
                        Source{}); // parallel; set for recurrence inputs below
    u.inputInitDist.assign(n, 1);
  }
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses) {
      SmallVector<Value> operands;
      dcpAddressing(acc.op, acc.addrMap, operands);
      acc.addr.assign(operands.size(), Source{});
    }
}

void DatapathBuilder::recordEdge(Resolved r, Source &slot, unsigned regionIdx) {
  if (!r.ok)
    return;
  if (r.depth == 0) {
    slot = r.base;
    return;
  }
  RegKey key{r.key, regionIdx};
  depthsByKey[key].push_back(r.depth);
  baseByKey[key] = r.base;
  pending.push_back({&slot, key, r.depth});
}

void DatapathBuilder::resolveUnitInputs() {
  for (FuncUnit &u : dp.units) {
    if (u.boundOps.empty())
      continue;
    Operation *op0 = u.boundOps.front().first;
    unsigned ridx = regionIdxOf.lookup(op0->getParentOp());
    unsigned ii = dp.regions[ridx].ii.value_or(1);
    unsigned nPorts = op0->getNumOperands();
    if (u.boundOps.size() == 1) {
      for (unsigned k = 0; k < nPorts; ++k) {
        auto r = resolveOperand(op0->getOperand(k), op0, ii);
        recordEdge(r, u.inputs[k], ridx);
        u.inputInits[k] = r.init; // None unless k reads a loop-carried iter_arg
        u.inputInitDist[k] = r.initDist;
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
        auto r = resolveOperand(opj->getOperand(k), opj, ii);
        assert(r.init.kind == Source::Kind::None &&
               "sharing a recurrence (reduction) unit is not modelled");
        mb.ops.push_back(opj);
        recordEdge(r, mb.sources[j], ridx);
      }
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
        recordEdge(resolveOperand(operands[k], acc.op, ii), acc.addr[k], ridx);
      if (acc.isWrite)
        recordEdge(resolveOperand(cast<dcp::DCPathStoreOp>(acc.op).getValue(),
                                  acc.op, ii),
                   acc.data, ridx);
    }

  // A stream put's data driver resolves through the same reg-depth path as a
  // store's; a predicated get/put's i1 predicate is likewise delayed to the
  // access stage, so it gates the handshake in emitStreamAccesses.
  for (StreamChannel &s : dp.streams)
    for (StreamChannel::Access &acc : s.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      if (acc.isPut) {
        auto token = cast<StreamPutOp>(acc.op).getValue();
        auto r = resolveOperand(token, acc.op, ii);
        // AXI-S data stability: a STAGE>=1 put's valid pulse persists into the
        // drain under back-pressure, so a transient din (tracking a live
        // counter/read) could commit stale data; bump the put's schedule stage
        // by one to route it through an extra frozen register instead (a
        // stage-0 put's valid freezes atomically with its din, so needs no
        // fix).
        if (r.ok && r.depth == 0 && dcpStart(acc.op) >= 1 &&
            isTransientDin(token)) {
          acc.op->setAttr(
              "start",
              IntegerAttr::get(
                  cast<IntegerAttr>(acc.op->getAttr("start")).getType(),
                  dcpStart(acc.op) + 1));
          acc.stage = static_cast<unsigned>(dcpStart(acc.op));
          r = resolveOperand(token, acc.op, ii);
        }
        recordEdge(r, acc.data, ridx);
      }
      auto pred = isa<StreamGetOp>(acc.op)
                      ? cast<StreamGetOp>(acc.op).getPred()
                      : cast<StreamPutOp>(acc.op).getPred();
      if (pred) {
        // Unlike `acc.data` (a None Source trips an assert), a None
        // `acc.when` reads as "unconditional" everywhere in the emitter, so a
        // failed-to-resolve predicate would silently turn a masked get/put into
        // an every-cycle one (wrong token stream or deadlock).
        auto pr = resolveOperand(pred, acc.op, ii);
        assert(pr.ok &&
               "predicated stream access: predicate did not resolve (a "
               "None `when` emits an unconditional access)");
        recordEdge(pr, acc.when, ridx);
      }
    }
}

void DatapathBuilder::insertRegisters() {
  // One register per (value, region) key, keyed by its RegId, to patch the
  // pending slots that read it (each in the same region the register lives in).
  llvm::DenseMap<RegKey, RegId> keyToReg;
  for (auto &kv : depthsByKey)
    keyToReg[kv.first] = insertRegister(kv.first.first, kv.second,
                                        baseByKey[kv.first], kv.first.second);

  for (const RegDepth &p : pending)
    *p.slot = Source{Source::Kind::Reg, keyToReg[p.key], p.depth};

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

// Composition predecessors of each top-level region (`rb.predecessors`): the
// earlier top-level siblings it must start after. Two signals, both attributed
// to the top-level ancestor: (1) a shared memref: any two regions touching
// the same `MemUnit` are ordered (a RAW/WAR/WAW hazard, or, for two readers, a
// read-port conflict; functional units never conflict across regions under
// per-region binding, so shared *memory* is the only cross-region resource);
// (2) a cross-region SSA edge: an op in a later region uses a value produced
// in an earlier one (a scalar survivor handed between siblings). The emitter
// starts a predecessor-free region concurrently with the kernel `start` and
// gates the rest on their producers' joined `done`.
void DatapathBuilder::recordSiblingDeps(ArrayRef<Operation *> regionOps) {
  // Top-level ancestor of a region (walk the container chain to the root).
  auto topOf = [&](RegionId r) {
    while (dp.regions[r].parent)
      r = *dp.regions[r].parent;
    return r;
  };

  // Every op inside a top-level region maps to that region's id (a nested
  // child region folds into the enclosing top-level id, since deps track at
  // top-level granularity). A value defined outside any region has no entry and
  // is skipped by the SSA scan.
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

  // (1) Shared-memref order: each region depends on the previous top-level
  // region touching that memref (edges chain transitively). A CallUnit
  // masters memref operands without a MemUnit::Access, so it counts as a sharer
  // too, or a child could read a buffer concurrently with an earlier writer.
  for (const MemUnit &m : dp.mems) {
    SmallVector<RegionId, 4> tops;
    auto addSharer = [&](RegionId region) {
      RegionId t = topOf(region);
      if (!llvm::is_contained(tops, t))
        tops.push_back(t);
    };
    for (const MemUnit::Access &a : m.accesses)
      addSharer(a.region);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id)
          addSharer(cu.region);
    llvm::sort(tops);
    for (unsigned j = 1; j < tops.size(); ++j)
      addPred(tops[j - 1], tops[j]);
  }

  // (2) Cross-region SSA edges: an op in one top-level region uses a value
  // produced in an earlier one (a scalar survivor). SSA dominance guarantees
  // the producer precedes the consumer in program order.
  func.walk([&](Operation *o) {
    auto uit = opTop.find(o);
    if (uit == opTop.end())
      return;
    RegionId consumer = uit->second;
    for (Value v : o->getOperands())
      if (auto *def = v.getDefiningOp()) {
        auto dit = opTop.find(def);
        if (dit != opTop.end() && dit->second != consumer)
          addPred(dit->second, consumer);
      }
  });
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

  // Scalar-argument IO ports first: bindResource resolves an invoke's scalar
  // operand via boundSource, which reads `ioOf`, so it must be populated
  // before the region walk, since a scalar func arg passed straight to a child
  // is an IO source.
  bindIOArgs();

  for (unsigned ridx = 0, e = regionOps.size(); ridx < e; ++ridx) {
    Operation *regionOp = regionOps[ridx];
    auto rb = addRegion(regionOp, ridx);
    for (Operation &opRef : regionBody(regionOp)->without_terminator())
      bindResource(&opRef, rb);
    // A dual guard (dcp.select with a non-empty else) binds its else-branch
    // loose ops too, since regionBody returns only the then block. Nested
    // regions in either branch bind nothing here (walked in their own
    // iteration).
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
      if (!sel.getElseRegion().empty())
        for (Operation &opRef :
             sel.getElseRegion().front().without_terminator())
          bindResource(&opRef, rb);
    recordRegionResult(rb, regionOp);
    dp.regions.push_back(std::move(rb));
  }

  recordRegionBounds(
      regionOps); // dynamic-trip bounds (needs ioOf + regionIdxOf)
  recordCarryInfo(
      regionOps);          // container / while iter-arg recurrence (needs ioOf)
  recordGuards(regionOps); // guard (dcp.select) predicate Sources
  recordResults(); // scalar func-result output ports (needs ioOf + regionIdxOf)
  applyBinding(policy.plan(dp)); // trivial => no groups, no muxes
  deriveInterconnect();
  recordSiblingDeps(regionOps); // top-level composition DAG (concurrency gates)
  verifyBinding(dp); // MRT legality: no unit shared by conflicting ops
}

} // namespace mlir::allo::uarch
