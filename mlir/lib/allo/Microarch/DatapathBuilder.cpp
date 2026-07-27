/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h" // iface::ModuleInterface (CallUnit ports)
#include "allo/Microarch/Reservation.h" // verifyBinding (MRT legality)

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h"   // characterize (storage shape)
#include "allo/Support/Logging.h"          // unmodelled-op diagnostic
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

// The STORAGE IDENTITY of a memref or stream SSA value: the one definition
// every access to that buffer / channel must agree on.
//
// A buffer allocated inside a region and used by a later one cannot be named
// directly (SSA dominance), so the reifier threads it out through the region's
// terminator: the producer stores to `%alloc` while the consumer loads from the
// REGION RESULT forwarding it. Those are different Values, so keying storage by
// the operand as written builds ONE MEMORY PER REGION: the writes land in one
// and the reads come back from the other (uninitialized), and because the two
// halves are then distinct MemUnits `recordSiblingDeps` sees no shared memref
// and adds no ordering edge either. Both failures are silent.
//
// So peel every DCP region result back to the value it forwards, and a
// pipeline's iter-arg back to its init, until reaching the real definition.
static Value storageRoot(Value memref) {
  while (true) {
    if (auto res = dyn_cast<OpResult>(memref)) {
      Operation *owner = res.getOwner();
      unsigned k = res.getResultNumber();
      if (auto seq = dyn_cast<dcp::DCPathSequentialOp>(owner)) {
        memref = seq.getBody().front().getTerminator()->getOperand(k);
        continue;
      }
      if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(owner)) {
        // Terminator-kind agnostic: `uncondition` operands for a counted loop,
        // `condition`'s carried operands for a while (whose leading `i1` would
        // otherwise shift the indexing by one).
        memref = pipe.getCarriedValues()[k];
        continue;
      }
      // A guard yields from two arms, so a value crossing one has no single
      // definition to peel to. No frontend shape produces that; fail loudly
      // rather than silently splitting the buffer as above.
      assert(!isa<dcp::DCPathSelectOp>(owner) &&
             "a memref/stream yielded from a dcp.select has no single storage "
             "root");
      return memref;
    }
    // A pipeline iter-arg (block argument 0 is the counter) forwards its init.
    auto barg = dyn_cast<BlockArgument>(memref);
    if (!barg)
      return memref;
    auto pipe = dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
    if (!pipe || barg.getArgNumber() == 0)
      return memref; // a func argument, or the counter: already a root
    memref = pipe.getInits()[barg.getArgNumber() - 1];
  }
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
  memref = storageRoot(memref);
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
  auto mc = allo::characterize(memref, memLib.defaultImpl);
  // An initialized global the kernel stores to needs a real write port, so it
  // is a ROM only if nothing writes it. `isRom` is `MemoryChar::constantTable`,
  // the same predicate the scheduler's port model bills against.
  if (auto init = allo::globalInitOf(memref)) {
    m.romInit = *init;
    m.isRom = mc.constantTable;
  }
  // The one `allo.part` decode for this memref. `characterize` derives its bank
  // count from the same layout, so the assert guards drift between the two.
  // Anything still banked here is data-dependent or a partitioned argument.
  m.layout = allo::bankLayoutOf(memref);
  m.numBanks = m.layout.numBanks;
  assert(m.numBanks == std::max(1u, mc.numBanks) &&
         "the emitter's bank decomposition disagrees with the one the "
         "scheduler's port model was billed against");
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
  // threaded out of a region names different Values at its two ends, and split
  // in two a self-loop reads as two one-directional channels.
  stream = storageRoot(stream);
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
    // else body is an else-child, otherwise a then-child, found by walking up
    // to the child whose parent is the select.
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
    if (auto t = pipe.getTripAttr())
      rb.tripCount = t.getInt();
    // The induction bounds themselves are resolved by `recordRegionBounds`,
    // once the counter width is known and `resolveValue` can see the whole
    // region model.
  } else {
    assert(isa<dcp::DCPathSequentialOp>(regionOp) &&
           "a RegionBlock is a dcp pipeline / sequential / select");
    rb.kind = RegionBlock::Kind::Acyclic;
  }

  // Declared composition class + single-run latency, so the composer dispatches
  // on a declared property rather than re-deriving region shape. A
  // `counted_static` guard may carry no `latency` and hands off by handshake.
  if (auto d = regionOp->getAttrOfType<DeterminacyEnumAttr>("determinacy"))
    rb.determinacy = d.getValue();
  // A CONCURRENT region's `latency` is not a single-run span: its children run
  // to their own completion, ordered by back-pressure, so the number is a floor
  // and not a hand-off contract. Recording it would break the invariant below.
  if (auto lat = regionOp->getAttrOfType<IntegerAttr>("latency"))
    if (rb.determinacy != DeterminacyEnum::Concurrent)
      rb.staticLatency = lat.getInt();
  assert((!rb.staticLatency.has_value() ||
          rb.determinacy == DeterminacyEnum::CountedStatic) &&
         "a region with a static latency must be declared counted_static");
  return rb;
}

// The ONE derivation of the controller discriminant's structural axis. The
// emitter's dispatch and the validator's legality rules both read `rb.shape`;
// neither re-derives it from `children` / `guard` / `callUnits`. Order matters:
// a guard is a guard whichever arms it has, a region with child regions
// sequences them, and only then does a lone-call counted loop become the
// instance-substrate hand-off.
void DatapathBuilder::deriveShapes() {
  for (RegionBlock &rb : dp.regions) {
    if (rb.guard)
      rb.shape = RegionBlock::Shape::Guard;
    else if (!rb.children.empty())
      rb.shape = RegionBlock::Shape::Container;
    else if (rb.kind == RegionBlock::Kind::Cyclic && !rb.callUnits.empty())
      rb.shape = RegionBlock::Shape::CallNode;
    else
      rb.shape = RegionBlock::Shape::Leaf;

    // The invariants each shape carries, stated where the shape is decided
    // rather than at the consumer that would otherwise trip over them.
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
  // memref operand contributes one MemArg per child port; a boundary port's top
  // name is `<name>_<role>`, indexed per role and paired by order.
  for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
    if (isa<StreamType>(operand.getType())) {
      // A channel END: the child handshakes on three ports of its own, and the
      // channel records which call/arg pair they belong to so the realization
      // (one FIFO per consumer) can wire them without scanning the calls.
      const iface::FIFO *f = mi.streamForArg(static_cast<int>(k));
      assert(f && "a stream operand with no matching callee stream port");
      StreamId sid = getOrCreateStream(operand, f->isInput);
      dp.streams[sid].callEnds.push_back(
          {cu.id, static_cast<unsigned>(cu.streamArgs.size())});
      // Buffering is a throughput hint and deeper is always KPN-safe, so a
      // channel takes the deepest request among its container-side type and
      // every end.
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
      cu.scalarIns.push_back({Source{}, sc->name});
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
      // one port group per bank. emitCalls routes an internal buffer via
      // memBanks[mem][bank]; a boundary group carries (bank, factor) for cosim.
      ma.bank = static_cast<unsigned>(m->bank);
      ma.factor = static_cast<unsigned>(m->factor);
      ma.addr = m->addr;
      ma.data = m->data;
      ma.we = m->we;
      // `ma.topBase` (the boundary port group) is assigned by
      // enumerateBoundaryPorts, once every access is bound.
      cu.memArgs.push_back(std::move(ma));
    }
  }
  // Each scalar result is a Source::Call this region yields: register
  // producerOf per result so recordRegionResults latches each as its own
  // survivor, and record the child's result-output ports for emitCalls.
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
  // A ROM has no write path, so a store bound to it would be silently
  // dropped. `getOrCreateMem` clears `isRom` for any memref a `dcp.store`
  // names, so this fires only if that scan and this binding disagree.
  assert(!(isWrite && dp.mems[mid].isRom) &&
         "store bound to a memory classified read-only");
  // The MemUnit's device-resolved latency IS the latency the scheduler stamped
  // on this access; a mismatch would build a port timed against a cycle the
  // consumer's register depth was not solved for. Both read the same table.
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
    u.latency = dcpLatency(comp);
    u.pipelined = comp->getAttrOfType<BoolAttr>("pipelined").getValue();
    u.stall = comp->getAttrOfType<StallContractEnumAttr>("stall").getValue();
  }
  u.resultType = comp.getResult().getType();
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
  // One arm per kind of resource a body op binds, plus the kinds that bind
  // nothing. Falling out the bottom is the loud case: an unmodelled op would
  // otherwise be dropped from the hardware while compilation reported success.
  if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op))
    // A sub-kernel call: a CallUnit owned by this region. The child instance
    // masters its memref operands' memory ports; a scalar operand becomes a
    // Source input, a scalar result a survivor.
    return bindCall(inv, rb);
  if (isa<StreamGetOp, StreamPutOp>(op))
    return bindStream(op, rb); // a handshaked FIFO access
  if (auto mr = dcpMemref(op))
    return bindMemory(op, mr, rb); // a MemUnit port
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return bindCompute(comp, rb); // a FuncUnit

  // A nested region op (a loop wrapper, or a dcp.select guard) is a child
  // region, walked in its own iteration.
  if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp, dcp::DCPathSelectOp>(
          op))
    return;
  // Literals are pre-registered as ConstCells (see collectConstants).
  if (isa<arith::ConstantOp>(op))
    return;
  // A declaration binds no resource: the memref / stream it defines is
  // materialized on first access. The reifier keeps these verbatim in a region
  // body, so an `alloc` reaches here whenever an imperfect nest decomposes.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
          StreamCreateOp>(op))
    return;

  unsupported(Stage::Emit, op)
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
      // A while's continue condition is a scheduled compute producer: solved
      // in-body for a leaf while, reified to a start-0 compute over the
      // iter-arg survivors for a sequential wrapper. A counted loop has none.
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
  // The MemIds a call touches, by role. Two calls share an array iff they are
  // passed the same one, which is exactly MemId identity (`getOrCreateMem` keys
  // on the storage root).
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
        // A CONCURRENT container places every child at 0, so the hazard
        // DIRECTION (RAW / WAW / WAR) is the whole ordering. A SCHEDULED one
        // orders by `start`: gate an earlier or indeterminate neighbour only.
        bool hazard =
            concurrent
                ? (shares(memsOf(p, true), memsOf(cu, std::nullopt)) ||
                   shares(memsOf(cu, true), memsOf(p, false)))
                : (p.start < cu.start || !p.latency) &&
                      shares(memsOf(p, std::nullopt), memsOf(cu, std::nullopt));
        if (hazard)
          add(p.id, /*viaResult=*/false);
      }
      // A child consuming an earlier child's scalar RESULT is ordered after it
      // too, and this edge is never inert: the result port only holds from the
      // producer's `done`.
      for (const CallUnit::ScalarArg &sa : cu.scalarIns)
        if (sa.src.kind == Source::Kind::Call)
          add(sa.src.id, /*viaResult=*/true);
    }
  }
}

void DatapathBuilder::enumerateBoundaryPorts() {
  // ONE numbering for every boundary memory port: each external access is a
  // port group with an INDEX (into read/writePorts) and a NAME, off a counter
  // per (memory, role) that call-mastered ports continue after the parent's.
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
    for (auto [a, acc] : llvm::enumerate(m.accesses)) {
      auto &ports = acc.isWrite ? dp.writePorts : dp.readPorts;
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
    // latched it into, and the ONLY channel a value leaves a region by. A loop
    // bound, a carried next-value and a data operand all cross it.
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
            dcp::DCPathSelectOp>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
    return {}; // an unmodelled producer
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second; // a scalar function argument
  // A `dcp.pipeline` block argument. Arg 0 is the induction counter, e.g. the
  // `i` in `for ii in range(i, i_max)`: its region's counter register, held
  // stable for the whole of a nested run.
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

// The counter width of every cyclic region: i32, except a loop-over-call, whose
// counter drives the callee's index port directly and so must be built at that
// port's width. Recorded here rather than dug back out at emission, where the
// only place that can adapt it is a controller rebuilding the terminator it was
// just handed.
void DatapathBuilder::deriveCounterTypes() {
  Type i32 = IntegerType::get(func.getContext(), 32);
  for (RegionBlock &rb : dp.regions) {
    if (rb.kind != RegionBlock::Kind::Cyclic)
      continue;
    rb.counterType = i32;
    if (rb.shape != RegionBlock::Shape::CallNode)
      continue;
    assert(callees && "a loop-over-call needs callee context");
    const CallUnit &cu = dp.calls[rb.callUnits.front()];
    auto it = callees->ifaces.find(cu.callee);
    assert(it != callees->ifaces.end() && "the loop child must be registered");
    const iface::ModuleInterface &mi = it->second;
    // The IV operand is the scalar whose driver is this region's counter.
    Type ivType;
    for (const CallUnit::ScalarArg &sa : cu.scalarIns)
      if (sa.src.kind == Source::Kind::Counter && sa.src.id == rb.id)
        for (const iface::Scalar &s : mi.scalars)
          if (s.name == sa.port)
            ivType = IntegerType::get(func.getContext(), s.width);
    assert(ivType &&
           "a loop-over-call region has no induction-variable child port");
    rb.counterType = ivType;
  }
}

void DatapathBuilder::recordRegionBounds(ArrayRef<Operation *> regionOps) {
  // A runtime induction bound (ub / lb / step) crosses the same F->G channel a
  // data survivor does; an unresolvable one is reported, not silently run.
  auto recordBound = [&](Operation *pipe, Value b, Source &into) {
    if (!b)
      return;
    into = resolveValue(b);
    if (!into) {
      unsupported(Stage::Emit, pipe)
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
    // A stream arg is a FIFO channel (a StreamChannel, created lazily by
    // bindResource on its first get/put), not a scalar port.
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
  auto ret = cast<func::ReturnOp>(func.front().getTerminator());
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    assert(!isa<MemRefType>(v.getType()) &&
           "a memref result should be an out-param by emit "
           "(buffer-results-to-out-params)");
    Result r;
    // An unresolvable result Source is swept by `validateDatapath` (the same
    // cross-region hand-off error every other Source consumer reports), so the
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
      error(Stage::Emit, consumer)
          << "Infeasible schedule; the operand is not ready until cycle "
          << (static_cast<int64_t>(ready) - static_cast<int64_t>(distance) * ii)
          << " but its consumer is scheduled at cycle " << tY
          << " (producer ready " << ready << ", dependence distance "
          << distance << ", II " << ii << ")";
      dp.infeasible = true;
      depth = 0;
    }
    return {base, key, static_cast<unsigned>(depth), true};
  };

  // The one operand that does not read `v` at all: an unlatched iter_arg of the
  // consumer's OWN region is the loop RECURRENCE, so the edge runs back to the
  // previous iteration's producer, `distance` iterations away.
  if (auto barg = dyn_cast<BlockArgument>(v))
    if (auto pipe =
            dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
        pipe == regionOp && barg.getArgNumber() >= 1 &&
        !dp.regions[regionIdxOf.lookup(pipe)].container) {
      unsigned iterArg = barg.getArgNumber() - 1;
      auto [def, distance] = traceIterArgSource(pipe, iterArg);
      if (!def || def->getParentOp() != regionOp)
        return {};
      auto it = producerOf.find(def->getResult(0));
      if (it == producerOf.end())
        return {};
      // The emitter re-injects the iter_arg's init (reduction identity) on THIS
      // consumer input at the first iteration. It travels on the Resolved,
      // since the recurrence register may sit elsewhere in the cycle.
      auto r = edge(it->second, def->getResult(0), readyCycleOf(def), distance);
      r.init = resolveValue(pipe.getInits()[iterArg]);
      r.initDist = distance; // re-inject the init for the first `distance` runs
      // An unresolvable init is dropped by the non-None-keyed re-injection mux,
      // leaving the accumulator to free-run from reset. Only this site knows an
      // init was expected; None is normal elsewhere.
      if (!r.init) {
        unsupported(Stage::Emit, def)
            << "Loop-carried accumulator has an initial value this region "
               "cannot read; such a cross-region value hand-off is not "
               "lowered yet";
        dp.infeasible = true;
      }
      return r;
    }

  Source base = resolveValue(v);
  if (!base)
    return {};
  switch (base.kind) {
  // A held source is already valid when the consumer issues, so it ties
  // straight in: a survivor for the whole of the producing region's run, an
  // IO port and a literal for the whole kernel.
  case Source::Kind::Survivor:
  case Source::Kind::IO:
  case Source::Kind::Const:
    return {base, Value(), 0, true};
  // The counter presents its index at cycle 0 of ITS region (for an enclosing
  // loop's index, held across the whole nested run), so a consumer scheduled
  // at tY delays it that far.
  case Source::Kind::Counter:
    return edge(base, v, /*ready=*/0, /*distance=*/0);
  default:
    break;
  }
  // A scheduled producer: readable only from the region it issues in, and only
  // after it lands.
  Operation *def = v.getDefiningOp();
  assert(def && "a scheduled Source is produced by an op");
  if (def->getParentOp() != regionOp)
    return {}; // cross-region hand-off unsupported
  return edge(base, v, readyCycleOf(def), /*distance=*/0);
}

RegId DatapathBuilder::insertRegister(Value key, ArrayRef<unsigned> depths,
                                      Source input, RegionId region) {
  Register reg;
  reg.id = dp.regs.size();
  reg.value = key;
  reg.type = key.getType();
  // The chain is as long as its deepest consumer needs; the shallower ones read
  // their own tap off it (Source::Reg's `outPort`).
  reg.depth = *llvm::max_element(depths);
  reg.input = input;
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
    if (u.boundOps.empty())
      continue; // merged-away (dead) unit: dropped from its region
    unsigned n = u.repOp()->getNumOperands();
    u.inputs.assign(n, Source{});
    u.inputInits.assign(n,
                        Source{}); // parallel; set for recurrence inputs below
    u.inputInitDist.assign(n, 1);
  }
  for (MemUnit &m : dp.mems) {
    auto shape = cast<MemRefType>(m.memref.getType()).getShape();
    for (MemUnit::Access &acc : m.accesses) {
      SmallVector<Value> operands;
      dcpAddressing(acc.op, acc.addrMap, operands);
      acc.addr.assign(operands.size(), Source{});
      // Which bank this access reaches, decided by the `addrMap` just read: a
      // compile-time index routes to one, a data-dependent one spans every bank
      // (empty). An unbanked memref keeps the default 0.
      if (m.numBanks > 1) {
        std::optional<int64_t> b = staticBankOf(m.layout, acc.addrMap, shape);
        acc.staticBank = b ? std::optional<unsigned>(*b) : std::nullopt;
      }
    }
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
    Operation *op0 = u.repOp();
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
        // A shared unit reaches its input through the mux below, leaving
        // nowhere to time the reduction identity's re-injection against.
        // Vacuous under trivial binding; a sharing policy reaches it.
        if (r.init.kind != Source::Kind::None) {
          unsupported(Stage::Emit, opj)
              << "Binding policy shares one operator unit between a "
                 "loop-carried reduction and another op; re-injecting the "
                 "reduction identity through the shared input mux is not "
                 "modelled. Use binding='trivial' for this kernel";
          dp.infeasible = true;
        }
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
        recordEdge(r, acc.data, ridx);
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
          unsupported(Stage::Emit, acc.op)
              << "Predicate of this stream access is produced by a value the "
                 "region cannot read; such a cross-region value hand-off is "
                 "not lowered yet, and the access would otherwise fire "
                 "unconditionally";
          dp.infeasible = true;
        }
        recordEdge(pr, acc.when, ridx);
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
                                        baseByKey[kv.first], kv.first.second);

  for (const RegDepth &p : pending)
    *p.slot = Source{Source::Kind::Reg, keyToReg[p.key], p.depth};

  // Materialize sharing muxes: the sources are final once the registers are
  // built and the pending slots patched. A port whose bound ops all read one
  // driver needs no mux.
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
        producerOf[bo.first->getResult(0)] =
            Source{Source::Kind::Unit, into, 0};
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

  // Every op inside a top-level region maps to that region's id; a nested child
  // folds into the enclosing top-level id, since deps track at that
  // granularity. A value defined outside any region has no entry.
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
  // in program order, so each depends on the previous toucher and the rest
  // follows transitively.
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

  // Scalar-argument IO ports: one of the maps `resolveValue` reads, so every
  // pass below sees a scalar func arg as an IO source.
  bindIOArgs();

  for (unsigned ridx = 0, e = regionOps.size(); ridx < e; ++ridx) {
    Operation *regionOp = regionOps[ridx];
    auto rb = addRegion(regionOp, ridx);
    for (Operation &opRef : regionBody(regionOp)->without_terminator())
      bindResource(&opRef, rb);
    // A dual guard (dcp.select with a non-empty else) binds its else-branch
    // loose ops too, since regionBody returns only the then block. Nested
    // regions in either branch are walked in their own iteration.
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
      if (!sel.getElseRegion().empty())
        for (Operation &opRef :
             sel.getElseRegion().front().without_terminator())
          bindResource(&opRef, rb);
    dp.regions.push_back(std::move(rb));
  }

  deriveShapes();           // controller discriminant (needs every child)
  enumerateBoundaryPorts(); // module boundary ports (needs every access)
  // Everything below resolves Values to Sources, and so runs here rather than
  // during the walk: `resolveValue` needs the complete region model.
  recordRegionResults(regionOps); // per-region results/recurrence + predicate
  recordCallScalars();            // each dcp.instance's scalar operand drivers
  recordCallDeps();               // composition DAG on the instance substrate
  reclassifyRoms();               // read-only is a property of the USE
  deriveCounterTypes();           // counter width (needs the call IV operand)
  recordRegionBounds(regionOps);  // induction bounds, at that width
  recordResults();                // scalar func-result output ports
  applyBinding(policy.plan(dp));  // trivial => no groups, no muxes
  deriveInterconnect();
  recordSiblingDeps(regionOps); // top-level composition DAG (concurrency gates)
  verifyBinding(dp); // MRT legality: no unit shared by conflicting ops
}

} // namespace mlir::allo::uarch
