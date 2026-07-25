/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h" // StreamCreateOp, kAlloAsyncAttr
#include "allo/Microarch/HWEmitter.h"
#include "allo/Microarch/Interface.h" // iface field-name suffixes / helpers
#include "allo/Support/Logging.h"     // user-facing check reporting

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h" // is_contained

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// A loop-over-calls container: a single sync sub-kernel call inside one
// counted `dcp.pipeline` (the reified loop) directly in the container body:
// one child instance invoked N times, driven by a loop counter, rather than
// a flat call graph. Returns the loop, or null if `func` is not this shape
// (exactly one call, one counted loop).
static dcp::DCPathPipelineOp loopOverCall(func::FuncOp func) {
  SmallVector<func::CallOp> calls;
  func.walk([&](func::CallOp c) {
    if (!c->hasAttr(kAlloAsyncAttr))
      calls.push_back(c);
  });
  if (calls.size() != 1)
    return nullptr;
  auto loop = calls.front()->getParentOfType<dcp::DCPathPipelineOp>();
  if (loop && loop->getParentOp() == func && !loop.isWhileLoop())
    return loop;
  return nullptr;
}

namespace {
using Dir = hw::ModulePort::Direction;

// A callee port's declared type + whether it is a module input.
struct PortDesc {
  Type type;
  bool isInput;
};

llvm::StringMap<PortDesc> portMap(hw::HWModuleOp mod) {
  llvm::StringMap<PortDesc> m;
  for (const hw::PortInfo &p : mod.getPortList())
    m[p.name.getValue()] = {p.type, p.dir == Dir::Input};
  return m;
}

// One spawned process instance.
struct Inst {
  func::CallOp call;
  hw::HWModuleOp mod;
  const iface::ModuleInterface *mi; // the callee's port model (arg <-> names)
  llvm::StringMap<PortDesc> ports;
  llvm::StringMap<Value> outs;   // instance output values, by port name
  int64_t startOffset = 0;       // scheduled start cycle (the call's `start`
                                 // attr, 0 for a dataflow spawn / independent)
  bool isAsync = false;          // an `allo.async` spawn -> broadcast start
  bool determinate = false;      // callee has an exact static latency: as a
                                 // producer it releases a consumer by a static
                                 // offset (Route B); indeterminate -> handshake
  SmallVector<unsigned> gateOns; // producer insts whose `done`/completion gates
                                 // this child's `start`: it reads an array an
                                 // earlier child wrote. Empty = ungated.
};

// One internal FIFO channel (single producer, single consumer).
struct Chan {
  Type payload;
  unsigned depth = 2;
  int prod = -1, cons = -1;       // producing / consuming instance index
  std::string prodBase, consBase; // stream port base on each side
  ArrayAttr init; // initial tokens (feedback seeding), null when unseeded
};

// One container-boundary port mirrored onto the top: the top-side name (derived
// from the container argument, so distinct arguments never collide) and the
// instance-side name (the callee's own port), plus type/direction/owner.
struct Mirror {
  std::string topName, calleeName;
  Type type;
  bool isInput;
  unsigned inst;
};

// Per-channel body wires. The FIFO is built after the instances (which need its
// status/data), so its status/output are backedges. A seeded channel also
// carries an init-prepend shim on the consumer read port: the first k reads
// come from the init ROM and do not pop the FIFO, so the channel history is
// [init]++[produced] and a feedback cycle turns from cycle 0. The `seq.fifo`
// itself is untouched.
struct ChanWires {
  Backedge full, empty, dataOut;
  Value notFull, notEmpty;
  // Init-prepend shim (seeded channels only; null otherwise):
  Value servingInit; // rem != 0: still serving initial tokens
  Value dataMux;     // consumer data:  servingInit ? initROM : fifo.out
  Value validShim;   // consumer valid: servingInit | ~empty
  Value rem;         // remaining initial tokens (down-counter)
  Backedge remNext;  // resolved in the FIFO step (needs consumer ready)
};

//===----------------------------------------------------------------------===//
// Builds the structural top for one concurrent container, in phases: discover
// process instances and their channels, check feedback seeding, plan the
// boundary ports, then emit the module body (channel wires, instances, FIFOs,
// fork/join).
//===----------------------------------------------------------------------===//
struct ConcurrentTopBuilder {
  func::FuncOp container;
  // Callee module + port model tables, holding leaf kernels *and* inner
  // containers already emitted this pass (a container is composed exactly like
  // a leaf, so the two are not distinguished here).
  const llvm::StringMap<hw::HWModuleOp> &modules;
  const llvm::StringMap<iface::ModuleInterface> &ifaceModels;
  // The scheduled funcs, by symbol name: a callee's source `func.func`, which
  // still carries its `dcp.latency` (the emitted `hw.module` shares its name,
  // so a symbol lookup from a callsite is ambiguous).
  const llvm::StringMap<func::FuncOp> &scheduledFuncs;
  OpBuilder &b;
  MLIRContext *ctx;
  Location loc;
  Type i1;

  SmallVector<Inst> insts;
  llvm::MapVector<Value, Chan> chans; // keyed by the stream.create result
  SmallVector<Mirror> mirrors;
  iface::ModuleInterface topIface; // the composed top's port model

  // Body-scope wires, valid only while emitting the module body.
  Value clkRaw, clk, rst, start, tru;

  ConcurrentTopBuilder(
      func::FuncOp container, const llvm::StringMap<hw::HWModuleOp> &modules,
      const llvm::StringMap<iface::ModuleInterface> &ifaceModels,
      const llvm::StringMap<func::FuncOp> &scheduledFuncs, OpBuilder &b)
      : container(container), modules(modules), ifaceModels(ifaceModels),
        scheduledFuncs(scheduledFuncs), b(b), ctx(b.getContext()),
        loc(container.getLoc()), i1(b.getI1Type()) {}

  LogicalResult run(hw::HWModuleOp &modOut, iface::ModuleInterface &ifaceOut);

  void collectInstances();
  void computeStartGates();
  LogicalResult discoverChannels();
  LogicalResult checkFeedbackSeeding();
  void planBoundaryPorts();
  void buildBody(OpBuilder &ib, hw::HWModulePortAccessor &pa);
  llvm::MapVector<Value, ChanWires> buildChannelWires(OpBuilder &ib,
                                                      BackedgeBuilder &bb);
  void instantiateProcesses(OpBuilder &ib, hw::HWModulePortAccessor &pa,
                            llvm::MapVector<Value, ChanWires> &cw);
  void wireFifos(OpBuilder &ib, llvm::MapVector<Value, ChanWires> &cw);
  void forkJoin(OpBuilder &ib, hw::HWModulePortAccessor &pa);

  Value notBit(OpBuilder &ib, Value x) {
    return comb::XorOp::create(ib, loc, x, tru, false).getResult();
  }

  // Rising-edge pulse of a held level (high the one cycle it goes 0->1). A
  // child `done` is a held level; gating a Route-A successor's `start` (a
  // 1-cycle pulse) on it needs the edge, the same hand-off sequence() uses to
  // chain a kernel's regions, lifted to `hw.instance`s.
  Value risingEdge(OpBuilder &ib, Value level) {
    Value zero = hw::ConstantOp::create(ib, loc, i1, 0);
    Value prev =
        seq::CompRegOp::create(ib, loc, level, clk, rst, zero, "done_edge");
    return comb::AndOp::create(ib, loc, level, notBit(ib, prev), false)
        .getResult();
  }

  // A start pulse delayed \p n cycles: an n-deep 1-bit shift register (each
  // stage resets to 0, so no spurious start out of reset). Used to fire a
  // sequential child's `start` at its scheduled offset (n is small, bounded
  // by a region's depth; a counter+compare would scale better for very large
  // offsets).
  Value delayPulse(OpBuilder &ib, Value sig, int64_t n) {
    Value zero = hw::ConstantOp::create(ib, loc, i1, 0);
    Value d = sig;
    for (int64_t i = 0; i < n; ++i)
      d = seq::CompRegOp::create(ib, loc, d, clk, rst, zero, "start_delay");
    return d;
  }
};

// Collect the spawned process instances, in program order.
void ConcurrentTopBuilder::collectInstances() {
  container.walk([&](func::CallOp call) {
    auto mod = modules.lookup(call.getCallee());
    auto it = ifaceModels.find(call.getCallee());
    assert(mod && it != ifaceModels.end() &&
           "spawned callee has no emitted module / interface (a nested "
           "container must be emitted before its enclosing container)");
    int64_t off = 0;
    if (auto s = call->getAttrOfType<IntegerAttr>("start"))
      off = s.getInt();
    // Determinate (Route B): a non-async callee with an exact whole-kernel
    // latency (`dcp.determinacy` == counted_static) releases the consumer at
    // a static offset instead of its real `done`; resolved via scheduledFuncs.
    bool determinate = false;
    if (!call->hasAttr(kAlloAsyncAttr)) {
      auto callee = scheduledFuncs.lookup(call.getCallee());
      auto det =
          callee ? callee->getAttrOfType<DeterminacyEnumAttr>("dcp.determinacy")
                 : DeterminacyEnumAttr();
      determinate = det && det.getValue() == DeterminacyEnum::CountedStatic;
    }
    insts.push_back({call,
                     mod,
                     &it->second,
                     portMap(mod),
                     {},
                     off,
                     call->hasAttr(kAlloAsyncAttr),
                     determinate,
                     {}});
  });
}

// A child whose access to a shared array may hazard an earlier child's is
// ordered after that producer. Record each producer per child, from the
// callees' own read/write direction (`mi->reads`/`writes`) and SSA-operand
// identity (two calls share an array iff passed the same value). The gate
// drives the start policy: a determinate producer releases the consumer by a
// static offset (Route B); an indeterminate one (async / data-dependent)
// releases it by its real `done` (Route A), since an unknown span cannot be
// expressed as an offset. Pure-dataflow children hand off through FIFOs,
// never a shared array, so they stay ungated. Producers precede consumers in
// program order, so the gates form a DAG.
//
// This is a COARSER test than the scheduler's (whole-array, no element
// ranges), so a pair the scheduler proved disjoint may still be gated. That
// costs nothing when both children are determinate (the gate is inert and
// each still fires at its own scheduled offset, concurrently), and only
// forgoes overlap against an indeterminate producer, whose completion the
// emitter cannot place statically anyway.
void ConcurrentTopBuilder::computeStartGates() {
  auto memOperands = [](Inst &in, bool write) {
    SmallVector<Value> vs;
    for (const auto &grp : write ? in.mi->writes : in.mi->reads)
      for (const iface::Memory &m : grp)
        vs.push_back(in.call.getOperand(m.arg));
    return vs;
  };
  auto shares = [](ArrayRef<Value> a, ArrayRef<Value> b) {
    return llvm::any_of(a, [&](Value v) { return llvm::is_contained(b, v); });
  };
  for (unsigned j = 0; j < insts.size(); ++j) {
    auto reads = memOperands(insts[j], /*write=*/false);
    auto writes = memOperands(insts[j], /*write=*/true);
    for (unsigned i = 0; i < j; ++i) {
      auto pReads = memOperands(insts[i], /*write=*/false);
      auto pWrites = memOperands(insts[i], /*write=*/true);
      // This child reads or writes an array the earlier one wrote (RAW / WAW),
      // or writes one it read (WAR).
      if (shares(pWrites, reads) || shares(pWrites, writes) ||
          shares(writes, pReads))
        insts[j].gateOns.push_back(i); // one edge per producer suffices
    }
  }
}

// Discover channels from each `stream.create` and its producer/consumer.
// Channels are single-producer / single-consumer. A stream read by two
// processes (SPMC broadcast) or written by two (MPSC merge) is reported as a
// user-facing error rather than an internal assert: broadcast is not
// inserted automatically (write one channel per consumer), and deterministic
// merge is not supported.
LogicalResult ConcurrentTopBuilder::discoverChannels() {
  for (unsigned ii = 0; ii < insts.size(); ++ii) {
    Inst &in = insts[ii];
    for (const iface::FIFO &f : in.mi->streams) {
      Value stream = in.call.getOperand(f.arg);
      // A stream arg is either an internal channel (a local `stream.create`)
      // or a container boundary forwarded from the enclosing container;
      // boundaries get a top stream port instead, planned in planBoundaryPorts.
      if (isa<BlockArgument>(stream))
        continue;
      assert(stream.getDefiningOp<StreamCreateOp>() &&
             "a process stream arg must be an internal channel or a container "
             "boundary");
      Chan &c = chans[stream];
      c.depth = f.depth;
      c.payload = in.ports[f.data].type;
      c.init = stream.getDefiningOp<StreamCreateOp>().getInitAttr();
      if (f.isInput) {
        if (c.cons >= 0) {
          logging::error(logging::Stage::Emit, container)
              << "Stream channel is read by more than one process ('"
              << insts[c.cons].call.getCallee() << "' and '"
              << in.call.getCallee()
              << "'); a channel is single-consumer and broadcast is not "
                 "inserted automatically; give each consumer its own channel "
                 "and fan the producer's writes out across them";
          return failure();
        }
        c.cons = ii;
        c.consBase = f.base;
      } else {
        if (c.prod >= 0) {
          logging::error(logging::Stage::Emit, container)
              << "Stream channel is written by more than one process ('"
              << insts[c.prod].call.getCallee() << "' and '"
              << in.call.getCallee()
              << "'); a channel is single-producer and deterministic merge is "
                 "not supported yet";
          return failure();
        }
        c.prod = ii;
        c.prodBase = f.base;
      }
    }
  }
  for (auto &kv : chans) {
    Chan &c = kv.second;
    if (c.prod < 0 || c.cons < 0) {
      logging::error(logging::Stage::Emit, container)
          << "stream channel is "
          << (c.prod < 0 ? "never written" : "never read")
          << " by any process; every channel needs one producer and one "
             "consumer";
      return failure();
    }
    // A `seq.fifo` needs a >=1-bit address, i.e. depth >= 2 (depth 1 yields
    // zero-width pointers CIRCT cannot lower). Depth is a throughput hint and
    // deeper buffering is always KPN-safe, so raise a sub-2 depth to 2.
    if (c.depth < 2) {
      logging::warn(logging::Stage::Emit, container)
          << "Stream FIFO depth " << c.depth
          << " raised to 2 (the minimum a seq.fifo can express)";
      c.depth = 2;
    }
  }
  return success();
}

// Liveness: a directed cycle carrying no initial tokens deadlocks, since
// every process on it blocks reading an empty channel and the composed
// design would hang. A seeded channel breaks a cycle's start dependence, so
// it suffices that the graph of unseeded channels is acyclic. Report the
// first zero-token cycle through the logger, with the container as the error
// subject (which marks the failure for the Python caller to raise).
// Insufficient seeding (fewer tokens than the recurrence distance) is not
// caught here; it surfaces as a hang.
LogicalResult ConcurrentTopBuilder::checkFeedbackSeeding() {
  unsigned n = insts.size();
  SmallVector<SmallVector<unsigned>> adj(n); // producer -> consumer, unseeded
  for (auto &kv : chans) {
    const Chan &c = kv.second;
    if (!c.init || c.init.empty())
      adj[c.prod].push_back(c.cons);
  }
  SmallVector<int> color(n, 0), parent(n, -1); // 0 white / 1 gray / 2 black
  SmallVector<unsigned> cycle;
  // Self-parameter recursive lambda (`self(self, ...)`): a local DFS helper
  // with no std::function type-erasure / heap cost.
  auto visit = [&](auto &self, unsigned u) -> bool {
    color[u] = 1;
    for (unsigned v : adj[u]) {
      if (color[v] == 1) { // back edge -> cycle v .. u -> v
        for (int x = int(u); x != int(v); x = parent[x])
          cycle.push_back(unsigned(x));
        cycle.push_back(v);
        return true;
      }
      if (color[v] == 0) {
        parent[v] = int(u);
        if (self(self, v))
          return true;
      }
    }
    color[u] = 2;
    return false;
  };
  for (unsigned s = 0; s < n && cycle.empty(); ++s)
    if (color[s] == 0)
      visit(visit, s);
  if (cycle.empty())
    return success();

  std::reverse(cycle.begin(), cycle.end()); // producer order v -> ... -> u
  std::string path;
  {
    llvm::raw_string_ostream os(path);
    auto calleeOf = [&](unsigned idx) {
      return func::CallOp(insts[idx].call).getCallee();
    };
    for (unsigned idx : cycle)
      os << calleeOf(idx) << " -> ";
    os << calleeOf(cycle.front()); // close the loop
  }
  logging::error(logging::Stage::Emit, container)
      << "Dataflow feedback cycle [" << path
      << "] has no initial tokens and will deadlock; seed a channel on the "
         "cycle with an initializer, e.g. `s: Stream[T, depth] = [<init>]`";
  return failure();
}

// Forward each container argument to the process(es) that use it: mirror the
// callee's port(s) onto the top and record the top's interface entry. The
// top-side port name derives from the container argument (via its NameLoc) so
// two arguments never collide even when callees name their params the same;
// direction and width come straight from the callee port.
void ConcurrentTopBuilder::planBoundaryPorts() {
  // A boundary argument shared by several children costs differently by
  // kind: MEMORY = one port group per access (bound to one backing array);
  // SCALAR = one port fanned to all users; STREAM = single-producer/consumer,
  // rejected if shared.
  //
  // Whether two children may share an array is the SCHEDULE's call, not the
  // emitter's: a hazard makes the scheduler order the pair, which the
  // emitter realizes via start offset or `done`; an unordered pair is
  // concurrent by design.

  // Accessor groups seen per (top argument, role): the group index the top's
  // port base carries, so several children accessing one argument never
  // collide. Keyed on the argument's own owner token, not on a port position.
  llvm::StringMap<unsigned> groupSeq;
  llvm::DenseMap<int64_t, std::pair<std::string, Type>>
      scalarPort;                                // arg -> its one top port
  llvm::DenseMap<int64_t, unsigned> streamOwner; // arg -> owning instance
  // A boundary operand is always a container block argument, asserted in
  // `topArg`, so `ownerOf` resolves to a source name or `a<argNo>` and the
  // fallback is unreachable.
  auto owner = [&](Inst &in, unsigned calleeArg) {
    return ownerOf(in.call.getOperand(calleeArg), "");
  };
  auto boundaryMemBase = [&](Inst &in, unsigned calleeArg,
                             bool write) -> std::string {
    std::string own = owner(in, calleeArg);
    return memBase(own, write, groupSeq[own + (write ? "w" : "r")]++);
  };
  auto topArg = [&](Inst &in, unsigned calleeArg) -> int64_t {
    auto arg = cast<BlockArgument>(in.call.getOperand(calleeArg));
    assert(arg.getOwner()->getParentOp() == container &&
           "a boundary operand must be a container argument");
    return arg.getArgNumber();
  };
  auto mirror = [&](Inst &in, const std::string &calleeName,
                    const std::string &topName, unsigned ii) {
    PortDesc pd = in.ports[calleeName];
    mirrors.push_back({topName, calleeName, pd.type, pd.isInput, ii});
  };

  for (unsigned ii = 0; ii < insts.size(); ++ii) {
    Inst &in = insts[ii];
    for (const iface::Scalar &sc : in.mi->scalars) {
      // One top port per scalar argument, fanned out: children sharing an
      // argument bind the same port (a value costs nothing to share).
      int64_t arg = topArg(in, sc.arg);
      Type ty = in.ports[sc.name].type;
      auto it = scalarPort.find(arg);
      if (it == scalarPort.end()) {
        auto tn = scalarBase(owner(in, sc.arg));
        topIface.scalars.push_back({(int)arg, sc.width, tn});
        it = scalarPort.insert({arg, {tn, ty}}).first;
      }
      assert(it->second.second == ty &&
             "children sharing a scalar argument must agree on its width");
      mirror(in, sc.name, it->second.first, ii);
    }
    auto memPorts = [&](const std::vector<std::vector<iface::Memory>> &accs,
                        bool write) {
      for (const auto &grp : accs)
        for (const iface::Memory &cm : grp) {
          if (!isa<BlockArgument>(in.call.getOperand(cm.arg)))
            continue; // an internal buffer -> hlmem in the top, not a port
          auto tbase = boundaryMemBase(in, cm.arg, write);
          mirror(in, cm.addr, portAddr(tbase), ii);
          mirror(in, cm.data, portData(tbase), ii);
          if (write)
            mirror(in, cm.we, portWe(tbase), ii);
          iface::Memory mem{(int)topArg(in, cm.arg),
                            write,
                            cm.bank,
                            cm.factor,
                            cm.width,
                            cm.latency,
                            tbase,
                            portAddr(tbase),
                            portData(tbase),
                            write ? portWe(tbase) : std::string()};
          (write ? topIface.writes : topIface.reads)
              .push_back({std::move(mem)});
        }
    };
    memPorts(in.mi->reads, /*write=*/false);
    memPorts(in.mi->writes, /*write=*/true);
    // Stream boundaries: a stream arg bound to a container block argument
    // forwards as a top stream port, mirrored with the callee's own
    // directions (internal channels get a FIFO instead, in
    // discoverChannels/wireFifos).
    for (const iface::FIFO &f : in.mi->streams) {
      if (!isa<BlockArgument>(in.call.getOperand(f.arg)))
        continue;
      // A channel is single-producer / single-consumer, so a boundary stream
      // has exactly one owning child (unlike an array, which several may
      // share).
      bool fresh = streamOwner.try_emplace(topArg(in, f.arg), ii).second;
      assert(fresh && "a container boundary stream shared by two processes is "
                      "unsupported (a channel is single-producer / "
                      "single-consumer)");
      (void)fresh;
      // Single-owner, so no group index: the same `<arg>_st` a leaf presents.
      auto tbase = streamBase(owner(in, f.arg));
      mirror(in, f.data, portData(tbase), ii);
      mirror(in, f.valid, portValid(tbase), ii);
      mirror(in, f.ready, portReady(tbase), ii);
      topIface.streams.push_back({(int)topArg(in, f.arg), f.isInput, f.depth,
                                  f.width, tbase, portData(tbase),
                                  portValid(tbase), portReady(tbase)});
    }
    assert(in.mi->results.empty() &&
           "a process returning a scalar result is unsupported");
  }
}

// One ChanWires per channel: FIFO status/output backedges (resolved in
// wireFifos) plus, for a seeded channel, the consumer-side init-prepend shim.
llvm::MapVector<Value, ChanWires>
ConcurrentTopBuilder::buildChannelWires(OpBuilder &ib, BackedgeBuilder &bb) {
  llvm::MapVector<Value, ChanWires> cw;
  for (auto [idx, kv] : llvm::enumerate(chans)) {
    Chan &c = kv.second;
    ChanWires w{};
    w.full = bb.get(i1);
    w.empty = bb.get(i1);
    w.dataOut = bb.get(c.payload);
    w.notFull = notBit(ib, w.full);
    w.notEmpty = notBit(ib, w.empty);
    if (c.init && !c.init.empty()) {
      unsigned k = c.init.size();
      unsigned remW = 1;
      while ((1u << remW) <= k)
        ++remW;
      Type remTy = ib.getIntegerType(remW);
      auto rc = [&](int64_t v) -> Value {
        return hw::ConstantOp::create(ib, loc, remTy, v);
      };
      w.remNext = bb.get(remTy);
      w.rem = seq::CompRegOp::create(
          ib, loc, w.remNext, clk, rst, rc(k),
          channelSignal(ownerOf(kv.first, chanOwner(idx)), "init_rem"));
      w.servingInit =
          comb::ICmpOp::create(ib, loc, comb::ICmpPredicate::ne, w.rem, rc(0));
      // Data from the init ROM by the running index (idx = k-rem, served in
      // order as rem counts k..1; the rem==1 token falls through). A token is
      // the payload bit pattern (float carried as its bits).
      auto tok = [&](unsigned idx) -> Value {
        Attribute a = c.init[idx];
        APInt bits = isa<IntegerAttr>(a)
                         ? cast<IntegerAttr>(a).getValue()
                         : cast<FloatAttr>(a).getValue().bitcastToAPInt();
        unsigned pw = cast<IntegerType>(c.payload).getWidth();
        return hw::ConstantOp::create(ib, loc, c.payload,
                                      bits.zextOrTrunc(pw).getZExtValue());
      };
      Value dataInit = tok(k - 1);
      for (unsigned v = 2; v <= k; ++v) {
        Value isV = comb::ICmpOp::create(ib, loc, comb::ICmpPredicate::eq,
                                         w.rem, rc(v));
        dataInit = comb::MuxOp::create(ib, loc, isV, tok(k - v), dataInit);
      }
      w.dataMux =
          comb::MuxOp::create(ib, loc, w.servingInit, dataInit, w.dataOut);
      w.validShim =
          comb::OrOp::create(ib, loc, w.servingInit, w.notEmpty, false);
    }
    cw[kv.first] = w;
  }
  return cw;
}

// Instantiate each process: wire clk/rst/start, its stream ports (through the
// channel wires / init shim), and its mirrored boundary inputs; collect the
// instance's outputs by port name.
void ConcurrentTopBuilder::instantiateProcesses(
    OpBuilder &ib, hw::HWModulePortAccessor &pa,
    llvm::MapVector<Value, ChanWires> &cw) {
  for (unsigned ii = 0; ii < insts.size(); ++ii) {
    Inst &in = insts[ii];
    llvm::StringMap<Value> ins;
    ins[kClk] = clkRaw;
    ins[kRst] = rst;
    // Start policy, schedule-driven per child: a child gated on any
    // INDETERMINATE producer takes Route A (rising edge of joined `done`);
    // an async spawn broadcasts `start`; everything else fires at its Route B
    // scheduled offset.
    bool routeA = llvm::any_of(
        in.gateOns, [&](unsigned p) { return !insts[p].determinate; });
    if (routeA) {
      Value ready;
      for (unsigned p : in.gateOns) {
        Value d = insts[p].outs[kDone];
        ready = ready
                    ? comb::AndOp::create(ib, loc, ready, d, false).getResult()
                    : d;
      }
      ins[kStart] = risingEdge(ib, ready);
    } else if (in.isAsync)
      ins[kStart] = start;
    else
      ins[kStart] = delayPulse(ib, start, in.startOffset);
    for (const iface::FIFO &f : in.mi->streams) {
      // A boundary stream is a top stream port, wired through the mirror loop
      // below; only internal channels connect to a FIFO's status wires here.
      if (isa<BlockArgument>(in.call.getOperand(f.arg)))
        continue;
      ChanWires &w = cw[in.call.getOperand(f.arg)];
      if (f.isInput) {
        ins[f.data] = w.dataMux ? w.dataMux : w.dataOut;
        ins[f.valid] = w.validShim ? w.validShim : w.notEmpty;
      } else {
        ins[f.ready] = w.notFull;
      }
    }
    for (const Mirror &m : mirrors)
      if (m.inst == ii && m.isInput)
        ins[m.calleeName] = pa.getInput(m.topName);

    in.outs = instantiateChild(ib, loc, in.mod,
                               childInstanceName(in.call.getCallee(), ii), ins);
  }
}

// A `seq.fifo` per channel, wired to the producer/consumer handshakes; resolve
// the channel backedges. A seeded channel pops the FIFO only once its init
// tokens drain and advances the shim's down-counter on each init-served read.
void ConcurrentTopBuilder::wireFifos(OpBuilder &ib,
                                     llvm::MapVector<Value, ChanWires> &cw) {
  for (auto &kv : chans) {
    Chan &c = kv.second;
    ChanWires &w = cw[kv.first];
    Value pData = insts[c.prod].outs[portData(c.prodBase)];
    Value pValid = insts[c.prod].outs[portValid(c.prodBase)];
    Value cReady = insts[c.cons].outs[portReady(c.consBase)];
    Value wrEn = comb::AndOp::create(ib, loc, pValid, w.notFull, false);
    Value rdEn = comb::AndOp::create(ib, loc, cReady, w.notEmpty, false);
    if (w.rem) {
      rdEn =
          comb::AndOp::create(ib, loc, rdEn, notBit(ib, w.servingInit), false);
      Value doInit = comb::AndOp::create(ib, loc, w.servingInit, cReady, false);
      Value one = hw::ConstantOp::create(ib, loc, w.rem.getType(), 1);
      Value dec = comb::SubOp::create(ib, loc, w.rem, one);
      w.remNext.setValue(comb::MuxOp::create(ib, loc, doInit, dec, w.rem));
    }
    auto fifo = seq::FIFOOp::create(
        ib, loc, c.payload, i1, i1, Type(), Type(), pData, rdEn, wrEn, clk, rst,
        ib.getI64IntegerAttr(c.depth), ib.getI64IntegerAttr(0), IntegerAttr(),
        IntegerAttr());
    w.dataOut.setValue(fifo.getOutput());
    w.full.setValue(fifo.getFull());
    w.empty.setValue(fifo.getEmpty());
  }
}

// Join: done = AND of every process `done` (each a latched level); drive the
// mirrored boundary outputs from their owning instance.
void ConcurrentTopBuilder::forkJoin(OpBuilder &ib,
                                    hw::HWModulePortAccessor &pa) {
  Value done;
  for (Inst &in : insts) {
    Value d = in.outs[kDone];
    done = done ? comb::AndOp::create(ib, loc, done, d, false).getResult() : d;
  }
  pa.setOutput(kDone, done ? done : tru);
  for (const Mirror &m : mirrors)
    if (!m.isInput)
      pa.setOutput(m.topName, insts[m.inst].outs[m.calleeName]);
}

void ConcurrentTopBuilder::buildBody(OpBuilder &ib,
                                     hw::HWModulePortAccessor &pa) {
  BackedgeBuilder bb(ib, loc);
  clkRaw = pa.getInput(kClk);
  rst = pa.getInput(kRst);
  start = pa.getInput(kStart);
  clk = seq::ToClockOp::create(ib, loc, clkRaw);
  tru = hw::ConstantOp::create(ib, loc, i1, 1);
  auto cw = buildChannelWires(ib, bb);
  instantiateProcesses(ib, pa, cw);
  wireFifos(ib, cw);
  forkJoin(ib, pa);
}

LogicalResult ConcurrentTopBuilder::run(hw::HWModuleOp &modOut,
                                        iface::ModuleInterface &ifaceOut) {
  // The router sends only a CONCURRENT container here: the structural top
  // wires already-emitted callee instances, channels, and shared memory,
  // with no datapath of its own; loose top-level datapath ops are rejected
  // below rather than silently mis-wired.
  bool looseDatapath = false;
  container.walk([&](Operation *op) {
    if (isa<dcp::DCPathLoadOp, dcp::DCPathStoreOp, dcp::DCPathComputeOp>(op))
      looseDatapath = true;
  });
  if (looseDatapath)
    return container.emitError(
        "allo-datapath-to-hw: a concurrent (dataflow) container with its own "
        "top-level datapath ops (loose load/store/compute beside the process "
        "network) is not supported; the structural top composes child "
        "instances "
        "+ channels only");

  // A loop-over-calls container lowers to the leaf CallUnit path; reaching
  // the structural top means its callee is not leaf-eligible (a banked or
  // indeterminate child), which would fire once instead of once per iteration.
  if (loopOverCall(container))
    return container.emitError("allo-datapath-to-hw: a loop-over-calls "
                               "container must lower to the leaf CallUnit "
                               "path; its callee is not leaf-eligible (a "
                               "banked or indeterminate callee)");

  collectInstances();
  computeStartGates();
  if (failed(discoverChannels()))
    return failure();
  if (failed(checkFeedbackSeeding()))
    return failure();
  planBoundaryPorts();
  // Declare the top's ports from its composed port model, the same canonical
  // ABI declaration a leaf uses (declareModulePorts); `mirrors` still drives
  // the by-name body wiring below.
  auto ports = declareModulePorts(topIface, b);
  // Hand back the emitted module and its port model so the caller can
  // register them, since an enclosing container consumes this top exactly
  // like a leaf; the model (its toJSON() is the cosim manifest) needs no IR
  // attribute.
  topIface.symbol = container.getSymName().str();
  topIface.module = verilogName(topIface.symbol);
  modOut = hw::HWModuleOp::create(
      b, loc, StringAttr::get(ctx, topIface.module), hw::ModulePortInfo(ports),
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) { buildBody(ib, pa); });
  ifaceOut = std::move(topIface);
  return success();
}
} // namespace

// The structural-top emitter for a concurrent container: the per-child wiring
// (broadcast / static offset / `done` handshake) and the channel (FIFO /
// shared boundary) are derived from the schedule and the callees' determinacy,
// not a container-wide mode (see gateOns / the start policy in
// instantiateProcesses).
LogicalResult emitConcurrentTop(
    func::FuncOp container, const llvm::StringMap<hw::HWModuleOp> &modules,
    const llvm::StringMap<iface::ModuleInterface> &ifaceModels,
    const llvm::StringMap<func::FuncOp> &scheduledFuncs, OpBuilder &b,
    hw::HWModuleOp &modOut, iface::ModuleInterface &ifaceOut) {
  return ConcurrentTopBuilder(container, modules, ifaceModels, scheduledFuncs,
                              b)
      .run(modOut, ifaceOut);
}

} // namespace mlir::allo::uarch
