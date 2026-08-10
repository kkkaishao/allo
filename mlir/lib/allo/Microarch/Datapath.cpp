/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/DatapathBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/OperatorLibrary.h" // unit input delay
#include "allo/Support/BitAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR. One definition of the schedule
// cycle, the operator latency, and the derived result-ready cycle.
//===----------------------------------------------------------------------===//

unsigned dcpStart(Operation *op) {
  return cast<IntegerAttr>(op->getAttr("start")).getInt();
}

unsigned dcpLatency(Operation *op) {
  // An OPERATOR latency: the cycles between an op's issue and its result
  // landing. A region's `latency` is its whole start->done span, not an
  // operator delay.
  assert(!isa<dcp::DCPathRegionOpInterface>(op) &&
         "a region's `latency` is its whole span, not an operator latency");
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return static_cast<unsigned>(l.getLatency());
  // An IP compute takes its latency from the `dcp.operator` it names, which
  // outlives emission for this reason; a combinational one issues and lands in
  // the same cycle.
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op)) {
    FlatSymbolRefAttr sym = comp.getOpTypeAttr();
    if (!sym)
      return 0;
    auto opr =
        SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(comp, sym);
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    return static_cast<unsigned>(opr.getLatency());
  }
  // A store and a call carry their own `latency`, each an ODS field of its op.
  if (auto lat = op->getAttrOfType<IntegerAttr>("latency"))
    return static_cast<unsigned>(lat.getInt());
  return 0;
}

unsigned readyCycleOf(Operation *op) { return dcpStart(op) + dcpLatency(op); }

llvm::StringRef shapeName(RegionBlock::Shape s) {
  switch (s) {
  case RegionBlock::Shape::Leaf:
    return "leaf";
  case RegionBlock::Shape::Container:
    return "container";
  case RegionBlock::Shape::Guard:
    return "guard";
  case RegionBlock::Shape::CallNode:
    return "callnode";
  }
  llvm_unreachable("unhandled RegionBlock::Shape");
}

llvm::StringRef startPolicyName(CallUnit::StartPolicy p) {
  switch (p) {
  case CallUnit::StartPolicy::Handshake:
    return "handshake";
  case CallUnit::StartPolicy::Broadcast:
    return "broadcast";
  case CallUnit::StartPolicy::TimeTriggered:
    return "timed";
  }
  llvm_unreachable("unhandled CallUnit::StartPolicy");
}

Operation *Datapath::producingOp(const Source &s) const {
  switch (s.kind) {
  case Source::Kind::Unit:
    return units[s.id].boundOps[s.outPort].op;
  case Source::Kind::Mem: // outPort = the read access index
    return mems[s.id].accesses[s.outPort].op;
  case Source::Kind::Stream: // outPort = the get access index
    return streams[s.id].accesses[s.outPort].op;
  case Source::Kind::Call:
    return calls[s.id].invoke;
  case Source::Kind::None:
  case Source::Kind::Reg:
  case Source::Kind::Mux:
  case Source::Kind::IO:
  case Source::Kind::Const:
  case Source::Kind::Counter:
  case Source::Kind::Survivor:
    // At-issue, held, or produced outside this region.
    return nullptr;
  }
  llvm_unreachable("unhandled Source::Kind");
}

std::optional<int64_t> Datapath::constantOf(const Source &s) const {
  if (s.kind != Source::Kind::Const)
    return std::nullopt;
  auto ia = dyn_cast<IntegerAttr>(consts[s.id].value);
  return ia ? std::optional<int64_t>(ia.getInt()) : std::nullopt;
}

unsigned Datapath::readyCycle(const Source &s) const {
  switch (s.kind) {
  // A call is the one producer whose result does NOT land at `stage +
  // latency`: it lands at its region-relative issue plus the CALLEE's whole
  // start->done depth. Indeterminate calls are guarded earlier.
  case Source::Kind::Call: {
    const CallUnit &cu = calls[s.id];
    assert(cu.latency && "readyCycle of an indeterminate call result");
    return cu.start + static_cast<unsigned>(*cu.latency);
  }
  case Source::Kind::Unit: {
    const FuncUnit &u = units[s.id];
    return u.boundOps[s.outPort].stage + u.latency;
  }
  case Source::Kind::Mem: {
    const MemUnit &m = mems[s.id];
    return m.accesses[s.outPort].stage + m.readLatency;
  }
  // A get is a combinational front-read of the FIFO, so it lands at issue.
  case Source::Kind::Stream:
    return streams[s.id].accesses[s.outPort].stage;
  // A held source has no landing stage: a literal is constant, an IO port
  // stable for the whole kernel, and a counter or survivor a register settled
  // by the time the region reading it issues.
  case Source::Kind::Const:
  case Source::Kind::IO:
  case Source::Kind::Counter:
  case Source::Kind::Survivor:
    return 0;
  case Source::Kind::Reg:
  case Source::Kind::Mux:
  case Source::Kind::None:
    break;
  }
  llvm_unreachable("readyCycle only models a producing or held Source");
}

Datapath::Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
                   const DeviceModel &dev, float cycleTime,
                   const CalleeCtx &callees, bool isTop) {
  // What the model is OF, settled here rather than half here and half in the
  // builder: both are properties of the request, not of anything derived.
  this->func = func;
  atTop = isTop;
  DatapathBuilder builder(*this, func, policy, dev, cycleTime, callees);
  builder.build();
}

//===----------------------------------------------------------------------===//
// The model visitor.
//===----------------------------------------------------------------------===//

std::string SourceSite::describe() const {
  auto idx = [&](const char *noun) {
    return std::string(noun) + " " + std::to_string(index);
  };
  switch (slot) {
  case Slot::UnitInput:
    return idx("operand") + " of a compute unit";
  case Slot::UnitInit:
    return "the reduction identity of " + idx("operand");
  case Slot::RegisterInput:
    return "the input of a pipeline register";
  case Slot::MuxInput:
    return idx("arm") + " of a shared-unit mux";
  case Slot::MemAddress:
    return idx("address index") + " of a memory access";
  case Slot::MemWriteData:
    return "the data of a memory write";
  case Slot::StreamData:
    return "the token data of a stream put";
  case Slot::StreamPredicate:
    return "the predicate of a stream access";
  case Slot::CallScalarIn:
    return idx("scalar argument") + " of a sub-kernel call";
  case Slot::FuncResult:
    return idx("scalar function result");
  case Slot::RegionBound:
    return "a runtime loop bound";
  case Slot::RegionResult:
    return idx("result") + " of a region";
  case Slot::RegionResultInit:
    return "the loop-carried identity of " + idx("result");
  case Slot::RegionElseResult:
    return idx("else-branch result") + " of a guard";
  case Slot::RegionCondition:
    return "the control predicate of a region";
  }
  llvm_unreachable("unhandled SourceSite::Slot");
}

llvm::BitVector residualReads(const MemUnit::Access &acc) {
  llvm::BitVector read(acc.addr.size());
  unsigned numDims = acc.addrMap.getNumDims();
  for (AffineExpr e : {acc.offset.residual, acc.bank.residual}) {
    if (!e)
      continue;
    e.walk([&](AffineExpr x) {
      unsigned p;
      if (auto d = dyn_cast<AffineDimExpr>(x))
        p = d.getPosition();
      else if (auto s = dyn_cast<AffineSymbolExpr>(x))
        p = numDims + s.getPosition();
      else
        return;
      // Past the operand list: a digit `Reduced::reads` supplies instead.
      if (p < read.size())
        read.set(p);
    });
  }
  return read;
}

void forEachSource(
    const Datapath &dp,
    llvm::function_ref<void(const Source &, const SourceSite &)> fn) {
  using Slot = SourceSite::Slot;
  // `required` states whether a None source at that slot means "absent" or
  // "unresolved", so no consumer re-decides it.
  auto visit = [&](const Source &s, Slot slot, unsigned index, Operation *op,
                   bool required) {
    fn(s, SourceSite{slot, index, op, required});
  };

  for (const FuncUnit &u : dp.units) {
    for (auto [k, s] : llvm::enumerate(u.inputs))
      visit(s, Slot::UnitInput, k, u.repOp(), /*required=*/true);
    for (auto [k, inits] : llvm::enumerate(u.inputInits))
      for (const Source &s : inits)
        visit(s, Slot::UnitInit, k, u.repOp(), /*required=*/false);
  }
  for (const Register &r : dp.regs)
    visit(r.input, Slot::RegisterInput, r.id, nullptr, /*required=*/true);
  for (const Mux &x : dp.muxes)
    for (auto [k, s] : llvm::enumerate(x.sources))
      visit(s, Slot::MuxInput, k,
            x.selectOps.empty() ? nullptr : x.selectOps[k],
            /*required=*/true);

  for (const MemUnit &m : dp.mems)
    for (const MemUnit::Access &acc : m.accesses) {
      llvm::BitVector read = residualReads(acc);
      for (auto [k, s] : llvm::enumerate(acc.addr))
        visit(s, Slot::MemAddress, k, acc.op, /*required=*/read[k]);
      // A load leaves `data` None by construction.
      visit(acc.data, Slot::MemWriteData, 0, acc.op, /*required=*/acc.isWrite);
    }
  for (const StreamChannel &ch : dp.streams)
    for (const StreamChannel::Access &acc : ch.accesses) {
      visit(acc.data, Slot::StreamData, 0, acc.op, /*required=*/acc.isPut);
      visit(acc.when, Slot::StreamPredicate, 0, acc.op, /*required=*/false);
    }
  for (const CallUnit &cu : dp.calls)
    for (auto [k, sa] : llvm::enumerate(cu.scalarIns))
      visit(sa.src, Slot::CallScalarIn, k, cu.invoke, /*required=*/true);
  for (auto [k, r] : llvm::enumerate(dp.results))
    visit(r.source, Slot::FuncResult, k, nullptr, /*required=*/true);

  for (const RegionBlock &rb : dp.regions) {
    // Set for a counted region, None for an acyclic one; `ubSource` is also
    // None for the one derived bound (`tripCount` over a runtime lb/step), so
    // none of the three is required.
    for (const Source &s : {rb.lbSource, rb.ubSource, rb.stepSource})
      visit(s, Slot::RegionBound, rb.id, nullptr, /*required=*/false);
    // Only a Container threads its recurrence through `setupCarriedIterArgs`,
    // where an unresolved init or next has nothing to latch. Elsewhere a result
    // may be untracked.
    bool threaded = rb.shape == RegionBlock::Shape::Container;
    for (auto [k, r] : llvm::enumerate(rb.results)) {
      visit(r.value, Slot::RegionResult, k, nullptr, threaded);
      visit(r.init, Slot::RegionResultInit, k, nullptr, threaded);
      visit(r.elseValue, Slot::RegionElseResult, k, nullptr,
            /*required=*/false);
    }
    // A while and a guard both need their predicate; a counted region has none.
    visit(rb.condition, Slot::RegionCondition, rb.id, nullptr,
          /*required=*/rb.conditional || rb.shape == RegionBlock::Shape::Guard);
  }
}

//===----------------------------------------------------------------------===//
// Textual dump.
//===----------------------------------------------------------------------===//

static void printValueName(Value v, raw_ostream &os) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    os << "#arg" << arg.getArgNumber();
  else if (Operation *def = v.getDefiningOp())
    os << def->getName().getStringRef();
  else
    os << "<?>";
}

static void printSource(const Source &s, raw_ostream &os) {
  switch (s.kind) {
  case Source::Kind::None:
    os << "-";
    break;
  case Source::Kind::Unit:
    os << "u" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Reg:
    os << "r" << s.id << "@" << s.outPort;
    break;
  case Source::Kind::Mem:
    os << "m" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Mux:
    os << "x" << s.id;
    break;
  case Source::Kind::IO:
    os << "i" << s.id;
    break;
  case Source::Kind::Const:
    os << "c" << s.id;
    break;
  case Source::Kind::Counter:
    os << "iv" << s.id;
    break;
  case Source::Kind::Survivor:
    os << "sv" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Stream:
    os << "st" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Call:
    os << "call" << s.id << "#" << s.outPort;
    break;
  }
}

static void printSourceList(ArrayRef<Source> ss, raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(ss, os, [&](const Source &s) { printSource(s, os); });
  os << "]";
}

unsigned muxLevels(unsigned sources) {
  return sources <= 1 ? 0 : llvm::Log2_32_Ceil(sources);
}

double muxLevelDelay(const OperatorLibrary &lib) {
  // Priced at one bit, since every bit of an OR level settles in parallel, so
  // width buys mux LUTs (`set_mux_uses`) rather than levels. The full row delay
  // and not the marginal one: a level of a wide one-hot select pays routing
  // comparable to a whole register-to-register hop, not the LUT hop a narrow
  // cone pays.
  return lib.combDelay(OpKind::Or, 1);
}

double unitSlack(const FuncUnit &u, float cycleTime) {
  double slack = cycleTime;
  for (const FuncUnit::BoundOp &bo : u.boundOps)
    slack = std::min(slack, cycleTime - bo.z.value_or(0.0) - u.inDelay);
  return slack;
}

Datapath::PortRelation Datapath::portGraph(MemId id,
                                           std::optional<bool> writes) const {
  const MemUnit &m = mems[id];
  PortRelation rel;
  // Top-level ancestor of a region: the granularity `recordSiblingDeps` orders
  // at, and a container's children stay serial below it.
  auto topOf = [&](RegionId r) {
    while (regions[r].parent)
      r = *regions[r].parent;
    return r;
  };
  // Does call \p a precede \p b transitively? A channel-joined pair in a
  // concurrent container is deliberately NOT ordered, and writes from such a
  // pair really are simultaneous. Memoized: the pair loop below is quadratic in
  // the accesses and this walk is the only part of it that is not constant.
  llvm::DenseMap<std::pair<CallId, CallId>, bool> precedes;
  auto callPrecedes = [&](CallId a, CallId b) {
    auto [it, isNew] = precedes.try_emplace({a, b}, false);
    if (!isNew)
      return it->second;
    llvm::SmallVector<CallId> work{b};
    llvm::SmallDenseSet<CallId> seen{b};
    while (!work.empty()) {
      CallId c = work.pop_back_val();
      for (const CallUnit::Pred &p : calls[c].predecessors) {
        if (p.call == a)
          return precedes[{a, b}] = true;
        if (seen.insert(p.call).second)
          work.push_back(p.call);
      }
    }
    return false;
  };

  // When each vertex runs, beside the identity `PortVertex` publishes.
  struct When {
    RegionId top, region;
    unsigned residue;
    int call; // CallId, or -1 for a region-local access
    int bank; // the bank it commits to, or -1 when it may reach any
  };
  // `staticBank` is empty under a skew, where two slots rotate onto one bank
  // and neither names the memory an access reaches.
  auto bankOf = [](std::optional<unsigned> b) { return b ? int(*b) : -1; };
  llvm::SmallVector<When> ws;
  auto add = [&](const When &w, unsigned access, bool write, bool independent) {
    rel.verts.push_back({access, w.call, independent, write, w.bank});
    ws.push_back(w);
  };
  // Writes before reads, and this function's own accesses before the ports its
  // children master: the order every caller writes its colouring back in.
  for (bool dir : {true, false}) {
    if (writes && *writes != dir)
      continue;
    for (auto [i, acc] : llvm::enumerate(m.accesses))
      if (acc.isWrite == dir) {
        unsigned ii = regions[acc.region].ii.value_or(0);
        unsigned start = acc.stage;
        add({topOf(acc.region), acc.region, ii ? start % ii : start, -1,
             bankOf(acc.staticBank)},
            i, dir, /*independent=*/false);
      }
    for (const CallUnit &cu : calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == id && ma.isWrite == dir)
          add({topOf(cu.region), cu.region, 0, int(cu.id), bankOf(ma.bank)},
              kNoAccess, dir, ma.independent);
  }
  // A container drives its children serially, so two accesses in different
  // regions under one top are ordered UNLESS a concurrent container is in the
  // chain, which places every child at 0.
  auto underConcurrent = [&](RegionId r) {
    for (;; r = *regions[r].parent) {
      if (regions[r].determinacy == DeterminacyEnum::Concurrent)
        return true;
      if (!regions[r].parent)
        return false;
    }
  };
  auto overlaps = [&](const When &a, const When &b) {
    // A bank is its own `seq.hlmem`, so two accesses that name different ones
    // contend for nothing however they are scheduled.
    if (a.bank >= 0 && b.bank >= 0 && a.bank != b.bank)
      return false;
    if (a.top != b.top)
      return false;
    if (a.call >= 0 && b.call >= 0)
      return !callPrecedes(a.call, b.call) && !callPrecedes(b.call, a.call);
    if (a.call < 0 && b.call < 0) {
      if (a.region == b.region)
        return a.residue == b.residue;
      return underConcurrent(a.region) || underConcurrent(b.region);
    }
    return true;
  };
  rel.adj.assign(ws.size(), llvm::BitVector(ws.size()));
  for (unsigned i = 0; i < ws.size(); ++i)
    for (unsigned j = i + 1; j < ws.size(); ++j)
      if (overlaps(ws[i], ws[j]))
        rel.link(i, j);
  return rel;
}

unsigned Datapath::portConcurrency(MemId id, bool writes) const {
  PortRelation rel = portGraph(id, writes);
  unsigned n = rel.size();
  // Grow a clique from each vertex, always taking the lowest remaining
  // candidate. A vertex is never adjacent to itself, so intersecting with the
  // one just taken drops it from the candidate set.
  unsigned best = n ? 1 : 0;
  for (unsigned s = 0; s < n; ++s) {
    llvm::BitVector cand = rel.adj[s];
    unsigned size = 1;
    while (cand.any()) {
      cand &= rel.adj[cand.find_first()];
      ++size;
    }
    best = std::max(best, size);
  }
  return best;
}

void Datapath::dump(llvm::raw_ostream &os) const {
  auto func = this->func;
  os << "datapath @" << func.getSymName() << " {\n";

  // The controller discriminant as the emitter reads it: shape, then
  // termination class.
  for (const RegionBlock &rb : this->regions) {
    os << "  region " << rb.id << ": " << shapeName(rb.shape) << "/"
       << (rb.conditional                         ? "while"
           : rb.kind == RegionBlock::Kind::Cyclic ? "cyclic"
                                                  : "acyclic");
    if (rb.ii)
      os << " ii=" << *rb.ii;
    if (rb.tripCount)
      os << " trip=" << *rb.tripCount;
    os << " drain=" << rb.drainStage;
    if (!rb.predecessors.empty()) {
      os << " after=[";
      llvm::interleaveComma(rb.predecessors, os, [&](RegionId p) { os << p; });
      os << "]";
    }
    os << "\n";
    for (UnitId uid : rb.units) {
      const FuncUnit &u = this->units[uid];
      os << "    unit u" << uid << ": " << u.identity.realizationName()
         << " lat=" << u.latency << (u.pipelined ? " pipelined" : " sequential")
         << " : " << u.identity.resultType << "  [" << u.repOp()->getName()
         << " @" << u.boundOps.front().residue << "] <= ";
      printSourceList(u.inputs, os);
      // A recurrence input's reduction identities, one per early iteration.
      for (auto [k, inits] : llvm::enumerate(u.inputInits))
        if (!inits.empty()) {
          os << " init[" << k << "]=";
          printSourceList(inits, os);
        }
      os << "\n";
    }
    for (RegId rid : rb.regs) {
      const Register &r = this->regs[rid];
      os << "    reg r" << rid << ": depth=" << r.depth << " <= ";
      printSource(r.input, os);
      os << " : " << r.type << "\n";
    }
    for (MuxId xid : rb.muxes) {
      const Mux &x = this->muxes[xid];
      os << "    mux x" << xid << ": ";
      printSourceList(x.sources, os);
      // Per-arm select: the op's start cycle (the delayValid select stage),
      // suffixed by the iteration window a phased arm drives ('iN' the
      // reduction identity of iteration N, 'rN' the iterations from N on).
      os << " sel@[";
      for (auto [k, stage] : llvm::enumerate(x.selectStages)) {
        const Mux::Phase &ph = x.phases[k];
        os << (k ? ", " : "") << stage;
        if (ph.kind != Mux::Phase::Always)
          os << (ph.kind == Mux::Phase::At ? "i" : "r") << ph.iter;
      }
      os << "]\n";
    }
  }

  for (const MemUnit &m : this->mems) {
    os << "  mem m" << m.id << ": ";
    printValueName(m.memref, os);
    os << (m.external ? " external" : " internal") << " w=" << m.width
       << " depth=" << m.depthWords << " banks=" << m.numBanks
       << " storage=" << m.storage << "\n";
    for (const MemUnit::Access &acc : m.accesses) {
      os << "    " << (acc.isWrite ? "wr " : "rd ") << acc.op->getName()
         << " @r" << acc.region << " addr=";
      printSourceList(acc.addr, os);
      if (acc.isWrite) {
        os << " data=";
        printSource(acc.data, os);
      }
      os << "\n";
    }
  }

  for (const StreamChannel &s : this->streams) {
    os << "  chan s" << s.id << ": ";
    printValueName(s.stream, os);
    os << (s.internal  ? " internal"
           : s.isInput ? " in"
                       : " out")
       << " depth=" << s.depth;
    if (auto init = dyn_cast_or_null<ArrayAttr>(s.init))
      os << " init=" << init.size();
    for (const StreamChannel::CallEnd &e : s.callEnds)
      os << (this->calls[e.call].streamArgs[e.arg].isInput ? " get@k"
                                                           : " put@k")
         << e.call;
    os << "\n";
  }

  // The composition graph on the instance substrate: each child's start policy
  // inputs and the predecessors it waits for.
  for (const CallUnit &cu : this->calls) {
    os << "  call k" << cu.id << ": " << cu.callee << " @r" << cu.region
       << " start=" << cu.start << (cu.async ? " spawn" : "")
       << (cu.determinate ? " determinate" : " indeterminate") << " via "
       << startPolicyName(cu.startPolicy);
    if (!cu.predecessors.empty()) {
      os << " after=[";
      llvm::interleaveComma(cu.predecessors, os, [&](const CallUnit::Pred &p) {
        os << "k" << p.call << (p.viaResult ? "(result)" : "");
      });
      os << "]";
    }
    os << "\n";
  }

  for (const ConstCell &c : this->consts)
    os << "  const c" << c.id << ": " << c.value << "\n";

  for (const IOPort &io : this->ios)
    os << "  io i" << io.id << ": in " << io.type << "\n";

  // A region's results, each held for a sibling as a survivor (program order),
  // with the loop-carried identity / else-arm value where the regime has one.
  for (const RegionBlock &rb : this->regions) {
    if (rb.condition) {
      os << "  cond region " << rb.id << " <= ";
      printSource(rb.condition, os);
      os << "\n";
    }
    for (auto [k, r] : llvm::enumerate(rb.results)) {
      os << "  result region " << rb.id << "#" << k << " <= ";
      printSource(r.value, os);
      if (r.init) {
        os << " init=";
        printSource(r.init, os);
      }
      if (r.elseValue) {
        os << " else=";
        printSource(r.elseValue, os);
      }
      os << "\n";
    }
  }

  os << "}\n";
}

} // namespace mlir::allo::uarch
