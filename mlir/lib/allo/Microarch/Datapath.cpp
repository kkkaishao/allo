/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/DatapathBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/OperatorLibrary.h" // unit input delay

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

unsigned hwWidth(Type t) {
  if (isa<IndexType>(t))
    return kIndexWidth;
  if (auto f = dyn_cast<FloatType>(t))
    return f.getWidth();
  return cast<IntegerType>(t).getWidth();
}

Operation *Datapath::producingOp(const Source &s) const {
  switch (s.kind) {
  case Source::Kind::Unit:
    return units[s.id].boundOps[s.outPort].first;
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

Datapath::Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
                   const DeviceModel &dev, float cycleTime,
                   const CalleeCtx *callees, bool isTop) {
  atTop = isTop;
  maxWritePorts = dev.memory.maxWritePorts;
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
    for (auto [k, s] : llvm::enumerate(u.inputInits))
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
      for (auto [k, s] : llvm::enumerate(acc.addr))
        visit(s, Slot::MemAddress, k, acc.op, /*required=*/true);
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
  return lib.combDelay(OpKind::Or);
}

/// The delay `u`'s inputs must settle within, read from the same library row
/// the scheduler priced it against.
static double unitInDelay(const FuncUnit &u, const OperatorLibrary &lib) {
  if (u.identity.comb)
    return lib.combDelay(*u.identity.comb);
  auto opr = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(
      u.repOp(), cast<dcp::DCPathComputeOp>(u.repOp()).getOpTypeAttr());
  assert(opr && "an IP unit names a live dcp.operator");
  return opr.getInDelay().convertToDouble();
}

double unitSlack(const FuncUnit &u, float cycleTime,
                 const OperatorLibrary &lib) {
  double in = unitInDelay(u, lib);
  double slack = cycleTime;
  for (const auto &[op, residue] : u.boundOps) {
    auto z = op->getAttrOfType<FloatAttr>("z");
    slack = std::min(slack, cycleTime - (z ? z.getValueAsDouble() : 0.0) - in);
  }
  return slack;
}

/// The largest set of mutually adjacent vertices in \p adj, a bitset per
/// vertex, by Bron-Kerbosch with pivoting: every maximal clique contains the
/// pivot or a candidate NOT adjacent to it, so only those need branching. \p
/// budget bounds a recursion that stays exponential in the worst case;
/// exhausting it reports the whole vertex set, which only over-states.
static unsigned maxClique(llvm::ArrayRef<uint64_t> adj, uint64_t candidates,
                          uint64_t excluded, unsigned depth, unsigned &budget) {
  if (!candidates && !excluded)
    return depth;
  if (budget == 0)
    return adj.size();
  --budget;
  unsigned pivot = llvm::countr_zero(candidates | excluded);
  unsigned best = depth;
  for (uint64_t branch = candidates & ~adj[pivot]; branch;) {
    unsigned v = llvm::countr_zero(branch);
    uint64_t bit = uint64_t(1) << v;
    branch &= ~bit;
    best = std::max(best, maxClique(adj, candidates & adj[v], excluded & adj[v],
                                    depth + 1, budget));
    candidates &= ~bit;
    excluded |= bit;
  }
  return best;
}

llvm::SmallVector<uint64_t> Datapath::portGraph(
    MemId id, bool writesOnly, llvm::SmallVectorImpl<unsigned> &accessOf,
    llvm::SmallVectorImpl<std::pair<int, bool>> *callerOf) const {
  const MemUnit &m = mems[id];
  // Top-level ancestor of a region: the granularity `recordSiblingDeps` orders
  // at, and a container's children stay serial below it.
  auto topOf = [&](RegionId r) {
    while (regions[r].parent)
      r = *regions[r].parent;
    return r;
  };
  // Does call \p a precede \p b transitively? A channel-joined pair in a
  // concurrent container is deliberately NOT ordered, and writes from such a
  // pair really are simultaneous.
  auto callPrecedes = [&](CallId a, CallId b) {
    llvm::SmallVector<CallId> work{b};
    llvm::SmallDenseSet<CallId> seen{b};
    while (!work.empty()) {
      CallId c = work.pop_back_val();
      for (const CallUnit::Pred &p : calls[c].predecessors) {
        if (p.call == a)
          return true;
        if (seen.insert(p.call).second)
          work.push_back(p.call);
      }
    }
    return false;
  };

  struct Writer {
    RegionId top, region;
    unsigned residue;
    int call; // CallId, or -1 for a region-local access
    int bank; // the bank it commits to, or -1 when it may reach any
    bool independent = false; // a call port its child proved collision-free
  };
  // A skew records a SLOT in `staticBank`, and two slots rotate onto one bank,
  // so only an unskewed array's index names the memory an access reaches.
  auto bankOf = [&](std::optional<unsigned> b) {
    return m.skewed || !b ? -1 : int(*b);
  };
  llvm::SmallVector<Writer> ws;
  for (auto [i, acc] : llvm::enumerate(m.accesses))
    if (acc.isWrite || !writesOnly) {
      unsigned ii = regions[acc.region].ii.value_or(0);
      unsigned start = dcpStart(acc.op);
      ws.push_back({topOf(acc.region), acc.region, ii ? start % ii : start, -1,
                    bankOf(acc.staticBank)});
      accessOf.push_back(i);
      if (callerOf)
        callerOf->push_back({-1, false});
    }
  for (const CallUnit &cu : calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      if (ma.mem == id && (ma.isWrite || !writesOnly)) {
        ws.push_back({topOf(cu.region), cu.region, 0, int(cu.id),
                      bankOf(ma.bank), ma.independent});
        accessOf.push_back(kNoWritePort);
        if (callerOf)
          callerOf->push_back({int(cu.id), ma.independent});
      }
  // The bitsets are 64 wide. Above that the relation is not built and every
  // caller treats each access as simultaneous, which only over-states and so
  // never merges a port unsafely.
  if (ws.size() > 64)
    return {};

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
  auto overlaps = [&](const Writer &a, const Writer &b) {
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
  llvm::SmallVector<uint64_t> adj(ws.size(), 0);
  for (unsigned i = 0; i < ws.size(); ++i)
    for (unsigned j = i + 1; j < ws.size(); ++j)
      if (overlaps(ws[i], ws[j])) {
        adj[i] |= uint64_t(1) << j;
        adj[j] |= uint64_t(1) << i;
      }
  return adj;
}

unsigned Datapath::portsNeeded(MemId id, bool writesOnly) const {
  llvm::SmallVector<unsigned> accessOf;
  llvm::SmallVector<uint64_t> adj = portGraph(id, writesOnly, accessOf);
  unsigned n = accessOf.size();
  if (n < 2 || adj.size() != n)
    return n; // one access, or too many to relate: all of them at once
  unsigned budget = 1u << 20;
  uint64_t all = n == 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1;
  return maxClique(adj, all, /*excluded=*/0, /*depth=*/0, budget);
}

unsigned Datapath::callPortSlot(MemId id, CallId call, unsigned arg) const {
  unsigned slot = mems[id].accesses.size();
  for (const CallUnit &cu : calls)
    for (auto [k, ma] : llvm::enumerate(cu.memArgs)) {
      if (ma.mem != id || !ma.isWrite)
        continue;
      if (cu.id == call && k == arg)
        return slot;
      ++slot;
    }
  llvm_unreachable("no such call-mastered write of this array");
}

std::optional<llvm::SmallVector<unsigned>>
Datapath::writePortColouring(MemId id, unsigned maxPorts) const {
  const MemUnit &m = mems[id];
  llvm::SmallVector<unsigned> accessOf;
  llvm::SmallVector<std::pair<int, bool>> callerOf;
  llvm::SmallVector<uint64_t> adj =
      portGraph(id, /*writesOnly=*/true, accessOf, &callerOf);
  unsigned n = accessOf.size();
  if (adj.size() != n)
    return std::nullopt; // no relation to colour over

  // Greedy in vertex order, taking the lowest port no neighbour holds.
  llvm::SmallVector<unsigned> colour(n, 0);
  unsigned used = 0;
  for (unsigned i = 0; i < n; ++i) {
    uint64_t taken = 0;
    for (unsigned j = 0; j < i; ++j)
      if ((adj[i] >> j) & 1)
        taken |= uint64_t(1) << colour[j];
    colour[i] = llvm::countr_one(taken);
    used = std::max(used, colour[i] + 1);
  }
  if (used > maxPorts)
    return std::nullopt;

  // Splitting the writes across ports only orders a simultaneous pair if it is
  // PROVEN to address different words. Two accesses of one region are, and so
  // are two write ports of one child that declared them independent, which is
  // that child having proven the same thing about its own accesses. Anything
  // else and every write stays on the port it has today.
  auto proven = [&](unsigned i, unsigned j) {
    if (callerOf[i].first < 0 && callerOf[j].first < 0)
      return m.accesses[accessOf[i]].region == m.accesses[accessOf[j]].region;
    return callerOf[i].first >= 0 && callerOf[i].first == callerOf[j].first &&
           callerOf[i].second;
  };
  if (used > 1)
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = i + 1; j < n; ++j)
        if (((adj[i] >> j) & 1) && !proven(i, j))
          return std::nullopt;
  // Every surviving edge joins two writers of one region or one child, at one
  // bank and one modulo residue, an equivalence, so the graph is a disjoint
  // union of cliques and greedy colouring is exact.
  assert(used == portsNeeded(id, /*writesOnly=*/true) &&
         "the colouring must use as many ports as the model demands");

  // Accesses at their own index, then the call-mastered writes appended in
  // `portGraph` order, which is what `callPortSlot` reproduces.
  llvm::SmallVector<unsigned> port(m.accesses.size(), kNoWritePort);
  for (unsigned i = 0; i < n; ++i)
    if (accessOf[i] == kNoWritePort)
      port.push_back(colour[i]);
    else
      port[accessOf[i]] = colour[i];
  return port;
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
         << " @" << u.boundOps.front().second << "] <= ";
      printSourceList(u.inputs, os);
      for (unsigned k = 0; k < u.inputInits.size(); ++k)
        if (u.inputInits[k].kind != Source::Kind::None) {
          os << " init[" << k << "]="; // recurrence-input reduction identity
          printSource(u.inputInits[k], os);
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
      os << " sel@["; // per-source op start cycle (the delayValid select stage)
      llvm::interleaveComma(x.selectOps, os, [&](Operation *op) {
        os << cast<IntegerAttr>(op->getAttr("start")).getInt();
      });
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
       << (cu.determinate ? " determinate" : " indeterminate");
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
