/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/DatapathBuilder.h"

#include "allo/IR/AlloOps.h"

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
  // landing. A region carries a `latency` too, but that one is the whole
  // region's start->done span, so answering for it would report a span as an
  // operator delay.
  assert(!isa<dcp::DCPathRegionOpInterface>(op) &&
         "a region's `latency` is its whole span, not an operator latency");
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return static_cast<unsigned>(l.getLatency());
  // An IP compute takes its latency from the `dcp.operator` it names, which
  // outlives emission for this reason; a combinational one lands in the cycle
  // it issues.
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
    return units[s.id].repOp();
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
  case Source::Kind::Scope:
    // At-issue, held, or produced outside this region. A `Scope` cone HAS an op
    // but no schedule: it is combinational over settled registers, so reporting
    // it here would hand `readyCycleOf` an op with no `start` attribute.
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
                   const MemoryLibrary &memLib, const CalleeCtx *callees,
                   bool isTop) {
  atTop = isTop;
  DatapathBuilder builder(*this, func, policy, memLib, callees);
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
  case Slot::ScopeInput:
    return idx("operand") + " of a func-scope expression";
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
  // Every visit is one call; `required` states whether a None source there
  // means "absent" or "unresolved", so no consumer re-decides it.
  auto visit = [&](const Source &s, Slot slot, unsigned index, Operation *op,
                   bool required) {
    fn(s, SourceSite{slot, index, op, required});
  };

  for (const FuncUnit &u : dp.units) {
    // A merged-away unit was dropped from its region and drives nothing.
    if (u.boundOps.empty())
      continue;
    for (auto [k, s] : llvm::enumerate(u.inputs))
      visit(s, Slot::UnitInput, k, u.repOp(), /*required=*/true);
    for (auto [k, s] : llvm::enumerate(u.inputInits))
      visit(s, Slot::UnitInit, k, u.repOp(), /*required=*/false);
  }
  for (const ScopeUnit &su : dp.scopeUnits)
    for (auto [k, s] : llvm::enumerate(su.inputs))
      visit(s, Slot::ScopeInput, k, su.op, /*required=*/true);
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
    // Set for a counted region (literal bounds are synthesized cells), None for
    // an acyclic one. `ubSource` is also None for the one derived bound
    // (`tripCount` over a runtime lb/step), so none of the three is required.
    for (const Source &s : {rb.lbSource, rb.ubSource, rb.stepSource})
      visit(s, Slot::RegionBound, rb.id, nullptr, /*required=*/false);
    // Only a Container threads its recurrence through `setupCarriedIterArgs`,
    // where an unresolved init or next has nothing to latch. Elsewhere a result
    // may be untracked, and a consumer that reads one fails at its own slot.
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
    os << "u" << s.id;
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
  case Source::Kind::Scope:
    os << "g" << s.id;
    break;
  }
}

static void printSourceList(ArrayRef<Source> ss, raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(ss, os, [&](const Source &s) { printSource(s, os); });
  os << "]";
}

void Datapath::dump(llvm::raw_ostream &os) const {
  auto func = this->func;
  os << "datapath @" << func.getSymName() << " {\n";

  // The controller discriminant as the emitter reads it: shape, then the
  // termination class that picks the cell.
  auto shapeName = [](RegionBlock::Shape s) -> const char * {
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
  };
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
      os << "    unit u" << uid << ": " << u.opType << " lat=" << u.latency
         << (u.pipelined ? " pipelined" : " sequential") << " : "
         << u.resultType << "  [" << u.repOp()->getName() << " @"
         << u.boundOps.front().second << "] <= ";
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
       << " impl=" << stringifyMemoryImplEnum(m.impl) << "\n";
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
  // inputs (spawn / determinacy / offset) and the predecessors it waits for.
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

  for (const ScopeUnit &su : this->scopeUnits) {
    os << "  scope g" << su.id << ": " << su.opType << " : " << su.resultType
       << " <= ";
    printSourceList(su.inputs, os);
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
