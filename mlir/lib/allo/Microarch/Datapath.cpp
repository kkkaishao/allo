/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Public face of the L2 microarchitecture model: the `build` entry (a thin
// wrapper over DatapathBuilder), `hasDCPRegions` detection, and the textual
// `dump`. The construction itself lives in DatapathBuilder.{h,cpp}.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/DatapathBuilder.h"

#include "allo/IR/AlloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR (see Datapath.h). One definition of
// the schedule cycle, the operator latency, and the derived result-ready cycle.
//===----------------------------------------------------------------------===//

Operation *dcpOperatorOp(Operation *op) {
  FlatSymbolRefAttr sym;
  if (auto c = dyn_cast<dcp::DCPathComputeOp>(op))
    sym = c.getOpTypeAttr();
  else if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    sym = l.getOpTypeAttr();
  if (!sym)
    return nullptr;
  return SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(op, sym);
}

unsigned dcpStart(Operation *op) {
  return cast<IntegerAttr>(op->getAttr("start")).getInt();
}

unsigned dcpLatency(Operation *op) {
  auto opr = dyn_cast_or_null<dcp::DCPathOperatorOp>(dcpOperatorOp(op));
  return opr ? static_cast<unsigned>(opr.getLatency()) : 0;
}

unsigned readyCycleOf(Operation *op) { return dcpStart(op) + dcpLatency(op); }

Datapath::Datapath(func::FuncOp func) {
  DatapathBuilder builder(*this, func, TrivialBinding{});
  builder.build();
}

Datapath::Datapath(func::FuncOp func, const BindingPolicy &policy) {
  DatapathBuilder builder(*this, func, policy);
  builder.build();
}

//===----------------------------------------------------------------------===//
// Textual dump.
//===----------------------------------------------------------------------===//

namespace {

void printValueName(Value v, raw_ostream &os) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    os << "#arg" << arg.getArgNumber();
  else if (Operation *def = v.getDefiningOp())
    os << def->getName().getStringRef();
  else
    os << "<?>";
}

void printSource(const Source &s, raw_ostream &os) {
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
  }
}

void printSourceList(ArrayRef<Source> ss, raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(ss, os, [&](const Source &s) { printSource(s, os); });
  os << "]";
}

} // namespace

void Datapath::dump(llvm::raw_ostream &os) const {
  func::FuncOp func = this->func;
  os << "datapath @" << func.getSymName() << " {\n";

  for (const RegionBlock &rb : this->regions) {
    os << "  region " << rb.id << ": "
       << (rb.guard                               ? "guard"
           : rb.conditional                       ? "while"
           : rb.kind == RegionBlock::Kind::Cyclic ? "cyclic"
                                                  : "acyclic");
    if (rb.ii)
      os << " ii=" << *rb.ii;
    os << " length=" << rb.length;
    if (rb.tripCount)
      os << " trip=" << *rb.tripCount;
    os << "\n";
    for (UnitId uid : rb.units) {
      const FuncUnit &u = this->units[uid];
      os << "    unit u" << uid << ": " << u.opType << " lat=" << u.latency
         << (u.pipelined ? " pipelined" : " sequential") << " : "
         << u.resultType << "  [" << u.boundOps.front().first->getName() << " @"
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
      os << "    reg r" << rid << ": depth=" << r.depth << " taps=[";
      llvm::interleaveComma(r.taps, os);
      os << "] <= ";
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
       << " ports=" << m.portsPerBank
       << " impl=" << stringifyMemoryImplEnum(m.impl) << "\n";
    for (const MemUnit::Access &acc : m.accesses) {
      os << "    " << (acc.isWrite ? "wr " : "rd ") << acc.op->getName()
         << " addr=";
      printSourceList(acc.addr, os);
      if (acc.isWrite) {
        os << " data=";
        printSource(acc.data, os);
      }
      os << "\n";
    }
  }

  for (const ConstCell &c : this->consts)
    os << "  const c" << c.id << ": " << c.value << "\n";

  for (const IOPort &io : this->ios)
    os << "  io i" << io.id << ": " << (io.isInput ? "in " : "out ") << io.type
       << "\n";

  // A region's results, each held for a sibling as a survivor (program order).
  for (const RegionBlock &rb : this->regions)
    if (auto it = this->regionResult.find(rb.id);
        it != this->regionResult.end())
      for (auto [k, rs] : llvm::enumerate(it->second)) {
        os << "  result region " << rb.id << "#" << k << " <= ";
        printSource(rs, os);
        os << "\n";
      }

  os << "}\n";
}

} // namespace mlir::allo::uarch
