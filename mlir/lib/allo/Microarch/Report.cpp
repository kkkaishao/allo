/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Report.h"

#include "allo/Microarch/Datapath.h"
#include "allo/Microarch/Naming.h"       // operatorModuleName, ownerOf
#include "allo/Scheduling/MemoryModel.h" // bankKindName

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

namespace {

/// How an array's banks decompose, as one word. Several axes of one kind read
/// as that kind; a mix has no single name and says so rather than picking the
/// first, which would report a block-and-cyclic array as either one.
std::string layoutName(const MemUnit &m) {
  if (m.layout.registers)
    return "complete";
  if (m.layout.axes.empty())
    return "none";
  BankLayout::Kind first = m.layout.axes.front().kind;
  for (const BankLayout::Axis &a : m.layout.axes)
    if (a.kind != first)
      return "mixed";
  return bankKindName(first).str();
}

/// The structure the module built for an array, as one word.
llvm::StringRef realizationName(MemUnit::Realization r) {
  switch (r) {
  case MemUnit::Realization::Boundary:
    return "boundary";
  case MemUnit::Realization::Rom:
    return "rom";
  case MemUnit::Realization::Scatter:
    return "scatter";
  case MemUnit::Realization::Ram:
    return "ram";
  }
  llvm_unreachable("every realization is named");
}

/// The multiplexers of one region, aggregated by (fan-in, width).
std::vector<MuxClass> muxClasses(const Datapath &dp, const RegionBlock &rb) {
  std::map<std::pair<unsigned, unsigned>, unsigned> byClass;
  for (MuxId mid : rb.muxes)
    ++byClass[{(unsigned)dp.muxes[mid].sources.size(),
               datapathWidth(dp.muxes[mid].type)}];
  std::vector<MuxClass> out;
  out.reserve(byClass.size());
  for (const auto &[key, count] : byClass)
    out.push_back({key.first, key.second, count});
  return out;
}

} // namespace

FuncUarch::FuncUarch(const Datapath &dp, llvm::StringRef symbol,
                     llvm::StringRef module, const RegLedger &ledger)
    : func(symbol.str()), module(module.str()), top(dp.atTop),
      regs(ledger.classes()), readPorts(dp.readPorts.size()),
      writePorts(dp.writePorts.size()) {
  for (const RegionBlock &rb : dp.regions) {
    RegionUarch r;
    r.order = rb.id;
    r.shape = shapeName(rb.shape).str();
    r.kind = rb.kind == RegionBlock::Kind::Cyclic ? "cyclic" : "acyclic";
    if (rb.ii)
      r.interval = (int64_t)*rb.ii;
    r.cost.addrStrides = rb.addrStrides.size();
    if (rb.counterType)
      r.cost.counterWidth = datapathWidth(rb.counterType);
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      r.computeOps += u.boundOps.size();
      r.units.push_back(
          {u.identity.key(), u.identity.ipSymbol,
           u.identity.comb ? std::string() : operatorModuleName(u),
           datapathWidth(u.identity.resultType), u.latency,
           (unsigned)u.boundOps.size(), u.identity.comb.has_value(),
           u.pipelined});
    }
    r.muxes = muxClasses(dp, rb);
    for (MuxId mid : rb.muxes) {
      unsigned k = dp.muxes[mid].sources.size();
      r.cost.muxInputs += k;
      // A k:1 mux costs about (k-1) 2:1 muxes per bit.
      r.cost.muxBits += datapathWidth(dp.muxes[mid].type) * (k - 1);
    }
    regions.push_back(std::move(r));
  }

  for (const MemUnit &m : dp.mems) {
    MemReport mr;
    mr.owner = memArrayName(dp, m);
    auto mt = cast<MemRefType>(m.memref.getType());
    mr.shape.assign(mt.getShape().begin(), mt.getShape().end());
    mr.width = m.width;
    mr.banks = m.numBanks;
    mr.layout = layoutName(m);
    mr.storage = m.storage;
    mr.depthWords = m.depthWords;
    mr.readLatency = m.readLatency;
    mr.writeLatency = m.writeLatency;
    for (const MemUnit::Access &acc : m.accesses)
      (acc.isWrite ? mr.writes : mr.reads)++;
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id)
          (ma.isWrite ? mr.cost.callWrites : mr.cost.callReads)++;
    mr.cost.readPorts = m.readPortsBuilt;
    mr.cost.writePorts = m.writePortsBuilt;
    mr.cost.ports = m.portsBuilt;
    mr.cost.instances = m.instances;
    mr.cost.copiesBudget = m.ports.copies();
    mr.cost.rowReads = m.ports.instReads.value_or(0);
    mr.cost.rowWrites = m.ports.instWrites.value_or(0);
    mr.cost.readConcurrency = m.readConcurrency;
    mr.cost.writeConcurrency = m.writeConcurrency;
    mr.cost.boundaryPorts = m.boundaryPorts;
    mr.realization = realizationName(m.realization());
    mr.external = m.external;
    mr.scattered = m.scattered;
    mr.writesIndependent = m.writesIndependent;
    mr.rom = m.isRom;
    mr.skewed = m.skewed;
    mr.partitionResolved =
        m.numBanks <= 1 ||
        llvm::all_of(m.accesses, [](const MemUnit::Access &a) {
          return a.staticBank.has_value();
        });
    mems.push_back(std::move(mr));
  }

  for (const StreamChannel &s : dp.streams)
    streams.push_back({ownerOf(s.stream, chanOwner(s.id)),
                       datapathWidth(s.payload), s.depth, !s.callEnds.empty()});

  std::map<std::string, CallReport> byCallee;
  for (const CallUnit &cu : dp.calls) {
    CallReport &c = byCallee[cu.callee];
    c.callee = cu.callee;
    ++c.count;
    c.spawns += cu.async;
    c.latency = cu.latency;
  }
  for (auto &[name, c] : byCallee)
    calls.push_back(std::move(c));
}

std::string MicroarchReport::toJSON() const {
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::json::OStream j(os);
  j.object([&] {
    j.attribute("version", (int64_t)kVersion);
    j.attribute("binding", binding);
    j.attribute("cycle_time", cycleTime);
    j.attributeArray("funcs", [&] {
      for (const FuncUarch &f : funcs)
        j.object([&] {
          j.attribute("func", f.func);
          j.attribute("module", f.module);
          j.attribute("top", f.top);
          j.attribute("read_ports", (int64_t)f.readPorts);
          j.attribute("write_ports", (int64_t)f.writePorts);
          j.attributeArray("regions", [&] {
            for (const RegionUarch &r : f.regions)
              j.object([&] {
                j.attribute("order", r.order);
                j.attribute("shape", r.shape);
                j.attribute("kind", r.kind);
                if (r.interval)
                  j.attribute("interval", *r.interval);
                j.attribute("compute_ops", (int64_t)r.computeOps);
                j.attributeObject("cost", [&] {
                  j.attribute("mux_inputs", (int64_t)r.cost.muxInputs);
                  j.attribute("mux_bits", (int64_t)r.cost.muxBits);
                  j.attribute("counter_width", (int64_t)r.cost.counterWidth);
                  j.attribute("addr_strides", (int64_t)r.cost.addrStrides);
                });
                j.attributeArray("units", [&] {
                  for (const UnitReport &u : r.units)
                    j.object([&] {
                      j.attribute("identity", u.identity);
                      if (!u.impl.empty())
                        j.attribute("impl", u.impl);
                      if (!u.module.empty())
                        j.attribute("module", u.module);
                      j.attribute("width", (int64_t)u.width);
                      j.attribute("latency", (int64_t)u.latency);
                      j.attribute("bound_ops", (int64_t)u.boundOps);
                      j.attribute("comb", u.comb);
                      j.attribute("pipelined", u.pipelined);
                    });
                });
                j.attributeArray("muxes", [&] {
                  for (const MuxClass &m : r.muxes)
                    j.object([&] {
                      j.attribute("fanin", (int64_t)m.fanin);
                      j.attribute("width", (int64_t)m.width);
                      j.attribute("count", (int64_t)m.count);
                    });
                });
              });
          });
          j.attributeArray("regs", [&] {
            for (const RegClass &c : f.regs)
              j.object([&] {
                j.attribute("role", roleName(c.role));
                j.attribute("width", (int64_t)c.width);
                j.attribute("depth", (int64_t)c.depth);
                j.attribute("count", (int64_t)c.count);
              });
          });
          j.attributeArray("mems", [&] {
            for (const MemReport &m : f.mems)
              j.object([&] {
                j.attribute("owner", m.owner);
                j.attributeArray("shape", [&] {
                  for (int64_t d : m.shape)
                    j.value(d);
                });
                j.attribute("width", (int64_t)m.width);
                j.attribute("banks", (int64_t)m.banks);
                j.attribute("layout", m.layout);
                j.attribute("storage", m.storage);
                j.attribute("depth_words", (int64_t)m.depthWords);
                j.attribute("read_latency", (int64_t)m.readLatency);
                j.attribute("write_latency", (int64_t)m.writeLatency);
                j.attribute("reads", (int64_t)m.reads);
                j.attribute("writes", (int64_t)m.writes);
                j.attributeObject("cost", [&] {
                  j.attribute("call_reads", (int64_t)m.cost.callReads);
                  j.attribute("call_writes", (int64_t)m.cost.callWrites);
                  j.attribute("read_ports", (int64_t)m.cost.readPorts);
                  j.attribute("write_ports", (int64_t)m.cost.writePorts);
                  j.attribute("ports", (int64_t)m.cost.ports);
                  j.attribute("instances", (int64_t)m.cost.instances);
                  j.attribute("copies_budget", (int64_t)m.cost.copiesBudget);
                  j.attribute("row_reads", (int64_t)m.cost.rowReads);
                  j.attribute("row_writes", (int64_t)m.cost.rowWrites);
                  j.attribute("read_concurrency",
                              (int64_t)m.cost.readConcurrency);
                  j.attribute("write_concurrency",
                              (int64_t)m.cost.writeConcurrency);
                  j.attribute("boundary_ports", (int64_t)m.cost.boundaryPorts);
                });
                j.attribute("external", m.external);
                j.attribute("scattered", m.scattered);
                j.attribute("writes_independent", m.writesIndependent);
                j.attribute("realization", m.realization);
                j.attribute("rom", m.rom);
                j.attribute("skewed", m.skewed);
                j.attribute("partition_resolved", m.partitionResolved);
              });
          });
          j.attributeArray("streams", [&] {
            for (const StreamReport &s : f.streams)
              j.object([&] {
                j.attribute("owner", s.owner);
                j.attribute("width", (int64_t)s.width);
                j.attribute("depth", (int64_t)s.depth);
                j.attribute("crosses_call", s.crossesCall);
              });
          });
          j.attributeArray("calls", [&] {
            for (const CallReport &c : f.calls)
              j.object([&] {
                j.attribute("callee", c.callee);
                j.attribute("count", (int64_t)c.count);
                j.attribute("spawns", (int64_t)c.spawns);
                if (c.latency)
                  j.attribute("latency", *c.latency);
              });
          });
        });
    });
  });
  return out;
}

} // namespace mlir::allo::uarch
