/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Interface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::uarch; // the naming vocabulary, unqualified

namespace mlir::allo::iface {

namespace {
// The datapath's hardware width for a type: index -> 32, a float carried as its
// bit pattern, an integer verbatim (matches `uarch::hwType`, builder-free).
unsigned bitWidth(Type t) {
  if (isa<IndexType>(t))
    return 32;
  if (auto f = dyn_cast<FloatType>(t))
    return f.getWidth();
  return cast<IntegerType>(t).getWidth();
}

int argOf(Value v) {
  auto ba = dyn_cast<BlockArgument>(v);
  return ba ? (int)ba.getArgNumber() : -1;
}
} // namespace

ModuleInterface::ModuleInterface(const uarch::Datapath &dp,
                                 ArrayRef<uarch::AccRef> reads,
                                 ArrayRef<uarch::AccRef> writes) {
  for (const uarch::IOPort &io : dp.ios)
    if (io.isInput)
      scalars.push_back(
          {argOf(io.value), bitWidth(io.type), scalarPortName(dp, io)});

  for (const uarch::StreamChannel &s : dp.streams) {
    auto base = streamPortBase(dp, s);
    streams.push_back({argOf(s.stream), s.isInput, (int)s.depth,
                       bitWidth(s.payload), base, portData(base),
                       portValid(base), portReady(base)});
  }

  // Each external access expands to one interface per boundary bank (one when
  // unbanked / statically routed, N for a data-dependent access spanning
  // banks).
  auto group = [&](ArrayRef<uarch::AccRef> ports, unsigned i, bool write) {
    const auto &mu = dp.mems[ports[i].mem];
    unsigned w =
        bitWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
    int factor = externalBank(mu, mu.accesses[ports[i].idx]).factor;
    unsigned lat = write ? mu.writeLatency : mu.readLatency;
    std::vector<Memory> g;
    for (const auto &[bank, base] : extPorts(dp, ports, i, write))
      g.push_back({argOf(mu.memref), write, (int)bank, factor, w, lat, base,
                   portAddr(base), portData(base),
                   write ? portWe(base) : std::string()});
    return g;
  };
  for (unsigned i = 0; i < reads.size(); ++i)
    this->reads.push_back(group(reads, i, /*write=*/false));
  for (unsigned i = 0; i < writes.size(); ++i)
    this->writes.push_back(group(writes, i, /*write=*/true));

  // A CallUnit-mastered *boundary* argument has no MemUnit::Access (the child
  // drives the port), so it's declared here with the same `<name>_<role><i>`
  // naming as a normal port; emitCalls passes the child's ports through.
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (!ma.isBoundary)
        continue;
      // One port group per accessor, so concurrent or serial accessors of one
      // argument get separate groups backed by the same array. A cyclic
      // argument gets one group per bank.
      const auto &mu = dp.mems[ma.mem];
      unsigned w =
          bitWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
      const auto &base = ma.topBase; // indexed per role by the builder
      Memory m{argOf(mu.memref),
               ma.isWrite,
               (int)ma.bank,
               (int)ma.factor,
               w,
               ma.isWrite ? mu.writeLatency : mu.readLatency,
               base,
               portAddr(base),
               portData(base),
               ma.isWrite ? portWe(base) : std::string()};
      (ma.isWrite ? this->writes : this->reads).push_back({m});
    }

  for (const uarch::Result &r : dp.results)
    results.push_back({bitWidth(r.type), r.name});
}

std::string ModuleInterface::toJSON() const {
  using llvm::json::Array;
  using llvm::json::Object;
  using llvm::json::Value;

  auto mems = [](const std::vector<std::vector<Memory>> &accs) {
    Array out;
    for (const auto &acc : accs) {
      Array banks;
      for (const Memory &p : acc) {
        Object o{{"arg", p.arg},
                 {"bank", p.bank},
                 {"factor", p.factor},
                 {"width", (int64_t)p.width},
                 {"latency", (int64_t)p.latency},
                 {"base", p.base},
                 {"addr", p.addr},
                 {"data", p.data}};
        if (!p.we.empty())
          o["we"] = p.we;
        banks.push_back(std::move(o));
      }
      out.push_back(std::move(banks));
    }
    return out;
  };

  Array scalars;
  for (const Scalar &s : this->scalars)
    scalars.push_back(
        Object{{"arg", s.arg}, {"width", (int64_t)s.width}, {"name", s.name}});
  Array streams;
  for (const FIFO &s : this->streams)
    streams.push_back(Object{{"arg", s.arg},
                             {"input", s.isInput},
                             {"depth", s.depth},
                             {"width", (int64_t)s.width},
                             {"base", s.base},
                             {"data", s.data},
                             {"valid", s.valid},
                             {"ready", s.ready}});
  Array results;
  for (const Result &r : this->results)
    results.push_back(Object{{"width", (int64_t)r.width}, {"name", r.name}});
  Array operators;
  for (const Operator &o : this->operators) {
    Array ports;
    for (const Operator::Port &p : o.ports) {
      llvm::StringRef role = p.role == Operator::Role::Data  ? "data"
                             : p.role == Operator::Role::Clk ? "clk"
                             : p.role == Operator::Role::Ce  ? "ce"
                                                             : "out";
      ports.push_back(Object{{"name", p.name},
                             {"width", (int64_t)p.width},
                             {"role", role},
                             {"input", p.isInput()}});
    }
    operators.push_back(Object{{"module", o.module},
                               {"impl", o.impl},
                               {"predicate", o.predicate},
                               {"ports", std::move(ports)}});
  }

  Value root = Object{{"module", module},
                      {"symbol", symbol},
                      // The fixed control ABI, published so no consumer has to
                      // hard-code it on its own side.
                      {"control", Object{{"clk", uarch::kClk},
                                         {"rst", uarch::kRst},
                                         {"start", uarch::kStart},
                                         {"done", uarch::kDone}}},
                      {"scalars", std::move(scalars)},
                      {"streams", std::move(streams)},
                      {"reads", mems(reads)},
                      {"writes", mems(writes)},
                      {"results", std::move(results)},
                      {"operators", std::move(operators)}};
  std::string s;
  llvm::raw_string_ostream os(s);
  os << root;
  return s;
}

} // namespace mlir::allo::iface
