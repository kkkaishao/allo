/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Interface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::uarch;

namespace mlir::allo::iface {

namespace {
// The boundary carries a value exactly as wide as the datapath does, so the
// port model reads the ONE width rule rather than restating it (`uarch::hwType`
// is the same rule with an OpBuilder, for making the `IntegerType`).
using uarch::hwWidth;

int argOf(Value v) {
  auto ba = dyn_cast<BlockArgument>(v);
  return ba ? (int)ba.getArgNumber() : -1;
}

// The element-space bank decomposition of \p memref, in the manifest's shape:
// the host reproduces it to shard a numpy argument across the bank interfaces
// exactly as the emitted address arithmetic does.
std::pair<std::vector<int64_t>, std::vector<Memory::Axis>>
layoutOf(const uarch::MemUnit &mu) {
  auto shape = cast<MemRefType>(mu.memref.getType()).getShape();
  std::vector<Memory::Axis> axes;
  for (const BankLayout::Axis &a : mu.layout.axes) // decoded by the builder
    axes.push_back({(int)a.dim, a.factor, bankKindName(a.kind).str()});
  return {{shape.begin(), shape.end()}, std::move(axes)};
}
} // namespace

ModuleInterface::ModuleInterface(const uarch::Datapath &dp) {
  ArrayRef<uarch::AccRef> reads = dp.readPorts, writes = dp.writePorts;
  // Every IOPort is a scalar kernel argument; a scalar *result* is a
  // `dp.results` entry, declared further down.
  for (const uarch::IOPort &io : dp.ios)
    scalars.push_back(
        {argOf(io.value), hwWidth(io.type), scalarPortName(dp, io)});

  for (const uarch::StreamChannel &s : dp.streams) {
    if (s.internal)
      continue; // kernel-local: a seq.fifo in the body, not a boundary port
    auto base = streamPortBase(dp, s);
    streams.push_back({argOf(s.stream), s.isInput, (int)s.depth,
                       hwWidth(s.payload), base, portData(base),
                       portValid(base), portReady(base)});
  }

  // A scattered argument is declared per element, off the memory rather than
  // off its accesses, so it appears here and in neither `reads` nor `writes`
  // (its accesses take no port group at all).
  for (const uarch::MemUnit &mu : dp.mems) {
    if (!mu.scattered)
      continue;
    auto mt = cast<MemRefType>(mu.memref.getType());
    auto shape = mt.getShape();
    std::vector<RegisterFile::Element> elems;
    for (const uarch::MemUnit::ElemPort &p : mu.elemPorts)
      elems.push_back({p.in, p.out, p.we});
    registers.push_back({argOf(mu.memref),
                         hwWidth(mt.getElementType()),
                         {shape.begin(), shape.end()},
                         std::move(elems)});
  }

  // Each external access expands to one interface per boundary bank (one when
  // unbanked / statically routed, N for a data-dependent access spanning
  // banks).
  auto group = [&](uarch::AccRef r, bool write) {
    const auto &mu = dp.mems[r.id];
    const auto &acc = mu.accesses[r.idx];
    unsigned w =
        hwWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
    int factor = externalBank(mu, acc).factor;
    unsigned lat = write ? mu.writeLatency : mu.readLatency;
    auto [shape, axes] = layoutOf(mu);
    std::vector<Memory> g;
    for (const auto &[bank, base] : extPorts(mu, acc))
      g.push_back({argOf(mu.memref), write, (int)bank, factor, w, lat, base,
                   portAddr(base), portData(base),
                   write ? portWe(base) : std::string(), shape, axes});
    return g;
  };
  for (uarch::AccRef r : reads)
    this->reads.push_back(group(r, /*write=*/false));
  for (uarch::AccRef r : writes)
    this->writes.push_back(group(r, /*write=*/true));

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
          hwWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
      const auto &base = ma.topBase; // indexed per role by the builder
      auto [shape, axes] = layoutOf(mu);
      Memory m{argOf(mu.memref),
               ma.isWrite,
               (int)ma.bank,
               (int)ma.factor,
               w,
               ma.isWrite ? mu.writeLatency : mu.readLatency,
               base,
               portAddr(base),
               portData(base),
               ma.isWrite ? portWe(base) : std::string(),
               shape,
               axes};
      (ma.isWrite ? this->writes : this->reads).push_back({m});
    }

  for (const uarch::Result &r : dp.results)
    results.push_back({hwWidth(r.type), r.name});
}

llvm::SmallVector<const Memory *, 2>
ModuleInterface::portsForArg(int arg) const {
  llvm::SmallVector<const Memory *, 2> out;
  for (const std::vector<std::vector<Memory>> *side : {&reads, &writes})
    for (const std::vector<Memory> &grp : *side)
      for (const Memory &m : grp)
        if (m.arg == arg)
          out.push_back(&m);
  return out;
}

const FIFO *ModuleInterface::streamForArg(int arg) const {
  for (const FIFO &s : streams)
    if (s.arg == arg)
      return &s;
  return nullptr;
}

const Scalar *ModuleInterface::scalarForArg(int arg) const {
  for (const Scalar &s : scalars)
    if (s.arg == arg)
      return &s;
  return nullptr;
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
        // The bank decomposition, published only for a partitioned argument:
        // the host shards its numpy array with it (see `plan_mems`).
        if (!p.axes.empty()) {
          Array shape;
          for (int64_t d : p.shape)
            shape.push_back(d);
          o["shape"] = std::move(shape);
          Array axes;
          for (const Memory::Axis &a : p.axes)
            axes.push_back(
                Object{{"dim", a.dim}, {"factor", a.factor}, {"kind", a.kind}});
          o["axes"] = std::move(axes);
        }
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
  Array registers;
  for (const RegisterFile &rf : this->registers) {
    Array shape, elements;
    for (int64_t d : rf.shape)
      shape.push_back(d);
    // An unused direction has no port, so its key is absent rather than empty:
    // a consumer tests for the port it needs instead of for a sentinel.
    for (const RegisterFile::Element &e : rf.elements) {
      Object o;
      if (!e.in.empty())
        o["in"] = e.in;
      if (!e.out.empty()) {
        o["out"] = e.out;
        o["we"] = e.we;
      }
      elements.push_back(std::move(o));
    }
    registers.push_back(Object{{"arg", rf.arg},
                               {"width", (int64_t)rf.width},
                               {"shape", std::move(shape)},
                               {"elements", std::move(elements)}});
  }
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
                      {"registers", std::move(registers)},
                      {"results", std::move(results)},
                      {"operators", std::move(operators)}};
  std::string s;
  llvm::raw_string_ostream os(s);
  os << root;
  return s;
}

} // namespace mlir::allo::iface
