/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Interface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::allo;

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
          {argOf(io.value), bitWidth(io.type), uarch::scalarPortName(io)});

  for (const uarch::StreamChannel &s : dp.streams) {
    std::string base = uarch::streamPortBase(s);
    streams.push_back({argOf(s.stream), s.isInput, (int)s.depth,
                       bitWidth(s.payload), base, data_(base), valid(base),
                       ready(base)});
  }

  // Each external access expands to one interface per boundary bank (one when
  // unbanked / statically routed, N for a data-dependent access spanning
  // banks).
  auto group = [&](ArrayRef<uarch::AccRef> ports, unsigned i, bool write) {
    const uarch::MemUnit &mu = dp.mems[ports[i].mem];
    unsigned w =
        bitWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
    int factor = uarch::externalBank(mu, mu.accesses[ports[i].idx]).factor;
    std::vector<Memory> g;
    for (const auto &[bank, base] :
         uarch::extPorts(dp, ports, i, write ? "wr" : "rd"))
      g.push_back({argOf(mu.memref), write, (int)bank, factor, w, base,
                   addr(base), data_(base), write ? we(base) : std::string()});
    return g;
  };
  for (unsigned i = 0; i < reads.size(); ++i)
    this->reads.push_back(group(reads, i, /*write=*/false));
  for (unsigned i = 0; i < writes.size(); ++i)
    this->writes.push_back(group(writes, i, /*write=*/true));

  // A CallUnit-mastered *boundary* argument has no MemUnit::Access (the child
  // instance drives the port), so it is absent from the AccRef arrays above.
  // Declare its interface here -- the same `<name>_<role>` a normal
  // single-access boundary port gets -- so the top declares the port and the
  // cosim harness drives it; the leaf's own access loops still skip it (they
  // iterate the AccRefs), and emitCalls passes the child's ports through.
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (!ma.isBoundary)
        continue;
      // One port group PER ACCESSOR: the builder gave each a distinct `topBase`
      // (a running index per base), so several children accessing one argument
      // get separate concurrent groups -- the cosim harness backs every group
      // of the argument against its one array (no mux). A serial pair uses two
      // groups too (each drives in its own phase; the schedule keeps them
      // ordered). A cyclically partitioned argument exposes one group per
      // bank, each carrying its own `bank`/`factor` -- how the cosim harness
      // knows to back it with the argument's cyclic slice rather than a whole
      // copy.
      const uarch::MemUnit &mu = dp.mems[ma.mem];
      unsigned w =
          bitWidth(cast<MemRefType>(mu.memref.getType()).getElementType());
      const std::string &base = ma.topBase; // indexed per role by the builder
      Memory m{
          argOf(mu.memref), ma.isWrite,  (int)ma.bank,
          (int)ma.factor,   w,           base,
          addr(base),       data_(base), ma.isWrite ? we(base) : std::string()};
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
        Object o{{"arg", p.arg},       {"bank", p.bank},
                 {"factor", p.factor}, {"width", (int64_t)p.width},
                 {"base", p.base},     {"addr", p.addr},
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

  Value root = Object{{"scalars", std::move(scalars)},
                      {"streams", std::move(streams)},
                      {"reads", mems(reads)},
                      {"writes", mems(writes)},
                      {"results", std::move(results)}};
  std::string s;
  llvm::raw_string_ostream os(s);
  os << root;
  return s;
}

} // namespace mlir::allo::iface
