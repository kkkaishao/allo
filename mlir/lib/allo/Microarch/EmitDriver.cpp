/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/EmitDriver.h"

#include "allo/IR/AlloOps.h"
#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/HWEmitter.h" // HWEmitter
#include "allo/Microarch/Interface.h"
#include "allo/Microarch/Report.h"
#include "allo/Microarch/Verification.h" // validateDatapath
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Support/Logging.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt;

#define DEBUG_TYPE "hw-emitter"

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// emitModule: interface (ports, extern operator modules) + validation.
//===----------------------------------------------------------------------===//

// Declare an extern operator module for each IP-realized compute unit, named by
// `operatorModuleName` and deduplicated across the whole module (`opModules`).
// Native (comb) units emit inline, no extern. Returns unit id -> its extern
// module. Port order is one input per operand (`a`, `b`, `c`, ... at its
// width), clk, `ce` when the realization is clock-enabled (`ce == 0` freezes it
// in lockstep with the shell), then the output. Port shape is a function of the
// unit's identity, so deduplicating by module name is safe only as far as the
// name separates identities, which the assert below checks.
//
// The module name stems from the `dcp.operator`'s own `sym_name`, and that
// declaration stays live until every kernel has emitted, so the symbol is
// briefly duplicated. `SymbolTable::lookupSymbolIn` returns the first match in
// block order, so the `dcp.operator` has to stay ahead of these declarations.
static DenseMap<unsigned, Operation *>
declareOperatorModules(dcp::DCPathModuleOp func, const uarch::Datapath &dp,
                       OpBuilder &b, llvm::StringMap<Operation *> &opModules,
                       std::vector<iface::Operator> &declared) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  DenseMap<unsigned, Operation *> unitModule;
  // One manifest entry per module, not per unit; the value is the identity
  // that claimed the name.
  llvm::StringMap<const allo::OperatorIdentity *> listed;
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.identity.comb)
      continue;
    IntegerType outW = datapathType(u.identity.resultType, b);
    std::string modName = operatorModuleName(u);
    iface::Operator entry{
        modName, u.identity.ipSymbol, operatorPredicate(u), {}};
    SmallVector<PortInfo> ep;
    // Operand widths off the IDENTITY, which is what decides whether two units
    // may share a module name at all, so the ports cannot disagree with it.
    for (unsigned k = 0; k < u.identity.argTypes.size(); ++k) {
      IntegerType w = datapathType(u.identity.argTypes[k], b);
      std::string pn(1, static_cast<char>('a' + k));
      ep.push_back({{StringAttr::get(ctx, pn), w, Dir::Input}});
      entry.ports.push_back({pn, w.getWidth(), iface::Operator::Role::Data});
    }
    ep.push_back({{StringAttr::get(ctx, kClk), b.getI1Type(), Dir::Input}});
    entry.ports.push_back({kClk.str(), 1, iface::Operator::Role::Clk});
    if (u.stall == allo::StallContractEnum::Ce) {
      ep.push_back({{StringAttr::get(ctx, kCe), b.getI1Type(), Dir::Input}});
      entry.ports.push_back({kCe.str(), 1, iface::Operator::Role::Ce});
    }
    ep.push_back({{StringAttr::get(ctx, kOpOut), outW, Dir::Output}});
    entry.ports.push_back(
        {kOpOut.str(), outW.getWidth(), iface::Operator::Role::Out});

    Operation *&mod = opModules[modName];
    if (!mod)
      mod = hw::HWModuleExternOp::create(b, loc, StringAttr::get(ctx, modName),
                                         hw::ModulePortInfo(ep));
    auto [claim, fresh] = listed.try_emplace(modName, &u.identity);
    assert(*claim->second == u.identity &&
           "two operator identities share one module name");
    if (fresh)
      declared.push_back(std::move(entry));
    unitModule[u.id] = mod;
  }
  return unitModule;
}

llvm::SmallVector<hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b) {
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  auto *ctx = b.getContext();
  Type i1 = b.getI1Type(), i32 = b.getIntegerType(32);
  // A data port's hw width is its field bit width, so `iType(w)` reproduces
  // `datapathType`/`memElemType` for the data ports.
  auto iType = [&](unsigned w) -> Type { return b.getIntegerType(w); };
  SmallVector<PortInfo> ports;
  // The port names are the manifest, authored before CIRCT's LegalizeNames
  // runs, so a name ExportVerilog would rewrite or uniquify desyncs cosim from
  // the Verilog. These check the composed result.
  llvm::StringSet<> seen;
  auto port = [&](const Twine &n, Type t, Dir d) {
    std::string s = n.str();
    assert(sv::isNameValid(s, /*caseInsensitiveKeywords=*/false) &&
           "module port name is not a legal SystemVerilog identifier; the JSON "
           "manifest would desync from the emitted Verilog");
    bool fresh = seen.insert(s).second;
    assert(fresh && "duplicate module port name; the JSON manifest would "
                    "desync from the emitted Verilog");
    (void)fresh;
    ports.push_back(PortInfo{{StringAttr::get(ctx, s), t, d}});
  };
  port(kClk, i1, Dir::Input);
  port(kRst, i1, Dir::Input);
  port(kStart, i1, Dir::Input);
  // Scalar kernel arguments; memref args become memory ports instead. One
  // named after a control port trips the duplicate check above.
  for (const iface::Scalar &s : model.scalars)
    port(s.name, iType(s.width), Dir::Input);
  // Stream FIFO ports, input side. Module inputs must stay contiguous at the
  // front, since HWModulePortAccessor maps body args to the first `numInputs`
  // ports positionally, so {data, valid} / {ready} go here.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.data, iType(s.width), Dir::Input);
      port(s.valid, i1, Dir::Input);
    } else {
      port(s.ready, i1, Dir::Input);
    }
  }
  // A partitioned argument presents one interface per bank (a data-dependent
  // access spans all of them, a static access one); `model.reads[i]` holds an
  // access's per-bank interfaces.
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.data, iType(r.width), Dir::Input);
  // A fully-partitioned argument gets one input per element, no address or
  // latency, read combinationally in any number at once. A write-only argument
  // has no input side.
  for (const iface::RegisterFile &rf : model.registers)
    for (const iface::RegisterFile::Element &e : rf.elements)
      if (!e.in.empty())
        port(e.in, iType(rf.width), Dir::Input);
  port(kDone, i1, Dir::Output);
  // Stream FIFO ports, output side: an input stream's back-pressure {ready}, an
  // output stream's {data, valid}.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.ready, i1, Dir::Output);
    } else {
      port(s.data, iType(s.width), Dir::Output);
      port(s.valid, i1, Dir::Output);
    }
  }
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.addr, i32, Dir::Output);
  for (const auto &acc : model.writes)
    for (const iface::Memory &w : acc) {
      port(w.addr, i32, Dir::Output);
      port(w.data, iType(w.width), Dir::Output);
      port(w.we, i1, Dir::Output);
    }
  // A written scattered argument leaves on one data + write-enable pair per
  // element: the storage is the driver's, so an element commits only where the
  // module says it did.
  for (const iface::RegisterFile &rf : model.registers)
    for (const iface::RegisterFile::Element &e : rf.elements)
      if (!e.out.empty()) {
        port(e.out, iType(rf.width), Dir::Output);
        port(e.we, i1, Dir::Output);
      }
  // Scalar function results: one output port each, driven by the returning
  // region's survivor and valid when `done` rises (emit()).
  for (const iface::Result &r : model.results)
    port(r.name, iType(r.width), Dir::Output);
  return ports;
}

llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins) {
  using Dir = hw::ModulePort::Direction;
  SmallVector<Value> operands(mod.getNumInputPorts());
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Input) {
      auto it = ins.find(p.name.getValue());
      assert(it != ins.end() && "unwired child input port");
      operands[p.argNum] = it->second;
    }
  auto inst =
      hw::InstanceOp::create(b, loc, mod, b.getStringAttr(name), operands);
  llvm::StringMap<Value> outs;
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Output)
      outs[p.name.getValue()] = inst.getResult(p.argNum);
  return outs;
}

/// Flip-flops in \p mod's own body, which is what the ledger claims to count.
/// A child instance's registers live in the child's body and are not walked.
static unsigned compRegBits(hw::HWModuleOp mod) {
  unsigned bits = 0;
  mod.walk([&](seq::CompRegOp r) {
    bits += datapathWidth(r.getResult().getType());
  });
  return bits;
}

// Emit an hw.module for one scheduled function's datapath. Returns failure with
// a diagnostic if the datapath is outside the supported subset
// (validateDatapath). `opModules` caches extern operator modules across
// functions.
static FailureOr<std::pair<hw::HWModuleOp, iface::ModuleInterface>>
emitModule(dcp::DCPathModuleOp func, const uarch::Datapath &dp, OpBuilder &b,
           llvm::StringMap<Operation *> &opModules, float cycleTime,
           const OperatorLibrary &lib, MicroarchReport &report,
           const uarch::CalleeCtx *callees = nullptr) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  if (failed(validateDatapath(func, dp, cycleTime, lib)))
    return failure();

  Type i1 = b.getI1Type();
  Type i32 = b.getIntegerType(32);

  // The single source for every boundary port name, shared by declaration,
  // manifest and cosim harness; it also carries the extern operator modules
  // this kernel instantiates.
  iface::ModuleInterface model(dp);
  auto unitModule =
      declareOperatorModules(func, dp, b, opModules, model.operators);
  auto ports = declareModulePorts(model, b);

  hw::ModulePortInfo portInfo(ports);
  // Legalized here, so the key the manifest uses is the emitted Verilog module
  // name: a nested callee `top.child` would otherwise be rewritten downstream
  // by ExportVerilog.
  model.symbol = func.getSymName().str();
  model.module = verilogName(model.symbol);
  StringAttr modName = StringAttr::get(ctx, model.module);

  RegLedger ledger;
  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, unitModule, bb, i1, i32, callees);
        e.ctx.clk = e.ctx.R(seq::ToClockOp::create(ib, loc, pa.getInput(kClk)));
        e.ctx.clkRaw = pa.getInput(kClk);
        e.ctx.rst = pa.getInput(kRst);
        e.emit();
        ledger = std::move(e.ctx.ledger);
      });
  // Every register came through `EmitContext::reg`, so the ledger is the
  // emitted design's own flip-flop count and not a model of it. Checked here
  // rather than in one test, so every emission the suite runs holds it.
  assert(compRegBits(hwMod) == ledger.bits() &&
         "a register was built outside EmitContext::reg, so the ledger is no "
         "longer a count of the emitted design");
  report.funcs.emplace_back(dp, model.symbol, model.module, ledger);

  // The caller derives the cosim manifest JSON from this port model and threads
  // it back in as a callee model.
  return std::make_pair(hwMod, std::move(model));
}

static void cleanupDcpOps(ModuleOp module) {
  // cleanup non-hw ops to avoid Verilog export errors
  for (dcp::DCPathModuleOp f :
       llvm::make_early_inc_range(module.getOps<dcp::DCPathModuleOp>()))
    f.erase();
  for (memref::GlobalOp g :
       llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
    g.erase();
  // Spent declarations, dropped last: a `dcp.compute` reads its timing off the
  // `dcp.operator` it names, and dropping them leaves each extern operator
  // module sole owner of its `sym_name`.
  SmallVector<Operation *> spent;
  module.walk([&](Operation *op) {
    if (isa<dcp::DCPathOperatorOp, dcp::DCPathDeviceOp, dcp::DCPathUnitOp>(op))
      spent.push_back(op);
  });
  for (Operation *op : spent)
    op->erase();
}

LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top, float cycleTime,
                               llvm::StringMap<std::string> &interfaces,
                               MicroarchReport &report) {
  report.binding = binding.str();
  report.cycleTime = cycleTime;
  // Called directly (not via the pass manager), so load the dialects this
  // emits, the ones the pass declares as dependent, into the context.
  auto *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  // Storage and comb timing have no per-access carrier, so they thread into the
  // datapath builder as a library; an IP's timing rides the `dcp.operator` its
  // `dcp.compute` names, which stays live for the whole of emission.
  DeviceModel dev = DeviceModel::fromModule(module);

  auto scheduled = llvm::to_vector(module.getOps<dcp::DCPathModuleOp>());

  auto policy = bindingPolicyFor(binding);
  if (!policy) {
    error(Stage::Emit, Code::UnknownOption, module)
        << "Unknown binding policy '" << binding
        << "'; the policies are 'trivial', 'greedy-share' and 'planned'";
    return failure();
  }

  // Bottom-up over the call DAG: a container always finds its children already
  // registered.
  llvm::StringMap<dcp::DCPathModuleOp> byName;
  for (dcp::DCPathModuleOp f : scheduled)
    byName[f.getSymName()] = f;
  dcp::DCPathModuleOp topFunc = byName.lookup(top);
  if (!topFunc) {
    error(Stage::Emit, Code::TopFunctionMissing, module)
        << "Top function '" << top << "' is not a scheduled function";
    return failure();
  }

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  // Callee tables, keyed by symbol name: leaf kernels plus the containers
  // emitted so far, which compose exactly like a leaf.
  llvm::StringMap<hw::HWModuleOp> modules;
  llvm::StringMap<iface::ModuleInterface> ifaceModels;
  llvm::StringSet<> visited;

  auto registerModule = [&](StringRef name, hw::HWModuleOp mod,
                            iface::ModuleInterface model) {
    // The callee tables key on the func symbol, which a callsite names; the
    // manifest keys on the emitted module name, which the simulator names.
    interfaces[mod.getModuleName()] = model.toJSON();
    modules[name] = mod;
    ifaceModels[name] = std::move(model);
  };

  // Post-order over the call DAG, which is acyclic: the frontend rejects
  // recursion.
  auto emitOne = [&](auto &self, dcp::DCPathModuleOp f) -> LogicalResult {
    if (!visited.insert(f.getSymName()).second)
      return success(); // a shared callee already emitted
    // Children first: a `dcp.instance` is the only way a kernel reaches another
    // one, so it is the only edge to recurse on, and a leaf call misses
    // `byName`.
    WalkResult wr = f.walk([&](dcp::DCPathInstanceOp inv) -> WalkResult {
      auto it = byName.find(inv.getCallee());
      if (it != byName.end() && failed(self(self, it->second)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // One emission path, whichever way the function composes: leaf, sequential
    // container and dataflow differ only in the start policy they pick.
    bool hasInvoke = false;
    f.walk([&](dcp::DCPathInstanceOp) {
      hasInvoke = true;
      return WalkResult::interrupt();
    });
    uarch::CalleeCtx cc{modules, ifaceModels};
    const uarch::CalleeCtx *callees = hasInvoke ? &cc : nullptr;
    // Sealed on construction: the builder decides, and everything below reads.
    const Datapath dp(f, *policy, dev, cycleTime, callees,
                      /*isTop=*/f == topFunc);
    LLVM_DEBUG({
      llvm::dbgs() << "// datapath for @" << f.getSymName() << "\n";
      dp.dump(llvm::dbgs());
    });
    b.setInsertionPoint(f);
    auto pairOr = emitModule(f, dp, b, opModules, cycleTime, dev.operators,
                             report, callees);
    if (failed(pairOr))
      return failure();
    registerModule(f.getSymName(), pairOr->first, std::move(pairOr->second));
    return success();
  };

  if (failed(emitOne(emitOne, topFunc)))
    return failure();

  cleanupDcpOps(module);
  return success();
}

} // namespace mlir::allo::uarch
