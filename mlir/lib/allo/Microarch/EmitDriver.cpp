/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/EmitDriver.h"

#include "allo/IR/AlloOps.h"
#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/HWEmitter.h" // HWEmitter
#include "allo/Microarch/Interface.h"
#include "allo/Microarch/Verify.h" // validateDatapath
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Support/Logging.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
// Native (comb) units emit inline, no extern. One input port per operand (named
// `a`, `b`, `c`, ... at each operand's width: a unary cast/`sqrt` gets `a`
// only, a binary op `a`+`b`, a compare two operands yielding i1) then the
// output at the result width. The interface follows the realization's stall
// contract:
// `(a.., clk) -> y` free-running, or `(a.., clk, ce) -> y` when clock-enabled
// (`ce == 0` freezes the pipe in lockstep with the shell). Both are a function
// of `impl`, so every instance of a module name shares one port shape, which is
// what makes the dedup safe. Returns unit id -> its extern module.
static DenseMap<unsigned, Operation *>
declareOperatorModules(func::FuncOp func, const uarch::Datapath &dp,
                       OpBuilder &b, llvm::StringMap<Operation *> &opModules,
                       std::vector<iface::Operator> &declared) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  DenseMap<unsigned, Operation *> unitModule;
  llvm::StringSet<> listed; // one manifest entry per module, not per unit
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.comb || u.boundOps.empty())
      continue;
    Operation *srcOp = u.repOp();
    assert(u.inputs.size() == srcOp->getNumOperands() &&
           "IP unit input count must match its bound op's operand count");
    IntegerType outW = hwType(u.resultType, b);
    std::string modName = operatorModuleName(u);
    // The port shape is a function of the realization, so every instance of a
    // module name shares it: build the manifest entry alongside the ports.
    iface::Operator entry{modName, u.impl, operatorPredicate(u), {}};
    SmallVector<PortInfo> ep;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      IntegerType w = hwType(srcOp->getOperand(k).getType(), b);
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
    if (listed.insert(modName).second)
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
  // `hwType`/`memElemType` for the data ports.
  auto iType = [&](unsigned w) -> Type { return b.getIntegerType(w); };
  SmallVector<PortInfo> ports;
  // The port names are the manifest, authored before CIRCT's LegalizeNames
  // runs, so a name ExportVerilog would rewrite or uniquify desyncs cosim from
  // the Verilog. `verilogName` prevents that; these check the composed result.
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
  // Stream FIFO ports, input side: module inputs must stay contiguous at the
  // front (HWModulePortAccessor maps body args to the first `numInputs` ports
  // positionally), so {data, valid} / {ready} go here; outputs follow `done`.
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
  port(kDone, i1, Dir::Output);
  // Stream FIFO ports, output side (after `done`, among the module outputs): an
  // input stream's back-pressure {ready}; an output stream's {data, valid}.
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

// Emit an hw.module for one scheduled function's datapath. Returns failure with
// a diagnostic if the datapath is outside the supported subset
// (validateDatapath). `opModules` caches extern operator modules across
// functions.
static FailureOr<std::pair<hw::HWModuleOp, iface::ModuleInterface>>
emitModule(func::FuncOp func, uarch::Datapath &dp, OpBuilder &b,
           llvm::StringMap<Operation *> &opModules,
           const uarch::CalleeCtx *callees = nullptr) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  if (failed(validateDatapath(func, dp)))
    return failure();

  Type i1 = b.getI1Type();
  Type i32 = b.getIntegerType(32);

  // The single source for every boundary port name, shared by declaration,
  // manifest and cosim harness; it also carries the extern operator modules
  // this kernel instantiates. Its port lists come enumerated from the builder.
  iface::ModuleInterface model(dp);
  auto unitModule =
      declareOperatorModules(func, dp, b, opModules, model.operators);
  auto ports = declareModulePorts(model, b);

  hw::ModulePortInfo portInfo(ports);
  // Legalized here rather than left to ExportVerilog, so the key the manifest
  // uses is the emitted Verilog module name. A nested callee `top.child` would
  // otherwise be rewritten downstream.
  model.symbol = func.getSymName().str();
  model.module = verilogName(model.symbol);
  StringAttr modName = StringAttr::get(ctx, model.module);

  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, unitModule, bb, i1, i32, callees);
        e.ctx.clk = e.ctx.R(seq::ToClockOp::create(ib, loc, pa.getInput(kClk)));
        e.ctx.clkRaw = pa.getInput(kClk);
        e.ctx.rst = pa.getInput(kRst);
        e.emit();
      });

  // Hand the port model back to the caller: it derives the cosim manifest
  // JSON and, for a dataflow container, threads the leaf models into the
  // structural-top emitter, keeping one in-memory port representation.
  return std::make_pair(hwMod, std::move(model));
}

static bool hasDCPRegions(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// The IP operators' timing lives on module-level `dcp.operator` symbols. Fold
// each onto its referencing `dcp.compute` (its `latency` + `pipelined`) so the
// datapath reads timing locally, then drop the spent `dcp.operator` /
// `dcp.device` declarations. This lets each extern operator module share the
// operator's `sym_name` (the RTL module name) with no symbol clash: there is
// no live same-named symbol once the declarations are gone. Runs on the emit
// clone, so the canonical scheduled module keeps the normalized form.
static void stampOperatorTiming(ModuleOp module) {
  Builder bd(module.getContext());
  module.walk([&](dcp::DCPathComputeOp comp) {
    FlatSymbolRefAttr sym = comp.getOpTypeAttr();
    if (!sym)
      return; // a combinational compute (comb_kind); no operator timing
    auto opr =
        SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(comp, sym);
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    comp->setAttr("latency", bd.getI64IntegerAttr(opr.getLatency()));
    comp->setAttr("pipelined", bd.getBoolAttr(opr.getPipelined()));
    comp->setAttr("stall",
                  StallContractEnumAttr::get(bd.getContext(), opr.getStall()));
  });
  SmallVector<Operation *> spent;
  module.walk([&](Operation *op) {
    if (isa<dcp::DCPathOperatorOp, dcp::DCPathDeviceOp>(op))
      spent.push_back(op);
  });
  for (Operation *op : spent)
    op->erase();
}

LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top,
                               llvm::StringMap<std::string> &interfaces) {
  // Called directly (not via the pass manager), so load the dialects this
  // emits, the ones the pass declares as dependent, into the context.
  auto *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  // The device's storage timing is read BEFORE `stampOperatorTiming` drops
  // `dcp.device`. Compute timing folds onto each `dcp.compute`, but memory
  // latency has no such carrier, so it threads into the datapath builder.
  MemoryLibrary memLib = OperatorLibrary::fromModule(module).memoryLibrary();

  // Fold operator timing onto the compute ops and drop the declarations, before
  // any datapath is built or an extern operator module is named.
  stampOperatorTiming(module);

  SmallVector<func::FuncOp> scheduled;
  module.walk([&](func::FuncOp f) {
    if (hasDCPRegions(f))
      scheduled.push_back(f);
  });

  // A memref result is unsupported: hardware writes output through a memory
  // port, and the prepass that would rewrite it is deliberately not run.
  // Reject cleanly here rather than deep in the builder.
  for (func::FuncOp f : scheduled)
    if (auto ret = dyn_cast<func::ReturnOp>(f.front().getTerminator()))
      for (Value v : ret.getOperands())
        if (isa<MemRefType>(v.getType())) {
          logging::error(logging::Stage::Emit, f)
              << "Returning a memref is unsupported; write the result through "
                 "an output argument (out-parameter) instead";
          return failure();
        }

  auto policy = bindingPolicyFor(binding);
  if (!policy)
    return module.emitError("allo-datapath-to-hw: unknown binding policy '")
           << binding << "'";

  // Emission is rooted at the top function and runs bottom-up over the call
  // DAG: each callee emits before its caller, so a container always finds
  // its children already registered. Mirrors the scheduler's traversal order.
  llvm::StringMap<func::FuncOp> byName;
  for (func::FuncOp f : scheduled)
    byName[f.getSymName()] = f;
  func::FuncOp topFunc = byName.lookup(top);
  if (!topFunc)
    return module.emitError("allo-datapath-to-hw: top function '")
           << top << "' is not a scheduled function";

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  // Callee tables, keyed by symbol name: leaf kernels plus containers
  // emitted so far. A container composes exactly like a leaf, so both
  // live here.
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

  // Post-order over the call DAG (acyclic; the frontend rejects recursion),
  // via a self-parameter recursive lambda (`self(self, ...)`).
  auto emitOne = [&](auto &self, func::FuncOp f) -> LogicalResult {
    if (!visited.insert(f.getSymName()).second)
      return success(); // a shared callee already emitted
    // Children first: emit every scheduled callee (a leaf call misses
    // `byName`). A callee is referenced by a `func.call` or a `dcp.instance`
    // (a leaf CallUnit); both must recurse before their caller emits.
    WalkResult wr = f.walk([&](Operation *op) -> WalkResult {
      StringRef callee;
      if (auto call = dyn_cast<func::CallOp>(op))
        callee = call.getCallee();
      else if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op))
        callee = inv.getCallee();
      else
        return WalkResult::advance();
      auto it = byName.find(callee);
      if (it != byName.end() && failed(self(self, it->second)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // ONE emission path, whichever way the function composes: leaf, sequential
    // container and dataflow differ only in composition class (the start policy
    // it picks). Every child is a `dcp.instance` CallUnit in one `Datapath`.
    bool hasInvoke = false;
    f.walk([&](dcp::DCPathInstanceOp) {
      hasInvoke = true;
      return WalkResult::interrupt();
    });
    uarch::CalleeCtx cc{modules, ifaceModels};
    const uarch::CalleeCtx *callees = hasInvoke ? &cc : nullptr;
    Datapath dp(f, *policy, memLib, callees);
    LLVM_DEBUG({
      llvm::dbgs() << "// datapath for @" << f.getSymName() << "\n";
      dp.dump(llvm::dbgs());
    });
    b.setInsertionPoint(f);
    auto pairOr = emitModule(f, dp, b, opModules, callees);
    if (failed(pairOr))
      return failure();
    registerModule(f.getSymName(), pairOr->first, std::move(pairOr->second));
    return success();
  };

  if (failed(emitOne(emitOne, topFunc)))
    return failure();

  // cleanup non-hw ops to avoid Verilog export errors
  for (func::FuncOp f : scheduled)
    f.erase();
  for (memref::GlobalOp g :
       llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
    g.erase();
  return success();
}

} // namespace mlir::allo::uarch
