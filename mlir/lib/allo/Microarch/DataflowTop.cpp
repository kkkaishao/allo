/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Dataflow composition -- the structural top (Route S).
//
// A container function whose scheduled body spawns concurrent processes (each
// an `await callee(...)` -> `func.call` carrying the `allo.async` attr) is NOT
// a datapath. It is lowered here to a thin *structural* `hw.module` that:
//
//   * instantiates each spawned process's already-emitted leaf `hw.module`
//     (EmitHW emits the leaves first; this file wires them);
//   * allocates a `seq.fifo` per internal channel (each `allo.stream.create`)
//   and
//     wires the producer's {data,valid,ready} handshake to the consumer's
//     through it (the FIFO's `full`/`empty` drive the shell
//     back-pressure/starvation signals the latency-insensitive leaves already
//     expose from P0);
//   * broadcasts the region `start` pulse to every process (fork);
//   * AND-reduces every process `done` into the region `done` (join).
//
// The container's own memref/scalar arguments are container *boundaries*: each
// is forwarded straight through to the one process that uses it, mirrored as a
// top port so cosim drives the composed design exactly as it would a single
// kernel. v1 scope (see drafts/dataflow-composition-design.md §12, P1):
// single-producer / single-consumer channels, counted processes, no feedback /
// fan-out / merge.
//===----------------------------------------------------------------------===//

#include "allo/IR/AlloOps.h" // StreamCreateOp, kAlloAsyncAttr
#include "allo/Microarch/HWEmitter.h"
#include "allo/Microarch/Interface.h" // iface field-name suffixes / helpers

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

bool isDataflowContainer(func::FuncOp func) {
  // A dataflow region is one whose body spawns concurrent processes -- a
  // `func.call` carrying the `allo.async` attr (an `await` in the frontend). A
  // leaf compute kernel has none; a plain (non-async) call is a sequential
  // compose, left on the normal datapath path (§7.1).
  bool found = false;
  func.walk([&](func::CallOp call) {
    if (call->hasAttr(kAlloAsyncAttr))
      found = true;
  });
  return found;
}

namespace {
using Dir = hw::ModulePort::Direction;

// A callee port's declared type + whether it is a module input.
struct PortDesc {
  Type type;
  bool isInput;
};

llvm::StringMap<PortDesc> portMap(hw::HWModuleOp mod) {
  llvm::StringMap<PortDesc> m;
  for (const hw::PortInfo &p : mod.getPortList())
    m[p.name.getValue()] = {p.type, p.dir == Dir::Input};
  return m;
}

// One spawned process instance.
struct Inst {
  func::CallOp call;
  hw::HWModuleOp mod;
  const iface::ModuleInterface *mi; // the callee's port model (arg <-> names)
  llvm::StringMap<PortDesc> ports;
  llvm::StringMap<Value> outs; // instance output values, by port name
};

// One internal FIFO channel (v1: single producer, single consumer).
struct Chan {
  Type payload;
  unsigned depth = 2;
  int prod = -1, cons = -1;       // producing / consuming instance index
  std::string prodBase, consBase; // stream port base on each side
};

// One container-boundary port to mirror onto the top: the top-side name
// (derived from the container argument, so distinct arguments never collide)
// and the instance-side name (the callee's own port), plus
// type/direction/owner.
struct Mirror {
  std::string topName, calleeName;
  Type type;
  bool isInput;
  unsigned inst;
};
} // namespace

LogicalResult
emitDataflowTop(func::FuncOp container,
                const llvm::StringMap<hw::HWModuleOp> &leaves,
                const llvm::StringMap<iface::ModuleInterface> &leafInterfaces,
                OpBuilder &b, std::string *jsonOut) {
  MLIRContext *ctx = b.getContext();
  Location loc = container.getLoc();
  Type i1 = b.getI1Type();

  // -- 1. Collect the spawned process instances (in program order). ----------
  SmallVector<Inst> insts;
  container.walk([&](func::CallOp call) {
    assert(
        call->hasAttr(kAlloAsyncAttr) &&
        "sequential compose inside a dataflow container is unsupported (v1)");
    hw::HWModuleOp mod = leaves.lookup(call.getCallee());
    auto it = leafInterfaces.find(call.getCallee());
    assert(mod && it != leafInterfaces.end() &&
           "spawned callee has no emitted leaf module / interface");
    insts.push_back({call, mod, &it->second, portMap(mod), {}});
  });

  // -- 2. Discover channels from stream.create -> {producer, consumer}. ------
  llvm::MapVector<Value, Chan> chans; // keyed by the stream.create result
  for (unsigned ii = 0; ii < insts.size(); ++ii) {
    Inst &in = insts[ii];
    for (const iface::FIFO &f : in.mi->streams) {
      Value stream = in.call.getOperand(f.arg);
      assert(stream.getDefiningOp<StreamCreateOp>() &&
             "a process stream arg must be an internal channel "
             "(v1: no top-level stream boundary)");
      Chan &c = chans[stream];
      c.depth = f.depth;
      c.payload = in.ports[f.data].type;
      if (f.isInput) {
        assert(c.cons < 0 && "channel has >1 consumer (v1 is SPSC)");
        c.cons = ii;
        c.consBase = f.base;
      } else {
        assert(c.prod < 0 && "channel has >1 producer (v1 is SPSC)");
        c.prod = ii;
        c.prodBase = f.base;
      }
    }
  }
  for (auto &kv : chans)
    assert(kv.second.prod >= 0 && kv.second.cons >= 0 &&
           "an SPSC channel needs exactly one producer and one consumer");

  // -- 3. Boundary ports: forward each container arg to the process using it. -
  // The top-side port name derives from the container argument (via its
  // NameLoc) so two arguments never collide even when the callees name their
  // params the same; the manifest records that top name for cosim. Direction
  // and width come straight from the callee port.
  SmallVector<Mirror> mirrors;
  llvm::DenseMap<int64_t, unsigned> argOwner; // boundary arg -> owning instance
  // The composed top's port interface (its JSON manifest): boundary scalars /
  // memories forwarded from the processes; streams are internal (no port) and
  // v1 has no scalar results.
  iface::ModuleInterface topIface;

  auto boundaryBase = [&](Inst &in, unsigned calleeArg,
                          StringRef role) -> std::string {
    auto arg = cast<BlockArgument>(in.call.getOperand(calleeArg));
    return cellName(arg.getLoc(), ("arg" + Twine(arg.getArgNumber())).str()) +
           "_" + role.str();
  };
  auto topArg = [&](Inst &in, unsigned calleeArg, unsigned ii) -> int64_t {
    auto arg = cast<BlockArgument>(in.call.getOperand(calleeArg));
    assert(arg.getOwner()->getParentOp() == container &&
           "a boundary operand must be a container argument");
    auto [it, fresh] = argOwner.try_emplace(arg.getArgNumber(), ii);
    assert((fresh || it->second == ii) &&
           "a container boundary shared by two processes is unsupported (v1)");
    return arg.getArgNumber();
  };
  auto mirror = [&](Inst &in, const std::string &calleeName,
                    const std::string &topName, unsigned ii) {
    PortDesc pd = in.ports[calleeName];
    mirrors.push_back({topName, calleeName, pd.type, pd.isInput, ii});
  };

  for (unsigned ii = 0; ii < insts.size(); ++ii) {
    Inst &in = insts[ii];
    for (const iface::Scalar &sc : in.mi->scalars) {
      std::string tn = cellName(
          cast<BlockArgument>(in.call.getOperand(sc.arg)).getLoc(), sc.name);
      mirror(in, sc.name, tn, ii);
      topIface.scalars.push_back({(int)topArg(in, sc.arg, ii), sc.width, tn});
    }
    // Forward each memory access: mirror its concrete callee field ports onto
    // the top (a top-side name derived from the container argument) and record
    // the top's own interface entry.
    auto memPorts = [&](const std::vector<std::vector<iface::Memory>> &accs,
                        StringRef role, bool write) {
      for (const auto &grp : accs)
        for (const iface::Memory &cm : grp) {
          std::string tbase = boundaryBase(in, cm.arg, role);
          mirror(in, cm.addr, iface::addr(tbase), ii);
          mirror(in, cm.data, iface::data_(tbase), ii);
          if (write)
            mirror(in, cm.we, iface::we(tbase), ii);
          iface::Memory mem{(int)topArg(in, cm.arg, ii),
                            write,
                            cm.bank,
                            cm.factor,
                            cm.width,
                            tbase,
                            iface::addr(tbase),
                            iface::data_(tbase),
                            write ? iface::we(tbase) : std::string()};
          (write ? topIface.writes : topIface.reads)
              .push_back({std::move(mem)});
        }
    };
    memPorts(in.mi->reads, "rd", /*write=*/false);
    memPorts(in.mi->writes, "wr", /*write=*/true);
    assert(in.mi->results.empty() &&
           "a process returning a scalar result is unsupported (v1)");
  }

  // -- 4. Declare the top ports (inputs contiguous at the front). ------------
  SmallVector<hw::PortInfo> ports;
  auto addPort = [&](StringRef n, Type t, Dir d) {
    ports.push_back(hw::PortInfo{{StringAttr::get(ctx, n), t, d}});
  };
  addPort("clk", i1, Dir::Input);
  addPort("rst", i1, Dir::Input);
  addPort("start", i1, Dir::Input);
  for (const Mirror &m : mirrors)
    if (m.isInput)
      addPort(m.topName, m.type, Dir::Input);
  addPort("done", i1, Dir::Output);
  for (const Mirror &m : mirrors)
    if (!m.isInput)
      addPort(m.topName, m.type, Dir::Output);

  // -- 5. Build the structural body. -----------------------------------------
  hw::HWModuleOp::create(
      b, loc, StringAttr::get(ctx, container.getSymName()),
      hw::ModulePortInfo(ports),
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        Value clkRaw = pa.getInput("clk");
        Value rst = pa.getInput("rst");
        Value start = pa.getInput("start");
        Value clk = seq::ToClockOp::create(ib, loc, clkRaw);
        Value tru = hw::ConstantOp::create(ib, loc, i1, 1);
        auto notv = [&](Value x) {
          return comb::XorOp::create(ib, loc, x, tru, false).getResult();
        };

        // Per-channel wires: the FIFO is built after the instances (which need
        // its status/data), so back the status/output with backedges.
        struct ChanWires {
          Backedge full, empty, dataOut;
          Value notFull, notEmpty;
        };
        llvm::MapVector<Value, ChanWires> cw;
        for (auto &kv : chans) {
          ChanWires w{
              bb.get(i1), bb.get(i1), bb.get(kv.second.payload), {}, {}};
          w.notFull = notv(w.full);
          w.notEmpty = notv(w.empty);
          cw[kv.first] = w;
        }

        // Instantiate each process.
        for (unsigned ii = 0; ii < insts.size(); ++ii) {
          Inst &in = insts[ii];
          llvm::StringMap<Value> ins;
          ins["clk"] = clkRaw;
          ins["rst"] = rst;
          ins["start"] = start;
          for (const iface::FIFO &f : in.mi->streams) {
            ChanWires &w = cw[in.call.getOperand(f.arg)];
            if (f.isInput) {
              ins[f.data] = w.dataOut;
              ins[f.valid] = w.notEmpty;
            } else {
              ins[f.ready] = w.notFull;
            }
          }
          for (const Mirror &m : mirrors)
            if (m.inst == ii && m.isInput)
              ins[m.calleeName] = pa.getInput(m.topName);

          SmallVector<Value> operands(in.mod.getNumInputPorts());
          for (const hw::PortInfo &p : in.mod.getPortList())
            if (p.dir == Dir::Input) {
              auto it = ins.find(p.name.getValue());
              assert(it != ins.end() && "unwired process input port");
              operands[p.argNum] = it->second;
            }
          auto inst = hw::InstanceOp::create(ib, loc, in.mod,
                                             in.call.getCallee(), operands);
          for (const hw::PortInfo &p : in.mod.getPortList())
            if (p.dir == Dir::Output)
              in.outs[p.name.getValue()] = inst.getResult(p.argNum);
        }

        // A FIFO per channel; resolve the channel backedges.
        for (auto &kv : chans) {
          Chan &c = kv.second;
          ChanWires &w = cw[kv.first];
          Value pData = insts[c.prod].outs[iface::data_(c.prodBase)];
          Value pValid = insts[c.prod].outs[iface::valid(c.prodBase)];
          Value cReady = insts[c.cons].outs[iface::ready(c.consBase)];
          Value wrEn = comb::AndOp::create(ib, loc, pValid, w.notFull, false);
          Value rdEn = comb::AndOp::create(ib, loc, cReady, w.notEmpty, false);
          auto fifo = seq::FIFOOp::create(
              ib, loc, c.payload, i1, i1, Type(), Type(), pData, rdEn, wrEn,
              clk, rst, ib.getI64IntegerAttr(c.depth), ib.getI64IntegerAttr(0),
              IntegerAttr(), IntegerAttr());
          w.dataOut.setValue(fifo.getOutput());
          w.full.setValue(fifo.getFull());
          w.empty.setValue(fifo.getEmpty());
        }

        // Fork/join: done = AND of every process `done` (each a latched level).
        Value done;
        for (Inst &in : insts) {
          Value d = in.outs["done"];
          done = done ? comb::AndOp::create(ib, loc, done, d, false).getResult()
                      : d;
        }
        pa.setOutput("done", done ? done : tru);
        for (const Mirror &m : mirrors)
          if (!m.isInput)
            pa.setOutput(m.topName, insts[m.inst].outs[m.calleeName]);
      });

  // -- 6. Hand back the composed top's port-interface JSON (the cosim
  // manifest); no IR attribute is attached -- the model is the single
  // representation. ----
  if (jsonOut)
    *jsonOut = topIface.toJSON();
  return success();
}

} // namespace mlir::allo::uarch
