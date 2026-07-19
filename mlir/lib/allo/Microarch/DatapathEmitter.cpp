/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/Microarch/Interface.h" // iface field-name helpers

#include "allo/Scheduling/MemoryModel.h" // partitionOf (crossbar shape check)
#include "allo/Scheduling/OperatorLibrary.h" // isNativeImpl

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// --- Memory-banking crossbar primitives (see HWEmitter.h) ------------------

BankSplit splitBank(EmitContext &c, Value addr, unsigned factor) {
  assert(llvm::isPowerOf2_64(factor) &&
         "cyclic bank factor must be a power of 2");
  Value bank = c.R(
      comb::AndOp::create(c.b, c.loc, addr, c.konst(c.i32, factor - 1), false));
  Value offset = c.R(comb::ShrUOp::create(
      c.b, c.loc, addr, c.konst(c.i32, llvm::Log2_64(factor)), false));
  return {bank, offset};
}

Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank) {
  Value out = bankValues[0]; // bank 0 falls through the priority chain
  for (unsigned k = 1; k < bankValues.size(); ++k)
    out = c.mux(c.icmpEq(bank, k), bankValues[k], out);
  return out;
}

Value bankWe(EmitContext &c, Value we, Value bank, unsigned k) {
  return c.R(comb::AndOp::create(c.b, c.loc, we, c.icmpEq(bank, k), false));
}

ExternalBanking externalBank(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  if (m.numBanks == 1)
    return {1u, 0u};
  PartitionInfo p = partitionOf(m.memref);
  assert(p.cyclicAxes.size() == 1 && !p.hasBlock &&
         cast<MemRefType>(m.memref.getType()).getRank() == 1 &&
         llvm::isPowerOf2_64(m.numBanks) &&
         "external banking: only 1-D power-of-two cyclic supported");
  auto [dim, factor] = p.cyclicAxes[0];
  ExternalBanking eb;
  eb.factor = static_cast<unsigned>(factor);
  // A statically-banked access routes to one interface; a data-dependent one
  // (empty bank) crosses over all bank interfaces.
  if (std::optional<int64_t> b = staticBank(acc.addrMap, dim, factor))
    eb.bank = static_cast<unsigned>(*b);
  return eb;
}

// Resolve a datapath Source to the SSA value driving it. Exhaustive over
// Source::Kind: the switch is the single extension point for new source kinds
// (muxes in the binding phase).
Value DatapathEmitter::resolveSource(const uarch::Source &s) {
  switch (s.kind) {
  case uarch::Source::Kind::Unit:
    return unitVal.lookup(s.id);
  case uarch::Source::Kind::Reg:
    return regStages[s.id].tap(s.outPort);
  case uarch::Source::Kind::Mem:
    return readData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Stream:
    // An input stream's loaded token: the `_data` module port, bound by
    // bindStreamReads before any consumer (like a memory read).
    return streamReadData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Counter: {
    // The iteration counter of Source's region (an outer container's counter is
    // live while its nested region emits).
    Value cv = controlOf.lookup(s.id).counter;
    assert(cv && "counter source with no emitted region counter");
    return cv;
  }
  case uarch::Source::Kind::Const: {
    // The datapath carries a value as its bit pattern, so a float literal ties
    // in as its bitcast integer (a float constant reaching a Source is e.g. a
    // reduction identity `0.0`).
    IntegerType t = hwType(dp.consts[s.id].type, c.b);
    Attribute v = dp.consts[s.id].value;
    if (auto ia = dyn_cast<IntegerAttr>(v))
      return c.konst(t, ia.getInt());
    return c.konst(
        t, cast<FloatAttr>(v).getValue().bitcastToAPInt().getZExtValue());
  }
  case uarch::Source::Kind::IO:
    // A scalar kernel argument, exposed as its own module input port.
    return pa.getInput(scalarPortName(dp.ios[s.id]));
  case uarch::Source::Kind::Mux: {
    // A shared unit's input: drive each source on the cycle its op consumes it.
    // The selects are mutually exclusive (disjoint residues, MRT-verified), so
    // a priority chain suffices -- source 0 is the default (its own cycle sees
    // no other select high). Each select is the op's `activationPulse`, the
    // same signal a store's write-enable uses.
    if (Value v = muxVal.lookup(s.id))
      return v;
    const uarch::Mux &mx = dp.muxes[s.id];
    Value issue = controlOf.lookup(mx.region).issue;
    assert(issue && "mux in a region with no controller");
    Value v = resolveSource(mx.sources[0]);
    for (unsigned i = 1; i < mx.sources.size(); ++i) {
      Value sel = c.activationPulse(issue, mx.selectOps[i]);
      v = c.mux(sel, resolveSource(mx.sources[i]), v);
    }
    muxVal[s.id] = v;
    return v;
  }
  case uarch::Source::Kind::Survivor: {
    // A sibling region's held result: the orchestrator latched it (setSurvivor)
    // when the producing region completed, before this consumer emitted.
    Value sv = survivorOf.lookup(accKey(s.id, s.outPort));
    assert(sv && "survivor source read before its region was captured");
    return sv;
  }
  case uarch::Source::Kind::Call: {
    // A sub-kernel call's scalar result: the child instance's result output,
    // populated by emitCalls before any consumer (captureResults latches it
    // into this region's survivor; a same-region later child reads it live).
    Value cv = callResultVal.lookup(accKey(s.id, s.outPort));
    assert(cv && "call result source read before its CallUnit was emitted");
    return cv;
  }
  case uarch::Source::Kind::None:
    assert(false && "unresolved (None) source");
    return {};
  }
  llvm_unreachable("unhandled Source::Kind");
}

unsigned DatapathEmitter::readyCycle(const uarch::Source &s) const {
  // Map the Source to its producing op, then defer to the one ready-cycle
  // definition (readyCycleOf). A Const is at-issue (no op, cycle 0).
  switch (s.kind) {
  case uarch::Source::Kind::Unit:
    return readyCycleOf(dp.units[s.id].boundOps.front().first);
  case uarch::Source::Kind::Mem:
    return readyCycleOf(dp.mems[s.id].accesses[s.outPort].op);
  case uarch::Source::Kind::Stream:
    return readyCycleOf(dp.streams[s.id].accesses[s.outPort].op);
  case uarch::Source::Kind::Const:
    return 0;
  case uarch::Source::Kind::Call: {
    // A determinate call's scalar result lands at its region-relative issue +
    // the callee's whole-kernel latency (its start->done depth). Indeterminate
    // calls carry no latency and are guarded before emit.
    const uarch::CallUnit &cu = dp.calls[s.id];
    assert(cu.latency && "readyCycle of an indeterminate call result");
    return cu.start + static_cast<unsigned>(*cu.latency);
  }
  default:
    assert(false && "readyCycle only modelled for a Unit / memory read / "
                    "stream get / constant / call result");
    return 0;
  }
}

// Evaluate an affine index expression to an i32 hw value, emitting comb ops.
// `idx` holds the resolved value of each map operand (dims then symbols).
// Affine index arithmetic degenerates to adds and multiply-by-constant; a
// genuine multiplier survives only a non-reducible map, and identity addressing
// emits nothing. Shared by the two places a map reaches the datapath: a memory
// access's address (computeAddr) and a standalone affine.apply (emitCompute).
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims) {
  auto konst = [&](int64_t v) {
    return hw::ConstantOp::create(b, loc, b.getIntegerType(32), v).getResult();
  };
  if (auto cst = dyn_cast<AffineConstantExpr>(e))
    return konst(cst.getValue());
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return idx[d.getPosition()];
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return idx[numDims + sym.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Value lhs = evalAffine(b, loc, bin.getLHS(), idx, numDims);
  Value rhs = evalAffine(b, loc, bin.getRHS(), idx, numDims);
  if (e.getKind() == AffineExprKind::Add)
    return comb::AddOp::create(b, loc, lhs, rhs, false).getResult();
  if (e.getKind() == AffineExprKind::Mul)
    return comb::MulOp::create(b, loc, lhs, rhs, false).getResult();
  // floordiv / mod by a constant are the delinearization a coalesced nest
  // leaves behind (`iv floordiv N`, `iv mod N`) over a non-negative index. A
  // power-of-two divisor is a shift / bit-mask (no divider); any other constant
  // is a general unsigned divide / remainder -- synthesis folds a constant
  // divisor to a multiply-shift, so no runtime divider is instantiated.
  auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS());
  assert(rc && rc.getValue() > 0 &&
         "affine div/mod by a non-constant or non-positive divisor");
  int64_t f = rc.getValue();
  bool pow2 = llvm::isPowerOf2_64(f);
  if (e.getKind() == AffineExprKind::FloorDiv)
    return pow2
               ? comb::ShrUOp::create(b, loc, lhs, konst(llvm::Log2_64(f)),
                                      false)
                     .getResult()
               : comb::DivUOp::create(b, loc, lhs, konst(f), false).getResult();
  assert(e.getKind() == AffineExprKind::Mod && "unexpected affine op");
  return pow2
             ? comb::AndOp::create(b, loc, lhs, konst(f - 1), false).getResult()
             : comb::ModUOp::create(b, loc, lhs, konst(f), false).getResult();
}

// The linear element address of a memory access: evaluate its affine map over
// the (already stage-delayed) index sources, then linearize by the memref's
// row-major strides. Single-index identity addressing emits no ops.
Value DatapathEmitter::computeAddr(const uarch::MemUnit &m,
                                   const uarch::MemUnit::Access &acc) {
  SmallVector<Value> idx;
  for (const uarch::Source &s : acc.addr)
    idx.push_back(resolveSource(s));
  AffineMap map = acc.addrMap;
  assert(map && "dcp memory access without an affine map");
  ArrayRef<int64_t> shape = cast<MemRefType>(m.memref.getType()).getShape();
  unsigned rank = map.getNumResults();
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * shape[k + 1];
  Value addr;
  for (unsigned k = 0; k < rank; ++k) {
    Value term =
        evalAffine(c.b, c.loc, map.getResult(k), idx, map.getNumDims());
    if (stride[k] != 1)
      term = c.R(comb::MulOp::create(c.b, c.loc, term,
                                     c.konst(c.i32, stride[k]), false));
    addr =
        k == 0 ? term : c.R(comb::AddOp::create(c.b, c.loc, addr, term, false));
  }
  return addr;
}

// Narrow a linear element address to a memory's clog2(depth)-bit index, the
// width seq.hlmem addressing expects. A single-element buffer (clog2 == 0)
// takes the register spill path, not an hlmem.
Value DatapathEmitter::memAddr(const uarch::MemUnit &m, Value addr) {
  unsigned w = llvm::Log2_64_Ceil(m.depthWords);
  assert(w > 0 && "single-element internal memory should spill to a register");
  return c.R(
      comb::ExtractOp::create(c.b, c.loc, c.b.getIntegerType(w), addr, 0));
}

// Bind the read-data input ports into readData, once, before the per-region
// loop (external memories only; internal ones read via seq.read below). A
// data-dependent banked read has one data port per bank and is muxed in-region
// by emitExternalReads, so it is bound there, not here.
void DatapathEmitter::bindReadPorts() {
  for (unsigned i = 0; i < reads.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[reads[i].mem];
    ExternalBanking eb = externalBank(m, m.accesses[reads[i].idx]);
    if (eb.factor > 1 && !eb.bank)
      continue; // data-dependent: bound by emitExternalReads
    readData[accKey(reads[i].mem, reads[i].idx)] =
        pa.getInput(iface::data_(extPorts(dp, reads, i, "rd").front().second));
  }
}

// Instantiate on-chip storage for each internal (non-argument) memory: one
// seq.hlmem, or -- when the array reached emit still partitioned (a
// data-dependent bank dcp-resolve-banking could not split statically) -- one
// per bank, addressed through the crossbar (splitBank/readCrossbar/bankWe). The
// handles are module-scope so writes and reads in different regions share them.
void DatapathEmitter::createInternalMemories() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    if (m.numBanks > 1) {
      // The emitter crossbar handles a 1-D power-of-two cyclic partition; block
      // / multi-dim / external banking are not supported.
      PartitionInfo p = partitionOf(m.memref);
      assert(
          p.cyclicAxes.size() == 1 && !p.hasBlock &&
          cast<MemRefType>(m.memref.getType()).getRank() == 1 &&
          llvm::isPowerOf2_64(m.numBanks) &&
          "emitter banking crossbar: only 1-D power-of-two cyclic supported");
    }
    // Name the on-chip storage after the frontend buffer (its memref NameLoc,
    // e.g. "buf"); fall back to m<id> for an unnamed internal memory. A bank
    // appends its index (buf_b0, buf_b1, ...).
    std::string base = cellName(m.memref.getLoc(), ("m" + Twine(m.id)).str());
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k) {
      std::string name =
          m.numBanks > 1 ? base + "_b" + std::to_string(k) : base;
      auto mem = seq::HLMemOp::create(c.b, c.loc, c.clk, c.rst, name,
                                      {static_cast<int64_t>(m.depthWords)},
                                      memElemType(m, c.b));
      banks.push_back(mem.getHandle());
    }
    memBanks[m.id] = std::move(banks);
  }
}

// Shift-register chains for region \p rb's registers (index delays, pipeline
// holds). Each chain's head input is a backedge resolved once the units exist.
void DatapathEmitter::emitRegisters(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs) {
    const uarch::Register &rg = dp.regs[rid];
    Backedge head = c.bb.get(hwType(rg.type, c.b));
    regHeadBE.try_emplace(rg.id, head);
    // A register is a plain delay chain -- reduction-identity re-injection
    // rides the consuming unit's recurrence input (emitUnits), not the
    // register.
    regStages[rg.id] = c.shiftChain(head, rg.depth);
    // Name each held stage after the value it delays (its NameLoc); an IV-delay
    // register whose value lost its name stays unnamed (best-effort).
    for (Value stage : regStages[rg.id].stages)
      nameValue(stage, rg.value.getLoc());
  }
}

// seq.read for each internal-memory read scheduled in region \p rb, bound into
// readData *before* emitUnits consumes it (a Source::Mem input). Read latency
// follows the storage impl (register: comb; ram: 1-cycle, as scheduled).
void DatapathEmitter::emitInternalReads(const uarch::RegionBlock &rb) {
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    ArrayRef<Value> banks = memBanks[m.id];
    unsigned lat = memReadLatency(m.impl);
    for (unsigned a = 0; a < m.accesses.size(); ++a) {
      const uarch::MemUnit::Access &acc = m.accesses[a];
      if (acc.isWrite || acc.region != rb.id)
        continue;
      Value flat = computeAddr(m, acc);
      Value rd;
      if (banks.size() == 1) {
        rd = c.R(seq::ReadPortOp::create(c.b, c.loc, banks[0],
                                         ValueRange{memAddr(m, flat)},
                                         /*rdEn=*/Value(), lat));
      } else {
        // Read every bank at the (bank-independent) offset, then select by the
        // runtime bank -- aligned with the read data (delayed by the latency).
        BankSplit bs = splitBank(c, flat, m.numBanks);
        Value addr = memAddr(m, bs.offset);
        SmallVector<Value> vals;
        for (Value h : banks)
          vals.push_back(c.R(seq::ReadPortOp::create(
              c.b, c.loc, h, ValueRange{addr}, /*rdEn=*/Value(), lat)));
        Value sel = lat ? c.shiftChain(bs.bank, lat).last() : bs.bank;
        rd = readCrossbar(c, vals, sel);
      }
      readData[accKey(m.id, a)] = rd;
    }
  }
}

// Read crossbar for each data-dependent external (argument) read in region
// \p rb: drive every bank interface's address with the in-bank offset, read
// each bank's data port, and mux by the runtime bank -- delayed to the read
// latency so the select aligns with the (1-cycle) memory data. Bound into
// readData before emitUnits, the twin of emitInternalReads for boundary ports
// instead of hlmems.
void DatapathEmitter::emitExternalReads(const uarch::RegionBlock &rb) {
  for (unsigned i = 0; i < reads.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[reads[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[reads[i].idx];
    ExternalBanking eb = externalBank(m, acc);
    if (acc.region != rb.id || eb.factor == 1 || eb.bank)
      continue; // only data-dependent banked reads
    BankSplit bs = splitBank(c, computeAddr(m, acc), eb.factor);
    SmallVector<Value> vals;
    for (const auto &[bank, base] : extPorts(dp, reads, i, "rd")) {
      pa.setOutput(iface::addr(base), bs.offset);
      vals.push_back(pa.getInput(iface::data_(base)));
    }
    unsigned lat = memReadLatency(m.impl);
    Value sel = lat ? c.shiftChain(bs.bank, lat).last() : bs.bank;
    readData[accKey(reads[i].mem, reads[i].idx)] = readCrossbar(c, vals, sel);
  }
}

// Compute units of region \p rb: native -> comb; IP -> an instance of the
// extern operator module (internally pipelined by its latency).
void DatapathEmitter::emitUnits(const uarch::RegionBlock &rb) {
  // Backedge every unit output before wiring, so an input may reference a unit
  // emitted later: the widened-reduction idiom reads the accumulator as a
  // depth-0 loop-carry from a later unit, and a fused recurrence reads its own
  // output. A register elsewhere in the recurrence cycle keeps the hardware
  // acyclic -- the backedges only free emission from topological order.
  DenseMap<unsigned, Backedge> outBE;
  for (uarch::UnitId uid : rb.units) {
    Backedge b = c.bb.get(hwType(dp.units[uid].resultType, c.b));
    outBE[uid] = b;
    unitVal[uid] = b;
  }
  for (uarch::UnitId uid : rb.units) {
    const uarch::FuncUnit &u = dp.units[uid];
    SmallVector<Value> operands;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      Value v =
          resolveSource(u.inputs[k]); // a self-reference reads its own backedge
      // Re-inject the reduction identity at a recurrence input -- the port
      // reading a loop-carried iter_arg -- on the first iteration, so a
      // retriggered reduction restarts from the identity. Gate = the
      // first-iteration issue pulse (`issue & counter == lb`) delayed to this
      // op's stage, the cycle it consumes the first iteration. The counter
      // holds the real IV, so the first iteration is `iv == lb` (== 0 for a
      // lb=0 loop). One gate for both regimes: free-running (counter = cycle)
      // and modulo (counter advances at the issue cycle, so `counter == lb`
      // alone is stale by the op's stage). The recurrence's register, if any,
      // is a plain delay -- the init rides the input, since the widened idiom
      // reads acc through a bare wire, not a tap.
      if (u.inputInits[k].kind != uarch::Source::Kind::None) {
        const RegionControl rc = controlOf.lookup(rb.id);
        Value iv = rc.counter, issue = rc.issue;
        assert(iv && issue &&
               "recurrence input in a region with no controller");
        // The first iteration is `iv == lb`; the lb is a runtime Source (a
        // data-dependent range start) or the constant fast path.
        Value lb =
            rb.lbSource ? resolveSource(rb.lbSource) : c.konst(c.i32, rb.lb);
        Value iter0 = c.R(
            comb::AndOp::create(c.b, c.loc, issue, c.icmpEqV(iv, lb), false));
        Value gate = c.activationPulse(iter0, u.boundOps.front().first);
        v = c.mux(gate, resolveSource(u.inputInits[k]), v);
      }
      operands.push_back(v);
    }

    Value result;
    if (allo::isNativeImpl(u.impl)) {
      result = emitCompute(c.b, c.loc, u.opType, operands,
                           hwType(u.resultType, c.b), u.boundOps.front().first);
    } else {
      // An IP operator instance takes its data operands, then the clock, then
      // (for a clock-enabled contract) a `ce` freeze bit. `ce` rides the
      // region's clock-enable so the IP pipeline freezes in lockstep with the
      // shell's shift chains under back-pressure; outside a stream region
      // `regionEnable` is null, so `ce` is a constant 1 (free-running).
      operands.push_back(c.clkRaw);
      if (allo::stallContract(u.impl) == allo::StallContract::ClockEnable)
        operands.push_back(c.regionEnable ? c.regionEnable : c.t1);
      result = hw::InstanceOp::create(c.b, c.loc, unitModule.lookup(u.id),
                                      ("u" + Twine(u.id)).str(), operands)
                   ->getResult(0);
    }
    outBE[uid].setValue(result);
    unitVal[u.id] = result;
    // Name the result wire after the frontend variable this op computes (the
    // dcp op carries the assignment-target NameLoc, e.g. "acc").
    nameValue(result, u.boundOps.front().first->getLoc());
  }
}

// A container's own combinational units: its continue-condition (a
// sequential-wrapper while) or a child guard's predicate, reified
// into start-0 `dcp.compute`s bound in the container. Unlike `emitUnits` there
// is no reduction-identity re-injection -- these read the container counter /
// iter-arg survivors / constants, never a loop-carried accumulator -- so no
// issue pulse is needed (a container has none). Emitted after the counter and
// survivors are set and before the children are sequenced, so a child guard
// resolves its parent-emitted predicate (Source::Unit). Backedges let the tree
// wire in any order, exactly as `emitUnits` does.
void DatapathEmitter::emitCombUnits(const uarch::RegionBlock &rb) {
  DenseMap<unsigned, Backedge> outBE;
  for (uarch::UnitId uid : rb.units) {
    Backedge be = c.bb.get(hwType(dp.units[uid].resultType, c.b));
    outBE[uid] = be;
    unitVal[uid] = be;
  }
  for (uarch::UnitId uid : rb.units) {
    const uarch::FuncUnit &u = dp.units[uid];
    assert(llvm::all_of(u.inputInits,
                        [](const uarch::Source &s) {
                          return s.kind == uarch::Source::Kind::None;
                        }) &&
           "a container's combinational unit carries no recurrence init");
    assert(allo::isNativeImpl(u.impl) &&
           "a container condition/predicate must be a native (comb) unit");
    SmallVector<Value> operands;
    for (const uarch::Source &in : u.inputs)
      operands.push_back(resolveSource(in));
    Value result =
        emitCompute(c.b, c.loc, u.opType, operands, hwType(u.resultType, c.b),
                    u.boundOps.front().first);
    outBE[uid].setValue(result);
    unitVal[u.id] = result;
    nameValue(result, u.boundOps.front().first->getLoc());
  }
}

// Resolve region \p rb's register head inputs now that its units exist.
void DatapathEmitter::resolveRegHeads(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs)
    regHeadBE.find(rid)->second.setValue(resolveSource(dp.regs[rid].input));
}

// Read/write address + data outputs of the accesses scheduled in region \p rb,
// driven by that region's controller (counter / \p issue). Returns the region's
// store feedback: `storeDrain`, the deepest store's schedule stage (max schedT
// over its stores), which the region's `done` waits on.
DatapathFeedback DatapathEmitter::emitAccesses(const uarch::RegionBlock &rb,
                                               Value issue) {
  unsigned ridx = rb.id;
  // Address an external port: the in-bank offset for a partitioned argument
  // (the boundary presents one interface per bank), else the flat element
  // index.
  auto extAddr = [&](const uarch::MemUnit &m,
                     const uarch::MemUnit::Access &acc) -> Value {
    Value flat = computeAddr(m, acc);
    unsigned factor = externalBank(m, acc).factor;
    return factor > 1 ? splitBank(c, flat, factor).offset : flat;
  };
  for (unsigned i = 0; i < reads.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[reads[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[reads[i].idx];
    ExternalBanking eb = externalBank(m, acc);
    // A data-dependent read's addresses + crossbar are emitted in
    // emitExternalReads (before emitUnits); here handle the single-interface
    // case (unbanked or statically banked).
    if (acc.region == ridx && (eb.factor == 1 || eb.bank))
      pa.setOutput(iface::addr(memPortBase(dp, reads, i, "rd")),
                   extAddr(m, acc));
  }
  DatapathFeedback fb;
  for (unsigned i = 0; i < writes.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[writes[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[writes[i].idx];
    if (acc.region != ridx)
      continue;
    Value we = c.activationPulse(issue, acc.op);
    Value addr = extAddr(m, acc), data = resolveSource(acc.data);
    ExternalBanking eb = externalBank(m, acc);
    // A data-dependent write drives every bank interface; its runtime bank
    // gates each interface's write-enable so only the target bank commits (an
    // N-way demux). A static / unbanked write is a single interface.
    Value dynBank =
        eb.bank ? Value() : splitBank(c, computeAddr(m, acc), eb.factor).bank;
    for (const auto &[bank, base] : extPorts(dp, writes, i, "wr")) {
      pa.setOutput(iface::addr(base), addr);
      pa.setOutput(iface::data_(base), data);
      pa.setOutput(iface::we(base),
                   dynBank ? bankWe(c, we, dynBank, bank) : we);
    }
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, schedT(acc.op));
  }
  // Internal-memory writes drive seq.write (registered, latency 1) instead of
  // module ports. They still contribute to the region's store drain (schedT) so
  // its done waits for them -- a region storing only to an internal buffer
  // completes after that buffer's deepest write commits.
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    ArrayRef<Value> banks = memBanks[m.id];
    for (unsigned a = 0; a < m.accesses.size(); ++a) {
      const uarch::MemUnit::Access &acc = m.accesses[a];
      if (!acc.isWrite || acc.region != ridx)
        continue;
      Value we = c.activationPulse(issue, acc.op);
      Value flat = computeAddr(m, acc), data = resolveSource(acc.data);
      auto wlat = c.b.getI64IntegerAttr(1);
      if (banks.size() == 1) {
        seq::WritePortOp::create(c.b, c.loc, banks[0],
                                 ValueRange{memAddr(m, flat)}, data, we, wlat);
      } else {
        // Drive every bank; the runtime bank gates the write-enable so only the
        // selected bank commits (an N-way we-demux).
        BankSplit bs = splitBank(c, flat, m.numBanks);
        Value addr = memAddr(m, bs.offset);
        for (unsigned k = 0; k < banks.size(); ++k)
          seq::WritePortOp::create(c.b, c.loc, banks[k], ValueRange{addr}, data,
                                   bankWe(c, we, bs.bank, k), wlat);
      }
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, schedT(acc.op));
    }
  }
  return fb;
}

void DatapathEmitter::bindStreamReads(const uarch::RegionBlock &rb) {
  for (const uarch::StreamChannel &s : dp.streams)
    for (unsigned a = 0; a < s.accesses.size(); ++a) {
      const uarch::StreamChannel::Access &acc = s.accesses[a];
      if (acc.isPut || acc.region != rb.id)
        continue;
      streamReadData[accKey(s.id, a)] =
          pa.getInput(iface::data_(streamPortBase(s)));
    }
}

// The latency-insensitive shell's port drives + control signals for region
// \p rb (freeze only on output back-pressure). A put drives
// `_data`/`_valid`; a get drives `_ready`. Only a full output freezes the
// pipeline (`chainEnable = ~outputFull`); an empty input injects a bubble by
// dropping `issueEnable`, never a freeze -- freezing on starvation would hold a
// mid-flight `valid` high and let a ready consumer double-capture the token. A
// stage-0 access keys on the UNgated `wantIssue` so the signals stay
// combinationally acyclic, a deeper access on the (registered) delayed issue.
// A predicated access (`acc.when` set) additionally gates its handshake on
// the predicate so a token is consumed/produced only where it holds; the
// predicate is a datapath value (no FIFO status), so acyclicity is preserved.
void DatapathEmitter::emitStreamAccesses(const uarch::RegionBlock &rb,
                                         Value issue, DatapathFeedback &fb) {
  // LI-shell scope invariants -- checked only for a region that actually runs
  // the shell (emit() calls this for every region; it is a no-op for a
  // stream-free one). Fail loudly rather than silently miscompile an
  // out-of-scope topology:
  //  * II == 1: the shell freezes every shift chain as one (chainEnable); a
  //    modulo (II>1) phase counter rides `issueEnable` instead, so on an input
  //    bubble it would diverge from the still-advancing chains -> broken tap
  //    alignment. The global-stall shell is defined only for the II==1
  //    invariant.
  //  * one access per channel: each channel drives exactly one
  //  {data,valid,ready}
  //    bundle, so two get/put on one channel would silently overwrite its port.
  //  * a multi-input region fires only when every input it needs this firing is
  //    present. Stage-0 gets (read at issue) pop together and gate the issue;
  //    an empty stage-0 input injects a bubble (drop `issueEnable`), never a
  //    freeze. A mid-pipeline get (stage > 0 -- e.g. a merge whose selector is
  //    a stream token read at stage 0, so the selected `get` lands a cycle
  //    later) cannot bubble: its in-flight iteration is already past issue, so
  //    a needed-but- empty deeper input FREEZES the whole pipeline
  //    (`chainEnable`) until the token arrives, preserving tap alignment. A
  //    predicated get counts as needed only where its predicate holds -- which
  //    makes a data-selected merge (one of N inputs popped per firing) a
  //    special case of this.
  bool hasStream = false;
  for (const uarch::StreamChannel &s : dp.streams) {
    bool here = false;
    for (const uarch::StreamChannel::Access &acc : s.accesses)
      if (acc.region == rb.id)
        here = hasStream = true;
    assert((!here || s.accesses.size() <= 1) &&
           "one access per stream channel");
  }
  if (hasStream)
    assert(rb.ii.value_or(1) == 1 && "stream LI shell assumes II == 1");

  Value atIssue =
      controlOf.lookup(rb.id).wantIssue; // ungated stage-0 activation
  if (!atIssue)
    atIssue = issue; // acyclic region: no separate wantIssue
  // Outputs: drive data + valid, accumulate the output-full hazard (the sole
  // freeze cause).
  Value outHazard; // OR over the region's puts of (valid & ~ready)
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (!acc.isPut || acc.region != rb.id)
        continue;
      std::string base = streamPortBase(s);
      // A predicated put produces a token only where its predicate holds:
      // gate `valid`, and suppress the output-full hazard when it is low (do
      // not freeze the pipeline waiting for space we will not write this
      // firing).
      Value pred = acc.when ? resolveSource(acc.when) : Value();
      Value valid = c.activationPulse(issue, acc.op);
      if (pred)
        valid = c.andBits(valid, pred);
      pa.setOutput(iface::data_(base), resolveSource(acc.data));
      pa.setOutput(iface::valid(base), valid);
      // A stage-0 put keys its hazard on wantIssue (ungated) & pred; a stage>=1
      // put's valid is already registered (delayed) and predicate-gated.
      Value active = acc.stage == 0 ? atIssue : valid;
      if (pred && acc.stage == 0)
        active = c.andBits(active, pred);
      Value hz = c.andBits(active, c.notBit(pa.getInput(iface::ready(base))));
      outHazard = outHazard ? c.orBits(outHazard, hz) : hz;
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, acc.stage);
    }
  // Mid-pipeline freeze: a get at stage > 0 whose input is needed-but-empty
  // freezes the whole pipeline -- its in-flight iteration cannot bubble past a
  // token it has not received -- so fold each such stall into `chainEnable`
  // alongside the output-full freeze. `active` and the predicate are registered
  // (delayed to the get's stage), so this reads only stored state -- no cycle.
  Value midStall;
  unsigned stage0Gets = 0;
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (acc.isPut || acc.region != rb.id)
        continue;
      if (acc.stage == 0) {
        ++stage0Gets;
        continue;
      }
      Value active = c.delayValid(issue, acc.stage);
      Value want =
          acc.when ? c.andBits(active, resolveSource(acc.when)) : active;
      Value miss = c.andBits(
          want, c.notBit(pa.getInput(iface::valid(streamPortBase(s)))));
      midStall = midStall ? c.orBits(midStall, miss) : miss;
    }
  Value chainEnable = outHazard ? c.notBit(outHazard) : c.t1;
  if (midStall)
    chainEnable = c.andBits(chainEnable, c.notBit(midStall));

  // Stage-0 inputs (read at issue): fold each effective valid into
  // `stage0Valid`, the issue gate. A predicated get treats a non-needed input
  // as available
  // (`valid | ~pred`), so a skipped input never blocks. With >1 stage-0 get
  // they must pop together (an elastic join), so their readies are gated on it
  // too.
  Value stage0Valid;
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (acc.isPut || acc.region != rb.id || acc.stage != 0)
        continue;
      Value valid = pa.getInput(iface::valid(streamPortBase(s)));
      if (acc.when)
        valid = c.orBits(valid, c.notBit(resolveSource(acc.when)));
      stage0Valid = stage0Valid ? c.andBits(stage0Valid, valid) : valid;
    }
  bool join0 = stage0Gets > 1;

  // Drive each `_ready`. A stage-0 get accepts when we want to issue and are
  // not frozen (independent of its OWN valid, per the handshake contract; a
  // join additionally waits for all stage-0 inputs). A deeper get accepts when
  // the chain advances -- `chainEnable` already withholds that if this get's
  // own input is the missing one. A predicated get pops only where its
  // predicate holds, so a data-selected merge consumes exactly the chosen
  // input.
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (acc.isPut || acc.region != rb.id)
        continue;
      Value pred = acc.when ? resolveSource(acc.when) : Value();
      Value active = acc.stage == 0 ? atIssue : c.delayValid(issue, acc.stage);
      Value ready = c.andBits(active, chainEnable);
      if (acc.stage == 0 && join0)
        ready = c.andBits(ready, stage0Valid);
      if (pred)
        ready = c.andBits(ready, pred);
      pa.setOutput(iface::ready(streamPortBase(s)), ready);
    }
  fb.chainEnable = chainEnable;
  fb.issueEnable =
      stage0Valid ? c.andBits(chainEnable, stage0Valid) : chainEnable;
}

// Instantiate each CallUnit (dcp.invoke) in region \p rb as a child
// hw.instance. The child masters each memref operand's memory: it drives the
// addr/data/we, so the leaf wires those instance-output ports to the buffer's
// hlmem (a seq.read whose data feeds back to the child, a seq.write). The
// region's completion is the child's real `done` (fb.callDone). Serial
// execution (a producer region drains before the child starts, the child before
// a consumer) means one master per port at a time -- no arbitration mux.
void DatapathEmitter::emitCalls(const uarch::RegionBlock &rb, Value issue,
                                DatapathFeedback &fb) {
  // Multiple calls share one region when they form a straight-line span (calls
  // with no loose op between them reify to one dcp.sequential). Each call
  // starts on the joined `done` of its predecessors -- an earlier call it
  // depends on (a schedule-serialized shared buffer/boundary, or a scalar
  // result it consumes); a call with no predecessor starts on the region
  // `issue`, CONCURRENT with its independent siblings (the intra-region
  // analogue of composeSiblings). The region completes when EVERY call has --
  // the AND of their (held) dones, last-to-finish under concurrency.
  SmallVector<Value> dones;                        // each call's done, by index
  SmallVector<SmallVector<uarch::MemId>> callMems; // each call's touched MemIds
  SmallVector<unsigned> callStart; // each call's scheduled start
  SmallVector<bool> callIndet;     // each call's indeterminacy
  llvm::DenseMap<uarch::CallId, Value>
      doneByCid; // done by id (scalar hand-off)
  for (uarch::CallId cid : rb.callUnits) {
    const uarch::CallUnit &cu = dp.calls[cid];
    // Shared-memref predecessors: an earlier call the SCHEDULE places strictly
    // before this one (`start` smaller) and touching a common MemId -- a real
    // hazard the scheduler serialized. Starting on their joined `done` realizes
    // it; calls the scheduler left at the same offset share only reads/disjoint
    // elements, so they carry no edge and run concurrently (each contended
    // array exposes one port group per accessor). A call sharing nothing starts
    // on `issue`. An INDETERMINATE producer (a `while` leaf) is the
    // exception: the scheduler could not offset a consumer past it (its latency
    // is unknown), so it collapses to the same `start` -- a program-order
    // sharer must still await its real `done`, or it reads what the producer
    // has not yet written.
    SmallVector<uarch::MemId> myMems;
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs)
      myMems.push_back(ma.mem);
    SmallVector<Value> predDones;
    for (auto [j, jmems] : llvm::enumerate(callMems))
      if ((callStart[j] < cu.start || callIndet[j]) &&
          llvm::any_of(jmems, [&](uarch::MemId m) {
            return llvm::is_contained(myMems, m);
          }))
        predDones.push_back(dones[j]);
    // A scalar hand-off is a dependence too: a child consuming an earlier
    // child's result (Source::Call) must start after that producer's `done`.
    for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
      if (sa.src.kind == uarch::Source::Kind::Call)
        if (Value d = doneByCid.lookup(sa.src.id))
          predDones.push_back(d);
    Value startK = c.startFor(issue, predDones);
    assert(callees && "a CallUnit needs callee context");
    auto mit = callees->modules.find(cu.callee);
    assert(mit != callees->modules.end() &&
           "the callee module must be registered (emitted bottom-up first)");
    hw::HWModuleOp child = mit->second;

    // Instance inputs by child port name: clk/rst/`start` (the region's issue
    // pulse) + each read's data input. An internal read consumes a backedge
    // (the seq.read output, resolved after the instance); a boundary read
    // passes the top's data input port straight through to the child.
    llvm::StringMap<Value> ins;
    ins["clk"] = c.clkRaw;
    ins["rst"] = c.rst;
    ins["start"] = startK;
    llvm::StringMap<circt::Backedge> rdBackedge;
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (ma.isWrite)
        continue;
      if (ma.isBoundary)
        ins[ma.data] = pa.getInput(iface::data_(ma.topBase));
      else {
        Backedge be = c.bb.get(memElemType(dp.mems[ma.mem], c.b));
        ins[ma.data] = be;
        rdBackedge.try_emplace(ma.data, be);
      }
    }
    // Scalar operands: drive each child scalar-input port from its
    // resolved Source (an IO port, a sibling survivor latched earlier, an
    // earlier child's live result, or a constant). Sampled at the child's
    // start.
    for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
      ins[sa.port] = resolveSource(sa.src);

    // Wire the child instance: inputs by port name from `ins`, outputs by name.
    llvm::StringMap<Value> outs =
        instantiateChild(c.b, c.loc, child, cu.callee, ins);

    // Scalar results: the child holds each result on its output port
    // from `done` onward, so the port value IS the survivor a sibling region
    // reads (no separate capture -- the `done` handshake gates the consumer on
    // a valid result). callResultVal serves a same-region later child that
    // reads it live (Source::Call), survivorOf a cross-region sibling
    // (Source::Survivor over the call's region); both resolve to the same held
    // wire.
    for (auto [r, port] : llvm::enumerate(cu.resultPorts)) {
      callResultVal[accKey(cu.id, r)] = outs[port];
      setSurvivor(cu.region, r, outs[port]);
    }

    // Master each buffer from the child's addr/data/we outputs. A boundary arg
    // passes them through to the top boundary port (flat i32
    // address); an internal buffer drives its hlmem (1-cycle registered, the
    // RAM contract the child was compiled against; address narrowed to the
    // clog2(depth) index).
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (ma.isBoundary) {
        // One port group per accessor: drive it DIRECTLY from the child's
        // addr/data/we. Concurrent masters of an argument have distinct groups
        // (no mux); a serial pair also uses two groups, each driven only in its
        // own phase (a child self-gates we == 0 outside its run).
        pa.setOutput(iface::addr(ma.topBase), outs[ma.addr]);
        if (ma.isWrite) {
          pa.setOutput(iface::data_(ma.topBase), outs[ma.data]);
          pa.setOutput(iface::we(ma.topBase), outs[ma.we]);
        }
        continue;
      }
      const uarch::MemUnit &m = dp.mems[ma.mem];
      // One hlmem per bank: the child masters bank `ma.bank`, already
      // addressing that bank's own index space (propagate-partition gave every
      // callee the same `allo.part`), so route straight to memBanks[m.id][bank]
      // -- no crossbar. An unbanked buffer is bank 0. The parent's bank count
      // (characterize) and the child's (iface factor) agree by construction
      // (propagate-partition), so the index is in range -- assert it loudly.
      assert(ma.bank < memBanks[m.id].size() &&
             "child bank index exceeds the buffer's bank count (parent/callee "
             "partition-factor disagreement)");
      Value hlmem = memBanks[m.id][ma.bank];
      Value addr = memAddr(m, outs[ma.addr]);
      auto lat = c.b.getI64IntegerAttr(1);
      if (ma.isWrite)
        seq::WritePortOp::create(c.b, c.loc, hlmem, ValueRange{addr},
                                 outs[ma.data], outs[ma.we], lat);
      else
        rdBackedge[ma.data].setValue(c.R(seq::ReadPortOp::create(
            c.b, c.loc, hlmem, ValueRange{addr}, /*rdEn=*/Value(), 1)));
    }
    doneByCid[cu.id] = outs["done"];
    dones.push_back(outs["done"]);
    callMems.push_back(std::move(myMems));
    callStart.push_back(cu.start);
    callIndet.push_back(!cu.latency);
  }
  // The region completes when every call has: the AND of their held dones
  // (last-to-finish; a serial chain degenerates to the last call's done).
  Value all;
  for (Value d : dones)
    all = all ? c.andBits(all, d) : d;
  if (all)
    fb.callDone = all;
}

// The child induction-variable scalar port's type for a loop-over-call region:
// the IV scalar operand is the one whose Source is this region's
// `Counter`, so the counter emitLoopCall builds must be that port's width
// (resolveSource(Counter) drives the port with no cast).
Type DatapathEmitter::loopIndexPortType(const uarch::RegionBlock &rb) {
  assert(rb.callUnits.size() == 1 && "a loop-over-call region has one child");
  assert(callees && "a loop-over-call needs callee context");
  const uarch::CallUnit &cu = dp.calls[rb.callUnits.front()];
  auto mit = callees->modules.find(cu.callee);
  assert(mit != callees->modules.end() && "the loop child must be registered");
  hw::HWModuleOp child = mit->second;
  for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
    if (sa.src.kind == uarch::Source::Kind::Counter && sa.src.id == rb.id)
      for (const hw::PortInfo &p : child.getPortList())
        if (p.name.getValue() == sa.port)
          return p.type;
  llvm_unreachable(
      "a loop-over-call region has no induction-variable child port");
}

// Emit region \p rb's whole datapath given the controller's \p issue; returns
// its store feedback. Stream reads bound first (before any consumer), then
// registers, internal reads, units, register heads, accesses, and finally the
// stream shell (port drives + stall).
DatapathFeedback DatapathEmitter::emit(const uarch::RegionBlock &rb,
                                       Value issue) {
  bindStreamReads(rb);
  emitRegisters(rb);
  emitInternalReads(rb);
  emitExternalReads(rb);
  emitUnits(rb);
  resolveRegHeads(rb);
  DatapathFeedback fb = emitAccesses(rb, issue);
  emitStreamAccesses(rb, issue, fb);
  emitCalls(rb, issue, fb);
  return fb;
}

} // namespace mlir::allo::uarch
