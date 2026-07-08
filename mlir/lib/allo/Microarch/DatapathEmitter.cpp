/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Datapath (F): register chains, compute units, memory access + addressing, and
// the uniform Source resolution `src`. The controller's `issue` arrives as an
// argument and its `counter` via `setCounter`; the region's store drain (the
// deepest store's stage) is returned to the controller. See HWEmit.h.
//===----------------------------------------------------------------------===//

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
Value DatapathEmitter::src(const uarch::Source &s) {
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
    Value cv = counterOf.lookup(s.id);
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
    Value issue = issueOf.lookup(mx.region);
    assert(issue && "mux in a region with no controller");
    Value v = src(mx.sources[0]);
    for (unsigned i = 1; i < mx.sources.size(); ++i) {
      Value sel = c.activationPulse(issue, mx.selectOps[i]);
      v = c.mux(sel, src(mx.sources[i]), v);
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
  case uarch::Source::Kind::None:
    assert(false && "unresolved (None) source");
    return {};
  }
  llvm_unreachable("unhandled Source::Kind");
}

unsigned DatapathEmitter::readyCycle(const uarch::Source &s) const {
  if (s.kind == uarch::Source::Kind::Unit) {
    const uarch::FuncUnit &u = dp.units[s.id];
    return schedT(u.boundOps.front().first) + u.latency;
  }
  if (s.kind == uarch::Source::Kind::Mem) {
    const uarch::MemUnit &m = dp.mems[s.id];
    return schedT(m.accesses[s.outPort].op) + memReadLatency(m.impl);
  }
  if (s.kind == uarch::Source::Kind::Stream)
    // A stream get is a combinational FIFO-front read: the token is ready at
    // the get's stage (read latency 0).
    return dp.streams[s.id].accesses[s.outPort].stage;
  // A Reg result lands elsewhere; only Unit/Mem/Const ready cycles are modelled
  // (a constant is at-issue, cycle 0).
  assert(s.kind == uarch::Source::Kind::Const &&
         "readyCycle only modelled for a Unit (schedT+lat), a memory read "
         "(schedT+read-lat), or a constant (at-issue)");
  return 0;
}

// Evaluate an affine index expression to an i32 hw value, emitting comb ops.
// `idx` holds the resolved value of each map operand (dims then symbols). Fork
// (3) address-value: affine addressing degenerates to adds and multiply-by-
// constant; a genuine multiplier survives only a non-reducible map, and
// identity addressing emits nothing.
Value DatapathEmitter::evalAffine(AffineExpr e, ArrayRef<Value> idx,
                                  unsigned numDims) {
  if (auto cst = dyn_cast<AffineConstantExpr>(e))
    return c.konst(c.i32, cst.getValue());
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return idx[d.getPosition()];
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return idx[numDims + sym.getPosition()];
  auto bin = cast<AffineBinaryOpExpr>(e);
  Value lhs = evalAffine(bin.getLHS(), idx, numDims);
  Value rhs = evalAffine(bin.getRHS(), idx, numDims);
  if (e.getKind() == AffineExprKind::Add)
    return c.R(comb::AddOp::create(c.b, c.loc, lhs, rhs, false));
  if (e.getKind() == AffineExprKind::Mul)
    return c.R(comb::MulOp::create(c.b, c.loc, lhs, rhs, false));
  // floordiv / mod by a constant are the delinearization a coalesced nest
  // leaves in an address (`iv floordiv N`, `iv mod N`) over a non-negative
  // index. A power-of-two divisor is a shift / bit-mask (no divider); any other
  // constant is a general unsigned divide / remainder -- synthesis folds a
  // constant divisor to a multiply-shift, so no runtime divider is
  // instantiated.
  auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS());
  assert(rc && rc.getValue() > 0 &&
         "affine div/mod by a non-constant or non-positive divisor in address");
  int64_t f = rc.getValue();
  Value fv = c.konst(c.i32, f);
  bool pow2 = llvm::isPowerOf2_64(f);
  if (e.getKind() == AffineExprKind::FloorDiv)
    return pow2 ? c.R(comb::ShrUOp::create(
                      c.b, c.loc, lhs, c.konst(c.i32, llvm::Log2_64(f)), false))
                : c.R(comb::DivUOp::create(c.b, c.loc, lhs, fv, false));
  assert(e.getKind() == AffineExprKind::Mod &&
         "unexpected affine op in address");
  return pow2 ? c.R(comb::AndOp::create(c.b, c.loc, lhs, c.konst(c.i32, f - 1),
                                        false))
              : c.R(comb::ModUOp::create(c.b, c.loc, lhs, fv, false));
}

// The linear element address of a memory access: evaluate its affine map over
// the (already stage-delayed) index sources, then linearize by the memref's
// row-major strides. Single-index identity addressing emits no ops.
Value DatapathEmitter::computeAddr(const uarch::MemUnit &m,
                                   const uarch::MemUnit::Access &acc) {
  SmallVector<Value> idx;
  for (const uarch::Source &s : acc.addr)
    idx.push_back(src(s));
  AffineMap map = acc.addrMap;
  assert(map && "dcp memory access without an affine map");
  ArrayRef<int64_t> shape = cast<MemRefType>(m.memref.getType()).getShape();
  unsigned rank = map.getNumResults();
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * shape[k + 1];
  Value addr;
  for (unsigned k = 0; k < rank; ++k) {
    Value term = evalAffine(map.getResult(k), idx, map.getNumDims());
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
      // / multi-dim / external banking are not lowered yet.
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
      Value v = src(u.inputs[k]); // a self-reference reads its own backedge
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
        Value iv = counterOf.lookup(rb.id), issue = issueOf.lookup(rb.id);
        assert(iv && issue &&
               "recurrence input in a region with no controller");
        // The first iteration is `iv == lb`; the lb is a runtime Source (a
        // data-dependent range start) or the constant fast path.
        Value lb = rb.lbSource ? src(rb.lbSource) : c.konst(c.i32, rb.lb);
        Value iter0 = c.R(
            comb::AndOp::create(c.b, c.loc, issue, c.icmpEqV(iv, lb), false));
        Value gate = c.activationPulse(iter0, u.boundOps.front().first);
        v = c.mux(gate, src(u.inputInits[k]), v);
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
      // `regionEnable` is null, so `ce` is a constant 1 (free-running,
      // identical to before).
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

// Resolve region \p rb's register head inputs now that its units exist.
void DatapathEmitter::resolveRegHeads(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs)
    regHeadBE.find(rid)->second.setValue(src(dp.regs[rid].input));
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
    Value addr = extAddr(m, acc), data = src(acc.data);
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
      Value flat = computeAddr(m, acc), data = src(acc.data);
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
// \p rb (Approach 2 -- freeze only on output back-pressure). A put drives
// `_data`/`_valid`; a get drives `_ready`. Only a full output freezes the
// pipeline (`chainEnable = ~outputFull`); an empty input injects a bubble by
// dropping `issueEnable`, never a freeze -- freezing on starvation would hold a
// mid-flight `valid` high and let a ready consumer double-capture the token. A
// stage-0 access keys on the UNgated `wantIssue` so the signals stay
// combinationally acyclic, a deeper access on the (registered) delayed issue.
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
  //  * at most one input get per region: a multi-input join needs each `_ready`
  //    gated on the OTHER inputs' valids (else a leading input pops a token
  //    while `issueEnable` is low and it is lost); not implemented (topology
  //    has no merge yet).
  bool hasStream = false;
  unsigned inGets = 0;
  for (const uarch::StreamChannel &s : dp.streams) {
    bool here = false;
    for (const uarch::StreamChannel::Access &acc : s.accesses)
      if (acc.region == rb.id) {
        here = hasStream = true;
        inGets += !acc.isPut;
      }
    assert((!here || s.accesses.size() <= 1) &&
           "one access per stream channel (P0)");
  }
  if (hasStream) {
    assert(rb.ii.value_or(1) == 1 && "stream LI shell assumes II == 1");
    assert(inGets <= 1 && "multi-input stream join not implemented");
  }

  Value atIssue = wantIssueOf.lookup(rb.id); // ungated stage-0 activation
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
      Value valid = c.activationPulse(issue, acc.op);
      pa.setOutput(iface::data_(base), src(acc.data));
      pa.setOutput(iface::valid(base), valid);
      // A stage-0 put keys its hazard on wantIssue (ungated); a stage>=1 put's
      // valid is already registered (delayed), so it is safe to key on.
      Value active = acc.stage == 0 ? atIssue : valid;
      Value hz = c.andBits(active, c.notBit(pa.getInput(iface::ready(base))));
      outHazard = outHazard ? c.orBits(outHazard, hz) : hz;
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, acc.stage);
    }
  Value chainEnable = outHazard ? c.notBit(outHazard) : c.t1;
  // Inputs: drive `_ready` (accept a token when we want to issue and are not
  // output-blocked -- independent of the input's own valid, per the handshake
  // contract), and AND every input valid into the issue gate (an empty input
  // yields a bubble, not a freeze).
  Value allValid;
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (acc.isPut || acc.region != rb.id)
        continue;
      std::string base = streamPortBase(s);
      Value active = acc.stage == 0 ? atIssue : c.delayValid(issue, acc.stage);
      pa.setOutput(iface::ready(base), c.andBits(active, chainEnable));
      Value valid = pa.getInput(iface::valid(base));
      allValid = allValid ? c.andBits(allValid, valid) : valid;
    }
  fb.chainEnable = chainEnable;
  fb.issueEnable = allValid ? c.andBits(chainEnable, allValid) : chainEnable;
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
  return fb;
}

} // namespace mlir::allo::uarch
