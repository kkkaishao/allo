/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/Microarch/Interface.h" // iface field-name helpers

#include "allo/Scheduling/MemoryModel.h" // partitionOf (crossbar shape check)
#include "allo/Scheduling/OperatorLibrary.h" // stallContract

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// --- Memory-banking crossbar primitives -------------------------------------

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

ExternalBanking externalBank(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  if (m.numBanks == 1)
    return {1u, 0u};
  auto p = partitionOf(m.memref);
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
    return pa.getInput(scalarPortName(dp, dp.ios[s.id]));
  case uarch::Source::Kind::Mux: {
    // A shared unit's input: drive each source on the cycle its op consumes
    // it, via a priority chain (mutually exclusive, MRT-verified selects)
    // with source 0 as default; the select is the op's `activationPulse`.
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
  // floordiv/mod by a constant is delinearization left by a coalesced nest
  // over a non-negative index. A power-of-two divisor is a shift/bit-mask;
  // any other constant divide/remainder is synthesis-folded to a
  // multiply-shift.
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
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k) {
    // Row-major strides are a product of the trailing static extents. A
    // dynamic non-leading dim would poison the stride and mis-address every
    // multi-dim access; a leading dynamic dim is safe since shape[0] is never
    // read.
    assert(!ShapedType::isDynamic(shape[k + 1]) &&
           "row-major linearization needs static non-leading memref dims");
    stride[k] = stride[k + 1] * shape[k + 1];
  }
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
  // A stream-region read must freeze its address on stall, or the counter
  // over-runs and the in-flight read is lost, violating KPN semantics; a
  // write needs no hold since its gated write-enable simply skips a stall.
  return acc.isWrite ? addr : c.stallHold(addr);
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
    auto eb = externalBank(m, m.accesses[reads[i].idx]);
    if (eb.factor > 1 && !eb.bank)
      continue; // data-dependent: bound by emitExternalReads
    readData[accKey(reads[i].mem, reads[i].idx)] = pa.getInput(
        portData(extPorts(dp, reads, i, /*write=*/false).front().second));
  }
}

// Instantiate on-chip storage for each internal (non-argument) memory: one
// seq.hlmem, or, when the array reached emit still partitioned (a
// data-dependent bank dcp-resolve-banking could not split statically), one
// per bank, addressed through the crossbar (splitBank/readCrossbar). The
// handles are module-scope so writes and reads in different regions share them.
void DatapathEmitter::createInternalMemories() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    if (m.isRom) {
      // A constant table: one hw.aggregate_constant holding the global's
      // initializer, read combinationally by hw.array_get (registered to the
      // read latency in emitInternalReads). No writable hlmem, no write ports.
      assert(m.numBanks == 1 && "a banked ROM is not supported");
      auto data = cast<ElementsAttr>(m.romInit);
      IntegerType elemTy = memElemType(m, c.b);
      SmallVector<Attribute> fields;
      fields.reserve(m.depthWords);
      for (const APInt &v : data.getValues<APInt>())
        fields.push_back(
            IntegerAttr::get(elemTy, v.zextOrTrunc(elemTy.getWidth())));
      // A hw.array indexes element 0 as the LAST aggregate_constant field, so
      // the natural-order initializer is reversed to make array_get(i) ==
      // data[i].
      std::reverse(fields.begin(), fields.end());
      auto arrTy = hw::ArrayType::get(elemTy, m.depthWords);
      romArray[m.id] = hw::AggregateConstantOp::create(
          c.b, c.loc, arrTy, c.b.getArrayAttr(fields));
      continue;
    }
    if (m.numBanks > 1) {
      // The emitter crossbar handles a 1-D power-of-two cyclic partition; block
      // / multi-dim / external banking are not supported.
      auto p = partitionOf(m.memref);
      assert(
          p.cyclicAxes.size() == 1 && !p.hasBlock &&
          cast<MemRefType>(m.memref.getType()).getRank() == 1 &&
          llvm::isPowerOf2_64(m.numBanks) &&
          "emitter banking crossbar: only 1-D power-of-two cyclic supported");
    }
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k) {
      auto mem = seq::HLMemOp::create(
          c.b, c.loc, c.clk, c.rst, memCellName(dp, m, k),
          {static_cast<int64_t>(m.depthWords)}, memElemType(m, c.b));
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
    auto head = c.bb.get(hwType(rg.type, c.b));
    regHeadBE.try_emplace(rg.id, head);
    // A register is a plain delay chain; reduction-identity re-injection
    // rides the consuming unit's recurrence input (emitUnits), not the
    // register.
    regStages[rg.id] = c.shiftChain(head, rg.depth);
    // Name each held stage `<value>_d<k>`: the value it delays, plus how many
    // cycles late it is. Stage 0 is the undelayed input, already named by its
    // producer, so leave it alone rather than relabel a shared wire.
    std::string owner = ownerOf(rg.value, regOwner(rg.id));
    for (auto [k, stage] : llvm::enumerate(regStages[rg.id].stages))
      if (k)
        nameValue(stage, regTapName(owner, k));
  }
}

// seq.read for each internal-memory read scheduled in region \p rb, bound into
// readData *before* emitUnits consumes it (a Source::Mem input). Read latency
// is the memory's device-resolved `readLatency`, the same number the
// scheduler timed the access at, so the port lands the datum on exactly the
// cycle the consumer's register depth was solved against.
void DatapathEmitter::emitInternalReads(const uarch::RegionBlock &rb) {
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    unsigned lat = m.readLatency;
    if (m.isRom) {
      // A constant table read: index the aggregate_constant combinationally,
      // then register to the (scheduled) read latency so timing matches a RAM.
      Value arr = romArray[m.id];
      for (unsigned a = 0; a < m.accesses.size(); ++a) {
        const uarch::MemUnit::Access &acc = m.accesses[a];
        if (acc.isWrite || acc.region != rb.id)
          continue; // (a ROM carries no writes)
        Value idx = memAddr(m, computeAddr(m, acc));
        Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, arr, idx));
        readData[accKey(m.id, a)] = lat ? c.shiftChain(elem, lat).last() : elem;
      }
      continue;
    }
    ArrayRef<Value> banks = memBanks[m.id];
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
        // Read every bank at the (bank-independent) offset, then select by
        // the runtime bank, aligned with the read data (delayed by the
        // latency).
        auto bs = splitBank(c, flat, m.numBanks);
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
// each bank's data port, and mux by the runtime bank, delayed to the memory's
// device read latency so the select aligns with its data. Bound into
// readData before emitUnits, the twin of emitInternalReads for boundary ports
// instead of hlmems.
void DatapathEmitter::emitExternalReads(const uarch::RegionBlock &rb) {
  for (unsigned i = 0; i < reads.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[reads[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[reads[i].idx];
    auto eb = externalBank(m, acc);
    if (acc.region != rb.id || eb.factor == 1 || eb.bank)
      continue; // only data-dependent banked reads
    auto bs = splitBank(c, computeAddr(m, acc), eb.factor);
    SmallVector<Value> vals;
    for (const auto &[bank, base] : extPorts(dp, reads, i, /*write=*/false)) {
      pa.setOutput(portAddr(base), bs.offset);
      vals.push_back(pa.getInput(portData(base)));
    }
    unsigned lat = m.readLatency;
    Value sel = lat ? c.shiftChain(bs.bank, lat).last() : bs.bank;
    readData[accKey(reads[i].mem, reads[i].idx)] = readCrossbar(c, vals, sel);
  }
}

// Compute units of region \p rb: native -> comb; IP -> an instance of the
// extern operator module (internally pipelined by its latency).
// Backedge every unit output before wiring, so an input may reference a unit
// emitted later: the widened-reduction idiom reads the accumulator as a depth-0
// loop-carry from a later unit, a fused recurrence reads its own output, and a
// data-dependent read address (emitInternalReads, which runs before emitUnits)
// reads a unit that computes it. A register elsewhere in the recurrence cycle
// keeps the hardware acyclic; the backedges only free emission from
// topological order.
void DatapathEmitter::declareUnits(const uarch::RegionBlock &rb) {
  for (uarch::UnitId uid : rb.units) {
    auto b = c.bb.get(hwType(dp.units[uid].resultType, c.b));
    unitBE[uid] = b;
    unitVal[uid] = b;
  }
}

void DatapathEmitter::emitUnits(const uarch::RegionBlock &rb) {
  for (uarch::UnitId uid : rb.units) {
    const uarch::FuncUnit &u = dp.units[uid];
    SmallVector<Value> operands;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      Value v =
          resolveSource(u.inputs[k]); // a self-reference reads its own backedge
      // Re-inject the reduction identity at a recurrence input (a
      // loop-carried iter_arg read) while `iv` is in `[lb, lb + dist*step)`,
      // gated by the issue pulse delayed to this op's stage; one gate covers
      // both regimes.
      if (u.inputInits[k].kind != uarch::Source::Kind::None) {
        const auto rc = controlOf.lookup(rb.id);
        Value iv = rc.counter, issue = rc.issue;
        assert(iv && issue &&
               "recurrence input in a region with no controller");
        // lb is a runtime Source (a data-dependent range start) or the constant
        // fast path.
        Value lb =
            rb.lbSource ? resolveSource(rb.lbSource) : c.konst(c.i32, rb.lb);
        unsigned dist = u.inputInitDist[k];
        Value cond;
        if (dist <= 1) {
          cond = c.icmpEqV(iv, lb); // iv == lb
        } else {
          // iv < lb + dist*step  ==  !(iv >= lb + dist*step)
          Value distStep =
              rb.stepSource.kind != uarch::Source::Kind::None
                  ? c.R(comb::MulOp::create(
                        c.b, c.loc, c.konst(c.i32, static_cast<int64_t>(dist)),
                        resolveSource(rb.stepSource), false))
                  : c.konst(c.i32, static_cast<int64_t>(dist) * rb.step);
          Value bound =
              c.R(comb::AddOp::create(c.b, c.loc, lb, distStep, false));
          cond = c.notBit(c.icmpUgeV(iv, bound));
        }
        Value iter0 = c.R(comb::AndOp::create(c.b, c.loc, issue, cond, false));
        Value gate = c.activationPulse(iter0, u.boundOps.front().first);
        v = c.mux(gate, resolveSource(u.inputInits[k]), v);
      }
      operands.push_back(v);
    }

    Value result;
    if (u.comb) {
      result = emitCompute(c.b, c.loc, u.opType, operands,
                           hwType(u.resultType, c.b), u.boundOps.front().first);
    } else {
      // An IP instance takes its data operands, then clock, then (for a
      // clock-enabled contract) a `ce` bit that rides the region's
      // clock-enable, freezing with the shift chains under back-pressure.
      operands.push_back(c.clkRaw);
      if (u.stall == allo::StallContractEnum::Ce)
        operands.push_back(c.regionEnable ? c.regionEnable : c.t1);
      else
        // A free-running/elastic IP has no `ce`: in a back-pressured region
        // it would keep advancing while the shell's shift chains stall,
        // folding a stale result, so a stallable region needs a Ce operator.
        assert(!c.regionEnable &&
               "a free-running/elastic IP operator cannot participate in a "
               "back-pressured region; use a clock-enabled (ce) operator");
      result = hw::InstanceOp::create(c.b, c.loc, unitModule.lookup(u.id),
                                      unitInstanceName(u), operands)
                   ->getResult(0);
    }
    unitBE[uid].setValue(result);
    unitVal[u.id] = result;
    // Name the result wire after the frontend variable this op computes (the
    // dcp op carries the assignment-target NameLoc, e.g. "acc").
    nameValue(result, u.boundOps.front().first->getLoc());
  }
}

// A container's own combinational units: its continue-condition (a
// sequential-wrapper while) or a child guard's predicate, reified
// into start-0 `dcp.compute`s bound in the container. Unlike `emitUnits` there
// is no reduction-identity re-injection (these read the container counter,
// iter-arg survivors, or constants, never a loop-carried accumulator), so no
// issue pulse is needed (a container has none). Emitted after the counter and
// survivors are set and before the children are sequenced, so a child guard
// resolves its parent-emitted predicate (Source::Unit). Backedges let the tree
// wire in any order, exactly as `emitUnits` does.
void DatapathEmitter::emitCombUnits(const uarch::RegionBlock &rb) {
  DenseMap<unsigned, Backedge> outBE;
  for (uarch::UnitId uid : rb.units) {
    auto be = c.bb.get(hwType(dp.units[uid].resultType, c.b));
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
    assert(u.comb &&
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

// The condition cone of a sequential (CHECK/RUN) while: emit the container's
// OWN condition memory reads plus its combinational compute, and return the
// settled condition value + its ready latency t_cond. Unlike a leaf region's
// `emit`, there is no per-iteration issue pulse: the read address is the
// frozen iter-arg survivor, so the load is a continuous read of a stable
// element and its data is a stable wire from `checkStart + t_cond` onward (the
// survivors do not advance until after the body drains, which is after CHECK
// decides). A combinational condition has no read, so this reduces to
// `emitCombUnits` with t_cond == 0.
std::pair<Value, unsigned>
DatapathEmitter::emitConditionRegion(const uarch::RegionBlock &rb,
                                     const uarch::Source &condSrc) {
  // Same emission order as a leaf region's `emit` (registers, unit
  // backedges, reads, units, register heads); a container has no
  // per-iteration recurrence, so `emitUnits`'s reduction re-injection stays
  // inert.
  emitRegisters(rb);
  declareUnits(rb);
  emitInternalReads(rb);
  emitExternalReads(rb);
  emitUnits(rb);
  resolveRegHeads(rb);
  // Unbanked/statically-banked external condition reads: drive the
  // read-address port with the survivor-addressed element, after emitUnits
  // so it resolves to the filled unit value, not a dangling backedge.
  for (unsigned i = 0; i < reads.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[reads[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[reads[i].idx];
    if (acc.region != rb.id)
      continue;
    auto eb = externalBank(m, acc);
    if (eb.factor > 1 && !eb.bank)
      continue; // data-dependent: emitExternalReads drove it
    Value flat = computeAddr(m, acc);
    Value off = eb.factor > 1 ? splitBank(c, flat, eb.factor).offset : flat;
    pa.setOutput(portAddr(memPortBase(dp, reads, i, /*write=*/false)), off);
  }
  return {resolveSource(condSrc), readyCycle(condSrc)};
}

// Resolve region \p rb's register head inputs now that its units exist.
void DatapathEmitter::resolveRegHeads(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs)
    regHeadBE.find(rid)->second.setValue(resolveSource(dp.regs[rid].input));
}

// The drain stage a store contributes to its region's `done`. The write is
// PRESENTED at `dcpStart` and COMMITS `writeLatency` cycles later; `emitDone`
// rides its own latch register for the last of those cycles (done reads 1 at
// `lastIssue + drainStage + 1`), so the stage is the commit cycle minus that
// one.
static unsigned storeDrainOf(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  assert(m.writeLatency >= 1 &&
         "a zero-cycle write has no commit edge for the done latch to ride");
  return dcpStart(acc.op) + m.writeLatency - 1;
}

// Read/write address + data outputs of the accesses scheduled in region \p rb,
// driven by that region's controller (counter / \p issue). Returns the region's
// store feedback: `storeDrain`, the stage its deepest store commits at (see
// `storeDrainOf`), which the region's `done` waits on.
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
    auto eb = externalBank(m, acc);
    // A data-dependent read's addresses + crossbar are emitted in
    // emitExternalReads (before emitUnits); here handle the single-interface
    // case (unbanked or statically banked).
    if (acc.region == ridx && (eb.factor == 1 || eb.bank))
      pa.setOutput(portAddr(memPortBase(dp, reads, i, /*write=*/false)),
                   extAddr(m, acc));
  }
  DatapathFeedback fb;
  // A store's write-enable is the issue pulse delayed to the store's stage.
  // A leaf `while`'s doomed exit iteration still issues, so its store is
  // also gated by the continue-condition; container/guard stores are gated
  // structurally by not-issuing.
  Value gatedIssue;
  auto commitPulse = [&]() -> Value {
    if (!rb.conditional)
      return issue;
    if (!gatedIssue) {
      auto ci = dp.carryInfo.find(rb.id);
      assert(ci != dp.carryInfo.end() &&
             "conditional (while) region has no carryInfo entry; its continue "
             "condition is required to gate in-loop store commits");
      gatedIssue = c.andBits(issue, resolveSource(ci->second.condition));
    }
    return gatedIssue;
  };
  for (unsigned i = 0; i < writes.size(); ++i) {
    const uarch::MemUnit &m = dp.mems[writes[i].mem];
    const uarch::MemUnit::Access &acc = m.accesses[writes[i].idx];
    if (acc.region != ridx)
      continue;
    Value we = c.activationPulse(commitPulse(), acc.op);
    Value addr = extAddr(m, acc), data = resolveSource(acc.data);
    auto eb = externalBank(m, acc);
    // A data-dependent write drives every bank interface; its runtime bank
    // gates each interface's write-enable so only the target bank commits (an
    // N-way demux). A static / unbanked write is a single interface.
    Value dynBank =
        eb.bank ? Value() : splitBank(c, computeAddr(m, acc), eb.factor).bank;
    for (const auto &[bank, base] : extPorts(dp, writes, i, /*write=*/true)) {
      pa.setOutput(portAddr(base), addr);
      pa.setOutput(portData(base), data);
      pa.setOutput(portWe(base),
                   dynBank ? c.andBits(we, c.icmpEq(dynBank, bank)) : we);
    }
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
  }
  // Internal-memory writes drive seq.write instead of module ports, but
  // still set the region's store drain so `done` waits for them: a region
  // storing only to an internal buffer completes after its deepest write
  // commits.
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    ArrayRef<Value> banks = memBanks[m.id];
    for (unsigned a = 0; a < m.accesses.size(); ++a) {
      const uarch::MemUnit::Access &acc = m.accesses[a];
      if (!acc.isWrite || acc.region != ridx)
        continue;
      Value we = c.activationPulse(commitPulse(), acc.op);
      Value flat = computeAddr(m, acc), data = resolveSource(acc.data);
      auto wlat = c.b.getI64IntegerAttr(m.writeLatency);
      if (banks.size() == 1) {
        seq::WritePortOp::create(c.b, c.loc, banks[0],
                                 ValueRange{memAddr(m, flat)}, data, we, wlat);
      } else {
        // Drive every bank; the runtime bank gates the write-enable so only the
        // selected bank commits (an N-way we-demux).
        auto bs = splitBank(c, flat, m.numBanks);
        Value addr = memAddr(m, bs.offset);
        for (unsigned k = 0; k < banks.size(); ++k)
          seq::WritePortOp::create(c.b, c.loc, banks[k], ValueRange{addr}, data,
                                   c.andBits(we, c.icmpEq(bs.bank, k)), wlat);
      }
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
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
          pa.getInput(portData(streamPortBase(dp, s)));
    }
}

// The latency-insensitive shell's port drives + control signals for region
// \p rb (freeze only on output back-pressure). A put drives
// `_data`/`_valid`; a get drives `_ready`. Only a full output freezes the
// pipeline (`chainEnable = ~outputFull`); an empty input injects a bubble by
// dropping `issueEnable`, never a freeze: freezing on starvation would hold a
// mid-flight `valid` high and let a ready consumer double-capture the token. A
// stage-0 access keys on the UNgated `wantIssue` so the signals stay
// combinationally acyclic, a deeper access on the (registered) delayed issue.
// A predicated access (`acc.when` set) additionally gates its handshake on
// the predicate so a token is consumed/produced only where it holds; the
// predicate is a datapath value (no FIFO status), so acyclicity is preserved.
void DatapathEmitter::emitStreamAccesses(const uarch::RegionBlock &rb,
                                         Value issue, DatapathFeedback &fb) {
  // LI-shell invariants, checked only when in scope: one access per channel;
  // a starved stage-0 input bubbles at II==1 but freezes `chainEnable` at
  // II>1 (phase/chain sync); a starved stage>0 input always freezes it.
  bool hasStream = false;
  for (const uarch::StreamChannel &s : dp.streams) {
    bool here = false;
    for (const uarch::StreamChannel::Access &acc : s.accesses)
      if (acc.region == rb.id)
        here = hasStream = true;
    assert((!here || s.accesses.size() <= 1) &&
           "one access per stream channel");
    (void)here;
  }
  (void)hasStream;

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
      auto base = streamPortBase(dp, s);
      // A predicated put produces a token only where its predicate holds:
      // gate `valid`, and suppress the output-full hazard when it is low, so
      // the pipeline never freezes waiting for space it won't write this
      // firing.
      Value pred = acc.when ? resolveSource(acc.when) : Value();
      Value valid = c.activationPulse(issue, acc.op);
      if (pred)
        valid = c.andBits(valid, pred);
      pa.setOutput(portData(base), resolveSource(acc.data));
      pa.setOutput(portValid(base), valid);
      // A stage-0 put keys its hazard on wantIssue (ungated) & pred; a stage>=1
      // put's valid is already registered (delayed) and predicate-gated.
      Value active = acc.stage == 0 ? atIssue : valid;
      if (pred && acc.stage == 0)
        active = c.andBits(active, pred);
      Value hz = c.andBits(active, c.notBit(pa.getInput(portReady(base))));
      outHazard = outHazard ? c.orBits(outHazard, hz) : hz;
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, acc.stage);
    }
  // Mid-pipeline freeze: a stage>0 get with a needed-but-empty input can't
  // bubble past a missing token, so fold that stall into `chainEnable`
  // alongside the output-full freeze; `active`/predicate are registered, so
  // this reads only stored state.
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
          want, c.notBit(pa.getInput(portValid(streamPortBase(dp, s)))));
      midStall = midStall ? c.orBits(midStall, miss) : miss;
    }
  Value chainEnable = outHazard ? c.notBit(outHazard) : c.t1;
  if (midStall)
    chainEnable = c.andBits(chainEnable, c.notBit(midStall));

  // Stage-0 inputs (read at issue) fold into `stage0Valid`, the issue gate; a
  // predicated get treats a non-needed input as available (`valid | ~pred`).
  // With >1 stage-0 get they must pop together, so their readies gate on it
  // too.
  Value stage0Valid;
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses) {
      if (acc.isPut || acc.region != rb.id || acc.stage != 0)
        continue;
      Value valid = pa.getInput(portValid(streamPortBase(dp, s)));
      if (acc.when)
        valid = c.orBits(valid, c.notBit(resolveSource(acc.when)));
      stage0Valid = stage0Valid ? c.andBits(stage0Valid, valid) : valid;
    }
  bool join0 = stage0Gets > 1;

  // Modulo (II>1) cadence: a starved stage-0 slot can't be a mere issue-skip,
  // since the phase counter and shift chains must freeze together or tap
  // alignment desyncs. Fold it into `chainEnable` gated by `atIssue`, so
  // `issueEnable == chainEnable` gates phase, counter, chains, and Ce operators
  // as one.
  bool modulo = rb.ii.value_or(1) > 1;
  if (modulo && stage0Valid)
    chainEnable = c.andBits(
        chainEnable, c.notBit(c.andBits(atIssue, c.notBit(stage0Valid))));

  // Drive each `_ready`: a stage-0 get accepts when issuing and not frozen
  // (a join also waits for all stage-0 inputs); a deeper get accepts when
  // the chain advances; a predicated get pops only where its predicate holds.
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
      pa.setOutput(portReady(streamPortBase(dp, s)), ready);
    }
  nameValue(chainEnable, regionSignal(rb.id, "ce"));
  fb.chainEnable = chainEnable;
  fb.issueEnable = modulo ? chainEnable
                          : (stage0Valid ? c.andBits(chainEnable, stage0Valid)
                                         : chainEnable);
}

// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
// hw.instance. The child masters each memref operand's memory: it drives the
// addr/data/we, so the leaf wires those instance-output ports to the buffer's
// hlmem (a seq.read whose data feeds back to the child, a seq.write). The
// region's completion is the child's real `done` (fb.callDone). Serial
// execution (a producer region drains before the child starts, the child before
// a consumer) means one master per port at a time: no arbitration mux.
void DatapathEmitter::emitCalls(const uarch::RegionBlock &rb, Value issue,
                                DatapathFeedback &fb) {
  // Calls sharing a region (a straight-line span) each start on the joined
  // `done` of predecessors they depend on (a shared buffer/boundary or a
  // scalar result), or on region `issue` if none, running concurrently with
  // siblings; the region completes when every call's done is set.
  SmallVector<Value> dones;                        // each call's done, by index
  SmallVector<SmallVector<uarch::MemId>> callMems; // each call's touched MemIds
  SmallVector<unsigned> callStart; // each call's scheduled start
  SmallVector<bool> callIndet;     // each call's indeterminacy
  llvm::DenseMap<uarch::CallId, Value>
      doneByCid; // done by id (scalar hand-off)
  for (uarch::CallId cid : rb.callUnits) {
    const uarch::CallUnit &cu = dp.calls[cid];
    // A shared-memref predecessor is an earlier call (by schedule `start`)
    // touching a common MemId; the consumer starts on their joined `done`.
    // Same-offset calls run concurrently; an indeterminate (while-leaf)
    // producer still gates its sharer via `done` despite sharing its start.
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

    // Instance inputs by child port name: clk/rst/`start` (the region's
    // issue pulse) plus each read's data input. An internal read consumes a
    // backedge (resolved after the instance); a boundary read passes the top's
    // data input straight through.
    llvm::StringMap<Value> ins;
    ins[kClk] = c.clkRaw;
    ins[kRst] = c.rst;
    ins[kStart] = startK;
    llvm::StringMap<circt::Backedge> rdBackedge;
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (ma.isWrite)
        continue;
      if (ma.isBoundary)
        ins[ma.data] = pa.getInput(portData(ma.topBase));
      else {
        auto be = c.bb.get(memElemType(dp.mems[ma.mem], c.b));
        ins[ma.data] = be;
        rdBackedge.try_emplace(ma.data, be);
      }
    }
    // Scalar operands: drive each child scalar-input port from its resolved
    // Source (an IO port, a latched sibling survivor, an earlier child's
    // live result, or a constant), sampled at the child's start.
    for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
      ins[sa.port] = resolveSource(sa.src);

    // Wire the child instance: inputs by port name from `ins`, outputs by name.
    auto outs = instantiateChild(c.b, c.loc, child,
                                 childInstanceName(cu.callee, cu.id), ins);

    // Scalar results: the child holds each result on its output port from
    // `done` onward, so that port value IS the survivor a sibling reads (the
    // `done` handshake gates it, no separate capture); callResultVal serves a
    // live same-region reader, survivorOf a cross-region one, both the same
    // wire.
    for (auto [r, port] : llvm::enumerate(cu.resultPorts)) {
      callResultVal[accKey(cu.id, r)] = outs[port];
      setSurvivor(cu.region, r, outs[port]);
    }

    // Master each buffer from the child's addr/data/we outputs: a boundary
    // arg passes through to the top boundary port (flat i32 address); an
    // internal buffer drives its hlmem, narrowed to the clog2(depth) index, at
    // the RAM latency the child was compiled against.
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (ma.isBoundary) {
        // One port group per accessor, driven directly from the child's
        // addr/data/we: concurrent masters get distinct groups (no mux); a
        // serial pair also uses two groups, each active only in its own phase
        // (self-gated we==0 elsewhere).
        pa.setOutput(portAddr(ma.topBase), outs[ma.addr]);
        if (ma.isWrite) {
          pa.setOutput(portData(ma.topBase), outs[ma.data]);
          pa.setOutput(portWe(ma.topBase), outs[ma.we]);
        }
        continue;
      }
      const uarch::MemUnit &m = dp.mems[ma.mem];
      // One hlmem per bank: the child masters bank `ma.bank` (already
      // indexed in that bank's own space via `allo.part`), so route straight
      // to memBanks[m.id][bank], no crossbar; parent and child bank counts
      // agree by construction, so assert the index is in range.
      assert(ma.bank < memBanks[m.id].size() &&
             "child bank index exceeds the buffer's bank count (parent/callee "
             "partition-factor disagreement)");
      Value hlmem = memBanks[m.id][ma.bank];
      Value addr = memAddr(m, outs[ma.addr]);
      // The child was compiled against this buffer's device latency (carried
      // on its own `dcp.load`/`dcp.store`), so the parent drives the port at
      // the same latency, read from the MemUnit since the parent never accesses
      // the buffer itself.
      if (ma.isWrite)
        seq::WritePortOp::create(c.b, c.loc, hlmem, ValueRange{addr},
                                 outs[ma.data], outs[ma.we],
                                 c.b.getI64IntegerAttr(m.writeLatency));
      else
        rdBackedge[ma.data].setValue(
            c.R(seq::ReadPortOp::create(c.b, c.loc, hlmem, ValueRange{addr},
                                        /*rdEn=*/Value(), m.readLatency)));
    }
    doneByCid[cu.id] = outs[kDone];
    dones.push_back(outs[kDone]);
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
  declareUnits(rb); // unit backedges must exist before a read address resolves
  emitInternalReads(rb);
  emitExternalReads(rb);
  emitUnits(rb);
  resolveRegHeads(rb);
  auto fb = emitAccesses(rb, issue);
  emitStreamAccesses(rb, issue, fb);
  emitCalls(rb, issue, fb);
  return fb;
}

} // namespace mlir::allo::uarch
