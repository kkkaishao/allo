/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h" // kIndependentWritesAttr
#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/AddressModel.h" // addressExprsOf
#include "allo/Scheduling/MemoryModel.h"  // BankLayout

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// --- Memory-banking crossbar primitives -------------------------------------

// A literal of \p v's own width. Address arithmetic is carried at whatever
// width the addressed memory needs, so every operand of a `comb` op below has
// to be built against the value it accompanies rather than a fixed i32.
static Value konstLike(OpBuilder &b, Location loc, Value v, int64_t k) {
  return hw::ConstantOp::create(b, loc, v.getType(), k).getResult();
}

// An address, a bank digit and a scaled counter are all non-negative by
// construction, so every width change on the address path is the UNSIGNED
// resize (`uarch::resize`).
static Value addrAt(OpBuilder &b, Location loc, Value v, unsigned width) {
  return resize(b, loc, v, width, /*isSigned=*/false);
}

// Unsigned divide by a compile-time constant: a shift for a power of two, else
// a real divider (synthesis folds a constant divisor into a multiply-shift).
static Value divConst(OpBuilder &b, Location loc, Value v, int64_t d) {
  if (d == 1)
    return v;
  if (llvm::isPowerOf2_64(d))
    return comb::ShrUOp::create(b, loc, v,
                                konstLike(b, loc, v, llvm::Log2_64(d)), false)
        .getResult();
  return comb::DivUOp::create(b, loc, v, konstLike(b, loc, v, d), false)
      .getResult();
}

// Multiply by a compile-time constant. A power-of-two coefficient is a shift;
// anything else stays a `comb.mul` deliberately, since synthesis recodes a
// constant multiplier into a shift-add network better than a decomposition
// emitted here could.
static Value mulConst(OpBuilder &b, Location loc, Value v, int64_t k) {
  if (k == 1)
    return v;
  if (k > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(k)))
    return comb::ShlOp::create(b, loc, v,
                               konstLike(b, loc, v, llvm::Log2_64(k)), false)
        .getResult();
  return comb::MulOp::create(b, loc, v, konstLike(b, loc, v, k), false)
      .getResult();
}

static Value modConst(OpBuilder &b, Location loc, Value v, int64_t d) {
  if (d == 1)
    return konstLike(b, loc, v, 0);
  if (llvm::isPowerOf2_64(d))
    return comb::AndOp::create(b, loc, v, konstLike(b, loc, v, d - 1), false)
        .getResult();
  return comb::ModUOp::create(b, loc, v, konstLike(b, loc, v, d), false)
      .getResult();
}

// The bits one bank of \p m needs to address itself, which is the width its
// whole address cone is carried at.
static unsigned addrWidth(const uarch::MemUnit &m) {
  return llvm::Log2_64_Ceil(declaredDepth(m.depthWords));
}

// An address on its way OUT of the module. A boundary address port is
// `kDatapathAddressWidth` wide for every argument, the fixed contract the
// manifest and the cosim harness are written against, so a narrow in-bank
// address widens back here.
static Value boundaryAddr(EmitContext &c, Value addr) {
  return addrAt(c.b, c.loc, addr, kDatapathAddressWidth);
}

Value readCrossbar(EmitContext &c, ArrayRef<Value> bankValues, Value bank) {
  Value out = bankValues[0]; // bank 0 falls through the priority chain
  for (unsigned k = 1; k < bankValues.size(); ++k)
    out = c.mux(c.icmpEq(bank, k), bankValues[k], out);
  return out;
}

Value writeDemux(EmitContext &c, Value we, Value bank, unsigned k) {
  return bank ? c.andBits(we, c.icmpEq(bank, k)) : we;
}

// Which of several sources bank \p k takes, each tagged with the bank IT
// reaches: the inverse of `readCrossbar`. At most one tag equals `k` at a time,
// because a lane holds distinct slots and distinct slots are distinct banks at
// every rotation, so the priority order carries no meaning and the first arm is
// a fall-through.
static Value laneSelect(EmitContext &c,
                        ArrayRef<std::pair<Value, Value>> tagged, unsigned k) {
  Value out = tagged.front().second;
  for (const auto &[bank, val] : tagged.drop_front())
    out = c.mux(c.icmpEq(bank, k), val, out);
  return out;
}

// Resolve a datapath Source to the SSA value driving it, exhaustive over
// Source::Kind.
Value DatapathEmitter::resolveSource(const uarch::Source &s) {
  switch (s.kind) {
  case uarch::Source::Kind::Unit: {
    // A same-region operator result. `declareUnits` declares its backedge
    // before any read resolves, so a miss means this consumer sits outside the
    // owning region.
    Value v = unitVal.lookup(s.id);
    assert(v && "unit source read outside the region that declared it");
    return v;
  }
  case uarch::Source::Kind::Reg:
    return regStages[s.id].tap(s.outPort);
  case uarch::Source::Kind::Mem:
    return readData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Stream:
    // An input stream's loaded token, bound by bindStreamReads before any
    // consumer, like a memory read.
    return streamReadData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Counter: {
    // The iteration counter of Source's region (an outer container's counter is
    // live while its nested region emits), at `kIndexWidth` whatever width the
    // region built its register at.
    Value cv = counterIndex.lookup(s.id);
    assert(cv && "counter source with no emitted region counter");
    return cv;
  }
  case uarch::Source::Kind::Const: {
    // The datapath carries a value as its bit pattern, so a float literal ties
    // in as its bitcast integer.
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
    // A shared unit's input: the bound ops hold disjoint MRT residues, so the
    // `activationPulse` selects are one-hot and an AND-OR reduction serves.
    // With no op issuing the result is zero, which no consumer samples.
    if (Value v = muxVal.lookup(s.id))
      return v;
    const uarch::Mux &mx = dp.muxes[s.id];
    Value issue = controlOf.lookup(mx.region).issue;
    assert(issue && "mux in a region with no controller");
    // Timed against the OWNING region's shell (`mx.region`), not whichever
    // region is emitting: the select rides that region's issue pulse.
    StallShell sh = shellFor(mx.region);
    // A recurrence operand's iteration windows, delayed to their op's stage and
    // built once per (op, iteration). The At arms and the From arm that
    // complements them partition that op's pulse by construction. The two kinds
    // are cached apart because one op may carry recurrences of different
    // distances, whose `iter` numbers then mean different things.
    DenseMap<std::pair<Operation *, unsigned>, Value> atOf, fromOf;
    SmallVector<Value> values, selects;
    for (auto [k, src] : llvm::enumerate(mx.sources)) {
      Operation *op = mx.selectOps[k];
      const uarch::Mux::Phase &ph = mx.phases[k];
      const uarch::RegionBlock &rb = dp.regions[mx.region];
      Value sel = c.activationPulse(issue, op, sh);
      if (ph.kind == uarch::Mux::Phase::At) {
        Value &window = atOf[{op, ph.iter}];
        if (!window)
          window = c.activationPulse(atIteration(rb, ph.iter), op, sh);
        sel = c.andBits(sel, window);
      } else if (ph.kind == uarch::Mux::Phase::From) {
        Value &window = fromOf[{op, ph.iter}];
        if (!window)
          window = c.activationPulse(firstIterations(rb, ph.iter), op, sh);
        sel = c.andBits(sel, c.notBit(window));
      }
      values.push_back(resolveSource(src));
      selects.push_back(sel);
    }
    Value v = c.oneHotSelect(values, selects);
    muxVal[s.id] = v;
    return v;
  }
  case uarch::Source::Kind::Survivor: {
    // A sibling region's held result, latched by setSurvivor when the producing
    // region completed, before this consumer emitted.
    Value sv = survivorOf.lookup(accKey(s.id, s.outPort));
    assert(sv && "survivor source read before its region was captured");
    return sv;
  }
  case uarch::Source::Kind::Call: {
    // A sub-kernel call's scalar result: the child instance's result output,
    // populated by emitCalls before any consumer.
    Value cv = callResultVal.lookup(accKey(s.id, s.outPort));
    assert(cv && "call result source read before its CallUnit was emitted");
    return cv;
  }
  case uarch::Source::Kind::None:
    // `validateDatapath` rejects a None Source earlier. Not an `assert`: under
    // NDEBUG that would fall through and hand the caller a null Value.
    llvm_unreachable("unresolved (None) source reached emission");
  }
  llvm_unreachable("unhandled Source::Kind");
}

Value DatapathEmitter::ivAt(const uarch::RegionBlock &rb, unsigned n,
                            Value lb) {
  if (!n)
    return {};
  auto ivTy = cast<IntegerType>(rb.counterType);
  std::optional<int64_t> kstep = dp.constantOf(rb.stepSource);
  Value nStep = kstep ? c.konst(ivTy, static_cast<int64_t>(n) * *kstep)
                      : c.R(comb::MulOp::create(
                            c.b, c.loc, c.konst(ivTy, static_cast<int64_t>(n)),
                            resize(c.b, c.loc, resolveSource(rb.stepSource),
                                   ivTy.getWidth(), /*isSigned=*/true),
                            false));
  return c.R(comb::AddOp::create(c.b, c.loc, lb, nStep, false));
}

// Both at the raw counter register's width, the width its terminator compares
// at, so a bound from elsewhere resizes into it.
std::pair<Value, Value>
DatapathEmitter::counterAndLb(const uarch::RegionBlock &rb) {
  Value iv = controlOf.lookup(rb.id).counter;
  assert(iv && "a recurrence input in a region with no iteration counter");
  unsigned w = cast<IntegerType>(rb.counterType).getWidth();
  return {iv, resize(c.b, c.loc, resolveSource(rb.lbSource), w,
                     /*isSigned=*/true)};
}

Value DatapathEmitter::firstIterations(const uarch::RegionBlock &rb,
                                       unsigned dist) {
  auto [iv, lb] = counterAndLb(rb);
  if (dist <= 1)
    return c.icmpEqV(iv, lb);
  // iv < lb + dist*step == !(iv >= lb + dist*step). Signed, as
  // `Terminator::isLast` compares the same counter against the same kind of
  // bound; an unsigned predicate would order a negative `lb` wrongly.
  return c.notBit(c.icmpSgeV(iv, ivAt(rb, dist, lb)));
}

Value DatapathEmitter::atIteration(const uarch::RegionBlock &rb,
                                   unsigned iter) {
  auto [iv, lb] = counterAndLb(rb);
  Value at = ivAt(rb, iter, lb);
  return c.icmpEqV(iv, at ? at : lb);
}

unsigned DatapathEmitter::readyCycle(const uarch::Source &s) const {
  // A call is the one producing op whose result does NOT land at
  // `dcpStart + dcpLatency`: it lands at its region-relative issue plus the
  // CALLEE's whole start->done depth. Indeterminate calls are guarded earlier.
  if (s.kind == uarch::Source::Kind::Call) {
    const uarch::CallUnit &cu = dp.calls[s.id];
    assert(cu.latency && "readyCycle of an indeterminate call result");
    return cu.start + static_cast<unsigned>(*cu.latency);
  }
  // A held source has no landing stage: a literal is constant, an IO port
  // stable for the whole kernel, and a counter or survivor a register settled
  // by the time the region reading it issues.
  if (s.kind == uarch::Source::Kind::Const ||
      s.kind == uarch::Source::Kind::IO ||
      s.kind == uarch::Source::Kind::Counter ||
      s.kind == uarch::Source::Kind::Survivor)
    return 0;
  Operation *op = dp.producingOp(s);
  assert(op && "readyCycle only modelled for a Unit / memory read / "
               "stream get / constant / call result");
  return readyCycleOf(op);
}

// Evaluate an affine index expression to a hw value \p width bits wide,
// emitting comb ops. `idx` holds the resolved value of each map operand (dims
// then symbols), each at the datapath width. Shared by the two places a map
// reaches the datapath: a memory access's address (bankAddress) and a
// standalone affine.apply (emitCompute).
Value evalAffine(OpBuilder &b, Location loc, AffineExpr e, ValueRange idx,
                 unsigned numDims, unsigned width) {
  Type t = b.getIntegerType(width);
  if (auto cst = dyn_cast<AffineConstantExpr>(e))
    return hw::ConstantOp::create(b, loc, t, cst.getValue()).getResult();
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return addrAt(b, loc, idx[d.getPosition()], width);
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return addrAt(b, loc, idx[numDims + sym.getPosition()], width);
  auto bin = cast<AffineBinaryOpExpr>(e);
  if (e.getKind() == AffineExprKind::Add)
    return comb::AddOp::create(
               b, loc, evalAffine(b, loc, bin.getLHS(), idx, numDims, width),
               evalAffine(b, loc, bin.getRHS(), idx, numDims, width), false)
        .getResult();
  if (e.getKind() == AffineExprKind::Mul) {
    Value lhs = evalAffine(b, loc, bin.getLHS(), idx, numDims, width);
    // An affine coefficient is always constant, so this is a shift-or-multiply
    // rather than a general multiplier. A semi-affine map is representable
    // though, so a non-constant one still lowers.
    if (auto k = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return mulConst(b, loc, lhs, k.getValue());
    return comb::MulOp::create(
               b, loc, lhs,
               evalAffine(b, loc, bin.getRHS(), idx, numDims, width), false)
        .getResult();
  }
  // floordiv/mod by a constant is delinearization left by a coalesced nest over
  // a non-negative index. Neither is congruent modulo 2^width, so both compute
  // wide and narrow afterwards.
  auto rc = dyn_cast<AffineConstantExpr>(bin.getRHS());
  assert(rc && rc.getValue() > 0 &&
         "affine div/mod by a non-constant or non-positive divisor");
  int64_t f = rc.getValue();
  // With one congruent exception, the one a bank digit always ends in: `x mod
  // 2^k` IS the low k bits, so that subtree is built k bits wide and the mask
  // disappears with it. `addressCost` prices it at the same narrowed width.
  if (e.getKind() == AffineExprKind::Mod && f > 1 &&
      llvm::isPowerOf2_64(static_cast<uint64_t>(f))) {
    unsigned k =
        std::min<unsigned>(width, llvm::Log2_64(static_cast<uint64_t>(f)));
    return addrAt(b, loc, evalAffine(b, loc, bin.getLHS(), idx, numDims, k),
                  width);
  }
  Value lhs =
      evalAffine(b, loc, bin.getLHS(), idx, numDims, kDatapathAddressWidth);
  assert((e.getKind() == AffineExprKind::FloorDiv ||
          e.getKind() == AffineExprKind::Mod) &&
         "unexpected affine op");
  Value q = e.getKind() == AffineExprKind::FloorDiv ? divConst(b, loc, lhs, f)
                                                    : modConst(b, loc, lhs, f);
  return addrAt(b, loc, q, width);
}

// The resolved (already stage-delayed) index sources of an access, dims then
// symbols.
SmallVector<Value>
DatapathEmitter::addrSources(const uarch::MemUnit::Access &acc) {
  SmallVector<Value> idx;
  for (const uarch::Source &s : acc.addr)
    idx.push_back(resolveSource(s));
  return idx;
}

// Build one cone \p r of this access's address as hardware at \p width, out of
// the parts `planAddressGenerators` split it into: a constant, one register per
// strength-reduced term (`RegionBlock::addrStrides`, advanced by the
// controller), and whatever did not reduce. The residual is added after the
// delay chain because ITS operands arrive already delayed, where the counters
// run live, which puts both halves in the access's own cycle.
Value DatapathEmitter::buildAddr(const uarch::MemUnit::Access &acc,
                                 const uarch::MemUnit::Access::Reduced &r,
                                 unsigned width) {
  Value addr;
  auto add = [&](Value v) {
    addr =
        addr ? comb::AddOp::create(c.b, c.loc, addr, v, false).getResult() : v;
  };
  if (r.base)
    add(c.konst(c.b.getIntegerType(width), r.base));
  for (const uarch::MemUnit::Access::ScaledTerm &t : r.terms) {
    const uarch::RegionControl &rc = controlOf.lookup(t.region);
    assert(t.slot < rc.scaledCounters.size() &&
           "a reduced address term has no scaled counter in its region");
    add(addrAt(c.b, c.loc, rc.scaledCounters[t.slot], width));
  }
  if (addr && acc.addrDelay)
    addr = c.shiftChain(addr, acc.addrDelay, shellFor(acc.region)).last();
  if (r.residual) {
    // A register the residual reads runs live like a counter, so each is
    // delayed on its own (summed terms share one delay only by being summed
    // first). Appended at the datapath width, which is what `evalAffine` reads
    // its operands at.
    SmallVector<Value> idx = addrSources(acc);
    for (const uarch::MemUnit::Access::ScaledTerm &t : r.reads) {
      const uarch::RegionControl &rc = controlOf.lookup(t.region);
      assert(t.slot < rc.scaledCounters.size() &&
             "a residual's digit has no scaled counter in its region");
      Value v =
          addrAt(c.b, c.loc, rc.scaledCounters[t.slot], kDatapathAddressWidth);
      if (acc.addrDelay)
        v = c.shiftChain(v, acc.addrDelay, shellFor(acc.region)).last();
      idx.push_back(v);
    }
    add(evalAffine(c.b, c.loc, r.residual, idx, acc.addrMap.getNumDims(),
                   width));
  }
  // Nothing at all: the access sits at a fixed element of a one-word bank.
  return addr ? addr : c.konst(c.b.getIntegerType(width), 0);
}

// The address hardware of one access: the element index within the bank it
// reaches, plus the bank digit when that is decided at runtime. Uniform over
// banked and unbanked, since an unpartitioned memref is a one-bank layout whose
// offset is the flat index and whose digit nothing builds.
//
// Both halves are derived SYMBOLICALLY (`addressExprsOf`) and only then
// evaluated: composing the row-major strides on a coalesced nest's
// `iv -> (iv floordiv N, iv mod N)` cancels it back to `iv`, where the same
// thing built out of `comb` ops is a multiply, a mask and a shift that no later
// pass can fold away.
BankSplit DatapathEmitter::bankAddress(const uarch::MemUnit &m,
                                       const uarch::MemUnit::Access &acc) {
  assert(acc.addrMap && "dcp memory access without an affine map");
  ArrayRef<int64_t> shape = cast<MemRefType>(m.memref.getType()).getShape();
  AddressExprs e = addressExprsOf(m.layout, acc.addrMap, shape, acc.staticBank);
  assert(e.width == addrWidth(m) &&
         "the width the address was priced at is not the one it is built at");
  Value offset = buildAddr(acc, acc.offset, e.width);
  // The digit stays at the datapath width, being compared against literal bank
  // numbers rather than used as an address. It reduces like the offset:
  // `counter mod F` is a register that wraps, not a `mod` on the setup path.
  Value bank =
      e.bank ? buildAddr(acc, acc.bank, kDatapathAddressWidth) : Value();
  // A read freezes its address on stall or the in-flight read is lost (KPN);
  // a write skips the stalled cycle through its gated write enable. Both
  // halves freeze together or they name different elements.
  if (!acc.isWrite) {
    StallShell sh = shellFor(acc.region);
    if (bank)
      bank = c.stallHold(bank, sh);
    offset = c.stallHold(offset, sh);
  }
  return {bank, offset};
}

// Narrow to the clog2(depth)-bit index `seq.hlmem` / `hw.array_get` expects,
// which is also the width `bankAddress` carries its arithmetic at.
Value DatapathEmitter::memAddr(const uarch::MemUnit &m, Value addr) {
  return addrAt(c.b, c.loc, addr, addrWidth(m));
}

// Which element of a scattered argument \p acc names, at the DATAPATH width.
// The crossbar and the write demux compare it against literal element numbers
// (`icmpEq` builds those at that width).
Value DatapathEmitter::scatterIndex(const uarch::MemUnit &m,
                                    const uarch::MemUnit::Access &acc) {
  assert(m.scattered && "an element index belongs to a scattered argument");
  return addrAt(c.b, c.loc, bankAddress(m, acc).offset, kDatapathAddressWidth);
}

// Bind the read-data input ports into readData, once, before the per-region
// loop (external memories only; internal ones read via seq.read below). A
// data-dependent banked read has one data port per bank and is bound by
// emitExternalReads, which muxes them in-region.
void DatapathEmitter::bindReadPorts() {
  for (uarch::AccRef r : dp.readPorts) {
    const uarch::MemUnit &m = dp.mems[r.id];
    auto eb = externalBank(m, m.accesses[r.idx]);
    if (eb.factor > 1 && !eb.bank)
      continue; // data-dependent: bound by emitExternalReads
    readData[accKey(r.id, r.idx)] =
        pa.getInput(portData(extPorts(m, m.accesses[r.idx]).front().second));
  }
}

// Instantiate on-chip storage for each internal (non-argument) memory: one
// seq.hlmem, or one per bank when the array reached emit still partitioned (a
// data-dependent bank `dcp-resolve-banking` could not split statically). The
// handles are module-scope so writes and reads in different regions share them.
void DatapathEmitter::createInternalMemories() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (m.external)
      continue;
    IntegerType elemTy = memElemType(m, c.b);
    unsigned depth = declaredDepth(m.depthWords);
    if (m.isRom) {
      // A constant table: one hw.aggregate_constant holding the global's
      // initializer, read combinationally by hw.array_get and registered to the
      // read latency in emitInternalReads. No writable hlmem, no write ports.
      SmallVector<Attribute> fields;
      for (const APInt &w :
           initWords(cast<ElementsAttr>(m.romInit), m.width, depth))
        fields.push_back(IntegerAttr::get(elemTy, w));
      // A hw.array indexes element 0 as the LAST aggregate_constant field, so
      // the natural-order initializer is reversed to make array_get(i) ==
      // data[i].
      std::reverse(fields.begin(), fields.end());
      romArray[m.id] = hw::AggregateConstantOp::create(
          c.b, c.loc, hw::ArrayType::get(elemTy, depth),
          c.b.getArrayAttr(fields));
      continue;
    }
    // Stores that provably never issue together share a write port. A skewed
    // array presents no single addressable port, and a dynamically banked store
    // drives every bank behind a demux, so neither has a port to be coloured
    // onto. `Datapath::maxWritePorts` is the ceiling: past it the RAM inference
    // fails outright, so a further colour would buy nothing and still cost its
    // address and data muxes.
    std::optional<SmallVector<unsigned>> ports;
    if (!m.skewed &&
        llvm::all_of(m.accesses, [](const uarch::MemUnit::Access &a) {
          return !a.isWrite || a.staticBank;
        }))
      ports = dp.writePortColouring(m.id, dp.maxWritePorts);
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k) {
      auto mem =
          seq::HLMemOp::create(c.b, c.loc, c.clk, c.rst, memCellName(dp, m, k),
                               {static_cast<int64_t>(depth)}, elemTy);
      // The colouring is exactly the promise the lowering needs to describe
      // each port in its own `always` block, and so to infer a true dual port.
      // A port per static write instead drops the array into a register file.
      if (ports)
        mem->setAttr(kIndependentWritesAttr, c.b.getUnitAttr());
      // An initialized array the kernel also WRITES is a real memory that
      // merely starts with contents. `seq.hlmem` carries no initializer, so the
      // words ride to the seq->SV pipeline, which gives the backing reg an
      // `initial` block.
      if (m.romInit)
        recordMemoryInit(
            mem, initWords(cast<ElementsAttr>(m.romInit), m.width, depth));
      banks.push_back(mem.getHandle());
    }
    memBanks[m.id] = std::move(banks);
    if (ports)
      writePortOf[m.id] = std::move(*ports);
  }
}

// Shift-register chains for region \p rb's registers (index delays, pipeline
// holds). Each chain's head input is a backedge resolved once the units exist.
void DatapathEmitter::emitRegisters(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  // A published phase is the foldability condition: only a schedule-paced
  // controller at II > 1 publishes one, and only there does one iteration land
  // every `ii` cycles. A depth-1 chain is one register either way.
  Value phase = controlOf.lookup(rb.id).phase;
  unsigned ii = rb.ii.value_or(1);
  assert((!phase || ii > 1) && "a phase was published for a region at II 1");
  for (uarch::RegId rid : rb.regs) {
    const uarch::Register &rg = dp.regs[rid];
    auto head = c.bb.get(hwType(rg.type, c.b));
    regHeadBE.try_emplace(rg.id, head);
    // A register is a plain delay chain; reduction-identity re-injection rides
    // the consuming unit's recurrence input (emitUnits), not the register.
    regStages[rg.id] =
        phase && rg.depth > 1
            ? c.foldedChain(head, rg.depth, ii, phase, rg.ready, sh)
            : c.shiftChain(head, rg.depth, sh);
    // Name each held stage `<value>_d<k>`. Stage 0 is the undelayed input,
    // already named by its producer, so leave it alone rather than relabel a
    // shared wire. A folded chain repeats one register across the `ii` taps it
    // serves, so name it once, at the shallowest delay it provides.
    std::string owner = ownerOf(rg.value, regOwner(rg.id));
    auto &taps = regStages[rg.id].stages;
    for (unsigned k = 1; k < taps.size(); ++k)
      if (taps[k] != taps[k - 1])
        nameValue(taps[k], regTapName(owner, k));
  }
}

// The skewed twin of `emitInternalReads`: one read port per bank per LANE
// rather than per bank per access. A lane's accesses hold distinct slots, so
// they reach distinct banks at every rotation, and bank k can take the offset
// of whichever of them reaches it and hand its datum back to that one: F
// accesses over F banks at one port each, where a crossbar would take a port on
// every bank for every access.
void DatapathEmitter::emitSkewedInternalReads(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    if (m.skewed && !m.accesses[r.idx].isWrite)
      lanes[{r.id, m.accesses[r.idx].lane}].push_back(r.idx);
  }
  for (auto &[key, idxs] : lanes) {
    const uarch::MemUnit &m = dp.mems[key.first];
    ArrayRef<Value> banks = memBanks[m.id];
    unsigned lat = m.readLatency;
    SmallVector<std::pair<Value, Value>> tagged; // (runtime bank, in-bank addr)
    for (unsigned i : idxs) {
      BankSplit bs = bankAddress(m, m.accesses[i]);
      tagged.emplace_back(bs.bank, memAddr(m, bs.offset));
    }
    SmallVector<Value> vals;
    for (unsigned k = 0; k < banks.size(); ++k)
      vals.push_back(c.R(seq::ReadPortOp::create(
          c.b, c.loc, banks[k], ValueRange{laneSelect(c, tagged, k)},
          /*rdEn=*/Value(), lat)));
    // Each access picks its own bank's datum back out, delayed with it.
    for (auto [i, t] : llvm::zip(idxs, tagged)) {
      Value sel = lat ? c.shiftChain(t.first, lat, sh).last() : t.first;
      readData[accKey(m.id, i)] = readCrossbar(c, vals, sel);
    }
  }
}

// seq.read for each internal-memory read scheduled in region \p rb, bound into
// readData BEFORE emitUnits consumes it. Read latency is the memory's
// device-resolved `readLatency`, the number the scheduler timed the access at,
// so the datum lands on exactly the cycle the consumer's register depth was
// solved against.
void DatapathEmitter::emitInternalReads(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  emitSkewedInternalReads(rb);
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (m.external || acc.isWrite || m.skewed)
      continue;
    unsigned lat = m.readLatency;
    if (m.isRom) {
      // A constant table read: index the aggregate_constant combinationally,
      // then register to the scheduled read latency so timing matches a RAM.
      Value idx = memAddr(m, bankAddress(m, acc).offset);
      Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id], idx));
      readData[accKey(m.id, r.idx)] =
          lat ? c.shiftChain(elem, lat, sh).last() : elem;
      continue;
    }
    ArrayRef<Value> banks = memBanks[m.id];
    auto readAt = [&](Value handle, Value addr) {
      return c.R(seq::ReadPortOp::create(c.b, c.loc, handle, ValueRange{addr},
                                         /*rdEn=*/Value(), lat));
    };
    Value rd;
    if (acc.staticBank) {
      // A compile-time bank reads its own memory: no crossbar, and no read port
      // on the other banks. An unbanked memref is the same case at bank 0.
      rd = readAt(banks[*acc.staticBank],
                  memAddr(m, bankAddress(m, acc).offset));
    } else {
      // Read every bank at the (bank-independent) offset, then select by the
      // runtime bank, aligned with the read data (delayed by the latency).
      auto bs = bankAddress(m, acc);
      Value addr = memAddr(m, bs.offset);
      SmallVector<Value> vals;
      for (Value h : banks)
        vals.push_back(readAt(h, addr));
      Value sel = lat ? c.shiftChain(bs.bank, lat, sh).last() : bs.bank;
      rd = readCrossbar(c, vals, sel);
    }
    readData[accKey(m.id, r.idx)] = rd;
  }
}

// Read crossbar for each data-dependent external (argument) read in region
// \p rb: drive every bank interface's address with the in-bank offset, read
// each bank's data port, and mux by the runtime bank, delayed to the memory's
// device read latency so the select aligns with its data. The twin of
// emitInternalReads for boundary ports instead of hlmems.
void DatapathEmitter::emitExternalReads(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (!m.external || acc.isWrite)
      continue;
    // A scattered argument has no address port: a read is a crossbar over every
    // element selected by the index, and a constant index folds the crossbar
    // away. Read latency is 0.
    if (m.scattered) {
      SmallVector<Value> elems;
      for (const uarch::MemUnit::ElemPort &p : m.elemPorts)
        elems.push_back(pa.getInput(p.in));
      readData[accKey(r.id, r.idx)] =
          readCrossbar(c, elems, scatterIndex(m, acc));
      continue;
    }
    auto eb = externalBank(m, acc);
    if (eb.factor == 1 || eb.bank)
      continue; // only data-dependent banked reads
    auto bs = bankAddress(m, acc);
    SmallVector<Value> vals;
    Value addr = boundaryAddr(c, bs.offset);
    for (const auto &[bank, base] : extPorts(m, acc)) {
      pa.setOutput(portAddr(base), addr);
      vals.push_back(pa.getInput(portData(base)));
    }
    unsigned lat = m.readLatency;
    Value sel = lat ? c.shiftChain(bs.bank, lat, sh).last() : bs.bank;
    readData[accKey(r.id, r.idx)] = readCrossbar(c, vals, sel);
  }
}

// Drive the read-address port of each single-interface external read in region
// \p rb: the in-bank offset for a statically-banked argument (the boundary
// presents one interface per bank), the flat element index for an unbanked one.
// A data-dependent banked read spans every interface, and emitExternalReads
// drives all of its addresses.
void DatapathEmitter::emitExternalReadAddrs(const uarch::RegionBlock &rb) {
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (!m.external || acc.isWrite || m.scattered)
      continue; // a scattered argument has no address port to drive
    auto eb = externalBank(m, acc);
    if (eb.factor > 1 && !eb.bank)
      continue;
    pa.setOutput(portAddr(acc.portBase),
                 boundaryAddr(c, bankAddress(m, acc).offset));
  }
}

// Backedge every unit output before wiring, so an input may reference a unit
// emitted later: a fused recurrence reads its own output, and a data-dependent
// read address (emitInternalReads, which runs before emitUnits) reads a unit
// that computes it. A register elsewhere in the recurrence cycle keeps the
// hardware acyclic; the backedges only free emission from topological order.
void DatapathEmitter::declareUnits(const uarch::RegionBlock &rb) {
  for (uarch::UnitId uid : rb.units) {
    auto b = c.bb.get(hwType(dp.units[uid].identity.resultType, c.b));
    unitBE[uid] = b;
    unitVal[uid] = b;
  }
}

// Compute units of region \p rb: native -> comb; IP -> an instance of the
// extern operator module, internally pipelined by its latency.
void DatapathEmitter::emitUnits(const uarch::RegionBlock &rb, UnitMode mode) {
  // A leaf's backedges are declared earlier, before its reads resolve; a
  // container's units are the last thing it emits, so they declare their own.
  if (mode == UnitMode::Container)
    declareUnits(rb);
  StallShell sh = shellFor(rb.id);
  for (uarch::UnitId uid : rb.units) {
    const uarch::FuncUnit &u = dp.units[uid];
    if (mode != UnitMode::Leaf) {
      // Skipping the recurrence re-injection below is a no-op here rather than
      // a silent drop: a container has no per-iteration issue pulse to time one
      // against.
      assert(llvm::all_of(u.inputInits,
                          [](llvm::ArrayRef<uarch::Source> inits) {
                            return inits.empty();
                          }) &&
             "a container's own unit carries no recurrence init");
      // A guard predicate is a start-0 compute the children gate on, so the IP
      // path is unreachable for it; a while's condition cone may take cycles
      // (`t_cond`), which its mode says.
      assert((u.identity.comb || mode == UnitMode::Condition) &&
             "a container predicate must be a native (comb) unit");
    }
    SmallVector<Value> operands;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      Value v =
          resolveSource(u.inputs[k]); // a self-reference reads its own backedge
      // Re-inject a recurrence input's identities, one per iteration `iv`
      // spends below the recurrence distance, each gated by the issue pulse
      // delayed to this op's stage. Innermost first, so a later iteration's mux
      // sits nearer the port and the windows need no mutual exclusion. A shared
      // port carries none: its identities are arms of the input mux above.
      if (mode == UnitMode::Leaf)
        for (auto [n, init] : llvm::enumerate(u.inputInits[k])) {
          Value issue = controlOf.lookup(rb.id).issue;
          assert(issue && "recurrence input in a region with no controller");
          Value iterN = c.R(comb::AndOp::create(c.b, c.loc, issue,
                                                atIteration(rb, n), false));
          Value gate = c.activationPulse(iterN, u.repOp(), sh);
          v = c.mux(gate, resolveSource(init), v);
        }
      operands.push_back(v);
    }

    Value result;
    if (u.identity.comb) {
      result = emitCompute(c.b, c.loc, *u.identity.comb, operands,
                           hwType(u.identity.resultType, c.b), u.repOp());
    } else {
      // An IP instance takes its data operands, then clock, then (for a
      // clock-enabled contract) a `ce` bit that rides the region's
      // clock-enable, freezing with the shift chains under back-pressure.
      operands.push_back(c.clkRaw);
      if (u.stall == allo::StallContractEnum::Ce)
        operands.push_back(sh ? sh.chainEnable : c.t1);
      else
        // A free-running IP has no `ce`: under an elastic shell it would keep
        // advancing while the shell's shift chains stall, folding a stale
        // result. `validateDatapath` rejects that pairing up front.
        assert(!sh && "a free-running IP operator in a back-pressured region");
      result = hw::InstanceOp::create(c.b, c.loc, unitModule.lookup(u.id),
                                      unitInstanceName(u), operands)
                   ->getResult(0);
    }
    unitBE[uid].setValue(result);
    unitVal[u.id] = result;
    // Name the result wire after the frontend variable this op computes: the
    // dcp op carries the assignment-target NameLoc.
    nameValue(result, u.repOp()->getLoc());
  }
}

// The condition cone of a sequential (CHECK/RUN) while: the container's OWN
// condition memory reads plus its compute, returning the settled condition and
// its ready latency t_cond. There is no per-iteration issue pulse: the read
// address is the frozen iter-arg survivor, so the load is a continuous read of
// a stable element and its data is a stable wire from `checkStart + t_cond`
// onward, the survivors not advancing until after CHECK decides.
std::pair<Value, unsigned>
DatapathEmitter::emitConditionRegion(const uarch::RegionBlock &rb,
                                     const uarch::Source &condSrc) {
  // Same emission order as a leaf region's `emit`, but over this container's
  // OWN condition cone: `UnitMode::Condition`.
  emitRegisters(rb);
  declareUnits(rb);
  emitInternalReads(rb);
  emitExternalReads(rb);
  emitUnits(rb, UnitMode::Condition);
  resolveRegHeads(rb);
  // The condition's own external reads address by the survivor, so this runs
  // after emitUnits, exactly as in a leaf region's emitAccesses.
  emitExternalReadAddrs(rb);
  return {resolveSource(condSrc), readyCycle(condSrc)};
}

// Resolve region \p rb's register head inputs once its units exist.
void DatapathEmitter::resolveRegHeads(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs)
    regHeadBE.find(rid)->second.setValue(resolveSource(dp.regs[rid].input));
}

// The drain stage a store contributes to its region's `done`. The write is
// PRESENTED at `dcpStart` and COMMITS `writeLatency` cycles later; `emitDone`
// rides its own latch register for the last of those cycles (done reads 1 at
// `lastIssue + drainStage + 1`), so the stage is the commit cycle minus one.
static unsigned storeDrainOf(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  assert(m.writeLatency >= 1 &&
         "a zero-cycle write has no commit edge for the done latch to ride; "
         "checkDeviceCapability must have rejected the device row");
  return dcpStart(acc.op) + m.writeLatency - 1;
}

// The write twin of `emitSkewedInternalReads`: one write port per bank per
// lane. Bank k takes the address and data of whichever of the lane's accesses
// reaches it, and its write-enable is the OR of their demuxed enables, so an
// access commits on its own bank and nowhere else. The OR has at most one live
// arm for the same reason the address select does (`laneSelect`).
void DatapathEmitter::emitSkewedInternalWrites(const uarch::RegionBlock &rb,
                                               Value commit,
                                               DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    if (m.skewed && m.accesses[r.idx].isWrite)
      lanes[{r.id, m.accesses[r.idx].lane}].push_back(r.idx);
  }
  for (auto &[key, idxs] : lanes) {
    const uarch::MemUnit &m = dp.mems[key.first];
    ArrayRef<Value> banks = memBanks[m.id];
    // A `seq.hlmem` write port realizes exactly one cycle, so a deeper device
    // latency presents address/data/we `writeLatency - 1` cycles late (the
    // unskewed twin below says the rest).
    unsigned pre = m.writeLatency - 1;
    auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
    SmallVector<std::pair<Value, Value>> addrs, datas;
    SmallVector<Value> wes, bankOf;
    for (unsigned i : idxs) {
      const uarch::MemUnit::Access &acc = m.accesses[i];
      BankSplit bs = bankAddress(m, acc);
      Value bank = late(bs.bank);
      bankOf.push_back(bank);
      addrs.emplace_back(bank, late(memAddr(m, bs.offset)));
      datas.emplace_back(bank, late(resolveSource(acc.data)));
      wes.push_back(
          c.delayValid(c.activationPulse(commit, acc.op, sh), pre, sh));
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
    }
    auto wlat = c.b.getI64IntegerAttr(1);
    for (unsigned k = 0; k < banks.size(); ++k) {
      Value we = writeDemux(c, wes[0], bankOf[0], k);
      for (unsigned i = 1; i < idxs.size(); ++i)
        we = c.orBits(we, writeDemux(c, wes[i], bankOf[i], k));
      seq::WritePortOp::create(c.b, c.loc, banks[k],
                               ValueRange{laneSelect(c, addrs, k)},
                               laneSelect(c, datas, k), we, wlat);
    }
  }
}

// Read/write address + data outputs of the accesses scheduled in region \p rb,
// driven by that region's controller (counter / \p issue). Folds into \p fb the
// region's `storeDrain`: the stage its deepest store commits at (see
// `storeDrainOf`), which the region's `done` waits on.
void DatapathEmitter::emitAccesses(const uarch::RegionBlock &rb, Value issue,
                                   DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
  emitExternalReadAddrs(rb);
  // A store's write-enable is the issue pulse delayed to its stage. A leaf
  // while's doomed exit iteration still issues, so its store is additionally
  // gated by the continue-condition.
  Value gatedIssue;
  auto commitPulse = [&]() -> Value {
    if (!rb.conditional)
      return issue;
    if (!gatedIssue) {
      assert(rb.condition &&
             "a conditional (while) region has no continue condition; it is "
             "required to gate in-loop store commits");
      gatedIssue = c.andBits(issue, resolveSource(rb.condition));
    }
    return gatedIssue;
  };
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (!m.external || !acc.isWrite)
      continue;
    Value we = c.activationPulse(commitPulse(), acc.op, sh);
    Value data = resolveSource(acc.data);
    // A scattered argument's element ports are shared by every store: this only
    // records the terms, and `finalizeScatteredPorts` drives each element once
    // every region has contributed. Write latency is 1, no skew.
    if (m.scattered) {
      scatterWrites[m.id].push_back({we, scatterIndex(m, acc), data});
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
      continue;
    }
    auto eb = externalBank(m, acc);
    // A data-dependent write drives every bank interface; its runtime bank
    // gates each interface's write-enable so only the target bank commits (an
    // N-way demux). A static / unbanked write is a single interface.
    auto bs = bankAddress(m, acc);
    Value dynBank = eb.factor > 1 && !eb.bank ? bs.bank : Value();
    Value portAddrVal = boundaryAddr(c, bs.offset);
    // A merged group is one interface for several stores, so it is driven once
    // all of them have emitted. Merging happens only where every store reaches
    // a single interface, hence one `extPorts` pair and no demux.
    if (m.writesIndependent) {
      boundaryWrites[acc.portIdx].push_back({portAddrVal, data, we});
      fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
      continue;
    }
    for (const auto &[bank, base] : extPorts(m, acc)) {
      pa.setOutput(portAddr(base), portAddrVal);
      pa.setOutput(portData(base), data);
      pa.setOutput(portWe(base), writeDemux(c, we, dynBank, bank));
    }
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
  }
  emitSkewedInternalWrites(rb, commitPulse(), fb);
  // Internal-memory writes drive seq.write instead of module ports, but still
  // set the region's store drain.
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (m.external || !acc.isWrite || m.skewed)
      continue;
    ArrayRef<Value> banks = memBanks[m.id];
    // A `seq.hlmem` write port realizes exactly one cycle, so a deeper device
    // latency presents address/data/we `writeLatency - 1` cycles late. The
    // datum still lands at `dcpStart + writeLatency` (see `storeDrainOf`).
    unsigned pre = m.writeLatency - 1;
    auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
    Value we =
        c.delayValid(c.activationPulse(commitPulse(), acc.op, sh), pre, sh);
    Value data = late(resolveSource(acc.data));
    auto wlat = c.b.getI64IntegerAttr(1);
    auto ports = writePortOf.find(m.id);
    if (ports != writePortOf.end()) {
      sharedWrites[m.id].push_back(
          {*acc.staticBank, ports->second[r.idx],
           late(memAddr(m, bankAddress(m, acc).offset)), data, we});
    } else if (acc.staticBank) {
      // A compile-time bank writes its own memory: no demux, and no write port
      // on the other banks. An unbanked memref is the same case at bank 0.
      seq::WritePortOp::create(
          c.b, c.loc, banks[*acc.staticBank],
          ValueRange{late(memAddr(m, bankAddress(m, acc).offset))}, data, we,
          wlat);
    } else {
      // Drive every bank; the runtime bank gates the write-enable so only the
      // selected bank commits.
      auto bs = bankAddress(m, acc);
      Value addr = late(memAddr(m, bs.offset));
      Value bank = late(bs.bank);
      for (unsigned k = 0; k < banks.size(); ++k)
        seq::WritePortOp::create(c.b, c.loc, banks[k], ValueRange{addr}, data,
                                 writeDemux(c, we, bank, k), wlat);
    }
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
  }
}

// Drive each merged boundary write port group from the stores coloured onto
// it, a one-hot select for the same reason as the shared internal ports below.
void DatapathEmitter::finalizeBoundaryWritePorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.external || !m.writesIndependent)
      continue;
    for (const uarch::MemUnit::Access &acc : m.accesses) {
      auto it = boundaryWrites.find(acc.portIdx);
      if (!acc.isWrite || it == boundaryWrites.end())
        continue;
      Value addr, data, we;
      for (const BoundaryWrite &w : it->second) {
        addr = addr ? c.mux(w.we, w.addr, addr) : w.addr;
        data = data ? c.mux(w.we, w.data, data) : w.data;
        we = we ? c.orBits(we, w.we) : w.we;
      }
      pa.setOutput(portAddr(acc.portBase), addr);
      pa.setOutput(portData(acc.portBase), data);
      pa.setOutput(portWe(acc.portBase), we);
      boundaryWrites.erase(it); // the group's other stores are done with it
    }
  }
}

// Drive an array's shared write ports from the stores coloured onto each. Two
// stores on ONE port are provably never enabled in the same cycle, so the
// priority chain below is a one-hot select and its first arm is a don't-care.
void DatapathEmitter::finalizeSharedWritePorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    auto it = sharedWrites.find(m.id);
    if (it == sharedWrites.end())
      continue;
    ArrayRef<SharedWrite> writes = it->second;
    unsigned ports = 0;
    for (const SharedWrite &w : writes)
      ports = std::max(ports, w.port + 1);
    for (auto [k, hlmem] : llvm::enumerate(memBanks[m.id]))
      for (unsigned p = 0; p < ports; ++p) {
        Value addr, data, we;
        for (const SharedWrite &w : writes) {
          if (w.bank != k || w.port != p)
            continue;
          addr = addr ? c.mux(w.we, w.addr, addr) : w.addr;
          data = data ? c.mux(w.we, w.data, data) : w.data;
          we = we ? c.orBits(we, w.we) : w.we;
        }
        if (we)
          seq::WritePortOp::create(c.b, c.loc, hlmem, ValueRange{addr}, data,
                                   we, c.b.getI64IntegerAttr(1));
      }
  }
}

// Drive each scattered argument's element outputs from every store recorded
// against it: per element the datum is a priority mux over the stores that
// reach it, and the write-enable the OR of their demuxed pulses.
//
// At most one arm is live per element per cycle, so the priority order carries
// no meaning. Unlike a skewed lane that is not structural here: two stores to
// one element are ordered by the dependence analysis, while two stores to
// DIFFERENT elements in one cycle are what a complete partition's unlimited
// ports are for. A constant subscript folds its `icmpEq` away, so `A[3] = x`
// leaves element 3 driven and the other N-1 write-enables constant false.
void DatapathEmitter::finalizeScatteredPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    auto it = scatterWrites.find(m.id);
    if (!m.scattered || it == scatterWrites.end())
      continue; // not scattered, or scattered and never written
    ArrayRef<ScatterWrite> writes = it->second;
    for (auto [k, p] : llvm::enumerate(m.elemPorts)) {
      // Selected by the PULSE, not the index: two stores in different regions
      // can name element k at once (an idle region's stale address register),
      // so only the enabled one may drive; the first arm is a don't-care.
      Value data, we;
      for (const ScatterWrite &w : writes) {
        Value hits = writeDemux(c, w.we, w.index, k);
        data = data ? c.mux(hits, w.data, data) : w.data;
        we = we ? c.orBits(we, hits) : hits;
      }
      pa.setOutput(p.out, data);
      pa.setOutput(p.we, we);
    }
  }
}

// A kernel-local channel's `seq.fifo` cannot be built until every access has
// contributed its drive, and the accesses read its outputs. Declare those
// outputs as backedges here, before any region emits, and let
// `finalizeStreamPorts` build the FIFO and resolve them.
void DatapathEmitter::declareInternalChannels() {
  for (const uarch::StreamChannel &s : dp.streams) {
    // A channel wired between CHILD PORTS declares the other shape: one
    // `{data, valid}` pair per consumer end plus the producer's `ready`. Its
    // internal flag says which END is a module port, not whether wires exist.
    if (!s.callEnds.empty()) {
      ComposedWires &w = composedWires[s.id];
      for (const uarch::StreamChannel::CallEnd &e : s.callEnds)
        if (dp.calls[e.call].streamArgs[e.arg].isInput) {
          w.sinkData.push_back(c.bb.get(hwType(s.payload, c.b)));
          w.sinkValid.push_back(c.bb.get(c.i1));
        } else
          w.prodReady = c.bb.get(c.i1);
      continue;
    }
    if (!s.internal)
      continue;
    streamWires[s.id] = {c.bb.get(hwType(s.payload, c.b)), c.bb.get(c.i1),
                         c.bb.get(c.i1)};
  }
}

Value DatapathEmitter::streamData(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].data)
                    : pa.getInput(portData(streamPortBase(dp, s)));
}

Value DatapathEmitter::streamValid(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].valid)
                    : pa.getInput(portValid(streamPortBase(dp, s)));
}

Value DatapathEmitter::streamReady(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].ready)
                    : pa.getInput(portReady(streamPortBase(dp, s)));
}

void DatapathEmitter::bindStreamReads(const uarch::RegionBlock &rb) {
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    if (s.accesses[r.idx].isPut)
      continue;
    streamReadData[accKey(s.id, r.idx)] = streamData(s);
  }
}

// H for region \p rb: its stream handshakes, and the shell they derive. A put
// contributes to `_data`/`_valid`; a get to `_ready`. Both a full output and an
// empty input freeze the pipeline (`chainEnable`), so the phase counter, the
// shift chains and every Ce operator hold together and no bubble slips stale
// data into loop-carried state. A stage-0 access keys on the UNgated
// `wantIssue` so the signals stay combinationally acyclic, a deeper access on
// the registered delayed issue. A predicated access (`acc.when` set) also gates
// its handshake on the predicate, itself a datapath value rather than a FIFO
// status, so acyclicity is preserved.
//
// The pulses built here are timed against the region's PROMISED shell, which is
// what the returned enables resolve: the enable and the chains it freezes are
// mutually recursive, acyclic in hardware because the FIFO status it starts
// from is stored state, and the promise's backedges break that cycle for SSA
// construction. Several accesses may share one channel, interleaved inside the
// II by the FIFO dependence edges; each contributes its own term to
// `streamDrives[s.id]` and `finalizeStreamPorts` drives the port once.
StallShell DatapathEmitter::deriveStallShell(const uarch::RegionBlock &rb,
                                             Value issue,
                                             DatapathFeedback &fb) {
  // No stream accesses: nothing to be elastic about, so the region stays rigid.
  if (rb.streamAccesses.empty())
    return {};
  streamDrives.resize(dp.streams.size());
  StallShell sh = shellFor(rb.id); // the promise F and G were emitted against
  assert(sh && "a stream region must have its shell promise registered");

  Value atIssue =
      controlOf.lookup(rb.id).wantIssue; // ungated stage-0 activation
  assert(atIssue &&
         "a stream region's controller published no `wantIssue`: the shell "
         "defers a starved or back-pressured pass by GATING its issue, so a "
         "controller whose issue cannot be gated would drop the pass and "
         "sample `_data` with no regard for `_valid`");
  // Outputs: drive data + valid, accumulate the output-full hazard.
  Value outHazard; // OR over the region's puts of (valid & ~ready)
  // A stage>=1 put whose handshake fired while the pipeline is frozen: see the
  // `sent` latch below. Resolved once `chainEnable` is final.
  struct Sent {
    circt::Backedge in;
    Value flag, valid, ready;
  };
  SmallVector<Sent> sent;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (!acc.isPut)
      continue;
    // A predicated put produces a token only where its predicate holds: gate
    // `valid`, and suppress the output-full hazard when it is low, so the
    // pipeline never freezes waiting for space it will not write.
    Value pred = acc.when ? resolveSource(acc.when) : Value();
    Value valid = c.activationPulse(issue, acc.op, sh);
    if (pred)
      valid = c.andBits(valid, pred);
    // An input-side freeze holds a stage>=1 put's chain pulse high after the
    // handshake fired, so a ready consumer would recapture the token. The
    // `sent` latch retires it. A stage-0 pulse is `issue`, already gated.
    if (acc.stage >= 1) {
      circt::Backedge in = c.bb.get(c.i1);
      Value flag = c.reg(in, c.f1);
      valid = c.andBits(valid, c.notBit(flag));
      sent.push_back({in, flag, valid, streamReady(s)});
    }
    auto &drv = streamDrives[s.id];
    drv.data.push_back({valid, resolveSource(acc.data)});
    drv.valid = drv.valid ? c.orBits(drv.valid, valid) : valid;
    // A stage-0 put keys its hazard on wantIssue (ungated) & pred; a stage>=1
    // put's valid is already registered (delayed) and predicate-gated.
    Value active = acc.stage == 0 ? atIssue : valid;
    if (pred && acc.stage == 0)
      active = c.andBits(active, pred);
    Value hz = c.andBits(active, c.notBit(streamReady(s)));
    outHazard = outHazard ? c.orBits(outHazard, hz) : hz;
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, acc.stage);
  }
  // Mid-pipeline freeze: a stage>0 get with a needed-but-empty input cannot
  // bubble past a missing token, so fold that stall into `chainEnable` beside
  // the output-full freeze. Only registered state is read here.
  Value midStall;
  unsigned stage0Gets = 0;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut)
      continue;
    if (acc.stage == 0) {
      ++stage0Gets;
      continue;
    }
    Value active = c.delayValid(issue, acc.stage, sh);
    Value want = acc.when ? c.andBits(active, resolveSource(acc.when)) : active;
    Value miss = c.andBits(want, c.notBit(streamValid(s)));
    midStall = midStall ? c.orBits(midStall, miss) : miss;
  }
  Value chainEnable = outHazard ? c.notBit(outHazard) : c.t1;
  if (midStall)
    chainEnable = c.andBits(chainEnable, c.notBit(midStall));

  // Stage-0 inputs (read at issue) fold into `stage0Valid`, the issue gate; a
  // predicated get treats a non-needed input as available (`valid | ~pred`).
  // With >1 stage-0 get they must pop together, so their readies gate on it.
  Value stage0Valid;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut || acc.stage != 0)
      continue;
    Value valid = streamValid(s);
    if (acc.when)
      valid = c.orBits(valid, c.notBit(resolveSource(acc.when)));
    stage0Valid = stage0Valid ? c.andBits(stage0Valid, valid) : valid;
  }
  bool join0 = stage0Gets > 1;

  // A starved stage-0 slot freezes the whole shell rather than bubbling: a
  // bubble would desync the II>1 phase/chain alignment and clock stale data
  // into a recurrence. An acyclic region freezes on the same term.
  if (stage0Valid)
    chainEnable = c.andBits(
        chainEnable, c.notBit(c.andBits(atIssue, c.notBit(stage0Valid))));

  // `chainEnable` is final: retire each stage>=1 put's token until the chain
  // advances past it (`flag' = ~chainEnable & (flag | fired)`). Only the
  // register's input closes here, so nothing reads a half-built value.
  for (Sent &st : sent)
    st.in.setValue(c.andBits(c.notBit(chainEnable),
                             c.orBits(st.flag, c.andBits(st.valid, st.ready))));

  // Drive each `_ready`: a stage-0 get accepts when issuing and not frozen
  // (a join also waits for all stage-0 inputs); a deeper get accepts when
  // the chain advances; a predicated get pops only where its predicate holds.
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut)
      continue;
    Value pred = acc.when ? resolveSource(acc.when) : Value();
    Value active =
        acc.stage == 0 ? atIssue : c.delayValid(issue, acc.stage, sh);
    Value ready = c.andBits(active, chainEnable);
    if (acc.stage == 0 && join0)
      ready = c.andBits(ready, stage0Valid);
    if (pred)
      ready = c.andBits(ready, pred);
    auto &drv = streamDrives[s.id];
    drv.ready = drv.ready ? c.orBits(drv.ready, ready) : ready;
  }
  nameValue(chainEnable, regionSignal(rb.id, "ce"));
  // The two halves coincide: input starvation is already folded into the freeze
  // above, so the cycle the chain may advance is the cycle a pass may issue.
  return {chainEnable, chainEnable};
}

// Drive each channel from the terms every region contributed. A BOUNDARY
// channel drives its module ports, the port set following its direction: an
// input FIFO's `_data` / `_valid` are module inputs and only `_ready` is
// driven, an output's the reverse (`validateDatapath` rejects a boundary
// channel used both ways, so the two cases are exhaustive). A kernel-LOCAL
// channel instead OWNS its queue: one `seq.fifo` here.
void DatapathEmitter::finalizeStreamPorts() {
  streamDrives.resize(dp.streams.size());
  for (const uarch::StreamChannel &s : dp.streams) {
    // A channel wired between CHILD PORTS has no access of this module's own:
    // its handshake closes over the instances instead.
    if (!s.callEnds.empty()) {
      emitComposedChannel(s);
      continue;
    }
    const StreamDrive &drv = streamDrives[s.id];
    // The puts' pulses are mutually exclusive (one access per cycle), so the
    // data mux is a plain priority chain over them; the last arm is the
    // fall-through, read only when `valid` is low and thus a don't-care.
    auto putData = [&] {
      assert(!drv.data.empty() && "a written channel with no put");
      Value data = drv.data.back().second;
      for (unsigned k = drv.data.size() - 1; k-- > 0;)
        data = c.mux(drv.data[k].first, drv.data[k].second, data);
      return data;
    };
    if (s.internal) {
      emitInternalChannel(s, putData());
      continue;
    }
    auto base = streamPortBase(dp, s);
    if (s.isInput) {
      pa.setOutput(portReady(base), drv.ready ? drv.ready : c.f1);
      continue;
    }
    pa.setOutput(portData(base), putData());
    pa.setOutput(portValid(base), drv.valid);
  }
}

// The queue behind a kernel-local channel. Both ends are this module's, so the
// handshake closes here: a token is pushed where a put fires and the FIFO has
// space, popped where a get fires and it holds one. `seq.fifo`'s output is
// show-ahead, so {output, ~empty, ~full} present exactly the {data, valid,
// ready} triple the accesses were written against for a boundary port.
void DatapathEmitter::emitInternalChannel(const uarch::StreamChannel &s,
                                          Value data) {
  const StreamDrive &drv = streamDrives[s.id];
  StreamWires &w = streamWires[s.id];
  assert(drv.valid && drv.ready &&
         "a local channel is validated to have both ends");
  auto fifo = seq::FIFOOp::create(
      c.b, c.loc, hwType(s.payload, c.b), c.i1, c.i1, Type(), Type(), data,
      /*rdEn=*/c.andBits(drv.ready, w.valid),
      /*wrEn=*/c.andBits(drv.valid, w.ready), c.clk, c.rst,
      c.b.getI64IntegerAttr(declaredDepth(s.depth)), c.b.getI64IntegerAttr(0),
      IntegerAttr(), IntegerAttr());
  w.data.setValue(fifo.getOutput());
  w.valid.setValue(c.notBit(fifo.getEmpty()));
  w.ready.setValue(c.notBit(fifo.getFull()));
}

// The queue(s) behind a channel wired between CHILD PORTS: one `seq.fifo` per
// CONSUMER end, all pushed by the producer on the same cycle, the fan-out tee.
// The producer may write only when every consumer can accept (the bounded
// fork), so each copy sees the whole token sequence in order. A SEEDED channel
// additionally fronts each consumer with an init-prepend shim: while its `rem`
// down-counter is non-zero the consumer reads the initial tokens and does not
// pop, so the history it sees is [init] ++ [produced] and a feedback cycle
// turns from cycle 0.
//
// Where one end is a BOUNDARY port of this module rather than a child, that end
// needs no queue: the child's own handshake is the module's, so the three wires
// pass straight through. A fanned-out boundary input is the one mixed case: the
// module's port pushes the tee.
void DatapathEmitter::emitComposedChannel(const uarch::StreamChannel &s) {
  ComposedWires &w = composedWires[s.id];
  // A channel is composed OR accessed, never both: a stream operand makes its
  // call concurrent, and a concurrent region issues no access of its own.
  assert(s.accesses.empty() &&
         "a channel wired between child ports also has in-module accesses");
  Type payload = hwType(s.payload, c.b);
  std::string base = s.internal ? std::string() : streamPortBase(dp, s);

  // The push side: the producing child, or this module's own stream port for a
  // boundary INPUT argument.
  Value pData, pValid;
  SmallVector<const uarch::StreamChannel::CallEnd *> sinks;
  for (const uarch::StreamChannel::CallEnd &e : s.callEnds) {
    const uarch::CallUnit::StreamArg &sa = dp.calls[e.call].streamArgs[e.arg];
    if (sa.isInput) {
      sinks.push_back(&e);
      continue;
    }
    pData = callOuts[e.call][sa.data];
    pValid = callOuts[e.call][sa.valid];
  }
  // A boundary OUTPUT: the module's port is the consumer, so the producing
  // child's handshake IS the module's.
  if (sinks.empty()) {
    assert(!s.internal && pData && "a channel with no reader");
    pa.setOutput(portData(base), pData);
    pa.setOutput(portValid(base), pValid);
    w.prodReady.setValue(pa.getInput(portReady(base)));
    return;
  }
  auto consumerReady = [&](const uarch::StreamChannel::CallEnd &e) {
    return callOuts[e.call][dp.calls[e.call].streamArgs[e.arg].ready];
  };
  if (!pData) { // a boundary INPUT feeds the readers
    pData = pa.getInput(portData(base));
    pValid = pa.getInput(portValid(base));
    // A single reader takes it straight, queue-free.
    if (sinks.size() == 1) {
      w.sinkData[0].setValue(pData);
      w.sinkValid[0].setValue(pValid);
      pa.setOutput(portReady(base), consumerReady(*sinks.front()));
      return;
    }
  }

  unsigned depth = declaredDepth(s.depth);
  auto init = dyn_cast_or_null<ArrayAttr>(s.init);
  unsigned nInit = init ? init.size() : 0;
  // The status wires close a cycle: a consumer's `rdEn` reads its own FIFO's
  // `empty` through the shim, and the producer's `wrEn` every FIFO's `full`.
  // So the whole tee is built against promises and resolved at the end.
  SmallVector<Backedge> full, empty, out;
  Value allNotFull;
  for (unsigned k = 0; k < sinks.size(); ++k) {
    full.push_back(c.bb.get(c.i1));
    empty.push_back(c.bb.get(c.i1));
    out.push_back(c.bb.get(payload));
    Value nf = c.notBit(full[k]);
    allNotFull = allNotFull ? c.andBits(allNotFull, nf) : nf;
  }
  Value wrEn = c.andBits(pValid, allNotFull);
  if (!s.internal)
    pa.setOutput(portReady(base), allNotFull); // a fanned-out boundary input

  for (auto [k, e] : llvm::enumerate(sinks)) {
    Value notEmpty = c.notBit(empty[k]);
    Value cReady = consumerReady(*e);
    Value rdEn = c.andBits(cReady, notEmpty);
    Value data = out[k], valid = notEmpty;
    if (nInit) {
      // `rem` counts the initial tokens still to serve, k .. 1; the datum is
      // picked by the running index (idx = nInit - rem) and the rem==1 token
      // falls through as the chain's default.
      unsigned remW = 1;
      while ((1u << remW) <= nInit)
        ++remW;
      Type remTy = c.b.getIntegerType(remW);
      Backedge remNext = c.bb.get(remTy);
      Value rem = c.reg(remNext, c.konst(remTy, nInit));
      nameValue(rem,
                channelSignal(ownerOf(s.stream, chanOwner(s.id)),
                              sinks.size() > 1 ? "init_rem" + std::to_string(k)
                                               : std::string("init_rem")));
      Value serving = c.R(comb::ICmpOp::create(
          c.b, c.loc, comb::ICmpPredicate::ne, rem, c.konst(remTy, 0)));
      auto token = [&](unsigned idx) {
        Attribute a = init[idx];
        APInt bits = isa<IntegerAttr>(a)
                         ? cast<IntegerAttr>(a).getValue()
                         : cast<FloatAttr>(a).getValue().bitcastToAPInt();
        return c.konst(payload,
                       bits.zextOrTrunc(cast<IntegerType>(payload).getWidth())
                           .getZExtValue());
      };
      Value fromInit = token(nInit - 1);
      for (unsigned v = 2; v <= nInit; ++v)
        fromInit = c.mux(c.icmpEqV(rem, c.konst(remTy, v)), token(nInit - v),
                         fromInit);
      data = c.mux(serving, fromInit, out[k]);
      valid = c.orBits(serving, notEmpty);
      rdEn = c.andBits(rdEn, c.notBit(serving));
      Value dec = c.R(comb::SubOp::create(c.b, c.loc, rem, c.konst(remTy, 1)));
      remNext.setValue(c.mux(c.andBits(serving, cReady), dec, rem));
    }
    auto fifo = seq::FIFOOp::create(
        c.b, c.loc, payload, c.i1, c.i1, Type(), Type(), pData, rdEn, wrEn,
        c.clk, c.rst, c.b.getI64IntegerAttr(depth), c.b.getI64IntegerAttr(0),
        IntegerAttr(), IntegerAttr());
    // The consumer's promises resolve FIRST: for an unseeded end they resolve
    // *to* the status promises below, which would be erased out from under them
    // the other way round.
    w.sinkData[k].setValue(data);
    w.sinkValid[k].setValue(valid);
    out[k].setValue(fifo.getOutput());
    full[k].setValue(fifo.getFull());
    empty[k].setValue(fifo.getEmpty());
  }
  if (w.prodReady)
    w.prodReady.setValue(allNotFull);
}

// The start pulse of one child, read off the node's contract and its region's
// composition class:
//
//   * HANDSHAKE, the rising edge of the predecessors' joined `done`, for a
//     gated child of a SCHEDULED composition and, in a CONCURRENT one, for a
//     child whose ordering cannot be expressed as an offset: a spawn, a
//     consumer of a scalar result (that port only holds from the producer's
//     `done`), or a child gated by an indeterminate producer. A
//     CHANNEL-CONNECTED pair never reaches here, back-pressure already being
//     their ordering;
//   * BROADCAST, the container's own start, for an ungated spawn;
//   * TIME-TRIGGERED at the scheduled offset otherwise. An ungated call's
//     operands need not be ready at the region's issue pulse (a scalar argument
//     loaded from memory is the reachable case), so releasing it at issue would
//     latch garbage. The offset rides the region's shell, so it stretches with
//     a stall.
//
// A child's `done` is a level its own start clears, so on a retriggered region
// it still reads the previous pass's 1 until the child is released. The
// predecessor join and the region's completion conjunction mean "completed THIS
// pass" and therefore read it through `completedSince(issue)`, in a SCHEDULED
// composition only: there `issue` is the pass-start pulse the calls are placed
// against, where a CONCURRENT region has no such boundary.
Value DatapathEmitter::startForCall(const uarch::CallUnit &cu, Value issue,
                                    ArrayRef<Value> predDones, bool concurrent,
                                    const StallShell &sh) {
  if (!concurrent)
    return predDones.empty() ? c.delayValid(issue, cu.start, sh)
                             : c.startFor(issue, predDones);
  bool handshake =
      !predDones.empty() &&
      (cu.async ||
       llvm::any_of(cu.predecessors, [&](const uarch::CallUnit::Pred &p) {
         return p.viaResult || !dp.calls[p.call].determinate;
       }));
  if (handshake)
    return c.startFor(/*regionStart=*/Value(), predDones);
  if (cu.async)
    return issue;
  return c.delayValid(issue, cu.start, StallShell{});
}

// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
// hw.instance. The child masters each memref operand's memory: it drives the
// addr/data/we, so the leaf wires those instance-output ports to the buffer's
// hlmem. The region's completion is the child's real `done` (fb.callDone).
// Serial execution (a producer region drains before the child starts, the child
// before a consumer) means one master per port at a time: no arbitration mux.
void DatapathEmitter::emitCalls(const uarch::RegionBlock &rb, Value issue,
                                DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
  // Each call starts by the policy above, off the `done`s of the composition
  // predecessors the model derived (`recordCallDeps`); the region completes
  // when every call's done is set.
  bool concurrent = rb.determinacy == DeterminacyEnum::Concurrent;
  SmallVector<Value> dones; // each call's done, by index
  llvm::DenseMap<uarch::CallId, Value>
      doneByCid; // done by id (scalar hand-off)
  for (uarch::CallId cid : rb.callUnits) {
    const uarch::CallUnit &cu = dp.calls[cid];
    SmallVector<Value> predDones;
    for (const uarch::CallUnit::Pred &p : cu.predecessors) {
      Value d = doneByCid.lookup(p.call);
      assert(d && "a call predecessor must be instantiated before its "
                  "consumer (they are in program order)");
      predDones.push_back(d);
    }
    Value startK = startForCall(cu, issue, predDones, concurrent, sh);
    assert(callees && "a CallUnit needs callee context");
    auto mit = callees->modules.find(cu.callee);
    assert(mit != callees->modules.end() &&
           "the callee module must be registered (emitted bottom-up first)");
    hw::HWModuleOp child = mit->second;

    // Instance inputs by child port name: clk/rst/`start` plus each read's data
    // input. An internal read consumes a backedge resolved after the instance;
    // a boundary read passes the top's data input straight through.
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
    // Channel ends: the child drives two of the three handshake wires and reads
    // the third. What it reads is a promise the channel realization resolves
    // once every end exists.
    for (auto [k, sa] : llvm::enumerate(cu.streamArgs)) {
      ComposedWires &w = composedWires[sa.chan];
      if (!sa.isInput) {
        ins[sa.ready] = w.prodReady;
        continue;
      }
      unsigned slot = 0;
      for (const uarch::StreamChannel::CallEnd &e :
           dp.streams[sa.chan].callEnds)
        if (dp.calls[e.call].streamArgs[e.arg].isInput) {
          if (e.call == cu.id && e.arg == k)
            break;
          ++slot;
        }
      ins[sa.data] = w.sinkData[slot];
      ins[sa.valid] = w.sinkValid[slot];
    }
    // Scalar operands: drive each child scalar-input port from its resolved
    // Source, sampled at the child's start.
    for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
      ins[sa.port] = resize(c.b, c.loc, resolveSource(sa.src), sa.width,
                            /*isSigned=*/true);

    auto outs = instantiateChild(c.b, c.loc, child,
                                 childInstanceName(cu.callee, cu.id), ins);

    // Scalar results: the child holds each result on its output port from
    // `done` onward, so that port is the survivor a sibling reads, with no
    // separate capture (`captureResults` skips a Call result). A survivor is
    // keyed by the region result it is yielded as, which is the call's own
    // index only where the call is the whole of what the region yields.
    for (auto [r, port] : llvm::enumerate(cu.resultPorts)) {
      callResultVal[accKey(cu.id, r)] = outs[port];
      for (auto [k, res] : llvm::enumerate(dp.regions[cu.region].results))
        if (res.value.kind == uarch::Source::Kind::Call &&
            res.value.id == cu.id && res.value.outPort == r)
          setSurvivor(cu.region, k, outs[port]);
    }

    // Master each buffer from the child's addr/data/we outputs: a boundary arg
    // passes through to the top port (flat i32 address); an internal buffer
    // drives its hlmem at the clog2(depth) index and the child's RAM latency.
    for (auto [argIdx, ma] : llvm::enumerate(cu.memArgs)) {
      if (ma.isBoundary) {
        // One port group per accessor, driven from the child's addr/data/we:
        // concurrent masters get distinct groups (no mux); a serial pair also
        // uses two groups, each active only in its own phase.
        pa.setOutput(portAddr(ma.topBase), outs[ma.addr]);
        if (ma.isWrite) {
          pa.setOutput(portData(ma.topBase), outs[ma.data]);
          pa.setOutput(portWe(ma.topBase), outs[ma.we]);
        }
        continue;
      }
      const uarch::MemUnit &m = dp.mems[ma.mem];
      // A constant table the child only reads: one `hw.array_get` registered to
      // the latency the child was timed against, so the datum lands exactly
      // where a RAM's would.
      if (m.isRom) {
        Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id],
                                                memAddr(m, outs[ma.addr])));
        rdBackedge[ma.data].setValue(
            c.shiftChain(elem, m.readLatency, sh).last());
        continue;
      }
      // One hlmem per bank: the child masters bank `ma.bank`, already indexed
      // in that bank's own space via `allo.part`, so this routes straight to
      // it with no crossbar (validateDatapath rejects a partition mismatch).
      assert(ma.bank < memBanks[m.id].size() &&
             "child bank index exceeds the buffer's bank count; "
             "validateDatapath must have rejected the partition mismatch");
      Value hlmem = memBanks[m.id][ma.bank];
      Value addr = memAddr(m, outs[ma.addr]);
      // The child was compiled against this buffer's device latency, read here
      // from the MemUnit since the parent never accesses the buffer itself. A
      // deeper write pipelines into the fixed 1-cycle port, as emitAccesses.
      if (ma.isWrite) {
        unsigned pre = m.writeLatency - 1;
        Value a = c.shiftChain(addr, pre, sh).last();
        Value d = c.shiftChain(outs[ma.data], pre, sh).last();
        Value w = c.delayValid(outs[ma.we], pre, sh);
        // The colouring settles a call's write port too, so two ports of ONE
        // child that declared them independent land in separate `always`
        // blocks and the array still infers a true dual port.
        auto ports = writePortOf.find(m.id);
        if (ports != writePortOf.end())
          sharedWrites[m.id].push_back(
              {ma.bank,
               ports->second[dp.callPortSlot(m.id, cu.id, unsigned(argIdx))], a,
               d, w});
        else
          seq::WritePortOp::create(c.b, c.loc, hlmem, ValueRange{a}, d, w,
                                   c.b.getI64IntegerAttr(1));
      } else
        rdBackedge[ma.data].setValue(
            c.R(seq::ReadPortOp::create(c.b, c.loc, hlmem, ValueRange{addr},
                                        /*rdEn=*/Value(), m.readLatency)));
    }
    // Scoped to this pass for the join above and the conjunction below, which
    // would otherwise read the previous pass's latched 1.
    Value completed =
        concurrent ? outs[kDone] : c.completedSince(outs[kDone], issue);
    doneByCid[cu.id] = completed;
    dones.push_back(completed);
    if (!cu.streamArgs.empty())
      callOuts[cu.id] = std::move(outs);
  }
  // The region completes when every call has: the AND of their dones.
  Value all;
  for (Value d : dones)
    all = all ? c.andBits(all, d) : d;
  if (all)
    fb.callDone = all;
}

// Emit region \p rb's whole datapath (F) given the controller's \p issue;
// returns its store feedback. Every timing primitive here runs on the region's
// registered shell; deriving that shell (H) is the orchestrator's next step, on
// what this emits.
DatapathFeedback DatapathEmitter::emit(const uarch::RegionBlock &rb,
                                       Value issue) {
  bindStreamReads(rb);
  emitRegisters(rb);
  declareUnits(rb); // unit backedges must exist before a read address resolves
  emitInternalReads(rb);
  emitExternalReads(rb);
  DatapathFeedback fb;
  // Calls precede units/reg-heads/accesses: a call's scalar result is an
  // ordinary Source a chained unit reads directly. The reverse edge (a call
  // operand computed by this region's unit) closes through the unit backedges.
  emitCalls(rb, issue, fb);
  emitUnits(rb);
  resolveRegHeads(rb);
  emitAccesses(rb, issue, fb);
  return fb;
}

} // namespace mlir::allo::uarch
