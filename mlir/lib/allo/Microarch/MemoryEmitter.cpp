/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The memory subsystem's EMIT half: how an access reaches its storage. Three
// dispatches on `PortPlan` -- `emitReads`, `emitWrites` and `masterCallPorts`
// -- plus the address hardware they share and the finalizers a port shared
// between regions needs. What the storage IS, and which ports each access
// holds, is decided in Memory.cpp.
//===----------------------------------------------------------------------===//

#include "allo/IR/AlloOps.h" // kIndependentWritesAttr
#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/AddressModel.h" // addressExprsOf

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

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
    SmallVector<Value> idx; // the access's own index sources, dims then symbols
    for (const uarch::Source &s : acc.addr)
      idx.push_back(resolveSource(s));
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

// Which element of a scattered memory \p acc names, at the DATAPATH width. The
// crossbar and the write demux compare it against literal element numbers
// (`icmpEq` builds those at that width).
Value DatapathEmitter::scatterIndex(const uarch::MemUnit &m,
                                    const uarch::MemUnit::Access &acc) {
  assert(m.scattered && "an element index belongs to a scattered memory");
  return addrAt(c.b, c.loc, bankAddress(m, acc).offset, kDatapathAddressWidth);
}

// The element registers of scattered internal array \p id, in element order.
// They are backedges until `finalizeScatteredPorts` resolves them, so a reader
// takes them without waiting for the stores that drive them.
SmallVector<Value> DatapathEmitter::scatterValues(unsigned id) {
  auto it = scatterElems.find(id);
  assert(it != scatterElems.end() && "no element registers for this array");
  return {it->second.begin(), it->second.end()};
}

// Bind the read-data input ports into readData, once, before the per-region
// loop (external memories only; internal ones read via seq.read below). Every
// access of a port group takes the same data input, since they never issue
// together and each finds its own datum there on its own cycle. A
// data-dependent banked read has one data port per bank and is bound by
// emitReads, which muxes them in-region.
void DatapathEmitter::bindReadPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.external)
      continue;
    for (auto [i, acc] : llvm::enumerate(m.accesses))
      // A write, or a plan whose datum is a select over several ports rather
      // than one port's: both are bound by `emitReads`.
      if (!acc.isWrite && acc.plan == PortPlan::Coloured)
        readData[accKey(m.id, i)] =
            pa.getInput(portData(extPorts(m, acc).front().second));
  }
}

// Instantiate on-chip storage for each internal (non-argument) memory: one
// seq.hlmem, or one per bank when the array reached emit still partitioned (a
// data-dependent bank `dcp-resolve-banking` could not split statically). The
// handles are module-scope so writes and reads in different regions share them.
void DatapathEmitter::createInternalMemories() {
  using R = uarch::MemUnit::Realization;
  for (const uarch::MemUnit &m : dp.mems) {
    R realization = m.realization();
    if (realization == R::Boundary)
      continue;
    IntegerType elemTy = memElemType(m, c.b);
    unsigned depth = declaredDepth(m.depthWords);
    if (realization == R::Rom) {
      // A constant table: one hw.aggregate_constant holding the global's
      // initializer, read combinationally by hw.array_get and registered to the
      // read latency in emitReads. No writable hlmem, no write ports.
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
    // A completely partitioned array is one register per element, not an
    // addressed memory: that is what buys the unlimited combinational ports the
    // scheduler was billed against. Only the backedges here; the registers
    // themselves need every store, so `finalizeScatteredPorts` builds them.
    // Exactly `depthWords` of them, not `declaredDepth`: the padding word only
    // exists to keep an hlmem's address at least one bit wide, and an element
    // is selected by comparison rather than addressed.
    if (realization == R::Scatter) {
      SmallVector<Backedge> elems;
      for (unsigned k = 0; k < m.depthWords; ++k)
        elems.push_back(c.bb.get(elemTy));
      scatterElems[m.id] = std::move(elems);
      continue;
    }
    // One cell per instance of each bank, bank-major. Reads past what one
    // instance of the row has are served by another copy of the whole array,
    // so the copies are what the module builds and what it was priced for.
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k)
      for (unsigned i = 0; i < m.instances; ++i) {
        auto mem = seq::HLMemOp::create(c.b, c.loc, c.clk, c.rst,
                                        memCellName(dp, m, k, i),
                                        {static_cast<int64_t>(depth)}, elemTy);
        // The port binding proved these writes never collide, which is exactly
        // the promise the lowering needs to describe each in its own `always`
        // block, and so to build a true dual port. Without it they share one
        // block, which arbitrates the collision they might have.
        if (m.writesIndependent)
          mem->setAttr(kIndependentWritesAttr, c.b.getUnitAttr());
        // Pin the array to the row it is REALIZED in: the row is this module's
        // decision, and leaving it unsaid hands the structure to the
        // synthesizer, which then builds something the cost model did not
        // price.
        if (!m.ramStyle.empty())
          mem->setAttr(kRamStyleAttr, c.b.getStringAttr(m.ramStyle));
        // An initialized array the kernel also WRITES is a real memory that
        // merely starts with contents. `seq.hlmem` carries no initializer, so
        // the words ride to the seq->SV pipeline, which gives the backing reg
        // an `initial` block. Every copy starts with them.
        if (m.romInit)
          recordMemoryInit(
              mem, initWords(cast<ElementsAttr>(m.romInit), m.width, depth));
        banks.push_back(mem.getHandle());
      }
    memBanks[m.id] = std::move(banks);
  }
}

Value DatapathEmitter::atReadData(const uarch::MemUnit &m, Value v,
                                  const StallShell &sh) {
  return c.shiftChain(v, m.readLatency, sh).last();
}

// One arm per `PortPlan`, and within an arm the only question left is whose
// cells these are, this module's or the caller's. Read latency is the memory's
// device-resolved `readLatency`, the number the scheduler timed the access at,
// so the datum lands on exactly the cycle the consumer's register depth was
// solved against.
void DatapathEmitter::emitReads(const uarch::RegionBlock &rb, Value issue) {
  StallShell sh = shellFor(rb.id);
  // The two plans that serve several accesses from one port, collected here and
  // built below, once the region's whole demand on the port is known.
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  llvm::MapVector<std::tuple<unsigned, unsigned, unsigned>,
                  SmallVector<unsigned>>
      shared;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (acc.isWrite)
      continue;
    switch (acc.plan) {
    case PortPlan::Table: {
      // A constant table read: index the aggregate_constant combinationally,
      // then register to the scheduled read latency so timing matches a RAM.
      Value idx = memAddr(m, bankAddress(m, acc).offset);
      readData[accKey(m.id, r.idx)] = atReadData(
          m, c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id], idx)), sh);
      break;
    }
    case PortPlan::ElementWise: {
      // No address port: a read selects over the cells, and a constant
      // subscript folds the select away. An argument's cells arrive on its
      // element input ports, an internal array's are this module's registers;
      // either way they are timed at read latency 0.
      SmallVector<Value> elems;
      if (m.external)
        for (const uarch::MemUnit::ElemPort &p : m.elemPorts)
          elems.push_back(pa.getInput(p.in));
      else
        elems = scatterValues(m.id);
      readData[accKey(m.id, r.idx)] =
          readCrossbar(c, elems, scatterIndex(m, acc));
      break;
    }
    case PortPlan::Lane:
      lanes[{r.id, acc.lane}].push_back(r.idx);
      break;
    case PortPlan::Coloured:
      // A compile-time bank reads its own memory: no crossbar, and no read port
      // on the other banks. An unbanked memref is the same case at bank 0. An
      // ARGUMENT's group is not built here: its datum is the port's, bound once
      // by `bindReadPorts`, and its address may be computed by a unit this
      // region has not emitted yet (`emitExternalReadAddrs`).
      if (!m.external)
        shared[{r.id, *acc.staticBank, acc.port}].push_back(r.idx);
      break;
    case PortPlan::Crossbar: {
      // Read every bank at the (bank-independent) offset, then select by the
      // runtime bank, aligned with the read data. Such an access reaches every
      // bank, so it holds a port of its own on each and shares none.
      auto bs = bankAddress(m, acc);
      SmallVector<Value> vals;
      if (m.external) {
        Value addr = boundaryAddr(c, bs.offset);
        for (const auto &[bank, base] : extPorts(m, acc)) {
          pa.setOutput(portAddr(base), addr);
          vals.push_back(pa.getInput(portData(base)));
        }
      } else {
        Value addr = memAddr(m, bs.offset);
        for (unsigned k = 0; k < m.numBanks; ++k)
          vals.push_back(
              c.R(atPort(seq::ReadPortOp::create(
                             c.b, c.loc, memReadCell(m, k, acc.port),
                             ValueRange{addr}, /*rdEn=*/Value(), m.readLatency),
                         acc.port)));
      }
      readData[accKey(m.id, r.idx)] =
          readCrossbar(c, vals, atReadData(m, bs.bank, sh));
      break;
    }
    }
  }
  // One read port per bank per LANE rather than per bank per access. A lane's
  // accesses hold distinct slots, so they reach distinct banks at every
  // rotation, and bank k can take the offset of whichever of them reaches it
  // and hand its datum back to that one: F accesses over F banks at one port
  // each, where a crossbar would take a port on every bank for every access.
  for (auto &[key, idxs] : lanes) {
    const uarch::MemUnit &m = dp.mems[key.first];
    SmallVector<std::pair<Value, Value>> tagged; // (runtime bank, in-bank addr)
    for (unsigned i : idxs) {
      BankSplit bs = bankAddress(m, m.accesses[i]);
      tagged.emplace_back(bs.bank, memAddr(m, bs.offset));
    }
    // Untagged for the same reason its stores are: a lane is assigned by the
    // skew rather than by the port graph, so it proves nothing about what else
    // touches this bank.
    SmallVector<Value> vals;
    for (unsigned k = 0; k < m.numBanks; ++k)
      vals.push_back(
          c.R(seq::ReadPortOp::create(c.b, c.loc, memReadCell(m, k, key.second),
                                      ValueRange{laneSelect(c, tagged, k)},
                                      /*rdEn=*/Value(), m.readLatency)));
    // Each access picks its own bank's datum back out, delayed with it.
    for (auto [i, t] : llvm::zip(idxs, tagged))
      readData[accKey(m.id, i)] =
          readCrossbar(c, vals, atReadData(m, t.first, sh));
  }
  // Reads coloured onto one port of one bank: `bindMemoryPorts` proved they
  // never issue in the same cycle, so one bus carries them all under a select
  // on their own activation.
  for (auto &[key, idxs] : shared) {
    auto [id, bank, port] = key;
    const uarch::MemUnit &m = dp.mems[id];
    Value rd = sharedReadPort(m, bank, port);
    for (unsigned i : idxs)
      readData[accKey(m.id, i)] = rd;
    // This region's own accesses select between themselves here, where their
    // addresses and their shell are; the bus itself is driven by
    // `finalizeSharedReadPorts`, once every region holding the port has done
    // the same.
    Value fired;
    Value addr =
        sharedAddress(m, idxs, issue, sh,
                      sharedInternalPort(m, bank, port) ? &fired : nullptr);
    sharedReads[key].arms.push_back({addr, fired});
  }
}

bool DatapathEmitter::sharedInternalPort(const uarch::MemUnit &m, unsigned bank,
                                         unsigned port) const {
  // A region is one holder however many of its accesses reach the port, since
  // they have already selected between themselves; a call is another.
  llvm::SmallDenseSet<uint64_t> holders;
  for (const uarch::MemUnit::Access &acc : m.accesses)
    if (!acc.isWrite && acc.staticBank == bank && acc.port == port)
      holders.insert(uint64_t(acc.region) << 1);
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs)
      if (!ma.isWrite && ma.mem == m.id && ma.bank == bank && ma.port == port)
        holders.insert((uint64_t(cu.id) << 1) | 1);
  return holders.size() > 1;
}

bool DatapathEmitter::multiRegionPort(const uarch::MemUnit &m,
                                      unsigned portIdx) {
  std::optional<unsigned> one;
  for (const uarch::MemUnit::Access &acc : m.accesses) {
    if (acc.isWrite || acc.portIdx != portIdx)
      continue;
    if (!one)
      one = acc.region;
    else if (*one != acc.region)
      return true;
  }
  return false;
}

Value DatapathEmitter::sharedReadPort(const uarch::MemUnit &m, unsigned bank,
                                      unsigned port) {
  SharedReadPort &p = sharedReads[{m.id, bank, port}];
  if (!p.data) {
    p.addr = c.bb.get(c.b.getIntegerType(addrWidth(m)));
    p.data = c.R(
        atPort(seq::ReadPortOp::create(c.b, c.loc, memReadCell(m, bank, port),
                                       ValueRange{Value(p.addr)},
                                       /*rdEn=*/Value(), m.readLatency),
               port));
  }
  return p.data;
}

Value DatapathEmitter::regionSelectedAddress(ArrayRef<SharedRead> arms) {
  assert(!arms.empty() && "a read port was built for no access");
  if (arms.size() == 1) {
    assert(!arms.front().fired &&
           "a port the binding gave to two regions got one arm, so a region "
           "holding it never emitted its accesses");
    return arms.front().addr;
  }
  // Between drives the bus keeps the last address: a read frozen by
  // back-pressure re-presents it, and an idle region must not put its stale
  // address back on a bus another region has taken. At most one arm is live in
  // a cycle (`portGraph` separates two that can overlap), so the priority order
  // carries no meaning.
  Type ty = arms.front().addr.getType();
  Backedge next = c.bb.get(ty);
  Value out = c.reg(next, c.konst(ty, 0));
  for (const SharedRead &a : llvm::reverse(arms)) {
    assert(a.fired && "an arm sharing a port with another region has to say "
                      "when it is presenting");
    out = c.mux(a.fired, a.addr, out);
  }
  next.setValue(out);
  return out;
}

void DatapathEmitter::finalizeSharedReadPorts() {
  for (auto &[key, p] : sharedReads)
    p.addr.setValue(regionSelectedAddress(p.arms));
  for (auto &[portIdx, arms] : boundaryReads) {
    uarch::AccRef r = dp.readPorts[portIdx];
    pa.setOutput(portAddr(dp.mems[r.id].accesses[r.idx].portBase),
                 regionSelectedAddress(arms));
  }
}

// The address ONE REGION's accesses on a read port present: each drives it on
// its own issue cycle, and the select is held with the datapath so a read
// frozen by back-pressure keeps re-presenting its address until its datum is
// taken. A port with one access here is that access's address.
Value DatapathEmitter::sharedAddress(const uarch::MemUnit &m,
                                     ArrayRef<unsigned> idxs, Value issue,
                                     const StallShell &sh, Value *fired) {
  auto addrOf = [&](unsigned i) {
    Value off = bankAddress(m, m.accesses[i]).offset;
    return m.external ? boundaryAddr(c, off) : memAddr(m, off);
  };
  // Every pulse below says when its access is presenting; only an access alone
  // on a port no one else holds needs none and drives it unconditionally.
  assert((issue || (idxs.size() == 1 && !fired)) &&
         "a region with no issue pulse cannot say when it is driving a port; "
         "`bindMemoryPorts` leaves such a read alone on one");
  // One access needs no select, and its address is already held against this
  // region's shell (`bankAddress`), so a second hold would be the same register
  // twice.
  if (idxs.size() == 1) {
    if (fired)
      *fired = c.activationPulse(issue, m.accesses[idxs.front()].stage, sh);
    return addrOf(idxs.front());
  }
  SmallVector<Value> addrs, sels;
  for (unsigned i : idxs) {
    addrs.push_back(addrOf(i));
    sels.push_back(c.activationPulse(issue, m.accesses[i].stage, sh));
  }
  // Any of them presenting is this region driving the port, which is what a
  // port another region also holds selects on.
  if (fired)
    for (Value s : sels)
      *fired = *fired ? c.orBits(*fired, s) : s;
  return c.stallHold(c.oneHotSelect(addrs, sels), sh);
}

// Drive the read-address port of each single-interface external read in region
// \p rb: the in-bank offset for a statically-banked argument (the boundary
// presents one interface per bank), the flat element index for an unbanked one.
// A data-dependent banked read spans every interface, and emitReads
// drives all of its addresses.
void DatapathEmitter::emitExternalReadAddrs(const uarch::RegionBlock &rb,
                                            Value issue) {
  StallShell sh = shellFor(rb.id);
  // One address per port group, the accesses sharing it selecting on their own
  // activation as they do on an internal port. A group's accesses are all in
  // this region, the granularity `bindMemoryPorts` binds a read port at.
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> shared;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    // A scattered argument has no address port to drive, and a data-dependent
    // banked one spans every interface (`emitReads`).
    if (!m.external || acc.isWrite || acc.plan != PortPlan::Coloured)
      continue;
    shared[{r.id, acc.portIdx}].push_back(r.idx);
  }
  for (auto &[key, idxs] : shared) {
    const uarch::MemUnit &m = dp.mems[key.first];
    unsigned portIdx = key.second;
    // The group is one module output, so a second region holding it cannot
    // drive it here; `finalizeSharedReadPorts` does, once both have presented.
    Value fired;
    Value addr = sharedAddress(m, idxs, issue, sh,
                               multiRegionPort(m, portIdx) ? &fired : nullptr);
    boundaryReads[portIdx].push_back({addr, fired});
  }
}

// The drain stage a store contributes to its region's `done`. The write is
// PRESENTED at its stage and COMMITS `writeLatency` cycles later; `emitDone`
// rides its own latch register for the last of those cycles (done reads 1 at
// `lastIssue + drainStage + 1`), so the stage is the commit cycle minus one.
static unsigned storeDrainOf(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  assert(m.writeLatency >= 1 &&
         "a zero-cycle write has no commit edge for the done latch to ride; "
         "checkDeviceCapability must have rejected the device row");
  return acc.stage + m.writeLatency - 1;
}

void DatapathEmitter::emitWrites(const uarch::RegionBlock &rb, Value issue,
                                 DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
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
  llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>> lanes;
  for (uarch::AccRef r : rb.memAccesses) {
    const uarch::MemUnit &m = dp.mems[r.id];
    const uarch::MemUnit::Access &acc = m.accesses[r.idx];
    if (!acc.isWrite)
      continue;
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, storeDrainOf(m, acc));
    if (acc.plan == PortPlan::Lane) {
      lanes[{r.id, acc.lane}].push_back(r.idx);
      continue;
    }
    // A `seq.hlmem` write port realizes exactly one cycle, so an internal
    // memory whose device latency is deeper presents address, data and enable
    // `writeLatency - 1` cycles late; the datum still lands at `stage +
    // writeLatency` (see `storeDrainOf`). A boundary port is the caller's
    // memory rather than an hlmem, and takes its terms at its stage.
    unsigned pre = m.external ? 0 : m.writeLatency - 1;
    auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
    Value we =
        c.delayValid(c.activationPulse(commitPulse(), acc.stage, sh), pre, sh);
    Value data = late(resolveSource(acc.data));
    switch (acc.plan) {
    case PortPlan::Table:
      llvm_unreachable("a constant table has no write port; `verifyDatapath` "
                       "refuses a store to one");
    case PortPlan::Lane:
      llvm_unreachable("a lane's stores are delayed and demuxed together, "
                       "below, so they leave the loop above this");
    case PortPlan::ElementWise:
      // The cells are shared by every store, so this only records the terms:
      // `finalizeScatteredPorts` drives an ARGUMENT's element ports, or builds
      // an internal array's registers, once every region and call has
      // contributed.
      scatterWrites[m.id].push_back({we, scatterIndex(m, acc), data});
      break;
    case PortPlan::Coloured: {
      // A compile-time bank writes its own memory: no demux, and no write port
      // on the other banks. An unbanked memref is the same case at bank 0. One
      // interface (or one bus) carries every store bound to the port, so it is
      // driven once all of them have emitted, by `finalizeBoundaryWritePorts`
      // or `finalizeSharedWritePorts`.
      auto bs = bankAddress(m, acc);
      if (m.external)
        boundaryWrites[acc.portIdx].push_back(
            {boundaryAddr(c, bs.offset), data, we});
      else
        sharedWrites[m.id].push_back(
            {*acc.staticBank, acc.port, late(memAddr(m, bs.offset)), data, we});
      break;
    }
    case PortPlan::Crossbar: {
      // Drive every bank; the runtime bank gates each write-enable so only the
      // target bank commits (an N-way demux). Such a store reaches every bank,
      // so it holds a port of its own on each and shares none.
      auto bs = bankAddress(m, acc);
      if (m.external) {
        Value addr = boundaryAddr(c, bs.offset);
        for (const auto &[bank, base] : extPorts(m, acc)) {
          pa.setOutput(portAddr(base), addr);
          pa.setOutput(portData(base), data);
          pa.setOutput(portWe(base), writeDemux(c, we, bs.bank, bank));
        }
        break;
      }
      Value addr = late(memAddr(m, bs.offset));
      Value bank = late(bs.bank);
      for (unsigned k = 0; k < m.numBanks; ++k)
        for (Value cell : memWriteCells(m, k))
          atPort(seq::WritePortOp::create(c.b, c.loc, cell, ValueRange{addr},
                                          data, writeDemux(c, we, bank, k),
                                          c.b.getI64IntegerAttr(1)),
                 acc.port);
      break;
    }
    }
  }
  // One write port per bank per lane. Bank k takes the address and data of
  // whichever of the lane's accesses reaches it, and its write-enable is the OR
  // of their demuxed enables, so an access commits on its own bank and nowhere
  // else. The OR has at most one live arm for the same reason the address
  // select does (`laneSelect`).
  for (auto &[key, idxs] : lanes) {
    const uarch::MemUnit &m = dp.mems[key.first];
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
      wes.push_back(c.delayValid(
          c.activationPulse(commitPulse(), acc.stage, sh), pre, sh));
    }
    auto wlat = c.b.getI64IntegerAttr(1);
    for (unsigned k = 0; k < m.numBanks; ++k) {
      Value we = writeDemux(c, wes[0], bankOf[0], k);
      for (unsigned i = 1; i < idxs.size(); ++i)
        we = c.orBits(we, writeDemux(c, wes[i], bankOf[i], k));
      // Deliberately untagged: a skew assigns its ports by lane rather than by
      // the port graph, so nothing proves this store and a read of the same
      // bank stay out of each other's cycle, and only that proof lets the two
      // share one address.
      for (Value cell : memWriteCells(m, k))
        seq::WritePortOp::create(c.b, c.loc, cell,
                                 ValueRange{laneSelect(c, addrs, k)},
                                 laneSelect(c, datas, k), we, wlat);
    }
  }
}

// Drive each boundary write port group from the stores bound to it: a one-hot
// select for the same reason as the shared internal ports below, and a single
// store's own terms where it has the group to itself.
void DatapathEmitter::finalizeBoundaryWritePorts() {
  for (auto &[portIdx, writes] : boundaryWrites) {
    uarch::AccRef r = dp.writePorts[portIdx];
    llvm::StringRef base = dp.mems[r.id].accesses[r.idx].portBase;
    Value addr, data, we;
    for (const BoundaryWrite &w : writes) {
      addr = addr ? c.mux(w.we, w.addr, addr) : w.addr;
      data = data ? c.mux(w.we, w.data, data) : w.data;
      we = we ? c.orBits(we, w.we) : w.we;
    }
    pa.setOutput(portAddr(base), addr);
    pa.setOutput(portData(base), data);
    pa.setOutput(portWe(base), we);
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
    for (unsigned k = 0; k < m.numBanks; ++k)
      for (unsigned p = 0; p < ports; ++p) {
        Value addr, data, we;
        for (const SharedWrite &w : writes) {
          if (w.bank != k || w.port != p)
            continue;
          addr = addr ? c.mux(w.we, w.addr, addr) : w.addr;
          data = data ? c.mux(w.we, w.data, data) : w.data;
          we = we ? c.orBits(we, w.we) : w.we;
        }
        if (!we)
          continue;
        // The same port on every instance of the bank: a copy that missed a
        // write would stop holding the same array.
        for (Value cell : memWriteCells(m, k))
          atPort(seq::WritePortOp::create(c.b, c.loc, cell, ValueRange{addr},
                                          data, we, c.b.getI64IntegerAttr(1)),
                 p);
      }
  }
}

// Settle each scattered memory's elements from every store recorded against it:
// per element the datum is a priority mux over the stores that reach it, and
// the write-enable the OR of their demuxed pulses. An ARGUMENT's cells are the
// caller's and this drives its element ports; an INTERNAL array's are this
// module's and this builds them, one enabled register per element.
//
// At most one arm is live per element per cycle, so the priority order carries
// no meaning. Unlike a skewed lane that is not structural here: two stores to
// one element are ordered by the dependence analysis, while two stores to
// DIFFERENT elements in one cycle are what a complete partition's unlimited
// ports are for. A constant subscript folds its `icmpEq` away, so `A[3] = x`
// leaves element 3 driven and the other N-1 write-enables constant false.
void DatapathEmitter::finalizeScatteredPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.scattered)
      continue;
    ArrayRef<ScatterWrite> writes;
    if (auto it = scatterWrites.find(m.id); it != scatterWrites.end())
      writes = it->second;
    // Selected by the PULSE, not the index: two stores in different regions can
    // name element k at once (an idle region's stale address register), so only
    // the enabled one may drive; the first arm is a don't-care.
    auto driveOf = [&](unsigned k) {
      Value data, we;
      for (const ScatterWrite &w : writes) {
        Value hits = writeDemux(c, w.we, w.index, k);
        data = data ? c.mux(hits, w.data, data) : w.data;
        we = we ? c.orBits(we, hits) : hits;
      }
      return std::pair{data, we};
    };
    if (m.external) {
      if (writes.empty())
        continue; // read-only: the caller's cells arrive and never leave
      for (auto [k, p] : llvm::enumerate(m.elemPorts)) {
        auto [data, we] = driveOf(k);
        pa.setOutput(p.out, data);
        pa.setOutput(p.we, we);
      }
      continue;
    }
    // An element no store reaches holds its reset value for the whole run, so
    // it is that constant rather than a register.
    IntegerType elemTy = memElemType(m, c.b);
    for (auto [k, be] : llvm::enumerate(scatterElems[m.id])) {
      auto [data, we] = driveOf(k);
      Value zero = c.konst(elemTy, 0);
      if (!we) {
        be.setValue(zero);
        continue;
      }
      Value cell = c.enabledReg(data, we, zero, RegRole::Storage);
      nameValue(cell, memElemName(dp, m, k));
      be.setValue(cell);
    }
  }
}

// Master each buffer from child \p cu's addr/data/we outputs (\p outs): a
// boundary argument passes straight through to the top port, an internal one
// reaches its storage the way the parent's own accesses do. One arm per
// `PortPlan`, as `emitReads` and `emitWrites`.
void DatapathEmitter::masterCallPorts(
    const uarch::CallUnit &cu, llvm::StringMap<Value> &outs,
    llvm::StringMap<circt::Backedge> &rdBackedge,
    llvm::function_ref<Value()> runWindow, const StallShell &sh) {
  for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
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
    switch (ma.plan) {
    case PortPlan::Crossbar:
      llvm_unreachable("a child masters one bank, indexed in that bank's own "
                       "space, so it never crossbars");
    case PortPlan::Lane:
      llvm_unreachable("a child masters a port on a skewed array; a lane is "
                       "assigned from this module's own accesses and the "
                       "child holds none. `checkEmitterSubset` refuses it");
    case PortPlan::Table: {
      // A constant table the child only reads: one `hw.array_get` registered
      // to the latency the child was timed against, so the datum lands
      // exactly where a RAM's would.
      Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id],
                                              memAddr(m, outs[ma.addr])));
      rdBackedge[ma.data].setValue(atReadData(m, elem, sh));
      break;
    }
    case PortPlan::ElementWise: {
      // A scattered array holds no addressable port, so the child's addressed
      // one is served off the element registers: a select for its read, a
      // term per store for its write. The child keeps the ordinary port ABI,
      // and a constant address folds both away.
      assert(ma.bank == 0 && "a scattered array is one bank, so a child "
                             "masters it in whole-array element space");
      Value idx = addrAt(c.b, c.loc, outs[ma.addr], kDatapathAddressWidth);
      if (ma.isWrite)
        scatterWrites[m.id].push_back({outs[ma.we], idx, outs[ma.data]});
      else
        rdBackedge[ma.data].setValue(readCrossbar(c, scatterValues(m.id), idx));
      break;
    }
    case PortPlan::Coloured: {
      // One hlmem per bank: the child masters bank `ma.bank`, already indexed
      // in that bank's own space via `allo.part`, so this routes straight to
      // it with no crossbar (validateDatapath rejects a partition mismatch).
      assert(ma.bank < m.numBanks &&
             "child bank index exceeds the buffer's bank count; "
             "validateDatapath must have rejected the partition mismatch");
      Value addr = memAddr(m, outs[ma.addr]);
      // The child was compiled against this buffer's device latency, read
      // here from the MemUnit since the parent never accesses the buffer
      // itself. A deeper write pipelines into the fixed 1-cycle port, as
      // emitWrites.
      if (ma.isWrite) {
        unsigned pre = m.writeLatency - 1;
        Value a = c.shiftChain(addr, pre, sh).last();
        Value d = c.shiftChain(outs[ma.data], pre, sh).last();
        Value w = c.delayValid(outs[ma.we], pre, sh);
        // The binding settles a call's write port too, so two ports of one
        // child that declared them independent land in separate `always`
        // blocks and the array still infers a true dual port.
        sharedWrites[m.id].push_back({ma.bank, ma.port, a, d, w});
        break;
      }
      // The port may also be held by a sibling call or by the parent's own
      // accesses, so the datum comes off the one `seq.read` they share and
      // the address joins its arms.
      rdBackedge[ma.data].setValue(sharedReadPort(m, ma.bank, ma.port));
      Value fired;
      if (sharedInternalPort(m, ma.bank, ma.port))
        fired = runWindow();
      sharedReads[{m.id, ma.bank, ma.port}].arms.push_back({addr, fired});
      break;
    }
    }
  }
}

} // namespace mlir::allo::uarch
