/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The memory subsystem's emit half: how an access reaches its storage. Three
// dispatches on `PortPlan` (`emitReads`, `emitWrites`, `masterCallPorts`), plus
// the address hardware they share and the finalizers a port shared between
// regions needs. What the storage is, and which ports each access holds, is
// decided in Memory.cpp.
//===----------------------------------------------------------------------===//

#include "allo/IR/AlloOps.h" // kIndependentWritesAttr
#include "allo/Microarch/HWEmitter.h"

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

// An address on its way out of the module. A boundary address port is
// `kDatapathAddressWidth` wide for every argument, the contract the manifest
// and the cosim harness are written against, so a narrow in-bank address widens
// back here.
static Value boundaryAddr(EmitContext &c, Value addr) {
  return addrAt(c.b, c.loc, addr, kDatapathAddressWidth);
}

// Which of several sources bank \p k takes, each tagged with the bank it
// reaches: the inverse of `readCrossbar`. At most one tag equals `k` at a time,
// since a lane holds distinct slots and distinct slots are distinct banks at
// every rotation, so the selects are one-hot; with no tag on `k` the result
// is 0, a don't-care behind the port's own enable.
static Value laneSelect(EmitContext &c,
                        ArrayRef<std::pair<Value, Value>> tagged, unsigned k) {
  if (tagged.size() == 1)
    return tagged.front().second;
  c.muxLedger.add(MuxRole::Crossbar, tagged.size(),
                  datapathWidth(tagged.front().second.getType()));
  SmallVector<Value> vals, sels;
  for (const auto &[bank, val] : tagged) {
    vals.push_back(val);
    sels.push_back(c.icmpEq(bank, k));
  }
  return c.oneHotSelect(vals, sels);
}

// Build one cone \p r of this access's address as hardware at \p width, out of
// the parts `planAddressGenerators` split it into: a constant, one register per
// strength-reduced term (`RegionBlock::addrStrides`, advanced by the
// controller), and whatever did not reduce. The residual is added after the
// delay chain, its operands arriving already delayed where the counters run
// live, which puts both halves in the access's own cycle.
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
    // delayed on its own. Appended at the datapath width, which is what
    // `evalAffine` reads its operands at.
    SmallVector<Value> idx; // the access's own index sources, dims then symbols
    // An operand the reduction folded into a scaled counter has an empty slot
    // and no position in this residual, so nothing reads the gap.
    for (const uarch::Source &s : acc.addr)
      idx.push_back(s ? resolveSource(s) : Value());
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
// offset is the flat index and whose digit nothing builds. Both halves are the
// `Reduced` cones `planAddressGenerators` already split.
BankSplit DatapathEmitter::bankAddress(const uarch::MemUnit &m,
                                       const uarch::MemUnit::Access &acc) {
  assert(acc.addrMap && "dcp memory access without an affine map");
  Value offset = buildAddr(acc, acc.offset, addrWidth(m));
  // The digit's cone is built at the datapath width so its intermediates keep
  // their range, then narrowed to clog2(numBanks): consumers delay it and
  // compare it against literal bank numbers, and `icmpEq` follows its width.
  // It reduces like the offset: `counter mod F` is a register that wraps, not
  // a `mod` on the setup path.
  Value bank =
      acc.hasBankCone
          ? addrAt(c.b, c.loc, buildAddr(acc, acc.bank, kDatapathAddressWidth),
                   std::max(1u, llvm::Log2_64_Ceil(m.numBanks)))
          : Value();
  // No hold here: a boundary read address is held once where it leaves the
  // module (`sharedAddress`, the crossbar read), and an internal port keeps
  // its in-flight datum through its read enable instead.
  return {bank, offset};
}

// Narrow to the clog2(depth)-bit index `seq.hlmem` / `hw.array_get` expects,
// which is also the width `bankAddress` carries its arithmetic at.
Value DatapathEmitter::memAddr(const uarch::MemUnit &m, Value addr) {
  return addrAt(c.b, c.loc, addr, addrWidth(m));
}

// Which element of a scattered memory \p acc names, at the memory's own
// address width. The crossbar and the write demux compare it against literal
// element numbers (`icmpEq` builds those at its width).
Value DatapathEmitter::scatterIndex(const uarch::MemUnit &m,
                                    const uarch::MemUnit::Access &acc) {
  assert(m.scattered && "an element index belongs to a scattered memory");
  return bankAddress(m, acc).offset;
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
// together. A data-dependent banked read has one data port per bank and is
// bound by emitReads, which muxes them in-region.
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
      // read latency in emitReads. No writable hlmem and no write ports.
      SmallVector<Attribute> fields;
      for (const APInt &w :
           initWords(cast<ElementsAttr>(m.romInit), m.width, depth))
        fields.push_back(IntegerAttr::get(elemTy, w));
      // A hw.array indexes element 0 as the last aggregate_constant field, so
      // the natural-order initializer is reversed to make array_get(i) ==
      // data[i].
      std::reverse(fields.begin(), fields.end());
      romArray[m.id] = hw::AggregateConstantOp::create(
          c.b, c.loc, hw::ArrayType::get(elemTy, depth),
          c.b.getArrayAttr(fields));
      continue;
    }
    // A completely partitioned array is one register per element rather than an
    // addressed memory, which is what buys the unlimited combinational ports
    // the scheduler was billed against. Only the backedges here; the registers
    // need every store, so `finalizeScatteredPorts` builds them. Exactly
    // `depthWords` of them, not `declaredDepth`: the padding word only keeps an
    // hlmem's address one bit wide, and an element is selected by comparison.
    if (realization == R::Scatter) {
      SmallVector<Backedge> elems;
      for (unsigned k = 0; k < m.depthWords; ++k)
        elems.push_back(c.bb.get(elemTy));
      scatterElems[m.id] = std::move(elems);
      continue;
    }
    // One cell per instance of each bank, bank-major. Reads past what one
    // instance of the row has are served by another copy of the whole array.
    SmallVector<Value> banks;
    for (unsigned k = 0; k < m.numBanks; ++k)
      for (unsigned i = 0; i < m.instances; ++i) {
        auto mem = seq::HLMemOp::create(c.b, c.loc, c.clk, c.rst,
                                        memCellName(dp, m, k, i),
                                        {static_cast<int64_t>(depth)}, elemTy);
        // The port binding proved these writes never collide, which lets the
        // lowering put each in its own `always` block and build a true dual
        // port. Without it they share one block, which arbitrates.
        if (m.writesIndependent)
          mem->setAttr(kIndependentWritesAttr, c.b.getUnitAttr());
        // Pin the array to the row it is realized in. Leaving it unsaid hands
        // the structure to the synthesizer, which then builds something the
        // cost model did not price.
        if (!m.ramStyle.empty())
          mem->setAttr(kRamStyleAttr, c.b.getStringAttr(m.ramStyle));
        // An initialized array the kernel also writes is a real memory that
        // starts with contents. `seq.hlmem` carries no initializer, so the
        // words ride to the seq->SV pipeline, which gives the backing reg an
        // `initial` block. Every copy starts with them.
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

// Emit region \p rb's reads, one arm per `PortPlan`. Read latency is the
// memory's device-resolved `readLatency`, the number the scheduler timed the
// access at, so the datum lands on exactly the cycle the consumer's register
// depth was solved against.
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
    case PortPlan::ElementWise: {
      // No address port: a read selects over the cells, and a constant
      // subscript folds the select away. An argument's cells arrive on its
      // element input ports, an internal array's are this module's registers.
      // Either way they are timed at read latency 0.
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
    case PortPlan::Table: {
      // A constant table read: index the aggregate_constant combinationally,
      // then register to the scheduled read latency so timing matches a RAM.
      Value idx = memAddr(m, bankAddress(m, acc).offset);
      readData[accKey(m.id, r.idx)] = atReadData(
          m, c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id], idx)), sh);
      break;
    }
    case PortPlan::Coloured:
      // A compile-time bank reads its own memory: no crossbar, and no read port
      // on the other banks. An unbanked memref is the same case at bank 0. An
      // argument's group is not built here: its datum is the port's, bound by
      // `bindReadPorts`, and its address by `emitExternalReadAddrs`.
      if (!m.external)
        shared[{r.id, *acc.staticBank, acc.port}].push_back(r.idx);
      break;
    case PortPlan::Lane:
      lanes[{r.id, acc.lane}].push_back(r.idx);
      break;
    case PortPlan::Crossbar: {
      // Read every bank at the (bank-independent) offset, then select by the
      // runtime bank, aligned with the read data. Such an access reaches every
      // bank, so it holds a port of its own on each. A boundary address is
      // held against back-pressure before it widens; an internal port freezes
      // through its read enable instead.
      auto bs = bankAddress(m, acc);
      SmallVector<Value> vals;
      if (m.external) {
        Value addr = boundaryAddr(c, c.stallHold(bs.offset, sh));
        for (const auto &[bank, base] : extPorts(m, acc)) {
          pa.setOutput(portAddr(base), addr);
          vals.push_back(pa.getInput(portData(base)));
        }
      } else {
        Value addr = memAddr(m, bs.offset);
        for (unsigned k = 0; k < m.numBanks; ++k)
          vals.push_back(c.R(atPort(
              seq::ReadPortOp::create(
                  c.b, c.loc, memReadCell(m, k, acc.port), ValueRange{addr},
                  /*rdEn=*/sh ? sh.chainEnable : Value(), m.readLatency),
              acc.port)));
      }
      readData[accKey(m.id, r.idx)] =
          readCrossbar(c, vals, atReadData(m, bs.bank, sh));
      break;
    }
    }
  }
  // One read port per bank per lane rather than per bank per access. A lane's
  // accesses hold distinct slots, so bank k takes the offset of whichever of
  // them reaches it and hands its datum back to that one: F accesses over F
  // banks at one port each, where a crossbar would take a port on every bank
  // for every access.
  for (auto &[key, idxs] : lanes) {
    const uarch::MemUnit &m = dp.mems[key.first];
    SmallVector<std::pair<Value, Value>> tagged; // (runtime bank, in-bank addr)
    for (unsigned i : idxs) {
      BankSplit bs = bankAddress(m, m.accesses[i]);
      tagged.emplace_back(bs.bank, memAddr(m, bs.offset));
    }
    // Untagged: a lane is assigned by the skew rather than by the port graph,
    // so it proves nothing about what else touches this bank.
    SmallVector<Value> vals;
    for (unsigned k = 0; k < m.numBanks; ++k)
      vals.push_back(c.R(seq::ReadPortOp::create(
          c.b, c.loc, memReadCell(m, k, key.second),
          ValueRange{laneSelect(c, tagged, k)},
          /*rdEn=*/sh ? sh.chainEnable : Value(), m.readLatency)));
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
    // addresses and their shell are. The bus itself is driven by
    // `finalizeSharedReadPorts`, once every region holding the port has
    // contributed its arm.
    Value fired;
    Value addr =
        sharedAddress(m, idxs, issue, sh,
                      sharedInternalPort(m, bank, port) ? &fired : nullptr);
    SharedReadPort &p = sharedReads[key];
    p.arms.push_back({fired, addr, Value()});
    ++p.owners;
    p.ownerRegion = rb.id;
  }
}

bool DatapathEmitter::sharedInternalPort(const uarch::MemUnit &m, unsigned bank,
                                         unsigned port) const {
  // A region is one holder however many of its accesses reach the port, since
  // they have already selected between themselves; a call is another.
  llvm::SmallDenseSet<uint64_t> holders;
  for (const uarch::MemUnit::Access &acc : m.accesses)
    if (!acc.isWrite && acc.staticBank.value_or(0) == bank && acc.port == port)
      holders.insert(uint64_t(acc.region) << 1);
  for (const uarch::CallUnit &cu : dp.calls)
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs)
      if (!ma.isWrite && ma.mem == m.id && ma.bank == bank && ma.port == port)
        holders.insert((uint64_t(cu.id) << 1) | 1);
  return holders.size() > 1;
}

Value DatapathEmitter::sharedReadPort(const uarch::MemUnit &m, unsigned bank,
                                      unsigned port) {
  SharedReadPort &p = sharedReads[{m.id, bank, port}];
  if (!p.data) {
    p.addr = c.bb.get(c.b.getIntegerType(addrWidth(m)));
    // The read enable is a promise too: whether one owner's shell may freeze
    // the port is only known once every holder has contributed.
    p.rdEnBE = c.bb.get(c.i1);
    p.data = c.R(
        atPort(seq::ReadPortOp::create(c.b, c.loc, memReadCell(m, bank, port),
                                       ValueRange{Value(p.addr)},
                                       Value(p.rdEnBE), m.readLatency),
               port));
  }
  return p.data;
}

DatapathEmitter::SinkArm DatapathEmitter::commitSink(ArrayRef<SinkArm> arms,
                                                     Idle idle) {
  assert(!arms.empty() && "a shared port was built for no driver");
  // One unconditional arm is the port: nothing to select between, and nothing
  // for an idle cycle to take it away from.
  if (arms.size() == 1 && !arms.front().fired)
    return arms.front();
  SinkArm out;
  for (const SinkArm &a : arms) {
    assert(a.fired && "an arm sharing a sink has to say when it is presenting");
    out.fired = out.fired ? c.orBits(out.fired, a.fired) : a.fired;
  }
  auto reduce = [&](llvm::function_ref<Value(const SinkArm &)> term) -> Value {
    if (!term(arms.front()))
      return {}; // a term this sink does not carry
    // A held sink has one more arm than the drivers: the idle register.
    c.muxLedger.add(MuxRole::Commit, arms.size() + (idle == Idle::Hold ? 1 : 0),
                    datapathWidth(term(arms.front()).getType()));
    // The arms are exclusive by construction (the binding proved two drivers
    // never enabled together), so the reduction is the log-depth AND-OR
    // `muxLevels` prices rather than an arms-1 priority chain. With nothing
    // fired it reads 0, a don't-care behind `out.fired`.
    SmallVector<Value> vals, sels;
    for (const SinkArm &a : arms) {
      vals.push_back(term(a));
      sels.push_back(a.fired);
    }
    Value hot = c.oneHotSelect(vals, sels);
    if (idle == Idle::DontCare)
      return hot;
    // Between drives the bus holds its last value: a read frozen by
    // back-pressure re-presents its address, and an idle region must not put a
    // stale one back on a bus another region has taken.
    Type ty = term(arms.front()).getType();
    Backedge next = c.bb.get(ty);
    Value held = c.reg(next, c.konst(ty, 0));
    Value res = c.mux(out.fired, hot, held);
    next.setValue(res);
    return res;
  };
  out.addr = reduce([](const SinkArm &a) { return a.addr; });
  out.data = reduce([](const SinkArm &a) { return a.data; });
  return out;
}

void DatapathEmitter::finalizeSharedReadPorts() {
  auto address = [&](ArrayRef<SinkArm> arms) {
    assert(!arms.empty() && "a read port was built for no access");
    assert((arms.size() > 1 || !arms.front().fired) &&
           "a port the binding gave to two regions got one arm, so a region "
           "holding it never emitted its accesses");
    return commitSink(arms, Idle::Hold).addr;
  };
  for (auto &[key, p] : sharedReads) {
    // The port freezes with its owner where that is unambiguous: a lone
    // region's chain enable keeps the in-flight datum in the port's own
    // register. Several holders read every cycle off the held bus instead
    // (a constant-true enable, which the hlmem lowering folds away). The
    // shell is read here, resolved, not captured at contribution time.
    StallShell sh = p.owners == 1 && p.ownerRegion ? shellFor(*p.ownerRegion)
                                                   : StallShell{};
    p.rdEnBE.setValue(sh ? sh.chainEnable : c.t1);
    p.addr.setValue(address(p.arms));
  }
  for (auto &[base, arms] : boundaryReads)
    pa.setOutput(portAddr(base), address(arms));
}

// The address one region's accesses on a read port present: each drives it on
// its own issue cycle, and the select is held with the datapath so a read
// frozen by back-pressure keeps re-presenting its address until its datum is
// taken. A port with one access here is that access's address.
Value DatapathEmitter::sharedAddress(const uarch::MemUnit &m,
                                     ArrayRef<unsigned> idxs, Value issue,
                                     const StallShell &sh, Value *fired) {
  // Select and hold at the bank's own address width; a boundary port widens
  // after, so neither runs at the 32-bit boundary contract.
  auto addrOf = [&](unsigned i) {
    return memAddr(m, bankAddress(m, m.accesses[i]).offset);
  };
  // One hold after the select: a read frozen by back-pressure keeps
  // re-presenting its address until its datum is taken.
  auto out = [&](Value addr) {
    addr = c.stallHold(addr, sh);
    return m.external ? boundaryAddr(c, addr) : addr;
  };
  // Every pulse below says when its access is presenting; only an access alone
  // on a port no one else holds needs none and drives it unconditionally.
  assert((issue || (idxs.size() == 1 && !fired)) &&
         "a region with no issue pulse cannot say when it is driving a port; "
         "`bindMemoryPorts` leaves such a read alone on one");
  if (idxs.size() == 1) {
    if (fired)
      *fired = c.activationPulse(issue, m.accesses[idxs.front()].stage, sh);
    return out(addrOf(idxs.front()));
  }
  SmallVector<Value> addrs, sels;
  for (unsigned i : idxs) {
    addrs.push_back(addrOf(i));
    sels.push_back(c.activationPulse(issue, m.accesses[i].stage, sh));
  }
  c.muxLedger.add(MuxRole::Address, addrs.size(),
                  datapathWidth(addrs.front().getType()));
  // Any of them presenting is this region driving the port, which is what a
  // port held by another region as well selects on.
  if (fired)
    for (Value s : sels)
      *fired = *fired ? c.orBits(*fired, s) : s;
  return out(c.oneHotSelect(addrs, sels));
}

// Drive the read-address port of each single-interface external read in region
// \p rb: the in-bank offset for a statically-banked argument (the boundary
// presents one interface per bank), the flat element index for an unbanked one.
// A data-dependent banked read spans every interface, and emitReads drives all
// of its addresses.
void DatapathEmitter::emitExternalReadAddrs(const uarch::RegionBlock &rb,
                                            Value issue) {
  StallShell sh = shellFor(rb.id);
  // One address per port group, the accesses sharing it selecting on their own
  // activation as they do on an internal port.
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
    const uarch::MemUnit::Access &acc = m.accesses[idxs.front()];
    // The group is one module output, so a second holder (another region's
    // accesses, or a child mastered on the colour) cannot drive it here;
    // `finalizeSharedReadPorts` does, once every holder has presented.
    Value fired;
    bool held = sharedInternalPort(m, acc.staticBank.value_or(0), acc.port);
    Value addr = sharedAddress(m, idxs, issue, sh, held ? &fired : nullptr);
    boundaryReads[acc.portBase].push_back({fired, addr, Value()});
  }
}

// The drain stage a store contributes to its region's `done`. The write is
// presented at its stage and commits `writeLatency` cycles later; `emitDone`
// rides its own latch register for the last of those cycles (done reads 1 at
// `lastIssue + drainStage + 1`), so the stage is the commit cycle minus one.
static unsigned storeDrainOf(const uarch::MemUnit &m,
                             const uarch::MemUnit::Access &acc) {
  assert(m.writeLatency >= 1 &&
         "a zero-cycle write has no commit edge for the done latch to ride; "
         "`assertModelInvariants` holds the device row to that");
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
    // writeLatency` (see `storeDrainOf`). A boundary port takes its terms at
    // its stage.
    unsigned pre = m.external ? 0 : m.writeLatency - 1;
    auto late = [&](Value v) { return c.shiftChain(v, pre, sh).last(); };
    Value we =
        c.delayValid(c.activationPulse(commitPulse(), acc.stage, sh), pre, sh);
    Value data = late(resolveSource(acc.data));
    switch (acc.plan) {
    case PortPlan::ElementWise:
      // The cells are shared by every store, so this only records the terms:
      // `finalizeScatteredPorts` drives an argument's element ports, or builds
      // an internal array's registers, once every region and call has
      // contributed.
      scatterWrites[m.id].push_back({we, scatterIndex(m, acc), data});
      break;
    case PortPlan::Table:
      llvm_unreachable("a constant table has no write port; an array "
                       "anything writes is never classified as one");
    case PortPlan::Coloured: {
      // A compile-time bank writes its own memory: no demux, and no write port
      // on the other banks. An unbanked memref is the same case at bank 0. One
      // interface carries every store bound to the port, driven once all of
      // them have emitted, by `finalizeBoundaryWritePorts` or
      // `finalizeSharedWritePorts`.
      auto bs = bankAddress(m, acc);
      if (m.external)
        boundaryWrites[acc.portBase].push_back(
            {we, boundaryAddr(c, bs.offset), data});
      else
        sharedWrites[m.id].push_back({*acc.staticBank,
                                      acc.port,
                                      {we, late(memAddr(m, bs.offset)), data}});
      break;
    }
    case PortPlan::Lane:
      llvm_unreachable("a lane's stores are delayed and demuxed together, "
                       "below, so they leave the loop above this");
    case PortPlan::Crossbar: {
      // Drive every bank; the runtime bank gates each write-enable so only the
      // target bank commits (an N-way demux).
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
  // else. The OR has at most one live arm, as `laneSelect` does.
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
      // Untagged: a skew assigns its ports by lane rather than by the port
      // graph, so nothing proves this store and a read of the same bank stay
      // out of each other's cycle, and only that proof lets the two share one
      // address.
      for (Value cell : memWriteCells(m, k))
        seq::WritePortOp::create(c.b, c.loc, cell,
                                 ValueRange{laneSelect(c, addrs, k)},
                                 laneSelect(c, datas, k), we, wlat);
    }
  }
}

// Drive each boundary write port group from the stores bound to it: a one-hot
// select over them, or a single store's own terms where it has the group to
// itself.
void DatapathEmitter::finalizeBoundaryWritePorts() {
  for (auto &[base, writes] : boundaryWrites) {
    SinkArm out = commitSink(writes, Idle::DontCare);
    pa.setOutput(portAddr(base), out.addr);
    pa.setOutput(portData(base), out.data);
    pa.setOutput(portWe(base), out.fired);
  }
}

// Drive an array's shared write ports from the stores coloured onto each. Two
// stores on one port are provably never enabled in the same cycle, which is
// what lets `commitSink` reduce them as a one-hot select.
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
        SmallVector<SinkArm, 2> onPort;
        for (const SharedWrite &w : writes)
          if (w.bank == k && w.port == p)
            onPort.push_back(w.arm);
        if (onPort.empty())
          continue;
        SinkArm out = commitSink(onPort, Idle::DontCare);
        // The same port on every instance of the bank: a copy that missed a
        // write would stop holding the same array.
        for (Value cell : memWriteCells(m, k))
          atPort(seq::WritePortOp::create(c.b, c.loc, cell,
                                          ValueRange{out.addr}, out.data,
                                          out.fired, c.b.getI64IntegerAttr(1)),
                 p);
      }
  }
}

// Settle each scattered memory's elements from every store recorded against it:
// per element the datum is a one-hot select over the stores that reach it (at
// most one is live per cycle, since two stores to one element are ordered by
// the dependence analysis), and the write-enable the OR of their decoded
// pulses. An argument's cells are the caller's and this drives its element
// ports; an internal array's are this module's and this builds them, one
// enabled register per element.
void DatapathEmitter::finalizeScatteredPorts() {
  for (const uarch::MemUnit &m : dp.mems) {
    if (!m.scattered)
      continue;
    ArrayRef<SinkArm> writes;
    if (auto it = scatterWrites.find(m.id); it != scatterWrites.end())
      writes = it->second;
    // One narrow decode per store, shared by every element: the alternative,
    // a compare per (store, element) pair at the index's carried width, was
    // the scatter's whole LUT bill.
    SmallVector<SmallVector<Value>> hot;
    for (const SinkArm &w : writes) {
      hot.push_back(oneHotDecode(c, w.addr, m.depthWords));
      c.muxLedger.add(MuxRole::Crossbar, m.depthWords,
                      llvm::Log2_32_Ceil(std::max(2u, m.depthWords)));
    }
    // Demuxed onto element k first, so the select is the pulse and not the
    // index: two stores in different regions can name element k at once (an
    // idle region's stale address register), and only the enabled one may
    // drive.
    auto driveOf = [&](unsigned k) {
      SmallVector<SinkArm, 1> at;
      for (auto [s, w] : llvm::enumerate(writes))
        at.push_back({c.andBits(w.fired, hot[s][k]), Value(), w.data});
      return commitSink(at, Idle::DontCare);
    };
    if (m.external) {
      if (writes.empty())
        continue; // read-only: the caller's cells arrive and never leave
      for (auto [k, p] : llvm::enumerate(m.elemPorts)) {
        SinkArm out = driveOf(k);
        pa.setOutput(p.out, out.data);
        pa.setOutput(p.we, out.fired);
      }
      continue;
    }
    // An element no store reaches holds its reset value for the whole run, so
    // it is that constant rather than a register.
    IntegerType elemTy = memElemType(m, c.b);
    for (auto [k, be] : llvm::enumerate(scatterElems[m.id])) {
      SinkArm out = driveOf(k);
      Value zero = c.konst(elemTy, 0);
      if (!out.fired) {
        be.setValue(zero);
        continue;
      }
      Value cell = c.enabledReg(out.data, out.fired, zero, RegRole::Storage);
      nameValue(cell, memElemName(dp, m, k));
      be.setValue(cell);
    }
  }
}

// Master each buffer from child \p cu's addr/data/we outputs (\p outs): a
// boundary argument passes straight through to the top port, an internal one
// reaches its storage the way the parent's own accesses do. One arm per
// `PortPlan`.
void DatapathEmitter::masterCallPorts(
    const uarch::CallUnit &cu, llvm::StringMap<Value> &outs,
    llvm::StringMap<circt::Backedge> &rdBackedge,
    llvm::function_ref<Value()> runWindow, const StallShell &sh) {
  for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
    if (ma.isBoundary) {
      // The child's drive is one arm of its colour's boundary group, so a
      // holder it provably never issues with (another child, or a region's own
      // accesses) shares the bus, selected on the run window. Concurrent
      // masters carry distinct colours and keep distinct groups.
      if (ma.isWrite) {
        boundaryWrites[ma.topBase].push_back(
            {outs[ma.we], outs[ma.addr], outs[ma.data]});
      } else {
        Value fired;
        if (sharedInternalPort(dp.mems[ma.mem], ma.bank, ma.port))
          fired = runWindow();
        boundaryReads[ma.topBase].push_back({fired, outs[ma.addr], Value()});
      }
      continue;
    }
    const uarch::MemUnit &m = dp.mems[ma.mem];
    switch (ma.plan) {
    case PortPlan::ElementWise: {
      // A scattered array holds no addressable port, so the child's addressed
      // one is served off the element registers: a select for its read, a term
      // per store for its write. The child keeps the ordinary port ABI.
      assert(ma.bank == 0 && "a scattered array is one bank, so a child "
                             "masters it in whole-array element space");
      Value idx = addrAt(c.b, c.loc, outs[ma.addr], kDatapathAddressWidth);
      if (ma.isWrite)
        scatterWrites[m.id].push_back({outs[ma.we], idx, outs[ma.data]});
      else
        rdBackedge[ma.data].setValue(readCrossbar(c, scatterValues(m.id), idx));
      break;
    }
    case PortPlan::Table: {
      // A constant table the child only reads: one `hw.array_get` registered
      // to the latency the child was timed against, so the datum lands where
      // a RAM's would.
      Value elem = c.R(hw::ArrayGetOp::create(c.b, c.loc, romArray[m.id],
                                              memAddr(m, outs[ma.addr])));
      rdBackedge[ma.data].setValue(atReadData(m, elem, sh));
      break;
    }
    case PortPlan::Coloured: {
      // One hlmem per bank: the child masters bank `ma.bank`, already indexed
      // in that bank's own space via `allo.part`, so this routes straight to it
      // with no crossbar.
      assert(ma.bank < m.numBanks &&
             "child bank index exceeds the buffer's bank count; "
             "validateDatapath must have rejected the partition mismatch");
      Value addr = memAddr(m, outs[ma.addr]);
      // The child was compiled against this buffer's device latency, read here
      // from the MemUnit since the parent never accesses the buffer itself. A
      // deeper write pipelines into the fixed 1-cycle port, as in emitWrites.
      if (ma.isWrite) {
        unsigned pre = m.writeLatency - 1;
        Value a = c.shiftChain(addr, pre, sh).last();
        Value d = c.shiftChain(outs[ma.data], pre, sh).last();
        Value w = c.delayValid(outs[ma.we], pre, sh);
        // The binding settles a call's write port too, so two ports of one
        // child that declared them independent land in separate `always`
        // blocks and the array still infers a true dual port.
        sharedWrites[m.id].push_back({ma.bank, ma.port, {w, a, d}});
        break;
      }
      // The port may also be held by a sibling call or by the parent's own
      // accesses, so the datum comes off the one `seq.read` they share and the
      // address joins its arms. A child paces itself, so it brings no read
      // enable; as an owner it keeps the port unfrozen.
      rdBackedge[ma.data].setValue(sharedReadPort(m, ma.bank, ma.port));
      Value fired;
      if (sharedInternalPort(m, ma.bank, ma.port))
        fired = runWindow();
      SharedReadPort &p = sharedReads[{m.id, ma.bank, ma.port}];
      p.arms.push_back({fired, addr, Value()});
      ++p.owners;
      break;
    }

    case PortPlan::Lane:
      llvm_unreachable("a child masters a port on a skewed array; a lane is "
                       "assigned from this module's own accesses and the "
                       "child holds none. `checkEmitterSubset` refuses it");
    case PortPlan::Crossbar:
      llvm_unreachable("a child masters one bank, indexed in that bank's own "
                       "space, so it never crossbars");
    }
  }
}

} // namespace mlir::allo::uarch
