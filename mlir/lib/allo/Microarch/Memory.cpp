/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The memory subsystem's MODEL half: one `MemUnit` per array, what each access
// reaches it by (`PortPlan`), which port of which bank it drives, what this
// module builds to hold it (`MemUnit::Realization`), and the boundary port
// groups an argument publishes. How those decisions become hardware is
// MemoryEmitter.cpp.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h"    // iface::ModuleInterface (child ports)
#include "allo/Microarch/Naming.h"       // uniqueOwnerOf, memBase, elemBase
#include "allo/Scheduling/MemoryModel.h" // characterize (storage shape)
#include "allo/Scheduling/OperatorLibrary.h" // DeviceModel (the storage rows)
#include "allo/Support/AliasAnalysis.h"      // resolveRoot (storage identity)
#include "allo/Support/Logging.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

MemId DatapathBuilder::memIdOf(Value memref) {
  // Key on the storage root, not the operand as written, so a buffer threaded
  // out of a region is the SAME memory to its producer and its consumer.
  auto it = memOf.find(resolveRoot(memref));
  assert(it != memOf.end() &&
         "`collectStorageFacts` builds a MemUnit for every array the function "
         "touches, so a lookup here cannot miss");
  return it->second;
}

// Build the MemUnit for \p memref, or hand back the one it already has.
static MemId createMem(Datapath &dp, llvm::DenseMap<Value, MemId> &memOf,
                       const DeviceModel &dev, Value memref) {
  memref = resolveRoot(memref);
  if (auto it = memOf.find(memref); it != memOf.end())
    return it->second;
  MemId id = dp.mems.size();
  MemUnit m;
  m.id = id;
  m.memref = memref;
  m.external = isa<BlockArgument>(memref);
  auto mt = cast<MemRefType>(memref.getType());
  m.width = mt.getElementTypeBitWidth();
  // Banking / ports from the same storage model the scheduler binds against
  // (allo.part / allo.bind.storage): ONE characterization, so the ports billed
  // and the ports built cannot disagree.
  MemoryChar mc = allo::characterize(memref, dev.memory);
  // The power-on contents, when the array reads through an initialized global.
  // Whether that makes it a constant TABLE is a property of the use, settled by
  // `collectStorageFacts` once every writer is in view.
  if (auto init = allo::globalInitOf(memref))
    m.romInit = *init;
  m.layout = mc.layout;
  m.numBanks = m.layout.numBanks;
  // THE expression behind `scattered` (see its declaration for where the cells
  // live): the ROW says whether the array is held one cell per element, which a
  // complete partition is one way of reaching and several writing accessors
  // another. A callee's array argument is the one place neither changes
  // anything: the storage is the parent's and the child masters an addressed
  // port on it.
  m.scattered = dev.memory.isScatter(mc.storage) && (!m.external || dp.atTop);
  m.storage = mc.storage;
  // Everything the device states about the resolved realization, from the same
  // rows the scheduler timed this memref's accesses against. The emitter builds
  // ports at these latencies; do not re-derive from the name.
  const StorageRealization *sr = dev.memory.row(m.storage);
  assert(sr && "`PreVerification` rejects an array whose storage realization "
               "the device does not declare");
  m.readLatency = sr->timing.latency.read;
  m.writeLatency = sr->timing.latency.write;
  m.ramStyle = sr->ramStyle;
  // Port budget from that same characterization, so the ports the scheduler
  // reserved and the ports `bindMemoryPorts` assigns are one number.
  m.ports = mc.ports;
  assert(mt.hasStaticShape() &&
         "datapath memory requires a static shape (a dynamic memref sizes to "
         "depthWords 0)");
  // Per-bank depth from the same element-space decomposition the emitter's
  // crossbar and the host-side layout use, so a bank's address space is exactly
  // the elements it holds (`ceil` per partitioned dim, not of the total).
  m.depthWords = static_cast<unsigned>(m.layout.bankWords());
  dp.mems.push_back(std::move(m));
  memOf[memref] = id;
  return id;
}

void DatapathBuilder::collectStorageFacts(ArrayRef<Operation *> regionOps) {
  // Whether anything writes each array, indexed by MemId. This is the whole
  // reason the sweep exists: read-only is a property of the USE, and an array's
  // uses are in view only once every region body and every callee interface has
  // been looked at. Deciding it per access, as it is first touched, can only
  // answer conservatively.
  llvm::SmallVector<bool> written;
  auto touch = [&](Value memref, bool isWrite) {
    MemId id = createMem(dp, memOf, dev, memref);
    written.resize(dp.mems.size(), false);
    written[id] = written[id] || isWrite;
  };
  for (Operation *regionOp : regionOps)
    forEachBodyOp(regionOp, [&](Operation *op) {
      if (Value memref = dcpMemref(op)) {
        touch(memref, isa<dcp::DCPathStoreOp>(op));
        return;
      }
      auto inv = dyn_cast<dcp::DCPathInstanceOp>(op);
      if (!inv)
        return;
      // A child's array operand, and the direction of every port it masters on
      // it. The callee interface is registered before this caller is built, so
      // a child that only reads leaves the array a table.
      assert(callees && "a dcp.instance in a leaf datapath needs callee "
                        "context (a rerouted container)");
      auto it = callees->ifaces.find(inv.getCallee());
      assert(it != callees->ifaces.end() &&
             "the callee interface must be registered (emitted bottom-up)");
      for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        bool isWrite = false;
        for (const iface::Memory *p : it->second.portsForArg(int(k)))
          isWrite |= p->write;
        touch(operand, isWrite);
      }
    });

  // An initialized array nothing writes is a combinational constant table. An
  // ARGUMENT never is: its cells are the caller's, and a block argument reads
  // through no global, so it carries no contents to begin with.
  for (MemUnit &m : dp.mems) {
    assert(!(m.romInit && m.external) &&
           "an argument array reads through no initialized global");
    m.isRom = m.romInit && !m.external && !written[m.id];
  }
}

// Plan \p m: which port of its bank each access would drive, and how many ports
// one bank would therefore be built with.
//
// Two accesses share a port only where `portGraph` has no edge between them,
// which proves they never issue in the same cycle, so the port carries a select
// over them rather than an arbiter. Two shapes take a port of their own: an
// access `contendsWithAll` relates to every other, and, on the write side,
// every write of an array whose splitting is not proven safe.
std::optional<DatapathBuilder::PortAssignment>
DatapathBuilder::planPorts(const MemUnit &m, std::optional<bool> writes,
                           unsigned base) {
  Datapath::PortRelation rel = dp.portGraph(m.id, writes);
  ArrayRef<Datapath::PortVertex> verts = rel.verts;
  unsigned n = rel.size();
  llvm::SmallVector<unsigned> colour(n, 0);
  unsigned used = 0;

  // What a pair on one port shares its address by decides how far apart they
  // may sit, and every pair the relation leaves unrelated may share. Two writes
  // need nothing, an address being a don't-care in any cycle its enable is low.
  // A read and a write share by the write's own enable, likewise a signal in
  // its own right. Two reads share by a select over their activation pulses,
  // and where they belong to different ACCESSORS the emitter adds a second
  // select over which of them is driving: a region's accesses presenting, or a
  // child's run window.
  //
  // Two shapes share with nothing. An access with no bank of its own is routed
  // to every bank by a crossbar, so it reaches whatever any other reaches. And
  // a read with no pulse of its own cannot be selected between: a container's
  // own reads form its condition cone, live on every cycle its children run,
  // and a guard sequences its arms rather than running a datapath at all.
  auto contendsWithAll = [&](unsigned i) {
    if (verts[i].bank < 0)
      return true;
    if (verts[i].write || verts[i].call >= 0)
      return false;
    RegionBlock::Shape s = dp.regions[m.accesses[verts[i].access].region].shape;
    return s == RegionBlock::Shape::Container || s == RegionBlock::Shape::Guard;
  };
  for (unsigned i = 0; i < n; ++i)
    if (contendsWithAll(i))
      for (unsigned j = 0; j < n; ++j)
        if (j != i)
          rel.link(i, j);
  // Greedy in vertex order, taking the lowest port no neighbour holds.
  for (unsigned i = 0; i < n; ++i) {
    llvm::BitVector taken(n);
    for (unsigned j = 0; j < i; ++j)
      if (rel.adj[i].test(j))
        taken.set(colour[j]);
    colour[i] = taken.find_first_unset();
    used = std::max(used, colour[i] + 1);
  }
  // Greedy first fit bounds the ports, it does not minimize them. Two read
  // ports with no edge across them are one port carrying both selects, which is
  // one address bus fewer and, past what one instance serves, one copy of the
  // array fewer. Write ports stay as they are: dropping below two clears
  // `writesIndependent`, which puts every write in one `always` block and
  // infers no RAM at all.
  llvm::SmallVector<llvm::BitVector> members(used, llvm::BitVector(n));
  llvm::SmallVector<llvm::BitVector> nbrs(used, llvm::BitVector(n));
  llvm::BitVector reads(used, true);
  for (unsigned i = 0; i < n; ++i) {
    members[colour[i]].set(i);
    nbrs[colour[i]] |= rel.adj[i];
    if (verts[i].write)
      reads.reset(colour[i]);
  }
  // Into the lowest port that will take it. A port merged away is never a
  // target afterwards, so one pass over the ports in order is a fixed point.
  for (unsigned b = 1; b < used; ++b)
    if (reads[b])
      for (unsigned a = 0; a < b; ++a)
        if (reads[a] && members[a].any() && !nbrs[a].anyCommon(members[b])) {
          members[a] |= members[b];
          nbrs[a] |= nbrs[b];
          members[b].reset();
          break;
        }
  used = 0;
  for (llvm::BitVector &group : members)
    if (group.any()) {
      for (unsigned i : group.set_bits())
        colour[i] = used;
      ++used;
    }

  // Whether the writes may go on separate ports, which are separate `always`
  // blocks with nothing between them to resolve a collision. Only a pair proven
  // to address different words may: two accesses of one region, which a memory
  // dependence made the scheduler separate, and two write ports of one child
  // that declared them independent. Two different children, or a child and a
  // local access, are related by nothing.
  bool split = true;
  auto proven = [&](unsigned i, unsigned j) {
    if (verts[i].call < 0 && verts[j].call < 0)
      return m.accesses[verts[i].access].region ==
             m.accesses[verts[j].access].region;
    return verts[i].call >= 0 && verts[i].call == verts[j].call &&
           verts[i].independent;
  };
  for (unsigned i = 0; split && used > 1 && i < n; ++i)
    for (unsigned j = i + 1; j < n; ++j)
      if (verts[i].write && verts[j].write && rel.adj[i].test(j) &&
          !proven(i, j)) {
        split = false;
        break;
      }
  // An unsplittable set of writes stays on one `always` block, which arbitrates
  // the collision it might have. Each still keeps a port of its own so the
  // block holds one assignment per write: two writes to different words in one
  // cycle must both commit, and a select would drop one.
  //
  // That block is per direction, so a both-directions pass cannot express it
  // and declines; a binding giving every access its own port is never fewer
  // ports than the per-direction one it falls back to.
  if (!split) {
    if (!writes)
      return std::nullopt;
    for (unsigned i = 0; i < n; ++i)
      colour[i] = i;
    used = n;
  }
  PortAssignment out;
  out.writes = writes;
  // Only a colouring that included the writes has anything to say about them.
  if (!writes || *writes) {
    llvm::SmallDenseSet<unsigned> writeColours;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].write)
        writeColours.insert(colour[i]);
    out.writesIndependent = split && writeColours.size() > 1;
  }

  // Ports one bank is built with: a bank is its own `seq.hlmem` and only the
  // accesses reaching it take its ports.
  out.counts.colours = used;
  for (unsigned k = 0; k < m.numBanks; ++k) {
    llvm::SmallDenseSet<unsigned> all, rd, wr;
    for (unsigned i = 0; i < n; ++i)
      if (verts[i].bank < 0 || verts[i].bank == int(k)) {
        all.insert(colour[i]);
        (verts[i].write ? wr : rd).insert(colour[i]);
      }
    out.counts.total = std::max<unsigned>(out.counts.total, all.size());
    out.counts.reads = std::max<unsigned>(out.counts.reads, rd.size());
    out.counts.writes = std::max<unsigned>(out.counts.writes, wr.size());
  }
  for (unsigned c : colour)
    out.colour.push_back(base + c);
  return out;
}

void DatapathBuilder::commitPorts(MemUnit &m, const PortAssignment &pa) {
  // The vertex order `portGraph` builds: writes before reads, and within each
  // this function's accesses before the ports its children master.
  unsigned v = 0;
  for (bool dir : {true, false}) {
    if (pa.writes && *pa.writes != dir)
      continue;
    for (MemUnit::Access &acc : m.accesses)
      if (acc.isWrite == dir)
        acc.port = pa.colour[v++];
    for (CallUnit &cu : dp.calls)
      for (CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && ma.isWrite == dir)
          ma.port = pa.colour[v++];
  }
  assert(v == pa.colour.size() &&
         "the port binding walks `portGraph`'s vertex order");
  if (!pa.writes || *pa.writes)
    m.writesIndependent = pa.writesIndependent;
}

// Group a skewed memory's accesses into LANES: within a lane the slots are
// distinct, so the accesses reach distinct banks and share one port on each.
// Same-slot accesses always collide, so each takes the next lane, the port the
// model billed it. Numbered per region and reads apart from writes, the
// granularity a port is contended at.
void DatapathBuilder::assignLanes() {
  for (MemUnit &m : dp.mems) {
    // A constant table has no ports to share (it is combinational), and an
    // argument's ports are boundary interfaces the manifest already published,
    // one set per access, which is why `assign-banks` assigns it no slot
    // either.
    if (!m.layout.skew() || m.external || m.isRom)
      continue;
    // One access without a slot and the array is back to crossbarring: a lane
    // shares a port on the strength of every user holding a distinct slot.
    if (llvm::any_of(m.accesses,
                     [](const MemUnit::Access &a) { return !a.staticBank; }))
      continue;
    m.skewed = true;
    llvm::DenseMap<std::tuple<unsigned, unsigned, unsigned>, unsigned> used;
    for (MemUnit::Access &acc : m.accesses) {
      assert(*acc.staticBank < m.numBanks && "a slot indexes the skew's banks");
      acc.lane = used[{acc.region, acc.isWrite, *acc.staticBank}]++;
    }
  }
}

void DatapathBuilder::planAccessPorts() {
  // What the STORAGE or the layout decides, which every access of the array
  // then takes; empty where the access's own bank decides it.
  auto uniform = [](const MemUnit &m) -> std::optional<PortPlan> {
    if (m.isRom)
      return PortPlan::Table;
    if (m.scattered)
      return PortPlan::ElementWise;
    if (m.skewed)
      return PortPlan::Lane;
    return std::nullopt;
  };
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses)
      acc.plan = uniform(m).value_or(acc.staticBank ? PortPlan::Coloured
                                                    : PortPlan::Crossbar);
  for (CallUnit &cu : dp.calls)
    for (CallUnit::MemArg &ma : cu.memArgs)
      ma.plan = uniform(dp.mems[ma.mem]).value_or(PortPlan::Coloured);
}

void DatapathBuilder::bindMemoryPorts() {
  for (MemUnit &m : dp.mems) {
    // Neither is addressed, so neither has a port to contend for: a scattered
    // array is one cell per element and a constant table is combinational.
    if (m.scattered || m.isRom)
      continue;
    // A skew answers this at the same granularity: a lane's accesses hold
    // distinct slots, so they reach distinct banks and share one port on each.
    if (m.skewed) {
      llvm::SmallDenseSet<unsigned> lanes[2];
      for (MemUnit::Access &acc : m.accesses) {
        acc.port = acc.lane;
        lanes[acc.isWrite].insert(acc.lane);
      }
      m.readPortsBuilt = lanes[0].size();
      m.writePortsBuilt = lanes[1].size();
      m.portsBuilt = m.readPortsBuilt + m.writePortsBuilt;
      continue;
    }
    // A direction at a time, reads numbered past the writes so no port carries
    // both. On a row whose directions are separate structures, merging them
    // buys an address multiplexer and nothing else.
    PortAssignment w = planPorts(m, /*writes=*/true, /*base=*/0).value();
    PortAssignment r =
        planPorts(m, /*writes=*/false, /*base=*/w.counts.colours).value();
    unsigned separateTotal = w.counts.writes + r.counts.reads;
    // Where the row's ports are a pool, each serving either direction, a read
    // may ride a write's port and one address bus carries both. Worth the
    // multiplexer only where the array does not otherwise fit, and possible
    // only where the writes were split, an unsplittable set already being one
    // `always` block.
    std::optional<PortAssignment> pooled;
    if (m.ports.instPool && !m.external &&
        !m.fitsStorage(w.counts.writes, separateTotal) &&
        (w.counts.writes <= 1 || w.writesIndependent)) {
      // The shared bus carries the write's address on the cycle it commits, so
      // a write that presents its terms early would drive the read's cycle too.
      assert(m.writeLatency == 1 &&
             "a pooled row's write port realizes in one cycle");
      // A pooled binding that saves no port is the multiplexer for nothing.
      pooled = planPorts(m, /*writes=*/std::nullopt, /*base=*/0);
      if (pooled && pooled->counts.total >= separateTotal)
        pooled.reset();
    }
    if (pooled) {
      commitPorts(m, *pooled);
      m.readPortsBuilt = pooled->counts.reads;
      m.writePortsBuilt = pooled->counts.writes;
      m.portsBuilt = pooled->counts.total;
      continue;
    }
    commitPorts(m, w);
    commitPorts(m, r);
    m.writePortsBuilt = w.counts.writes;
    m.readPortsBuilt = r.counts.reads;
    m.portsBuilt = separateTotal;
  }
  // Instances of its row each bank is held in, decided here because the bound
  // ports are what it follows from. A skew binds its ports by lane and leaves
  // the loop early, so this runs over every memory rather than inside.
  for (MemUnit &m : dp.mems) {
    // Off `instReads`, what ONE instance serves, not off the array's own
    // allowance, which is already a multiple of it. Not bounded by the copies
    // budget either: the budget is what a CYCLE may issue, and a binding that
    // needs more address buses than that still builds one copy per bus.
    unsigned per = m.ports.instReads.value_or(0);
    // A bus carrying a write is on every copy, so a read riding one is served
    // wherever it lands and costs no port of its own.
    unsigned riding = 0;
    if (m.ports.instPool) {
      // A pooled port serves either direction, and the binding may already have
      // ridden a write on a read's where the two never issue together, so what
      // the bank was BUILT with is the question and not the two directions
      // separately. Within one instance's pool, one instance holds it.
      if (m.portsBuilt <= *m.ports.instPool)
        continue;
      riding = m.readPortsBuilt + m.writePortsBuilt - m.portsBuilt;
      // Past it every copy takes every write and the reads share what is left,
      // so a written block RAM serves one read a copy, which is what the part
      // does: 1024x32 measures one tile at one read and two at two. Nothing
      // left is an array this row cannot hold at all.
      per = std::min(per, *m.ports.instPool > m.writePortsBuilt
                              ? *m.ports.instPool - m.writePortsBuilt
                              : 0u);
    }
    unsigned own = m.readPortsBuilt - riding;
    if (!per || own <= per)
      continue;
    m.instances = (own + per - 1) / per;
    // Each bank ranks the read ports that reach it and hands them out a whole
    // instance at a time. Per bank, not over the memory: `readPortsBuilt` is
    // the largest any one bank holds, so ranking every colour together would
    // run past it wherever two banks hold different ones and would put more
    // reads on an instance than it has. A read on a write's bus goes to the
    // first instance, where the port it rides already exists.
    llvm::SmallDenseSet<unsigned> writePorts;
    for (const MemUnit::Access &acc : m.accesses)
      if (acc.isWrite)
        writePorts.insert(acc.port);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && ma.isWrite)
          writePorts.insert(ma.port);
    llvm::SmallVector<llvm::SmallVector<unsigned>> byBank(m.numBanks);
    auto reaches = [&](std::optional<unsigned> bank, unsigned port) {
      // A skew's `staticBank` is a slot, not a bank, and its accesses read
      // every bank through the crossbar, as an unassigned access does.
      if (bank && !m.skewed)
        byBank[*bank].push_back(port);
      else
        for (auto &ports : byBank)
          ports.push_back(port);
    };
    for (const MemUnit::Access &acc : m.accesses)
      if (!acc.isWrite)
        reaches(acc.staticBank, acc.port);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id && !ma.isWrite)
          reaches(ma.bank, ma.port);
    for (auto [k, ports] : llvm::enumerate(byBank)) {
      llvm::sort(ports);
      ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
      unsigned rank = 0;
      for (unsigned port : ports) {
        unsigned inst = writePorts.contains(port) ? 0 : rank++ / per;
        assert(inst < m.instances && "a read ranked past the instances");
        m.readInstance[MemUnit::instanceKey(k, port)] = inst;
      }
    }
  }
}

void DatapathBuilder::measurePorts() {
  for (MemUnit &m : dp.mems) {
    // Neither is addressed, so neither contends for a port and the comparison
    // has nothing to compare: the same pair `bindMemoryPorts` skips.
    if (!m.scattered && !m.isRom) {
      m.readConcurrency = dp.portConcurrency(m.id, /*writes=*/false);
      m.writeConcurrency = dp.portConcurrency(m.id, /*writes=*/true);
    }
    // A scattered argument publishes its cells rather than an address bus, so
    // its groups are the elements; every other array publishes one per bound
    // port, plus one for each group a child masters on it.
    m.boundaryPorts = m.elemPorts.size();
    for (AccRef r : dp.readPorts)
      m.boundaryPorts += r.id == m.id;
    for (AccRef r : dp.writePorts)
      m.boundaryPorts += r.id == m.id;
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        m.boundaryPorts += ma.mem == m.id && ma.isBoundary;

    // The copies the scheduler priced the array at are what it reserved its
    // read bandwidth against, so a binding taking more of them has bought
    // bandwidth no cycle was cut for. Nothing here can refuse it, the schedule
    // being already fixed, so it is reported rather than dropped. The
    // concurrency beside it says which of the two is at fault: equal to the
    // ports, the schedule really does ask for them all at once and the array
    // wants partitioning or a wider row; below them, the binding separated
    // accesses that never meet.
    if (m.instances > m.ports.copies())
      logging::log(Level::Warn, Stage::Emit, m.memref.getLoc())
          << ownerOfMem(m.id) << ": " << m.readPortsBuilt << " read ports on "
          << m.storage << " take " << m.instances
          << " copies of it per bank, past the " << m.ports.copies()
          << " the schedule reserved (" << m.readConcurrency
          << " of its reads may issue in one cycle)";

    // One boundary group is one interface the CALLER has to build, and the
    // ports bound above are all this module can drive at once on a bank. Past
    // them the caller backs bandwidth nothing here asks for. Only an ADDRESSED
    // argument has a budget at all: an internal array publishes no group, and a
    // scattered one publishes cells rather than buses, which is the whole of
    // what its realization buys. Reported and not refused, the interface being
    // the manifest the caller was already compiled against.
    unsigned budget = m.numBanks * m.portsBuilt;
    if (budget && m.boundaryPorts > budget)
      logging::log(Level::Warn, Stage::Emit, m.memref.getLoc())
          << ownerOfMem(m.id) << ": the caller provides " << m.boundaryPorts
          << " interface groups for this argument, "
          << (m.boundaryPorts - budget)
          << " past what this module can drive at once (" << m.portsBuilt
          << " ports per bank, " << m.numBanks
          << " banks). Every accessor takes a group of its own, so a "
             "sub-kernel reaching the array adds one whether or not its port "
             "already shares a bus with another's";
  }
}

std::string DatapathBuilder::ownerOfMem(MemId id) const {
  llvm::SmallVector<Value> memRefs;
  for (const MemUnit &m : dp.mems)
    memRefs.push_back(m.memref);
  return uniqueOwnerOf(dp.mems[id].memref, memRefs, memOwner(id));
}

void DatapathBuilder::enumerateBoundaryPorts() {
  auto key = [](MemId mem, bool write) {
    return (uint64_t(mem) << 1) | unsigned(write);
  };
  llvm::DenseMap<uint64_t, unsigned> group;

  for (MemUnit &m : dp.mems) {
    if (!m.external)
      continue;
    std::string owner = ownerOfMem(m.id);
    // A scattered argument's ports are per element, enumerated once for the
    // memory (not per access), since every access reads them all and selects.
    // Its accesses keep the default portIdx/portBase; nothing addresses them.
    if (m.scattered) {
      // The directions actually used decide the names: an argument used one way
      // takes the bare `A_k`, used both ways its two ports need telling apart.
      bool reads = false, writes = false;
      for (const MemUnit::Access &acc : m.accesses)
        (acc.isWrite ? writes : reads) = true;
      for (unsigned k = 0, e = m.depthWords; k < e; ++k) {
        MemUnit::ElemPort p;
        if (reads)
          p.in = elemBase(owner, k, writes ? ElemDir::In : ElemDir::Only);
        if (writes) {
          p.out = elemBase(owner, k, reads ? ElemDir::Out : ElemDir::Only);
          p.we = portWe(p.out);
        }
        m.elemPorts.push_back(std::move(p));
      }
      continue;
    }
    // One boundary port group per bound port: accesses that provably never
    // issue together share a port, and so share the interface the caller backs
    // the array with, driving it through a select on their own activation. A
    // group per access instead makes every caller provide that many interfaces
    // for bandwidth the schedule never asks for.
    //
    // Keyed by bank as well as port, since a port index is one per bank and two
    // accesses routed to different banks are different interfaces. A
    // data-dependent banked access spans every interface, and `bindMemoryPorts`
    // already gave it a port of its own. One map per direction, since the two
    // number their groups in their own port list.
    llvm::SmallDenseMap<std::pair<unsigned, unsigned>, unsigned> groupOfPort[2];
    for (auto [a, acc] : llvm::enumerate(m.accesses)) {
      auto &ports = acc.isWrite ? dp.writePorts : dp.readPorts;
      auto [it, isNew] = groupOfPort[acc.isWrite].try_emplace(
          {acc.staticBank.value_or(~0u), acc.port}, ports.size());
      if (!isNew) {
        acc.portIdx = it->second;
        acc.portBase = m.accesses[ports[acc.portIdx].idx].portBase;
        continue;
      }
      acc.portIdx = ports.size();
      acc.portBase =
          memBase(owner, acc.isWrite, group[key(m.id, acc.isWrite)]++);
      ports.push_back({m.id, unsigned(a)});
    }
  }
  // One port group per accessor of a (memory, role), concurrent and un-muxed: a
  // mux would time-share the port a second accessor exists to avoid.
  for (CallUnit &cu : dp.calls)
    for (CallUnit::MemArg &ma : cu.memArgs)
      if (ma.isBoundary)
        ma.topBase = memBase(ownerOfMem(ma.mem), ma.isWrite,
                             group[key(ma.mem, ma.isWrite)]++);
}

} // namespace mlir::allo::uarch
