/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h"

#include "allo-c/Schedule.h" // kPartitionAttr, kBindStorageAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloOps.h" // dcp::DCPathStoreOp (post-reification)
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (the access substrate)

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GlobalOp / GetGlobalOp
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::allo;

// The storage root an access operates on (views peeled), or null for a non-
// access. Arrays and streams are BOTH port-limited storage, an array by its
// memory ports and a stream by its handshake, so both belong to the model. A
// FIFO carries exactly one transfer per end per cycle; without a resource for
// it, several accesses to one channel are free to land on the SAME cycle, which
// the emitter can only reject (their token order would be lost).
static Value storageOf(Operation *op) {
  auto a = asMemAccess(op);
  return a ? a->root : Value();
}

// Look up attribute \p name on \p memRef's carrier: its defining op, else the
// function-argument attrs if it is a func argument.
template <typename AttrT>
static AttrT carrierAttr(Value memRef, StringRef name) {
  if (Operation *def = memRef.getDefiningOp())
    return def->getAttrOfType<AttrT>(name);
  if (auto barg = dyn_cast<BlockArgument>(memRef))
    if (auto func = dyn_cast<func::FuncOp>(barg.getOwner()->getParentOp()))
      return func.template getArgAttrOfType<AttrT>(barg.getArgNumber(), name);
  return {};
}

// The three orthogonal axes of an `allo.bind.storage` directive, mapped from
// its `type` string (port topology + RAM/ROM) and `impl` string (storage
// primitive). An ABSENT `type` is the dual-port RAM default and an absent
// `impl` is `Auto` (the array falls to the library's default implementation);
// an unrecognized one is a frontend/scheduler vocabulary mismatch, asserted.
namespace {
struct BindStorage {
  MemoryPortEnum port = MemoryPortEnum::TrueDualPort;
  MemoryKindEnum kind = MemoryKindEnum::RAM;
  MemoryImplEnum impl = MemoryImplEnum::Auto;
};
} // namespace

static BindStorage parseBindStorage(DictionaryAttr bind) {
  BindStorage bs;
  if (!bind)
    return bs;
  // Both vocabularies mirror a Python enum the frontend validates, so every
  // string reaching here is a known case; mapping through an optional makes
  // a drifted vocabulary a loud bug instead of a silent fall to the default.
  if (auto ty = bind.getAs<StringAttr>("type")) {
    auto t = ty.getValue();
    auto port =
        llvm::StringSwitch<std::optional<MemoryPortEnum>>(t)
            .Cases({"ram_1p", "rom_1p"}, MemoryPortEnum::SinglePort)
            .Cases({"ram_2p", "ram_s2p"}, MemoryPortEnum::SimpleDualPort)
            // 2 shared R/W ports; `fifo` is not a topology, but a stream is
            // never characterized through here so its mapping is immaterial.
            .Cases({"ram_t2p", "ram_1wnr", "rom_2p", "rom_np", "fifo"},
                   MemoryPortEnum::TrueDualPort)
            .Default(std::nullopt);
    assert(port && "unknown allo.bind.storage type= (the frontend's "
                   "BindStorageType vocabulary drifted from this switch)");
    bs.port = port.value_or(MemoryPortEnum::TrueDualPort);
    bs.kind = t.starts_with("rom") ? MemoryKindEnum::ROM : MemoryKindEnum::RAM;
  }
  if (auto im = bind.getAs<StringAttr>("impl")) {
    auto impl =
        llvm::StringSwitch<std::optional<MemoryImplEnum>>(im.getValue())
            .Case("bram", MemoryImplEnum::BRAM)
            .Case("uram", MemoryImplEnum::URAM)
            .Case("lutram", MemoryImplEnum::LUTRAM)
            .Case("register", MemoryImplEnum::Register)
            .Case("srl", MemoryImplEnum::LUTRAM) // shift-register: LUT-based
            .Default(std::nullopt);
    assert(impl && "unknown allo.bind.storage impl= (the frontend's "
                   "BindStorageImpl vocabulary drifted from this switch)");
    bs.impl = impl.value_or(MemoryImplEnum::Auto);
  }
  return bs;
}

// Concurrent ports of a topology (per bank).
static unsigned portCount(MemoryPortEnum p) {
  return p == MemoryPortEnum::SinglePort ? 1u : 2u;
}

std::optional<Attribute> mlir::allo::globalInitOf(Value memRef) {
  auto gg = memRef.getDefiningOp<memref::GetGlobalOp>();
  if (!gg)
    return std::nullopt;
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      gg, gg.getNameAttr());
  assert(global && "get_global references an undefined memref.global");
  if (auto init = global.getInitialValue())
    return *init;
  return std::nullopt;
}

bool mlir::allo::isConstantTable(Value memRef) {
  if (!globalInitOf(memRef))
    return false;
  // A write is an `affine`/`memref` store before reification and a `dcp.store`
  // after, so cover both. Handing the array to a sub-kernel also disqualifies
  // it, whichever way the child accesses it (see the header).
  return llvm::none_of(memRef.getUsers(), [](Operation *u) {
    if (isa<dcp::DCPathStoreOp, func::CallOp, dcp::DCPathInstanceOp>(u))
      return true;
    auto a = asMemAccess(u);
    return a && a->isWrite;
  });
}

// The storage implementation a memref resolves to: a complete partition
// scatters into registers (regardless of any bind.storage impl); else an
// explicit `bind.storage impl`; else the library's default (unbound on-chip
// arrays default to LUTRAM in Vitis). This is the memref's position on the
// implementation axis, the input to per-impl access timing.
static MemoryImplEnum resolveImpl(Value memRef, MemoryImplEnum defaultImpl) {
  if (partitionOf(memRef).unlimited)
    return MemoryImplEnum::Register;
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
  return bs.impl != MemoryImplEnum::Auto ? bs.impl : defaultImpl;
}

void MemoryBankModel::observe(Operation *op) {
  if (Value root = storageOf(op))
    byMemref[root].accesses.push_back(op);
}

void MemoryBankModel::finalize() {
  for (auto &entry : byMemref) {
    Value memRef = entry.first;
    MemInfo &info = entry.second;
    // A stream channel is a FIFO, not an array: one transfer per end per cycle,
    // no banking or storage-impl axis, and two independent ends, i.e. `splitRW`
    // at one port each. `partitionOf` below would cast its type to MemRefType.
    if (isa<StreamType>(memRef.getType())) {
      info.splitRW = true;
      info.readPorts = 1;
      info.writePorts = 1;
      continue;
    }
    // Port topology from `allo.bind.storage`. A SimpleDualPort (S2P) RAM has a
    // dedicated read and write port; every other topology shares its ports for
    // reads and writes. A ROM has no write port, so it uses shared ports.
    auto bs =
        parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
    bool constTable = isConstantTable(memRef);
    bool readOnly = bs.kind == MemoryKindEnum::ROM || constTable;
    if (bs.port == MemoryPortEnum::SimpleDualPort && !readOnly) {
      info.splitRW = true;
      info.readPorts = 1;
      info.writePorts = 1;
    } else {
      info.sharedPorts = portCount(bs.port);
    }
    auto part = partitionOf(memRef);
    // A constant table has no port to contend for (see `isConstantTable`); a
    // complete partition scattered the array into registers. Either way there
    // is nothing to bind against.
    info.unlimited = part.unlimited || constTable;
    info.partitionFactor = part.factor;
    info.cyclicAxes = std::move(part.cyclicAxes);
    bool hasBlock = part.hasBlock;
    // Per-bank refinement holds only when every access is statically banked on
    // every cyclic axis; block/complete/mixed or roaming access instead falls
    // back to the aggregate limit.
    if (info.unlimited || info.cyclicAxes.empty() || hasBlock)
      continue;
    info.perBank = llvm::all_of(info.accesses, [&](Operation *access) {
      return llvm::all_of(info.cyclicAxes, [&](const auto &ax) {
        return staticBank(access, ax.first, ax.second).has_value();
      });
    });
  }
}

std::optional<std::pair<std::string, unsigned>>
MemoryBankModel::resource(Operation *op) const {
  auto memRef = storageOf(op);
  if (!memRef)
    return std::nullopt;
  auto it = byMemref.find(memRef);
  if (it == byMemref.end())
    return std::nullopt;
  const MemInfo &info = it->second;
  std::string base = "mem_" + std::to_string(hash_value(memRef));
  if (info.unlimited)
    return std::make_pair(base + "_rsrc", 0u); // unlimited -> no binding

  // The pool this access draws from, and its ports per bank. Split (S2P) ->
  // dedicated read/write pools that never contend; shared -> one `_rw` pool for
  // both directions.
  auto a = asMemAccess(op);
  bool isWrite = a && a->isWrite;
  StringRef dir;
  unsigned portsPerBank;
  if (info.splitRW) {
    dir = isWrite ? "_w" : "_r";
    portsPerBank = isWrite ? info.writePorts : info.readPorts;
  } else {
    dir = "_rw";
    portsPerBank = info.sharedPorts;
  }

  if (info.perBank) {
    std::string key = base + "_bank";
    for (auto [dim, factor] : info.cyclicAxes)
      key += "_" + std::to_string(*staticBank(op, dim, factor));
    return std::make_pair(key + dir.str(), portsPerBank);
  }
  return std::make_pair(base + "_rsrc" + dir.str(),
                        portsPerBank * info.partitionFactor);
}

namespace mlir::allo {

int64_t BankLayout::bankWords() const {
  int64_t n = 1;
  for (int64_t e : bankShape)
    n *= e;
  return n;
}

// The banking a memref's `allo.part` implies, in element space: each block or
// cyclic axis splits its dimension into `factor` banks of
// `ceil(extent/factor)`, and the axes compose in mixed radix; a complete
// partition scatters into registers (no banks at all). See BankLayout for the
// single definition this implements.
BankLayout bankLayoutOf(Value memRef) {
  BankLayout l;
  auto mt = cast<MemRefType>(memRef.getType());
  ArrayRef<int64_t> shape = mt.getShape();
  l.bankShape.assign(shape.begin(), shape.end());
  auto part = carrierAttr<PartitionAttr>(memRef, kPartitionAttr);
  if (!part)
    return l;
  for (PartitionAxisAttr axis : part.getPartitions()) {
    // A complete partition scatters the array into registers: no banked
    // storage to describe, so drop any axis seen so far.
    if (axis.getKind() == PartitionKindEnum::CompletePartition) {
      l.axes.clear();
      l.bankShape.assign(shape.begin(), shape.end());
      l.numBanks = 1;
      l.registers = true;
      return l;
    }
    int64_t f = axis.getFactor();
    bool block = axis.getKind() == PartitionKindEnum::BlockPartition;
    auto addAxis = [&](unsigned d) {
      int64_t extent = (l.bankShape[d] + f - 1) / f;
      l.axes.push_back({d, f, block, extent});
      l.bankShape[d] = extent;
      l.numBanks *= static_cast<unsigned>(f);
    };
    // `dim == 0` partitions every dimension by this factor.
    if (axis.getDim() == 0)
      for (unsigned d = 0, e = mt.getRank(); d < e; ++d)
        addAxis(d);
    else
      addAxis(static_cast<unsigned>(axis.getDim() - 1));
  }
  return l;
}

// The coordinate expression of dimension \p k for an address map that is either
// in element space (one result per dimension) or already linearized by
// `dcp-flatten-memref` (one result), delinearized row-major in the latter case.
// Dimension 0 needs no `mod`: a linear index is always below the total size.
static AffineExpr coordExpr(AffineMap map, ArrayRef<int64_t> shape,
                            unsigned k) {
  unsigned rank = shape.size();
  if (map.getNumResults() == rank)
    return map.getResult(k);
  AffineExpr e = map.getResult(0);
  int64_t stride = 1;
  for (unsigned d = k + 1; d < rank; ++d)
    stride *= shape[d];
  if (stride != 1)
    e = e.floorDiv(stride);
  return k == 0 ? e : e % shape[k];
}

std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    ArrayRef<int64_t> shape) {
  if (!map)
    return std::nullopt;
  int64_t bank = 0;
  for (const BankLayout::Axis &a : layout.axes) {
    AffineExpr e = coordExpr(map, shape, a.dim);
    auto one = AffineMap::get(map.getNumDims(), map.getNumSymbols(), e,
                              map.getContext());
    std::optional<int64_t> digit;
    if (a.block) {
      // A block bank is `i floordiv extent`, which is fixed only when the
      // subscript itself is: unlike a cyclic residue, no coefficient condition
      // pins it (a varying `i` walks from one chunk into the next).
      if (auto cst = dyn_cast<AffineConstantExpr>(
              simplifyAffineExpr(e, map.getNumDims(), map.getNumSymbols())))
        digit = cst.getValue() / a.extent;
    } else {
      digit = staticBank(one, 0, a.factor);
    }
    if (!digit)
      return std::nullopt;
    bank = bank * a.factor + *digit;
  }
  return bank;
}

AffineMap linearizeAccessMap(AffineMap map, ArrayRef<int64_t> shape) {
  // Already linear (dcp-flatten-memref's form): the same discriminant
  // `coordExpr` reads, so the two directions agree on which form they see.
  unsigned rank = shape.size();
  if (map.getNumResults() != rank) {
    assert(map.getNumResults() == 1 &&
           "an address map is either in element space (one result per memref "
           "dimension) or already linearized (exactly one result)");
    return map;
  }
  // Row-major strides are a product of the TRAILING extents, so a dynamic
  // non-leading dim poisons every stride; shape[0] is never read, which is why
  // a leading dynamic dim is safe. A rank-0 memref (a `Stateful` scalar's
  // backing storage) has no trailing dim and no stride to poison.
  assert((shape.empty() ||
          llvm::none_of(shape.drop_front(),
                        [](int64_t d) { return ShapedType::isDynamic(d); })) &&
         "row-major linearization needs static non-leading memref dims");
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * shape[k + 1];
  AffineExpr lin = getAffineConstantExpr(0, map.getContext());
  for (unsigned k = 0; k < rank; ++k)
    lin = lin + map.getResult(k) * stride[k];
  lin = simplifyAffineExpr(lin, map.getNumDims(), map.getNumSymbols());
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), lin,
                        map.getContext());
}

// The aggregate projection of `bankLayoutOf`, for consumers that only need the
// bank count / kind rather than the full decomposition.
PartitionInfo partitionOf(Value memRef) {
  BankLayout l = bankLayoutOf(memRef);
  PartitionInfo p;
  p.unlimited = l.registers;
  p.factor = l.numBanks;
  for (const BankLayout::Axis &a : l.axes) {
    if (a.block)
      p.hasBlock = true;
    else
      p.cyclicAxes.push_back({a.dim, a.factor});
  }
  return p;
}

// The compile-time bank of \p op on a cyclic axis (0-based `dim`, `factor`), or
// nullopt when the subscript is not iteration-invariant modulo the factor (a
// roaming or non-affine access): a fixed bank needs every variable coefficient
// of the partition-dim subscript to vanish modulo the factor.
std::optional<int64_t> staticBank(AffineMap map, unsigned dim, int64_t factor) {
  if (!map || dim >= map.getNumResults())
    return std::nullopt;
  SmallVector<int64_t> flat;
  if (failed(getFlattenedAffineExpr(map.getResult(dim), map.getNumDims(),
                                    map.getNumSymbols(), &flat)))
    return std::nullopt;
  // flat = [dim/sym/local coeffs..., constant].
  for (unsigned i = 0, e = flat.size() - 1; i < e; ++i)
    if (flat[i] % factor != 0)
      return std::nullopt;
  return ((flat.back() % factor) + factor) % factor;
}

std::optional<int64_t> staticBank(Operation *op, unsigned dim, int64_t factor) {
  auto a = asMemAccess(op);
  return a ? staticBank(a->map, dim, factor) : std::nullopt;
}

} // namespace mlir::allo

//===----------------------------------------------------------------------===//
// Memory timing library
//===----------------------------------------------------------------------===//

MemKindTiming MemoryLibrary::timing(MemoryImplEnum impl) const {
  for (const MemPrimitive &p : primitives)
    if (p.impl == impl)
      return p.timing;
  // An undeclared primitive gets zero timing. Reaching here with a concrete
  // (non-Auto) impl means a storage kind the device declares no timing for
  // would be scheduled at latency 0 (a bug); only a stream (Auto) is zero.
  assert(impl == MemoryImplEnum::Auto &&
         "storage impl not declared by the device -> silent latency-0 access");
  static constexpr MemKindTiming zero;
  return zero;
}

MemoryImplEnum MemoryLibrary::resolvedImpl(Operation *op) const {
  auto a = asMemAccess(op);
  if (!a || a->kind == AccessKind::Stream)
    return MemoryImplEnum::Auto;
  return resolveImpl(a->root, defaultImpl);
}

bool MemoryLibrary::declares(MemoryImplEnum impl) const {
  return llvm::any_of(primitives,
                      [&](const MemPrimitive &p) { return p.impl == impl; });
}

MemoryLibrary::Timing MemoryLibrary::timing(Operation *op) const {
  auto a = asMemAccess(op);
  if (!a)
    return {};
  MemoryImplEnum impl = a->kind == AccessKind::Stream
                            ? MemoryImplEnum::Auto
                            : resolveImpl(a->root, defaultImpl);
  MemKindTiming t = a->kind == AccessKind::Stream ? fifo : timing(impl);
  return a->isWrite ? Timing{t.latency.write, t.delay.write, true, impl}
                    : Timing{t.latency.read, t.delay.read, true, impl};
}

MemoryChar allo::characterize(Value memref, MemoryImplEnum defaultImpl) {
  using namespace detail;
  MemoryChar c;
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memref, kBindStorageAttr));
  c.constantTable = isConstantTable(memref);
  c.readOnly = bs.kind == MemoryKindEnum::ROM || c.constantTable;
  c.portsPerBank = portCount(bs.port);
  auto part = partitionOf(memref);
  c.numBanks = part.factor;
  c.registers = part.unlimited;
  c.impl = resolveImpl(memref, defaultImpl);
  return c;
}
