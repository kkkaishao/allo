/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h"

#include "allo-c/Schedule.h" // kPartitionAttr, kBindStorageAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloOps.h" // dcp::DCPathStoreOp (post-reification)
#include "allo/Scheduling/AddressModel.h" // simplifiedForHardware
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (the access substrate)

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GlobalOp / GetGlobalOp
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::allo;

// The storage root an access operates on (views peeled), or null for a
// non-access. Arrays and streams are BOTH port-limited storage: an array by its
// memory ports, a stream by its handshake, a FIFO carrying exactly one transfer
// per end per cycle.
static Value storageOf(Operation *op) {
  auto a = asMemAccess(op);
  return a ? a->root : Value();
}

// Look up attribute \p name on \p memRef's carrier: its defining op, else the
// function-argument attrs if it is a func argument. A `memref.get_global` is a
// REFERENCE to storage, so its carrier is the `memref.global` that declares it,
// which is where the schedule primitives write.
template <typename AttrT>
static AttrT carrierAttr(Value memRef, StringRef name) {
  if (Operation *def = memRef.getDefiningOp()) {
    if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          get, get.getNameAttr());
      assert(global && "get_global references an undefined memref.global");
      return global->getAttrOfType<AttrT>(name);
    }
    return def->getAttrOfType<AttrT>(name);
  }
  // Asked of the `func.func` the scheduler works on and of the `dcp.module` it
  // closes into, so it keys on the interface rather than on either op.
  if (auto barg = dyn_cast<BlockArgument>(memRef))
    if (auto func =
            dyn_cast<FunctionOpInterface>(barg.getOwner()->getParentOp()))
      return llvm::dyn_cast_or_null<AttrT>(
          func.getArgAttr(barg.getArgNumber(), name));
  return {};
}

// The three orthogonal axes of an `allo.bind.storage` directive, mapped from
// its `type` string (port topology + RAM/ROM) and `impl` string (which storage
// realization). An ABSENT `type` is the dual-port RAM default and an absent
// `impl` is an absent CHOICE, resolved against the library's default.
namespace {
struct BindStorage {
  MemoryPortEnum port = MemoryPortEnum::TrueDualPort;
  MemoryKindEnum kind = MemoryKindEnum::RAM;
  StringRef storage; // empty: no explicit choice, not "no storage"
};
} // namespace

static BindStorage parseBindStorage(DictionaryAttr bind) {
  BindStorage bs;
  if (!bind)
    return bs;
  // Both vocabularies mirror a Python enum the frontend validates, so every
  // string reaching here is a known case; the optional makes a drifted
  // vocabulary a loud bug instead of a silent fall to the default.
  if (auto ty = bind.getAs<StringAttr>("type")) {
    auto t = ty.getValue();
    auto port =
        llvm::StringSwitch<std::optional<MemoryPortEnum>>(t)
            .Cases({"ram_1p", "rom_1p"}, MemoryPortEnum::SinglePort)
            .Cases({"ram_2p", "ram_s2p"}, MemoryPortEnum::SimpleDualPort)
            // 2 shared R/W ports. `fifo` is not a topology, but a stream is
            // never characterized through here.
            .Cases({"ram_t2p", "ram_1wnr", "rom_2p", "rom_np", "fifo"},
                   MemoryPortEnum::TrueDualPort)
            .Default(std::nullopt);
    assert(port && "unknown allo.bind.storage type= (the frontend's "
                   "BindStorageType vocabulary drifted from this switch)");
    bs.port = port.value_or(MemoryPortEnum::TrueDualPort);
    bs.kind = t.starts_with("rom") ? MemoryKindEnum::ROM : MemoryKindEnum::RAM;
  }
  // `impl` NAMES a `dcp.storage` of the device, so there is no table here to
  // drift: a name the device does not declare is reported by `PreVerification`
  // against the array, which is where the user can act on it.
  if (auto im = bind.getAs<StringAttr>("impl"))
    bs.storage = im.getValue();
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
  // after, so cover both. A sub-kernel call disqualifies the array whichever
  // way the child accesses it (see the header).
  return llvm::none_of(memRef.getUsers(), [](Operation *u) {
    if (isa<dcp::DCPathStoreOp, func::CallOp, dcp::DCPathInstanceOp>(u))
      return true;
    auto a = asMemAccess(u);
    return a && a->isWrite;
  });
}

// The name of the storage realization a memref resolves to, the input to
// per-realization access timing: a complete partition takes the device's
// `scatter` row whatever `bind.storage impl` says, since once every bank holds
// one word there is no addressed structure left; else an explicit
// `bind.storage impl`; else the device's `default` row. An empty result is a
// device that marks no `scatter`, which `PreVerification` reports.
static std::string resolveStorage(Value memRef, const MemoryLibrary &lib) {
  if (bankLayoutOf(memRef).registers)
    return lib.scatterStorage;
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
  return (bs.storage.empty() ? StringRef(lib.defaultStorage) : bs.storage)
      .str();
}

StringRef allo::boundStorageOf(Value memRef) {
  return parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr))
      .storage;
}

void MemoryBankModel::observe(Operation *op) {
  if (Value root = storageOf(op))
    byMemref.try_emplace(root);
}

void MemoryBankModel::finalize(const MemoryLibrary &lib) {
  for (auto &[root, info] : byMemref) {
    // A stream channel is a FIFO, not an array: one transfer per end per cycle,
    // no banking or storage-impl axis to characterize, two independent ends,
    // i.e. `splitRW` at one port each. `characterize` would cast its type to
    // MemRefType.
    if (isa<StreamType>(root.getType())) {
      info.splitRW = true;
      info.readPorts = 1;
      info.writePorts = 1;
      // Its default `layout` is the single unbanked one, and it resolves no
      // `storage` realization, which nothing here asks a stream for.
      continue;
    }
    info = characterize(root, lib);
  }
}

SmallVector<std::pair<std::string, unsigned>>
MemoryBankModel::resources(Operation *op) const {
  auto memRef = storageOf(op);
  if (!memRef)
    return {};
  auto it = byMemref.find(memRef);
  if (it == byMemref.end())
    return {};
  const MemoryChar &info = it->second;
  if (info.unlimited())
    return {};

  // The pool this access draws from, and its ports per bank. Split (S2P) gives
  // dedicated read/write pools that never contend, shared one `_rw` pool.
  auto a = asMemAccess(op);
  assert(a && "storageOf named a storage root, so this is a memory access");
  StringRef dir;
  unsigned portsPerBank;
  if (info.splitRW) {
    dir = a->isWrite ? "_w" : "_r";
    portsPerBank = a->isWrite ? info.writePorts : info.readPorts;
  } else {
    dir = "_rw";
    portsPerBank = info.sharedPorts;
  }
  std::string base = "mem_" + std::to_string(hash_value(memRef));

  // The banks this access occupies: its assigned bank alone, or every one of
  // them when it has none and reaches the emitter's crossbar. READ rather than
  // derived, so the ports billed and the routing built are one fact.
  unsigned numBanks = info.layout.numBanks;
  std::optional<unsigned> bank;
  if (numBanks > 1)
    bank = assignedBankOf(op);
  SmallVector<std::pair<std::string, unsigned>> ports;
  auto take = [&](unsigned k) {
    ports.emplace_back(base + "_b" + std::to_string(k) + dir.str(),
                       portsPerBank);
  };
  if (numBanks == 1 || bank)
    take(bank.value_or(0));
  else
    for (unsigned k = 0; k < numBanks; ++k)
      take(k);
  return ports;
}

namespace mlir::allo {

int64_t BankLayout::bankWords() const {
  int64_t n = 1;
  for (int64_t e : bankShape)
    n *= e;
  return n;
}

StringRef bankKindName(BankLayout::Kind kind) {
  switch (kind) {
  case BankLayout::Kind::Cyclic:
    return "cyclic";
  case BankLayout::Kind::Block:
    return "block";
  case BankLayout::Kind::Skew:
    return "skew";
  }
  llvm_unreachable("unhandled bank layout kind");
}

const BankLayout::Axis *BankLayout::skew() const {
  const Axis *found = nullptr;
  for (const Axis &a : axes)
    if (a.kind == Kind::Skew) {
      assert(!found && "a layout carries at most one skewed axis");
      found = &a;
    }
  return found;
}

// The banking a memref's `allo.part` implies, in element space: each block or
// cyclic axis splits its dimension into `factor` banks of
// `ceil(extent/factor)`, the axes composing in mixed radix; a complete
// partition scatters into registers. See BankLayout for the full definition.
BankLayout bankLayoutOf(Value memRef) {
  BankLayout l;
  auto mt = cast<MemRefType>(memRef.getType());
  ArrayRef<int64_t> shape = mt.getShape();
  l.bankShape.assign(shape.begin(), shape.end());
  auto part = carrierAttr<PartitionAttr>(memRef, kPartitionAttr);
  if (!part)
    return l;
  for (PartitionAxisAttr axis : part.getPartitions()) {
    // A complete partition leaves no banked storage to describe, so drop any
    // axis seen so far.
    if (axis.getKind() == PartitionKindEnum::CompletePartition) {
      l.axes.clear();
      l.bankShape.assign(shape.begin(), shape.end());
      l.numBanks = 1;
      l.registers = true;
      return l;
    }
    int64_t f = axis.getFactor();
    BankLayout::Kind kind = axis.getKind() == PartitionKindEnum::BlockPartition
                                ? BankLayout::Kind::Block
                            : axis.getKind() == PartitionKindEnum::SkewPartition
                                ? BankLayout::Kind::Skew
                                : BankLayout::Kind::Cyclic;
    auto addAxis = [&](unsigned d) {
      int64_t extent = (l.bankShape[d] + f - 1) / f;
      l.axes.push_back({d, f, kind, extent});
      l.bankShape[d] = extent;
      l.numBanks *= static_cast<unsigned>(f);
    };
    // `dim == 0` partitions every dimension by this factor (never a skew, whose
    // verifier requires a named distribution dimension).
    if (axis.getDim() == 0)
      for (unsigned d = 0, e = mt.getRank(); d < e; ++d)
        addAxis(d);
    else
      addAxis(static_cast<unsigned>(axis.getDim() - 1));
  }
  return l;
}

//===--------------------------------------------------------------------===//
// The partition lattice: the canonical spelling of a banking, and the coarsest
// banking that satisfies two of them.
//===--------------------------------------------------------------------===//

static bool isCompletePartition(PartitionAttr part) {
  return part && llvm::any_of(part.getPartitions(), [](PartitionAxisAttr a) {
           return a.getKind() == PartitionKindEnum::CompletePartition;
         });
}

static bool hasSkewAxis(PartitionAttr part) {
  return part && llvm::any_of(part.getPartitions(), [](PartitionAxisAttr a) {
           return a.getKind() == PartitionKindEnum::SkewPartition;
         });
}

// The whole-array top, spelled once. `bankLayoutOf` scatters into registers on
// ANY complete axis whatever dimension it names, so normalizing the dimension
// away is what lets two spellings of "registers" compare equal.
static PartitionAttr completePartition(MLIRContext *ctx) {
  return PartitionAttr::get(
      ctx, {PartitionAxisAttr::get(ctx, PartitionKindEnum::CompletePartition,
                                   /*factor=*/0, /*dim=*/0)});
}

PartitionAttr canonicalizePartition(PartitionAttr part, MemRefType type) {
  if (!part)
    return {};
  MLIRContext *ctx = part.getContext();
  if (isCompletePartition(part))
    return completePartition(ctx);
  // `dim == 0` means every dimension, which `bankLayoutOf` expands in
  // increasing dimension order; expanding it here lets an axis list be compared
  // one dimension at a time.
  SmallVector<PartitionAxisAttr, 4> axes;
  for (PartitionAxisAttr axis : part.getPartitions()) {
    if (axis.getDim() != 0) {
      axes.push_back(axis);
      continue;
    }
    for (int64_t d = 1, e = type.getRank(); d <= e; ++d)
      axes.push_back(
          PartitionAxisAttr::get(ctx, axis.getKind(), axis.getFactor(), d));
  }
  llvm::sort(axes, [](PartitionAxisAttr x, PartitionAxisAttr y) {
    return x.getDim() < y.getDim();
  });
  return PartitionAttr::get(ctx, axes);
}

// The single axis refining both \p x and \p y on the dimension they share, of
// static extent \p extent. A cyclic residue class modulo F is a union of the
// classes modulo kF, so a multiple of the factor always refines; a block chunk
// of `ceil(extent / F)` splits into finer chunks only where the division leaves
// no remainder, else a finer chunk straddles a coarser boundary.
static llvm::FailureOr<PartitionAxisAttr> joinAxis(PartitionAxisAttr x,
                                                   PartitionAxisAttr y,
                                                   int64_t extent,
                                                   std::string &why) {
  assert(x.getDim() == y.getDim() && "joining axes of different dimensions");
  assert(x.getKind() != PartitionKindEnum::SkewPartition &&
         y.getKind() != PartitionKindEnum::SkewPartition &&
         "a skew is handled whole-attribute, being its array's only axis");
  llvm::raw_string_ostream os(why);
  if (x.getKind() != y.getKind()) {
    os << "dimension " << x.getDim() << " is "
       << ConvertToPartitionString(x.getKind()) << "-partitioned on one side, "
       << ConvertToPartitionString(y.getKind())
       << " on the other; a chunked layout and an interleaved one place the "
          "same elements in different banks, so no single banking serves both. "
          "Give both sides the same kind, or partition the array with a Skew, "
          "which stays conflict-free along either axis";
    return failure();
  }
  int64_t lo = std::min(x.getFactor(), y.getFactor());
  int64_t hi = std::max(x.getFactor(), y.getFactor());
  if (hi % lo != 0) {
    os << "dimension " << x.getDim() << " is partitioned by " << lo
       << " on one side and by " << hi
       << " on the other; the factors must divide, so that the finer banking "
          "keeps apart everything the coarser one does";
    return failure();
  }
  if (x.getKind() == PartitionKindEnum::BlockPartition &&
      (ShapedType::isDynamic(extent) || extent % hi != 0)) {
    os << "dimension " << x.getDim() << " is block-partitioned by " << lo
       << " on one side and by " << hi << " on the other, but its extent ("
       << extent
       << ") is not a multiple of the larger factor, so the two chunkings cut "
          "the dimension at different points";
    return failure();
  }
  return PartitionAxisAttr::get(x.getContext(), x.getKind(), hi, x.getDim());
}

llvm::FailureOr<PartitionAttr> joinPartitions(PartitionAttr a, PartitionAttr b,
                                              MemRefType type,
                                              std::string &why) {
  a = canonicalizePartition(a, type);
  b = canonicalizePartition(b, type);
  if (!a || a == b)
    return b;
  if (!b)
    return a;
  MLIRContext *ctx = type.getContext();
  // A complete partition is the top: every element its own register, which
  // distinguishes every pair and so serves every consumer.
  if (isCompletePartition(a) || isCompletePartition(b))
    return completePartition(ctx);
  if (hasSkewAxis(a) || hasSkewAxis(b)) {
    llvm::raw_string_ostream(why)
        << "a skew partition must be an array's only axis (its bank already "
           "reads every subscript), so "
        << a << " and " << b << " cannot be combined";
    return failure();
  }
  // Axes on different dimensions compose in mixed radix; only a shared
  // dimension folds into one axis. The ordered map also puts the result in
  // canonical order.
  std::map<int64_t, PartitionAxisAttr> byDim;
  for (PartitionAxisAttr axis : a.getPartitions())
    byDim.emplace(axis.getDim(), axis);
  for (PartitionAxisAttr axis : b.getPartitions()) {
    auto [slot, fresh] = byDim.try_emplace(axis.getDim(), axis);
    if (fresh)
      continue;
    auto joined =
        joinAxis(slot->second, axis, type.getDimSize(axis.getDim() - 1), why);
    if (failed(joined))
      return failure();
    slot->second = *joined;
  }
  SmallVector<PartitionAxisAttr, 4> axes;
  for (auto &[dim, axis] : byDim)
    axes.push_back(axis);
  return PartitionAttr::get(ctx, axes);
}

// The linear form a skewed axis reads its bank digit from: every subscript,
// summed. A skew is the only axis of its layout (`PartitionAttr::verify`), so
// this sees the access's own coordinates rather than a partly peeled set, which
// lets `skewSlotOf` reproduce it from the map alone.
static AffineExpr skewSum(ArrayRef<AffineExpr> coord) {
  AffineExpr s = coord.front();
  for (AffineExpr c : coord.drop_front())
    s = s + c;
  return s;
}

BankSplitExpr bankSplitOf(const BankLayout &layout, AffineMap map,
                          ArrayRef<int64_t> shape) {
  assert(map && "bank split of an access with no address map");
  assert(map.getNumResults() == shape.size() &&
         "an address map is in element space, one result per memref dimension");
  // The per-bank strides below are products of the trailing extents, so a
  // dynamic non-leading dim poisons them.
  assert((shape.empty() ||
          llvm::none_of(shape.drop_front(),
                        [](int64_t d) { return ShapedType::isDynamic(d); })) &&
         "banked addressing needs static non-leading memref dims");
  unsigned rank = shape.size();
  unsigned nd = map.getNumDims(), ns = map.getNumSymbols();
  MLIRContext *ctx = map.getContext();

  SmallVector<AffineExpr> coord(map.getResults());

  // Peel each axis's digit off its own subscript, cyclic taking the residue and
  // block the quotient; a skew reads every subscript and divides only its
  // distribution dimension. The digits compose in mixed radix.
  AffineExpr bank;
  for (const BankLayout::Axis &a : layout.axes) {
    AffineExpr ci = coord[a.dim];
    AffineExpr digit;
    switch (a.kind) {
    case BankLayout::Kind::Block:
      digit = ci.floorDiv(a.extent);
      coord[a.dim] = ci % a.extent;
      break;
    case BankLayout::Kind::Cyclic:
      digit = ci % a.factor;
      coord[a.dim] = ci.floorDiv(a.factor);
      break;
    case BankLayout::Kind::Skew:
      digit = skewSum(coord) % a.factor;
      coord[a.dim] = ci.floorDiv(a.factor);
      break;
    }
    bank = bank ? bank * a.factor + digit : digit;
  }
  if (!bank)
    bank = getAffineConstantExpr(0, ctx); // unpartitioned: the one bank

  // What remains linearizes over the PER-BANK extents, the address space one
  // bank actually has.
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * layout.bankShape[k + 1];
  AffineExpr offset = getAffineConstantExpr(0, ctx);
  for (unsigned k = 0; k < rank; ++k) {
    offset = offset + coord[k] * stride[k];
    coord[k] = simplifiedForHardware(coord[k], nd, ns);
  }

  return {simplifiedForHardware(bank, nd, ns),
          simplifiedForHardware(offset, nd, ns), std::move(coord)};
}

// The interval \p e takes over \p ranges, or nullopt when an operand in it is
// unbounded. Endpoint arithmetic, exact since every operator is monotone in its
// argument; the one over-approximation is a residue whose argument straddles a
// multiple of the divisor, widened to the whole residue class. Widening is
// SOUND: a caller acts on `lo == hi`, so a wider interval only declines.
static std::optional<std::pair<int64_t, int64_t>>
rangeOf(AffineExpr e, ArrayRef<DimRange> ranges) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return std::pair{c.getValue(), c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e)) {
    if (d.getPosition() >= ranges.size() || !ranges[d.getPosition()].known)
      return std::nullopt;
    const DimRange &r = ranges[d.getPosition()];
    return std::pair{r.lo, r.hi};
  }
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return std::nullopt; // a symbol: loop-invariant, but not bounded here
  std::optional<std::pair<int64_t, int64_t>> l = rangeOf(bin.getLHS(), ranges),
                                             r = rangeOf(bin.getRHS(), ranges);
  if (!l || !r)
    return std::nullopt;
  if (bin.getKind() == AffineExprKind::Add)
    return std::pair{l->first + r->first, l->second + r->second};
  // Every other operator's right operand is a constant in a well-formed map.
  if (r->first != r->second)
    return std::nullopt;
  int64_t k = r->first;
  if (bin.getKind() == AffineExprKind::Mul)
    return k >= 0 ? std::pair{l->first * k, l->second * k}
                  : std::pair{l->second * k, l->first * k};
  if (k <= 0)
    return std::nullopt;
  int64_t qlo = llvm::divideFloorSigned(l->first, k),
          qhi = llvm::divideFloorSigned(l->second, k);
  if (bin.getKind() == AffineExprKind::FloorDiv)
    return std::pair{qlo, qhi};
  if (bin.getKind() == AffineExprKind::CeilDiv)
    return std::pair{llvm::divideCeilSigned(l->first, k),
                     llvm::divideCeilSigned(l->second, k)};
  assert(bin.getKind() == AffineExprKind::Mod && "unhandled affine operator");
  if (qlo != qhi)
    return std::pair{int64_t{0}, k - 1}; // straddles: the whole residue class
  return std::pair{l->first - qlo * k, l->second - qlo * k};
}

std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    ArrayRef<int64_t> shape,
                                    ArrayRef<DimRange> ranges) {
  if (!map)
    return std::nullopt;
  // "Statically banked" is the bank expression taking ONE value, which a
  // constant fold covers directly and a bounded iteration domain (a block
  // partition's digit) can cover too.
  AffineExpr bank = bankSplitOf(layout, map, shape).bank;
  if (auto cst = dyn_cast<AffineConstantExpr>(bank))
    return cst.getValue();
  if (std::optional<std::pair<int64_t, int64_t>> r = rangeOf(bank, ranges))
    if (r->first == r->second)
      return r->first;
  return std::nullopt;
}

std::optional<SkewSlot> skewSlotOf(const BankLayout &layout, AffineMap map,
                                   ArrayRef<int64_t> shape) {
  const BankLayout::Axis *ax = layout.skew();
  if (!map || !ax)
    return std::nullopt;
  assert(layout.axes.size() == 1 && "a skew is its layout's only axis");
  assert(map.getNumResults() == shape.size() &&
         "an address map is in element space, one result per memref dimension");
  unsigned nd = map.getNumDims(), ns = map.getNumSymbols();
  AffineExpr sum = skewSum(map.getResults());
  // The constant part is what the sum reads with every operand at zero, so the
  // rest is the runtime part. One substitution rather than a walk, which no
  // shape of affine sum can defeat.
  AffineExpr zero = getAffineConstantExpr(0, map.getContext());
  SmallVector<AffineExpr> zeroDims(nd, zero), zeroSyms(ns, zero);
  auto cst = dyn_cast<AffineConstantExpr>(
      simplifyAffineExpr(sum.replaceDimsAndSymbols(zeroDims, zeroSyms), 0, 0));
  if (!cst)
    return std::nullopt;
  int64_t f = ax->factor;
  return SkewSlot{simplifyAffineExpr(sum - cst.getValue(), nd, ns),
                  static_cast<unsigned>(((cst.getValue() % f) + f) % f)};
}

AffineMap linearizeAccessMap(AffineMap map, ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  assert(map.getNumResults() == rank &&
         "an address map is in element space, one result per memref dimension");
  // Row-major strides are a product of the TRAILING extents, so a dynamic
  // non-leading dim poisons every stride; shape[0] is never read, so a leading
  // dynamic dim is safe.
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
  lin = simplifiedForHardware(lin, map.getNumDims(), map.getNumSymbols());
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), lin,
                        map.getContext());
}

std::optional<unsigned> assignedBankOf(Operation *op) {
  // Two carriers, one fact: a discardable attribute while the access is still
  // affine, the reified op's own attribute afterwards.
  IntegerAttr bank;
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    bank = l.getBankAttr();
  else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
    bank = s.getBankAttr();
  else
    bank = op->getAttrOfType<IntegerAttr>(kBankAttr);
  if (!bank)
    return std::nullopt;
  return static_cast<unsigned>(bank.getInt());
}

} // namespace mlir::allo

//===----------------------------------------------------------------------===//
// Memory timing library
//===----------------------------------------------------------------------===//

MemoryLibrary MemoryLibrary::fromModule(ModuleOp module) {
  MemoryLibrary m;
  dcp::DCPathDeviceOp device;
  module.walk([&](dcp::DCPathDeviceOp d) { device = d; });
  if (!device)
    return m;
  // Both rows carry the same four fields under the same accessor names, so one
  // template reads a `dcp.storage` and a `dcp.stream_timing` alike.
  auto timing = [](auto row) {
    MemKindTiming t;
    t.latency.read = (unsigned)row.getRdLatency();
    t.latency.write = (unsigned)row.getWrLatency();
    t.delay.read = row.getRdDelay().convertToDouble();
    t.delay.write = row.getWrDelay().convertToDouble();
    return t;
  };
  Block &body = device.getBody().front();
  for (auto s : body.getOps<dcp::DCPathStorageOp>()) {
    m.storage.push_back({s.getSymName().str(), timing(s)});
    if (s.getIsDefault())
      m.defaultStorage = s.getSymName().str();
    if (s.getIsScatter())
      m.scatterStorage = s.getSymName().str();
  }
  for (auto st : body.getOps<dcp::DCPathStreamTimingOp>())
    m.fifo = timing(st);
  return m;
}

MemKindTiming MemoryLibrary::timing(StringRef name) const {
  for (const StorageRealization &s : storage)
    if (s.name == name)
      return s.timing;
  // `PreVerification` rejects an array resolving to a realization the device
  // does not declare, so reaching here means that check was bypassed and the
  // access would schedule at latency 0.
  assert(false && "storage realization not declared by the device -> silent "
                  "latency-0 access");
  static constexpr MemKindTiming zero;
  return zero;
}

bool MemoryLibrary::declares(StringRef name) const {
  return llvm::any_of(
      storage, [&](const StorageRealization &s) { return s.name == name; });
}

MemoryLibrary::Timing MemoryLibrary::timing(Operation *op) const {
  auto a = asMemAccess(op);
  if (!a)
    return {};
  // A stream is a FIFO, timed by its own row rather than via `resolveStorage`
  // (which also returns empty for an array with no declared realization).
  // Branch on the access kind, not the resolved name, or the two cases collide.
  std::string name;
  MemKindTiming t = fifo;
  if (a->kind != AccessKind::Stream) {
    name = resolveStorage(a->root, *this);
    // The one way a resolution comes back empty is a completely partitioned
    // array on a device marking no `scatter` row, which `PreVerification`
    // rejects; reaching here means that check was bypassed.
    assert(!name.empty() &&
           "an array access resolves to a storage realization");
    t = timing(name);
  }
  return a->isWrite ? Timing{t.latency.write, t.delay.write, name}
                    : Timing{t.latency.read, t.delay.read, name};
}

MemoryChar allo::characterize(Value memref, const MemoryLibrary &lib) {
  MemoryChar c;
  auto bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memref, kBindStorageAttr));
  c.constantTable = isConstantTable(memref);
  // A SimpleDualPort (S2P) RAM has a dedicated read and write port; every other
  // topology shares its ports, a ROM having no write port to dedicate.
  bool readOnly = bs.kind == MemoryKindEnum::ROM || c.constantTable;
  if (bs.port == MemoryPortEnum::SimpleDualPort && !readOnly) {
    c.splitRW = true;
    c.readPorts = 1;
    c.writePorts = 1;
  } else {
    c.sharedPorts = portCount(bs.port);
  }
  c.layout = bankLayoutOf(memref);
  c.storage = resolveStorage(memref, lib);
  return c;
}
