/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryModel.h"

#include "allo-c/Schedule.h" // kPartitionAttr, kBindStorageAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess (the access substrate)

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::detail {

// The array root a load/store accesses (views peeled), or null. Streams are not
// array accesses -> null (the bank model is array-only).
static Value memRefOf(Operation *op) {
  std::optional<MemAccess> a = asMemAccess(op);
  return a && a->kind == AccessKind::Array ? a->root : Value();
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
// primitive). Absent/unknown `type` -> the dual-port RAM default;
// absent/unknown `impl` -> `Auto` (the array falls to the library's default
// implementation).
struct BindStorage {
  MemoryPortEnum port = MemoryPortEnum::TrueDualPort;
  MemoryKindEnum kind = MemoryKindEnum::RAM;
  MemoryImplEnum impl = MemoryImplEnum::Auto;
};
static BindStorage parseBindStorage(DictionaryAttr bind) {
  BindStorage bs;
  if (!bind)
    return bs;
  if (auto ty = bind.getAs<StringAttr>("type")) {
    StringRef t = ty.getValue();
    bs.port =
        llvm::StringSwitch<MemoryPortEnum>(t)
            .Cases({"ram_1p", "rom_1p"}, MemoryPortEnum::SinglePort)
            .Cases({"ram_2p", "ram_s2p"}, MemoryPortEnum::SimpleDualPort)
            .Default(MemoryPortEnum::TrueDualPort); // t2p / 1wnr / rom_2p / np
    bs.kind = t.starts_with("rom") ? MemoryKindEnum::ROM : MemoryKindEnum::RAM;
  }
  if (auto im = bind.getAs<StringAttr>("impl"))
    bs.impl =
        llvm::StringSwitch<MemoryImplEnum>(im.getValue())
            .Case("bram", MemoryImplEnum::BRAM)
            .Case("uram", MemoryImplEnum::URAM)
            .Case("lutram", MemoryImplEnum::LUTRAM)
            .Case("register", MemoryImplEnum::Register)
            .Case("srl", MemoryImplEnum::LUTRAM) // shift-register: LUT-based
            .Default(MemoryImplEnum::Auto);
  return bs;
}

// Concurrent ports of a topology (per bank).
static unsigned portCount(MemoryPortEnum p) {
  return p == MemoryPortEnum::SinglePort ? 1u : 2u;
}

// The banking a memref's `allo.part` implies: block/cyclic axes multiply the
// bank count; a complete partition scatters into registers (no banks). Shared
// by the scheduler's per-bank ResII model and the microarch's MemUnit shape.
static PartitionInfo partitionOf(Value memRef) {
  PartitionInfo p;
  auto part = carrierAttr<PartitionAttr>(memRef, kPartitionAttr);
  if (!part)
    return p;
  unsigned rank = cast<MemRefType>(memRef.getType()).getRank();
  for (PartitionAxisAttr axis : part.getPartitions()) {
    // A complete partition scatters the array into registers -> unlimited.
    if (axis.getKind() == PartitionKindEnum::CompletePartition) {
      p.unlimited = true;
      break;
    }
    // Block/cyclic partitioning into `factor` banks multiplies the count.
    p.factor *= static_cast<unsigned>(axis.getFactor());
    if (axis.getKind() == PartitionKindEnum::CyclicPartition) {
      // `dim == 0` partitions every dimension by this factor.
      if (axis.getDim() == 0)
        for (unsigned d = 0; d < rank; ++d)
          p.cyclicAxes.push_back({d, axis.getFactor()});
      else
        p.cyclicAxes.push_back(
            {static_cast<unsigned>(axis.getDim() - 1), axis.getFactor()});
    } else {
      p.hasBlock = true;
    }
  }
  return p;
}

// The storage implementation a memref resolves to: a complete partition
// scatters into registers (regardless of any bind.storage impl); else an
// explicit `bind.storage impl`; else the library's default (unbound on-chip
// arrays -- Vitis: LUTRAM). This is the memref's position on the implementation
// axis, the input to per-impl access timing.
static MemoryImplEnum resolveImpl(Value memRef, MemoryImplEnum defaultImpl) {
  if (partitionOf(memRef).unlimited)
    return MemoryImplEnum::Register;
  BindStorage bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
  return bs.impl != MemoryImplEnum::Auto ? bs.impl : defaultImpl;
}

// The compile-time bank of \p op on a cyclic axis (0-based `dim`, `factor`), or
// nullopt when the subscript is not iteration-invariant modulo the factor (a
// roaming or non-affine access): a fixed bank needs every variable coefficient
// of the partition-dim subscript to vanish modulo the factor.
static std::optional<int64_t> staticBank(AffineMap map, unsigned dim,
                                         int64_t factor) {
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

static std::optional<int64_t> staticBank(Operation *op, unsigned dim,
                                         int64_t factor) {
  std::optional<MemAccess> a = asMemAccess(op);
  return a ? staticBank(a->map, dim, factor) : std::nullopt;
}

void MemoryBankModel::observe(Operation *op) {
  if (Value memRef = memRefOf(op))
    byMemref[memRef].accesses.push_back(op);
}

void MemoryBankModel::finalize() {
  for (auto &entry : byMemref) {
    Value memRef = entry.first;
    MemInfo &info = entry.second;
    // Port topology from `allo.bind.storage`. A SimpleDualPort (S2P) RAM has a
    // dedicated read and write port; every other topology shares its ports for
    // reads and writes. A ROM has no write port, so it uses shared ports.
    BindStorage bs =
        parseBindStorage(carrierAttr<DictionaryAttr>(memRef, kBindStorageAttr));
    bool readOnly = bs.kind == MemoryKindEnum::ROM;
    if (bs.port == MemoryPortEnum::SimpleDualPort && !readOnly) {
      info.splitRW = true;
      info.readPorts = 1;
      info.writePorts = 1;
    } else {
      info.sharedPorts = portCount(bs.port);
    }
    PartitionInfo part = partitionOf(memRef);
    info.unlimited = part.unlimited;
    info.partitionFactor = part.factor;
    info.cyclicAxes = std::move(part.cyclicAxes);
    bool hasBlock = part.hasBlock;
    // Per-bank refinement is sound only when every access is statically banked
    // on every cyclic axis (its bank is a fixed constant, so a per-bank
    // resource is exact even under modulo scheduling). Block/complete/mixed or
    // any roaming access falls back to the aggregate limit.
    if (info.unlimited || info.cyclicAxes.empty() || hasBlock)
      continue;
    info.perBank = llvm::all_of(info.accesses, [&](Operation *access) {
      return llvm::all_of(
          info.cyclicAxes, [&](std::pair<unsigned, int64_t> ax) {
            return staticBank(access, ax.first, ax.second).has_value();
          });
    });
  }
}

std::optional<std::pair<std::string, unsigned>>
MemoryBankModel::resource(Operation *op) const {
  Value memRef = memRefOf(op);
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
  std::optional<MemAccess> a = asMemAccess(op);
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

} // namespace mlir::allo::detail

namespace mlir::allo {

// Public forwarders (the banking pass reuses the scheduler's exact partition /
// static-bank logic, so they can never diverge).
PartitionInfo partitionOf(Value memRef) { return detail::partitionOf(memRef); }

std::optional<int64_t> staticBank(AffineMap map, unsigned dim, int64_t factor) {
  return detail::staticBank(map, dim, factor);
}

std::optional<int64_t> staticBank(Operation *op, unsigned dim, int64_t factor) {
  return detail::staticBank(op, dim, factor);
}

} // namespace mlir::allo

//===----------------------------------------------------------------------===//
// Memory timing library
//===----------------------------------------------------------------------===//

const MemKindTiming &MemoryLibrary::forImpl(MemoryImplEnum impl) const {
  for (const MemPrimitive &p : primitives)
    if (p.impl == impl)
      return p.timing;
  // An undeclared primitive is a zero (combinational) timing -- the same
  // "absent -> zero" convention the whole `memory:` section uses. Reaching here
  // with a concrete (non-Auto) impl means an array resolved to a storage kind
  // the device declares no timing for (a `bind.storage impl=` /
  // `default_memory` naming an undeclared primitive): its multi-cycle access
  // would be scheduled at latency 0 -- read before valid. Auto (a stream, timed
  // via `fifo`) is the only legitimate zero.
  assert(impl == MemoryImplEnum::Auto &&
         "storage impl not declared by the device -> silent latency-0 access");
  static const MemKindTiming zero;
  return zero;
}

MemoryLibrary::Timing MemoryLibrary::timing(Operation *op) const {
  std::optional<MemAccess> a = asMemAccess(op);
  if (!a)
    return {};
  MemoryImplEnum impl = a->kind == AccessKind::Stream
                            ? MemoryImplEnum::Auto
                            : detail::resolveImpl(a->root, defaultImpl);
  const MemKindTiming &t = a->kind == AccessKind::Stream ? fifo : forImpl(impl);
  return a->isWrite ? Timing{t.latency.write, t.delay.write, true, impl}
                    : Timing{t.latency.read, t.delay.read, true, impl};
}

MemoryChar mlir::allo::characterize(Value memref) {
  using namespace detail;
  MemoryChar c;
  BindStorage bs =
      parseBindStorage(carrierAttr<DictionaryAttr>(memref, kBindStorageAttr));
  c.readOnly = bs.kind == MemoryKindEnum::ROM;
  c.portsPerBank = portCount(bs.port);
  PartitionInfo part = partitionOf(memref);
  c.numBanks = part.factor;
  c.registers = part.unlimited;
  c.impl = resolveImpl(memref, MemoryImplEnum::LUTRAM);
  return c;
}
