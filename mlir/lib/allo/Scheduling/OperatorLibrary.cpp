/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // addressDelayOf (per-site address)
#include "allo/Support/Logging.h"         // logging::info

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Abstract-kind <-> string
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::allo::opKindString(OpKind kind) {
  switch (kind) {
  case OpKind::Add:
    return "add";
  case OpKind::Sub:
    return "sub";
  case OpKind::Mul:
    return "mul";
  case OpKind::Div:
    return "div";
  case OpKind::Rem:
    return "rem";
  case OpKind::Max:
    return "max";
  case OpKind::Min:
    return "min";
  case OpKind::MaxNum:
    return "maxnum";
  case OpKind::MinNum:
    return "minnum";
  case OpKind::CeilDiv:
    return "ceildiv";
  case OpKind::FloorDiv:
    return "floordiv";
  case OpKind::Neg:
    return "neg";
  case OpKind::Cmp:
    return "cmp";
  case OpKind::And:
    return "and";
  case OpKind::Or:
    return "or";
  case OpKind::Xor:
    return "xor";
  case OpKind::Shl:
    return "shl";
  case OpKind::Shr:
    return "shr";
  case OpKind::Select:
    return "select";
  case OpKind::ICastI:
    return "icast";
  case OpKind::FCastI:
    return "ifcast";
  case OpKind::FCastF:
    return "fcast";
  default:
    return "";
  }
}

std::optional<OpKind> mlir::allo::parseOpKind(llvm::StringRef s) {
  return llvm::StringSwitch<std::optional<OpKind>>(s)
      .Case("add", OpKind::Add)
      .Case("sub", OpKind::Sub)
      .Case("mul", OpKind::Mul)
      .Case("div", OpKind::Div)
      .Case("rem", OpKind::Rem)
      .Case("max", OpKind::Max)
      .Case("min", OpKind::Min)
      .Case("maxnum", OpKind::MaxNum)
      .Case("minnum", OpKind::MinNum)
      .Case("ceildiv", OpKind::CeilDiv)
      .Case("floordiv", OpKind::FloorDiv)
      .Case("neg", OpKind::Neg)
      .Case("cmp", OpKind::Cmp)
      .Case("and", OpKind::And)
      .Case("or", OpKind::Or)
      .Case("xor", OpKind::Xor)
      .Case("shl", OpKind::Shl)
      .Case("shr", OpKind::Shr)
      .Case("select", OpKind::Select)
      .Case("icast", OpKind::ICastI)
      .Case("ifcast", OpKind::FCastI)
      .Case("fcast", OpKind::FCastF)
      .Default(std::nullopt);
}

std::optional<CombOpKindEnum> mlir::allo::combKindOf(Operation *op) {
  using E = CombOpKindEnum;
  return llvm::TypeSwitch<Operation *, std::optional<E>>(op)
      .Case<arith::AddIOp>([](auto) { return E::Addi; })
      .Case<arith::SubIOp>([](auto) { return E::Subi; })
      .Case<arith::MulIOp>([](auto) { return E::Muli; })
      .Case<arith::DivSIOp>([](auto) { return E::Divsi; })
      .Case<arith::DivUIOp>([](auto) { return E::Divui; })
      .Case<arith::RemSIOp>([](auto) { return E::Remsi; })
      .Case<arith::RemUIOp>([](auto) { return E::Remui; })
      .Case<arith::AndIOp>([](auto) { return E::Andi; })
      .Case<arith::OrIOp>([](auto) { return E::Ori; })
      .Case<arith::XOrIOp>([](auto) { return E::Xori; })
      .Case<arith::ShLIOp>([](auto) { return E::Shli; })
      .Case<arith::ShRSIOp>([](auto) { return E::Shrsi; })
      .Case<arith::ShRUIOp>([](auto) { return E::Shrui; })
      .Case<arith::CmpIOp>([](auto) { return E::Cmpi; })
      .Case<arith::SelectOp>([](auto) { return E::Select; })
      .Case<arith::ExtSIOp>([](auto) { return E::Extsi; })
      .Case<arith::ExtUIOp>([](auto) { return E::Extui; })
      .Case<arith::TruncIOp>([](auto) { return E::Trunci; })
      .Case<arith::IndexCastOp, arith::IndexCastUIOp>(
          [](auto) { return E::IndexCast; })
      .Case<affine::AffineApplyOp>([](auto) { return E::Apply; })
      .Case<arith::NegFOp>([](auto) { return E::Negf; })
      .Case<arith::MinSIOp>([](auto) { return E::Minsi; })
      .Case<arith::MaxSIOp>([](auto) { return E::Maxsi; })
      .Case<arith::MinUIOp>([](auto) { return E::Minui; })
      .Case<arith::MaxUIOp>([](auto) { return E::Maxui; })
      .Default([](auto) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// Classification: concrete IR op -> abstract kind
//===----------------------------------------------------------------------===//

OpKind mlir::allo::classify(Operation *op) {
  return llvm::TypeSwitch<Operation *, OpKind>(op)
      .Case<arith::AddIOp, arith::AddFOp>([](auto) { return OpKind::Add; })
      .Case<arith::SubIOp, arith::SubFOp>([](auto) { return OpKind::Sub; })
      .Case<arith::MulIOp, arith::MulFOp>([](auto) { return OpKind::Mul; })
      .Case<arith::DivSIOp, arith::DivUIOp, arith::DivFOp>(
          [](auto) { return OpKind::Div; })
      .Case<arith::RemSIOp, arith::RemUIOp, arith::RemFOp>(
          [](auto) { return OpKind::Rem; })
      .Case<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(
          [](auto) { return OpKind::Max; })
      .Case<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(
          [](auto) { return OpKind::Min; })
      .Case<arith::MaxNumFOp>([](auto) { return OpKind::MaxNum; })
      .Case<arith::MinNumFOp>([](auto) { return OpKind::MinNum; })
      .Case<arith::CeilDivSIOp, arith::CeilDivUIOp>(
          [](auto) { return OpKind::CeilDiv; })
      .Case<arith::FloorDivSIOp>([](auto) { return OpKind::FloorDiv; })
      .Case<arith::NegFOp>([](auto) { return OpKind::Neg; })
      .Case<arith::CmpIOp, arith::CmpFOp>([](auto) { return OpKind::Cmp; })
      .Case<arith::AndIOp>([](auto) { return OpKind::And; })
      .Case<arith::OrIOp>([](auto) { return OpKind::Or; })
      .Case<arith::XOrIOp>([](auto) { return OpKind::Xor; })
      .Case<arith::ShLIOp>([](auto) { return OpKind::Shl; })
      .Case<arith::ShRSIOp, arith::ShRUIOp>([](auto) { return OpKind::Shr; })
      .Case<arith::SelectOp>([](auto) { return OpKind::Select; })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>([](auto) { return OpKind::ICastI; })
      .Case<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(
          [](auto) { return OpKind::FCastI; })
      .Case<arith::ExtFOp, arith::TruncFOp>([](auto) { return OpKind::FCastF; })
      .Case<affine::AffineLoadOp, memref::LoadOp>(
          [](auto) { return OpKind::MemRead; })
      .Case<affine::AffineStoreOp, memref::StoreOp>(
          [](auto) { return OpKind::MemWrite; })
      .Case<StreamGetOp>([](auto) { return OpKind::StreamRead; })
      .Case<StreamPutOp>([](auto) { return OpKind::StreamWrite; })
      .Default([](auto) { return OpKind::Unknown; });
}

//===----------------------------------------------------------------------===//
// Matching helpers
//===----------------------------------------------------------------------===//

namespace {

// The element type of each shaped type in `types`, else the type itself: what
// an IP row is matched against.
llvm::SmallVector<Type> elementTypes(TypeRange types) {
  llvm::SmallVector<Type> out;
  for (Type t : types) {
    if (auto sh = dyn_cast<ShapedType>(t))
      t = sh.getElementType();
    out.push_back(t);
  }
  return out;
}

// Whether every data operand of `op` has integer element type: what an
// integer-arithmetic comb row matches on.
bool allIntegerOperands(Operation *op) {
  auto ts = elementTypes(op->getOperandTypes());
  return !ts.empty() &&
         llvm::all_of(ts, [](Type t) { return isa<IntegerType>(t); });
}

// The library row matching \p op, or null. Advanced (raw-mnemonic) rows match
// first on an exact type list; abstract rows match last-wins, so a
// later-injected operator overrides an earlier one of the same signature
// (user @ip > built-in IP > comb fallback).
const OperatorEntry *matchEntry(const std::vector<OperatorEntry> &advanced,
                                const std::vector<OperatorEntry> &entries,
                                Operation *op) {
  auto kind = classify(op);
  auto mnem = op->getName().stripDialect();
  auto aTys = elementTypes(op->getOperandTypes());
  auto rTys = elementTypes(op->getResultTypes());
  ArrayRef<Type> a = aTys, r = rTys;
  for (const OperatorEntry &e : advanced)
    if (e.mlirOp == mnem && ArrayRef<Type>(e.argTypes) == a &&
        ArrayRef<Type>(e.resTypes) == r)
      return &e;
  for (const OperatorEntry &e : llvm::reverse(entries)) {
    if (e.kind != kind)
      continue;
    if (e.comb) {
      // `select`/`neg` comb rows match any operand type: a mux over any
      // datatype, a float sign flip. Every other comb kind is integer
      // arithmetic.
      if (kind == OpKind::Select || kind == OpKind::Neg ||
          allIntegerOperands(op))
        return &e;
    } else if (ArrayRef<Type>(e.argTypes) == a &&
               ArrayRef<Type>(e.resTypes) == r) {
      return &e;
    }
  }
  return nullptr;
}

// Whether \p op needs an IP realization: a float arithmetic op or compare
// other than neg/select, any cast to or from float, or a math.* advanced op.
bool needsIP(Operation *op) {
  auto isFloat = [](Type t) { return isa<FloatType>(t); };
  bool floaty = llvm::any_of(elementTypes(op->getOperandTypes()), isFloat) ||
                llvm::any_of(elementTypes(op->getResultTypes()), isFloat);
  switch (classify(op)) {
  case OpKind::Add:
  case OpKind::Sub:
  case OpKind::Mul:
  case OpKind::Div:
  case OpKind::Rem:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::MaxNum:
  case OpKind::MinNum:
  case OpKind::Cmp:
    return floaty;
  case OpKind::CeilDiv:
  case OpKind::FloorDiv:
    // No native comb realization; `legalize-arith` expands these unless the
    // device provides an IP, so one reaching the scheduler must be an IP.
    return true;
  case OpKind::FCastI:
  case OpKind::FCastF:
    return true;
  case OpKind::Unknown:
    return isa<math::MathDialect>(op->getDialect());
  default:
    return false;
  }
}

// The identity of the unit \p op runs on. Empty without a realization, or when
// \p op is not the single-result compute a `FuncUnit` is built from.
OperatorIdentity identityOf(Operation *op, std::string realization, bool comb) {
  OperatorIdentity id;
  if (realization.empty() || op->getNumResults() != 1)
    return id;
  id.realization = std::move(realization);
  id.comb = comb;
  id.argTypes.assign(op->getOperandTypes().begin(),
                     op->getOperandTypes().end());
  id.resultType = op->getResult(0).getType();
  id.predicate = op->getAttr("predicate");
  id.map = op->getAttr("map");
  return id;
}

MemoryLibrary memoryFromDevice(dcp::DCPathDeviceOp device) {
  MemoryLibrary m;
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
    m.storage.push_back({s.getSymName().str(), s.getPorts(), timing(s)});
    if (s.getIsDefault())
      m.defaultStorage = s.getSymName().str();
    if (s.getIsScatter())
      m.scatterStorage = s.getSymName().str();
  }
  for (auto st : body.getOps<dcp::DCPathStreamTimingOp>())
    m.fifo = timing(st);
  return m;
}

} // namespace

//===----------------------------------------------------------------------===//
// Building the library from injected `dcp.device` / `dcp.operator` IR
//===----------------------------------------------------------------------===//

OperatorLibrary OperatorLibrary::fromModule(ModuleOp module) {
  OperatorLibrary lib;
  // The default row: ops that match nothing (constants, address arithmetic) are
  // 0-latency combinational.
  lib.defaultEntry.latency = 0;
  lib.defaultEntry.inDelay = lib.defaultEntry.outDelay = 0.1;

  dcp::DCPathDeviceOp device;
  module.walk([&](dcp::DCPathDeviceOp d) { device = d; });

  // Comb rows first: `entries` is matched last-wins, so comb is the
  // lowest-priority fallback and an injected IP of the same kind overrides it.
  if (device) {
    for (dcp::DCPathCombOp comb :
         device.getBody().getOps<dcp::DCPathCombOp>()) {
      auto kind = parseOpKind(comb.getKind());
      // `OpKind` is this layer's vocabulary, so the dialect verifier cannot
      // check it and a row naming something outside it is reported here rather
      // than dropped into a silent zero delay.
      if (!kind) {
        logging::error(logging::Stage::Prep, comb)
            << "Device declares a combinational delay for '" << comb.getKind()
            << "', which is not an operator kind";
        continue;
      }
      OperatorEntry e;
      e.kind = *kind;
      e.comb = true;
      e.latency = 0;
      e.inDelay = e.outDelay = comb.getDelay().convertToDouble();
      e.uses = comb.getUsesAttr();
      lib.entries.push_back(std::move(e));
    }
    lib.memory = memoryFromDevice(device);

    // The currency: the most plentiful resource sets the scale, so a price is
    // how scarce a resource is relative to the one the part has most of.
    int64_t widest = 1;
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>())
      widest = std::max<int64_t>(widest, r.getCapacity());
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>())
      lib.resourcePrices[r.getSymName()] = std::max<int64_t>(
          1, llvm::divideNearest<int64_t>(kPriceResolution * widest,
                                          r.getCapacity()));
    for (auto m : device.getBody().getOps<dcp::DCPathMuxOp>())
      lib.muxUses = m.getUsesAttr();
    for (auto c : device.getBody().getOps<dcp::DCPathChainOp>())
      lib.chainUses = c.getUsesAttr();
  }

  // IP rows in injection order (built-in, then user), matched last-wins: a user
  // `@ip` overrides a built-in of the same signature.
  module.walk([&](dcp::DCPathOperatorOp op) {
    OperatorEntry e;
    e.latency = (uint32_t)op.getLatency();
    e.pipelined = op.getPipelined();
    e.inDelay = op.getInDelay().convertToDouble();
    e.outDelay = op.getOutDelay().convertToDouble();
    e.symbol = op.getSymName().str();
    e.uses = op.getUsesAttr();
    auto sig = op.getSignature();
    e.argTypes = elementTypes(sig.getInputs());
    e.resTypes = elementTypes(sig.getResults());
    if (std::optional<OpKind> kind = parseOpKind(op.getKind())) {
      e.kind = *kind;
      lib.entries.push_back(std::move(e));
    } else {
      e.mlirOp = op.getKind().str(); // advanced: matched by stripped mnemonic
      lib.advancedEntries.push_back(std::move(e));
    }
  });
  return lib;
}

//===----------------------------------------------------------------------===//
// Lookup
//===----------------------------------------------------------------------===//

OpKind mlir::allo::opKindOf(CombOpKindEnum kind) {
  using E = CombOpKindEnum;
  switch (kind) {
  case E::Addi:
    return OpKind::Add;
  case E::Subi:
    return OpKind::Sub;
  case E::Muli:
    return OpKind::Mul;
  case E::Divsi:
  case E::Divui:
    return OpKind::Div;
  case E::Remsi:
  case E::Remui:
    return OpKind::Rem;
  case E::Andi:
    return OpKind::And;
  case E::Ori:
    return OpKind::Or;
  case E::Xori:
    return OpKind::Xor;
  case E::Shli:
    return OpKind::Shl;
  case E::Shrsi:
  case E::Shrui:
    return OpKind::Shr;
  case E::Cmpi:
    return OpKind::Cmp;
  case E::Select:
    return OpKind::Select;
  case E::Extsi:
  case E::Extui:
  case E::Trunci:
  case E::IndexCast:
    return OpKind::ICastI;
  case E::Negf:
    return OpKind::Neg;
  case E::Minsi:
  case E::Minui:
    return OpKind::Min;
  case E::Maxsi:
  case E::Maxui:
    return OpKind::Max;
  case E::Apply:
    return OpKind::Unknown; // no abstract row; priced by the default one
  }
  llvm_unreachable("every comb realization names an abstract kind or Unknown");
}

const OperatorEntry *OperatorLibrary::combEntry(OpKind kind) const {
  // `dcp.device.comb` is a dictionary, so there is at most one row per kind in
  // practice.
  const OperatorEntry *found = nullptr;
  for (const OperatorEntry &e : entries)
    if (e.comb && e.kind == kind)
      found = &e;
  return found;
}

double OperatorLibrary::combDelay(CombOpKindEnum kind) const {
  const OperatorEntry *e = combEntry(opKindOf(kind));
  return e ? e->outDelay : defaultEntry.outDelay;
}

double OperatorLibrary::combDelay(OpKind kind) const {
  const OperatorEntry *e = combEntry(kind);
  return e ? e->outDelay : 0.0;
}

int64_t OperatorLibrary::priceOf(ArrayAttr uses,
                                 ArrayRef<int64_t> params) const {
  int64_t total = 0;
  for (auto [resource, count] : evaluateResourceUse(uses, params)) {
    auto it = resourcePrices.find(resource.getLeafReference().getValue());
    assert(it != resourcePrices.end() &&
           "a realization spends a resource the device does not declare, which "
           "the dialect verifier resolves before this point");
    assert(count >= 0 && "a realization spends a negative resource count");
    total += it->second * count;
  }
  return total;
}

int64_t OperatorLibrary::muxPrice(int64_t sources, int64_t width) const {
  return priceOf(muxUses, {sources, width});
}

int64_t OperatorLibrary::chainPrice(int64_t depth, int64_t width) const {
  // A chain of no stages is a wire. The device row characterizes a structure
  // that exists, so its head and tail terms are not zero at depth zero.
  if (depth <= 0)
    return 0;
  return priceOf(chainUses, {depth, width});
}

int64_t OperatorLibrary::pulsePrice() const {
  return chainPrice(2, 1) - chainPrice(1, 1);
}

OperatorChar OperatorLibrary::lookup(Operation *op) const {
  auto kind = classify(op);

  // Memory / stream accesses are the storage dimension.
  switch (kind) {
  case OpKind::MemRead:
  case OpKind::MemWrite:
  case OpKind::StreamRead:
  case OpKind::StreamWrite: {
    auto t = memory.timing(op);
    OperatorChar c;
    c.typeName = (kind == OpKind::MemRead      ? "mem.rd"
                  : kind == OpKind::MemWrite   ? "mem.wr"
                  : kind == OpKind::StreamRead ? "srm.rd"
                                               : "srm.wr");
    if (kind == OpKind::MemRead || kind == OpKind::MemWrite) {
      assert(!t.storage.empty() &&
             "an array access resolves to a storage realization");
      c.typeName += t.storage;
    }
    c.latency = t.latency;
    c.inDelay = c.outDelay = t.delay;
    // The address cone is no operation of its own, so no dependence carries its
    // delay: charge it to the port it feeds. The type NAME carries it too, or
    // two sites costing differently would share one characterization.
    if (double addr = addressDelayOf(op, *this)) {
      // A registered port takes the cone on its input side alone, ending at its
      // own address register. A zero-latency port has none, and CIRCT requires
      // its two delays to agree, so there the cone lands on both.
      c.inDelay += addr;
      if (c.latency == 0)
        c.outDelay += addr;
      c.typeName += "@" + llvm::formatv("{0:F2}", addr).str();
    }
    return c;
  }
  default:
    break;
  }

  const auto *e = matchEntry(advancedEntries, entries, op);
  if (!e) {
    // The default row (0-latency comb) is a miscompile for a float->float arith
    // op that is neither combinational nor already rejected by `needsIP`.
    auto isFloat = [](Type t) { return isa<FloatType>(t); };
    bool floatIn = llvm::any_of(elementTypes(op->getOperandTypes()), isFloat);
    bool floatOut = llvm::any_of(elementTypes(op->getResultTypes()), isFloat);
    assert((needsIP(op) || combKindOf(op) || !(floatIn && floatOut)) &&
           "unrecognized arith float->float op fell through to the latency-0 "
           "default row (no IP requirement, no comb lowering); add it to "
           "classify()/needsIP(). This is an early duplicate of the operator "
           "realizability check in validateDatapath, which is where a release "
           "build reports it");
    e = &defaultEntry;
  }

  // The stable Problem::OperatorType key: an IP row's symbol, a comb row's
  // `comb.<kind>`, else `default`.
  OperatorChar c;
  c.typeName = !e->symbol.empty() ? e->symbol
               : e->comb          ? ("comb." + opKindString(e->kind)).str()
                                  : std::string("default");
  c.latency = e->latency;
  c.pipelined = e->pipelined;
  c.inDelay = e->inDelay;
  c.outDelay = e->outDelay;
  // Every row is characterized over one parameter, an operand width; an IP's
  // signature pins it, so there the factors are constants and this is the
  // measured core.
  if (op->getNumResults() == 1)
    if (Type t = elementTypes(op->getResultTypes()).front(); t.isIntOrFloat())
      c.price = priceOf(e->uses, {(int64_t)t.getIntOrFloatBitWidth()});
  // The realization is the row's own symbol when it is an IP, else the native
  // lowering the reifier picks; the default row reaches the comb arm too.
  if (!e->symbol.empty())
    c.identity = identityOf(op, e->symbol, /*comb=*/false);
  else if (std::optional<CombOpKindEnum> ck = combKindOf(op))
    c.identity = identityOf(op, stringifyCombOpKindEnum(*ck).str(), true);
  return c;
}

std::string OperatorIdentity::key() const {
  std::string s = realization;
  llvm::raw_string_ostream os(s);
  os << '(';
  llvm::interleaveComma(argTypes, os);
  os << ")->" << resultType;
  if (predicate)
    os << " p" << predicate;
  if (map)
    os << " m" << map;
  return os.str();
}

OperatorIdentity mlir::allo::operatorIdentity(dcp::DCPathComputeOp comp) {
  if (std::optional<CombOpKindEnum> ck = comp.getCombKind())
    return identityOf(comp, stringifyCombOpKindEnum(*ck).str(), true);
  return identityOf(comp, comp.getOpTypeAttr().getValue().str(), false);
}

OperatorIdentity mlir::allo::operatorIdentity(Operation *op,
                                              const OperatorLibrary &lib) {
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return operatorIdentity(comp);
  return lib.lookup(op).identity;
}

void mlir::allo::reportOperatorClassSplit(circt::scheduling::Problem &problem,
                                          const OperatorLibrary &lib) {
  if (!logging::detail::enabled(logging::Level::Info))
    return;
  // Operator type -> {ops priced under it, their distinct identities}. Sorted,
  // so two compiles report the classes in the same order.
  std::map<std::string, std::pair<unsigned, llvm::StringSet<>>> byType;
  for (Operation *op : problem.getOperations()) {
    OperatorIdentity id = operatorIdentity(op, lib);
    if (!id.realized())
      continue;
    auto &[count, classes] = byType[lib.lookup(op).typeName];
    ++count;
    classes.insert(id.key());
  }

  // Only a type pricing several ops under several identities over-approximates.
  llvm::SmallVector<std::pair<std::string, std::pair<unsigned, unsigned>>>
      split;
  for (auto &[type, seen] : byType)
    if (seen.first > 1 && seen.second.size() > 1)
      split.push_back({type, {seen.first, (unsigned)seen.second.size()}});
  if (split.empty())
    return;

  auto d = logging::info(logging::Stage::Sched, problem.getContainingOp());
  d << "Operator classes: " << split.size() << " of " << byType.size()
    << " operator types cover several operator identities:";
  for (auto &[type, counts] : split)
    d << " " << type << " " << counts.first << " ops / " << counts.second
      << " classes,";
}

bool OperatorLibrary::requiresUnmatchedIP(Operation *op) const {
  return needsIP(op) && matchEntry(advancedEntries, entries, op) == nullptr;
}

bool OperatorLibrary::hasDirectRealization(Operation *op) const {
  return matchEntry(advancedEntries, entries, op) != nullptr;
}
