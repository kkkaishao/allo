/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // addressDelayOf (per-site address)

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

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

// The element types of `types` (element type of a shaped type, else the type
// itself): the concrete operand/result types an IP row is matched against.
llvm::SmallVector<Type> elementTypes(TypeRange types) {
  llvm::SmallVector<Type> out;
  for (Type t : types) {
    if (auto sh = dyn_cast<ShapedType>(t))
      t = sh.getElementType();
    out.push_back(t);
  }
  return out;
}

// Whether every data operand of `op` is an integer (element type): the
// predicate an integer-arithmetic comb row matches on.
bool allIntegerOperands(Operation *op) {
  auto ts = elementTypes(op->getOperandTypes());
  return !ts.empty() &&
         llvm::all_of(ts, [](Type t) { return isa<IntegerType>(t); });
}

// The library row matching \p op, or null. Advanced (raw-mnemonic) rows match
// first (exact type list); abstract rows match last-wins, so a later-injected
// operator overrides an earlier one of the same signature (user @ip > built-in
// IP > comb fallback). An IP row matches by kind + exact operand/result element
// types; a comb row by kind + integer operands (or any type for select/neg).
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
      // `select`/`neg` comb rows match any operand type (a mux over any
      // datatype / a float sign flip, neither with an IP counterpart); every
      // other comb kind is integer arithmetic.
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

// Whether \p op needs an IP realization: a float arithmetic/compare, any cast
// to or from float, or a math.* advanced op. Integer arithmetic, integer
// resize, and memory/stream accesses are combinational or storage, and never
// require an IP.
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

MemoryLibrary memoryFromDevice(dcp::DCPathDeviceOp device) {
  MemoryLibrary m;
  auto i64 = [](DictionaryAttr d, StringRef k) {
    return (unsigned)cast<IntegerAttr>(d.get(k)).getInt();
  };
  auto f = [](DictionaryAttr d, StringRef k) {
    return cast<FloatAttr>(d.get(k)).getValueAsDouble();
  };
  auto timing = [&](DictionaryAttr d) {
    MemKindTiming t;
    t.latency.read = i64(d, "rd_lat");
    t.latency.write = i64(d, "wr_lat");
    t.delay.read = f(d, "rd_delay");
    t.delay.write = f(d, "wr_delay");
    return t;
  };
  for (NamedAttribute na : device.getMemory()) {
    auto impl = symbolizeMemoryImplEnum(na.getName().strref());
    if (!impl)
      continue;
    MemPrimitive p;
    p.impl = *impl;
    p.timing = timing(cast<DictionaryAttr>(na.getValue()));
    m.primitives.push_back(p);
  }
  if (DictionaryAttr fifo = device.getFifoAttr())
    m.fifo = timing(fifo);
  if (StringAttr def = device.getDefaultMemoryAttr())
    if (auto impl = symbolizeMemoryImplEnum(def.strref()))
      m.defaultImpl = *impl;
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

  // Comb rows first: `entries` is matched last-wins (see `matchEntry`), so
  // combinational integer arithmetic is the lowest-priority fallback; an
  // injected IP of the same kind (built-in or user) overrides it.
  if (device) {
    for (NamedAttribute na : device.getComb()) {
      auto kind = parseOpKind(na.getName().strref());
      if (!kind)
        continue;
      OperatorEntry e;
      e.kind = *kind;
      e.comb = true;
      e.latency = 0;
      e.inDelay = e.outDelay =
          cast<FloatAttr>(na.getValue()).getValueAsDouble();
      lib.entries.push_back(std::move(e));
    }
    lib.memory = memoryFromDevice(device);
  }

  // IP rows in injection order (built-in, then user), matched last-wins: a
  // user `@ip` appended after the built-ins overrides a built-in of the same
  // signature. Match types are the operator's declared signature element types.
  module.walk([&](dcp::DCPathOperatorOp op) {
    OperatorEntry e;
    e.latency = (uint32_t)op.getLatency();
    e.inDelay = op.getInDelay().convertToDouble();
    e.outDelay = op.getOutDelay().convertToDouble();
    e.pipelined = op.getPipelined();
    e.symbol = op.getSymName().str();
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

double OperatorLibrary::combDelay(OpKind kind) const {
  // Last wins, like `matchEntry`. `dcp.device.comb` is a dictionary, so there
  // is at most one row per kind in practice.
  double delay = 0.0;
  for (const OperatorEntry &e : entries)
    if (e.comb && e.kind == kind)
      delay = e.outDelay;
  return delay;
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
    if (kind == OpKind::MemRead || kind == OpKind::MemWrite)
      c.typeName += stringifyMemoryImplEnum(t.impl).str();
    c.latency = t.latency;
    c.inDelay = c.outDelay = t.delay;
    c.pipelined = t.pipelined;
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
    // The default row (0-latency comb fallback) is only safe for a
    // float->float arith op when it is genuinely combinational (`combKindOf`)
    // or `needsIP` already rejected the schedule; otherwise it is a miscompile.
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
  // `comb.<kind>`, else `default`. A non-empty `symbol` denotes the IP
  // realization path (`op_type`); an empty one, the combinational path.
  OperatorChar c;
  c.typeName = !e->symbol.empty() ? e->symbol
               : e->comb          ? ("comb." + opKindString(e->kind)).str()
                                  : std::string("default");
  c.latency = e->latency;
  c.inDelay = e->inDelay;
  c.outDelay = e->outDelay;
  c.pipelined = e->pipelined;
  c.symbol = e->symbol;
  return c;
}

bool OperatorLibrary::requiresUnmatchedIP(Operation *op) const {
  return needsIP(op) && matchEntry(advancedEntries, entries, op) == nullptr;
}

bool OperatorLibrary::hasDirectRealization(Operation *op) const {
  return matchEntry(advancedEntries, entries, op) != nullptr;
}
