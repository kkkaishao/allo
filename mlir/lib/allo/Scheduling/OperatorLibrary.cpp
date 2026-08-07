/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // addressDelayOf (per-site address)
#include "allo/Support/BitAnalysis.h"     // isBitRename

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Native realizations: the one table the three views below are generated from.
//
// A row is (`CombOpKindEnum` case, abstract `OpKind` case, the MLIR ops it
// realizes). One table rather than three switches, because the three have to
// agree: `classify(op) == opKindOf(*combKindOf(op))` wherever an op has a
// native lowering, and nothing outside the table could enforce that. Adding a
// native operator is one row here plus one `emitCompute` case.
//
// A row's kind is FINER than its abstract kind, which is what the two columns
// are for: a device characterizes "an integer add", while the emitter has to
// know it is emitting `addi` and not `subi`.
//===----------------------------------------------------------------------===//

#define ALLO_COMB_KINDS(X)                                                     \
  X(Addi, Add, arith::AddIOp)                                                  \
  X(Subi, Sub, arith::SubIOp)                                                  \
  X(Muli, Mul, arith::MulIOp)                                                  \
  X(Divsi, Div, arith::DivSIOp)                                                \
  X(Divui, Div, arith::DivUIOp)                                                \
  X(Remsi, Rem, arith::RemSIOp)                                                \
  X(Remui, Rem, arith::RemUIOp)                                                \
  X(Andi, And, arith::AndIOp)                                                  \
  X(Ori, Or, arith::OrIOp)                                                     \
  X(Xori, Xor, arith::XOrIOp)                                                  \
  X(Shli, Shl, arith::ShLIOp)                                                  \
  X(Shrsi, Shr, arith::ShRSIOp)                                                \
  X(Shrui, Shr, arith::ShRUIOp)                                                \
  X(Cmpi, Cmp, arith::CmpIOp)                                                  \
  X(Select, Select, arith::SelectOp)                                           \
  X(Extsi, ICastI, arith::ExtSIOp)                                             \
  X(Extui, ICastI, arith::ExtUIOp)                                             \
  X(Trunci, ICastI, arith::TruncIOp)                                           \
  X(IndexCast, ICastI, arith::IndexCastOp, arith::IndexCastUIOp)               \
  X(Negf, Neg, arith::NegFOp)                                                  \
  X(Minsi, Min, arith::MinSIOp)                                                \
  X(Minui, Min, arith::MinUIOp)                                                \
  X(Maxsi, Max, arith::MaxSIOp)                                                \
  X(Maxui, Max, arith::MaxUIOp)                                                \
  /* An address expression, priced by the DEFAULT row: no device row covers */ \
  /* a whole affine map, whose delay is its own operators' (`addressDelay`) */ \
  X(Apply, Unknown, affine::AffineApplyOp)

std::optional<CombOpKindEnum> mlir::allo::combKindOf(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<CombOpKindEnum>>(op)
#define X(comb, abstract, ...)                                                 \
  .Case<__VA_ARGS__>([](auto) { return CombOpKindEnum::comb; })
      ALLO_COMB_KINDS(X)
#undef X
          .Default([](auto) { return std::nullopt; });
}

OpKind mlir::allo::opKindOf(CombOpKindEnum kind) {
  switch (kind) {
#define X(comb, abstract, ...)                                                 \
  case CombOpKindEnum::comb:                                                   \
    return OpKind::abstract;
    ALLO_COMB_KINDS(X)
#undef X
  }
  llvm_unreachable("every comb realization names an abstract kind or Unknown");
}

//===----------------------------------------------------------------------===//
// Classification: concrete IR op -> abstract kind
//
// Total, so it also covers what has no native lowering: float arithmetic, the
// float casts, and the composite integer kinds `legalize-arith` expands.
// `Unknown` for everything else, an access included: an access is timed by its
// storage (`accessCharacterization`), so no operator row answers for it.
//===----------------------------------------------------------------------===//

OpKind mlir::allo::classify(Operation *op) {
  return llvm::TypeSwitch<Operation *, OpKind>(op)
#define X(comb, abstract, ...)                                                 \
  .Case<__VA_ARGS__>([](auto) { return OpKind::abstract; })
      ALLO_COMB_KINDS(X)
#undef X
          .Case<arith::AddFOp>([](auto) { return OpKind::Add; })
          .Case<arith::SubFOp>([](auto) { return OpKind::Sub; })
          .Case<arith::MulFOp>([](auto) { return OpKind::Mul; })
          .Case<arith::DivFOp>([](auto) { return OpKind::Div; })
          .Case<arith::RemFOp>([](auto) { return OpKind::Rem; })
          .Case<arith::MaximumFOp>([](auto) { return OpKind::Max; })
          .Case<arith::MinimumFOp>([](auto) { return OpKind::Min; })
          .Case<arith::MaxNumFOp>([](auto) { return OpKind::MaxNum; })
          .Case<arith::MinNumFOp>([](auto) { return OpKind::MinNum; })
          .Case<arith::CeilDivSIOp, arith::CeilDivUIOp>(
              [](auto) { return OpKind::CeilDiv; })
          .Case<arith::FloorDivSIOp>([](auto) { return OpKind::FloorDiv; })
          .Case<arith::CmpFOp>([](auto) { return OpKind::Cmp; })
          .Case<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                arith::FPToUIOp>([](auto) { return OpKind::FCastI; })
          .Case<arith::ExtFOp, arith::TruncFOp>(
              [](auto) { return OpKind::FCastF; })
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
// integer-arithmetic comb row matches on. An `index` counts, and has to: a
// bound, a counter and an address are index-typed, and a row that skipped them
// would leave the device's own adder and divider priced at the DEFAULT row,
// which is 0.1 ns whatever it builds.
bool allIntegerOperands(Operation *op) {
  auto ts = elementTypes(op->getOperandTypes());
  return !ts.empty() && llvm::all_of(ts, [](Type t) {
    return isa<IntegerType, IndexType>(t);
  });
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

// The identity of the unit \p op runs on: a native \p comb realization or the
// `dcp.operator` \p symbol, exactly one of which a caller gives. Empty without
// either, or when \p op is not the single-result compute a `FuncUnit` is built
// from.
OperatorIdentity identityOf(Operation *op, std::optional<CombOpKindEnum> comb,
                            StringRef symbol) {
  assert((!comb || symbol.empty()) && "a compute takes one realization path");
  OperatorIdentity id;
  if ((!comb && symbol.empty()) || op->getNumResults() != 1)
    return id;
  id.comb = comb;
  id.ipSymbol = symbol.str();
  id.argTypes.assign(op->getOperandTypes().begin(),
                     op->getOperandTypes().end());
  id.resultType = op->getResult(0).getType();
  id.predicate = op->getAttr("predicate");
  id.map = op->getAttr("map");
  return id;
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
    lib.regFloor = device.getRegDelay().convertToDouble();
    for (dcp::DCPathCombOp comb :
         device.getBody().getOps<dcp::DCPathCombOp>()) {
      OperatorEntry e;
      e.kind = comb.getKind();
      e.comb = true;
      e.latency = 0;
      // Left as a curve: what width to evaluate it at is the matched
      // OPERATION's, which `lookup` knows and this does not.
      e.delay = comb.getDelayAttr();
      e.uses = comb.getUsesAttr();
      lib.entries.push_back(std::move(e));
    }

    // The currency: the most plentiful resource sets the scale, so a price is
    // how scarce a resource is relative to the one the part has most of.
    int64_t widest = 1;
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>())
      widest = std::max<int64_t>(widest, r.getCapacity());
    for (auto r : device.getBody().getOps<dcp::DCPathResourceOp>())
      lib.resourcePrices[r.getSymName()] =
          std::max<int64_t>(1, llvm::divideNearest<int64_t>(
                                   kPriceResolution * widest, r.getCapacity()));
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
    if (std::optional<OpKind> kind = symbolizeOpKindEnum(op.getKind())) {
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

const OperatorEntry *OperatorLibrary::combEntry(OpKind kind) const {
  // `dcp.device.comb` is a dictionary, so there is at most one row per kind in
  // practice.
  const OperatorEntry *found = nullptr;
  for (const OperatorEntry &e : entries)
    if (e.comb && e.kind == kind)
      found = &e;
  return found;
}

double OperatorLibrary::combDelay(CombOpKindEnum kind, int64_t width) const {
  const OperatorEntry *e = combEntry(opKindOf(kind));
  return e ? e->delay.evaluate(width) : defaultEntry.outDelay;
}

double OperatorLibrary::combDelay(OpKind kind, int64_t width) const {
  const OperatorEntry *e = combEntry(kind);
  return e ? e->delay.evaluate(width) : 0.0;
}

double OperatorLibrary::combMarginalDelay(OpKind kind, int64_t width) const {
  return std::max(0.0, combDelay(kind, width) - regFloor);
}

double OperatorLibrary::combMarginalDelay(CombOpKindEnum kind,
                                          int64_t width) const {
  return std::max(0.0, combDelay(kind, width) - regFloor);
}

int64_t mlir::allo::combParamWidth(Operation *op) {
  int64_t width = 0;
  for (Type t : elementTypes(op->getOperandTypes()))
    if (t.isIntOrFloat())
      width = std::max<int64_t>(width, t.getIntOrFloatBitWidth());
  if (width)
    return width;
  for (Type t : elementTypes(op->getResultTypes()))
    if (t.isIntOrFloat())
      width = std::max<int64_t>(width, t.getIntOrFloatBitWidth());
  return width ? width : 1;
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

NodeTiming mlir::allo::accessCharacterization(Operation *op,
                                              const OperatorLibrary &opLib,
                                              const MemoryLibrary &memLib) {
  std::optional<MemAccess> a = asMemAccess(op);
  assert(a && "accessCharacterization was handed something that is not an "
              "access");
  MemoryLibrary::Timing t = memLib.timing(op);
  NodeTiming c;
  bool stream = a->kind == AccessKind::Stream;
  c.typeName = stream ? (a->isWrite ? "srm.wr" : "srm.rd")
                      : (a->isWrite ? "mem.wr" : "mem.rd");
  if (!stream) {
    assert(!t.storage.empty() &&
           "an array access resolves to a storage realization");
    c.typeName += t.storage;
  }
  c.latency = t.latency;
  c.inDelay = c.outDelay = t.delay;
  // The address cone is no operation of its own, so no dependence carries its
  // delay: charge it to the port it feeds. The type NAME carries it too, or
  // two sites costing differently would share one characterization.
  if (double addr = addressDelayOf(op, opLib)) {
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

OperatorChar OperatorLibrary::lookup(Operation *op) const {
  // Neither a sub-kernel call nor a memory access is an operator: no device
  // row, identity, or price. A call's length is its callee's own schedule
  // (`scheduledCallLatency`), an access's is its storage's
  // (`accessCharacterization`); each caller decides what that means for its
  // own question.
  assert(!isSyncSubKernelCall(op) &&
         "the operator library was asked to time a sub-kernel call");
  assert(!asMemAccess(op) &&
         "the operator library was asked to time a memory access");

  const auto *e = matchEntry(advancedEntries, entries, op);
  if (!e) {
    // Matching nothing is ordinary: a constant, a yield terminator, or
    // `affine.apply` cost nothing real here (apply takes the default row's
    // delay, matching `combDelay(CombOpKindEnum::Apply)`). A float->float
    // arith op reaching here would miscompile at latency 0, so assert
    // instead; extend `classify()`/`needsIP()` (`validateDatapath` repeats
    // this check for a release build).
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

  // Every row is characterized over one parameter, an operand width.
  int64_t width = combParamWidth(op);

  // The stable Problem::OperatorType key: an IP row's symbol, a comb row's
  // `comb.<kind>.w<N>`, else `default`.
  OperatorChar c;
  c.timing.typeName =
      !e->symbol.empty() ? e->symbol
      : e->comb
          ? ("comb." + stringifyOpKindEnum(e->kind) + ".w" + Twine(width)).str()
          : std::string("default");
  c.timing.latency = e->latency;
  // A comb row carries its MARGINAL delay: what the operator adds to a path
  // that already left a register. The fabric floor the measurement also saw is
  // paid once per CYCLE, so it comes off the chaining BUDGET instead
  // (`runSDCScheduler`); charging it per operator costs a four-deep chain three
  // floors it does not spend.
  //
  // The two are the same number here because they must be:
  // `ChainingProblem::checkDelays` rejects a zero-latency operator whose
  // incoming and outgoing delays differ, since for a combinational cell they
  // describe one path.
  if (e->comb)
    c.timing.inDelay = c.timing.outDelay =
        std::max(0.0, e->delay.evaluate(width) - regFloor);
  else {
    c.timing.inDelay = e->inDelay;
    c.timing.outDelay = e->outDelay;
  }
  c.pipelined = e->pipelined;
  // A shift by a literal is wiring, not a shifter. Its own type name because
  // the problem registers timing per NAME: leaving it on the shift row would
  // make the two spellings of that row disagree, and the last one populated
  // would win for both.
  if (isBitRename(op)) {
    c.timing.typeName = "rename." + c.timing.typeName;
    c.timing.inDelay = c.timing.outDelay = 0.0;
  }
  // An IP's signature pins the width, so there the factors are constants and
  // this is the measured core.
  c.price = priceOf(e->uses, {width});
  // The realization is the row's own symbol when it is an IP, else the native
  // lowering the reifier picks; the default row reaches the comb arm too.
  if (!e->symbol.empty())
    c.identity = identityOf(op, std::nullopt, e->symbol);
  else
    c.identity = identityOf(op, combKindOf(op), "");
  return c;
}

std::string OperatorIdentity::key() const {
  std::string s = realizationName().str();
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
  return identityOf(comp, comp.getCombKind(),
                    comp.getOpType().value_or(StringRef()));
}

OperatorIdentity mlir::allo::operatorIdentity(Operation *op,
                                              const OperatorLibrary &lib) {
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return operatorIdentity(comp);
  return lib.lookup(op).identity;
}

bool OperatorLibrary::requiresUnmatchedIP(Operation *op) const {
  return needsIP(op) && matchEntry(advancedEntries, entries, op) == nullptr;
}

bool OperatorLibrary::hasDirectRealization(Operation *op) const {
  return matchEntry(advancedEntries, entries, op) != nullptr;
}
