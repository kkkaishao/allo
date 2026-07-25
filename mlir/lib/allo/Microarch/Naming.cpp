/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Naming.h"
#include "allo/Microarch/HWEmitter.h"      // externalBank (banked port set)
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid (the SV keyword set)
#include "circt/Dialect/Seq/SeqOps.h"   // seq::CompRegOp (the reg name channel)

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpFPredicate (IP module name)

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

namespace {
// The suffix vocabulary: the one place these strings are spelled.
constexpr llvm::StringLiteral kAddr = "_addr";
constexpr llvm::StringLiteral kData = "_data";
constexpr llvm::StringLiteral kWe = "_we";
constexpr llvm::StringLiteral kValid = "_valid";
constexpr llvm::StringLiteral kReady = "_ready";
constexpr llvm::StringLiteral kRead = "_rd";
constexpr llvm::StringLiteral kWrite = "_wr";
constexpr llvm::StringLiteral kStream = "_st";
constexpr llvm::StringLiteral kIn = "_in";
constexpr llvm::StringLiteral kOut = "_out";
constexpr llvm::StringLiteral kBank = "_b";

std::string join(llvm::StringRef base, llvm::StringLiteral suffix) {
  return base.str() + suffix.str();
}
} // namespace

std::string verilogName(llvm::StringRef name) {
  std::string s = sanitizeCppIdentifier(name); // charset + leading digit
  // ExportVerilog renames a reserved word (`input` -> `input_0`), desyncing the
  // manifest from the Verilog, so escape it here instead. One '_' suffices
  // since no keyword ends in one; the loop states the invariant.
  while (!sv::isNameValid(s, /*caseInsensitiveKeywords=*/false))
    s.push_back('_');
  return s;
}

//===----------------------------------------------------------------------===//
// Owner tokens.
//===----------------------------------------------------------------------===//

std::string argOwner(unsigned argNo) { return "a" + std::to_string(argNo); }
std::string memOwner(MemId m) { return "m" + std::to_string(m); }
std::string unitOwner(UnitId u) { return "u" + std::to_string(u); }
std::string chanOwner(StreamId s) { return "ch" + std::to_string(s); }
std::string regOwner(RegId r) { return "reg" + std::to_string(r); }

std::string ownerOf(Location loc, llvm::StringRef fallback) {
  // Charset only: the keyword escape belongs to the composed name, so an
  // `input` array yields `input_rd0` rather than `input__rd0`.
  if (auto name = nameFromLoc(loc))
    return sanitizeCppIdentifier(*name);
  return fallback.str();
}

std::string ownerOf(Value v, llvm::StringRef fallback) {
  if (auto name = nameFromLoc(v.getLoc()))
    return sanitizeCppIdentifier(*name);
  // An unnamed value keys on its own identity, the argument position, never
  // on where its port lands in the port list.
  if (auto ba = dyn_cast<BlockArgument>(v))
    return argOwner(ba.getArgNumber());
  return fallback.str();
}

std::string uniqueOwnerOf(Value v, llvm::ArrayRef<Value> siblings,
                          llvm::StringRef fallback) {
  std::string own = ownerOf(v, fallback);
  unsigned ties = 0;
  for (Value s : siblings)
    ties += ownerOf(s, fallback) == own;
  if (ties <= 1)
    return own;
  // Two arguments carry the same source name, which would give their port
  // groups one set of names. The colliding group falls back to its own
  // argument position.
  auto ba = dyn_cast<BlockArgument>(v);
  return ba ? own + "_" + argOwner(ba.getArgNumber()) : own;
}

//===----------------------------------------------------------------------===//
// Fields and bases.
//===----------------------------------------------------------------------===//

std::string portAddr(llvm::StringRef base) { return join(base, kAddr); }
std::string portData(llvm::StringRef base) { return join(base, kData); }
std::string portWe(llvm::StringRef base) { return join(base, kWe); }
std::string portValid(llvm::StringRef base) { return join(base, kValid); }
std::string portReady(llvm::StringRef base) { return join(base, kReady); }

std::string memBase(llvm::StringRef owner, bool write, unsigned group) {
  return verilogName(join(owner, write ? kWrite : kRead) +
                     std::to_string(group));
}
std::string streamBase(llvm::StringRef owner) {
  return verilogName(join(owner, kStream));
}
std::string scalarBase(llvm::StringRef owner) {
  return verilogName(join(owner, kIn));
}
std::string resultBase(llvm::StringRef owner) {
  return verilogName(join(owner, kOut));
}
std::string bankBase(llvm::StringRef base, unsigned bank) {
  return verilogName(join(base, kBank) + std::to_string(bank));
}

// The memory owners of a datapath, for the same-source-name tie-break.
static llvm::SmallVector<Value> memRefs(const Datapath &dp) {
  llvm::SmallVector<Value> vs;
  for (const MemUnit &m : dp.mems)
    vs.push_back(m.memref);
  return vs;
}

std::string memPortBase(const Datapath &dp, llvm::ArrayRef<AccRef> ports,
                        unsigned i, bool write) {
  const MemUnit &mu = dp.mems[ports[i].mem];
  // Which access of this argument this is: the group index is per (argument,
  // role), so adding an access to another argument never renumbers it.
  unsigned group = 0;
  for (unsigned j = 0; j < i; ++j)
    group += ports[j].mem == ports[i].mem;
  return memBase(uniqueOwnerOf(mu.memref, memRefs(dp), memOwner(mu.id)), write,
                 group);
}

std::string memBoundaryPortBase(const Datapath &dp, MemId mem, bool write,
                                unsigned group) {
  const MemUnit &mu = dp.mems[mem];
  return memBase(uniqueOwnerOf(mu.memref, memRefs(dp), memOwner(mu.id)), write,
                 group);
}

llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const Datapath &dp, llvm::ArrayRef<AccRef> ports, unsigned i,
         bool write) {
  const MemUnit &m = dp.mems[ports[i].mem];
  ExternalBanking eb = externalBank(m, m.accesses[ports[i].idx]);
  std::string base = memPortBase(dp, ports, i, write);
  if (eb.factor == 1)
    return {{0u, base}};
  if (eb.bank)
    return {{*eb.bank, base}}; // statically routed to one interface
  // Data-dependent: one interface per bank (the crossbar drives every bank).
  llvm::SmallVector<std::pair<unsigned, std::string>> all;
  for (unsigned k = 0; k < eb.factor; ++k)
    all.push_back({k, bankBase(base, k)});
  return all;
}

std::string streamPortBase(const Datapath &dp, const StreamChannel &s) {
  auto own = [](const StreamChannel &c) {
    return streamBase(ownerOf(c.stream, chanOwner(c.id)));
  };
  std::string base = own(s);
  // Count the siblings this name ties with. A tie gives two handshakes one set
  // of port names, which ExportVerilog uniquifies, desyncing the manifest and
  // collapsing the by-name instance wiring.
  unsigned sameBase = 0, sameDir = 0;
  for (const StreamChannel &o : dp.streams) {
    if (own(o) != base)
      continue;
    ++sameBase;
    sameDir += o.isInput == s.isInput;
  }
  if (sameBase == 1)
    return base;
  base += (s.isInput ? kIn : kOut).str(); // the systolic shape: a get, a put
  return sameDir == 1 ? base : base + "_s" + std::to_string(s.id);
}

std::string scalarPortName(const Datapath &dp, const IOPort &io) {
  llvm::SmallVector<Value> siblings;
  for (const IOPort &o : dp.ios)
    if (o.isInput)
      siblings.push_back(o.value);
  return scalarBase(
      uniqueOwnerOf(io.value, siblings, "s" + std::to_string(io.id)));
}

std::string resultPortName(unsigned i, unsigned n) {
  // A result is 1:1 with the source signature, so it carries an index only
  // when the signature itself declares several.
  return resultBase(n == 1 ? "ret" : "ret" + std::to_string(i));
}

//===----------------------------------------------------------------------===//
// Internal cells.
//===----------------------------------------------------------------------===//

std::string memCellName(const Datapath &dp, const MemUnit &m, unsigned bank) {
  // The only name with no role suffix, so it escapes itself: a buffer named
  // `buf` collides with the Verilog gate primitive.
  std::string base = uniqueOwnerOf(m.memref, memRefs(dp), memOwner(m.id));
  return m.numBanks > 1 ? bankBase(base, bank) : verilogName(base);
}

std::string regionSignal(llvm::StringRef tag, llvm::StringRef sig) {
  return verilogName(tag.str() + "_" + sig.str());
}

std::string regionSignal(unsigned region, llvm::StringRef sig) {
  return regionSignal("r" + std::to_string(region), sig);
}

std::string regTapName(llvm::StringRef owner, unsigned k) {
  return verilogName(owner.str() + "_d" + std::to_string(k));
}

std::string survivorName(unsigned region, unsigned k) {
  return regionSignal(region, "sv" + std::to_string(k));
}

std::string unitInstanceName(const FuncUnit &u) {
  std::string own = ownerOf(u.boundOps.front().first->getLoc(), "");
  return verilogName(own.empty() ? unitOwner(u.id)
                                 : own + "_" + unitOwner(u.id));
}

std::string childInstanceName(llvm::StringRef callee, unsigned n) {
  return verilogName(callee.str() + "_i" + std::to_string(n));
}

std::string channelSignal(llvm::StringRef chan, llvm::StringRef sig) {
  return verilogName(chan.str() + "_" + sig.str());
}

std::string operatorPredicate(const FuncUnit &u) {
  // A compare is the only IP carrying a `predicate` attr, copied onto the op
  // by the reifier. Integer compare is combinational, so an IP compare is
  // always floating-point.
  if (auto pred =
          u.boundOps.front().first->getAttrOfType<arith::CmpFPredicateAttr>(
              "predicate"))
    return arith::stringifyCmpFPredicate(pred.getValue()).str();
  return "";
}

std::string operatorModuleName(const FuncUnit &u) {
  std::string pred = operatorPredicate(u);
  return pred.empty() ? u.impl : u.impl + "_" + pred;
}

void nameValue(Value v, llvm::StringRef name) {
  if (name.empty())
    return;
  Operation *op = v.getDefiningOp();
  if (!op) // a block argument / unresolved backedge is named elsewhere
    return;
  // Pick the channel ExportVerilog reads: a register names from its own `name`
  // attr, since sv.namehint is ignored on a reg; any other value uses namehint.
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    reg.setNameAttr(StringAttr::get(op->getContext(), name));
  else
    op->setAttr("sv.namehint", StringAttr::get(op->getContext(), name));
}

void nameValue(Value v, Location loc) {
  if (auto name = nameFromLoc(loc))
    nameValue(v, sanitizeCppIdentifier(*name));
}

} // namespace mlir::allo::uarch
