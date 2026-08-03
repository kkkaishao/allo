/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/Primitives.h"

#include "allo-c/Schedule.h"       // kMemoryInitAttr
#include "allo/Microarch/Naming.h" // regionSignal

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpIPredicate
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Shared free helpers.
//===----------------------------------------------------------------------===//

IntegerType hwType(Type t, OpBuilder &b) {
  return b.getIntegerType(hwWidth(t));
}

IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b) {
  return hwType(cast<MemRefType>(m.memref.getType()).getElementType(), b);
}

Value resize(OpBuilder &b, Location loc, Value v, unsigned width,
             bool isSigned) {
  auto want = b.getIntegerType(width);
  unsigned have = cast<IntegerType>(v.getType()).getWidth();
  if (have == width)
    return v;
  if (have > width)
    return comb::ExtractOp::create(b, loc, want, v, 0).getResult();
  return isSigned ? comb::createOrFoldSExt(b, loc, v, want)
                  : comb::createZExt(b, loc, v, width);
}

unsigned declaredDepth(unsigned words) { return std::max(2u, words); }

SmallVector<APInt> initWords(ElementsAttr init, unsigned width,
                             unsigned depth) {
  SmallVector<APInt> words;
  if (isa<FloatType>(init.getElementType()))
    for (const APFloat &v : init.getValues<APFloat>())
      words.push_back(v.bitcastToAPInt().zextOrTrunc(width));
  else
    for (const APInt &v : init.getValues<APInt>())
      words.push_back(v.zextOrTrunc(width));
  assert(words.size() <= depth &&
         "an array's declared depth must cover its initializer");
  words.resize(depth, APInt(width, 0));
  return words;
}

void recordMemoryInit(seq::HLMemOp mem, ArrayRef<APInt> words) {
  Type elemTy = mem.getMemType().getElementType();
  SmallVector<Attribute> vals;
  for (const APInt &w : words)
    vals.push_back(IntegerAttr::get(elemTy, w));
  mem->setAttr(kMemoryInitAttr, ArrayAttr::get(mem.getContext(), vals));
}

// Integer/logic mnemonics EmitHW lowers to a native `comb` primitive. The
// single source of truth for `emitCompute`'s coverage; a native op outside this
// set has no EmitHW lowering (realizability errors, see `emitModule`).
bool combEmitted(StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases({"addi", "subi", "muli", "andi", "ori", "xori"}, true)
      .Cases({"extsi", "extui", "trunci", "index_cast", "index_castui"}, true)
      .Cases({"cmpi", "select", "shli", "shrsi", "shrui"}, true)
      .Cases({"divsi", "divui", "remsi", "remui"}, true)
      .Cases({"minsi", "maxsi", "minui", "maxui"}, true)
      .Cases({"apply", "negf"}, true)
      .Default(false);
}

// arith and comb name the same ten integer-compare predicates; map across the
// two enums (comb adds 4-state predicates that are never produced here).
static comb::ICmpPredicate combICmpPredicate(arith::CmpIPredicate p) {
  using A = arith::CmpIPredicate;
  using C = comb::ICmpPredicate;
  switch (p) {
  case A::eq:
    return C::eq;
  case A::ne:
    return C::ne;
  case A::slt:
    return C::slt;
  case A::sle:
    return C::sle;
  case A::sgt:
    return C::sgt;
  case A::sge:
    return C::sge;
  case A::ult:
    return C::ult;
  case A::ule:
    return C::ule;
  case A::ugt:
    return C::ugt;
  case A::uge:
    return C::uge;
  }
  llvm_unreachable("unknown arith::CmpIPredicate");
}

Value emitCompute(OpBuilder &b, Location loc, StringRef kind,
                  ValueRange operands, Type resultType, Operation *srcOp) {
  // affine.apply: a map carried on the op (like arith.cmpi's predicate),
  // left by loop-canonicalization when reading an IV outside an address.
  // Uses evalAffine, so a power-of-two divisor stays shift+mask.
  if (kind == "apply") {
    assert(srcOp->getAttr("map") &&
           "dcp.compute<apply> must carry the original affine map");
    AffineMap map = cast<AffineMapAttr>(srcOp->getAttr("map")).getValue();
    assert(map.getNumResults() == 1 && "affine.apply yields one result");
    return evalAffine(b, loc, map.getResult(0), operands, map.getNumDims());
  }
  Value lhs = operands[0];
  // Width-changing unary casts (the widened-reduction idiom
  // trunc(add(ext,ext))) resize operand[0] via comb sign/zero-extend or a
  // low-bit extract; 0-latency, so they slot into the schedule like any comb.
  if (kind == "extsi")
    return comb::createOrFoldSExt(b, loc, lhs, resultType);
  if (kind == "extui")
    return comb::createZExt(b, loc, lhs,
                            cast<IntegerType>(resultType).getWidth());
  if (kind == "trunci")
    return comb::ExtractOp::create(b, loc, resultType, lhs, 0).getResult();
  if (kind == "index_cast")
    return resize(b, loc, lhs, cast<IntegerType>(resultType).getWidth(),
                  /*isSigned=*/true);
  // Float negate: arith.negf flips the sign bit of the float, which rides as
  // its integer bit pattern here, so this is a single XOR, no IP. Unary, so
  // it precedes the `rhs = operands[1]` read below.
  if (kind == "negf") {
    unsigned w = cast<IntegerType>(resultType).getWidth();
    // The mask is the width's signed minimum: exactly the top bit set, at any
    // width. Built as an APInt rather than `1 << (w-1)`, which shifts into the
    // sign bit of an int64 at w == 64 (UB before C++20) and past it beyond.
    Value signBit = hw::ConstantOp::create(
        b, loc, IntegerAttr::get(resultType, APInt::getSignedMinValue(w)));
    return comb::XorOp::create(b, loc, lhs, signBit, false)->getResult(0);
  }
  // 3-input value mux: arith.select(cond, t, f) == comb.mux (cond ? t : f). The
  // if-conversion of a guarded store lowers to this over the two speculated
  // values, so it must be native (no binary-IP shape).
  if (kind == "select")
    return comb::MuxOp::create(b, loc, operands[0], operands[1], operands[2])
        ->getResult(0);
  // Width-preserving binary integer/logic ops.
  Value rhs = operands[1];
  if (kind == "addi")
    return comb::AddOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "subi")
    return comb::SubOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "muli")
    return comb::MulOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "andi")
    return comb::AndOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "ori")
    return comb::OrOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "xori")
    return comb::XorOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "shli")
    return comb::ShlOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "shrsi")
    return comb::ShrSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "shrui")
    return comb::ShrUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  // Signed / unsigned divide, emitted for a flattened guard's delinearization
  // (an affine `i floordiv N` in the predicate lowers to signed-divide over
  // the coalesced counter); a scheduled data divide is multi-cycle IP instead.
  if (kind == "divsi")
    return comb::DivSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "divui")
    return comb::DivUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  // Signed / unsigned remainder (int rem is combinational under the operator
  // model). Both operands share the result width.
  if (kind == "remsi")
    return comb::ModSOp::create(b, loc, lhs, rhs, false)->getResult(0);
  if (kind == "remui")
    return comb::ModUOp::create(b, loc, lhs, rhs, false)->getResult(0);
  // Integer min/max: a compare feeds a mux (canonicalize folds a
  // `select(a<b,a,b)` idiom into arith.minsi/maxsi/minui/maxui).
  auto minmax = [&](comb::ICmpPredicate p) -> Value {
    Value c = comb::ICmpOp::create(b, loc, p, lhs, rhs, false)->getResult(0);
    return comb::MuxOp::create(b, loc, c, lhs, rhs)->getResult(0);
  };
  if (kind == "minsi")
    return minmax(comb::ICmpPredicate::slt);
  if (kind == "maxsi")
    return minmax(comb::ICmpPredicate::sgt);
  if (kind == "minui")
    return minmax(comb::ICmpPredicate::ult);
  if (kind == "maxui")
    return minmax(comb::ICmpPredicate::ugt);
  // Integer compare -> comb.icmp with the predicate carried from arith.cmpi
  // (preserved onto the compute op by convert-schedule-to-dcp).
  if (kind == "cmpi") {
    auto pred =
        cast<arith::CmpIPredicateAttr>(srcOp->getAttr("predicate")).getValue();
    return comb::ICmpOp::create(b, loc, combICmpPredicate(pred), lhs, rhs,
                                false)
        ->getResult(0);
  }
  // Not an `assert`: under NDEBUG that would fall through and hand the caller a
  // null Value to wire into the datapath.
  llvm_unreachable("combEmitted mnemonic without an emitCompute case");
}

//===----------------------------------------------------------------------===//
// EmitContext: the shared builder substrate.
//===----------------------------------------------------------------------===//

Value EmitContext::konst(Type t, int64_t v) {
  return R(hw::ConstantOp::create(b, loc, t, v));
}

Value EmitContext::reg(Value in, Value rstVal) {
  return R(seq::CompRegOp::create(b, loc, in, clk, rst, rstVal));
}

Value EmitContext::enabledReg(Value in, Value ce, Value rstVal) {
  Backedge selfNext = bb.get(in.getType());
  Value self = reg(selfNext, rstVal);
  selfNext.setValue(mux(ce, in, self));
  return self;
}

Value EmitContext::stallHold(Value in, const StallShell &sh) {
  if (!sh)
    return in; // rigid: the address is just the live index
  Backedge heldNext = bb.get(in.getType());
  Value held = reg(heldNext, konst(in.getType(), 0));
  Value out = mux(sh.chainEnable, in, held);
  heldNext.setValue(out);
  return out;
}

Value EmitContext::latchReg(Value init, Value next, Value load, Value advance) {
  Backedge selfNext = bb.get(init.getType());
  Value self = reg(selfNext, konst(init.getType(), 0));
  selfNext.setValue(mux(load, init, mux(advance, next, self)));
  return self;
}

Value EmitContext::mux(Value sel, Value t, Value f) {
  return R(comb::MuxOp::create(b, loc, sel, t, f));
}

ShiftChain EmitContext::shiftChain(Value in, unsigned depth,
                                   const StallShell &sh) {
  ShiftChain chain;
  chain.stages.push_back(in); // stage 0 = the source (a depth-0 tap reads it)
  Value rz = konst(in.getType(), 0);
  Value cur = in;
  for (unsigned s = 1; s <= depth; ++s) {
    // Under an elastic shell every stage advances only while enabled, so all
    // taps freeze together and their "index == cycles delayed" contract still
    // holds under stall; a rigid shell is a plain unconditional shift.
    cur = sh ? enabledReg(cur, sh.chainEnable, rz) : reg(cur, rz);
    chain.stages.push_back(cur);
  }
  return chain;
}

ShiftChain EmitContext::foldedChain(Value in, unsigned depth, unsigned ii,
                                    Value phase, unsigned ready,
                                    const StallShell &sh) {
  assert(ii > 1 && "a fold at II 1 is the plain chain, one register per tap");
  // A stall freezes the phase, so the capture term stays high across it and
  // would otherwise shift the chain once per stalled cycle.
  Value capture = icmpEq(phase, ready % ii);
  Value ce = sh ? andBits(sh.chainEnable, capture) : capture;
  Value rz = konst(in.getType(), 0);
  llvm::SmallVector<Value> held;
  Value cur = in;
  for (unsigned j = 0, n = (depth + ii - 1) / ii; j < n; ++j) {
    cur = enabledReg(cur, ce, rz);
    held.push_back(cur);
  }
  ShiftChain chain;
  chain.stages.push_back(in); // stage 0 = the source, as in a plain chain
  for (unsigned k = 1; k <= depth; ++k)
    chain.stages.push_back(held[(k - 1) / ii]); // register ceil(k / ii)
  return chain;
}

// Above this many cycles a counter (log2(n) registers + a comparator) is
// cheaper than a chain (n registers). Set well clear of ordinary pipeline-stage
// delays so the shape of a small chain, which structural tests read, is left
// alone.
static constexpr unsigned kCountedDelayCycles = 64;

Value EmitContext::delayPulseCounted(Value pulse, unsigned n,
                                     const StallShell &sh) {
  assert(regionSinglePass && "a counted delay drops every pulse but the first, "
                             "so it needs a region that issues one pass");
  assert(n >= 1 && "a zero-cycle delay is the signal itself");
  // `pulse` arms the counter at 0; it counts every advancing cycle and fires
  // at n-1, so the output rises exactly n cycles after the input (a chain
  // tap's contract). Under an elastic shell it counts only while enabled.
  Backedge armedNext = bb.get(i1);
  Backedge countNext = bb.get(i32);
  Value armed =
      sh ? enabledReg(armedNext, sh.chainEnable, f1) : reg(armedNext, f1);
  Value count = sh ? enabledReg(countNext, sh.chainEnable, zero32)
                   : reg(countNext, zero32);
  Value fire = andBits(armed, icmpEq(count, n - 1));
  armedNext.setValue(mux(pulse, t1, mux(fire, f1, armed)));
  countNext.setValue(mux(
      pulse, zero32,
      mux(armed, R(comb::AddOp::create(b, loc, count, one32, false)), count)));
  if (!regionTag.empty()) {
    nameValue(armed, regionSignal(regionTag, "wait" + std::to_string(n)));
    nameValue(count,
              regionSignal(regionTag, "wait" + std::to_string(n) + "_c"));
  }
  return fire;
}

Value EmitContext::delayValid(Value sig, unsigned n, const StallShell &sh) {
  if (n >= kCountedDelayCycles && regionSinglePass)
    return delayPulseCounted(sig, n, sh);
  ShiftChain chain = shiftChain(sig, n, sh);
  // The densest cluster of otherwise-anonymous state in a pipelined region.
  // Label each stage with the cycle it is valid at, so a waveform reads
  // `r1_v3`: region 1, three cycles after issue.
  for (auto [k, stage] : llvm::enumerate(chain.stages))
    if (k && !regionTag.empty())
      nameValue(stage, regionSignal(regionTag, "v" + std::to_string(k)));
  return chain.last();
}

Value EmitContext::activationPulse(Value pulse, Operation *op,
                                   const StallShell &sh) {
  return delayValid(pulse, dcpStart(op), sh);
}

Value EmitContext::icmpEq(Value a, int64_t cst) {
  return R(comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, a,
                                konst(i32, cst), false));
}

Value EmitContext::icmpEqV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, lhs, rhs, false));
}

Value EmitContext::icmpUgeV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::uge, lhs, rhs, false));
}
Value EmitContext::icmpSgeV(Value lhs, Value rhs) {
  return R(
      comb::ICmpOp::create(b, loc, comb::ICmpPredicate::sge, lhs, rhs, false));
}

Value EmitContext::isNonZero(Value v) {
  return R(comb::ICmpOp::create(b, loc, comb::ICmpPredicate::ne, v,
                                konst(v.getType(), 0), false));
}

Value EmitContext::notBit(Value v) {
  return R(comb::XorOp::create(b, loc, v, t1, false));
}

Value EmitContext::andBits(Value lhs, Value rhs) {
  return R(comb::AndOp::create(b, loc, lhs, rhs, false));
}

Value EmitContext::orBits(Value lhs, Value rhs) {
  return R(comb::OrOp::create(b, loc, lhs, rhs, false));
}

Value EmitContext::risingEdge(Value level) {
  Value prev = reg(level, f1);
  return R(comb::AndOp::create(
      b, loc, level, R(comb::XorOp::create(b, loc, prev, t1, false)), false));
}

Value EmitContext::startFor(Value regionStart, ArrayRef<Value> predDones) {
  if (predDones.empty())
    return regionStart;
  Value ready = predDones.front();
  for (Value d : predDones.drop_front())
    ready = andBits(ready, d);
  return risingEdge(ready);
}

Value EmitContext::holdDone(Value setPulse, Value start) {
  circt::Backedge doneNext = bb.get(i1);
  Value done = reg(doneNext, f1);
  doneNext.setValue(mux(start, f1, mux(setPulse, t1, done)));
  return done;
}

Value EmitContext::completedSince(Value level, Value passStart) {
  Value edge = risingEdge(level);
  return andBits(orBits(holdDone(edge, passStart), edge), notBit(passStart));
}

std::pair<Value, Value> EmitContext::branchPulse(Value when, Value cond) {
  return {andBits(when, cond), andBits(when, notBit(cond))};
}

void EmitContext::initLiterals() {
  zero32 = konst(i32, 0);
  one32 = konst(i32, 1);
  f1 = konst(i1, 0);
  t1 = konst(i1, 1);
}

} // namespace mlir::allo::uarch
