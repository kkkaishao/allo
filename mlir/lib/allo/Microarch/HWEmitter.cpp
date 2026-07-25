/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/IR/AlloOps.h"
#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/Interface.h"
#include "allo/Scheduling/OperatorLibrary.h" // stallContract
#include "allo/Support/Logging.h"
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h" // sv::isNameValid (the SV keyword set)
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpIPredicate (cmpi predicate)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GlobalOp (constant ROM)
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt;

#define DEBUG_TYPE "hw-emitter"

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Shared free helpers.
//===----------------------------------------------------------------------===//

IntegerType hwType(Type t, OpBuilder &b) {
  if (isa<IndexType>(t))
    return b.getIntegerType(32);
  if (auto ft = dyn_cast<FloatType>(t))
    return b.getIntegerType(ft.getWidth());
  return cast<IntegerType>(t);
}

IntegerType memElemType(const uarch::MemUnit &m, OpBuilder &b) {
  return hwType(cast<MemRefType>(m.memref.getType()).getElementType(), b);
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
  // left by flatten-perfect-loops when reading an IV outside an address.
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
  if (kind == "index_cast") {
    // index <-> integer: both carried at their hw integer width (hwType maps
    // index to i32), so this is a signed resize to the result width: sExt,
    // low-bit extract, or identity when the widths already match.
    unsigned dst = cast<IntegerType>(resultType).getWidth();
    unsigned src = cast<IntegerType>(lhs.getType()).getWidth();
    if (dst == src)
      return lhs;
    return dst > src ? comb::createOrFoldSExt(b, loc, lhs, resultType)
                     : comb::ExtractOp::create(b, loc, resultType, lhs, 0)
                           .getResult();
  }
  // Float negate: arith.negf flips the sign bit of the float, which rides as
  // its integer bit pattern here, so this is a single XOR, no IP. Unary, so
  // it precedes the `rhs = operands[1]` read below.
  if (kind == "negf") {
    unsigned w = cast<IntegerType>(resultType).getWidth();
    // The sign-bit mask is built as an int64 (`1 << (w-1)`), so a float wider
    // than 64 bits (f80/f128) would shift by >= 64 (UB) and cannot carry the
    // pattern; such a type needs an APInt mask.
    assert(w <= 64 &&
           "negf sign-bit mask uses int64 (1 << (w-1)); a float wider "
           "than 64 bits needs an APInt bit pattern");
    Value signBit = hw::ConstantOp::create(b, loc, resultType,
                                           static_cast<int64_t>(1) << (w - 1));
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
  assert(false && "combEmitted mnemonic without an emitCompute case");
  return {};
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

Value EmitContext::stallHold(Value in) {
  if (!regionEnable)
    return in; // no stall shell: the address is just the live index
  Backedge heldNext = bb.get(in.getType());
  Value held = reg(heldNext, konst(in.getType(), 0));
  Value out = mux(regionEnable, in, held);
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

ShiftChain EmitContext::shiftChain(Value in, unsigned depth) {
  ShiftChain chain;
  chain.stages.push_back(in); // stage 0 = the source (a depth-0 tap reads it)
  Value rz = konst(in.getType(), 0);
  Value cur = in;
  for (unsigned s = 1; s <= depth; ++s) {
    // In a stall shell (regionEnable set) every stage advances only while
    // enabled, so all taps freeze together and their "index == cycles delayed"
    // contract still holds under stall; otherwise a plain unconditional shift.
    cur = regionEnable ? enabledReg(cur, regionEnable, rz) : reg(cur, rz);
    chain.stages.push_back(cur);
  }
  return chain;
}

Value EmitContext::delayValid(Value sig, unsigned n) {
  ShiftChain chain = shiftChain(sig, n);
  // The densest cluster of otherwise-anonymous state in a pipelined region.
  // Label each stage with the cycle it is valid at, so a waveform reads
  // `r1_v3`: region 1, three cycles after issue.
  for (auto [k, stage] : llvm::enumerate(chain.stages))
    if (k && !regionTag.empty())
      nameValue(stage, regionSignal(regionTag, "v" + std::to_string(k)));
  return chain.last();
}

Value EmitContext::activationPulse(Value pulse, Operation *op) {
  return delayValid(pulse, dcpStart(op));
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

std::pair<Value, Value> EmitContext::branchPulse(Value when, Value cond) {
  return {andBits(when, cond), andBits(when, notBit(cond))};
}

void EmitContext::initLiterals() {
  zero32 = konst(i32, 0);
  one32 = konst(i32, 1);
  f1 = konst(i1, 0);
  t1 = konst(i1, 1);
}

//===----------------------------------------------------------------------===//
// HWEmitter: the orchestrator.
//===----------------------------------------------------------------------===//

// The counted induction bounds (lb/ub/step) of region \p rb: the IV runs
// `lb, lb+step, ...` and terminates on `iv+step >= ub`. Each of lb/step is a
// resolved runtime Source (a data-dependent range start/stride) or a constant.
// ub is a resolved runtime count (`ubSource`, a dynamic trip) or, for a
// constant trip K, `lb + K*step`: a konst when lb/step are compile-time, else
// a datapath Value tracking the runtime lb/step (the `range(i, i+K)` window).
// Empty (default) for an acyclic region (no counter) or a while (which builds
// its own Terminator::conditional from the resolved condition).
Terminator HWEmitter::terminatorOf(const uarch::RegionBlock &rb) {
  // Counts up via SIGNED compares (isLast/isEmpty), so a negative lower bound
  // is fine; a non-positive/decreasing step is unsupported (frontend-rejected,
  // so this is a dormant backstop) and a runtime step's sign goes unchecked.
  assert((rb.stepSource || rb.step > 0) &&
         "counted-loop counter is up-counting; a non-positive/decreasing step "
         "is unsupported (the frontend rejects it)");
  auto bound = [&](const uarch::Source &s, int64_t c) {
    return s ? datapath.resolveSource(s) : ctx.konst(ctx.i32, c);
  };
  Value lb = bound(rb.lbSource, rb.lb), step = bound(rb.stepSource, rb.step);
  if (rb.ubSource)
    return Terminator::counted(lb, datapath.resolveSource(rb.ubSource), step,
                               /*dynamic=*/true);
  if (rb.tripCount) {
    // Constant trip K: ub = lb + K*step. Compile-time lb/step fold to a
    // constant; runtime lb/step (the fixed-window idiom `for j in
    // range(i, i+K)`) resolve it the same way, giving a stable `ub`.
    int64_t trip = *rb.tripCount;
    Value ub;
    if (!rb.lbSource && !rb.stepSource) {
      ub = ctx.konst(ctx.i32, rb.lb + trip * rb.step);
    } else {
      Value span = rb.stepSource
                       ? ctx.R(comb::MulOp::create(ctx.b, ctx.loc, step,
                                                   ctx.konst(ctx.i32, trip),
                                                   /*twoState=*/false))
                       : ctx.konst(ctx.i32, trip * rb.step);
      ub = ctx.R(
          comb::AddOp::create(ctx.b, ctx.loc, lb, span, /*twoState=*/false));
    }
    return Terminator::counted(lb, ub, step, /*dynamic=*/false);
  }
  return {};
}

// One imperative path for every leaf region (counted / dynamic-trip / while):
// control -> datapath -> resolve the F->G condition, capture results, done. The
// regimes differ only in the Terminator and the survivor mechanism (see
// captureResults); the shared skeleton reads as a linear sequence.
Value HWEmitter::emitRegion(const uarch::RegionBlock &rb, Value start,
                            bool retrig) {
  RegionTag tag(ctx, rb.id); // naming scope for this region's pipeline cells
  if (!rb.children.empty()) {
    if (rb.guard)
      return emitGuard(rb, start);
    return rb.conditional ? emitConditionalContainer(rb, start)
                          : emitContainer(rb, start);
  }
  assert(!rb.guard && "a guard region has no children to predicate");

  // A loop-over-call region (a counted `dcp.pipeline` wrapping one CallUnit)
  // runs a dedicated done-driven controller: one child fired per iteration,
  // advancing on its real `done`, not the per-cycle pipeline cadence.
  if (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.callUnits.empty())
    return emitLoopCall(rb, start);

  // Control: the terminator + the control skeleton. A while's
  // continue-condition is a datapath value not emitted yet, so it rides a
  // backedge resolved after the datapath; a counted loop's bound resolves now.
  Backedge condBE;
  Terminator term;
  if (rb.conditional) {
    condBE = ctx.bb.get(ctx.i1);
    term = Terminator::conditional(condBE, ctx.zero32, ctx.one32);
  } else {
    term = terminatorOf(rb);
  }

  // A region with stream accesses gets two backedges resolved after the
  // datapath: `chainEnable` freezes shift chains/done-drain on back-pressure;
  // `issueEnable` also requires inputs-available, so empty input bubbles.
  bool hasStream = false;
  for (const uarch::StreamChannel &s : dp.streams)
    for (const uarch::StreamChannel::Access &acc : s.accesses)
      hasStream |= acc.region == rb.id;
  Backedge chainEnableBE, issueEnableBE;
  Value enable = ctx.t1;
  if (hasStream) {
    chainEnableBE = ctx.bb.get(ctx.i1);
    issueEnableBE = ctx.bb.get(ctx.i1);
    ctx.regionEnable = chainEnableBE;
    enable = issueEnableBE;
  }

  auto rc = control.emitPipelineControl(rb, term, start, enable);
  datapath.setControl(rb.id, rc); // seam G -> F (counter + issue)

  // Datapath -> feedback (store drain + shell signals; a while's condition
  // and next-value producers are now emitted). Shell backedges resolve last,
  // after the done drain reads ctx.regionEnable, so setValue still RAUWs it.
  auto fb = datapath.emit(rb, rc.issue);

  // Resolve the F->G condition backedge now the datapath has emitted it, and
  // re-point the terminator: `setValue` RAUWs and erases the placeholder, so
  // a later `term.cond` read (lastIssuePulse's exit test) needs the real value.
  if (rb.conditional) {
    Value cond =
        datapath.resolveSource(dp.carryInfo.find(rb.id)->second.condition);
    condBE.setValue(cond);
    term.cond = cond;
  }

  // Survivors: capture the region's results (returning their drain stage) and
  // pin the last iteration's issue pulse, the one pulse both done and the
  // captures share.
  Value lastIssue = lastIssuePulse(rc, term);
  unsigned resultDrain = captureResults(rb, rc, lastIssue, start);
  unsigned drainStage = std::max(fb.storeDrain, resultDrain);

  // A counted leaf that is empty (lb >= ub) issues nothing, so it completes
  // on `start` via `emptyDone`, delayed one cycle so the pulse doesn't land
  // on `start` itself (`done` is a level; retrigger needs a real 0->1 edge).
  Value emptyDone =
      (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional)
          ? ctx.delayValid(ctx.andBits(start, term.isEmpty(ctx)), 1)
          : Value();
  // emitDone's drain chain must still see the shell's enable; leave it after.
  // A CallUnit region completes on the child's real `done` (fb.callDone):
  // correct for a determinate child and the only option for an indeterminate
  // one, since a call region has no parent-issued stores to drain.
  Value done = fb.callDone ? fb.callDone
                           : control.emitDone(rb.id, drainStage, lastIssue,
                                              emptyDone, start, retrig);
  // Every use of the shell signals (datapath + done drain) is now emitted, so
  // resolving the backedges here RAUWs them all.
  if (hasStream) {
    assert(fb.chainEnable && fb.issueEnable &&
           "a stream region must produce the shell signals");
    chainEnableBE.setValue(fb.chainEnable);
    issueEnableBE.setValue(fb.issueEnable);
  }
  ctx.regionEnable = Value();
  return done;
}

// The final iteration's issue pulse: a counted region's last iteration (the
// next induction value iv+step reaches the bound) or a while's exit (its
// condition false); an acyclic region has no counter, so its single
// (registered) issue pulse is itself the last. The one pulse the done
// (emitDone) and the survivor captures key off.
Value HWEmitter::lastIssuePulse(const RegionControl &rc,
                                const Terminator &term) {
  if (!rc.counter)
    return rc.issue; // acyclic: a single pass
  Value ivStep =
      ctx.R(comb::AddOp::create(ctx.b, ctx.loc, rc.counter, term.step, false));
  return ctx.andBits(rc.issue, term.isLast(ctx, ivStep));
}

unsigned HWEmitter::captureResults(const uarch::RegionBlock &rb,
                                   const RegionControl &rc, Value lastIssue,
                                   Value start) {
  return rb.conditional ? captureWhileResults(rb, rc, start)
                        : captureCountedResults(rb, lastIssue, start);
}

// Capture each of a result-yielding region's results into its own survivor
// register, on the cycle it lands, while the result is still on its Source (a
// free-running datapath overwrites it once the run ends). Returns the
// LATEST-landing result's stage (its ready cycle after \p lastIssue), which the
// region folds into its `drainStage`: the done then rises with the deepest
// survivor latched, so a sibling that starts on it reads every survivor valid.
// A store-ful region yields no result and returns stage 0 (its store drain
// governs).
unsigned HWEmitter::captureCountedResults(const uarch::RegionBlock &rb,
                                          Value lastIssue, Value start) {
  auto it = dp.regionResult.find(rb.id);
  if (it == dp.regionResult.end())
    return 0;
  auto initIt = dp.regionResultInit.find(rb.id);
  unsigned maxStage = 0;
  for (auto [k, rs] : llvm::enumerate(it->second)) {
    if (rs.kind == uarch::Source::Kind::None)
      continue; // an untracked result: no survivor (asserts if read)
    if (rs.kind == uarch::Source::Kind::Call)
      continue; // a call result: emitCalls set the survivor from the child's
                // held output port (self-timed by `done`), not a static capture
    // Capture the result on the cycle it lands (its ready cycle after the last
    // issue); the region's done drains on the latest-landing result.
    unsigned stage = datapath.readyCycle(rs);
    Value cap = ctx.delayValid(lastIssue, stage);
    Value res = datapath.resolveSource(rs);
    // A loop-carried result preloads its init at `start`, then latches the
    // final value when it lands, so a zero-trip run (cap never fires) keeps
    // the init instead of a stale prior value; an init-less result always
    // lands.
    uarch::Source initSrc =
        initIt != dp.regionResultInit.end() && k < initIt->second.size()
            ? initIt->second[k]
            : uarch::Source{};
    Value survivor =
        initSrc.kind == uarch::Source::Kind::None
            ? ctx.enabledReg(res, cap, ctx.konst(res.getType(), 0))
            : ctx.latchReg(datapath.resolveSource(initSrc), res, start, cap);
    nameValue(survivor, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, survivor);
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// A while region's loop-carried results: each freezes into a latch (loaded with
// its init at \p start, advanced to its next-value while the loop continues,
// held once it exits), so a sibling reads the loop's final value (or the init
// for a zero-iteration loop). Returns the deepest carried-value stage, which
// the region folds into its `drainStage` so the done rises with the deepest
// in-flight survivor latched (0 for a single-stage while). The exit pulse the
// done keys off is the while's `lastIssuePulse` (issue & ~cond).
unsigned HWEmitter::captureWhileResults(const uarch::RegionBlock &rb,
                                        const RegionControl &rc, Value start) {
  const auto &wi = dp.carryInfo.find(rb.id)->second;
  Value cond = datapath.resolveSource(
      wi.condition); // memoized (the resolved continue-condition)
  // A while continues (advances recurrences) on each issued iteration whose
  // condition is true; a next-value produced at a later stage delays its
  // advance pulse to match, reducing to one shared pulse for a stage-0 body.
  Value cont = ctx.andBits(rc.issue, cond);
  unsigned maxStage = 0;
  for (auto [k, nextS] : llvm::enumerate(wi.nexts)) {
    if (nextS.kind == uarch::Source::Kind::None)
      continue; // an untracked carried value: no survivor (asserts if read)
    unsigned stage = datapath.readyCycle(nextS);
    Value advance = ctx.delayValid(cont, stage);
    Value next = datapath.resolveSource(nextS);
    Value init = wi.inits[k].kind == uarch::Source::Kind::None
                     ? ctx.konst(next.getType(), 0)
                     : datapath.resolveSource(wi.inits[k]);
    Value survivor = ctx.latchReg(init, next, start, advance);
    nameValue(survivor, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, survivor);
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// Run `regions` in program order, each region starting when its predecessor
// drains (the first on `start`); returns the last region's done. The shared
// sequencer: func-scope siblings (a single pass) and a container's children
// (once per outer iteration) follow the same "start k+1 when k drains" pattern.
Value HWEmitter::sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                          bool retrig) {
  Value done;
  Value startK = start;
  for (auto [i, rid] : llvm::enumerate(regions)) {
    const auto &rb = dp.regions[rid];
    done = emitRegion(rb, startK, retrig);
    if (i + 1 < regions.size())
      startK = ctx.startFor(/*regionStart=*/Value(), done);
  }
  return done;
}

// Compose the func-scope siblings by their dependence DAG (rb.predecessors): a
// region with no predecessors starts with the kernel `start` (independent
// siblings run concurrently), the rest on the rising edge of their
// predecessors' joined `done`. Emission is in program order, so a predecessor's
// `done` is already built when its consumer reads it. The kernel `done` is the
// conjunction of every region's `done`: it rises when the last region
// completes, which under concurrency need not be the last in program order. A
// pure chain (every region depends on the previous) reproduces `sequence`
// exactly: each start is the rising edge of the prior `done` and the
// conjunction equals the final `done`.
Value HWEmitter::composeSiblings(llvm::ArrayRef<uarch::RegionId> regions,
                                 Value start) {
  llvm::DenseMap<uarch::RegionId, Value> doneOf;
  Value allDone;
  for (uarch::RegionId rid : regions) {
    const auto &rb = dp.regions[rid];
    // No predecessors: run concurrently with the kernel `start`. Otherwise
    // start on the rising edge of the predecessors' joined `done`, which
    // waits for the last producer to complete.
    llvm::SmallVector<Value, 2> predDones;
    for (uarch::RegionId p : rb.predecessors) {
      Value d = doneOf.lookup(p);
      assert(d && "a predecessor's done must be emitted before its consumer");
      predDones.push_back(d);
    }
    Value startK = ctx.startFor(start, predDones);
    Value done = emitRegion(rb, startK, /*retrig=*/true);
    doneOf[rid] = done;
    allDone = allDone ? ctx.andBits(allDone, done) : done;
  }
  return allDone;
}

// Set up a container's loop-carried iter-args as frozen survivor registers:
// each latches its `inits[k]` at `start` and advances to a next-value on
// `advance` (a Backedge resolved after the children emit). Records each as
// Source::Survivor{rb, k}, read by the children's init reads and, for the
// final value, a sibling; returns the per-arg next-value backedges the
// caller sets to `resolveSource(nexts[k])` once the children have produced
// them. Shared by the counted (emitContainer) and conditional
// (emitConditionalContainer) regimes.
SmallVector<circt::Backedge>
HWEmitter::setupCarriedIterArgs(const uarch::RegionBlock &rb,
                                ArrayRef<uarch::Source> inits, Value start,
                                Value advance) {
  SmallVector<circt::Backedge> nextBE;
  for (auto [k, initS] : llvm::enumerate(inits)) {
    assert(initS && "a container iter-arg has no resolvable init");
    Value init = datapath.resolveSource(initS);
    circt::Backedge nb = ctx.bb.get(init.getType());
    nextBE.push_back(nb);
    Value carried = ctx.latchReg(init, nb, start, advance);
    nameValue(carried, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, carried);
  }
  return nextBE;
}

// A loop-over-call region (a counted `dcp.pipeline` wrapping one CallUnit):
// the counter is `rc.counter` (so emitCalls wires the child's index port to
// it via Source::Counter) and the child start is `rc.issue` (the loop-fire
// pulse); the region completes when the last iteration's `done` latches. One
// child instance fires N times, each invocation advancing on its real
// `done`, a held level cleared on its start, so its rising edge marks each
// completion.
Value HWEmitter::emitLoopCall(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id);
  assert(rb.callUnits.size() == 1 && rb.units.empty() && rb.regs.empty() &&
         "a loop-over-call region is one child with no loose datapath");
  assert(rb.tripCount && rb.lb == 0 && rb.step == 1 &&
         "loop-over-call first cut supports a `0 to N step 1` trip");
  int64_t n = *rb.tripCount;
  // A zero-trip loop (`dcp.pipeline 0 to 0`) issues nothing, so it needs the
  // same escape as the leaf/container paths: `N - 1` wraps to all-ones, so
  // the unsigned `more` test never fires. Complete one cycle after `start`.
  const bool empty = n <= 0;
  auto ivType = cast<IntegerType>(datapath.loopIndexPortType(rb));
  auto kconst = [&](int64_t v) { return ctx.konst(ivType, v); };

  // The child `done` rides a backedge resolved after the datapath (emitCalls)
  // drives it; its rising edge is each invocation's completion.
  Backedge doneBE = ctx.bb.get(ctx.i1);
  Value doneEdge = ctx.risingEdge(doneBE);
  // Loop counter: reset to 0 on `start`, +1 the cycle the child completes with
  // iterations still left.
  Backedge kNextBE = ctx.bb.get(ivType);
  Value k = ctx.reg(kNextBE, kconst(0));
  nameValue(k, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                      : rb.counterName);
  // iterations after k (never, for an empty loop, so `advance` stays low and
  // the counter never leaves 0)
  Value more = empty ? ctx.f1 : ctx.notBit(ctx.icmpUgeV(k, kconst(n - 1)));
  Value advance = ctx.andBits(doneEdge, more);
  Value kInc = ctx.R(comb::AddOp::create(ctx.b, ctx.loc, k, kconst(1), false));
  kNextBE.setValue(ctx.mux(start, kconst(0), ctx.mux(advance, kInc, k)));
  // Fire the next iteration one cycle after the done edge, once k has settled.
  Value fireNext = ctx.reg(advance, ctx.f1);
  nameValue(fireNext, regionSignal(rb.id, "fire"));
  // An empty loop never fires the child at all (it is still instantiated; its
  // own run gating keeps every write-enable low, so the arrays stay untouched).
  Value childStart = empty ? ctx.f1 : ctx.orBits(start, fireNext);

  // Datapath: emitCalls wires the single child: start = rc.issue = childStart,
  // index port = resolveSource(Counter) = k, boundary/internal mems mastered
  // as usual.
  RegionControl rc{/*issue=*/childStart, /*counter=*/k, /*wantIssue=*/Value()};
  datapath.setControl(rb.id, rc);
  auto fb = datapath.emit(rb, rc.issue);
  assert(fb.callDone && "a loop-over-call region produced no child done");
  doneBE.setValue(fb.callDone);

  // done: latch the last iteration's completion (a done edge with none left),
  // cleared on `start` so a re-invocation re-arms.
  Value last = empty ? ctx.delayValid(start, 1)
                     : ctx.andBits(doneEdge, ctx.notBit(more));
  Backedge doneHeldBE = ctx.bb.get(ctx.i1);
  Value doneHeld = ctx.reg(doneHeldBE, ctx.f1);
  nameValue(doneHeld, regionSignal(rb.id, "done"));
  doneHeldBE.setValue(ctx.mux(start, ctx.f1, ctx.mux(last, ctx.t1, doneHeld)));
  return doneHeld;
}

// A container region: a cyclic loop whose body nests one or more child regions,
// run once per outer iteration. The outer counter is materialized first (its
// value feeds the children's addressing while they emit); the children are then
// sequenced within one outer iteration (`sequence`), and the outer counter
// advances (restarting child 0) when the LAST child drains, until the trip is
// exhausted. Non-overlapping (II_outer >= sum of child latencies), so the outer
// index is stable across one pass. A value handed child-to-child crosses as a
// survivor register (captured in the producer, read in the consumer). Returns a
// latched completion level.
Value HWEmitter::emitContainer(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id);
  // Induction bounds: compile-time constants, or runtime Sources for a
  // variable-trip container (bound = an enclosing loop's counter or a
  // prologue survivor); a runtime bound terminates on `iv+step >= ub`.
  auto term = terminatorOf(rb);
  // The outer counter is the source IV (init `lb`, advance by `step`); both
  // it and the child start lag the iteration pulse by one cycle, so a child
  // sampling the counter sees the iteration it's starting, not the one just
  // left.
  Backedge ivNext = ctx.bb.get(ctx.i32);
  Value iv = ctx.reg(ivNext, term.lb);
  // A container's counter is materialized here, not by the ControlEmitter, so
  // it needs the same source-IV label its leaf counterpart gets.
  nameValue(iv, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                       : rb.counterName);
  datapath.setCounter(rb.id,
                      iv); // live while the children emit (their outer index)

  // Loop-carried iter-args (e.g. an outer loop carrying an accumulator into
  // an inner reduction): each latches its init at `start`, advances on each
  // outer-iteration drain, and its final value is this region's survivor.
  Backedge advanceEdge = ctx.bb.get(ctx.i1);
  auto ci = dp.carryInfo.find(rb.id);
  SmallVector<Backedge> nextBE;
  if (ci != dp.carryInfo.end())
    nextBE = setupCarriedIterArgs(rb, ci->second.inits, start, advanceEdge);

  // Emit the container's own combinational units (a nested guard's predicate
  // over this counter) now the counter + iter-arg survivors are live, so a
  // guard child reads its predicate as a Source::Unit when it emits below.
  datapath.emitCombUnits(rb);

  // Child 0 starts on `child0Start` (resolved below); `lastEdge` is the last
  // child's done edge, the outer iteration's completion.
  Backedge child0Start = ctx.bb.get(ctx.i1);
  Value lastEdge = ctx.risingEdge(sequence(rb.children, child0Start,
                                           /*retrig=*/true));
  advanceEdge.setValue(lastEdge); // advance the iter-args on each outer drain
  for (auto [k, nb] : llvm::enumerate(nextBE))
    nb.setValue(datapath.resolveSource(ci->second.nexts[k]));
  Value ivStep =
      ctx.R(comb::AddOp::create(ctx.b, ctx.loc, iv, term.step, false));
  Value last = term.isLast(ctx, ivStep); // this outer iteration is the last
  Value advance = ctx.andBits(lastEdge, ctx.notBit(last));
  // Restart child 0 one cycle after the outer start pulse or each
  // outer-iteration drain, registered so the counter has settled before a
  // child samples it as its own bound; `gateStart` masks a zero-trip start.
  child0Start.setValue(
      ctx.reg(ctx.mux(term.gateStart(ctx, start), ctx.t1, advance), ctx.f1));
  Value ivAdv = ctx.mux(advance, ivStep, iv);
  ivNext.setValue(ctx.mux(start, term.lb, ivAdv));
  // Latch done when the last child of the last outer iteration drains, and
  // clear it on `start` so a retriggered container gives a fresh edge each
  // pass. A zero-trip container completes one cycle after `start` instead.
  Value emptyDone = ctx.reg(ctx.andBits(start, term.isEmpty(ctx)), ctx.f1);
  Value done =
      ctx.holdDone(ctx.orBits(emptyDone, ctx.andBits(lastEdge, last)), start);
  nameValue(done, regionSignal(rb.id, "done"));
  return done;
}

// A conditional container: a sequential-wrapper while whose body nests child
// regions (an outer while enclosing an inner while). Each outer iteration
// runs the children once (as emitContainer), but the loop is data-dependent:
// the outer iter-args are frozen survivor registers advanced by the children's
// results, and the loop ends when the combinational continue-condition (a raw
// arith tree over those registers) goes false. A done-based CHECK/RUN FSM
// times it: one cycle after `start`, and after each outer-iteration drain,
// the condition is re-checked on the settled iter-args; if it holds the
// children (re)start, else the container is done and the iter-args hold
// their final values (a sibling reads them as this region's survivors). No
// squash or stall: the same non-speculative flushing family as a leaf while.
Value HWEmitter::emitConditionalContainer(const uarch::RegionBlock &rb,
                                          Value start) {
  RegionTag tag(ctx, rb.id);
  const auto &wi = dp.carryInfo.find(rb.id)->second;
  unsigned nArgs = wi.inits.size();

  // Outer iter-arg registers = this region's survivors: each latches its
  // init at `start`, then advances to a child survivor's value when an
  // outer iteration drains (`advanceEdge`, resolved after the children emit).
  Backedge advanceEdge = ctx.bb.get(ctx.i1);
  SmallVector<Backedge> nextBE =
      setupCarriedIterArgs(rb, wi.inits, start, advanceEdge);

  // CHECK-start pulse: one cycle after `start` or after each outer-iteration
  // drain, when the iter-arg survivor registers have settled. The condition
  // cone reads those (frozen) survivors, so it launches here.
  Value checkStart = ctx.reg(ctx.orBits(start, advanceEdge), ctx.f1);
  nameValue(checkStart, regionSignal(rb.id, "check"));
  // Emit the condition cone and get the continue-condition plus its ready
  // latency t_cond: 0 for a combinational condition, or several cycles for a
  // memory-/IP-dependent one, which the CHECK/RUN regime waits for.
  auto [cond, tCond] = datapath.emitConditionRegion(rb, wi.condition);
  Value condValid = ctx.delayValid(checkStart, tCond);
  // Start the children only if the condition holds when it settles; otherwise
  // the container completes this cycle.
  auto [child0Start, donePulse] = ctx.branchPulse(condValid, cond);

  // Sequence the children within one outer iteration; the last child's drain
  // edge advances the iter-args (resolving the survivor next-values) and drives
  // the next CHECK.
  Value lastEdge =
      ctx.risingEdge(sequence(rb.children, child0Start, /*retrig=*/true));
  advanceEdge.setValue(lastEdge);
  for (unsigned k = 0; k < nArgs; ++k)
    nextBE[k].setValue(datapath.resolveSource(wi.nexts[k]));

  // Latch done (a level) when the condition first fails; clear on `start` so a
  // retriggered container presents a fresh edge each pass (harmless top-level).
  Value done = ctx.holdDone(donePulse, start);
  nameValue(done, regionSignal(rb.id, "done"));
  return done;
}

// A guard region (a dcp.select): its two arms run mutually-exclusively under
// the predicate. The then-arm (`children`) runs iff the predicate holds; the
// else-arm (`elseChildren`) runs iff it does not (a *dual* guard). The
// predicate is a held value (the condition region's survivor, captured before
// the guard emits, valid at `start`). The not-taken arm's children never
// issue, so their stores never fire: the predicate reaches every store
// write-enable structurally, via the missing issue pulse, not a per-store
// gate. An empty arm (a then-only guard's absent else, or a pass-through else
// that yields a value but runs no schedule) completes in one cycle: its start
// pulse IS its drain. Either way the region produces a done edge, so an
// enclosing container advances past it in both branches. Run-once: no
// iteration or iter-args, unlike emitConditionalContainer, since the
// predicate is independent of the children.
Value HWEmitter::emitGuard(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id);
  const auto &gi = dp.guardCond.find(rb.id)->second;
  // The predicate as a Source: a scheduled condition region's survivor (a
  // data-dependent scf guard), or the parent container's combinational
  // predicate unit (an affine guard, emitted by emitCombUnits beforehand).
  Value cond = datapath.resolveSource(gi.condition);
  // CHECK one cycle after start (as in emitConditionalContainer): this
  // decouples the completion pulse from the start-clear below, since a
  // skipped guard's done would otherwise coincide with `start` and be masked.
  Value checkTime = ctx.reg(start, ctx.f1);
  nameValue(checkTime, regionSignal(rb.id, "check"));
  // Two mutually-exclusive arm pulses: thenStart and elseStart.
  auto [thenStart, elseStart] = ctx.branchPulse(checkTime, cond);
  // Each arm runs its children once (retrig so a re-entered guard presents
  // fresh edges each enclosing pass); an empty arm drains on its own start
  // pulse.
  Value thenDrained =
      rb.children.empty()
          ? thenStart
          : ctx.risingEdge(sequence(rb.children, thenStart, /*retrig=*/true));
  Value elseDrained = rb.elseChildren.empty()
                          ? elseStart
                          : ctx.risingEdge(sequence(rb.elseChildren, elseStart,
                                                    /*retrig=*/true));
  // Result-mux: each yielded result is `cond ? then-value : else-value`.
  // Latch each branch's value when that arm drains (only the taken arm
  // fires, so the mux ignores the other's stale survivor) and select by cond.
  auto rit = dp.regionResult.find(rb.id);
  if (rit != dp.regionResult.end()) {
    const auto &elseR = dp.selectElseResult.find(rb.id)->second;
    for (auto [k, thenSrc] : llvm::enumerate(rit->second)) {
      Value tv = datapath.resolveSource(thenSrc);
      Value ev = datapath.resolveSource(elseR[k]);
      Value thenSurv =
          ctx.enabledReg(tv, thenDrained, ctx.konst(tv.getType(), 0));
      Value elseSurv =
          ctx.enabledReg(ev, elseDrained, ctx.konst(ev.getType(), 0));
      nameValue(thenSurv, survivorName(rb.id, k));
      nameValue(elseSurv, survivorName(rb.id, k));
      datapath.setSurvivor(rb.id, k, ctx.mux(cond, thenSurv, elseSurv));
    }
  }
  // Exactly one arm runs, so the region completes on whichever drains. Latch
  // done (a level); clear on start so a retriggered guard re-edges.
  Value done = ctx.holdDone(ctx.orBits(thenDrained, elseDrained), start);
  nameValue(done, regionSignal(rb.id, "done"));
  return done;
}

// Emit the whole module body: preamble (literals + read ports + internal
// memories) once, then the func-scope sibling regions in program order, chained
// by the done-based sequencer. Nested regions emit inside their container.
void HWEmitter::emit() {
  ctx.initLiterals();
  datapath.bindReadPorts();
  datapath.createInternalMemories();
  SmallVector<uarch::RegionId> top;
  for (const uarch::RegionBlock &rb : dp.regions)
    if (!rb.parent) // a child region emits inside its container
      top.push_back(rb.id);
  // Compose the top-level siblings by their dependence DAG: independent
  // regions start together, the rest gate on their producers' `done`; retrig
  // keeps the module re-invocable with a fresh `done` edge each drive.
  pa.setOutput(kDone, composeSiblings(top, pa.getInput(kStart)));
  // Scalar results: the returning region's survivor register, stable once
  // its region (and thus `done`) has risen; the cosim samples it at `done`.
  for (const uarch::Result &r : dp.results)
    pa.setOutput(r.name, datapath.resolveSource(r.source));
}

//===----------------------------------------------------------------------===//
// emitModule: interface (ports, extern operator modules) + validation.
//===----------------------------------------------------------------------===//

// Reject a datapath outside the emittable subset, with a source diagnostic. The
// preconditions the leaf lowering relies on: a schedulable region set, a trip
// for every cyclic region, a combinational while/guard condition, no in-loop
// store under a while, no cross-region value hand-off (spilling), and an
// emittable realization for every compute unit.
static LogicalResult validateDatapath(func::FuncOp func,
                                      const uarch::Datapath &dp) {
  // Supported subset: top-level siblings in program order, plus container
  // loops whose children sequence within one outer iteration (crossing as a
  // survivor register); a counted cyclic region needs a trip, a while flushes.
  if (dp.regions.empty())
    return func.emitError("allo-datapath-to-hw: no schedulable region");
  // The builder already reported the offending edge; fail before any
  // hardware is built from the placeholder depths it left.
  if (dp.infeasible)
    return failure();
  // Access latencies the emitted structure cannot realize. Both are device
  // rows the SCHEDULER honors, so silently emitting a 1-cycle port instead
  // would place every consumer of that array on the wrong cycle.
  for (const uarch::MemUnit &m : dp.mems) {
    // An internal array is a `seq.hlmem`: CIRCT realizes read latency > 1 (it
    // delays the address and registers the data) but only a 1-cycle write.
    if (!m.external && !m.isRom && m.writeLatency != 1)
      return func.emitError("allo-datapath-to-hw: on-chip array with a ")
             << m.writeLatency
             << "-cycle write is unsupported (seq.hlmem realizes only a "
                "1-cycle write); declare wr_lat 1 for this storage impl";
    // A boundary array's port latency is a contract with the driver, not
    // enforced by the emitted RTL; the interface manifest carries it, so any
    // latency >= 1 works, but 0 is rejected (an edge-triggered port can't).
    if (m.external && (m.readLatency < 1 || m.writeLatency < 1))
      return func.emitError("allo-datapath-to-hw: argument array with a ")
             << m.readLatency << "-cycle read / " << m.writeLatency
             << "-cycle write is unsupported; a boundary port is "
                "edge-triggered "
                "and needs at least 1 cycle. Use an internal buffer, or bind "
                "this argument to a storage impl with a >= 1 cycle access";
  }
  for (const uarch::RegionBlock &rb : dp.regions)
    if (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional &&
        !rb.tripCount && !rb.ubSource)
      return func.emitError("allo-datapath-to-hw: cyclic region needs a "
                            "constant or dynamic trip");
  // Condition timing: a flushing leaf while or guard samples it in-cycle,
  // needing a stage-0 Unit or settled Survivor; a sequential CHECK/RUN while
  // instead waits t_cond cycles, so a multi-stage condition works there.
  auto conditionOk = [&](const uarch::Source &s, bool sequential) {
    switch (s.kind) {
    case uarch::Source::Kind::Survivor:
      return true; // a scheduled prologue predicate, valid at the region start
    case uarch::Source::Kind::Unit:
      return sequential || dcpStart(dp.units[s.id].boundOps.front().first) == 0;
    default:
      return false; // None (raw / unliftable)
    }
  };
  for (const uarch::RegionBlock &rb : dp.regions) {
    if (rb.conditional &&
        !conditionOk(dp.carryInfo.find(rb.id)->second.condition,
                     /*sequential=*/!rb.children.empty()))
      return func.emitError("allo-datapath-to-hw: a while loop with a non-"
                            "combinational (memory-/IP-dependent) condition is "
                            "not yet lowered");
    if (rb.guard && !conditionOk(dp.guardCond.find(rb.id)->second.condition,
                                 /*sequential=*/false))
      return func.emitError("allo-datapath-to-hw: a guard with a "
                            "non-combinational predicate is not yet lowered");
  }
  // A leaf `while` with an in-loop store lowers: emitAccesses gates each
  // store's write-enable by `issue & cond`, so a doomed exit iteration
  // commits nothing, matching the non-speculative loop-carried-survivor rule.

  // An unresolved (None) input is a cross-region SSA value hand-off (a
  // scalar produced in one region, consumed in another); reject it cleanly
  // here rather than asserting deep in `src`, since spilling is unsupported.
  auto none = [](const uarch::Source &s) {
    return s.kind == uarch::Source::Kind::None;
  };
  for (const uarch::FuncUnit &u : dp.units)
    if (llvm::any_of(u.inputs, none))
      return func.emitError("allo-datapath-to-hw: cross-region value hand-off "
                            "not yet supported");
  for (const uarch::MemUnit &m : dp.mems)
    for (const uarch::MemUnit::Access &acc : m.accesses)
      if (llvm::any_of(acc.addr, none) || (acc.isWrite && none(acc.data)))
        return func.emitError(
            "allo-datapath-to-hw: cross-region value hand-off "
            "not yet supported");

  // Realizability: every compute unit needs an emittable realization. A
  // combinational unit needs an EmitHW comb lowering (`combEmitted`); an IP
  // unit needs a non-empty module name. Fail by op name, not deep in emission.
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.comb) {
      if (!combEmitted(u.opType)) {
        error(Stage::Emit, u.boundOps.back().first)
            << "Combinational operator '" << u.opType
            << "' has no native EmitHW "
               "lowering; provide an IP "
               "or add native support";
        return failure();
      }
    } else if (u.impl.empty()) {
      error(Stage::Emit, u.boundOps.back().first)
          << "Operator '" << u.opType
          << "' has no IP module realization; provide an IP for this operator "
             "or add native support";
      return failure();
    }
  }
  // Reject a CallUnit shape the leaf cannot lower, loudly, rather than
  // mis-emitting it.
  for (const uarch::CallUnit &cu : dp.calls) {
    // A multi-scalar-result call yields several survivors from one op, but
    // producerOf (and thus regionResult) is keyed per op, so only result 0
    // is tracked. Reject more than one: per-result tracking is unsupported.
    if (cu.resultPorts.size() > 1)
      return func.emitError("allo-datapath-to-hw: a sub-kernel call returning "
                            "more than one scalar is not yet lowered");
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      // Several serial calls mastering one boundary arg time-share the top
      // port via emitCalls' master mux. A call sharing that arg with a
      // *parent* access needs the same mux via emitAccesses; unsupported.
      if (ma.isBoundary && !dp.mems[ma.mem].accesses.empty())
        return func.emitError(
            "allo-datapath-to-hw: a boundary argument mastered by a sub-kernel "
            "call and a parent access needs a port-sharing mux -- not yet "
            "lowered");
    }
    // A void indeterminate call (no static latency) lowers fine: the region
    // completes on the child's real `done`. Only a scalar-returning
    // indeterminate call is rejected: its result timing is data-dependent.
    if (!cu.latency && !cu.resultPorts.empty())
      return func.emitError(
          "allo-datapath-to-hw: an indeterminate sub-kernel call returning a "
          "scalar has a data-dependent result timing, not yet lowered");
  }
  return success();
}

// Declare an extern operator module for each IP-realized compute unit, named by
// `operatorModuleName` and deduplicated across the whole module (`opModules`).
// Native (comb) units emit inline, no extern. One input port per operand (named
// `a`, `b`, `c`, ... at each operand's width: a unary cast/`sqrt` gets `a`
// only, a binary op `a`+`b`, a compare two operands yielding i1) then the
// output at the result width. The interface follows the realization's stall
// contract:
// `(a.., clk) -> y` free-running, or `(a.., clk, ce) -> y` when clock-enabled
// (`ce == 0` freezes the pipe in lockstep with the shell). Both are a function
// of `impl`, so every instance of a module name shares one port shape, which is
// what makes the dedup safe. Returns unit id -> its extern module.
static DenseMap<unsigned, Operation *>
declareOperatorModules(func::FuncOp func, const uarch::Datapath &dp,
                       OpBuilder &b, llvm::StringMap<Operation *> &opModules,
                       std::vector<iface::Operator> &declared) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  DenseMap<unsigned, Operation *> unitModule;
  llvm::StringSet<> listed; // one manifest entry per module, not per unit
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.comb || u.boundOps.empty())
      continue;
    Operation *srcOp = u.boundOps.front().first;
    assert(u.inputs.size() == srcOp->getNumOperands() &&
           "IP unit input count must match its bound op's operand count");
    IntegerType outW = hwType(u.resultType, b);
    std::string modName = operatorModuleName(u);
    // The port shape is a function of the realization, so every instance of a
    // module name shares it: build the manifest entry alongside the ports.
    iface::Operator entry{modName, u.impl, operatorPredicate(u), {}};
    SmallVector<PortInfo> ep;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      IntegerType w = hwType(srcOp->getOperand(k).getType(), b);
      std::string pn(1, static_cast<char>('a' + k));
      ep.push_back({{StringAttr::get(ctx, pn), w, Dir::Input}});
      entry.ports.push_back({pn, w.getWidth(), iface::Operator::Role::Data});
    }
    ep.push_back({{StringAttr::get(ctx, kClk), b.getI1Type(), Dir::Input}});
    entry.ports.push_back({kClk.str(), 1, iface::Operator::Role::Clk});
    if (u.stall == allo::StallContractEnum::Ce) {
      ep.push_back({{StringAttr::get(ctx, kCe), b.getI1Type(), Dir::Input}});
      entry.ports.push_back({kCe.str(), 1, iface::Operator::Role::Ce});
    }
    ep.push_back({{StringAttr::get(ctx, kOpOut), outW, Dir::Output}});
    entry.ports.push_back(
        {kOpOut.str(), outW.getWidth(), iface::Operator::Role::Out});

    Operation *&mod = opModules[modName];
    if (!mod)
      mod = hw::HWModuleExternOp::create(b, loc, StringAttr::get(ctx, modName),
                                         hw::ModulePortInfo(ep));
    if (listed.insert(modName).second)
      declared.push_back(std::move(entry));
    unitModule[u.id] = mod;
  }
  return unitModule;
}

llvm::SmallVector<hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b) {
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  auto *ctx = b.getContext();
  Type i1 = b.getI1Type(), i32 = b.getIntegerType(32);
  // A data port's hw width is its field bit width, so `iType(w)` reproduces
  // `hwType`/`memElemType` for the data ports.
  auto iType = [&](unsigned w) -> Type { return b.getIntegerType(w); };
  SmallVector<PortInfo> ports;
  // The port names are the manifest, authored before CIRCT's LegalizeNames
  // runs, so a name ExportVerilog would rewrite or uniquify desyncs cosim from
  // the Verilog. `verilogName` prevents that; these check the composed result.
  llvm::StringSet<> seen;
  auto port = [&](const Twine &n, Type t, Dir d) {
    std::string s = n.str();
    assert(sv::isNameValid(s, /*caseInsensitiveKeywords=*/false) &&
           "module port name is not a legal SystemVerilog identifier; the JSON "
           "manifest would desync from the emitted Verilog");
    bool fresh = seen.insert(s).second;
    assert(fresh && "duplicate module port name; the JSON manifest would "
                    "desync from the emitted Verilog");
    (void)fresh;
    ports.push_back(PortInfo{{StringAttr::get(ctx, s), t, d}});
  };
  port(kClk, i1, Dir::Input);
  port(kRst, i1, Dir::Input);
  port(kStart, i1, Dir::Input);
  // Scalar kernel arguments; memref args become memory ports instead. One
  // named after a control port trips the duplicate check above.
  for (const iface::Scalar &s : model.scalars)
    port(s.name, iType(s.width), Dir::Input);
  // Stream FIFO ports, input side: module inputs must stay contiguous at the
  // front (HWModulePortAccessor maps body args to the first `numInputs`
  // ports positionally), so {data, valid} / {ready} go here; outputs follow
  // `done`.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.data, iType(s.width), Dir::Input);
      port(s.valid, i1, Dir::Input);
    } else {
      port(s.ready, i1, Dir::Input);
    }
  }
  // A partitioned argument presents one interface per bank (a data-dependent
  // access spans all of them, a static access one); `model.reads[i]` holds an
  // access's per-bank interfaces.
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.data, iType(r.width), Dir::Input);
  port(kDone, i1, Dir::Output);
  // Stream FIFO ports, output side (after `done`, among the module outputs): an
  // input stream's back-pressure {ready}; an output stream's {data, valid}.
  for (const iface::FIFO &s : model.streams) {
    if (s.isInput) {
      port(s.ready, i1, Dir::Output);
    } else {
      port(s.data, iType(s.width), Dir::Output);
      port(s.valid, i1, Dir::Output);
    }
  }
  for (const auto &acc : model.reads)
    for (const iface::Memory &r : acc)
      port(r.addr, i32, Dir::Output);
  for (const auto &acc : model.writes)
    for (const iface::Memory &w : acc) {
      port(w.addr, i32, Dir::Output);
      port(w.data, iType(w.width), Dir::Output);
      port(w.we, i1, Dir::Output);
    }
  // Scalar function results: one output port each, driven by the returning
  // region's survivor and valid when `done` rises (emit()).
  for (const iface::Result &r : model.results)
    port(r.name, iType(r.width), Dir::Output);
  return ports;
}

llvm::StringMap<Value> instantiateChild(OpBuilder &b, Location loc,
                                        hw::HWModuleOp mod,
                                        llvm::StringRef name,
                                        llvm::StringMap<Value> &ins) {
  using Dir = hw::ModulePort::Direction;
  SmallVector<Value> operands(mod.getNumInputPorts());
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Input) {
      auto it = ins.find(p.name.getValue());
      assert(it != ins.end() && "unwired child input port");
      operands[p.argNum] = it->second;
    }
  auto inst =
      hw::InstanceOp::create(b, loc, mod, b.getStringAttr(name), operands);
  llvm::StringMap<Value> outs;
  for (const hw::PortInfo &p : mod.getPortList())
    if (p.dir == Dir::Output)
      outs[p.name.getValue()] = inst.getResult(p.argNum);
  return outs;
}

// Emit an hw.module for one scheduled function's datapath. Returns failure with
// a diagnostic if the datapath is outside the supported subset
// (validateDatapath). `opModules` caches extern operator modules across
// functions.
static FailureOr<std::pair<hw::HWModuleOp, iface::ModuleInterface>>
emitModule(func::FuncOp func, uarch::Datapath &dp, OpBuilder &b,
           llvm::StringMap<Operation *> &opModules,
           const uarch::CalleeCtx *callees = nullptr) {
  auto *ctx = b.getContext();
  Location loc = func.getLoc();
  if (failed(validateDatapath(func, dp)))
    return failure();

  Type i1 = b.getI1Type();
  Type i32 = b.getIntegerType(32);

  // Enumerate *external* (argument) memory accesses as read / write ports.
  // Internal memories become on-chip seq.hlmem storage (emitted in the body),
  // so they take no module ports.
  SmallVector<AccRef> reads, writes;
  for (const uarch::MemUnit &m : dp.mems)
    if (m.external)
      for (unsigned a = 0; a < m.accesses.size(); ++a)
        (m.accesses[a].isWrite ? writes : reads).push_back({m.id, a});

  // The single source for every boundary port name, shared by the declaration
  // (declareModulePorts), the manifest and the cosim harness. It also carries
  // the extern operator modules this kernel instantiates.
  iface::ModuleInterface model(dp, reads, writes);
  auto unitModule =
      declareOperatorModules(func, dp, b, opModules, model.operators);
  auto ports = declareModulePorts(model, b);

  hw::ModulePortInfo portInfo(ports);
  // Legalized here rather than left to ExportVerilog, so the key the manifest
  // uses is the emitted Verilog module name. A nested callee `top.child` would
  // otherwise be rewritten downstream.
  model.symbol = func.getSymName().str();
  model.module = verilogName(model.symbol);
  StringAttr modName = StringAttr::get(ctx, model.module);

  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, reads, writes, unitModule, bb, i1, i32,
                    callees);
        e.ctx.clk = e.ctx.R(seq::ToClockOp::create(ib, loc, pa.getInput(kClk)));
        e.ctx.clkRaw = pa.getInput(kClk);
        e.ctx.rst = pa.getInput(kRst);
        e.emit();
      });

  // Hand the port model back to the caller: it derives the cosim manifest
  // JSON and, for a dataflow container, threads the leaf models into the
  // structural-top emitter, keeping one in-memory port representation.
  return std::make_pair(hwMod, std::move(model));
}

static bool hasDCPRegions(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// The IP operators' timing lives on module-level `dcp.operator` symbols. Fold
// each onto its referencing `dcp.compute` (its `latency` + `pipelined`) so the
// datapath reads timing locally, then drop the now-spent `dcp.operator` /
// `dcp.device` declarations. This lets each extern operator module share the
// operator's `sym_name` (the RTL module name) with no symbol clash: there is
// no live same-named symbol once the declarations are gone. Runs on the emit
// clone, so the canonical scheduled module keeps the normalized form.
static void stampOperatorTiming(ModuleOp module) {
  Builder bd(module.getContext());
  module.walk([&](dcp::DCPathComputeOp comp) {
    FlatSymbolRefAttr sym = comp.getOpTypeAttr();
    if (!sym)
      return; // a combinational compute (comb_kind); no operator timing
    auto opr =
        SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(comp, sym);
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    comp->setAttr("latency", bd.getI64IntegerAttr(opr.getLatency()));
    comp->setAttr("pipelined", bd.getBoolAttr(opr.getPipelined()));
    comp->setAttr("stall",
                  StallContractEnumAttr::get(bd.getContext(), opr.getStall()));
  });
  SmallVector<Operation *> spent;
  module.walk([&](Operation *op) {
    if (isa<dcp::DCPathOperatorOp, dcp::DCPathDeviceOp>(op))
      spent.push_back(op);
  });
  for (Operation *op : spent)
    op->erase();
}

LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top,
                               llvm::StringMap<std::string> &interfaces) {
  // Called directly (not via the pass manager), so load the dialects this
  // emits, the ones the pass declares as dependent, into the context.
  auto *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  // The device's storage timing is read BEFORE `stampOperatorTiming` drops
  // `dcp.device`. Compute timing folds onto each `dcp.compute`, but memory
  // latency has no such carrier, so it threads into the datapath builder.
  MemoryLibrary memLib = OperatorLibrary::fromModule(module).memoryLibrary();

  // Fold operator timing onto the compute ops and drop the declarations, before
  // any datapath is built or an extern operator module is named.
  stampOperatorTiming(module);

  SmallVector<func::FuncOp> scheduled;
  module.walk([&](func::FuncOp f) {
    if (hasDCPRegions(f))
      scheduled.push_back(f);
  });

  // A memref/array function result is unsupported: hardware writes output
  // through a memory port, not a returned value, and the prepass that would
  // rewrite it is deliberately not run. Reject cleanly here, not deep in the
  // builder.
  for (func::FuncOp f : scheduled)
    if (auto ret = dyn_cast<func::ReturnOp>(f.front().getTerminator()))
      for (Value v : ret.getOperands())
        if (isa<MemRefType>(v.getType())) {
          logging::error(logging::Stage::Emit, f)
              << "Returning a memref is unsupported; write the result through "
                 "an output argument (out-parameter) instead";
          return failure();
        }

  auto policy = bindingPolicyFor(binding);
  if (!policy)
    return module.emitError("allo-datapath-to-hw: unknown binding policy '")
           << binding << "'";

  // Emission is rooted at the top function and runs bottom-up over the call
  // DAG: each callee emits before its caller, so a container always finds
  // its children already registered. Mirrors the scheduler's traversal order.
  llvm::StringMap<func::FuncOp> byName;
  for (func::FuncOp f : scheduled)
    byName[f.getSymName()] = f;
  func::FuncOp topFunc = byName.lookup(top);
  if (!topFunc)
    return module.emitError("allo-datapath-to-hw: top function '")
           << top << "' is not a scheduled function";

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  // Callee tables, keyed by symbol name: leaf kernels plus containers
  // emitted so far. A container composes exactly like a leaf, so both
  // live here.
  llvm::StringMap<hw::HWModuleOp> modules;
  llvm::StringMap<iface::ModuleInterface> ifaceModels;
  llvm::StringSet<> visited;

  auto registerModule = [&](StringRef name, hw::HWModuleOp mod,
                            iface::ModuleInterface model) {
    // The callee tables key on the func symbol, which a callsite names; the
    // manifest keys on the emitted module name, which the simulator names.
    interfaces[mod.getModuleName()] = model.toJSON();
    modules[name] = mod;
    ifaceModels[name] = std::move(model);
  };

  // Post-order over the call DAG (acyclic; the frontend rejects recursion),
  // via a self-parameter recursive lambda (`self(self, ...)`).
  auto emitOne = [&](auto &self, func::FuncOp f) -> LogicalResult {
    if (!visited.insert(f.getSymName()).second)
      return success(); // a shared callee already emitted
    // Children first: emit every scheduled callee (a leaf call misses
    // `byName`). A callee is referenced by a `func.call` or a `dcp.instance`
    // (a leaf CallUnit); both must recurse before their caller emits.
    WalkResult wr = f.walk([&](Operation *op) -> WalkResult {
      StringRef callee;
      if (auto call = dyn_cast<func::CallOp>(op))
        callee = call.getCallee();
      else if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op))
        callee = inv.getCallee();
      else
        return WalkResult::advance();
      auto it = byName.find(callee);
      if (it != byName.end() && failed(self(self, it->second)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // A container wires its children into a structural top (inserted at the
    // module-body start, so the outermost lands at position 0). Only a
    // CONCURRENT container takes this path; everything else emits as a leaf.
    auto det = f->getAttrOfType<DeterminacyEnumAttr>("dcp.determinacy");
    if (det && det.getValue() == DeterminacyEnum::Concurrent) {
      b.setInsertionPointToStart(module.getBody());
      hw::HWModuleOp topMod;
      iface::ModuleInterface topModel;
      if (failed(emitConcurrentTop(f, modules, ifaceModels, byName, b, topMod,
                                   topModel)))
        return failure();
      registerModule(f.getSymName(), topMod, std::move(topModel));
    } else {
      // A plain leaf, or a rerouted mixed container whose sync calls are
      // `dcp.instance`s. The latter needs its already-emitted callees' modules
      // + port models to build/emit each CallUnit; a plain leaf passes null.
      bool hasInvoke = false;
      f.walk([&](dcp::DCPathInstanceOp) { hasInvoke = true; });
      uarch::CalleeCtx cc{modules, ifaceModels};
      const uarch::CalleeCtx *callees = hasInvoke ? &cc : nullptr;
      Datapath dp(f, *policy, memLib, callees);
      LLVM_DEBUG({
        llvm::dbgs() << "// datapath for @" << f.getSymName() << "\n";
        dp.dump(llvm::dbgs());
      });
      b.setInsertionPoint(f);
      auto pairOr = emitModule(f, dp, b, opModules, callees);
      if (failed(pairOr))
        return failure();
      registerModule(f.getSymName(), pairOr->first, std::move(pairOr->second));
    }
    return success();
  };

  if (failed(emitOne(emitOne, topFunc)))
    return failure();

  // cleanup non-hw ops to avoid Verilog export errors
  for (func::FuncOp f : scheduled)
    f.erase();
  for (memref::GlobalOp g :
       llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
    g.erase();
  return success();
}

} // namespace mlir::allo::uarch
