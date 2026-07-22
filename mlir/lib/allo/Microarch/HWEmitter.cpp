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

unsigned memReadLatency(MemoryImplEnum impl) {
  return impl == MemoryImplEnum::Register ? 0 : 1;
}

// The schedule cycle at which `op` fires (its `start`); the emit-side spelling
// of the shared reader.
unsigned schedT(Operation *op) { return uarch::dcpStart(op); }

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
// two enums (comb adds 4-state predicates we never produce).
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
  // An affine.apply: index arithmetic parameterized by a map (carried on the op
  // the way arith.cmpi carries its predicate) rather than by arity. This is the
  // delinearization `flatten-perfect-loops` leaves behind when a coalesced
  // nest's body reads an original IV outside an address -- a guard's condition,
  // materialized by the if-conversion over the surviving IV. The same map
  // evaluator the address path uses (evalAffine): same non-negative index, same
  // constant divisors, so a power-of-two delinearization stays a shift and a
  // mask rather than the signed correction a generic affine expansion emits.
  if (kind == "apply") {
    assert(srcOp->getAttr("map") &&
           "dcp.compute<apply> must carry the original affine map");
    AffineMap map = cast<AffineMapAttr>(srcOp->getAttr("map")).getValue();
    assert(map.getNumResults() == 1 && "affine.apply yields one result");
    return evalAffine(b, loc, map.getResult(0), operands, map.getNumDims());
  }
  Value lhs = operands[0];
  // Width-changing unary casts (the widened-reduction idiom
  // trunc(add(ext,ext))) resize operand[0] to the unit's result width: comb
  // sign/zero-extend and a low-bit extract. All 0-latency, so they slot into
  // the schedule like any comb.
  if (kind == "extsi")
    return comb::createOrFoldSExt(b, loc, lhs, resultType);
  if (kind == "extui")
    return comb::createZExt(b, loc, lhs,
                            cast<IntegerType>(resultType).getWidth());
  if (kind == "trunci")
    return comb::ExtractOp::create(b, loc, resultType, lhs, 0).getResult();
  if (kind == "index_cast") {
    // index <-> integer: both carried at their hw integer width (hwType maps
    // index to i32), so a signed resize to the result width -- sExt / low-bit
    // extract / identity when the widths already match.
    unsigned dst = cast<IntegerType>(resultType).getWidth();
    unsigned src = cast<IntegerType>(lhs.getType()).getWidth();
    if (dst == src)
      return lhs;
    return dst > src ? comb::createOrFoldSExt(b, loc, lhs, resultType)
                     : comb::ExtractOp::create(b, loc, resultType, lhs, 0)
                           .getResult();
  }
  // Float negate: arith.negf flips the sign bit of the float, which rides as
  // its integer bit pattern here -- a single XOR, no IP. Unary, so it precedes
  // the `rhs = operands[1]` read below.
  if (kind == "negf") {
    unsigned w = cast<IntegerType>(resultType).getWidth();
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
  // Signed / unsigned divide. Emitted for a flattened guard's delinearization
  // (an affine `i floordiv N` in the predicate lowers to the signed-divide
  // idiom over the coalesced counter); a scheduled data divide is a multi-cycle
  // IP, not this comb path.
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

std::string memPortBase(const uarch::Datapath &dp, ArrayRef<AccRef> ports,
                        unsigned i, StringRef role) {
  Value memref = dp.mems[ports[i].mem].memref;
  auto name = nameFromLoc(memref.getLoc());
  if (!name)
    return (role + Twine(i)).str(); // unnamed argument: positional fallback
  std::string base = sanitizeCppIdentifier(*name) + "_" + role.str();
  // Index only when this argument backs more than one port of the same role.
  unsigned total = 0, index = 0;
  for (unsigned j = 0; j < ports.size(); ++j)
    if (ports[j].mem == ports[i].mem) {
      if (j < i)
        ++index;
      ++total;
    }
  if (total > 1)
    base += std::to_string(index);
  return base;
}

std::string scalarPortName(const uarch::IOPort &io) {
  if (auto name = nameFromLoc(io.value.getLoc()))
    return sanitizeCppIdentifier(*name);
  return ("s" + Twine(io.id)).str();
}

std::string memBoundaryPortBase(const uarch::Datapath &dp, uarch::MemId mem,
                                StringRef role) {
  Value memref = dp.mems[mem].memref;
  if (auto name = nameFromLoc(memref.getLoc()))
    return sanitizeCppIdentifier(*name) + "_" + role.str();
  return (role + Twine(mem)).str(); // unnamed argument: stable fallback
}

llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const uarch::Datapath &dp, ArrayRef<AccRef> ports, unsigned i,
         StringRef role) {
  const uarch::MemUnit &m = dp.mems[ports[i].mem];
  ExternalBanking eb = externalBank(m, m.accesses[ports[i].idx]);
  std::string base = memPortBase(dp, ports, i, role);
  if (eb.factor == 1)
    return {{0u, base}};
  if (eb.bank)
    return {{*eb.bank, base}}; // statically routed to one interface
  // Data-dependent: one interface per bank (the crossbar drives every bank).
  llvm::SmallVector<std::pair<unsigned, std::string>> all;
  for (unsigned k = 0; k < eb.factor; ++k)
    all.push_back({k, base + "_b" + std::to_string(k)});
  return all;
}

void nameValue(Value v, StringRef name) {
  if (name.empty())
    return;
  Operation *op = v.getDefiningOp();
  if (!op) // a block argument / unresolved backedge is named elsewhere
    return;
  // Pick the channel ExportVerilog reads: a register names from its own `name`
  // attr (sv.namehint is ignored on a reg), any other value from `sv.namehint`.
  if (auto reg = dyn_cast<seq::CompRegOp>(op))
    reg.setNameAttr(StringAttr::get(op->getContext(), name));
  else
    op->setAttr("sv.namehint", StringAttr::get(op->getContext(), name));
}

void nameValue(Value v, Location loc) {
  if (auto name = nameFromLoc(loc))
    nameValue(v, sanitizeCppIdentifier(*name));
}

std::string cellName(Location loc, StringRef fallback) {
  if (auto name = nameFromLoc(loc))
    return sanitizeCppIdentifier(*name);
  return fallback.str();
}

std::string streamPortBase(const uarch::StreamChannel &s) {
  std::string fallback = "stream" + std::to_string(s.id);
  return cellName(s.stream.getLoc(), fallback);
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
  return shiftChain(sig, n).last();
}

Value EmitContext::activationPulse(Value pulse, Operation *op) {
  return delayValid(pulse, schedT(op));
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
// `lb, lb+step, ...` and terminates on `iv+step >= ub`. Each bound is a
// resolved runtime Source (a data-dependent range start/count/stride) or the
// constant fast path (the `lb`/`step` integers, `ub = lb + trip*step`). Empty
// (default) for an acyclic region (no counter) or a while (which builds its own
// Terminator::conditional from the resolved condition).
Terminator HWEmitter::terminatorOf(const uarch::RegionBlock &rb) {
  auto bound = [&](const uarch::Source &s, int64_t c) {
    return s ? datapath.resolveSource(s) : ctx.konst(ctx.i32, c);
  };
  Value lb = bound(rb.lbSource, rb.lb), step = bound(rb.stepSource, rb.step);
  if (rb.ubSource)
    return Terminator::counted(lb, datapath.resolveSource(rb.ubSource), step,
                               /*dynamic=*/true);
  if (rb.tripCount)
    return Terminator::counted(
        lb, ctx.konst(ctx.i32, rb.lb + *rb.tripCount * rb.step), step,
        /*dynamic=*/false);
  return {};
}

// One imperative path for every leaf region (counted / dynamic-trip / while):
// control -> datapath -> resolve the F->G condition, capture results, done. The
// regimes differ only in the Terminator and the survivor mechanism (see
// captureResults); the shared skeleton reads as a linear sequence.
Value HWEmitter::emitRegion(const uarch::RegionBlock &rb, Value start,
                            bool retrig) {
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

  // Latency-insensitive shell: a region with stream accesses gets two signals
  // (both F->G datapath values, resolved after the datapath, so they ride
  // backedges) --
  // `chainEnable` (~output-full) drives ctx.regionEnable so every shift chain +
  // the done drain freeze coherently on back-pressure (preserving tap
  // alignment), and `issueEnable` (~output-full & inputs-available) gates issue
  // so an empty input is a bubble, not a freeze. A stream-free region keeps
  // enable == true and regionEnable null -- identical to a stream-free region.
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

  RegionControl rc = control.emitPipelineControl(rb, term, start, enable);
  datapath.setControl(rb.id, rc); // seam G -> F (counter + issue)

  // Datapath: -> feedback (the store drain + shell signals; a while's condition
  // + its next-value producers are now emitted). The shell backedges are
  // resolved at the very end (after the done drain, which also reads
  // ctx.regionEnable) -- a setValue before that last use would not RAUW it.
  DatapathFeedback fb = datapath.emit(rb, rc.issue);

  // Resolve the F->G condition backedge now the datapath has emitted it, and
  // re-point the terminator at the resolved value -- `setValue` RAUWs and
  // erases the placeholder, so a *later* read of `term.cond` (lastIssuePulse's
  // exit test) must use the real condition, not the dead backedge handle.
  if (rb.conditional) {
    Value cond =
        datapath.resolveSource(dp.carryInfo.find(rb.id)->second.condition);
    condBE.setValue(cond);
    term.cond = cond;
  }

  // Survivors: capture the region's results (returning their drain stage) and
  // pin the last iteration's issue pulse -- the one pulse the done and the
  // captures share.
  Value lastIssue = lastIssuePulse(rc, term);
  unsigned resultDrain = captureResults(rb, rc, lastIssue, start);
  unsigned drainStage = std::max(fb.storeDrain, resultDrain);

  // Completion: the region's done signal. A counted leaf that is empty (lb >=
  // ub
  // -- a static `range(1,1)` or a runtime zero-trip) issues nothing, so it
  // completes on `start` via `emptyDone` (else its store-drain done never fires
  // -> deadlock). Folds away for a statically non-empty loop. A while / acyclic
  // never reports empty here (null).
  //
  // Delayed one cycle so the pulse cannot land on `start` itself: `done` is a
  // latched LEVEL that consumers complete on the rising edge of, so it has to
  // read 0 for at least one cycle between two runs of a retriggered region --
  // which is exactly what emitDone's start-clear provides, and what a `fire` on
  // the same cycle would defeat. Firing on `start` a region whose done is still
  // held from the previous outer iteration would hold the level at 1 across the
  // restart, and the enclosing container would wait forever for an edge that
  // never comes. Every other `fire` path is already at least one cycle past
  // `start` (a store drains at stage >= 1; emitAcyclic registers its issue).
  Value emptyDone =
      (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional)
          ? ctx.delayValid(ctx.andBits(start, term.isEmpty(ctx)), 1)
          : Value();
  // emitDone's drain chain must still see the shell's enable; leave it after.
  // A CallUnit region completes on the child instance's real `done`
  // (fb.callDone) -- correct for a determinate child and the only option for an
  // indeterminate one -- bypassing the store-drain done (a call region has no
  // parent-issued stores of its own). A scalar result the child holds on its
  // output port until `done` is the region's survivor
  // *directly* (emitCalls set it), so the child's `done` handshake gates a
  // consumer on the valid result without a separate statically-timed capture.
  Value done = fb.callDone ? fb.callDone
                           : control.emitDone(drainStage, lastIssue, emptyDone,
                                              start, retrig);
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
// register, on the cycle it lands -- while the result is still on its Source (a
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
    // A loop-carried result (a reduction accumulator) preloads its init on the
    // region `start`, then latches the final value when it lands: a zero-trip
    // run issues nothing, so `cap` never fires and the survivor holds the init
    // (the reduction identity) rather than a stale accumulator from a prior
    // invocation. A result with no init (an acyclic once-computed survivor)
    // always lands, so a plain capture-when-ready suffices.
    uarch::Source initSrc =
        initIt != dp.regionResultInit.end() && k < initIt->second.size()
            ? initIt->second[k]
            : uarch::Source{};
    Value survivor =
        initSrc.kind == uarch::Source::Kind::None
            ? ctx.enabledReg(res, cap, ctx.konst(res.getType(), 0))
            : ctx.latchReg(datapath.resolveSource(initSrc), res, start, cap);
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
  const uarch::Datapath::CarryInfo &wi = dp.carryInfo.find(rb.id)->second;
  Value cond = datapath.resolveSource(
      wi.condition); // memoized (the resolved continue-condition)
  // A while continues (advances its recurrences) on each issued iteration whose
  // condition is true. A carried next-value produced at a later stage lands
  // that many cycles after the issue, so its advance pulse is delayed to match
  // -- multi-stage flush (a load in the body pushes `next` to stage 1). A
  // combinational body (every next at stage 0) reduces to the one shared pulse
  // (a single-stage body: delayValid by 0 is the identity).
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
    datapath.setSurvivor(rb.id, k, ctx.latchReg(init, next, start, advance));
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// Run `regions` in program order, each region starting when its predecessor
// drains (the first on `start`); returns the last region's done. The shared
// sequencer -- func-scope siblings (a single pass) and a container's children
// (once per outer iteration) are the same "start k+1 when k drains" pattern.
Value HWEmitter::sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                          bool retrig) {
  Value done;
  Value startK = start;
  for (auto [i, rid] : llvm::enumerate(regions)) {
    const uarch::RegionBlock &rb = dp.regions[rid];
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
// exactly -- each start is the rising edge of the prior `done` and the
// conjunction equals the final `done`.
Value HWEmitter::composeSiblings(llvm::ArrayRef<uarch::RegionId> regions,
                                 Value start) {
  llvm::DenseMap<uarch::RegionId, Value> doneOf;
  Value allDone;
  for (uarch::RegionId rid : regions) {
    const uarch::RegionBlock &rb = dp.regions[rid];
    // No predecessors: run concurrently with the kernel `start`. Else start on
    // the rising edge of the predecessors' joined `done` -- the join waits for
    // the last producer to complete. (A determinate predecessor's `done` edge
    // is its static offset, reused as a time-trigger.)
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
// Source::Survivor{rb, k} -- read by the children's init reads and, for the
// final value, a sibling -- and returns the per-arg next-value backedges the
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
    datapath.setSurvivor(rb.id, k, ctx.latchReg(init, nb, start, advance));
  }
  return nextBE;
}

// A loop-over-call region: see the header. The counter is `rc.counter` (so
// emitCalls wires the child's index port to it via Source::Counter) and the
// child start is `rc.issue` (the loop-fire pulse); the region completes when
// the last iteration's `done` latches. One child instance fires N times, each
// invocation advancing on its real `done` -- a held level cleared on its start,
// so its rising edge marks each completion.
Value HWEmitter::emitLoopCall(const uarch::RegionBlock &rb, Value start) {
  assert(rb.callUnits.size() == 1 && rb.units.empty() && rb.regs.empty() &&
         "a loop-over-call region is one child with no loose datapath");
  assert(rb.tripCount && rb.lb == 0 && rb.step == 1 &&
         "loop-over-call first cut supports a `0 to N step 1` trip");
  int64_t N = *rb.tripCount;
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
  nameValue(k, "loop_iv");
  Value more = ctx.notBit(ctx.icmpUgeV(k, kconst(N - 1))); // iterations after k
  Value advance = ctx.andBits(doneEdge, more);
  Value kInc = ctx.R(comb::AddOp::create(ctx.b, ctx.loc, k, kconst(1), false));
  kNextBE.setValue(ctx.mux(start, kconst(0), ctx.mux(advance, kInc, k)));
  // Fire the next iteration one cycle after the done edge, once k has settled.
  Value fireNext = ctx.reg(advance, ctx.f1);
  Value childStart = ctx.orBits(start, fireNext);

  // Datapath: emitCalls wires the single child -- start = rc.issue =
  // childStart, index port = resolveSource(Counter) = k, boundary/internal mems
  // mastered as usual.
  RegionControl rc{/*issue=*/childStart, /*counter=*/k, /*wantIssue=*/Value()};
  datapath.setControl(rb.id, rc);
  DatapathFeedback fb = datapath.emit(rb, rc.issue);
  assert(fb.callDone && "a loop-over-call region produced no child done");
  doneBE.setValue(fb.callDone);

  // done: latch the last iteration's completion (a done edge with none left),
  // cleared on `start` so a re-invocation re-arms.
  Value last = ctx.andBits(doneEdge, ctx.notBit(more));
  Backedge doneHeldBE = ctx.bb.get(ctx.i1);
  Value doneHeld = ctx.reg(doneHeldBE, ctx.f1);
  nameValue(doneHeld, "loop_done");
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
  // Induction bounds: compile-time constants (the common counted container) or
  // runtime Sources -- a variable-trip container, whose bound is an enclosing
  // loop's counter (Source::Counter, a triangular/tile `for ii in range(i,
  // ...)`) or a prologue survivor. `terminatorOf` resolves both; a runtime
  // bound carries no static trip, so termination is `iv+step >= ub` (`isLast`)
  // and a zero-trip
  // (`lb >= ub`) container runs no child at all (`isEmpty`/`gateStart`).
  Terminator term = terminatorOf(rb);
  // The outer counter is the source IV: init `lb`, advance by `step`, so the
  // children read the real outer index (Source::Counter). The counter register
  // updates the cycle AFTER an iteration's start/advance pulse, and the child
  // starts one cycle after that same pulse (child0Start is registered below),
  // so by the time a child samples the counter for its bound (isEmpty / its own
  // counter init) the register already holds the iteration it is starting --
  // not the one it just left.
  Backedge ivNext = ctx.bb.get(ctx.i32);
  Value iv = ctx.reg(ivNext, term.lb);
  datapath.setCounter(rb.id,
                      iv); // live while the children emit (their outer index)

  // Loop-carried iter-args (a counted reduction container: an outer loop
  // carrying an accumulator into an inner reduction, e.g. `for m: for n: temp
  // += …`). Each latches its init at `start` and advances to the child's
  // next-value when an outer iteration drains (`advanceEdge`, resolved once the
  // children have emitted). Placed before the children so their init reads
  // (Source::Survivor{rb, k}) resolve; the final value is this region's
  // survivor (a sibling store reads it).
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
  // child's done edge -- the outer iteration's completion.
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
  // Restart child 0 one cycle after the (non-empty) outer start pulse, then one
  // cycle after each outer-iteration drain -- registered so the counter has
  // settled to the iteration being started before a child samples it as a bound
  // (a child whose bound is this container's own counter, e.g. `for k in
  // range(i, j)` under `for j`). `gateStart` masks the start of a zero-trip
  // container so no child issues.
  child0Start.setValue(
      ctx.reg(ctx.mux(term.gateStart(ctx, start), ctx.t1, advance), ctx.f1));
  Value ivAdv = ctx.mux(advance, ivStep, iv);
  ivNext.setValue(ctx.mux(start, term.lb, ivAdv));
  // Latch done when the last child of the last outer iteration drains, and
  // clear it on `start` -- so a *retriggered* container (an inner nest re-run
  // by an enclosing container) presents a fresh 0->1 edge each pass. (Harmless
  // for a top-level container: its `start` pulses once, when done is already
  // 0.) A zero-trip container drains no child, so it completes one cycle after
  // `start` instead (delayed so the pulse follows the start-reset, the same
  // empty-region done pattern the leaf uses).
  Value emptyDone = ctx.reg(ctx.andBits(start, term.isEmpty(ctx)), ctx.f1);
  return ctx.holdDone(ctx.orBits(emptyDone, ctx.andBits(lastEdge, last)),
                      start);
}

// A conditional container -- a sequential-wrapper while whose body nests child
// regions (an outer while enclosing an inner while). Each outer iteration
// runs the children once (as emitContainer), but the loop is data-dependent:
// the outer iter-args are frozen survivor registers advanced by the children's
// results, and the loop ends when the combinational continue-condition (a raw
// arith tree over those registers) goes false. A done-based CHECK/RUN FSM times
// it -- one cycle after `start`, and after each outer-iteration drain, the
// condition is re-checked on the settled iter-args; if it holds the children
// (re)start, else the container is done and the iter-args hold their final
// values (a sibling reads them as this region's survivors). No squash / stall:
// the same non-speculative flushing family as a leaf while.
Value HWEmitter::emitConditionalContainer(const uarch::RegionBlock &rb,
                                          Value start) {
  const uarch::Datapath::CarryInfo &wi = dp.carryInfo.find(rb.id)->second;
  unsigned nArgs = wi.inits.size();

  // Outer iter-arg registers = this region's survivors. Each latches its init
  // at `start`, then advances to a child survivor's value when an outer
  // iteration drains (`advanceEdge`, resolved once the children have emitted).
  // Placed before the children so their init reads (Source::Survivor{rb, k})
  // resolve.
  Backedge advanceEdge = ctx.bb.get(ctx.i1);
  SmallVector<Backedge> nextBE =
      setupCarriedIterArgs(rb, wi.inits, start, advanceEdge);

  // CHECK-start pulse: one cycle after `start` or after each outer-iteration
  // drain, when the iter-arg survivor registers have settled. The condition
  // cone reads those (frozen) survivors, so it launches here.
  Value checkStart = ctx.reg(ctx.orBits(start, advanceEdge), ctx.f1);
  // Emit the condition cone (the container's own reads + combinational compute)
  // and get the continue-condition value + its ready latency t_cond. A
  // combinational condition has t_cond == 0 (the CHECK decides in-cycle, as
  // before); a memory-/IP-dependent condition lands t_cond cycles after
  // CHECK-start, so the decision WAITS for it -- the whole point of the
  // sequential CHECK/RUN regime. The survivors do not advance until the body
  // drains (after `child0Start`), so the cone's inputs are stable across the
  // wait.
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
  return ctx.holdDone(donePulse, start);
}

// A guard region (a dcp.select): its two arms run mutually-exclusively under
// the predicate. The then-arm (`children`) runs iff the predicate holds; the
// else-arm
// (`elseChildren`) runs iff it does not (a *dual* guard). The predicate is a
// held value (the condition region's survivor, captured before the guard emits,
// valid at `start`). The not-taken arm's children never issue -- so their
// stores never fire (the predicate reaches every store write-enable
// structurally, via the missing issue pulse, not a per-store gate). An empty
// arm (a then-only guard's absent else, or a pass-through else that yields a
// value but runs no schedule) completes in one cycle: its start pulse IS its
// drain. Either way the region produces a done edge, so an enclosing container
// advances past it in both branches. Run-once: no iteration / iter-args (unlike
// emitConditionalContainer
// -- the predicate is independent of the children).
Value HWEmitter::emitGuard(const uarch::RegionBlock &rb, Value start) {
  const uarch::Datapath::GuardInfo &gi = dp.guardCond.find(rb.id)->second;
  // The predicate as a Source: a scheduled condition region's survivor (a
  // data-dependent scf guard), or the parent container's combinational
  // predicate unit (an affine guard over the counter, reified + emitted by
  // emitCombUnits before this child sequences).
  Value cond = datapath.resolveSource(gi.condition);
  // CHECK one cycle after start (as in emitConditionalContainer): this
  // decouples the completion pulse from the start-clear below -- a skipped
  // guard's done pulse would otherwise coincide with `start` and be masked by
  // the clear.
  Value checkTime = ctx.reg(start, ctx.f1);
  // Two mutually-exclusive arm pulses (the else-arm is the old one-shot
  // `skip`).
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
  // Result-mux: each yielded result is `cond ? then-value : else-value`. Latch
  // each branch's value when that arm drains (only the taken arm fires, so its
  // survivor is fresh and the other holds a stale value the mux ignores), then
  // select by the held predicate. Resolvable now: sequence() above has set
  // every child survivor a then/else value reads. A result-less dual guard has
  // no regionResult entry and skips this.
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
      datapath.setSurvivor(rb.id, k, ctx.mux(cond, thenSurv, elseSurv));
    }
  }
  // Exactly one arm runs, so the region completes on whichever drains. Latch
  // done (a level); clear on start so a retriggered guard re-edges.
  return ctx.holdDone(ctx.orBits(thenDrained, elseDrained), start);
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
  // Compose the top-level siblings by their dependence DAG: independent regions
  // start together (concurrent), the rest gate on their producers' `done`.
  // `emitRegion`'s `retrig` (clear `done` on `start`) keeps the module re-
  // invocable -- a parent that drives it more than once (a loop-over-calls
  // controller) gets a fresh `done` edge per invocation. Harmless for a single
  // invocation: each `done` still rises at drain and holds.
  pa.setOutput("done", composeSiblings(top, pa.getInput("start")));
  // Scalar results: the returning region's survivor register, stable once its
  // region (and thus `done`) has risen -- the cosim samples it at `done`.
  for (const uarch::Result &r : dp.results)
    pa.setOutput(r.name, datapath.resolveSource(r.source));
}

//===----------------------------------------------------------------------===//
// emitModule: interface (ports, extern operator modules) + validation.
//===----------------------------------------------------------------------===//

// The extern operator-module name for an IP-realized unit: its `impl` (the
// operator's RTL module name), but a floating-point compare additionally
// encodes its predicate (one behavioral module per predicate), since `impl`
// alone
// (`fcmp_l1`) does not say which comparison. A compare is the only IP carrying
// a `predicate` attr (copied onto the op by the reifier); integer compare is
// combinational, so an IP compare is always floating-point.
static std::string ipModuleName(const uarch::FuncUnit &u) {
  if (auto pred =
          u.boundOps.front().first->getAttrOfType<arith::CmpFPredicateAttr>(
              "predicate"))
    return u.impl + "_" + arith::stringifyCmpFPredicate(pred.getValue()).str();
  return u.impl;
}

// Reject a datapath outside the emittable subset, with a source diagnostic. The
// preconditions the leaf lowering relies on: a schedulable region set, a trip
// for every cyclic region, a combinational while/guard condition, no in-loop
// store under a while, no cross-region value hand-off (spilling), and an
// emittable realization for every compute unit.
static LogicalResult validateDatapath(func::FuncOp func,
                                      const uarch::Datapath &dp) {
  // Supported subset: top-level sibling regions in program order (composed by
  // sequential hand-off) and container loops whose children are sequenced
  // within one outer iteration (a cross-region result crosses child-to-child as
  // a survivor register). A counted cyclic region needs a trip -- a constant
  // (`tripCount`) or a runtime upper bound (`ubSource`, a dynamic trip); a
  // while (`conditional`) region flushing-pipelines instead.
  if (dp.regions.empty())
    return func.emitError("allo-datapath-to-hw: no schedulable region");
  for (const uarch::RegionBlock &rb : dp.regions)
    if (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional &&
        !rb.tripCount && !rb.ubSource)
      return func.emitError("allo-datapath-to-hw: cyclic region needs a "
                            "constant or dynamic trip");
  // When and how the condition is read decides how settled it must be:
  //   * a flushing leaf while (rb.children empty) samples it the cycle it
  //   issues,
  //     and a guard samples it the cycle it is CHECKed -- both in-cycle, so the
  //     condition must be a stage-0 Unit or a settled prologue Survivor;
  //   * a sequential CHECK/RUN while (rb.children non-empty -- a container or a
  //     wrapped-body leaf) WAITS `t_cond` cycles for the condition
  //     (emitConditionRegion + delayValid) before deciding, so a multi-stage
  //     (memory-/IP-dependent) Unit condition is fine there.
  // A None condition (the reifier left an unschedulable tree raw) is always
  // rejected. A leaf while's multi-*stage body* is independently fine (a load
  // pushes a carried next-value to a later stage; captureWhileResults drains
  // it).
  auto conditionOk = [&](const uarch::Source &s, bool sequential) {
    switch (s.kind) {
    case uarch::Source::Kind::Survivor:
      return true; // a scheduled prologue predicate, valid at the region start
    case uarch::Source::Kind::Unit:
      return sequential || schedT(dp.units[s.id].boundOps.front().first) == 0;
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
  // store's write-enable by the continue-condition (`issue & cond`), so the
  // doomed exit iteration commits nothing -- the same non-speculative rule the
  // loop-carried survivors already follow. (Reaching here, every conditional
  // region has a combinational condition, checked just above, so the gate is a
  // valid stage-0 pulse.)

  // An unresolved (None) input is a cross-region SSA value hand-off (a scalar
  // produced in one region and consumed in another): build leaves the slot
  // empty (see resolveOperand). Reject it cleanly rather than asserting deep in
  // `src` -- spilling is unsupported. Memory-coupled regions (the common case)
  // resolve fully and pass this check.
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

  // Realizability: every compute unit must have an emittable realization. A
  // combinational unit needs an EmitHW comb lowering (`combEmitted`); an IP
  // unit needs a non-empty module name (instantiated below). Fail by op name
  // rather than asserting deep in emission.
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
    // producerOf (and thus regionResult) is keyed per op -- only result 0 is
    // tracked. Reject more than one: per-result tracking is unsupported.
    if (cu.resultPorts.size() > 1)
      return func.emitError("allo-datapath-to-hw: a sub-kernel call returning "
                            "more than one scalar is not yet lowered");
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      // Several serial calls mastering one boundary arg time-share the top port
      // via emitCalls' master mux. A call sharing a boundary arg with a
      // *parent* access still needs that access routed through the same mux (a
      // separate code path, emitAccesses) -- unsupported, so reject it loudly.
      if (ma.isBoundary && !dp.mems[ma.mem].accesses.empty())
        return func.emitError(
            "allo-datapath-to-hw: a boundary argument mastered by a sub-kernel "
            "call and a parent access needs a port-sharing mux -- not yet "
            "lowered");
    }
    // A void indeterminate call (a `while` leaf, no static latency) lowers on
    // the leaf: the region completes on the child's real `done` (fb.callDone),
    // needing no latency. Only a SCALAR-returning indeterminate call stays
    // rejected -- its result lands at a data-dependent cycle readyCycleOf
    // cannot place (a valid-signal result handshake is unsupported).
    if (!cu.latency && !cu.resultPorts.empty())
      return func.emitError(
          "allo-datapath-to-hw: an indeterminate sub-kernel call returning a "
          "scalar has a data-dependent result timing, not yet lowered");
  }
  return success();
}

// Declare an extern operator module for each IP-realized compute unit, named by
// `ipModuleName` and deduplicated across the whole module (`opModules`). Native
// (comb) units emit inline, no extern. One input port per operand (named `a`,
// `b`, `c`, ... at each operand's width -- a unary cast/`sqrt` gets `a` only, a
// binary op `a`+`b`, a compare two operands yielding i1) then the output at the
// result width. The interface follows the realization's stall contract:
// `(a.., clk) -> y` free-running, or `(a.., clk, ce) -> y` when clock-enabled
// (`ce == 0` freezes the pipe in lockstep with the shell). Signature + contract
// are a function of `impl`, so every instance of a given module name shares one
// port shape (dedup-safe). Returns unit id -> its extern module.
static DenseMap<unsigned, Operation *>
declareOperatorModules(func::FuncOp func, const uarch::Datapath &dp,
                       OpBuilder &b, llvm::StringMap<Operation *> &opModules) {
  MLIRContext *ctx = b.getContext();
  Location loc = func.getLoc();
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  DenseMap<unsigned, Operation *> unitModule;
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.comb || u.boundOps.empty())
      continue;
    Operation *srcOp = u.boundOps.front().first;
    assert(u.inputs.size() == srcOp->getNumOperands() &&
           "IP unit input count must match its bound op's operand count");
    IntegerType outW = hwType(u.resultType, b);
    std::string modName = ipModuleName(u);
    Operation *&mod = opModules[modName];
    if (!mod) {
      SmallVector<PortInfo> ep;
      for (unsigned k = 0; k < u.inputs.size(); ++k) {
        IntegerType w = hwType(srcOp->getOperand(k).getType(), b);
        std::string pn(1, static_cast<char>('a' + k));
        ep.push_back({{StringAttr::get(ctx, pn), w, Dir::Input}});
      }
      ep.push_back({{StringAttr::get(ctx, "clk"), b.getI1Type(), Dir::Input}});
      if (u.stall == allo::StallContractEnum::Ce)
        ep.push_back({{StringAttr::get(ctx, "ce"), b.getI1Type(), Dir::Input}});
      ep.push_back({{StringAttr::get(ctx, "y"), outW, Dir::Output}});
      mod = hw::HWModuleExternOp::create(b, loc, StringAttr::get(ctx, modName),
                                         hw::ModulePortInfo(ep));
    }
    unitModule[u.id] = mod;
  }
  return unitModule;
}

llvm::SmallVector<hw::PortInfo>
declareModulePorts(const iface::ModuleInterface &model, OpBuilder &b) {
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;
  MLIRContext *ctx = b.getContext();
  Type i1 = b.getI1Type(), i32 = b.getIntegerType(32);
  // A data port's hw width is its field bit width, so `iType(w)` reproduces
  // `hwType`/`memElemType` for the data ports.
  auto iType = [&](unsigned w) -> Type { return b.getIntegerType(w); };
  SmallVector<PortInfo> ports;
  auto port = [&](const Twine &n, Type t, Dir d) {
    ports.push_back(PortInfo{{StringAttr::get(ctx, n.str()), t, d}});
  };
  port("clk", i1, Dir::Input);
  port("rst", i1, Dir::Input);
  port("start", i1, Dir::Input);
  // Scalar kernel arguments (memref args become memory ports instead).
  for (const iface::Scalar &s : model.scalars)
    port(s.name, iType(s.width), Dir::Input);
  // Stream FIFO ports, input side. All module inputs must stay contiguous at
  // the front of the port list (HWModulePortAccessor maps body args to the
  // first `numInputs` ports positionally), so an input stream's {data, valid}
  // and an output stream's back-pressure {ready} go here; the output side is
  // declared after `done`.
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
  port("done", i1, Dir::Output);
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
  MLIRContext *ctx = b.getContext();
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

  DenseMap<unsigned, Operation *> unitModule =
      declareOperatorModules(func, dp, b, opModules);

  // The port-name model: the single source for every boundary port name, shared
  // by the declaration (declareModulePorts), the manifest, and the cosim
  // harness.
  iface::ModuleInterface model(dp, reads, writes);
  SmallVector<hw::PortInfo> ports = declareModulePorts(model, b);

  hw::ModulePortInfo portInfo(ports);
  StringAttr modName = StringAttr::get(ctx, func.getSymName());

  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, reads, writes, unitModule, bb, i1, i32,
                    callees);
        e.ctx.clk =
            e.ctx.R(seq::ToClockOp::create(ib, loc, pa.getInput("clk")));
        e.ctx.clkRaw = pa.getInput("clk");
        e.ctx.rst = pa.getInput("rst");
        e.emit();
      });

  // Hand the port model back to the caller (the pass): it derives the cosim
  // manifest JSON from it and, for a dataflow container, threads the leaf
  // models to the structural-top emitter -- so the model is the single
  // in-memory port representation, with no IR-attribute manifest to keep in
  // sync.
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
// operator's `sym_name` (the RTL module name) with no symbol clash -- there is
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
  // emits -- the ones the pass declares as dependent -- into the context.
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  // Fold operator timing onto the compute ops and drop the declarations, before
  // any datapath is built or an extern operator module is named.
  stampOperatorTiming(module);

  SmallVector<func::FuncOp> scheduled;
  module.walk([&](func::FuncOp f) {
    if (hasDCPRegions(f))
      scheduled.push_back(f);
  });

  // A memref/array function result is unsupported: hardware writes an output
  // through a memory port, not a returned value (the upstream
  // buffer-results-to-out-params prepass that would rewrite it is deliberately
  // not run). Reject cleanly here rather than asserting deep in the datapath
  // builder (recordResults), which runs while the Datapath is constructed
  // below.
  for (func::FuncOp f : scheduled)
    if (auto ret = dyn_cast<func::ReturnOp>(f.front().getTerminator()))
      for (Value v : ret.getOperands())
        if (isa<MemRefType>(v.getType())) {
          logging::error(logging::Stage::Emit, f)
              << "Returning a memref is unsupported; write the result through "
                 "an output argument (out-parameter) instead";
          return failure();
        }

  std::unique_ptr<BindingPolicy> policy = bindingPolicyFor(binding);
  if (!policy)
    return module.emitError("allo-datapath-to-hw: unknown binding policy '")
           << binding << "'";

  // Emission is rooted at the top function and runs bottom-up over the call
  // DAG: recurse into each callee before emitting `f`, so a container always
  // finds its children already emitted and registered. A leaf (no scheduled
  // callees) emits its own datapath; a container (dataflow or sequential) wires
  // its already-emitted children into a structural top. Mirrors the scheduler,
  // which also roots at `top` and schedules callees first.
  llvm::StringMap<func::FuncOp> byName;
  for (func::FuncOp f : scheduled)
    byName[f.getSymName()] = f;
  func::FuncOp topFunc = byName.lookup(top);
  if (!topFunc)
    return module.emitError("allo-datapath-to-hw: top function '")
           << top << "' is not a scheduled function";

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  // Callee tables, keyed by symbol name -- leaf kernels plus containers emitted
  // so far. A container is composed exactly like a leaf, so both live here.
  llvm::StringMap<hw::HWModuleOp> modules;
  llvm::StringMap<iface::ModuleInterface> ifaceModels;
  llvm::StringSet<> visited;

  auto registerModule = [&](StringRef name, hw::HWModuleOp mod,
                            iface::ModuleInterface model) {
    interfaces[name] = model.toJSON();
    modules[name] = mod;
    ifaceModels[name] = std::move(model);
  };

  // Post-order over the call DAG (acyclic -- the frontend rejects recursion). A
  // self-parameter recursive lambda (`self(self, ...)`) keeps the traversal
  auto emitOne = [&](auto &self, func::FuncOp f) -> LogicalResult {
    if (!visited.insert(f.getSymName()).second)
      return success(); // a shared callee already emitted
    // Children first: emit every scheduled callee (a leaf call misses
    // `byName`). A callee is referenced by a `func.call` (async spawn /
    // structural-top compose) or a `dcp.instance` (a leaf CallUnit) --
    // both must recurse so the child is emitted + registered before its caller.
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

    // A container wires its children into a structural top, inserted at the
    // module-body start so the outermost -- emitted last -- lands at position
    // 0. A leaf emits its own datapath before its source func. The one router
    // read: a CONCURRENT container (a dataflow network of `await` spawns,
    // stamped `dcp.determinacy = concurrent` by the reifier) wires a structural
    // top; every other kernel -- a leaf, or a non-concurrent container whose
    // sync calls reified to `dcp.instance`s (so it holds no `func.call`) --
    // lowers as a leaf datapath below.
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
      Datapath dp(f, *policy, callees);
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
