/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Lower the L2 datapath of a scheduled function to a structural `hw.module`
// (comb datapath, seq registers, an iteration counter, internal seq.hlmem
// storage, and bare external memory ports), ready for CIRCT's `lower-seq-to-sv`
// + `export-verilog`.
//
// This file holds the shared free helpers, the `EmitContext` primitives, the
// `HWEmitter` **orchestrator**, `emitModule` (port/interface setup + module
// validation), and the pass. The two halves of the body -- control (G) and
// datapath (F) -- live in ControlEmitter.cpp and DatapathEmitter.cpp; their
// interface is HWEmit.h.
//===----------------------------------------------------------------------===//

#include "allo/Microarch/HWEmitter.h"
#include "allo/IR/AlloOps.h"
#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/Interface.h"
#include "allo/Scheduling/OperatorLibrary.h" // isNativeImpl
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h" // arith::CmpIPredicate (cmpi predicate)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::allo;
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

unsigned schedT(Operation *op) {
  return cast<IntegerAttr>(op->getAttr("start")).getInt();
}

// Integer/logic mnemonics EmitHW lowers to a native `comb` primitive. The
// single source of truth for `emitCompute`'s coverage; a native op outside this
// set has no EmitHW lowering yet (realizability errors, see `emitModule`).
bool combEmitted(StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases({"addi", "subi", "muli", "andi", "ori", "xori"}, true)
      .Cases({"extsi", "extui", "trunci", "index_cast"}, true)
      .Cases({"cmpi", "select", "shli", "shrsi", "shrui"}, true)
      .Cases({"divsi", "divui"}, true)
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
    return s ? datapath.src(s) : ctx.konst(ctx.i32, c);
  };
  Value lb = bound(rb.lbSource, rb.lb), step = bound(rb.stepSource, rb.step);
  if (rb.ubSource)
    return Terminator::counted(lb, datapath.src(rb.ubSource), step,
                               /*dynamic=*/true);
  if (rb.tripCount)
    return Terminator::counted(
        lb, ctx.konst(ctx.i32, rb.lb + *rb.tripCount * rb.step), step,
        /*dynamic=*/false);
  return {};
}

// One imperative path for every leaf region (counted / dynamic-trip / while):
// P1 control -> P2 datapath -> P3 (resolve the F->G condition / capture results
// / done). The regimes differ only in the Terminator (P1) and the survivor
// mechanism (P3b, captureResults); the shared skeleton reads as a linear
// sequence.
Value HWEmitter::emitRegion(const uarch::RegionBlock &rb, Value start,
                            bool retrig) {
  if (!rb.children.empty()) {
    if (rb.guard)
      return emitGuard(rb, start);
    return rb.conditional ? emitConditionalContainer(rb, start)
                          : emitContainer(rb, start);
  }
  assert(!rb.guard && "a guard region has no children to predicate");

  // P1 (control): the terminator + the control skeleton. A while's
  // continue-condition is a datapath value not yet emitted, so it rides a
  // backedge resolved in P3a; a counted loop's bound resolves now.
  Backedge condBE;
  Terminator term;
  if (rb.conditional) {
    condBE = ctx.bb.get(ctx.i1);
    term = Terminator::conditional(condBE, ctx.zero32, ctx.one32);
  } else {
    term = terminatorOf(rb);
  }

  // Latency-insensitive shell: a region with stream accesses gets two signals
  // (both F->G datapath values, resolved after P2, so they ride backedges) --
  // `chainEnable` (~output-full) drives ctx.regionEnable so every shift chain +
  // the done drain freeze coherently on back-pressure (preserving tap
  // alignment), and `issueEnable` (~output-full & inputs-available) gates issue
  // so an empty input is a bubble, not a freeze. A stream-free region keeps
  // enable == true and regionEnable null -- byte-identical to before.
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

  // P2 (datapath): -> feedback (the store drain + shell signals; a while's
  // condition + its next-value producers are now emitted). The shell backedges
  // are resolved at the very end (after the done drain, which also reads
  // ctx.regionEnable) -- a setValue before that last use would not RAUW it.
  DatapathFeedback fb = datapath.emit(rb, rc.issue);

  // P3a: resolve the F->G condition backedge now the datapath has emitted it,
  // and re-point the terminator at the resolved value -- `setValue` RAUWs and
  // erases the placeholder, so a *later* read of `term.cond` (lastIssuePulse's
  // exit test) must use the real condition, not the dead backedge handle.
  if (rb.conditional) {
    Value cond = datapath.src(dp.carryInfo.find(rb.id)->second.condition);
    condBE.setValue(cond);
    term.cond = cond;
  }

  // P3b (survivors): capture the region's results (returning their drain stage)
  // and pin the last iteration's issue pulse -- the one pulse the done and the
  // captures share.
  Value lastIssue = lastIssuePulse(rc, term);
  unsigned resultDrain = captureResults(rb, rc, lastIssue, start);
  unsigned drainStage = std::max(fb.storeDrain, resultDrain);

  // P3c (control): the completion signal. A counted leaf that is empty (lb >=
  // ub
  // -- a static `range(1,1)` or a runtime zero-trip) issues nothing, so it
  // completes on `start` via `emptyDone` (else its store-drain done never fires
  // -> deadlock). Folds away for a statically non-empty loop. A while / acyclic
  // never reports empty here (null).
  Value emptyDone =
      (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional)
          ? ctx.andBits(start, term.isEmpty(ctx))
          : Value();
  // emitDone's drain chain must still see the shell's enable; leave it after.
  Value done =
      control.emitDone(drainStage, lastIssue, emptyDone, start, retrig);
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
                        : captureCountedResults(rb, lastIssue);
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
                                          Value lastIssue) {
  auto it = dp.regionResult.find(rb.id);
  if (it == dp.regionResult.end())
    return 0;
  unsigned maxStage = 0;
  for (auto [k, rs] : llvm::enumerate(it->second)) {
    if (rs.kind == uarch::Source::Kind::None)
      continue; // an untracked result: no survivor (asserts if read)
    // Capture the result on the cycle it lands (its ready cycle after the last
    // issue); the region's done drains on the latest-landing result.
    unsigned stage = datapath.readyCycle(rs);
    Value cap = ctx.delayValid(lastIssue, stage);
    Value res = datapath.src(rs);
    datapath.setSurvivor(rb.id, k,
                         ctx.enabledReg(res, cap, ctx.konst(res.getType(), 0)));
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
  Value cond =
      datapath.src(wi.condition); // memoized (== the P3a-resolved value)
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
    Value next = datapath.src(nextS);
    Value init = wi.inits[k].kind == uarch::Source::Kind::None
                     ? ctx.konst(next.getType(), 0)
                     : datapath.src(wi.inits[k]);
    datapath.setSurvivor(rb.id, k, ctx.latchReg(init, next, start, advance));
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// Run `regions` in program order, each starting on its predecessor's done edge
// (the first on `start`); returns the last region's done. The shared done-based
// sequencer -- func-scope siblings (a single pass) and a container's children
// (once per outer iteration) are the same "start k+1 when k drains" pattern.
Value HWEmitter::sequence(llvm::ArrayRef<uarch::RegionId> regions, Value start,
                          bool retrig) {
  Value done;
  Value startK = start;
  for (auto [i, rid] : llvm::enumerate(regions)) {
    done = emitRegion(dp.regions[rid], startK, retrig);
    if (i + 1 < regions.size())
      startK = ctx.risingEdge(done); // the next region starts on this done edge
  }
  return done;
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
  int64_t trip = *rb.tripCount;
  // A container has a compile-time trip, so its lb/step are compile-time too (a
  // runtime range start/stride yields no static trip); the constant counter
  // path below is exhaustive.
  assert(!rb.lbSource && !rb.stepSource &&
         "container with a runtime lb/step (no constant trip)");
  // The outer counter is the source IV: init `lb`, advance by `step`, so the
  // children read the real outer index (Source::Counter).
  Value lbV = ctx.konst(ctx.i32, rb.lb), stepV = ctx.konst(ctx.i32, rb.step);
  Backedge ivNext = ctx.bb.get(ctx.i32);
  Value iv = ctx.reg(ivNext, lbV);
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
  SmallVector<Backedge> nextBE;
  auto ci = dp.carryInfo.find(rb.id);
  if (ci != dp.carryInfo.end())
    for (auto [k, initS] : llvm::enumerate(ci->second.inits)) {
      assert(initS && "a container iter-arg has no resolvable init");
      Value init = datapath.src(initS);
      Backedge nb = ctx.bb.get(init.getType());
      nextBE.push_back(nb);
      datapath.setSurvivor(rb.id, k,
                           ctx.latchReg(init, nb, start, advanceEdge));
    }

  // Child 0 starts on `child0Start` (resolved below); `lastEdge` is the last
  // child's done edge -- the outer iteration's completion.
  Backedge child0Start = ctx.bb.get(ctx.i1);
  Value lastEdge = ctx.risingEdge(sequence(rb.children, child0Start,
                                           /*retrig=*/true));
  advanceEdge.setValue(lastEdge); // advance the iter-args on each outer drain
  for (auto [k, nb] : llvm::enumerate(nextBE))
    nb.setValue(datapath.src(ci->second.nexts[k]));
  Value last = ctx.icmpEq(iv, rb.lb + (trip - 1) * rb.step);
  Value advance = ctx.andBits(lastEdge, ctx.notBit(last));
  // Restart child 0 on the outer start pulse, then on each outer-iteration
  // drain.
  child0Start.setValue(ctx.mux(start, ctx.t1, advance));
  Value ivp1 = ctx.R(comb::AddOp::create(ctx.b, ctx.loc, iv, stepV, false));
  Value ivAdv = ctx.mux(advance, ivp1, iv);
  ivNext.setValue(ctx.mux(start, lbV, ivAdv));
  // Latch done when the last child of the last outer iteration drains, and
  // clear it on `start` -- so a *retriggered* container (an inner nest re-run
  // by an enclosing container) presents a fresh 0->1 edge each pass. (Harmless
  // for a top-level container: its `start` pulses once, when done is already
  // 0.)
  Backedge doneNext = ctx.bb.get(ctx.i1);
  Value done = ctx.reg(doneNext, ctx.f1);
  Value setDone = ctx.andBits(lastEdge, last);
  doneNext.setValue(ctx.mux(start, ctx.f1, ctx.mux(setDone, ctx.t1, done)));
  return done;
}

// Lower a conditional container's continue-condition -- an unscheduled
// combinational arith tree over the outer iter-args -- to comb. A block-arg is
// a container iter-arg, resolved to its survivor register; a literal is a
// constant; any arith op reuses emitCompute (the same mapping the scheduled
// units use). The wrapper body is unscheduled precisely because it is
// straight-line combinational, so every internal op is `combEmitted`.
Value HWEmitter::evalRawArith(Value v, const uarch::RegionBlock &container) {
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    // Block-arg 0 is the iteration counter (a guard's affine predicate reads
    // it); an iter-arg (>= 1) reads the container's frozen survivor register (a
    // while-wrapper's continue-condition).
    unsigned n = barg.getArgNumber();
    if (n == 0)
      return datapath.src(
          uarch::Source{uarch::Source::Kind::Counter, container.id, 0});
    return datapath.src(
        uarch::Source{uarch::Source::Kind::Survivor, container.id, n - 1});
  }
  Operation *def = v.getDefiningOp();
  assert(def && "raw condition operand with no defining op");
  if (auto cst = dyn_cast<arith::ConstantOp>(def))
    return ctx.konst(hwType(cst.getType(), ctx.b),
                     cast<IntegerAttr>(cst.getValue()).getInt());
  SmallVector<Value> operands;
  for (Value o : def->getOperands())
    operands.push_back(evalRawArith(o, container));
  StringRef kind = def->getName().stripDialect();
  assert(combEmitted(kind) &&
         "container condition uses a non-combinational op");
  return emitCompute(ctx.b, ctx.loc, kind, operands,
                     hwType(def->getResult(0).getType(), ctx.b), def);
}

// A conditional container -- a sequential-wrapper while whose body nests child
// regions (test_while_with_nested_while's outer while). Each outer iteration
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
  SmallVector<Backedge> nextBE;
  for (unsigned k = 0; k < nArgs; ++k) {
    assert(wi.inits[k].kind != uarch::Source::Kind::None &&
           "a conditional container's iter-arg has no resolvable init");
    Value init = datapath.src(wi.inits[k]);
    Backedge nb = ctx.bb.get(init.getType());
    nextBE.push_back(nb);
    datapath.setSurvivor(rb.id, k, ctx.latchReg(init, nb, start, advanceEdge));
  }

  // The combinational continue-condition over the iter-arg registers.
  Value cond = evalRawArith(wi.condValue, rb);

  // CHECK pulse: one cycle after `start` or after each outer-iteration drain,
  // when the iter-arg registers have settled. Start the children only if the
  // condition holds; otherwise the container completes this cycle.
  Value checkTime = ctx.reg(ctx.orBits(start, advanceEdge), ctx.f1);
  Value child0Start = ctx.andBits(checkTime, cond);
  Value donePulse = ctx.andBits(checkTime, ctx.notBit(cond));

  // Sequence the children within one outer iteration; the last child's drain
  // edge advances the iter-args (resolving the survivor next-values) and drives
  // the next CHECK.
  Value lastEdge =
      ctx.risingEdge(sequence(rb.children, child0Start, /*retrig=*/true));
  advanceEdge.setValue(lastEdge);
  for (unsigned k = 0; k < nArgs; ++k)
    nextBE[k].setValue(datapath.src(wi.nexts[k]));

  // Latch done (a level) when the condition first fails; clear on `start` so a
  // retriggered container presents a fresh edge each pass (harmless top-level).
  Backedge dNext = ctx.bb.get(ctx.i1);
  Value done = ctx.reg(dNext, ctx.f1);
  Value set = ctx.mux(donePulse, ctx.t1, done);
  dNext.setValue(ctx.mux(start, ctx.f1, set));
  return done;
}

// A guard region (a dcp.select): its children run once iff the predicate holds.
// The predicate is a held value (the condition region's survivor, captured
// before the guard emits, valid at `start`). When it holds, child 0 starts and
// the region drains on the last child; when it does not, the region completes
// in one cycle and the children never issue -- so their stores never fire (the
// predicate reaches every store write-enable structurally, via the missing
// issue pulse, not a per-store gate). Either way the region produces a done
// edge, so an enclosing container advances past it in both branches. Run-once:
// no iteration / iter-args (unlike emitConditionalContainer -- the predicate is
// independent of the children).
Value HWEmitter::emitGuard(const uarch::RegionBlock &rb, Value start) {
  const uarch::Datapath::GuardInfo &gi = dp.guardCond.find(rb.id)->second;
  // The predicate: a scheduled condition region's survivor, or a raw arith tree
  // over the enclosing container's counter (evalRawArith against `rb.parent`).
  Value cond = gi.condition.kind != uarch::Source::Kind::None
                   ? datapath.src(gi.condition)
                   : evalRawArith(gi.condValue, dp.regions[*rb.parent]);
  // CHECK one cycle after start (as in emitConditionalContainer): this
  // decouples the completion pulse from the start-clear below -- a skipped
  // guard's done pulse would otherwise coincide with `start` and be masked by
  // the clear.
  Value checkTime = ctx.reg(start, ctx.f1);
  Value child0Start = ctx.andBits(checkTime, cond);
  Value skip = ctx.andBits(checkTime,
                           ctx.notBit(cond)); // predicate false: one-shot done
  // Children run once (retrig so a re-entered guard presents fresh edges each
  // enclosing pass); `drained` is the last child's completion edge.
  Value drained =
      ctx.risingEdge(sequence(rb.children, child0Start, /*retrig=*/true));
  Value donePulse = ctx.orBits(skip, drained);
  // Latch done (a level); clear on start so a retriggered guard re-edges.
  Backedge dNext = ctx.bb.get(ctx.i1);
  Value done = ctx.reg(dNext, ctx.f1);
  Value set = ctx.mux(donePulse, ctx.t1, done);
  dNext.setValue(ctx.mux(start, ctx.f1, set));
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
  pa.setOutput("done", sequence(top, pa.getInput("start"), /*retrig=*/false));
  // Scalar results: the returning region's survivor register, stable once its
  // region (and thus `done`) has risen -- the cosim samples it at `done`.
  for (const uarch::Result &r : dp.results)
    pa.setOutput(r.name, datapath.src(r.source));
}

//===----------------------------------------------------------------------===//
// emitModule: interface (ports, extern operator modules) + validation.
//===----------------------------------------------------------------------===//

// The extern operator-module name for an IP-realized unit: usually its `impl`,
// but a floating-point compare additionally encodes its predicate (one
// behavioral module per predicate), since `impl` alone (`fcmp_l1`) does not say
// which comparison. The predicate is preserved onto the op by the reifier.
static std::string ipModuleName(const uarch::FuncUnit &u) {
  if (u.opType == "cmpf") {
    auto pred = cast<arith::CmpFPredicateAttr>(
                    u.boundOps.front().first->getAttr("predicate"))
                    .getValue();
    return u.impl + "_" + arith::stringifyCmpFPredicate(pred).str();
  }
  return u.impl;
}

// Whether \p v is a combinational arith tree over block-args (iter-args) and
// constants -- i.e. a conditional container's continue-condition the emitter
// can evaluate (evalRawArith). A memory-/IP-dependent wrapper condition is not.
static bool isCombArithTree(Value v) {
  if (isa<BlockArgument>(v))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ConstantOp>(def))
    return true;
  return combEmitted(def->getName().stripDialect()) &&
         llvm::all_of(def->getOperands(), isCombArithTree);
}

// Emit an hw.module for one scheduled function's datapath. Returns failure with
// a diagnostic if the datapath is outside the supported first-cut subset.
// `opModules` caches extern operator modules across functions.
static FailureOr<std::pair<hw::HWModuleOp, iface::ModuleInterface>>
emitModule(func::FuncOp func, uarch::Datapath &dp, OpBuilder &b,
           llvm::StringMap<Operation *> &opModules) {
  MLIRContext *ctx = b.getContext();
  Location loc = func.getLoc();

  // Supported subset: top-level sibling regions in program order (composed by
  // sequential hand-off) and container loops whose children are sequenced
  // within one outer iteration (a cross-region result crosses child-to-child as
  // a survivor register). A counted cyclic region needs a trip -- a constant
  // (`tripCount`) or a runtime upper bound (`ubSource`, a dynamic trip); a
  // while
  // (`conditional`) region flushing-pipelines instead (emitWhileRegion).
  if (dp.regions.empty())
    return func.emitError("allo-datapath-to-hw: no schedulable region");
  for (const uarch::RegionBlock &rb : dp.regions)
    if (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional &&
        !rb.tripCount && !rb.ubSource)
      return func.emitError("allo-datapath-to-hw: cyclic region needs a "
                            "constant or dynamic trip");
  // The while lowering requires a *combinational* continue-condition: the
  // flushing controller reads it the cycle each iteration issues (emitPipelined
  // / the container CHECK/RUN FSM clears `running` in-cycle). A leaf while's
  // body spanning several stages is fine (a load pushes a carried next-value to
  // stage 1; captureWhileResults drains it), but a memory-/IP-dependent
  // *condition* lands at a later stage (leaf: schedT > 0; container: a
  // non-arith cond op), so the gate would sample a stale value -- deferred.
  for (const uarch::RegionBlock &rb : dp.regions) {
    if (!rb.conditional)
      continue;
    const uarch::Datapath::CarryInfo &wi = dp.carryInfo.find(rb.id)->second;
    if (!rb.children.empty()) {
      // A conditional container: its unscheduled continue-condition must be a
      // combinational arith tree over the iter-arg registers (evalRawArith).
      if (!isCombArithTree(wi.condValue))
        return func.emitError(
            "allo-datapath-to-hw: a sequential-wrapper while with a "
            "non-combinational condition is not yet lowered");
      continue;
    }
    if (wi.condition.kind == uarch::Source::Kind::Unit &&
        schedT(dp.units[wi.condition.id].boundOps.front().first) > 0)
      return func.emitError("allo-datapath-to-hw: a while loop with a non-"
                            "combinational (memory-/IP-dependent) condition is "
                            "not yet lowered");
  }
  // A guard (dcp.select) with an unscheduled raw-arith predicate: emitGuard
  // evaluates it combinationally (evalRawArith), so a memory-/IP-dependent
  // guard that did NOT lower to a scheduled condition region
  // (`guardCond.condition` None) must be a combinational arith tree over the
  // counter -- else reject.
  for (const uarch::RegionBlock &rb : dp.regions) {
    if (!rb.guard)
      continue;
    const uarch::Datapath::GuardInfo &gi = dp.guardCond.find(rb.id)->second;
    if (gi.condition.kind == uarch::Source::Kind::None &&
        !isCombArithTree(gi.condValue))
      return func.emitError("allo-datapath-to-hw: a guard with a "
                            "non-combinational predicate is not yet lowered");
  }
  // A while region's loop-carried results become frozen survivor registers; an
  // in-loop store would commit at the exit iteration too (the store-enable is
  // not yet condition-gated), so reject one until that gate lands.
  for (const uarch::MemUnit &m : dp.mems)
    for (const uarch::MemUnit::Access &acc : m.accesses)
      if (acc.isWrite && dp.regions[acc.region].conditional)
        return func.emitError("allo-datapath-to-hw: a while loop with an "
                              "in-loop store is not yet lowered");

  // An unresolved (None) input is a cross-region SSA value hand-off (a scalar
  // produced in one region and consumed in another): build leaves the slot
  // empty (see resolveOperandDCP). Reject it cleanly rather than asserting deep
  // in `src` -- spilling is a later step. Memory-coupled regions (the common
  // case) resolve fully and pass this check.
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
  // native keyword (`comb`/`builtin`) needs an EmitHW comb lowering
  // (`combEmitted`); otherwise `impl` is an IP module name, instantiated below.
  // Fail by op name rather than asserting deep in emission.
  for (const uarch::FuncUnit &u : dp.units) {
    if (u.impl.empty())
      return func.emitError("allo-datapath-to-hw: operator '")
             << u.opType
             << "' has no realization; set 'impl' (a native keyword or an IP "
                "module name) on its operator library row";
    if (u.impl == "hwarith")
      return func.emitError(
                 "allo-datapath-to-hw: 'impl: hwarith' emission is not yet "
                 "implemented (operator '")
             << u.opType << "')";
    if (allo::isNativeImpl(u.impl) && !combEmitted(u.opType))
      return func.emitError("allo-datapath-to-hw: operator '")
             << u.opType
             << "' has no native EmitHW lowering; provide an IP (set 'impl' to "
                "an IP module name) or add native support";
  }

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

  // Ports: clk, rst, start, per-read data in; done, per-read addr out,
  // per-write {addr, data, we} out.
  using PortInfo = hw::PortInfo;
  using Dir = hw::ModulePort::Direction;

  // Declare an extern operator module for each IP-realized compute unit, named
  // by `ipModuleName` and deduplicated across the whole module. Native (comb)
  // units emit inline below, no extern. Inputs are the operand width and the
  // output the result width -- equal for a binary op, but a compare takes two
  // operands and yields i1. The interface follows the realization's stall
  // contract: `(a, b, clk) -> y` free-running, or `(a, b, clk, ce) -> y` when
  // clock-enabled (`ce == 0` freezes the pipe in lockstep with the shell). The
  // contract is a function of `impl`, so every instance of a given module name
  // shares one port shape (dedup-safe).
  DenseMap<unsigned, Operation *> unitModule;
  for (const uarch::FuncUnit &u : dp.units) {
    if (allo::isNativeImpl(u.impl) || u.boundOps.empty())
      continue;
    IntegerType inW =
        hwType(u.boundOps.front().first->getOperand(0).getType(), b);
    IntegerType outW = hwType(u.resultType, b);
    std::string modName = ipModuleName(u);
    Operation *&mod = opModules[modName];
    if (!mod) {
      SmallVector<PortInfo> ep{
          PortInfo{{StringAttr::get(ctx, "a"), inW, Dir::Input}},
          PortInfo{{StringAttr::get(ctx, "b"), inW, Dir::Input}},
          PortInfo{{StringAttr::get(ctx, "clk"), b.getI1Type(), Dir::Input}}};
      if (allo::stallContract(u.impl) == allo::StallContract::ClockEnable)
        ep.push_back({{StringAttr::get(ctx, "ce"), b.getI1Type(), Dir::Input}});
      ep.push_back({{StringAttr::get(ctx, "y"), outW, Dir::Output}});
      mod = hw::HWModuleExternOp::create(b, loc, StringAttr::get(ctx, modName),
                                         hw::ModulePortInfo(ep));
    }
    unitModule[u.id] = mod;
  }

  // The port-name model: the single source for every boundary port name (each
  // interface's concrete field names), shared by the declaration below, the
  // manifest, and the cosim harness. A data port's hw width is its field bit
  // width, so `iType(w)` reproduces `hwType`/`memElemType` for the data ports.
  iface::ModuleInterface model(dp, reads, writes);
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

  hw::ModulePortInfo portInfo(ports);
  StringAttr modName = StringAttr::get(ctx, func.getSymName());

  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, reads, writes, unitModule, bb, i1, i32);
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

LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               llvm::StringMap<std::string> &interfaces) {
  // Called directly (not via the pass manager), so load the dialects this
  // emits -- the ones the pass declares as dependent -- into the context.
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  SmallVector<func::FuncOp> scheduled;
  module.walk([&](func::FuncOp f) {
    if (hasDCPRegions(f))
      scheduled.push_back(f);
  });

  std::unique_ptr<BindingPolicy> policy = bindingPolicyFor(binding);
  if (!policy)
    return module.emitError("allo-datapath-to-hw: unknown binding policy '")
           << binding << "'";

  // A dataflow container (its body spawns concurrent processes via `func.call`)
  // is not a datapath: emit the leaf processes first as their own hw.modules,
  // then wire them structurally (§7.4, Route S). Everything else is a leaf
  // compute kernel emitted directly.
  SmallVector<func::FuncOp> leaves, containers;
  for (func::FuncOp f : scheduled)
    (isDataflowContainer(f) ? containers : leaves).push_back(f);

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  llvm::StringMap<hw::HWModuleOp> leafModules;
  llvm::StringMap<iface::ModuleInterface> leafInterfaces;
  for (func::FuncOp f : leaves) {
    Datapath dp(f, *policy);
    LLVM_DEBUG({
      llvm::dbgs() << "// datapath for @" << f.getSymName() << "\n";
      dp.dump(llvm::dbgs());
    });
    b.setInsertionPoint(f);
    auto pairOr = emitModule(f, dp, b, opModules);
    if (failed(pairOr))
      return failure();
    hw::HWModuleOp emitted = pairOr->first;
    iface::ModuleInterface &model = pairOr->second;
    // auto [emitted, model] = *pairOr;
    leafModules[f.getSymName()] = emitted;
    interfaces[f.getSymName()] = model.toJSON();
    leafInterfaces[f.getSymName()] = std::move(model);
  }
  // The structural top must be the *first* hw.module in the output (the cosim
  // reader keys on it), so insert each container's top module at the start of
  // the module body, before the leaf modules it instantiates.
  for (func::FuncOp f : containers) {
    b.setInsertionPointToStart(module.getBody());
    std::string json;
    if (failed(emitDataflowTop(f, leafModules, leafInterfaces, b, &json)))
      return failure();
    interfaces[f.getSymName()] = std::move(json);
  }
  for (func::FuncOp f : scheduled)
    f.erase();

  // Drop the now-consumed dcp.operator declarations so the module holds only hw
  // ops for CIRCT's Verilog export.
  SmallVector<Operation *> spent;
  module.walk([&](dcp::DCPathOperatorOp op) { spent.push_back(op); });
  for (Operation *op : spent)
    op->erase();
  return success();
}

} // namespace mlir::allo::uarch
