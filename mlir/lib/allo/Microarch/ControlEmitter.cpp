/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "circt/Dialect/Comb/CombOps.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

RegionControl ControlEmitter::emitPipelineControl(const uarch::RegionBlock &rb,
                                                  const Terminator &term,
                                                  Value start,
                                                  const StallShell &sh) const {
  if (rb.kind == uarch::RegionBlock::Kind::Acyclic)
    return emitAcyclic(rb.id, start, /*topLevel=*/!rb.parent, sh);
  assert(rb.ii && "a pipelined region reached control emission with no II");
  auto rc = emitPipelined(rb.id, *rb.ii, term, start, sh);
  // Label the counter register after the source loop variable. A loop whose
  // IV lost its name still reads as this region's counter.
  nameValue(rc.counter, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                               : rb.counterName);
  rc.scaledCounters = emitScaledCounters(
      rb, term, /*bypassStart=*/Value(),
      [&](Value cur, Value stepped, Value init) {
        // Exactly `iterNext` above, with `lb` and `step` scaled.
        return c.mux(rc.running, c.mux(rc.issue, stepped, cur), init);
      });
  return rc;
}

// \p update is passed rather than re-derived so that each family's scaled
// counters are written beside the counter they have to track. Drifting from
// that counter is the only way these can be wrong.
llvm::SmallVector<Value> ControlEmitter::emitScaledCounters(
    const uarch::RegionBlock &rb, const Terminator &term, Value bypassStart,
    llvm::function_ref<Value(Value, Value, Value)> update) const {
  llvm::SmallVector<Value> scaled;
  if (rb.addrStrides.empty())
    return scaled;
  auto ty = cast<IntegerType>(term.lb.getType());
  // Whether each slot's register wraps THIS advance, which is what a digit
  // above it advances on. A carry slot always precedes its consumer, so the
  // signal exists by the time it is read.
  llvm::SmallVector<Value> wrapped(rb.addrStrides.size());
  for (auto [slot, s] : llvm::enumerate(rb.addrStrides)) {
    Backedge next = c.bb.get(ty);
    Value init = c.konst(ty, s.init);
    Value reg = c.reg(next, init);
    nameValue(reg, regionSignal(rb.id, "addr" + std::to_string(slot)));
    // The same start-cycle bypass the counter takes, for the same reason: a
    // call region's first pass reads its index on `start` itself.
    Value cur = bypassStart ? c.mux(bypassStart, init, reg) : reg;
    Value raw = cur;
    if (s.step)
      raw =
          c.R(comb::AddOp::create(c.b, c.loc, raw, c.konst(ty, s.step), false));
    if (s.hasCarry) {
      assert(wrapped[s.carry] &&
             "a digit's carry slot is not emitted before it");
      raw = c.R(comb::AddOp::create(
          c.b, c.loc, raw,
          c.mux(wrapped[s.carry], c.konst(ty, s.bump), c.konst(ty, 0)), false));
    }
    Value stepped = raw;
    if (s.wrap) {
      // Unsigned throughout: a stride register holds an index, and a digit is a
      // residue, so neither is ever negative. Counting DOWN, the register goes
      // out of range by wrapping around zero, which is exactly `raw > cur`.
      Value wrapKonst = c.konst(ty, s.wrap);
      wrapped[slot] = c.R(comb::ICmpOp::create(
          c.b, c.loc,
          s.down ? comb::ICmpPredicate::ugt : comb::ICmpPredicate::uge, raw,
          s.down ? cur : wrapKonst, false));
      Value fixed =
          s.down ? c.R(comb::AddOp::create(c.b, c.loc, raw, wrapKonst, false))
                 : c.R(comb::SubOp::create(c.b, c.loc, raw, wrapKonst, false));
      stepped = c.mux(wrapped[slot], fixed, raw);
    }
    next.setValue(update(cur, stepped, init));
    scaled.push_back(cur);
  }
  return scaled;
}

// The one pipelined control skeleton, covering three regimes that differ only
// in their `Terminator` and (for II>1) a phase counter:
//   * free-running (II==1, counted): one iteration issued every cycle;
//   * modulo (II>1, counted): one issued every II cycles, gated by a [0,II)
//     phase counter (in-flight drain via the valid chain);
//   * while (II==1, conditional): a non-speculative flushing pipeline,
//     terminated by the condition going false.
// `running` is set by `start` and cleared the cycle the last iteration issues;
// the iteration counter advances on issue (feeding the counted bound test and
// the datapath's iteration-0 recurrence-init injection, mostly dead for a
// while, which rarely indexes by iteration). A conditional terminator is
// non-speculative (II >= t_cond, so no doomed iteration issues -> no squash)
// and stall-free (fixed-latency memory, no FIFO).
RegionControl ControlEmitter::emitPipelined(unsigned region, int64_t ii,
                                            const Terminator &term, Value start,
                                            const StallShell &sh) const {
  // G's half of H: a rigid region issues unconditionally.
  Value enable = sh ? sh.issueEnable : c.t1;
  auto runNext = c.bb.get(c.i1);
  Value running = c.reg(runNext, c.f1);
  nameValue(running, regionSignal(region, "run"));
  // The ungated per-cycle issue *desire*: modulo (II>1) a phase counter [0,II)
  // gates it to once per II; II==1 (and a while) wants to issue every running
  // cycle. The stall shell then gates this by `enable` below.
  Value wantIssue = running;
  if (ii > 1) {
    auto phaseNext = c.bb.get(c.i32);
    Value phase = c.reg(phaseNext, c.zero32);
    nameValue(phase, regionSignal(region, "phase"));
    wantIssue = c.R(
        comb::AndOp::create(c.b, c.loc, running, c.icmpEq(phase, 0), false));
    Value phasep1 = c.R(comb::AddOp::create(c.b, c.loc, phase, c.one32, false));
    Value phaseAdv = c.mux(c.icmpEq(phase, ii - 1), c.zero32, phasep1);
    // Freeze the phase while stalled (enable low) so the II cadence resumes
    // where it paused; a stall-free region advances it every cycle.
    phaseNext.setValue(
        c.mux(running, c.mux(enable, phaseAdv, phase), c.zero32));
  }
  // Gated issue: a stalled cycle (enable low) issues nothing, so the counter,
  // `running`, and (with the enabled shift chains) the whole datapath hold.
  Value issue = c.andBits(wantIssue, enable);
  nameValue(issue, regionSignal(region, "issue"));
  // Induction register: the counter IS the source IV, holding `lb` at start and
  // advancing by `step` on each gated issue, so Source::Counter reads the real
  // loop variable and a `lb != 0` / `step != 1` loop needs no body rewriting.
  auto iterNext = c.bb.get(c.i32);
  Value iv = c.reg(iterNext, term.lb);
  Value ivStep = c.R(comb::AddOp::create(c.b, c.loc, iv, term.step, false));
  iterNext.setValue(c.mux(running, c.mux(issue, ivStep, iv), term.lb));
  // Terminate on the last issued iteration (the next induction value reaches
  // the bound, or the condition is false), clearing running the next cycle.
  Value terminate = c.R(
      comb::AndOp::create(c.b, c.loc, issue, term.isLast(c, ivStep), false));
  Value runAfterLast = c.mux(terminate, c.f1, running);
  runNext.setValue(c.mux(term.gateStart(c, start), c.t1, runAfterLast));
  return {/*issue=*/issue, /*counter=*/iv, /*wantIssue=*/wantIssue,
          /*running=*/running, /*scaledCounters=*/{}};
}

// The one counted done-driven skeleton, covering the two cells whose iterations
// are paced by the body draining rather than by a schedule:
//   * Container: the body is a sequence of child regions;
//   * CallNode: the body is one instantiated sub-kernel.
// Both keep the same four cells: an induction register advancing on `advance`,
// the `isLast` test against the bound, the launch pulse, and a done latch
// cleared on `start`. The single difference is when the FIRST pass launches:
//   * a Container launches one cycle after `start`, off the settled counter
//     register, because its children read that counter as their own bound and
//     sample it at their own start;
//   * a CallNode launches on `start` itself, reading the counter through a
//     `start`-cycle bypass, because a call region's start->done latency is the
//     scheduled figure a caller composes against and a register there would add
//     a cycle to it.
// Either way the ADVANCE launch is registered, since the counter it feeds only
// settles the cycle after `advance`.
IterationControl
ControlEmitter::emitCountedIteration(const uarch::RegionBlock &rb,
                                     const Terminator &term, Value start,
                                     Value complete) const {
  assert(term.lb && "a counted iteration controller needs induction bounds");
  bool launchAtStart = rb.shape == uarch::RegionBlock::Shape::CallNode;

  Backedge ivNext = c.bb.get(term.lb.getType());
  Value ivReg = c.reg(ivNext, term.lb);
  nameValue(ivReg, rb.counterName.empty() ? regionSignal(rb.id, "iv")
                                          : rb.counterName);
  Value iv = launchAtStart ? c.mux(start, term.lb, ivReg) : ivReg;
  Value ivStep = c.R(comb::AddOp::create(c.b, c.loc, iv, term.step, false));
  // This pass is the last one; iterations remain otherwise (never for an empty
  // region, whose body never runs at all, so `advance` stays low and the
  // counter holds `lb`).
  Value last = term.isLast(c, ivStep);
  Value advance = c.andBits(complete, c.notBit(last));
  ivNext.setValue(c.mux(start, term.lb, c.mux(advance, ivStep, iv)));
  llvm::SmallVector<Value> scaled = emitScaledCounters(
      rb, term, /*bypassStart=*/launchAtStart ? start : Value(),
      [&](Value cur, Value stepped, Value init) {
        // Exactly `ivNext` above, with `lb` and `step` scaled.
        return c.mux(start, init, c.mux(advance, stepped, cur));
      });

  // `gateStart` masks the start launch of an empty region (a runtime zero trip
  // or a static lb >= ub), which completes through `empty` below instead.
  Value first = term.gateStart(c, start);
  Value launch = launchAtStart ? c.orBits(first, c.reg(advance, c.f1))
                               : c.reg(c.orBits(first, advance), c.f1);
  nameValue(launch, regionSignal(rb.id, "fire"));
  // An empty region completes one cycle after `start`, not on it: `done` is a
  // level cleared by `start`, so a pulse landing there would leave it high with
  // no 0->1 edge for the next node to start on.
  Value empty = c.reg(c.andBits(start, term.isEmpty(c)), c.f1);
  Value done = c.holdDone(c.orBits(empty, c.andBits(complete, last)), start);
  nameValue(done, regionSignal(rb.id, "done"));
  return {{/*issue=*/launch, /*counter=*/iv, /*wantIssue=*/Value(),
           /*running=*/Value(), /*scaledCounters=*/std::move(scaled)},
          done};
}

// The conditional done-driven skeleton: a sequential-wrapper while. Same
// boundary/continue/launch/done shape as the counted one, but the continue test
// is not available AT the boundary. The condition reads the iter-arg survivor
// registers, which only settle the cycle after a body pass drains, and may
// itself take `tCond` cycles (a memory- or IP-dependent condition). So the
// decision is a delayed CHECK pulse rather than a combinational test, and it
// forks directly into launch / finish. The zero-iteration case needs no
// separate empty term: the first CHECK already answers it, and it is a cycle
// after `start`, which is exactly the edge hygiene `done` needs.
IterationControl ControlEmitter::emitCheckedIteration(unsigned region,
                                                      Value cond,
                                                      unsigned tCond,
                                                      Value start,
                                                      Value complete) const {
  Value check = c.reg(c.orBits(start, complete), c.f1);
  nameValue(check, regionSignal(region, "check"));
  // A container derives no stall shell of its own, since its stream-touching
  // work sits in a child leaf under that leaf's shell, so the CHECK window is
  // rigid.
  Value settled = c.delayValid(check, tCond, StallShell{});
  auto [launch, finish] = c.branchPulse(settled, cond);
  nameValue(launch, regionSignal(region, "fire"));
  Value done = c.holdDone(finish, start);
  nameValue(done, regionSignal(region, "done"));
  return {{/*issue=*/launch, /*counter=*/Value(), /*wantIssue=*/Value(),
           /*running=*/Value(), /*scaledCounters=*/{}},
          done};
}

// Acyclic (straight-line) region: a single pass. A NESTED acyclic child arms
// `start` delayed one cycle, a registered pulse matching the cyclic regimes'
// registered `running`. This is what lets it read the outer counter correctly:
// the container advances its counter on the child's start pulse, so the new
// index only settles the next cycle (register semantics), exactly when this
// registered arming (and a cyclic child's `running`) rises. A TOP-LEVEL acyclic
// region has no outer counter, so that register would be pure latency: it arms
// on `start` directly, so a pure-seq call container's latency equals its
// reported schedule depth (no spurious +1). There is no iteration index of its
// own.
//
// Under an elastic shell the arming pulse is LATCHED into `pend`, the acyclic
// counterpart of the pipelined regime's `running`: a single one-shot pulse
// cannot be gated, only dropped, so a stage-0 stream access would sample its
// `_data` at the arming cycle whatever `_valid` said (and a stage-0 put would
// drop its token and never complete). The latch turns "issue now" into "issue
// as soon as the shell allows", which the whole region's timeline already
// follows, since every chain below it rides that same shell. `pend` is
// combinationally ORed with the arming pulse rather than replacing it, so an
// available token still issues at the arming cycle and the top-level latency
// above is unchanged. A rigid region has nothing to defer and stays a bare
// pulse.
RegionControl ControlEmitter::emitAcyclic(unsigned region, Value start,
                                          bool topLevel,
                                          const StallShell &sh) const {
  Value armed = topLevel ? start : c.reg(start, c.f1);
  if (!sh) {
    nameValue(armed, regionSignal(region, "issue"));
    return {armed, /*counter=*/Value(), /*wantIssue=*/Value(),
            /*running=*/Value(), /*scaledCounters=*/{}};
  }
  auto pendNext = c.bb.get(c.i1);
  Value pending = c.reg(pendNext, c.f1);
  nameValue(pending, regionSignal(region, "pend"));
  Value wantIssue = c.orBits(armed, pending);
  Value issue = c.andBits(wantIssue, sh.issueEnable);
  nameValue(issue, regionSignal(region, "issue"));
  // Hold the pass pending until it actually issues; the pass is a single one,
  // so `wantIssue` falls the cycle after and the latch stays down.
  pendNext.setValue(c.andBits(wantIssue, c.notBit(sh.issueEnable)));
  return {issue, /*counter=*/Value(), /*wantIssue=*/wantIssue,
          /*running=*/Value(), /*scaledCounters=*/{}};
}

// The region's completion signal: one latched level for every regime (cyclic,
// while, acyclic). It rises when the last iteration's deepest output has
// drained, that is, `lastIssue` (the final iteration's issue pulse) delayed by
// `drainStage` cycles, or immediately on `emptyDone` (an empty region, when
// reachable). The latch's register cycle is the LAST commit cycle, so a sibling
// starting on this done's edge reads every committed store and survivor.
// `drainStage` equals the deepest output's commit cycle minus that one, and
// `storeDrainOf` derives a store's half of it from the memory's write latency
// (a store-less region uses its result ready cycle instead). Keying on
// `lastIssue` (an actual issue pulse), rather than a store-retire count, keeps
// a region that retires several stores in one cycle from completing early. A
// `retrig` region (re-run by an enclosing container) resets its completion
// state on `start`.
Value ControlEmitter::emitDone(unsigned region, unsigned drainStage,
                               Value lastIssue, Value emptyDone, Value start,
                               bool retrig, const StallShell &sh) const {
  Value fire = c.delayValid(lastIssue, drainStage, sh);
  // The final put is not committed until accepted, so gate the completion pulse
  // on the region's clock-enable: `done` holds through back-pressure on the
  // last token. A no-op under a rigid shell.
  if (sh)
    fire = c.andBits(fire, sh.chainEnable);
  if (emptyDone)
    fire = c.orBits(emptyDone, fire);
  auto dNext = c.bb.get(c.i1);
  Value done = c.reg(dNext, c.f1);
  nameValue(done, regionSignal(region, "done"));
  // `retrig` clears the held `done` on `start`, giving a fresh 0->1 edge each
  // pass. Callers must keep `fire` off the `start` cycle: it wins over this
  // clear and would hold the level at 1 with no edge.
  Value held = retrig ? c.mux(start, c.f1, done) : done;
  dNext.setValue(c.mux(fire, c.t1, held));
  return done;
}

} // namespace mlir::allo::uarch
