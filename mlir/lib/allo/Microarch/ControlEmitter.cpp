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
                                                  Value start, Value enable) {
  if (rb.kind == uarch::RegionBlock::Kind::Acyclic)
    return emitAcyclic(start, /*topLevel=*/!rb.parent);
  RegionControl rc = emitPipelined(*rb.ii, term, start, enable);
  // Label the iteration-counter register after the source loop variable (i).
  nameValue(rc.counter, rb.counterName);
  return rc;
}

// The one pipelined control skeleton, covering three regimes that differ only
// in their `Terminator` and (for II>1) a phase counter:
//   * free-running (II==1, counted)  -- one iteration issued every cycle;
//   * modulo       (II>1,  counted)  -- one issued every II cycles, gated by a
//                                       [0,II) phase counter (in-flight drain
//                                       via the valid chain);
//   * while        (II==1, conditional) -- a non-speculative flushing pipeline,
//                                       terminated by the condition going
//                                       false.
// `running` is set by `start` and cleared the cycle the last iteration issues;
// the iteration counter advances on issue (feeding the counted bound test and
// the datapath's iteration-0 recurrence-init injection -- mostly dead for a
// while, which rarely indexes by iteration). A conditional terminator is
// non-speculative (II >= t_cond, so no doomed iteration issues -> no squash)
// and stall-free (fixed-latency memory, no FIFO).
RegionControl ControlEmitter::emitPipelined(int64_t ii, const Terminator &term,
                                            Value start, Value enable) {
  Backedge runNext = c.bb.get(c.i1);
  Value running = c.reg(runNext, c.f1);
  // The ungated per-cycle issue *desire*: modulo (II>1) a phase counter [0,II)
  // gates it to once per II; II==1 (and a while) wants to issue every running
  // cycle. The stall shell then gates this by `enable` below.
  Value wantIssue = running;
  if (ii > 1) {
    Backedge phaseNext = c.bb.get(c.i32);
    Value phase = c.reg(phaseNext, c.zero32);
    wantIssue = c.R(
        comb::AndOp::create(c.b, c.loc, running, c.icmpEq(phase, 0), false));
    Value phasep1 = c.R(comb::AddOp::create(c.b, c.loc, phase, c.one32, false));
    Value phaseAdv = c.mux(c.icmpEq(phase, ii - 1), c.zero32, phasep1);
    // Freeze the phase while stalled (enable low) so the II cadence resumes
    // where it paused; unchanged (advances every cycle) for a stall-free
    // region.
    phaseNext.setValue(
        c.mux(running, c.mux(enable, phaseAdv, phase), c.zero32));
  }
  // Gated issue: a stalled cycle (enable low) issues nothing, so the counter,
  // `running`, and (with the enabled shift chains) the whole datapath hold.
  Value issue = c.andBits(wantIssue, enable);
  // Induction register: the counter IS the source IV -- it holds `lb` at start
  // and advances by `step` on each issue. So Source::Counter reads the real
  // loop variable, and a `lb != 0` / `step != 1` loop indexes correctly with no
  // body rewriting. Advances on each (gated) issue; on a stalled cycle issue is
  // low, so it holds.
  Backedge iterNext = c.bb.get(c.i32);
  Value iv = c.reg(iterNext, term.lb);
  Value ivStep = c.R(comb::AddOp::create(c.b, c.loc, iv, term.step, false));
  iterNext.setValue(c.mux(running, c.mux(issue, ivStep, iv), term.lb));
  // Terminate on the last issued iteration (the next induction value reaches
  // the bound, or the condition is false), clearing running the next cycle.
  Value terminate = c.R(
      comb::AndOp::create(c.b, c.loc, issue, term.isLast(c, ivStep), false));
  Value runAfterLast = c.mux(terminate, c.f1, running);
  runNext.setValue(c.mux(term.gateStart(c, start), c.t1, runAfterLast));
  return {/*issue=*/issue, /*counter=*/iv, /*wantIssue=*/wantIssue};
}

// Acyclic (straight-line) region: a single pass. A NESTED acyclic child issues
// `start` delayed one cycle -- a registered pulse, matching the cyclic regimes'
// registered `running`. This is what lets it read the outer counter correctly:
// the container advances its counter on the child's start pulse, so the new
// index only settles the next cycle (register semantics), exactly when this
// registered issue (and a cyclic child's `running`) rises. A TOP-LEVEL acyclic
// region has no outer counter, so that register would be pure latency -- it
// issues on `start` directly, so a pure-seq call container's latency equals its
// reported schedule depth (no spurious +1). There is no iteration index of its
// own.
RegionControl ControlEmitter::emitAcyclic(Value start, bool topLevel) {
  return {/*issue=*/topLevel ? start : c.reg(start, c.f1),
          /*counter=*/Value(), /*wantIssue=*/Value()};
}

// The region's completion signal: one latched level for every regime (cyclic,
// while, acyclic). It rises when the last iteration's deepest output has
// drained -- `lastIssue` (the final iteration's issue pulse) delayed
// `drainStage` cycles -- or immediately on `emptyDone` (an empty region, when
// reachable). The latch's register cycle is the store/result commit cycle, so a
// sibling starting on this done's edge reads every committed store and
// survivor. `drainStage` is the deepest output's stage: a store-less region's
// result ready cycle, else its deepest store's stage. Keying on `lastIssue` (an
// actual issue pulse) rather than a store-retire count keeps a region that
// retires several stores in one cycle from completing early. A `retrig` region
// (re-run by an enclosing container) resets its completion state on `start`.
Value ControlEmitter::emitDone(unsigned drainStage, Value lastIssue,
                               Value emptyDone, Value start, bool retrig) {
  Value fire = c.delayValid(lastIssue, drainStage);
  // A stream region freezes on output back-pressure, and the final put is not
  // *committed* until it is accepted (valid & ready), not merely presented. So
  // gate the completion pulse on the region's clock-enable (chainEnable):
  // `done` holds through any back-pressure on the last token. The enabled
  // `fire` chain freezes too, so the pulse is simply held until acceptance.
  // No-op (regionEnable null) for a stall-free region.
  if (c.regionEnable)
    fire = c.andBits(fire, c.regionEnable);
  if (emptyDone)
    fire = c.orBits(emptyDone, fire);
  Backedge dNext = c.bb.get(c.i1);
  Value done = c.reg(dNext, c.f1);
  // `retrig` clears the held `done` on `start` so the region is re-invocable:
  // the level drops to 0 for the run, giving the consumer a fresh 0->1 edge
  // when it completes. Callers must therefore keep `fire` off the `start` cycle
  // (see emptyDone) -- it wins over the clear here, so the two coinciding would
  // hold the level at 1 across a restart and the consumer would see no edge.
  Value held = retrig ? c.mux(start, c.f1, done) : done;
  dNext.setValue(c.mux(fire, c.t1, held));
  return done;
}

} // namespace mlir::allo::uarch
