/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"
#include "allo/Scheduling/LatencyModel.h"

#include "circt/Dialect/Comb/CombOps.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// HWEmitter: the orchestrator.
//===----------------------------------------------------------------------===//

// The counted induction bounds (lb/ub/step) of region \p rb, each resolved to
// the region's `counterType`: the IV runs `lb, lb+step, ...` and terminates on
// `iv+step >= ub`. Empty for an acyclic region (no counter) or a while, which
// builds its own Terminator::conditional from the resolved condition.
//
// The counter counts up through SIGNED compares, so a negative lb is fine but
// the step must be positive. A runtime step's sign is a contract with the
// caller, since no static check settles it: step <= 0 would hang the loop and
// write out of bounds on the way.
Terminator HWEmitter::terminatorOf(const uarch::RegionBlock &rb) {
  if (!rb.lbSource)
    return {}; // acyclic: no counter, hence no bounds
  assert(dp.constantOf(rb.stepSource).value_or(1) > 0 &&
         "counted-loop counter is up-counting; a statically non-positive step "
         "must have been rejected by the frontend or the op verifier");
  auto ivType = cast<IntegerType>(rb.counterType);
  // A bound resized to the counter's width: identity for a literal bound, which
  // `recordRegionBounds` already tied in at that width, a real resize for a
  // runtime bound, which arrives as an ordinary index.
  auto at = [&](const uarch::Source &s) {
    return resize(ctx.b, ctx.loc, datapath.resolveSource(s), ivType.getWidth(),
                  /*isSigned=*/true);
  };
  Value lb = at(rb.lbSource), step = at(rb.stepSource);
  if (rb.ubSource)
    return Terminator::counted(lb, at(rb.ubSource), step);
  // No ubSource (see `RegionBlock::ubSource`): a constant trip K over a runtime
  // lb or step, so `ub = lb + K*step`. A literal step still folds its span.
  int64_t trip = *rb.tripCount;
  std::optional<int64_t> kstep = dp.constantOf(rb.stepSource);
  Value span = kstep ? ctx.konst(ivType, trip * *kstep)
                     : ctx.R(comb::MulOp::create(ctx.b, ctx.loc, step,
                                                 ctx.konst(ivType, trip),
                                                 /*twoState=*/false));
  return Terminator::counted(
      lb, ctx.R(comb::AddOp::create(ctx.b, ctx.loc, lb, span, false)), step);
}

// Emit one region: control -> datapath -> resolve the F->G condition, capture
// results, done. The leaf regimes (counted / dynamic-trip / while) differ only
// in the Terminator and the survivor mechanism.
Value HWEmitter::emitRegion(const uarch::RegionBlock &rb, Value start,
                            bool retrig) {
  RegionTag tag(ctx, rb.id,
                rb.singlePass()); // naming scope for this region's cells
  // The controller is selected by (shape x termination): one switch over the
  // table in `RegionBlock::Shape`. `Leaf` falls out and is built inline below.
  switch (rb.shape) {
  case uarch::RegionBlock::Shape::Guard:
    // Run-once under the predicate, either termination class.
    return emitGuard(rb, start);
  case uarch::RegionBlock::Shape::Container:
    return rb.conditional ? emitConditionalContainer(rb, start)
                          : emitContainer(rb, start);
  case uarch::RegionBlock::Shape::CallNode:
    // A counted loop whose body is one CallUnit, advancing on the child's real
    // `done` rather than on the per-cycle pipeline cadence.
    assert(!rb.conditional && "CallNode x Conditional is not a producible "
                              "shape; see RegionBlock::Shape");
    return emitLoopCall(rb, start);
  case uarch::RegionBlock::Shape::Leaf:
    break;
  }

  // A while's continue-condition is a datapath value not emitted yet, so it
  // rides a backedge resolved after the datapath; a counted bound resolves
  // here.
  Backedge condBE;
  Terminator term;
  if (rb.conditional) {
    condBE = ctx.bb.get(ctx.i1);
    term = Terminator::conditional(condBE, ctx.zero32, ctx.one32);
  } else {
    term = terminatorOf(rb);
  }

  // H (elasticity): a stream region's enables depend on handshakes not yet
  // emitted, so it registers a promise (two backedges) that G, F and the done
  // drain wire against, RAUWed at the end. A stream-free region is rigid.
  Backedge chainEnableBE, issueEnableBE;
  StallShell shell; // rigid unless the region has stream accesses
  if (!rb.streamAccesses.empty()) {
    chainEnableBE = ctx.bb.get(ctx.i1);
    issueEnableBE = ctx.bb.get(ctx.i1);
    shell = {chainEnableBE, issueEnableBE};
    datapath.setShell(rb.id, shell);
  }

  auto rc = control.emitPipelineControl(rb, term, start, shell);
  datapath.setControl(rb.id, rc); // seam G -> F (counter + issue)

  // This also emits a while's condition and its next-value producers.
  auto fb = datapath.emit(rb, rc.issue);
  // H runs on the emitted (F, G) pair, deriving the two promised enables.
  StallShell derived = datapath.deriveStallShell(rb, rc.issue, fb);

  // `setValue` RAUWs and erases the placeholder, so re-point the terminator: a
  // later `term.cond` read (lastIssuePulse's exit test) needs the real value.
  if (rb.conditional) {
    Value cond = datapath.resolveSource(rb.condition);
    condBE.setValue(cond);
    term.cond = cond;
  }

  Value lastIssue = lastIssuePulse(rc, term);
  // The one thing the two leaf terminations disagree about: a counted loop's
  // recurrence is final only on the last iteration, while a while advances on
  // every CONTINUING iteration (the doomed exit iteration must not commit).
  Value captureOn =
      rb.conditional ? ctx.andBits(rc.issue, term.cond) : lastIssue;
  unsigned resultDrain = captureResults(rb, captureOn, start);
  unsigned drainStage = std::max(fb.storeDrain, resultDrain);
  // The model against the hardware. A stream region is excluded because
  // `resolveAccessOperands` re-stamps its put stages, a call-holding leaf
  // because it also waits on the child's `done`.
  assert((!rb.modelledDrain || !rb.streamAccesses.empty() ||
          !rb.callUnits.empty() ||
          static_cast<int64_t>(drainStage) == *rb.modelledDrain) &&
         "the composed span's drain disagrees with the built datapath's; a "
         "consumer placed against it samples on the wrong cycle");

  // An empty counted leaf (lb >= ub) issues nothing, so it completes on
  // `start`, delayed one cycle so the pulse doesn't land on `start` itself:
  // `done` is a level and retrigger needs a real 0->1 edge.
  Value emptyDone =
      (rb.kind == uarch::RegionBlock::Kind::Cyclic && !rb.conditional)
          ? ctx.delayValid(ctx.andBits(start, term.isEmpty(ctx)), 1, shell)
          : Value();
  // A CallUnit region completes on the child's `done`; one that also has loose
  // datapath waits for both, ANDing two held levels so the later wins.
  bool looseWork = !rb.streamAccesses.empty() || !rb.units.empty() ||
                   !rb.memAccesses.empty();
  Value done = fb.callDone && !looseWork
                   ? fb.callDone
                   : control.emitDone(rb.id, drainStage, lastIssue, emptyDone,
                                      start, retrig, shell);
  if (fb.callDone && looseWork)
    done = ctx.andBits(fb.callDone, done);
  // Resolving the promise RAUWs every consumer and erases the placeholders, so
  // re-register the region with the resolved values; a later region must not
  // read the placeholders.
  if (shell) {
    assert(derived && "a stream region must derive its shell");
    chainEnableBE.setValue(derived.chainEnable);
    issueEnableBE.setValue(derived.issueEnable);
    datapath.setShell(rb.id, derived);
  }
  return done;
}

// The final iteration's issue pulse: a counted region's last iteration (iv+step
// reaches the bound) or a while's exit; an acyclic region has no counter, so
// its single issue pulse is itself the last. Both `emitDone` and the survivor
// captures key off it.
Value HWEmitter::lastIssuePulse(const RegionControl &rc,
                                const Terminator &term) {
  if (!rc.counter)
    return rc.issue; // acyclic: a single pass
  Value ivStep =
      ctx.R(comb::AddOp::create(ctx.b, ctx.loc, rc.counter, term.step, false));
  return ctx.andBits(rc.issue, term.isLast(ctx, ivStep));
}

// Capture each of a result-yielding LEAF region's results into its own survivor
// register on the cycle it lands, while the result is still on its Source: a
// free-running datapath overwrites it once the run ends. \p captureOn is the
// issue pulse the capture keys off; a result produced at a later stage delays
// its capture to match. Returns the LATEST-landing result's stage, which the
// region folds into its `drainStage` so `done` rises with the deepest survivor
// latched. A store-ful region yields no result and returns stage 0.
unsigned HWEmitter::captureResults(const uarch::RegionBlock &rb,
                                   Value captureOn, Value start) {
  StallShell sh = datapath.shellFor(rb.id);
  unsigned maxStage = 0;
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    if (!r.value)
      continue; // an untracked result: no survivor (asserts if read)
    if (r.value.kind == uarch::Source::Kind::Call)
      continue; // a call result: emitCalls sets the survivor from the child's
                // held output port (self-timed by `done`), not a static capture
    unsigned stage = datapath.readyCycle(r.value);
    Value cap = ctx.delayValid(captureOn, stage, sh);
    Value res = datapath.resolveSource(r.value);
    // A loop-carried result preloads its init at `start`, so a run that never
    // captures keeps the identity rather than a stale value. An init-less
    // result always lands: it powers on at 0.
    Value survivor =
        r.init ? ctx.latchReg(datapath.resolveSource(r.init), res, start, cap)
               : ctx.enabledReg(res, cap, ctx.konst(res.getType(), 0),
                                RegRole::Survivor);
    nameValue(survivor, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, survivor);
    maxStage = std::max(maxStage, stage);
  }
  return maxStage;
}

// Run `regions` in program order, each region starting when its predecessor
// drains (the first on `start`); returns the last region's done.
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
// predecessors' joined `done`. The kernel `done` is the conjunction of every
// region's `done`. Emission is in program order, so a predecessor's `done` is
// already built when its consumer reads it.
Value HWEmitter::composeSiblings(llvm::ArrayRef<uarch::RegionId> regions,
                                 Value start) {
  // Nothing to compose: complete a cycle after `start`, the shape an empty
  // counted region's `done` already takes.
  if (regions.empty())
    return ctx.holdDone(ctx.reg(start, ctx.f1), start);

  llvm::DenseMap<uarch::RegionId, Value> doneOf;
  Value allDone;
  for (uarch::RegionId rid : regions) {
    const auto &rb = dp.regions[rid];
    llvm::SmallVector<Value, 2> predDones;
    for (uarch::RegionId p : rb.predecessors) {
      Value d = doneOf.lookup(p);
      assert(d && "a predecessor's done must be emitted before its consumer");
      predDones.push_back(d);
    }
    Value startK = ctx.startFor(start, predDones);
    Value done = emitRegion(rb, startK, /*retrig=*/true);
    // A lone region is its own conjunction and has no consumer to hand a stale
    // level to, so it keeps the raw done.
    Value completed =
        regions.size() > 1 ? ctx.completedSince(done, start) : done;
    doneOf[rid] = completed;
    allDone = allDone ? ctx.andBits(allDone, completed) : completed;
  }
  return allDone;
}

// Set up a container's loop-carried iter-args as frozen survivor registers:
// each latches its `results[k].init` at `start` and advances to a next-value on
// `advance`, and is recorded as Source::Survivor{rb, k}. Returns the per-arg
// next-value backedges the caller sets to `resolveSource(results[k].value)`
// once the children have produced them; the recurrence splits in two halves
// because the register must exist before the children that feed it emit.
SmallVector<circt::Backedge>
HWEmitter::setupCarriedIterArgs(const uarch::RegionBlock &rb, Value start,
                                Value advance) {
  SmallVector<circt::Backedge> nextBE;
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    assert(r.init && "a container iter-arg has no resolvable init");
    Value init = datapath.resolveSource(r.init);
    circt::Backedge nb = ctx.bb.get(init.getType());
    nextBE.push_back(nb);
    Value carried = ctx.latchReg(init, nb, start, advance);
    nameValue(carried, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, carried);
  }
  return nextBE;
}

// A loop-over-call region (a counted `dcp.pipeline` wrapping one CallUnit): the
// counter is `rc.counter` and the child start is `rc.issue`, so one child
// instance fires N times, each invocation advancing on its real `done`, a held
// level cleared on its start whose rising edge marks each completion.
Value HWEmitter::emitLoopCall(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id, rb.singlePass());
  // A loop-over-call body is one child instance and nothing else
  // (`validateDatapath`), so the region is rigid: it derives no stall shell.
  assert(rb.streamAccesses.empty() &&
         "a loop-over-call region with stream accesses would need a stall "
         "shell, which this controller does not build");
  // Bounds are at the child's index-port width (this region's `counterType`).
  // The controller is paced by the child `done` edge, a backedge since
  // emitCalls needs the counter first.
  Backedge callDone = ctx.bb.get(ctx.i1);
  IterationControl ic = control.emitCountedIteration(
      rb, terminatorOf(rb), start, ctx.risingEdge(callDone));

  // An empty loop never fires the child, whose own run gating keeps every
  // write-enable low.
  datapath.setControl(rb.id, ic.rc);
  auto fb = datapath.emit(rb, ic.rc.issue);
  assert(fb.callDone && "a loop-over-call region produced no child done");
  callDone.setValue(fb.callDone);
  return ic.done;
}

// A container region: a cyclic loop whose body nests one or more child regions,
// run once per outer iteration. The outer counter is materialized first, then
// the children are sequenced within one outer iteration, and the counter
// advances when the LAST child drains. Non-overlapping (II_outer >= sum of
// child latencies), so the outer index is stable across one pass. A value
// handed child-to-child crosses as a survivor register. Returns a latched
// completion level.
Value HWEmitter::emitContainer(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id, rb.singlePass());
  // The controller is paced by `lastDrain`, the last child's done edge,
  // resolved once the children emit.
  Backedge lastDrain = ctx.bb.get(ctx.i1);
  IterationControl ic =
      control.emitCountedIteration(rb, terminatorOf(rb), start, lastDrain);
  // The counter must be live while the children emit: it is their outer index,
  // and (for a variable-trip child) its own bound.
  datapath.setControl(rb.id, ic.rc);

  // Loop-carried iter-args, advancing on each outer-iteration drain; the final
  // value is this region's survivor.
  SmallVector<Backedge> nextBE = setupCarriedIterArgs(rb, start, lastDrain);

  // The container's own combinational units (a nested guard's predicate over
  // this counter) emit once the counter and iter-arg survivors are live, so a
  // guard child reads its predicate as a Source::Unit when it emits below.
  datapath.emitUnits(rb, DatapathEmitter::UnitMode::Container);

  lastDrain.setValue(ctx.risingEdge(sequence(rb.children, ic.rc.issue,
                                             /*retrig=*/true)));
  for (auto [k, nb] : llvm::enumerate(nextBE))
    nb.setValue(datapath.resolveSource(rb.results[k].value));
  return ic.done;
}

// A conditional container: a sequential-wrapper while whose body nests child
// regions. Each outer iteration runs the children once (as emitContainer), but
// the loop is data-dependent: the outer iter-args are frozen survivor registers
// advanced by the children's results, and a done-based CHECK/RUN FSM re-checks
// the continue-condition on the settled iter-args after each drain, ending the
// loop when it goes false. No squash or stall: the same non-speculative
// flushing family as a leaf while.
Value HWEmitter::emitConditionalContainer(const uarch::RegionBlock &rb,
                                          Value start) {
  RegionTag tag(ctx, rb.id, rb.singlePass());

  // The outer iter-arg registers are this region's survivors, advanced when an
  // outer iteration drains (`lastDrain`, resolved after the children emit).
  Backedge lastDrain = ctx.bb.get(ctx.i1);
  SmallVector<Backedge> nextBE = setupCarriedIterArgs(rb, start, lastDrain);

  // The condition cone yields the continue-condition and its ready latency
  // t_cond (0 when combinational, several cycles when memory- or IP-dependent).
  // It reads only the frozen iter-args, so it emits before its sampler.
  auto [cond, tCond] = datapath.emitConditionRegion(rb, rb.condition);
  IterationControl ic =
      control.emitCheckedIteration(rb.id, cond, tCond, start, lastDrain);

  // The last child's drain edge advances the iter-args and drives the next
  // CHECK.
  lastDrain.setValue(
      ctx.risingEdge(sequence(rb.children, ic.rc.issue, /*retrig=*/true)));
  for (auto [k, nb] : llvm::enumerate(nextBE))
    nb.setValue(datapath.resolveSource(rb.results[k].value));
  return ic.done;
}

// A guard region (a dcp.select): the then-arm (`children`) runs iff the
// predicate holds, the else-arm (`elseChildren`) iff it does not. The predicate
// is a held value, valid at `start`. The not-taken arm's children never issue,
// so the predicate reaches every store write-enable structurally, via the
// missing issue pulse, not a per-store gate. An empty arm completes in one
// cycle, its start pulse IS its drain, so the region produces a done edge in
// both branches. Run-once: no iteration or iter-args, since the predicate is
// independent of the children.
Value HWEmitter::emitGuard(const uarch::RegionBlock &rb, Value start) {
  RegionTag tag(ctx, rb.id, rb.singlePass());
  // The predicate as a Source: a scheduled condition region's survivor (a
  // data-dependent scf guard), or the parent container's combinational
  // predicate unit (an affine guard, emitted by the container beforehand).
  Value cond = datapath.resolveSource(rb.condition);
  // CHECK after the guard's arm cost decouples the completion pulse from the
  // start-clear below: a skipped guard's done would otherwise coincide with
  // `start` and be masked.
  Value checkTime = ctx.delayValid(start, kGuardBoundary.arm, StallShell{});
  nameValue(checkTime, regionSignal(rb.id, "check"));
  auto [thenStart, elseStart] = ctx.branchPulse(checkTime, cond);
  // Each arm runs its children once, retrig so a re-entered guard presents
  // fresh edges each enclosing pass.
  Value thenDrained =
      rb.children.empty()
          ? thenStart
          : ctx.risingEdge(sequence(rb.children, thenStart, /*retrig=*/true));
  Value elseDrained = rb.elseChildren.empty()
                          ? elseStart
                          : ctx.risingEdge(sequence(rb.elseChildren, elseStart,
                                                    /*retrig=*/true));
  // Each yielded result is `cond ? then-value : else-value`, latched when its
  // arm drains; only the taken arm fires, so the mux ignores the other's stale
  // survivor.
  for (auto [k, r] : llvm::enumerate(rb.results)) {
    Value tv = datapath.resolveSource(r.value);
    Value ev = datapath.resolveSource(r.elseValue);
    Value thenSurv = ctx.enabledReg(tv, thenDrained,
                                    ctx.konst(tv.getType(), 0),
                                    RegRole::Survivor);
    Value elseSurv = ctx.enabledReg(ev, elseDrained,
                                    ctx.konst(ev.getType(), 0),
                                    RegRole::Survivor);
    nameValue(thenSurv, survivorName(rb.id, k));
    nameValue(elseSurv, survivorName(rb.id, k));
    datapath.setSurvivor(rb.id, k, ctx.mux(cond, thenSurv, elseSurv));
  }
  // Exactly one arm runs, so the region completes on whichever drains. Latch
  // done (a level); clear on start so a retriggered guard re-edges.
  Value done = ctx.holdDone(ctx.orBits(thenDrained, elseDrained), start);
  nameValue(done, regionSignal(rb.id, "done"));
  return done;
}

// Emit the whole module body: preamble (literals, read ports, internal memories
// and channels) once, then the func-scope sibling regions composed by their
// dependence DAG. Nested regions emit inside their container.
void HWEmitter::emit() {
  ctx.initLiterals();
  datapath.bindReadPorts();
  datapath.createInternalMemories();
  datapath.declareInternalChannels();
  SmallVector<uarch::RegionId> top;
  for (const uarch::RegionBlock &rb : dp.regions)
    if (!rb.parent) // a child region emits inside its container
      top.push_back(rb.id);
  // retrig keeps the module re-invocable with a fresh `done` edge each drive.
  pa.setOutput(kDone, composeSiblings(top, pa.getInput(kStart)));
  // Stream ports and the internal FIFOs last: a channel's single handshake is
  // shared by every access to it, so it can only be driven once every region
  // has contributed.
  datapath.finalizeStreamPorts();
  // Same reason: a scattered argument's N element outputs are shared by every
  // store to it.
  datapath.finalizeScatteredPorts();
  // And an internal array's write ports, each shared by the stores coloured
  // onto it so the array still infers a block RAM, and an external array's
  // boundary groups, merged onto the same colours so its OWNER can.
  datapath.finalizeSharedWritePorts();
  datapath.finalizeBoundaryWritePorts();
  // Scalar results: the returning region's survivor register, stable once its
  // region (and thus `done`) has risen; the cosim samples it at `done`.
  for (const uarch::Result &r : dp.results)
    pa.setOutput(r.name, datapath.resolveSource(r.source));
}

} // namespace mlir::allo::uarch
