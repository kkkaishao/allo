/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/HWEmitter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace mlir::allo::uarch {

// Resolve a datapath Source to the SSA value driving it, exhaustive over
// Source::Kind.
Value DatapathEmitter::resolveSource(const uarch::Source &s) {
  switch (s.kind) {
  case uarch::Source::Kind::Unit: {
    // A same-region operator result. `declareUnits` declares its backedge
    // before any read resolves, so a miss means this consumer sits outside the
    // owning region.
    Value v = unitVal.lookup(s.id);
    assert(v && "unit source read outside the region that declared it");
    return v;
  }
  case uarch::Source::Kind::Reg:
    return regStages[s.id].tap(s.outPort);
  case uarch::Source::Kind::Mem:
    return readData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Stream:
    // An input stream's loaded token, bound by bindStreamReads before any
    // consumer, like a memory read.
    return streamReadData.lookup(accKey(s.id, s.outPort));
  case uarch::Source::Kind::Counter: {
    // The iteration counter of Source's region (an outer container's counter is
    // live while its nested region emits), at `kIndexWidth` whatever width the
    // region built its register at.
    Value cv = counterIndex.lookup(s.id);
    assert(cv && "counter source with no emitted region counter");
    return cv;
  }
  case uarch::Source::Kind::Const: {
    // The datapath carries a value as its bit pattern, so a float literal ties
    // in as its bitcast integer.
    IntegerType t = datapathType(dp.consts[s.id].type, c.b);
    Attribute v = dp.consts[s.id].value;
    if (auto ia = dyn_cast<IntegerAttr>(v))
      return c.konst(t, ia.getInt());
    return c.konst(
        t, cast<FloatAttr>(v).getValue().bitcastToAPInt().getZExtValue());
  }
  case uarch::Source::Kind::IO:
    // A scalar kernel argument, exposed as its own module input port.
    return pa.getInput(scalarPortName(dp, dp.ios[s.id]));
  case uarch::Source::Kind::Mux: {
    // A shared unit's input: the bound ops hold disjoint MRT residues, so the
    // `activationPulse` selects are one-hot and an AND-OR reduction serves.
    // With no op issuing the result is zero, which no consumer samples.
    if (Value v = muxVal.lookup(s.id))
      return v;
    const uarch::Mux &mx = dp.muxes[s.id];
    Value issue = controlOf.lookup(mx.region).issue;
    assert(issue && "mux in a region with no controller");
    // Timed against the OWNING region's shell (`mx.region`), not whichever
    // region is emitting: the select rides that region's issue pulse.
    StallShell sh = shellFor(mx.region);
    // A recurrence operand's iteration windows, delayed to their op's stage and
    // built once per (op, iteration). The At arms and the From arm that
    // complements them partition that op's pulse by construction. The two kinds
    // are cached apart because one op may carry recurrences of different
    // distances, whose `iter` numbers then mean different things.
    DenseMap<std::pair<Operation *, unsigned>, Value> atOf, fromOf;
    SmallVector<Value> values, selects;
    for (auto [k, src] : llvm::enumerate(mx.sources)) {
      Operation *op = mx.selectOps[k];
      unsigned stage = mx.selectStages[k];
      const uarch::Mux::Phase &ph = mx.phases[k];
      const uarch::RegionBlock &rb = dp.regions[mx.region];
      Value sel = c.activationPulse(issue, stage, sh);
      if (ph.kind == uarch::Mux::Phase::At) {
        Value &window = atOf[{op, ph.iter}];
        if (!window)
          window = c.activationPulse(atIteration(rb, ph.iter), stage, sh);
        sel = c.andBits(sel, window);
      } else if (ph.kind == uarch::Mux::Phase::From) {
        Value &window = fromOf[{op, ph.iter}];
        if (!window)
          window = c.activationPulse(firstIterations(rb, ph.iter), stage, sh);
        sel = c.andBits(sel, c.notBit(window));
      }
      values.push_back(resolveSource(src));
      selects.push_back(sel);
    }
    Value v = c.oneHotSelect(values, selects);
    muxVal[s.id] = v;
    return v;
  }
  case uarch::Source::Kind::Survivor: {
    // A sibling region's held result, latched by setSurvivor when the producing
    // region completed, before this consumer emitted.
    Value sv = survivorOf.lookup(accKey(s.id, s.outPort));
    assert(sv && "survivor source read before its region was captured");
    return sv;
  }
  case uarch::Source::Kind::Call: {
    // A sub-kernel call's scalar result: the child instance's result output,
    // populated by emitCalls before any consumer.
    Value cv = callResultVal.lookup(accKey(s.id, s.outPort));
    assert(cv && "call result source read before its CallUnit was emitted");
    return cv;
  }
  case uarch::Source::Kind::None:
    // `validateDatapath` rejects a None Source earlier. Not an `assert`: under
    // NDEBUG that would fall through and hand the caller a null Value.
    llvm_unreachable("unresolved (None) source reached emission");
  }
  llvm_unreachable("unhandled Source::Kind");
}

Value DatapathEmitter::ivAt(const uarch::RegionBlock &rb, unsigned n,
                            Value lb) {
  if (!n)
    return {};
  auto ivTy = cast<IntegerType>(rb.counterType);
  std::optional<int64_t> kstep = dp.constantOf(rb.stepSource);
  Value nStep = kstep ? c.konst(ivTy, static_cast<int64_t>(n) * *kstep)
                      : c.R(comb::MulOp::create(
                            c.b, c.loc, c.konst(ivTy, static_cast<int64_t>(n)),
                            resize(c.b, c.loc, resolveSource(rb.stepSource),
                                   ivTy.getWidth(), /*isSigned=*/true),
                            false));
  return c.R(comb::AddOp::create(c.b, c.loc, lb, nStep, false));
}

// Both at the raw counter register's width, the width its terminator compares
// at, so a bound from elsewhere resizes into it.
std::pair<Value, Value>
DatapathEmitter::counterAndLb(const uarch::RegionBlock &rb) {
  Value iv = controlOf.lookup(rb.id).counter;
  assert(iv && "a recurrence input in a region with no iteration counter");
  unsigned w = cast<IntegerType>(rb.counterType).getWidth();
  return {iv, resize(c.b, c.loc, resolveSource(rb.lbSource), w,
                     /*isSigned=*/true)};
}

Value DatapathEmitter::firstIterations(const uarch::RegionBlock &rb,
                                       unsigned dist) {
  auto [iv, lb] = counterAndLb(rb);
  if (dist <= 1)
    return c.icmpEqV(iv, lb);
  // iv < lb + dist*step == !(iv >= lb + dist*step). Signed, as
  // `Terminator::isLast` compares the same counter against the same kind of
  // bound; an unsigned predicate would order a negative `lb` wrongly.
  return c.notBit(c.icmpSgeV(iv, ivAt(rb, dist, lb)));
}

Value DatapathEmitter::atIteration(const uarch::RegionBlock &rb,
                                   unsigned iter) {
  auto [iv, lb] = counterAndLb(rb);
  Value at = ivAt(rb, iter, lb);
  return c.icmpEqV(iv, at ? at : lb);
}

// Read off the model, never off the producing op: the stages were frozen when
// the model was built and the IR is provenance from here on.
unsigned DatapathEmitter::readyCycle(const uarch::Source &s) const {
  switch (s.kind) {
  // A call is the one producer whose result does NOT land at `stage +
  // latency`: it lands at its region-relative issue plus the CALLEE's whole
  // start->done depth. Indeterminate calls are guarded earlier.
  case uarch::Source::Kind::Call: {
    const uarch::CallUnit &cu = dp.calls[s.id];
    assert(cu.latency && "readyCycle of an indeterminate call result");
    return cu.start + static_cast<unsigned>(*cu.latency);
  }
  case uarch::Source::Kind::Unit: {
    const uarch::FuncUnit &u = dp.units[s.id];
    return u.boundOps[s.outPort].stage + u.latency;
  }
  case uarch::Source::Kind::Mem: {
    const uarch::MemUnit &m = dp.mems[s.id];
    return m.accesses[s.outPort].stage + m.readLatency;
  }
  // A get is a combinational front-read of the FIFO, so it lands at issue.
  case uarch::Source::Kind::Stream:
    return dp.streams[s.id].accesses[s.outPort].stage;
  // A held source has no landing stage: a literal is constant, an IO port
  // stable for the whole kernel, and a counter or survivor a register settled
  // by the time the region reading it issues.
  case uarch::Source::Kind::Const:
  case uarch::Source::Kind::IO:
  case uarch::Source::Kind::Counter:
  case uarch::Source::Kind::Survivor:
    return 0;
  case uarch::Source::Kind::Reg:
  case uarch::Source::Kind::Mux:
  case uarch::Source::Kind::None:
    break;
  }
  llvm_unreachable("readyCycle only models a producing or held Source");
}

// The resolved (already stage-delayed) index sources of an access, dims then
// symbols.
// Shift-register chains for region \p rb's registers (index delays, pipeline
// holds). Each chain's head input is a backedge resolved once the units exist.
void DatapathEmitter::emitRegisters(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  // A published phase is the foldability condition: only a schedule-paced
  // controller at II > 1 publishes one, and only there does one iteration land
  // every `ii` cycles. A depth-1 chain is one register either way.
  Value phase = controlOf.lookup(rb.id).phase;
  unsigned ii = rb.ii.value_or(1);
  assert((!phase || ii > 1) && "a phase was published for a region at II 1");
  for (uarch::RegId rid : rb.regs) {
    const uarch::Register &rg = dp.regs[rid];
    auto head = c.bb.get(datapathType(rg.type, c.b));
    regHeadBE.try_emplace(rg.id, head);
    // A register is a plain delay chain; reduction-identity re-injection rides
    // the consuming unit's recurrence input (emitUnits), not the register.
    regStages[rg.id] =
        phase && rg.depth > 1
            ? c.foldedChain(head, rg.depth, ii, phase, rg.ready, sh)
            : c.shiftChain(head, rg.depth, sh);
    // Name each held stage `<value>_d<k>`. Stage 0 is the undelayed input,
    // already named by its producer, so leave it alone rather than relabel a
    // shared wire. A folded chain repeats one register across the `ii` taps it
    // serves, so name it once, at the shallowest delay it provides.
    std::string owner = ownerOf(rg.value, regOwner(rg.id));
    auto &taps = regStages[rg.id].stages;
    for (unsigned k = 1; k < taps.size(); ++k)
      if (taps[k] != taps[k - 1])
        nameValue(taps[k], regTapName(owner, k));
  }
}

// Backedge every unit output before wiring, so an input may reference a unit
// emitted later: a fused recurrence reads its own output, and a data-dependent
// read address (emitReads, which runs before emitUnits) reads a unit
// that computes it. A register elsewhere in the recurrence cycle keeps the
// hardware acyclic; the backedges only free emission from topological order.
void DatapathEmitter::declareUnits(const uarch::RegionBlock &rb) {
  for (uarch::UnitId uid : rb.units) {
    auto b = c.bb.get(datapathType(dp.units[uid].identity.resultType, c.b));
    unitBE[uid] = b;
    unitVal[uid] = b;
  }
}

// Compute units of region \p rb: native -> comb; IP -> an instance of the
// extern operator module, internally pipelined by its latency.
void DatapathEmitter::emitUnits(const uarch::RegionBlock &rb) {
  StallShell sh = shellFor(rb.id);
  for (uarch::UnitId uid : rb.units) {
    const uarch::FuncUnit &u = dp.units[uid];
    SmallVector<Value> operands;
    for (unsigned k = 0; k < u.inputs.size(); ++k) {
      Value v =
          resolveSource(u.inputs[k]); // a self-reference reads its own backedge
      // Re-inject a recurrence input's identities, one per iteration `iv`
      // spends below the recurrence distance, each gated by the issue pulse
      // delayed to this op's stage. Innermost first, so a later iteration's mux
      // sits nearer the port and the windows need no mutual exclusion. A shared
      // port carries none: its identities are arms of the input mux above, and
      // a container's own units carry none at all.
      for (auto [n, init] : llvm::enumerate(u.inputInits[k])) {
        Value issue = controlOf.lookup(rb.id).issue;
        assert(issue && "recurrence input in a region with no controller");
        Value iterN = c.R(
            comb::AndOp::create(c.b, c.loc, issue, atIteration(rb, n), false));
        // A recurrence input belongs to an unshared unit (a shared port
        // carries its identities as mux arms), so the one bound op's stage is
        // the gate's.
        Value gate = c.activationPulse(iterN, u.boundOps.front().stage, sh);
        v = c.mux(gate, resolveSource(init), v);
      }
      operands.push_back(v);
    }

    Value result;
    if (u.identity.comb) {
      result = emitCompute(c.b, c.loc, u.identity, operands,
                           datapathType(u.identity.resultType, c.b));
    } else {
      // An IP instance takes its data operands, then clock, then (for a
      // clock-enabled contract) a `ce` bit that rides the region's
      // clock-enable, freezing with the shift chains under back-pressure.
      operands.push_back(c.clkRaw);
      if (u.stall == allo::StallContractEnum::Ce)
        operands.push_back(sh ? sh.chainEnable : c.t1);
      else
        // A free-running IP has no `ce`: under an elastic shell it would keep
        // advancing while the shell's shift chains stall, folding a stale
        // result. `validateDatapath` rejects that pairing up front.
        assert(!sh && "a free-running IP operator in a back-pressured region");
      // Keyed by the module name the manifest declared it under, which is the
      // same name `Naming.h` spells from this unit's identity.
      Operation *mod = opModules.lookup(operatorModuleName(u));
      assert(mod && "an IP unit with no extern operator module declared");
      result =
          hw::InstanceOp::create(c.b, c.loc, mod, unitInstanceName(u), operands)
              ->getResult(0);
    }
    unitBE[uid].setValue(result);
    unitVal[u.id] = result;
    // Name the result wire after the frontend variable this op computes: the
    // dcp op carries the assignment-target NameLoc.
    nameValue(result, u.repOp()->getLoc());
  }
}

// The condition cone of a sequential (CHECK/RUN) while: the container's OWN
// condition memory reads plus its compute, returning the settled condition and
// its ready latency t_cond. There is no per-iteration issue pulse: the read
// address is the frozen iter-arg survivor, so the load is a continuous read of
// a stable element and its data is a stable wire from `checkStart + t_cond`
// onward, the survivors not advancing until after CHECK decides.
std::pair<Value, unsigned>
DatapathEmitter::emitConditionRegion(const uarch::RegionBlock &rb,
                                     const uarch::Source &condSrc) {
  // The same order a leaf region's datapath emits in, over this container's own
  // condition cone. No issue pulse here, and none is needed: the cone's reads
  // are continuous reads of a stable element, which `bindMemoryPorts` keeps one
  // to a port for exactly that reason.
  emitBeforeUnits(rb, /*issue=*/Value());
  emitUnits(rb);
  emitAfterUnits(rb, /*issue=*/Value());
  return {resolveSource(condSrc), readyCycle(condSrc)};
}

// Resolve region \p rb's register head inputs once its units exist.
void DatapathEmitter::resolveRegHeads(const uarch::RegionBlock &rb) {
  for (uarch::RegId rid : rb.regs)
    regHeadBE.find(rid)->second.setValue(resolveSource(dp.regs[rid].input));
}

// A kernel-local channel's `seq.fifo` cannot be built until every access has
// contributed its drive, and the accesses read its outputs. Declare those
// outputs as backedges here, before any region emits, and let
// `finalizeStreamPorts` build the FIFO and resolve them.
void DatapathEmitter::declareInternalChannels() {
  for (const uarch::StreamChannel &s : dp.streams) {
    // A channel wired between CHILD PORTS declares the other shape: one
    // `{data, valid}` pair per consumer end plus the producer's `ready`. Its
    // internal flag says which END is a module port, not whether wires exist.
    if (!s.callEnds.empty()) {
      ComposedWires &w = composedWires[s.id];
      for (const uarch::StreamChannel::CallEnd &e : s.callEnds)
        if (dp.calls[e.call].streamArgs[e.arg].isInput) {
          w.sinkData.push_back(c.bb.get(datapathType(s.payload, c.b)));
          w.sinkValid.push_back(c.bb.get(c.i1));
        } else
          w.prodReady = c.bb.get(c.i1);
      continue;
    }
    if (!s.internal)
      continue;
    streamWires[s.id] = {c.bb.get(datapathType(s.payload, c.b)), c.bb.get(c.i1),
                         c.bb.get(c.i1)};
  }
}

Value DatapathEmitter::streamData(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].data)
                    : pa.getInput(portData(streamPortBase(dp, s)));
}

Value DatapathEmitter::streamValid(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].valid)
                    : pa.getInput(portValid(streamPortBase(dp, s)));
}

Value DatapathEmitter::streamReady(const uarch::StreamChannel &s) {
  return s.internal ? Value(streamWires[s.id].ready)
                    : pa.getInput(portReady(streamPortBase(dp, s)));
}

void DatapathEmitter::bindStreamReads(const uarch::RegionBlock &rb) {
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    if (s.accesses[r.idx].isPut)
      continue;
    streamReadData[accKey(s.id, r.idx)] = streamData(s);
  }
}

// H for region \p rb: its stream handshakes, and the shell they derive. A put
// contributes to `_data`/`_valid`; a get to `_ready`.
//
// The two halves split on what a blocked handshake blocks. A stage>=1 access
// belongs to an iteration in flight, which cannot advance, so it freezes the
// datapath (`chainEnable`); a stage-0 access belongs to the pass about to
// start, so it only defers that pass (`issueEnable`) and the region drains.
// Draining under starvation is what lets a feedback cycle between two
// processes turn on fewer tokens than the pipeline is deep. The bubble a
// deferral leaves is safe where every intra-iteration chain advances with its
// own iteration and every commit rides a pulse that reads 0 in it;
// `cycleIndexedState` marks the region holding state indexed in cycles
// instead, and there the two halves are one.
//
// A stage-0 access keys on the UNgated `wantIssue` so the signals stay
// combinationally acyclic, a deeper access on the registered delayed issue. A
// predicated access (`acc.when` set) also gates its handshake on the
// predicate, itself a datapath value rather than a FIFO status, so acyclicity
// is preserved.
//
// The pulses built here are timed against the region's PROMISED shell, which is
// what the returned enables resolve: the enable and the chains it freezes are
// mutually recursive, acyclic in hardware because the FIFO status it starts
// from is stored state, and the promise's backedges break that cycle for SSA
// construction. Several accesses may share one channel, interleaved inside the
// II by the FIFO dependence edges; each contributes its own term to
// `streamDrives[s.id]` and `finalizeStreamPorts` drives the port once.
StallShell DatapathEmitter::deriveStallShell(const uarch::RegionBlock &rb,
                                             Value issue,
                                             DatapathFeedback &fb) {
  // No stream accesses: nothing to be elastic about, so the region stays rigid.
  if (rb.streamAccesses.empty())
    return {};
  streamDrives.resize(dp.streams.size());
  StallShell sh = shellFor(rb.id); // the promise F and G were emitted against
  assert(sh && "a stream region must have its shell promise registered");

  Value atIssue =
      controlOf.lookup(rb.id).wantIssue; // ungated stage-0 activation
  assert(atIssue &&
         "a stream region's controller published no `wantIssue`: the shell "
         "defers a starved or back-pressured pass by GATING its issue, so a "
         "controller whose issue cannot be gated would drop the pass and "
         "sample `_data` with no regard for `_valid`");
  // Stage-0 inputs (read at issue) join into `stage0Valid`; a predicated get
  // treats a non-needed input as available (`valid | ~pred`). Built before the
  // puts because a stage-0 put shares the join: a pass that cannot issue
  // writes no token, so it needs no space.
  Value stage0Valid;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut || acc.stage != 0)
      continue;
    Value valid = streamValid(s);
    if (acc.when)
      valid = c.orBits(valid, c.notBit(resolveSource(acc.when)));
    stage0Valid = stage0Valid ? c.andBits(stage0Valid, valid) : valid;
  }
  Value fed = stage0Valid ? c.andBits(atIssue, stage0Valid) : atIssue;

  // Outputs: drive data + valid, accumulate the two back-pressure hazards.
  Value outHazard;   // stage>=1: the token is in flight and cannot advance
  Value issueHazard; // stage 0: the token would be written by the pass itself
  // A stage>=1 put whose handshake fired while the pipeline is frozen: see the
  // `sent` latch below. Resolved once `chainEnable` is final.
  struct Sent {
    circt::Backedge in;
    Value flag, valid, ready;
  };
  SmallVector<Sent> sent;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (!acc.isPut)
      continue;
    // A predicated put produces a token only where its predicate holds: gate
    // `valid`, and suppress the output-full hazard when it is low, so the
    // pipeline never freezes waiting for space it will not write.
    Value pred = acc.when ? resolveSource(acc.when) : Value();
    Value valid = c.activationPulse(issue, acc.stage, sh);
    if (pred)
      valid = c.andBits(valid, pred);
    // An input-side freeze holds a stage>=1 put's chain pulse high after the
    // handshake fired, so a ready consumer would recapture the token. The
    // `sent` latch retires it. A stage-0 pulse is `issue`, already gated.
    if (acc.stage >= 1) {
      circt::Backedge in = c.bb.get(c.i1);
      Value flag = c.reg(in, c.f1);
      valid = c.andBits(valid, c.notBit(flag));
      sent.push_back({in, flag, valid, streamReady(s)});
    }
    auto &drv = streamDrives[s.id];
    drv.puts.push_back({valid, Value(), resolveSource(acc.data)});
    drv.valid = drv.valid ? c.orBits(drv.valid, valid) : valid;
    // A stage-0 put keys its hazard on the pass that would write it (`fed` &
    // pred); a stage>=1 put's valid is already registered (delayed) and
    // predicate-gated.
    Value active = acc.stage == 0 ? fed : valid;
    if (pred && acc.stage == 0)
      active = c.andBits(active, pred);
    Value hz = c.andBits(active, c.notBit(streamReady(s)));
    Value &into = acc.stage == 0 ? issueHazard : outHazard;
    into = into ? c.orBits(into, hz) : hz;
    fb.storeDrain = std::max<unsigned>(fb.storeDrain, acc.stage);
  }
  // Mid-pipeline freeze: a stage>0 get with a needed-but-empty input cannot
  // bubble past a missing token, so fold that stall into `chainEnable` beside
  // the output-full freeze. Only registered state is read here.
  Value midStall;
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut || acc.stage == 0)
      continue;
    Value active = c.delayValid(issue, acc.stage, sh);
    Value want = acc.when ? c.andBits(active, resolveSource(acc.when)) : active;
    Value miss = c.andBits(want, c.notBit(streamValid(s)));
    midStall = midStall ? c.orBits(midStall, miss) : miss;
  }
  Value chainEnable = outHazard ? c.notBit(outHazard) : c.t1;
  if (midStall)
    chainEnable = c.andBits(chainEnable, c.notBit(midStall));

  // G additionally defers a pass whose stage-0 handshakes are not both open.
  // Spelled `~(wantIssue & ~stage0Valid)` rather than `stage0Valid` so it still
  // reads 1 outside a pass: under the merge below, a region draining after its
  // last issue must not freeze on an empty input.
  Value issueEnable = chainEnable;
  if (stage0Valid)
    issueEnable = c.andBits(
        issueEnable, c.notBit(c.andBits(atIssue, c.notBit(stage0Valid))));
  if (issueHazard)
    issueEnable = c.andBits(issueEnable, c.notBit(issueHazard));

  // A region holding cycle-indexed state cannot take that bubble, so its two
  // halves are one and it freezes where it would otherwise drain.
  if (rb.cycleIndexedState)
    chainEnable = issueEnable;

  // `chainEnable` is final: retire each stage>=1 put's token until the chain
  // advances past it (`flag' = ~chainEnable & (flag | fired)`). Only the
  // register's input closes here, so nothing reads a half-built value.
  for (Sent &st : sent)
    st.in.setValue(c.andBits(c.notBit(chainEnable),
                             c.orBits(st.flag, c.andBits(st.valid, st.ready))));

  // Drive each `_ready`: a stage-0 get pops exactly where the pass issues,
  // which already joins every stage-0 input; a deeper get accepts when the
  // chain advances; a predicated get pops only where its predicate holds.
  for (uarch::AccRef r : rb.streamAccesses) {
    const uarch::StreamChannel &s = dp.streams[r.id];
    const uarch::StreamChannel::Access &acc = s.accesses[r.idx];
    if (acc.isPut)
      continue;
    Value pred = acc.when ? resolveSource(acc.when) : Value();
    Value ready = acc.stage == 0 ? c.andBits(atIssue, issueEnable)
                                 : c.andBits(c.delayValid(issue, acc.stage, sh),
                                             chainEnable);
    if (pred)
      ready = c.andBits(ready, pred);
    auto &drv = streamDrives[s.id];
    drv.ready = drv.ready ? c.orBits(drv.ready, ready) : ready;
  }
  nameValue(chainEnable, regionSignal(rb.id, "ce"));
  if (issueEnable != chainEnable)
    nameValue(issueEnable, regionSignal(rb.id, "ien"));
  return {chainEnable, issueEnable};
}

// Drive each channel from the terms every region contributed. A BOUNDARY
// channel drives its module ports, the port set following its direction: an
// input FIFO's `_data` / `_valid` are module inputs and only `_ready` is
// driven, an output's the reverse (`validateDatapath` rejects a boundary
// channel used both ways, so the two cases are exhaustive). A kernel-LOCAL
// channel instead OWNS its queue: one `seq.fifo` here.
void DatapathEmitter::finalizeStreamPorts() {
  streamDrives.resize(dp.streams.size());
  for (const uarch::StreamChannel &s : dp.streams) {
    // A channel wired between CHILD PORTS has no access of this module's own:
    // its handshake closes over the instances instead.
    if (!s.callEnds.empty()) {
      emitComposedChannel(s);
      continue;
    }
    const StreamDrive &drv = streamDrives[s.id];
    // The puts' pulses are mutually exclusive (one access per cycle), so the
    // token is the same one-hot select every shared port takes; nothing reads
    // it while `valid` is low.
    auto putData = [&] {
      assert(!drv.puts.empty() && "a written channel with no put");
      return commitSink(drv.puts, Idle::DontCare).data;
    };
    if (s.internal) {
      emitInternalChannel(s, putData());
      continue;
    }
    auto base = streamPortBase(dp, s);
    if (s.isInput) {
      pa.setOutput(portReady(base), drv.ready ? drv.ready : c.f1);
      continue;
    }
    pa.setOutput(portData(base), putData());
    pa.setOutput(portValid(base), drv.valid);
  }
}

// The queue behind a kernel-local channel. Both ends are this module's, so the
// handshake closes here: a token is pushed where a put fires and the FIFO has
// space, popped where a get fires and it holds one. `seq.fifo`'s output is
// show-ahead, so {output, ~empty, ~full} present exactly the {data, valid,
// ready} triple the accesses were written against for a boundary port.
void DatapathEmitter::emitInternalChannel(const uarch::StreamChannel &s,
                                          Value data) {
  const StreamDrive &drv = streamDrives[s.id];
  StreamWires &w = streamWires[s.id];
  assert(drv.valid && drv.ready &&
         "a local channel is validated to have both ends");
  auto fifo = seq::FIFOOp::create(
      c.b, c.loc, datapathType(s.payload, c.b), c.i1, c.i1, Type(), Type(),
      data,
      /*rdEn=*/c.andBits(drv.ready, w.valid),
      /*wrEn=*/c.andBits(drv.valid, w.ready), c.clk, c.rst,
      c.b.getI64IntegerAttr(declaredDepth(s.depth)), c.b.getI64IntegerAttr(0),
      IntegerAttr(), IntegerAttr());
  w.data.setValue(fifo.getOutput());
  w.valid.setValue(c.notBit(fifo.getEmpty()));
  w.ready.setValue(c.notBit(fifo.getFull()));
}

// The queue(s) behind a channel wired between CHILD PORTS: one `seq.fifo` per
// CONSUMER end, all pushed by the producer on the same cycle, the fan-out tee.
// The producer may write only when every consumer can accept (the bounded
// fork), so each copy sees the whole token sequence in order. A SEEDED channel
// additionally fronts each consumer with an init-prepend shim: while its `rem`
// down-counter is non-zero the consumer reads the initial tokens and does not
// pop, so the history it sees is [init] ++ [produced] and a feedback cycle
// turns from cycle 0.
//
// Where one end is a BOUNDARY port of this module rather than a child, that end
// needs no queue: the child's own handshake is the module's, so the three wires
// pass straight through. A fanned-out boundary input is the one mixed case: the
// module's port pushes the tee.
void DatapathEmitter::emitComposedChannel(const uarch::StreamChannel &s) {
  ComposedWires &w = composedWires[s.id];
  // A channel is composed OR accessed, never both: a stream operand makes its
  // call concurrent, and a concurrent region issues no access of its own.
  assert(s.accesses.empty() &&
         "a channel wired between child ports also has in-module accesses");
  Type payload = datapathType(s.payload, c.b);
  std::string base = s.internal ? std::string() : streamPortBase(dp, s);

  // The push side: the producing child, or this module's own stream port for a
  // boundary INPUT argument.
  Value pData, pValid;
  SmallVector<const uarch::StreamChannel::CallEnd *> sinks;
  for (const uarch::StreamChannel::CallEnd &e : s.callEnds) {
    const uarch::CallUnit::StreamArg &sa = dp.calls[e.call].streamArgs[e.arg];
    if (sa.isInput) {
      sinks.push_back(&e);
      continue;
    }
    pData = callOuts[e.call][sa.data];
    pValid = callOuts[e.call][sa.valid];
  }
  // A boundary OUTPUT: the module's port is the consumer, so the producing
  // child's handshake IS the module's.
  if (sinks.empty()) {
    assert(!s.internal && pData && "a channel with no reader");
    pa.setOutput(portData(base), pData);
    pa.setOutput(portValid(base), pValid);
    w.prodReady.setValue(pa.getInput(portReady(base)));
    return;
  }
  auto consumerReady = [&](const uarch::StreamChannel::CallEnd &e) {
    return callOuts[e.call][dp.calls[e.call].streamArgs[e.arg].ready];
  };
  if (!pData) { // a boundary INPUT feeds the readers
    pData = pa.getInput(portData(base));
    pValid = pa.getInput(portValid(base));
    // A single reader takes it straight, queue-free.
    if (sinks.size() == 1) {
      w.sinkData[0].setValue(pData);
      w.sinkValid[0].setValue(pValid);
      pa.setOutput(portReady(base), consumerReady(*sinks.front()));
      return;
    }
  }

  unsigned depth = declaredDepth(s.depth);
  auto init = dyn_cast_or_null<ArrayAttr>(s.init);
  unsigned nInit = init ? init.size() : 0;
  // The status wires close a cycle: a consumer's `rdEn` reads its own FIFO's
  // `empty` through the shim, and the producer's `wrEn` every FIFO's `full`.
  // So the whole tee is built against promises and resolved at the end.
  SmallVector<Backedge> full, empty, out;
  Value allNotFull;
  for (unsigned k = 0; k < sinks.size(); ++k) {
    full.push_back(c.bb.get(c.i1));
    empty.push_back(c.bb.get(c.i1));
    out.push_back(c.bb.get(payload));
    Value nf = c.notBit(full[k]);
    allNotFull = allNotFull ? c.andBits(allNotFull, nf) : nf;
  }
  Value wrEn = c.andBits(pValid, allNotFull);
  if (!s.internal)
    pa.setOutput(portReady(base), allNotFull); // a fanned-out boundary input

  for (auto [k, e] : llvm::enumerate(sinks)) {
    Value notEmpty = c.notBit(empty[k]);
    Value cReady = consumerReady(*e);
    Value rdEn = c.andBits(cReady, notEmpty);
    Value data = out[k], valid = notEmpty;
    if (nInit) {
      // `rem` counts the initial tokens still to serve, k .. 1; the datum is
      // picked by the running index (idx = nInit - rem) and the rem==1 token
      // falls through as the chain's default.
      unsigned remW = 1;
      while ((1u << remW) <= nInit)
        ++remW;
      Type remTy = c.b.getIntegerType(remW);
      Backedge remNext = c.bb.get(remTy);
      Value rem = c.reg(remNext, c.konst(remTy, nInit));
      nameValue(rem,
                channelSignal(ownerOf(s.stream, chanOwner(s.id)),
                              sinks.size() > 1 ? "init_rem" + std::to_string(k)
                                               : std::string("init_rem")));
      Value serving = c.R(comb::ICmpOp::create(
          c.b, c.loc, comb::ICmpPredicate::ne, rem, c.konst(remTy, 0)));
      auto token = [&](unsigned idx) {
        Attribute a = init[idx];
        APInt bits = isa<IntegerAttr>(a)
                         ? cast<IntegerAttr>(a).getValue()
                         : cast<FloatAttr>(a).getValue().bitcastToAPInt();
        return c.konst(payload,
                       bits.zextOrTrunc(cast<IntegerType>(payload).getWidth())
                           .getZExtValue());
      };
      Value fromInit = token(nInit - 1);
      for (unsigned v = 2; v <= nInit; ++v)
        fromInit = c.mux(c.icmpEqV(rem, c.konst(remTy, v)), token(nInit - v),
                         fromInit);
      data = c.mux(serving, fromInit, out[k]);
      valid = c.orBits(serving, notEmpty);
      rdEn = c.andBits(rdEn, c.notBit(serving));
      Value dec = c.R(comb::SubOp::create(c.b, c.loc, rem, c.konst(remTy, 1)));
      remNext.setValue(c.mux(c.andBits(serving, cReady), dec, rem));
    }
    auto fifo = seq::FIFOOp::create(
        c.b, c.loc, payload, c.i1, c.i1, Type(), Type(), pData, rdEn, wrEn,
        c.clk, c.rst, c.b.getI64IntegerAttr(depth), c.b.getI64IntegerAttr(0),
        IntegerAttr(), IntegerAttr());
    // The consumer's promises resolve FIRST: for an unseeded end they resolve
    // *to* the status promises below, which would be erased out from under them
    // the other way round.
    w.sinkData[k].setValue(data);
    w.sinkValid[k].setValue(valid);
    out[k].setValue(fifo.getOutput());
    full[k].setValue(fifo.getFull());
    empty[k].setValue(fifo.getEmpty());
  }
  if (w.prodReady)
    w.prodReady.setValue(allNotFull);
}

// The start pulse of one child, read off the node's contract and its region's
// composition class:
//
//   * HANDSHAKE, the rising edge of the predecessors' joined `done`, for a
//     gated child of a SCHEDULED composition and, in a CONCURRENT one, for a
//     child whose ordering cannot be expressed as an offset: a spawn, a
//     consumer of a scalar result (that port only holds from the producer's
//     `done`), or a child gated by an indeterminate producer. A
//     CHANNEL-CONNECTED pair never reaches here, back-pressure already being
//     their ordering;
//   * BROADCAST, the container's own start, for an ungated spawn;
//   * TIME-TRIGGERED at the scheduled offset otherwise. An ungated call's
//     operands need not be ready at the region's issue pulse (a scalar argument
//     loaded from memory is the reachable case), so releasing it at issue would
//     latch garbage. The offset rides the region's shell, so it stretches with
//     a stall.
//
// A child's `done` is a level its own start clears, so on a retriggered region
// it still reads the previous pass's 1 until the child is released. The
// predecessor join and the region's completion conjunction mean "completed THIS
// pass" and therefore read it through `completedSince(issue)`, in a SCHEDULED
// composition only: there `issue` is the pass-start pulse the calls are placed
// against, where a CONCURRENT region has no such boundary.
Value DatapathEmitter::startForCall(const uarch::CallUnit &cu, Value issue,
                                    ArrayRef<Value> predDones, bool concurrent,
                                    const StallShell &sh) {
  if (!concurrent)
    return predDones.empty() ? c.delayValid(issue, cu.start, sh)
                             : c.startFor(issue, predDones);
  bool handshake =
      !predDones.empty() &&
      (cu.async ||
       llvm::any_of(cu.predecessors, [&](const uarch::CallUnit::Pred &p) {
         return p.viaResult || !dp.calls[p.call].determinate;
       }));
  if (handshake)
    return c.startFor(/*regionStart=*/Value(), predDones);
  if (cu.async)
    return issue;
  return c.delayValid(issue, cu.start, StallShell{});
}

// Instantiate each CallUnit (dcp.instance) in region \p rb as a child
// hw.instance. The child masters each memref operand's memory: it drives the
// addr/data/we, so the leaf wires those instance-output ports to the buffer's
// hlmem. The region's completion is the child's real `done` (fb.callDone).
// Serial execution (a producer region drains before the child starts, the child
// before a consumer) means one master per port at a time: no arbitration mux.
void DatapathEmitter::emitCalls(const uarch::RegionBlock &rb, Value issue,
                                DatapathFeedback &fb) {
  StallShell sh = shellFor(rb.id);
  // Each call starts by the policy above, off the `done`s of the composition
  // predecessors the model derived (`recordCallDeps`); the region completes
  // when every call's done is set.
  bool concurrent = rb.determinacy == DeterminacyEnum::Concurrent;
  SmallVector<Value> dones; // each call's done, by index
  llvm::DenseMap<uarch::CallId, Value>
      doneByCid; // done by id (scalar hand-off)
  for (uarch::CallId cid : rb.callUnits) {
    const uarch::CallUnit &cu = dp.calls[cid];
    SmallVector<Value> predDones;
    for (const uarch::CallUnit::Pred &p : cu.predecessors) {
      Value d = doneByCid.lookup(p.call);
      assert(d && "a call predecessor must be instantiated before its "
                  "consumer (they are in program order)");
      predDones.push_back(d);
    }
    Value startK = startForCall(cu, issue, predDones, concurrent, sh);
    assert(callees && "a CallUnit needs callee context");
    auto mit = callees->modules.find(cu.callee);
    assert(mit != callees->modules.end() &&
           "the callee module must be registered (emitted bottom-up first)");
    hw::HWModuleOp child = mit->second;

    // Instance inputs by child port name: clk/rst/`start` plus each read's data
    // input. An internal read consumes a backedge resolved after the instance;
    // a boundary read passes the top's data input straight through.
    llvm::StringMap<Value> ins;
    ins[kClk] = c.clkRaw;
    ins[kRst] = c.rst;
    ins[kStart] = startK;
    llvm::StringMap<circt::Backedge> rdBackedge;
    for (const uarch::CallUnit::MemArg &ma : cu.memArgs) {
      if (ma.isWrite)
        continue;
      if (ma.isBoundary)
        ins[ma.data] = pa.getInput(portData(ma.topBase));
      else {
        auto be = c.bb.get(memElemType(dp.mems[ma.mem], c.b));
        ins[ma.data] = be;
        rdBackedge.try_emplace(ma.data, be);
      }
    }
    // Channel ends: the child drives two of the three handshake wires and reads
    // the third. What it reads is a promise the channel realization resolves
    // once every end exists.
    for (auto [k, sa] : llvm::enumerate(cu.streamArgs)) {
      ComposedWires &w = composedWires[sa.chan];
      if (!sa.isInput) {
        ins[sa.ready] = w.prodReady;
        continue;
      }
      unsigned slot = 0;
      for (const uarch::StreamChannel::CallEnd &e :
           dp.streams[sa.chan].callEnds)
        if (dp.calls[e.call].streamArgs[e.arg].isInput) {
          if (e.call == cu.id && e.arg == k)
            break;
          ++slot;
        }
      ins[sa.data] = w.sinkData[slot];
      ins[sa.valid] = w.sinkValid[slot];
    }
    // Scalar operands: drive each child scalar-input port from its resolved
    // Source, sampled at the child's start.
    for (const uarch::CallUnit::ScalarArg &sa : cu.scalarIns)
      ins[sa.port] = resize(c.b, c.loc, resolveSource(sa.src), sa.width,
                            /*isSigned=*/true);

    auto outs = instantiateChild(c.b, c.loc, child,
                                 childInstanceName(cu.callee, cu.id), ins);

    // Scalar results: the child holds each result on its output port from
    // `done` onward, so that port is the survivor a sibling reads, with no
    // separate capture (`captureResults` skips a Call result). A survivor is
    // keyed by the region result it is yielded as, which is the call's own
    // index only where the call is the whole of what the region yields.
    for (auto [r, port] : llvm::enumerate(cu.resultPorts)) {
      callResultVal[accKey(cu.id, r)] = outs[port];
      for (auto [k, res] : llvm::enumerate(dp.regions[cu.region].results))
        if (res.value.kind == uarch::Source::Kind::Call &&
            res.value.id == cu.id && res.value.outPort == r)
          setSurvivor(cu.region, k, outs[port]);
    }

    // Scoped to this pass for the join above and the conjunction below, which
    // would otherwise read the previous pass's latched 1. Built on demand so a
    // region whose calls share no port emits it exactly where it used to.
    Value completed;
    auto completion = [&] {
      if (!completed)
        completed =
            concurrent ? outs[kDone] : c.completedSince(outs[kDone], issue);
      return completed;
    };
    // The window this child owns the ports it masters. A child drives its
    // addresses continuously and has no per-access pulse, so this is what a
    // port a second accessor also holds selects on.
    //
    // Armed from its release, which is in the window combinationally since the
    // register only carries it from the next cycle, and closed by its
    // completion COMBINATIONALLY as well: a gated sibling's start is the rising
    // edge of this one's `done` (`startFor`), so the two run on the very same
    // cycle and clearing a cycle later would leave both claiming the bus.
    Value driving;
    auto runWindow = [&] {
      if (!driving) {
        Backedge next = c.bb.get(c.i1);
        Value armed = c.orBits(startK, c.reg(next, c.f1));
        next.setValue(c.mux(completion(), c.f1, armed));
        driving = c.andBits(armed, c.notBit(completion()));
      }
      return driving;
    };

    masterCallPorts(cu, outs, rdBackedge, runWindow, sh);

    doneByCid[cu.id] = completion();
    dones.push_back(completion());
    if (!cu.streamArgs.empty())
      callOuts[cu.id] = std::move(outs);
  }
  // The region completes when every call has: the AND of their dones.
  Value all;
  for (Value d : dones)
    all = all ? c.andBits(all, d) : d;
  if (all)
    fb.callDone = all;
}

// Emit region \p rb's whole datapath (F) given the controller's \p issue;
// returns its store feedback. Every timing primitive here runs on the region's
// registered shell; deriving that shell (H) is the orchestrator's next step, on
// what this emits.
DatapathFeedback DatapathEmitter::emit(const uarch::RegionBlock &rb,
                                       Value issue) {
  bindStreamReads(rb);
  emitBeforeUnits(rb, issue);
  DatapathFeedback fb;
  // Calls precede the units: a call's scalar result is an ordinary Source a
  // chained unit reads directly. The reverse edge (a call operand computed by
  // this region's unit) closes through the unit backedges.
  emitCalls(rb, issue, fb);
  emitUnits(rb);
  emitAfterUnits(rb, issue);
  // Last of all: a store's data may be computed by a unit, and only here is
  // that a filled value rather than a dangling backedge.
  emitWrites(rb, issue, fb);
  return fb;
}

void DatapathEmitter::emitBeforeUnits(const uarch::RegionBlock &rb,
                                      Value issue) {
  emitRegisters(rb);
  declareUnits(rb);
  emitReads(rb, issue);
}

void DatapathEmitter::emitAfterUnits(const uarch::RegionBlock &rb,
                                     Value issue) {
  resolveRegHeads(rb);
  emitExternalReadAddrs(rb, issue);
}

} // namespace mlir::allo::uarch
