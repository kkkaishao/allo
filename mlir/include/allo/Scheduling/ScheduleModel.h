/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULE_MODEL_H
#define ALLO_SCHEDULING_SCHEDULE_MODEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo {

/// One op's place in its region's schedule.
struct OpSchedule {
  /// Start cycle, relative to the region's own start.
  int64_t start = 0;
  /// Sub-cycle start (ns within the cycle), from a chaining solve. Empty when
  /// the region was solved without one.
  std::optional<float> startInCycle;
};

/// The solved schedule of ONE region: what the solver DECIDED, and nothing the
/// IR already says. A trip count, a lower bound and a step are re-derived from
/// the loop; only a trip an ASSUMPTION bounded is here, because reification
/// keeps the loop's runtime bound operand and drops the assumption that bounded
/// it.
struct RegionSolution {
  /// Initiation interval. Empty for a straight-line span, which issues once.
  std::optional<int64_t> ii;
  /// Schedule DEPTH, the cycle by which every op has completed. A REPORT: it is
  /// stamped onto the region op for the schedule export, and a span composes
  /// from `drain` instead, because the solver may leave slack above the last
  /// commit.
  int64_t length = 0;
  /// The region's TERMINAL cycle: how long after its last issue pulse the
  /// deepest output commits, so its `done` rises `drain + 1` cycles after that
  /// pulse. What `leafSpan` composes.
  int64_t drain = 0;
  /// The region's own iteration count, one invocation of it. Empty for a
  /// straight-line span and for a data-dependent trip nothing bounds.
  ///
  /// A TRIP, not a latency: every field here is the region's own and
  /// per-invocation, so a span is composed where it is used (`composeSpan`) and
  /// no composed number is stored anywhere to go stale.
  std::optional<int64_t> trip;
  /// The trip above is a worst case derived from an `allo.assume.ssa` range
  /// rather than a compile-time constant, so every span composed from it is a
  /// bound.
  bool tripIsBound = false;
};

/// One scheduled op as the REPORT names it. Everything here reads off the
/// reified op, so an op is named the way its source op was rather than by the
/// dcp op that now stands for it.
struct ScheduledOpReport {
  /// An arith/affine-style mnemonic: `addi`, `mulf`, `load`, `stream.get`.
  std::string kind;
  /// Start cycle, relative to the region's own start.
  int64_t start = 0;
  /// The RTL module name of the IP instance that realizes it, empty for a
  /// combinational or memory op, which no IP realizes.
  std::string impl;
  /// Sub-cycle start (ns within the cycle), from a chaining solve.
  std::optional<float> z;
};

/// One scheduling region as the report names it: a `dcp.pipeline`,
/// `dcp.sequential` or `dcp.select`, with the ops it issues DIRECTLY. A nested
/// region's own ops are reported under that region, so an op appears once.
struct RegionReport {
  /// `cyclic` for a pipeline, `acyclic` for a straight-line span, `guard` for a
  /// select, which carries no compute of its own.
  std::string kind;
  /// Program order among its func's regions, and nesting depth among dcp
  /// regions (0 = outermost).
  int64_t order = 0, depth = 0;
  /// Whether a region nests inside it, which is what makes it a wrapper rather
  /// than a leaf, and whether its execution is predicated (a while pipeline or
  /// a guard).
  bool container = false, conditional = false;
  /// `length` is the schedule DEPTH, the cycle by which every op has completed;
  /// `drain` is the TERMINAL cycle its `done` composes off. Reported separately
  /// because a solver may leave slack between them, and only `drain` is
  /// charged.
  std::optional<int64_t> ii, trip, length, drain, latency;
  /// The latency above is an upper bound, not an exact count.
  bool latencyBound = false;
  /// The mnemonic of the region's determinacy class, the controller family that
  /// paces it. Empty only when the attribute is absent.
  std::string determinacy;
  std::vector<ScheduledOpReport> ops;
};

/// What ONE region's solve COST, which is a measurement of the compiler rather
/// than a fact about the hardware.
///
/// It rides beside the report rather than on a `dcp` op, because an attribute
/// would put a compile-time number into the IR the emitter reads, and nothing
/// downstream may ever build against how long a solve took.
///
/// Deliberately NOT joined to `RegionReport`: a solve is keyed by the affine
/// loop that owned the problem, and by the time the report is built off the
/// reified `dcp` ops that loop is gone. Both lists are in program order per
/// func, which is as much of a correspondence as is sound to claim.
struct SolveReport {
  /// The func it belongs to, and where its region is, as the log names it.
  std::string func, where;
  /// `cyclic` for a counted loop, `while` for a flushing while, `acyclic` for a
  /// straight-line span.
  std::string kind;
  /// Problem size: operations registered, and how many of those hold at least
  /// one limited unit.
  int64_t ops = 0, limitedOps = 0;
  /// The initiation interval settled, absent for an acyclic span.
  std::optional<int64_t> ii;
  /// Wall time of the whole solve in milliseconds.
  double millis = 0.0;
};

/// One kernel's schedule: an `allo.dcp.module` and the regions it holds.
struct FuncReport {
  std::string name;
  std::optional<int64_t> latency;
  bool latencyBound = false;
  /// The mnemonic of the kernel's determinacy class, the composition contract a
  /// caller holds its `latency` to. Empty only when the attribute is absent.
  std::string determinacy;
  std::vector<RegionReport> regions;
};

/// What the scheduling pipeline knows, in the two forms its two phases need.
///
/// The first is the SOLUTION, the hand-off from `runSDCScheduler` to
/// `runPostScheduleConversion`: per-op start times and per-region solutions.
///
/// The second is the REPORT, which the reify builds from the module it has just
/// written and Python reads back through `toJSON`. It is a second form rather
/// than a view of the first because by then the first has no keys left to read:
/// `forget` drops every op the reify erases, and everything the report names
/// (region kind, nesting, op mnemonic, IP impl, composed latency, determinacy)
/// is a fact about the reified `allo.dcp.*` ops that the solver never held. The
/// two are disjoint in time: the solution is valid only between the phases, the
/// report only after the second.
///
/// Keyed by `Operation *`, whose precondition is the invariant the two phases
/// already rely on: they run back to back over one module, with no pass between
/// them to fold, rebuild or erase an op. The reify may ADD to it, for a
/// condition cone it lifts or arithmetic it synthesizes for a symbolic bound,
/// and it reads a region's solution before it replaces the op that keys it, so
/// no lookup ever reaches a clone.
///
/// An `Operation *` key carries ONE obligation, `forget` below: a key is an
/// address, and an erased op's address is handed straight back out by the next
/// `create`.
///
/// A missing field is a compile error rather than a silent `nullopt`, and an
/// absent SOLUTION is distinguishable from a defaulted one, which is what keeps
/// a dropped descriptor from degrading an exact kernel to indeterminate with no
/// diagnostic.
class ScheduleModel {
public:
  /// Record \p op's solved start. An op is scheduled ONCE, by the solver or by
  /// the reify for a cone the solver never saw, never by both.
  void setStart(Operation *op, int64_t start) {
    bool inserted = ops.try_emplace(op, OpSchedule{start, std::nullopt}).second;
    assert(inserted && "an op carries one start time");
    (void)inserted;
  }

  /// Record \p op's sub-cycle start. Only meaningful alongside a start, and
  /// only ever read alongside one.
  void setStartInCycle(Operation *op, float z) {
    auto it = ops.find(op);
    assert(it != ops.end() && "a sub-cycle start belongs to a scheduled op");
    it->second.startInCycle = z;
  }

  /// \p op's place in its region's schedule, or null when it has none: a
  /// declaration, a terminator, a region anchor, an op no phase scheduled.
  const OpSchedule *scheduleOf(Operation *op) const {
    auto it = ops.find(op);
    return it == ops.end() ? nullptr : &it->second;
  }

  /// Open the solution OWNED by \p owner: the innermost loop of a counted band,
  /// a flushing `scf.while`, or a straight-line span's first op. That is the op
  /// both descents land on, which is what makes it the key rather than a
  /// synthetic id.
  RegionSolution &addRegion(Operation *owner) {
    auto [it, inserted] = regions.try_emplace(owner);
    assert(inserted && "a region is solved once");
    (void)inserted;
    return it->second;
  }

  /// \p owner's solution, or null when it owns none: a sequential wrapper, a
  /// `while` that cannot flush, an all-constant span the solver skipped.
  RegionSolution *regionOf(Operation *owner) {
    auto it = regions.find(owner);
    return it == regions.end() ? nullptr : &it->second;
  }

  /// Record that an `allo.assume.ssa` range bounds \p loop's iteration count at
  /// \p trip, for a loop whose exact count is not compile-time.
  void setTripBound(Operation *loop, int64_t trip) { tripBounds[loop] = trip; }
  /// The assumption-derived worst-case trip of \p loop, or empty when its trip
  /// is compile-time or nothing bounds it.
  std::optional<int64_t> tripBoundOf(Operation *loop) const {
    auto it = tripBounds.find(loop);
    return it == tripBounds.end() ? std::nullopt : std::optional(it->second);
  }

  /// Regions solved so far, module-wide. Sampled either side of one func to
  /// tell "this func solved nothing" from "this func solved a zero-cycle span",
  /// which are different answers to "what latency does it publish".
  size_t regionCount() const { return regions.size(); }

  /// Drop everything recorded about \p op, which every erase of a scheduled op
  /// owes the model.
  ///
  /// Not hygiene. MLIR frees an erased op and the next `create` may be handed
  /// that same address, so a stale entry would answer for an op no phase ever
  /// scheduled.
  void forget(Operation *op) {
    ops.erase(op);
    regions.erase(op);
    tripBounds.erase(op);
  }

  /// Read \p module's reified `allo.dcp.*` ops into `report`. Called once, at
  /// the tail of the reify, because the facts it collects exist only there:
  /// before it there are no dcp ops, and after it the pipeline is gone.
  void record(ModuleOp module);

  /// The report as the JSON document Python parses. Optional fields are
  /// OMITTED rather than null, as in the interface manifest, so a consumer
  /// tests for the field it needs instead of for a sentinel.
  std::string toJSON() const;

  /// The reified schedule, whole-module. Public because it is plain data: the
  /// reify fills it and the CAPI serializes it, and nothing else reads it.
  std::vector<FuncReport> report;

  /// What each region's solve cost, in solve order. Filled by the SOLVER, so
  /// unlike `report` it survives whether or not the reify runs, and it is the
  /// only per-region compile-time figure anything downstream can read.
  std::vector<SolveReport> solves;

private:
  llvm::DenseMap<Operation *, OpSchedule> ops;
  llvm::DenseMap<Operation *, RegionSolution> regions;
  llvm::DenseMap<Operation *, int64_t> tripBounds;
};

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULE_MODEL_H
