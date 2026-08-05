/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULE_MODEL_H
#define ALLO_SCHEDULING_SCHEDULE_MODEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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
  /// Which allocated instance runs it: an index into
  /// `ScheduleModel::allocatedUnits`. Empty unless an exact solve allocated.
  std::optional<unsigned> unit;
};

/// The solved schedule of ONE region: what the solver DECIDED, and nothing the
/// IR already says. A trip count, a lower bound and a step are re-derived from
/// the loop; only a trip an ASSUMPTION bounded is here, because reification
/// keeps the loop's runtime bound operand and drops the assumption that bounded
/// it.
struct RegionSolution {
  /// Initiation interval. Empty for a straight-line span, which issues once.
  std::optional<int64_t> ii;
  /// Schedule DEPTH, the cycle by which every op has completed. A REPORT only:
  /// a span composes from `drain` instead, since the solver may leave slack
  /// above the last commit.
  int64_t length = 0;
  /// The region's TERMINAL cycle: how long after its last issue pulse the
  /// deepest output commits, so its `done` rises `drain + 1` cycles after that
  /// pulse. What `leafSpan` composes.
  int64_t drain = 0;
  /// The region's own iteration count, one invocation of it. Empty for a
  /// straight-line span and for a data-dependent trip nothing bounds. A TRIP,
  /// not a latency: every field here is per-invocation, so a span is composed
  /// where it is used (`composeSpan`) rather than stored.
  std::optional<int64_t> trip;
  /// The trip above is a worst case derived from an `allo.assume.ssa` range
  /// rather than a compile-time constant, so every span composed from it is a
  /// bound.
  bool tripIsBound = false;
};

/// One scheduled op as the REPORT names it: read off the reified op, but named
/// the way its source op was rather than by the dcp op standing for it.
struct ScheduledOpReport {
  /// An arith/affine-style mnemonic: `addi`, `mulf`, `load`, `stream.get`.
  std::string kind;
  /// Start cycle, relative to the region's own start.
  int64_t start = 0;
  /// The `dcp.operator` symbol realizing it, empty for a combinational or
  /// memory op, which no IP realizes. The emitted module name derives from it
  /// (`operatorModuleName`).
  std::string impl;
  /// Sub-cycle start (ns within the cycle), from a chaining solve.
  std::optional<float> z;
};

/// One scheduling region as the report names it, with the ops it issues
/// DIRECTLY. A nested region's own ops are reported under that region, so an
/// op appears once.
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
  /// `drain` is the TERMINAL cycle its `done` composes off. Separate because a
  /// solver may leave slack between them, and only `drain` is charged.
  std::optional<int64_t> ii, trip, length, drain, latency;
  /// The latency above is an upper bound, not an exact count.
  bool latencyBound = false;
  /// The mnemonic of the region's determinacy class, the controller family that
  /// paces it. Empty only when the attribute is absent.
  std::string determinacy;
  std::vector<ScheduledOpReport> ops;
};

/// What ONE region's solve COST: a measurement of the compiler, not the
/// hardware, so it is never stamped as an IR attribute the emitter could read.
///
/// Kept separate from `RegionReport`: a solve is keyed by the affine loop that
/// owned the problem, which no longer exists once the report is built off the
/// reified `dcp` ops. Both lists are in program order per func.
struct SolveReport {
  /// The func it belongs to, and where its region is, as the log names it.
  std::string func, where;
  /// `cyclic` for a counted loop, `while` for a flushing while, `acyclic` for a
  /// straight-line span.
  std::string kind;
  /// Problem size: operations registered, and how many of those hold at least
  /// one limited unit.
  int64_t ops = 0, limitedOps = 0;
  /// What the solve allocated: operations whose operator count it decided, and
  /// the instances it decided to build for them. Both zero where nothing was
  /// allocatable, and always zero for a heuristic solve.
  int64_t allocatedOps = 0, allocatedUnits = 0;
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

/// What the scheduling pipeline knows, in the two forms its two phases need:
/// the SOLUTION hands `runSDCScheduler`'s start times and region solutions to
/// `runPostScheduleConversion`, and the REPORT is what the reify builds from
/// the module it has just written, read back by Python through `toJSON`.
///
/// The two are valid at disjoint times: the solution between the phases, the
/// report only after, since by then `forget` has dropped every op the reify
/// erased. Keyed by `Operation *`, which stays valid across the two phases,
/// running back to back with no pass between them to fold or rebuild an op.
class ScheduleModel {
public:
  /// Record \p op's solved start. An op is scheduled ONCE, by the solver or by
  /// the reify for a cone the solver never saw, never by both.
  void setStart(Operation *op, int64_t start) {
    bool inserted =
        ops.try_emplace(op, OpSchedule{start, std::nullopt, std::nullopt})
            .second;
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

  /// One functional-unit instance an allocation decided to build. The reify
  /// declares a `dcp.unit` per entry and points the operations running on it at
  /// that symbol.
  struct AllocatedUnit {
    std::string name;   // the `dcp.unit` symbol, unique across the module
    std::string opType; // the `dcp.operator` it realizes
  };

  /// Declare \p count instances of \p opType and return the index of the first.
  /// Names are minted here, the one object spanning a whole module's
  /// scheduling, so a `dcp.unit` symbol is unique across it.
  unsigned addUnits(llvm::StringRef opType, unsigned count) {
    unsigned base = units.size();
    for (unsigned k = 0; k < count; ++k)
      units.push_back(
          {(opType + "_u" + llvm::Twine(units.size())).str(), opType.str()});
    return base;
  }

  /// Record that \p op runs on `allocatedUnits()[index]`.
  void setUnit(Operation *op, unsigned index) {
    auto it = ops.find(op);
    assert(it != ops.end() && "an instance belongs to a scheduled op");
    it->second.unit = index;
  }

  /// Every instance the allocation decided to build, module-wide.
  llvm::ArrayRef<AllocatedUnit> allocatedUnits() const { return units; }

  /// \p op's place in its region's schedule, or null when it has none: a
  /// declaration, a terminator, a region anchor, an op no phase scheduled.
  const OpSchedule *scheduleOf(Operation *op) const {
    auto it = ops.find(op);
    return it == ops.end() ? nullptr : &it->second;
  }

  /// Open the solution OWNED by \p owner: the innermost loop of a counted band,
  /// a flushing `scf.while`, or a straight-line span's first op. That is the op
  /// both descents land on, hence the key.
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

  /// What a region's own BOUNDARY expression evaluates to: a counted loop's
  /// runtime bounds, a guard's predicate. The scheduler expands the `AffineMap`
  /// or `IntegerSet` into operations so the solve can cut them, and the reify
  /// wires these values rather than expanding the same expression a second
  /// time. An entry the region does not need stays null.
  ///
  /// It travels here rather than as an SSA operand because an `affine.for`
  /// bound and an `affine.if` set take valid affine symbols, which the arith an
  /// expansion produces is not; keeping the loop and the guard untouched is
  /// also what keeps the dependence analysis exact.
  struct EntryCone {
    Value lower, upper;
    Value predicate;
  };

  /// Record what \p op's boundary expression evaluates to.
  void setEntryCone(Operation *op, const EntryCone &cone) {
    bool inserted = entries.try_emplace(op, cone).second;
    assert(inserted && "a region's boundary is expanded once");
    (void)inserted;
    for (Value v : {cone.lower, cone.upper, cone.predicate})
      if (v)
        entryValues.insert(v);
  }

  /// \p op's boundary values, or null when it has none: a constant-bound loop,
  /// or an `scf` region carrying its bound and condition as operands already.
  const EntryCone *entryConeOf(Operation *op) const {
    auto it = entries.find(op);
    return it == entries.end() ? nullptr : &it->second;
  }

  /// Whether \p v is one a boundary reads. Its only consumer names it through
  /// this map rather than through a use, so the region computing it has to be
  /// told to yield it.
  bool isEntryValue(Value v) const { return entryValues.contains(v); }

  /// Follow a recorded boundary value through the rewrite that replaces it,
  /// which is the reify wrapping the span computing it into a region.
  void replaceEntryValue(Value from, Value to) {
    if (!entryValues.erase(from))
      return;
    entryValues.insert(to);
    for (auto &[op, cone] : entries)
      for (Value *slot : {&cone.lower, &cone.upper, &cone.predicate})
        if (*slot == from)
          *slot = to;
  }

  /// Regions solved so far, module-wide. Sampled either side of one func to
  /// tell a func that solved nothing from one that solved a zero-cycle span,
  /// which publish different latencies.
  size_t regionCount() const { return regions.size(); }

  /// Drop everything recorded about \p op, which every erase of a scheduled op
  /// owes the model. MLIR frees an erased op and the next `create` may be
  /// handed that same address, so a stale entry would answer for an op no
  /// phase ever scheduled.
  void forget(Operation *op) {
    ops.erase(op);
    regions.erase(op);
    tripBounds.erase(op);
    entries.erase(op);
    for (Value res : op->getResults())
      entryValues.erase(res);
  }

  /// Read \p module's reified `allo.dcp.*` ops into `report`. Called once, at
  /// the tail of the reify: before it there are no dcp ops, and after it the
  /// pipeline is gone.
  void record(ModuleOp module);

  /// The report as the JSON document Python parses. Optional fields are
  /// OMITTED rather than null, as in the interface manifest, so a consumer
  /// tests for the field it needs instead of for a sentinel.
  std::string toJSON() const;

  /// The reified schedule, whole-module. Plain data: the reify fills it and the
  /// CAPI serializes it.
  std::vector<FuncReport> report;

  /// What each region's solve cost, in solve order. Filled by the SOLVER, so
  /// unlike `report` it survives whether or not the reify runs.
  std::vector<SolveReport> solves;

private:
  llvm::DenseMap<Operation *, OpSchedule> ops;
  std::vector<AllocatedUnit> units;
  llvm::DenseMap<Operation *, RegionSolution> regions;
  llvm::DenseMap<Operation *, int64_t> tripBounds;
  llvm::DenseMap<Operation *, EntryCone> entries;
  llvm::DenseSet<Value> entryValues;
};

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULE_MODEL_H
