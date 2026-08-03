/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_LATENCY_MODEL_H
#define ALLO_SCHEDULING_LATENCY_MODEL_H

#include "allo/IR/AlloAttrs.h"           // DeterminacyEnum
#include "allo/Scheduling/RegionGraph.h" // RegionShape

#include "mlir/IR/Block.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::allo {

/// The cycles a controller family spends at its region's BOUNDARIES, outside
/// the region's own schedule.
///
/// Only STRUCTURAL constants live here; a datapath-derived delay (a condition
/// cone's `tCond`, a region's `drainStage`) is passed in as a parameter.
struct BoundaryCost {
  /// `start` -> the first body pass issues.
  unsigned arm;
  /// A body pass completing -> the next one issues. Meaningless for a run-once
  /// family, which sets it equal to `arm`.
  unsigned reArm;
};

/// A `done` level is a latch, so a region's completion is visible one cycle
/// after the pulse that sets it. Every family pays it.
inline constexpr unsigned kDoneLatchCycles = 1;

/// A container's children read its counter as their own bound and sample it at
/// their own start, so every launch is registered, the first one included.
inline constexpr BoundaryCost kContainerBoundary{/*arm=*/1, /*reArm=*/1};

/// A call region's start->done span is the figure a caller composes against, so
/// its first pass launches on `start` itself through a start-cycle bypass.
/// Advances still ride the settled register.
inline constexpr BoundaryCost kCallNodeBoundary{/*arm=*/0, /*reArm=*/1};

/// A sequential-wrapper while re-evaluates its condition on a CHECK pulse one
/// cycle after `start` and after each body drain; the condition cone's own
/// `tCond` is added on top by the controller.
inline constexpr BoundaryCost kCheckedBoundary{/*arm=*/1, /*reArm=*/1};

/// A guard checks its predicate one cycle after `start`. That also keeps a
/// skipped guard's completion pulse off the `done` latch's start-clear, which
/// would otherwise leave no rising edge.
inline constexpr BoundaryCost kGuardBoundary{/*arm=*/1, /*reArm=*/1};

/// A NESTED acyclic region arms one cycle after `start`: its container advances
/// the outer counter on the child's start pulse, so the new index only settles
/// the cycle after.
inline constexpr BoundaryCost kAcyclicNestedBoundary{/*arm=*/1, /*reArm=*/1};

/// A TOP-LEVEL acyclic region has no outer counter to wait for, so it arms on
/// `start` directly, which makes a pure-sequential call container's latency
/// equal its schedule depth.
inline constexpr BoundaryCost kAcyclicTopBoundary{/*arm=*/0, /*reArm=*/0};

/// A PIPELINED leaf's `running` is a register set by `start`, so its first
/// iteration issues one cycle in. Iterations then overlap at the SOLVED `ii`,
/// which `leafSpan` takes as a parameter, so `reArm` does not describe this
/// family and is set equal to `arm`.
inline constexpr BoundaryCost kPipelinedBoundary{/*arm=*/1, /*reArm=*/1};

/// An EMPTY region, a runtime zero trip or a static `lb >= ub`, never launches
/// a pass at all: `gateStart` masks the start launch and a register on
/// `start && isEmpty` feeds the `done` latch. Two cycles, whichever family
/// drives the region.
///
/// A separate constant, not the arithmetic below at trip zero: both
/// expressions describe the steady state and are written for `trip >= 1`.
inline constexpr int64_t kEmptyRegionCycles = 2;

/// A done-paced region's whole span, given what one pass of its body costs
/// (\p bodySpan, the sum of its children's spans). Evaluated identically by
/// the scheduler (over affine/scf loops) and the reifier (over the dcp
/// regions built from them).
inline int64_t containerSpan(const BoundaryCost &boundary, int64_t trip,
                             int64_t bodySpan) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * (boundary.reArm + bodySpan) + bodySpan +
         kDoneLatchCycles;
}

/// A LEAF's whole span: it arms, issues \p trip iterations at its solved \p ii,
/// and then drains.
///
/// \p drain is the TERMINAL quantity, the cycles from the last issue pulse to
/// the deepest output committing, so `done` rises `drain + 1` cycles after that
/// pulse. It is NOT the schedule depth, which is only the cycle by which every
/// op has ISSUED.
inline int64_t leafSpan(const BoundaryCost &boundary, int64_t trip, int64_t ii,
                        int64_t drain) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * ii + drain + kDoneLatchCycles;
}

/// One region as the latency model sees it: enough to compose a span, and
/// nothing else. Built by two structural walks (the scheduler over affine/scf
/// loops, the reifier over the dcp regions built from them) that both feed
/// the same composition arithmetic above.
struct SpanNode {
  RegionShape shape = RegionShape::Leaf;
  /// Iterations of this region's body. Empty when data-dependent (a `while`, a
  /// dynamic bound), which leaves every enclosing span unknown rather than
  /// guessed.
  std::optional<int64_t> trip;
  /// A LEAF's own solved schedule: its issue cadence and its TERMINAL cycle,
  /// the delay from the last issue pulse to the deepest output committing. `ii`
  /// stays empty for an acyclic leaf, which issues once.
  ///
  /// `drain` and not the schedule DEPTH: the two differ by however much slack
  /// the solver left above the last commit, which is a number no boundary
  /// constant can stand in for.
  std::optional<int64_t> drain, ii;
  /// An INSTANCE element's whole start->done contract (see `instance`).
  std::optional<int64_t> contract;
  /// A worst case the SCHEDULER bounded from an `allo.assume.ssa` range, for a
  /// node whose own `trip` is data-dependent. Stands in for a span this node
  /// cannot compose, and only where a bound is a usable answer: exported as a
  /// kernel's own `latency` (flagged `latency_bound`, so a caller waits it
  /// out), never as a container's body pass, which has to pace a real counter.
  /// Carried here because reification keeps the bounded LATENCY but drops the
  /// assumed TRIP that produced it, so this side cannot re-derive it.
  std::optional<int64_t> assumedSpan;
  /// A straight-line span rather than a counted loop.
  bool acyclic = false;
  /// Paced by back-pressure rather than by its own schedule (`isElastic`), so
  /// it has no static span. Set on whichever node holds the stream access and
  /// on every node above it, though either alone would answer.
  bool elastic = false;
  /// Whether an enclosing region drives this one, the one boundary cost that
  /// depends on CONTEXT rather than on the node (`kAcyclicNestedBoundary`
  /// against `kAcyclicTopBoundary`).
  bool nested = false;
  /// An INSTANCE element rather than a region of this func: `contract` is the
  /// callee's whole start->done span, counted to its own `done` rising.
  bool instance = false;
  /// Body elements of a done-paced region, in program order. `std::vector`
  /// rather than `SmallVector`: the element type is this one, still incomplete
  /// here, and only `std::vector` is specified to accept that.
  std::vector<SpanNode> children;
};

/// The per-invocation span of \p n: its start pulse to its `done` rising. It is
/// WHOLE, including the node's own arming cost, so a composer only ever sums
/// spans.
///
/// A LEAF runs its own solved schedule and drains it. A DONE-PACED region runs
/// no schedule of its own: one pass is its body elements in sequence, each
/// handed to the next through its own `done` latch, and its controller re-arms
/// between passes.
///
/// nullopt whenever any element is data-dependent, which leaves the enclosing
/// span unknown rather than guessed.
std::optional<int64_t> composeSpan(const SpanNode &n);

/// A run of nodes composed in PROGRAM ORDER, each starting on its predecessor's
/// `done` edge, which costs nothing (`startFor` is a rising edge, not a
/// register). So this is the sum of their spans, and both compositions in the
/// compiler are this one function: a done-paced region's body pass, and a
/// func's top-level regions along one path of their DAG.
std::optional<int64_t> composeSequence(llvm::ArrayRef<SpanNode> nodes);

/// For each top-level node, in program order, the earlier nodes it must run
/// after. \p nodeOps gives the ops each node owns, which is what makes this
/// IR-agnostic: the scheduler passes its affine scheduling regions and the
/// reifier its dcp regions.
///
/// Three signals, the same three the emitter composes on
/// (`DatapathBuilder::recordSiblingDeps`): a shared memref, a shared stream
/// channel, and a cross-region SSA use. Everything else runs CONCURRENTLY.
///
/// Deliberately CONSERVATIVE, and it has to stay that way: a value merely
/// PASSED through a node counts as a touch, and two nodes that only READ one
/// array are still ordered, because they share its ports. A spurious edge only
/// serializes the model; a missing one claims an overlap the hardware does not
/// have. That is also why `RegionGraph`'s polyhedral refinement is NOT used
/// here: it drops exactly the edges the emitter keeps.
std::vector<llvm::SmallVector<unsigned, 2>>
siblingPredecessors(llvm::ArrayRef<llvm::SmallVector<Operation *>> nodeOps);

/// The nodes \p def ultimately reads, given \p owner (op -> owning node) and a
/// \p def that \p owner does not name. The reifier leaves a pure arith cone at
/// FUNC SCOPE whenever a top-level loop's induction bound or a top-level
/// guard's predicate is an expression (`for i in range(start, m+1)` in a
/// callee, `if k == 0` before a `dcp.select`), and such an op belongs to no
/// node, so an SSA edge routed through it reads as no edge at all. The cone is
/// combinational, so it carries the dependence of everything it reads: chase
/// through it to the ops a node does own.
///
/// Shared by the two relations that must not disagree about it
/// (`siblingPredecessors` and `DatapathBuilder::recordSiblingDeps`), which are
/// otherwise deliberately separate: they read different substrates, but "what
/// does this value ultimately come from" is one graph question.
llvm::SmallVector<unsigned, 2>
ownersThroughScope(Operation *def,
                   const llvm::DenseMap<Operation *, unsigned> &owner);

/// A func's top-level span: its regions composed over their dependence DAG.
///
/// The LONGEST PATH, not the sum. Independent siblings overlap, so summing them
/// reports a kernel as slower than its own hardware. \p preds is
/// `siblingPredecessors`, indexed alongside \p nodes.
std::optional<int64_t>
composeDag(llvm::ArrayRef<SpanNode> nodes,
           llvm::ArrayRef<llvm::SmallVector<unsigned, 2>> preds);

/// One materialized dcp region as the latency model sees it: the REIFY-side
/// structural walk, over `dcp.pipeline` / `dcp.sequential` / `dcp.select` and
/// the `dcp.instance` elements they hold. `SDC.cpp` has the other, over the
/// affine/scf loops these were built from.
///
/// Public because the EMITTER composes with it too. A region's span and its
/// composition class are derived where they are used, not read back off an
/// attribute, which would be a second copy of a computed number.
///
/// \p topLevel is a func's entry block, which composes an exported contract; a
/// region body composes one pass of a counter. See `dcpSpanNodes`.
SpanNode dcpSpanNode(Operation *regionOp, bool topLevel);

/// The elements of a reified block, in program order.
///
/// The two scopes differ in what a span is FOR. A func's entry block composes
/// an exported contract, so an assume-bounded region contributes its bound,
/// which a caller then waits out. A region body composes one pass of a COUNTER,
/// where a bound cannot pace anything, so a bounded child leaves the container
/// done-paced instead.
std::vector<SpanNode> dcpSpanNodes(Block &block, bool topLevel);

/// How a materialized region is PACED: which controller family drives it, and
/// the single-run span a container may time-trigger it against.
struct RegionTiming {
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;
  /// Present only for `counted_static`, and then it is exact.
  std::optional<int64_t> staticLatency;
};

/// Derive \p regionOp's pacing from the region itself.
///
/// ONE definition, called twice for two different purposes: the reifier calls
/// it to STAMP `latency` / `latency_bound` / `determinacy` onto the op, and the
/// emitter calls it to DECIDE a controller family. Those attributes are
/// therefore a report of this function, never an input to it.
///
/// Four classes, tested in order since each shadows the ones after it.
/// CONCURRENT children run to their own completion ordered by back-pressure, so
/// a span over them is a floor rather than a hand-off contract. CONDITIONAL
/// control decides when it ends, so no static span describes it. COUNTED_STATIC
/// is what remains when a span composes exactly, which is the contract a
/// container may time-trigger against; INDETERMINATE completes on its real
/// `done`.
RegionTiming dcpRegionTiming(Operation *regionOp);

/// Func-level: whole-kernel latency in cycles, the top-level regions composed
/// over their dependence DAG (`publishKernelLatency`). Set only when every
/// region has a composable span. Whether it is an exact count or an
/// assume-bounded worst case is NOT recorded: a bound is an upper one, so it
/// times a caller safely either way.
constexpr llvm::StringLiteral kLatencyAttr = "allo.sched.latency";

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_LATENCY_MODEL_H
