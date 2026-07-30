/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_REGIONGRAPH_H
#define ALLO_SCHEDULING_REGIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::allo {

enum class RegionKind { Loop, StraightLine };

/// Coarse dependence kind between two regions. Memory edges distinguish
/// RAW/WAR/WAW; streams are elastic (any same-FIFO access is ordered, but a
/// FIFO decouples timing); SSA is an exact def-use edge.
enum class XEdgeKind { RAW, WAR, WAW, StreamElastic, SSA };

/// A scheduling region: a single affine loop, or a maximal run of non-loop ops.
struct SchedRegion {
  unsigned id;
  RegionKind kind;
  /// Top-level ops of the region (a Loop region holds its `affine.for`).
  SmallVector<Operation *> ops;

  Operation *anchor() const { return ops.front(); }
};

/// A coarse dependence edge; `src` precedes `dst` in program order.
struct XEdge {
  unsigned src;
  unsigned dst;
  XEdgeKind kind;
  Value root; // memref/stream root involved (null for SSA edges)
};

struct RegionGraph {
  SmallVector<SchedRegion> regions;
  SmallVector<XEdge> edges;

  /// True iff `from` can reach `to` via a directed path of length >= 1.
  bool reaches(unsigned from, unsigned to) const;
  /// Two regions are concurrent iff neither reaches the other.
  bool concurrent(unsigned a, unsigned b) const;
};

/// Partition a block into scheduling regions (loops + maximal straight-line
/// runs). The scheduler recurses this into imperfect-nest bodies.
///
/// Inside a *nested* block (a loop body, a while body, an `if` branch) a
/// synchronous sub-kernel call is additionally isolated into a region of its
/// own. Such a block is re-run by an enclosing container that drives its
/// children strictly serially, one per iteration, and completes each on its
/// real `done`, so a call there needs its own child region to be gated on.
/// The function's own entry block keeps a DETERMINATE call inside its span: its
/// regions are composed by the sequencer, where a fixed-latency call is a
/// time-triggered node that may legitimately overlap neighbouring work.
///
/// An INDETERMINATE call (`isIndeterminateCall`) is isolated in the entry block
/// too: it finishes at a data-dependent cycle, so a span sharing it would
/// schedule its own ops, loads of a buffer the child writes included, against a
/// start time that means nothing. Isolated, each consumer becomes a sibling
/// region the sequencer starts on the child's real `done`, and the call's
/// scalar results reach it as survivors (the child holds them on its output
/// ports from `done` onward). This applies only where a call becomes a leaf
/// CallUnit: a `composesOnStructuralTop` container wires every call as a
/// concurrent process, so nothing there is placed relative to one.
SmallVector<SchedRegion> enumerateRegions(Block &block);

/// Partition `func`'s entry block into scheduling regions (loops + maximal
/// straight-line runs).
SmallVector<SchedRegion> enumerateRegions(func::FuncOp func);

/// The structural axis of the controller discriminant. ONE rule, stated once:
/// the reifier reads it to charge a region's boundary cost, the emitter reads
/// it to pick a controller family, and neither derives it a second time.
enum class RegionShape {
  /// Runs a schedule itself: an II-paced pipeline or a straight-line
  /// sequential. A `dcp.instance` inside one is a fixed-latency datapath node
  /// (a `CallUnit`), not a child to sequence.
  Leaf,
  /// Drives child regions in its body (a loop wrapping an inner loop, or a
  /// sequential wrapper), one hierarchical pass per outer iteration.
  Container,
  /// Predicates its children: a `dcp.select`, run-once under its `condition`.
  Guard,
  /// Hands off to an instantiated module: a counted loop whose entire body is
  /// one `dcp.instance`, advanced by the child's real `done` rather than by a
  /// pipeline cadence. The child is on the *instance* substrate, which is why
  /// this is not a `Container` (it has no child regions).
  CallNode,
};

/// The shape of a reified region op (`dcp.pipeline` / `dcp.sequential` /
/// `dcp.select`), read off its body. Order matters: a select is a guard
/// whichever arms it has, a region holding child regions sequences them, and
/// only then does a lone-instance counted loop become the instance hand-off.
RegionShape dcpRegionShape(Operation *regionOp);

/// The same shape, asked of the SOURCE counted loop before its body is
/// materialized. THE shape decision on this side: the scheduler dispatches on
/// it to pick a problem, its composer to charge a boundary, and the reifier to
/// build the region, so none of the three can disagree about which loops
/// sequence children. `dcpRegionShape` asks it again of the region that comes
/// out, which is where a drift would be caught.
///
/// Asked of EVERY loop of a nest, including the outer levels of a perfect band:
/// each one above the innermost drives its child as a container. One SOLUTION
/// covers a whole band instead, which is why a flat walk of solutions cannot
/// see the levels a composition has to charge.
RegionShape countedLoopShape(LoopLikeOpInterface loop);

/// Whether a straight-line region carries a datapath, i.e. materializes into a
/// `dcp.sequential` at all. A span of nothing but declarations is left in place
/// (`isDeclarationOp`), so it forms no region and occupies no cycle.
bool spanFormsRegion(ArrayRef<Operation *> ops);

/// A DECLARATION: an op that names storage or a literal and binds no hardware.
/// A straight-line region of nothing but these carries no datapath, so the
/// reifier leaves it in place rather than wrapping it, and a level whose body
/// holds only these plus child loops has no work of its own to schedule.
bool isDeclarationOp(Operation *op);

/// A synchronous sub-kernel call: a plain (non-async) `func.call`, scheduled as
/// an opaque fixed-latency node. An async call composes structurally as
/// dataflow, ordered by its streams rather than by the schedule.
bool isSyncSubKernelCall(Operation *op);

/// The kernel a `func.call` names, whichever container the phase has: a
/// `func.func` while scheduling, a `dcp.module` once reified. Not filtered by
/// op type: a filter that misses returns null, and null reads as
/// "indeterminate callee" rather than as an error.
Operation *calleeOf(Operation *call);

/// A callee's whole-kernel static latency, from whichever carrier the current
/// phase has: `allo.sched.latency` on a `func.func` while scheduling, the
/// `dcp.module`'s own `latency` once the callee has been reified. Empty when
/// the callee's length is data-dependent. The partitioner runs in BOTH phases
/// and their descents must agree, so this reads either container; the reifier's
/// own `dcp.instance` timing reads it too, so one call cannot be indeterminate
/// to one of them and not the other.
///
/// Reification is post-order over the call graph, so a caller always asks this
/// of an already-reified callee and gets the exact number rather than the
/// scheduler's provisional one. The two carriers live on different OPS, so
/// violating that order means finding a `func.func` where a `dcp.module` is
/// asserted.
std::optional<int64_t> calleeStaticLatency(Operation *callee);

/// A sync call whose callee carries no static latency: its body is
/// data-dependent, so both its results and its writes land on the child's
/// `done`, at a cycle no static schedule can name. The region partitioner
/// isolates such a call; see `enumerateRegions`.
bool isIndeterminateCall(Operation *op);

/// Whether \p block holds a synchronous sub-kernel call anywhere under it. A
/// `while` body must decompose whenever it does: the flushing-pipeline schedule
/// issues an iteration per cycle, which no re-fired child instance can follow.
bool blockHasSyncCall(Block &block);

/// Whether \p op hands off through a STREAM anywhere under it, so the emitter
/// wraps it in a stall shell (`HWEmitter::emitRegion`) and back-pressure, not
/// its schedule, decides when it finishes.
///
/// The span such a region composes is a FLOOR, what the run costs with every
/// token available on time; a full output queue or a starved input stretches it
/// by an amount no static analysis names. So an elastic region carries no
/// static span at all (`composeSpan`), which makes its whole kernel
/// indeterminate and its callers gate on its real `done`.
///
/// Structural rather than an attempt to prove a given channel never stalls,
/// which is a whole-network analysis. Asked of affine/scf IR by the scheduler
/// and of the dcp regions by the reifier, so it keys on the stream ops
/// themselves, which reification keeps verbatim.
bool isElastic(Operation *op);

/// Whether a sync call can be modelled as a leaf CallUnit: every operand is a
/// memref or scalar and every result a scalar. It excludes a stream operand,
/// which is a latency-insensitive hand-off the leaf datapath cannot time.
bool callLowerable(func::CallOp call);

/// Whether \p func composes its children on the STRUCTURAL TOP rather than the
/// leaf: it has an `await` spawn (async), or wires children through a stream (a
/// plain KPN-style call whose operand is a `Stream`, concurrent even without
/// `await`). Read before reification (the scheduler,
/// `outline-loose-processes`); `spawnsConcurrently` is the same question asked
/// of one reified child.
bool composesOnStructuralTop(func::FuncOp func);

/// Whether \p invoke is a CONCURRENT child: an `await` spawn, or a call wired
/// to a sibling through a `Stream`. Either way its completion is ordered by
/// back-pressure rather than by a schedule, which is what makes its container a
/// process network. The reified counterpart of `composesOnStructuralTop`: that
/// one reads `func.call`s before reification, this one a `dcp.instance` after,
/// and they must agree about a container or the emitter would route it one way
/// and the latency model the other.
bool spawnsConcurrently(Operation *invoke);

/// Whether \p op is part of a concurrent container's own STRUCTURE: the calls
/// it composes, the channels / buffers / constant tables it declares, and the
/// constants feeding them. Everything else in such a container is loose
/// datapath, which `outline-loose-processes` lifts into a process of its own
/// and `verify-rtl-legality` rejects whatever the outliner had to leave behind.
bool isContainerStructure(Operation &op);

/// Whether a counted loop's body is decomposed into sub-regions, so the loop
/// becomes a sequential wrapper that runs its children in program order rather
/// than one flat modulo problem. True when the body nests a loop, and (via the
/// call isolation above) when it holds a sub-kernel call alongside anything
/// else: a flat modulo schedule has one issue cadence, and the loop controller
/// that re-fires a child per iteration advances on that child's `done`, so the
/// two cannot share a region. The scheduler and the reifier both read this, so
/// their descents stay in lockstep.
bool loopBodyDecomposes(LoopLikeOpInterface loop);

StringRef toString(XEdgeKind kind);

/// Emit the region graph as a DOT digraph (concurrent pairs as comments).
void printRegionGraphDot(const RegionGraph &graph, func::FuncOp func,
                         raw_ostream &os);

/// Topologically sort the synchronous call graph, as CALLSITES. Fails on a
/// cycle, diagnosed on the callsites that form it. For a consumer that binds
/// per callsite (two calls to one kernel pass different arrays), which is why
/// this granularity survives alongside `callGraphPostOrder` below.
llvm::FailureOr<SmallVector<Operation *>>
buildAndSortCallsiteGraph(func::FuncOp root);

/// The kernels reachable from \p root, CALLEES BEFORE CALLERS, with \p root
/// last. One entry per function, since the unit of work is the function and
/// several callsites may name one callee; external callees are dropped, having
/// no body to work on.
///
/// This order is what lets a caller read a fact its callee already published:
/// the pre-schedule verifier's stream directions, and the scheduler's callee
/// latency, on which the CALLER's own region partition depends
/// (`isIndeterminateCall`).
llvm::FailureOr<SmallVector<func::FuncOp>>
callGraphPostOrder(func::FuncOp root);
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_REGIONGRAPH_H
