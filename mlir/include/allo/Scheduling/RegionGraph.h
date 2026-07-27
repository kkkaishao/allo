/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Coarse cross-region dependence graph. Nodes are scheduling regions (loops +
// maximal straight-line runs of a func's entry block); edges are coarse,
// root-level memory/stream/SSA dependences between sibling regions. This is the
// second tier of the analysis (the first being the per-region affine/stream
// precision used to build each SDC problem). It drives concurrency reporting
// and cross-region composition. It does NOT reorder anything.
//===----------------------------------------------------------------------===//

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

/// A DECLARATION: an op that names storage or a literal and binds no hardware.
/// A straight-line region of nothing but these carries no datapath, so the
/// reifier leaves it in place rather than wrapping it, and a level whose body
/// holds only these plus child loops has no work of its own to schedule.
bool isDeclarationOp(Operation *op);

/// A synchronous sub-kernel call: a plain (non-async) `func.call`, scheduled as
/// an opaque fixed-latency node. An async call composes structurally as
/// dataflow, ordered by its streams rather than by the schedule.
bool isSyncSubKernelCall(Operation *op);

/// A callee's whole-kernel static latency, from whichever carrier the current
/// phase has: `allo.sched.latency` while scheduling, `dcp.latency` once the
/// callee has been reified (reification strips the schedule carrier). Empty
/// when the callee's length is data-dependent. The partitioner runs in BOTH
/// phases and their descents must agree, so this reads both carriers. The
/// reifier's own `dcp.instance` timing reads it too, so one call cannot be
/// indeterminate to one of them and not the other.
std::optional<int64_t> calleeStaticLatency(func::FuncOp callee);

/// A sync call whose callee carries no static latency: its body is
/// data-dependent, so both its results and its writes land on the child's
/// `done`, at a cycle no static schedule can name. The region partitioner
/// isolates such a call; see `enumerateRegions`.
bool isIndeterminateCall(Operation *op);

/// Whether \p block holds a synchronous sub-kernel call anywhere under it. A
/// `while` body must decompose whenever it does: the flushing-pipeline schedule
/// issues an iteration per cycle, which no re-fired child instance can follow.
bool blockHasSyncCall(Block &block);

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

llvm::FailureOr<SmallVector<Operation *>>
buildAndSortCallsiteGraph(func::FuncOp root);
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_REGIONGRAPH_H
