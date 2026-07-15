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
// and cross-region composition -- it does NOT reorder anything.
//===----------------------------------------------------------------------===//

#ifndef ALLO_SCHEDULING_REGIONGRAPH_H
#define ALLO_SCHEDULING_REGIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
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
SmallVector<SchedRegion> enumerateRegions(Block &block);

/// Partition `func`'s entry block into scheduling regions (loops + maximal
/// straight-line runs). Reused by the scheduler in S2.
SmallVector<SchedRegion> enumerateRegions(func::FuncOp func);

StringRef toString(XEdgeKind kind);

/// Emit the region graph as a DOT digraph (concurrent pairs as comments).
void printRegionGraphDot(const RegionGraph &graph, func::FuncOp func,
                         raw_ostream &os);

llvm::FailureOr<SmallVector<Operation *>>
buildAndSortCallsiteGraph(func::FuncOp root);
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_REGIONGRAPH_H
