/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_HIERARCHICALDEPENDENCE_H
#define ALLO_SCHEDULING_HIERARCHICALDEPENDENCE_H

#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/Footprint.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::allo {

/// A node in a level's problem: an immediate child of the level body -- a child
/// loop (aggregated over its subtree) or a leaf op -- with its footprint.
struct LevelNode {
  Operation *anchor;
  bool isLoop;
  Summary footprint;
};

/// A dependence between two nodes carried by the level's loop.
struct LevelEdge {
  unsigned src;
  unsigned dst;
  int64_t distance;     // 0 = same-iteration ordering, >= 1 = recurrence
  llvm::StringRef kind; // "raw" | "war" | "waw" | "stream" | "ssa" | "rec"
};

struct LevelAnalysis {
  llvm::SmallVector<LevelNode> nodes;
  llvm::SmallVector<LevelEdge> edges;
};

/// Compute the hierarchical dependence analysis of one loop `level`.
LevelAnalysis analyzeLevel(LoopLikeOpInterface level, DependenceAnalysis &deps);

/// DEBUG-log a level analysis (no-op unless the `debug` log level is enabled).
void logLevelAnalysis(const LevelAnalysis &analysis, LoopLikeOpInterface level);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_HIERARCHICALDEPENDENCE_H
