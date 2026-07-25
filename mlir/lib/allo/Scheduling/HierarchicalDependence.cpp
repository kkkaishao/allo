/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/HierarchicalDependence.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt::analysis;

LevelAnalysis mlir::allo::analyzeLevel(LoopLikeOpInterface level,
                                       DependenceAnalysis &deps) {
  LevelAnalysis result;
  Block &body = level.getLoopRegions().front()->front();
  Operation *levelOp = level.getOperation();
  // getLoopRegions().front() is a for-loop's sole body region; for scf.while
  // (before=condition, after=body) it would pick the condition, silently
  // dropping the body's ops/footprints/recurrences; callers gate on
  // AffineForOp/scf.ForOp so this never fires.
  assert(!isa<scf::WhileOp>(levelOp) &&
         "analyzeLevel: getLoopRegions().front() is the scf.while condition "
         "region, not its body");

  // Nodes = immediate children of the level body; map every op in a node's
  // subtree to that node and accumulate the node's footprint.
  llvm::DenseMap<Operation *, unsigned> owner;
  for (Operation &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    unsigned idx = result.nodes.size();
    LevelNode &n = result.nodes.emplace_back();
    n.anchor = &op;
    n.isLoop = isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op);
    op.walk([&](Operation *o) {
      owner[o] = idx;
      summarizeOp(o, result.nodes[idx].footprint);
    });
  }
  auto &nodes = result.nodes;

  // Collect raw edges, then keep one per directed node pair at the tightest
  // (smallest) distance, the binding constraint.
  llvm::MapVector<std::pair<unsigned, unsigned>, LevelEdge> best;
  auto add = [&](unsigned s, unsigned d, int64_t dist, StringRef kind) {
    LevelEdge e{s, d, dist, kind};
    auto *it = best.find({s, d});
    if (it == best.end() || dist < it->second.distance)
      best[{s, d}] = e;
  };

  // (1) Same-iteration ordering (dist 0): the per-op affine analysis omits
  // the loop-independent edge for a cross-nesting-depth pair, so program-order
  // footprint conflicts supply it, ordering the earlier node before a shared,
  // non-read-only, non-disjoint root.
  for (unsigned i = 0, e = nodes.size(); i < e; ++i)
    for (unsigned j = i + 1; j < e; ++j) {
      for (const auto &kv : nodes[i].footprint.mem) {
        auto it = nodes[j].footprint.mem.find(kv.first);
        if (it == nodes[j].footprint.mem.end())
          continue;
        auto c = footprintConflict(kv.second, it->second);
        if (c == Conflict::None)
          continue;
        add(i, j, 0,
            c == Conflict::WAW   ? "waw"
            : c == Conflict::RAW ? "raw"
                                 : "war");
      }
      for (Value s : nodes[i].footprint.streams)
        if (nodes[j].footprint.streams.count(s))
          add(i, j, 0, "stream");
    }

  // (2) SSA def-use across nodes (dist 0).
  body.walk([&](Operation *v) {
    auto vIt = owner.find(v);
    if (vIt == owner.end())
      return;
    for (Value operand : v->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      auto dIt = owner.find(def);
      if (dIt == owner.end() || dIt->second == vIt->second)
        continue;
      add(dIt->second, vIt->second, 0, "ssa");
    }
  });

  // (3) Recurrences carried by this level (dist >= 1): each dependence
  // component's loop is matched by IDENTITY, not positional nesting depth,
  // since an interleaved scf.for is absent from the *affine* components
  // (misread otherwise when affine and scf loops mix).
  body.walk([&](Operation *v) {
    auto vIt = owner.find(v);
    if (vIt == owner.end())
      return;
    for (const MemoryDependence &md : deps.getDependences(v)) {
      if (!hasDependence(md.dependenceType))
        continue;
      auto sIt = owner.find(md.source);
      if (sIt == owner.end() || sIt->second == vIt->second)
        continue;
      ArrayRef<affine::DependenceComponent> comps = md.dependenceComponents;
      const auto *lvl =
          llvm::find_if(comps, [&](const affine::DependenceComponent &c) {
            return c.op == levelOp;
          });
      if (lvl != comps.end()) {
        // The level is a loop common to both accesses: read the distance it
        // carries, dropping it when an enclosing loop already carries it.
        bool drop = false, valid = true;
        int64_t dist = carriedDistanceAtLevel(
            comps, std::distance(comps.begin(), lvl) + 1, drop, valid);
        if (!drop && valid && dist >= 1)
          add(sIt->second, vIt->second, dist, "rec");
        continue;
      }
      // The level is absent from the components for an scf.for level, whose
      // iterations the affine test can't model: an access under it has a
      // level-invariant address, so two aliasing accesses conflict every
      // iteration (distance-1), unless an enclosing loop already carries it.
      if (llvm::all_of(comps, [](const affine::DependenceComponent &c) {
            return c.lb.value_or(0) == 0;
          }))
        add(sIt->second, vIt->second, 1, "rec");
    }
  });

  for (const auto &kv : best)
    result.edges.push_back(kv.second);
  return result;
}

void mlir::allo::logLevelAnalysis(const LevelAnalysis &analysis,
                                  LoopLikeOpInterface level) {
  if (!logging::detail::enabled(logging::Level::Debug))
    return;
  debug(Stage::Sched) << "hier-level "
                      << logging::detail::describe(level.getOperation()) << ": "
                      << analysis.nodes.size() << " nodes";
  for (auto [idx, n] : llvm::enumerate(analysis.nodes))
    debug(Stage::Sched) << "  node " << idx
                        << (n.isLoop ? " [loop] " : " [op] ")
                        << logging::detail::describe(n.anchor);
  for (const LevelEdge &e : analysis.edges)
    debug(Stage::Sched) << "  edge " << e.src << " -> " << e.dst << "  dist "
                        << e.distance << "  [" << e.kind
                        << (e.src == e.dst ? " self]" : "]");
}
