/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Coarse cross-region dependence analysis (§6b of the design doc). Region
// footprints are summarized at memref/stream *root* granularity; a conflict
// between two sibling regions on a shared root (not both read-only) yields an
// edge. SSA def-use across regions yields an exact edge. Analysis only.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/DependenceAnalysis.h"

#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloTypes.h"

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Root resolution + region footprints
//===----------------------------------------------------------------------===//

// Walk view-like ops to the underlying alloc/arg/global root (A1). Distinct
// roots are assumed non-aliasing (A2); an unresolved root stays itself (A3,
// handled by conservative equality on the root value).
static Value resolveRoot(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::SubViewOp>(def)) {
      v = op.getSource();
    } else if (auto op = dyn_cast<memref::CastOp>(def)) {
      v = op.getSource();
    } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = op.getSource();
    } else if (auto op = dyn_cast<memref::ViewOp>(def)) {
      v = op.getSource();
    } else {
      break;
    }
  }
  return v;
}

static Value streamBaseOf(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getStream();
  return cast<StreamPutOp>(op).getStream();
}

namespace {
struct Access {
  bool reads = false;
  bool writes = false;
};
struct Summary {
  DenseMap<Value, Access> mem; // memref root -> access
  DenseSet<Value> streams;     // stream roots touched (get or put)
};
} // namespace

static void summarizeOp(Operation *op, Summary &s) {
  if (auto rd = dyn_cast<affine::AffineReadOpInterface>(op)) {
    s.mem[resolveRoot(rd.getMemRef())].reads = true;
    return;
  }
  if (auto wr = dyn_cast<affine::AffineWriteOpInterface>(op)) {
    s.mem[resolveRoot(wr.getMemRef())].writes = true;
    return;
  }
  if (auto ld = dyn_cast<memref::LoadOp>(op)) {
    s.mem[resolveRoot(ld.getMemRef())].reads = true;
    return;
  }
  if (auto st = dyn_cast<memref::StoreOp>(op)) {
    s.mem[resolveRoot(st.getMemRef())].writes = true;
    return;
  }
  if (isa<StreamGetOp, StreamPutOp>(op)) {
    s.streams.insert(resolveRoot(streamBaseOf(op)));
    return;
  }
  // Opaque call: conservatively read+write every memref/stream operand root.
  if (isa<func::CallOp>(op)) {
    for (Value operand : op->getOperands()) {
      Type t = operand.getType();
      if (isa<MemRefType>(t)) {
        Access &a = s.mem[resolveRoot(operand)];
        a.reads = a.writes = true;
      } else if (isa<allo::StreamType>(t)) {
        s.streams.insert(resolveRoot(operand));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Region enumeration
//===----------------------------------------------------------------------===//

SmallVector<SchedRegion> mlir::allo::enumerateRegions(func::FuncOp func) {
  SmallVector<SchedRegion> regions;
  if (func.getFunctionBody().empty())
    return regions;

  SmallVector<Operation *> pending; // accumulating straight-line run
  auto flush = [&]() {
    if (pending.empty())
      return;
    regions.push_back({(unsigned)regions.size(), RegionKind::StraightLine,
                       SmallVector<Operation *>(pending)});
    pending.clear();
  };

  for (Operation &op : func.getFunctionBody().front()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    if (isa<affine::AffineForOp>(&op)) {
      flush();
      regions.push_back(
          {(unsigned)regions.size(), RegionKind::Loop, {&op}});
    } else {
      pending.push_back(&op);
    }
  }
  flush();
  return regions;
}

//===----------------------------------------------------------------------===//
// Graph construction
//===----------------------------------------------------------------------===//

const RegionGraph &DependenceAnalysis::getRegionGraph() {
  if (regionGraph)
    return *regionGraph;

  RegionGraph g;
  g.regions = enumerateRegions(func);

  // Map every op to its region.
  DenseMap<Operation *, unsigned> opRegion;
  for (const SchedRegion &r : g.regions)
    for (Operation *top : r.ops)
      top->walk([&](Operation *o) { opRegion[o] = r.id; });

  // Footprint per region.
  SmallVector<Summary> sums(g.regions.size());
  for (const SchedRegion &r : g.regions)
    for (Operation *top : r.ops)
      top->walk([&](Operation *op) { summarizeOp(op, sums[r.id]); });

  // Coarse memory + stream edges between sibling regions (program order i < j).
  for (unsigned i = 0, e = g.regions.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      for (const auto &kv : sums[i].mem) {
        auto it = sums[j].mem.find(kv.first);
        if (it == sums[j].mem.end())
          continue;
        bool wi = kv.second.writes, wj = it->second.writes;
        bool ti = kv.second.reads || wi, tj = it->second.reads || wj;
        if (!((wi && tj) || (ti && wj)))
          continue; // both read-only: no conflict
        XEdgeKind kind = (wi && wj) ? XEdgeKind::WAW
                         : wi       ? XEdgeKind::RAW
                                    : XEdgeKind::WAR;
        g.edges.push_back({i, j, kind, kv.first});
      }
      for (Value s : sums[i].streams)
        if (sums[j].streams.count(s))
          g.edges.push_back({i, j, XEdgeKind::StreamElastic, s});
    }
  }

  // Exact SSA def-use edges across regions (deduplicated per region pair).
  DenseSet<std::pair<unsigned, unsigned>> ssaSeen;
  for (const SchedRegion &r : g.regions)
    for (Operation *top : r.ops)
      top->walk([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (!def)
            continue;
          auto it = opRegion.find(def);
          if (it == opRegion.end() || it->second == r.id)
            continue;
          if (ssaSeen.insert({it->second, r.id}).second)
            g.edges.push_back({it->second, r.id, XEdgeKind::SSA, Value()});
        }
      });

  regionGraph = std::move(g);
  return *regionGraph;
}

//===----------------------------------------------------------------------===//
// Reachability / concurrency
//===----------------------------------------------------------------------===//

bool RegionGraph::reaches(unsigned from, unsigned to) const {
  SmallVector<unsigned> stack{from};
  DenseSet<unsigned> seen{from};
  while (!stack.empty()) {
    unsigned cur = stack.pop_back_val();
    for (const XEdge &e : edges) {
      if (e.src != cur)
        continue;
      if (e.dst == to)
        return true;
      if (seen.insert(e.dst).second)
        stack.push_back(e.dst);
    }
  }
  return false;
}

bool RegionGraph::concurrent(unsigned a, unsigned b) const {
  if (a == b)
    return false;
  return !reaches(a, b) && !reaches(b, a);
}

//===----------------------------------------------------------------------===//
// DOT dump
//===----------------------------------------------------------------------===//

StringRef mlir::allo::toString(XEdgeKind kind) {
  switch (kind) {
  case XEdgeKind::RAW:
    return "RAW";
  case XEdgeKind::WAR:
    return "WAR";
  case XEdgeKind::WAW:
    return "WAW";
  case XEdgeKind::StreamElastic:
    return "stream";
  case XEdgeKind::SSA:
    return "ssa";
  }
  return "?";
}

void mlir::allo::printRegionGraphDot(const RegionGraph &g, func::FuncOp func,
                                     raw_ostream &os) {
  AsmState asmState(func);
  os << "digraph \"" << func.getSymName() << "\" {\n";
  for (const SchedRegion &r : g.regions) {
    StringRef kind = r.kind == RegionKind::Loop ? "loop" : "straightline";
    os << "  r" << r.id << " [label=\"r" << r.id << " " << kind << "\"];\n";
  }
  for (const XEdge &e : g.edges) {
    os << "  r" << e.src << " -> r" << e.dst << " [label=\"" << toString(e.kind);
    if (e.root) {
      os << " ";
      e.root.printAsOperand(os, asmState);
    }
    os << "\"];\n";
  }
  for (unsigned i = 0, e = g.regions.size(); i < e; ++i)
    for (unsigned j = i + 1; j < e; ++j)
      if (g.concurrent(i, j))
        os << "  // concurrent: r" << i << " r" << j << "\n";
  os << "}\n";
}
