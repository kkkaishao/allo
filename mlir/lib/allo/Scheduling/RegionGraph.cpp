/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/RegionGraph.h"
#include "allo/IR/AlloOps.h" // kAlloAsyncAttr
#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/Footprint.h"
#include "allo/Scheduling/Utils.h" // sched::kLatencyAttr
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

//===----------------------------------------------------------------------===//
// Region enumeration
//===----------------------------------------------------------------------===//

bool mlir::allo::isSyncSubKernelCall(Operation *op) {
  return isa<func::CallOp>(op) && !op->hasAttr(kAlloAsyncAttr);
}

// Whether a call node's operand/result types are the ones a leaf CallUnit can
// carry: memrefs the child masters and scalars it reads / returns. The one
// definition, asked of a `func.call` before reification and of a
// `dcp.instance` after (`spawnsConcurrently`).
static bool lowerableSignature(TypeRange operands, TypeRange results) {
  return llvm::all_of(operands,
                      [](Type t) {
                        return isa<MemRefType, IndexType>(t) ||
                               t.isIntOrFloat();
                      }) &&
         llvm::all_of(results, [](Type t) { return t.isIntOrFloat(); });
}

bool mlir::allo::callLowerable(func::CallOp call) {
  return lowerableSignature(call.getOperandTypes(), call.getResultTypes());
}

std::optional<int64_t> mlir::allo::calleeStaticLatency(func::FuncOp callee) {
  auto read = [&](llvm::StringRef name) -> std::optional<int64_t> {
    if (auto a = callee->getAttrOfType<IntegerAttr>(name))
      return a.getInt();
    return std::nullopt;
  };
  if (std::optional<int64_t> l = read("dcp.latency"))
    return l;
  return read(sched::kLatencyAttr);
}

bool mlir::allo::isIndeterminateCall(Operation *op) {
  if (!isSyncSubKernelCall(op))
    return false;
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      op, cast<func::CallOp>(op).getCalleeAttr());
  return !callee || !calleeStaticLatency(callee);
}

bool mlir::allo::composesOnStructuralTop(func::FuncOp func) {
  bool structural = false;
  func.walk([&](func::CallOp c) {
    if (c->hasAttr(kAlloAsyncAttr) || !callLowerable(c))
      structural = true;
  });
  return structural;
}

bool mlir::allo::isContainerStructure(Operation &op) {
  // Every one of these is operand-free (or a call), which is what lets
  // `outlineRun` place an outlined call at its run's last op without ever
  // using a value defined after it.
  return op.hasTrait<OpTrait::ConstantLike>() ||
         isa<func::CallOp, memref::AllocOp, memref::AllocaOp,
             memref::GetGlobalOp, StreamCreateOp>(op) ||
         op.hasTrait<OpTrait::IsTerminator>();
}

bool mlir::allo::spawnsConcurrently(Operation *invoke) {
  // `await` (broadcast start), or the same "not leaf-lowerable" signature test
  // `composesOnStructuralTop` applies pre-reification. In practice that is a
  // `Stream` operand, a back-pressured hand-off no schedule can time.
  return invoke->hasAttr(kAlloAsyncAttr) ||
         !lowerableSignature(invoke->getOperandTypes(),
                             invoke->getResultTypes());
}

SmallVector<SchedRegion> mlir::allo::enumerateRegions(Block &block) {
  SmallVector<SchedRegion> regions;
  // A DETERMINATE call is isolated only in a nested block; see the header for
  // why the function's own entry block keeps such calls inside their span.
  Operation *parent = block.getParentOp();
  bool isolateCalls = !isa_and_nonnull<func::FuncOp>(parent);
  // An INDETERMINATE one is isolated in the entry block too, but only where it
  // becomes a leaf CallUnit, the node other ops are scheduled against. `||`
  // short-circuits, so the cast runs only for a func's own block.
  bool isolateIndeterminate =
      isolateCalls || !composesOnStructuralTop(cast<func::FuncOp>(parent));

  SmallVector<Operation *> pending; // accumulating straight-line run
  auto flush = [&]() {
    if (pending.empty())
      return;
    regions.push_back({(unsigned)regions.size(), RegionKind::StraightLine,
                       SmallVector<Operation *>(pending)});
    pending.clear();
  };

  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    // A loop, or an `if` that survived if-conversion (guarding a loop/stream/
    // call, left opaque), is its own region: a single region-bearing op the
    // scheduler recurses into, not flattened into a straight-line span.
    if (isa<affine::AffineForOp, scf::ForOp, scf::WhileOp, affine::AffineIfOp,
            scf::IfOp>(&op)) {
      flush();
      regions.push_back({(unsigned)regions.size(), RegionKind::Loop, {&op}});
    } else if (isSyncSubKernelCall(&op) &&
               (isolateCalls ||
                (isolateIndeterminate && isIndeterminateCall(&op)))) {
      flush();
      regions.push_back(
          {(unsigned)regions.size(), RegionKind::StraightLine, {&op}});
    } else {
      pending.push_back(&op);
    }
  }
  flush();
  return regions;
}

bool mlir::allo::blockHasSyncCall(Block &block) {
  return block
      .walk([](Operation *op) {
        return isSyncSubKernelCall(op) ? WalkResult::interrupt()
                                       : WalkResult::advance();
      })
      .wasInterrupted();
}

bool mlir::allo::loopBodyDecomposes(LoopLikeOpInterface loop) {
  // A nested loop anywhere under the body (not just at its top level): an
  // `if` guarding a loop keeps that loop off the body's own op list, and an
  // affine.for enclosing an scf.for must not be treated as innermost either.
  for (Region *r : loop.getLoopRegions())
    if (r->walk([](Operation *op) {
           return isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op)
                      ? WalkResult::interrupt()
                      : WalkResult::advance();
         }).wasInterrupted())
      return true;
  return enumerateRegions(loop.getLoopRegions().front()->front()).size() > 1;
}

SmallVector<SchedRegion> mlir::allo::enumerateRegions(func::FuncOp func) {
  if (func.getFunctionBody().empty())
    return {};
  return enumerateRegions(func.getFunctionBody().front());
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
        // A shared-root conflict is a real ordering edge only when the regions'
        // footprints actually intersect (sub-range refinement inside).
        auto c = footprintConflict(kv.second, it->second);
        if (c == Conflict::None)
          continue;
        XEdgeKind kind = c == Conflict::WAW   ? XEdgeKind::WAW
                         : c == Conflict::RAW ? XEdgeKind::RAW
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
          auto *def = operand.getDefiningOp();
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

StringRef allo::toString(XEdgeKind kind) {
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

void allo::printRegionGraphDot(const RegionGraph &g, func::FuncOp func,
                               raw_ostream &os) {
  AsmState asmState(func);
  os << "digraph \"" << func.getSymName() << "\" {\n";
  for (const SchedRegion &r : g.regions) {
    StringRef kind = r.kind == RegionKind::Loop ? "loop" : "straightline";
    os << "  r" << r.id << " [label=\"r" << r.id << " " << kind << "\"];\n";
  }
  for (const XEdge &e : g.edges) {
    os << "  r" << e.src << " -> r" << e.dst << " [label=\""
       << toString(e.kind);
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

FailureOr<std::string>
allo::dumpRegionDependenceAnaysis(ModuleOp module,
                                  const std::string &funcName) {
  if (funcName.empty()) {
    return failure();
  }
  std::string s;
  llvm::raw_string_ostream os(s);
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (funcName.empty() || func.getSymName() == funcName)
      printRegionGraphDot(DependenceAnalysis(func).getRegionGraph(), func, os);
  if (s.empty())
    return failure();
  return os.str();
}

static void buildDepsRec(func::FuncOp fn, SymbolTableCollection &syms,
                         DenseMap<Operation *, SmallVector<Operation *>> &deps,
                         DenseSet<Operation *> &builtFns) {
  if (!builtFns.insert(fn).second)
    return; // already built this function's callsite deps
  fn.walk([&](func::CallOp call) {
    auto callee =
        syms.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (callee && !callee.isExternal()) {
      deps[call];
      buildDepsRec(callee, syms, deps, builtFns);
      callee.walk([&](func::CallOp inner) { deps[call].push_back(inner); });
    }
  });
}

// Topological sort of the synchronous call graph (callsites as nodes, edges to
// the callee's callsites). Returns false on a cycle, with a diagnostic on the
// first callsite in the cycle. The graph is a DAG if the program has no
// recursive synchronous calls (checked by `checkNoRecursiveCalls`).
static bool dfs(Operation *op,
                DenseMap<Operation *, SmallVector<Operation *>> &deps,
                llvm::SmallPtrSet<Operation *, 32> &visited,
                llvm::SmallPtrSet<Operation *, 32> &onStack,
                SmallVectorImpl<Operation *> &path,
                SmallVectorImpl<Operation *> &sorted) {
  if (visited.contains(op))
    return true;
  if (!onStack.insert(op).second) {
    auto *it = llvm::find(path, op);
    auto &diag = error(Stage::Prep, op)
                 << "Invalid cyclic call graph detected:";
    for (Operation *p : llvm::make_range(it, path.end()))
      diag << "\n  -> " << p->getLoc();
    diag << "\n  -> "
         << op->getLoc(); // repeat the first node to close the cycle
    return false;
  }
  path.push_back(op);
  for (Operation *dep : deps.lookup(op))
    if (!dfs(dep, deps, visited, onStack, path, sorted))
      return false;
  path.pop_back();
  onStack.erase(op);
  visited.insert(op);
  sorted.push_back(op);
  return true;
}

llvm::FailureOr<SmallVector<Operation *>>
allo::buildAndSortCallsiteGraph(func::FuncOp root) {
  SymbolTableCollection syms;
  DenseMap<Operation *, SmallVector<Operation *>> deps;
  DenseSet<Operation *> builtFns;
  SmallVector<Operation *> allCallsites;

  root->walk([&](func::CallOp call) {
    auto callee =
        syms.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (callee && !callee.isExternal()) {
      allCallsites.push_back(call);
      deps[call];
      buildDepsRec(callee, syms, deps, builtFns);
      callee.walk([&](func::CallOp inner) { deps[call].push_back(inner); });
    }
  });

  llvm::SmallPtrSet<Operation *, 32> visited, onStack;
  SmallVector<Operation *> path;
  SmallVector<Operation *> sorted;
  for (Operation *call : allCallsites)
    if (!dfs(call, deps, visited, onStack, path, sorted))
      return failure();
  return sorted;
}
