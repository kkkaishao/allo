/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/RegionGraph.h"
#include "allo/IR/AlloOps.h" // kAlloAsyncAttr
#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/Footprint.h"
#include "allo/Scheduling/LatencyModel.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

bool mlir::allo::isDeclarationOp(Operation *op) {
  return isa<arith::ConstantOp, memref::AllocOp, memref::AllocaOp,
             memref::GetGlobalOp, StreamCreateOp>(op);
}

bool mlir::allo::spanFormsRegion(ArrayRef<Operation *> ops) {
  return llvm::any_of(ops, [](Operation *op) {
    return !op->hasTrait<OpTrait::IsTerminator>() && !isDeclarationOp(op);
  });
}

RegionShape mlir::allo::dcpRegionShape(Operation *regionOp) {
  if (isa<dcp::DCPathSelectOp>(regionOp))
    return RegionShape::Guard;
  assert((isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp>(regionOp)) &&
         "a shape is a property of a dcp REGION op, not of any op");
  bool childRegion = false, instance = false;
  for (Operation &inner : regionOp->getRegion(0).front()) {
    if (isa<dcp::DCPathPipelineOp, dcp::DCPathSequentialOp,
            dcp::DCPathSelectOp>(inner))
      childRegion = true;
    else if (isa<dcp::DCPathInstanceOp>(inner))
      instance = true;
  }
  if (childRegion)
    return RegionShape::Container;
  // Only a CYCLIC region hands off once per iteration; an acyclic one holding
  // an instance is a leaf whose datapath happens to include a call node.
  if (instance && isa<dcp::DCPathPipelineOp>(regionOp))
    return RegionShape::CallNode;
  return RegionShape::Leaf;
}

// Whether a call node's operand/result types are the ones a leaf CallUnit can
// carry: memrefs the child masters and scalars it reads / returns.
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

Operation *mlir::allo::calleeOf(Operation *call) {
  return SymbolTable::lookupNearestSymbolFrom(
      call, cast<func::CallOp>(call).getCalleeAttr());
}

std::optional<int64_t> mlir::allo::calleeStaticLatency(Operation *callee) {
  if (auto mod = dyn_cast<dcp::DCPathModuleOp>(callee))
    return mod.getLatency();
  if (auto a = callee->getAttrOfType<IntegerAttr>(kLatencyAttr))
    return a.getInt();
  return std::nullopt;
}

bool mlir::allo::isIndeterminateCall(Operation *op) {
  if (!isSyncSubKernelCall(op))
    return false;
  Operation *callee = calleeOf(op);
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
  // await, or the same signature test `composesOnStructuralTop` applies
  // pre-reification, in practice a Stream operand.
  return invoke->hasAttr(kAlloAsyncAttr) ||
         !lowerableSignature(invoke->getOperandTypes(),
                             invoke->getResultTypes());
}

SmallVector<SchedRegion> mlir::allo::enumerateRegions(Block &block) {
  SmallVector<SchedRegion> regions;
  // A DETERMINATE call is isolated only in a nested block, not the entry block.
  Operation *parent = block.getParentOp();
  bool isolateCalls = !isa_and_nonnull<func::FuncOp>(parent);
  // `||` short-circuits, so the cast runs only for a func's own block.
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
    // A loop, or an `if` that survived if-conversion, is its own region: the
    // scheduler recurses into it rather than flattening it into a span.
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

bool mlir::allo::isElastic(Operation *op) {
  return op
      ->walk([](Operation *inner) {
        return isa<StreamGetOp, StreamPutOp>(inner) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
      })
      .wasInterrupted();
}

RegionShape mlir::allo::countedLoopShape(LoopLikeOpInterface loop) {
  assert((isa<affine::AffineForOp, scf::ForOp>(loop.getOperation())) &&
         "a counted loop is an affine.for or an scf.for");
  if (loopBodyDecomposes(loop))
    return RegionShape::Container;
  return blockHasSyncCall(loop.getLoopRegions().front()->front())
             ? RegionShape::CallNode
             : RegionShape::Leaf;
}

bool mlir::allo::loopBodyDecomposes(LoopLikeOpInterface loop) {
  // A nested loop anywhere under the body, not just at its top level: an `if`
  // guarding a loop keeps that loop off the body's own op list, and an
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
        // footprints actually intersect.
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

// Topological sort of the synchronous call graph: callsites as nodes, edges to
// the callee's callsites. Returns false on a cycle, with a diagnostic on the
// first callsite in it.
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

llvm::FailureOr<SmallVector<func::FuncOp>>
allo::callGraphPostOrder(func::FuncOp root) {
  auto callsOr = buildAndSortCallsiteGraph(root);
  if (failed(callsOr))
    return failure();
  SymbolTableCollection syms;
  SmallVector<func::FuncOp> order;
  llvm::SmallPtrSet<Operation *, 32> seen;
  for (Operation *call : *callsOr) {
    auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
        call, cast<func::CallOp>(call).getCalleeAttr());
    if (callee && !callee.isExternal() && seen.insert(callee).second)
      order.push_back(callee);
  }
  // The root is not the callee of anything reachable from itself, so it is
  // appended rather than found. The guard covers a self-recursive shape the
  // cycle check would already have rejected.
  if (seen.insert(root).second)
    order.push_back(root);
  return order;
}
