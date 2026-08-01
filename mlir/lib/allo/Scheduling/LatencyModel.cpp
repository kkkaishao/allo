/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The latency arithmetic, in one place. Every cycle a region costs is charged
// here; the two structural walks that feed it (`SDC.cpp` over affine/scf loops,
// `PostConversion.cpp` over the dcp regions built from them) only report shape.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/LatencyModel.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryAccess.h" // resolveRoot (storage identity)

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::dcp;

// An ODS `I64Attr` accessor hands back `uint64_t`; every count in this model is
// signed. Converted once, here, rather than cast at each of the seven reads.
static std::optional<int64_t> asInt64(std::optional<uint64_t> v) {
  if (v)
    return static_cast<int64_t>(*v);
  return std::nullopt;
}

std::optional<int64_t> mlir::allo::composeSpan(const SpanNode &n) {
  // An instance is a whole start->done contract already; nothing of this func's
  // composes inside it.
  if (n.instance)
    return n.contract;
  // A guard runs under a predicate, so it has no span until the predicate has a
  // value, which is not at compile time.
  if (n.shape == RegionShape::Guard)
    return std::nullopt;
  // A stall shell stretches a run by whatever back-pressure costs it, so what
  // composes below is a floor and not a contract. Ahead of the bound, since a
  // floor is the wrong direction to hand a consumer either way.
  if (n.elastic)
    return std::nullopt;
  // A data-dependent trip has no composable span; a carried bound stands in
  // where the builder judged one usable, and is otherwise absent.
  if (!n.trip)
    return n.assumedSpan;
  if (n.shape == RegionShape::Container || n.shape == RegionShape::CallNode) {
    // A DONE-PACED region runs no schedule of its own, so `drain`/`ii` do not
    // describe it: one pass is its body elements in sequence, and the
    // controller re-arms between passes.
    std::optional<int64_t> body = composeSequence(n.children);
    if (!body)
      return std::nullopt;
    return containerSpan(n.shape == RegionShape::CallNode ? kCallNodeBoundary
                                                          : kContainerBoundary,
                         *n.trip, *body);
  }
  // A LEAF issues on its own controller's cadence and then drains. The acyclic
  // families are the one place the boundary depends on context rather than on
  // the region: a nested one waits for its container's counter to settle.
  if (!n.drain || (!n.acyclic && !n.ii))
    return std::nullopt;
  const BoundaryCost &boundary =
      !n.acyclic ? kPipelinedBoundary
                 : (n.nested ? kAcyclicNestedBoundary : kAcyclicTopBoundary);
  return leafSpan(boundary, *n.trip, n.acyclic ? 0 : *n.ii, *n.drain);
}

std::optional<int64_t> mlir::allo::composeSequence(ArrayRef<SpanNode> nodes) {
  int64_t sum = 0;
  for (const SpanNode &n : nodes) {
    std::optional<int64_t> span = composeSpan(n);
    if (!span)
      return std::nullopt;
    sum += *span;
  }
  return sum;
}

llvm::SmallVector<unsigned, 2>
mlir::allo::ownersThroughScope(Operation *def,
                               const DenseMap<Operation *, unsigned> &owner) {
  llvm::SmallVector<unsigned, 2> roots;
  llvm::SmallVector<Operation *, 4> work{def};
  llvm::SmallPtrSet<Operation *, 8> seen{def};
  while (!work.empty()) {
    Operation *o = work.pop_back_val();
    // An owned op is a root: whoever reads through the cone waits for it, and
    // its own operands are that node's business, not this walk's.
    if (auto it = owner.find(o); it != owner.end()) {
      if (!llvm::is_contained(roots, it->second))
        roots.push_back(it->second);
      continue;
    }
    for (Value v : o->getOperands())
      if (Operation *d = v.getDefiningOp(); d && seen.insert(d).second)
        work.push_back(d);
  }
  return roots;
}

std::vector<llvm::SmallVector<unsigned, 2>>
mlir::allo::siblingPredecessors(ArrayRef<SmallVector<Operation *>> nodeOps) {
  // Which node owns each op, so a cross-node SSA use can name the producer's
  // node rather than the producing op.
  DenseMap<Operation *, unsigned> owner;
  for (auto [i, ops] : llvm::enumerate(nodeOps))
    for (Operation *top : ops)
      top->walk([&, i = i](Operation *o) { owner[o] = i; });

  std::vector<llvm::SmallVector<unsigned, 2>> preds(nodeOps.size());
  auto addPred = [&](unsigned p, unsigned c) {
    if (p < c && !llvm::is_contained(preds[c], p))
      preds[c].push_back(p);
  };

  // One shared resource orders every node touching it, chained in program order
  // so the rest follows transitively.
  llvm::MapVector<Value, llvm::SmallVector<unsigned, 4>> sharers;
  // A scalar survivor: SSA dominance already puts the producer first.
  for (auto [i, ops] : llvm::enumerate(nodeOps))
    for (Operation *top : ops)
      top->walk([&, i = i](Operation *o) {
        for (Value v : o->getOperands()) {
          if (isa<MemRefType, StreamType>(v.getType())) {
            auto &touchers = sharers[resolveRoot(v)];
            if (!llvm::is_contained(touchers, unsigned(i)))
              touchers.push_back(i);
          }
          Operation *def = v.getDefiningOp();
          if (!def)
            continue;
          auto it = owner.find(def);
          if (it != owner.end()) {
            if (it->second != i)
              addPred(it->second, i);
            continue;
          }
          // A def no node owns is a func-scope cone; it carries the dependence
          // of everything it reads (`ownersThroughScope`).
          for (unsigned p : ownersThroughScope(def, owner))
            if (p != i)
              addPred(p, i);
        }
      });
  for (auto &entry : sharers)
    for (unsigned j = 1; j < entry.second.size(); ++j)
      addPred(entry.second[j - 1], entry.second[j]);
  return preds;
}

std::optional<int64_t>
mlir::allo::composeDag(ArrayRef<SpanNode> nodes,
                       ArrayRef<llvm::SmallVector<unsigned, 2>> preds) {
  assert(nodes.size() == preds.size() && "one predecessor set per node");
  int64_t total = 0;
  llvm::SmallVector<int64_t> finish(nodes.size(), 0);
  for (auto [i, n] : llvm::enumerate(nodes)) {
    std::optional<int64_t> span = composeSpan(n);
    if (!span)
      return std::nullopt;
    // A region with no predecessors starts with the kernel; one with them waits
    // on the joined `done` of all of them.
    int64_t start = 0;
    for (unsigned p : preds[i])
      start = std::max(start, finish[p]);
    finish[i] = start + *span;
    total = std::max(total, finish[i]);
  }
  return total;
}

std::vector<SpanNode> mlir::allo::dcpSpanNodes(Block &block, bool topLevel) {
  std::vector<SpanNode> nodes;
  for (Operation &inner : block)
    // An instance inside a region is a body element like any other. At kernel
    // scope there are none: the reify wraps every call into a region, which
    // keeps this list index-aligned with `siblingPredecessors`.
    if (isa<DCPathRegionOpInterface>(inner) ||
        (!topLevel && isa<DCPathInstanceOp>(inner)))
      nodes.push_back(dcpSpanNode(&inner, topLevel));
  return nodes;
}

// Whether \p op is driven by an enclosing dcp region rather than by the func's
// own sequencer. Asked of the op and not of the walk's `topLevel`, because
// `dcpRegionTiming` enters the walk part-way down for a region of its own.
static bool hasEnclosingRegion(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (isa<DCPathRegionOpInterface>(p))
      return true;
  return false;
}

SpanNode mlir::allo::dcpSpanNode(Operation *op, bool topLevel) {
  SpanNode n;
  if (auto inv = dyn_cast<DCPathInstanceOp>(op)) {
    // A callee's `latency` is already a start->done contract, counted to its
    // own `done` rising. It crosses a module boundary, so it is the one
    // composed number this side cannot derive and has to be told.
    n.instance = true;
    n.contract = asInt64(inv.getLatency());
    return n;
  }
  auto region = cast<DCPathRegionOpInterface>(op);
  n.shape = dcpRegionShape(op);
  n.nested = hasEnclosingRegion(op);
  n.elastic = isElastic(op);
  if (n.shape == RegionShape::Guard)
    return n; // a `dcp.select`: predicated, so no static span
  if (n.shape == RegionShape::Container || n.shape == RegionShape::CallNode) {
    n.trip = asInt64(region.getTrip());
    n.children = dcpSpanNodes(op->getRegion(0).front(), /*topLevel=*/false);
    return n;
  }
  n.drain = asInt64(region.getDrain());
  if (isa<DCPathSequentialOp>(op)) {
    n.acyclic = true;
    n.trip = 1;
    return n;
  }
  auto pipe = cast<DCPathPipelineOp>(op);
  if (!pipe.isWhileLoop()) {
    n.trip = asInt64(pipe.getTrip()); // a while leaves it unset: data-dependent
    n.ii = asInt64(pipe.getIi());
  }
  // A dynamic trip is stamped with no `trip` but keeps the scheduler's
  // assume-bounded worst case, which this side cannot re-derive: reification
  // keeps the loop's runtime bound operand, not the assumption that bounded it.
  if (!n.trip && topLevel && region.getLatencyBound())
    n.assumedSpan = asInt64(region.getLatency());
  return n;
}

RegionTiming mlir::allo::dcpRegionTiming(Operation *regionOp) {
  RegionTiming t;
  // CONCURRENT: the region holds a child wired as a process. A concurrent child
  // belongs to its NEAREST enclosing region, the one whose composition operator
  // becomes the network.
  bool concurrent = false;
  regionOp->walk([&](DCPathInstanceOp inv) {
    if (!spawnsConcurrently(inv))
      return;
    Operation *p = inv->getParentOp();
    while (p && !isa<DCPathRegionOpInterface>(p))
      p = p->getParentOp();
    concurrent |= p == regionOp;
  });
  if (concurrent) {
    t.determinacy = DeterminacyEnum::Concurrent;
    return t;
  }
  // CONDITIONAL: a guard or a while. Its own control decides when it ends, so
  // no static span describes it.
  auto pipe = dyn_cast<DCPathPipelineOp>(regionOp);
  if (isa<DCPathSelectOp>(regionOp) || (pipe && pipe.isWhileLoop())) {
    t.determinacy = DeterminacyEnum::Conditional;
    return t;
  }
  // COUNTED_STATIC when a span composes exactly, which is the contract a
  // container may time-trigger against; INDETERMINATE otherwise, completing on
  // its real `done`.
  t.staticLatency = composeSpan(dcpSpanNode(regionOp, /*topLevel=*/false));
  t.determinacy = t.staticLatency ? DeterminacyEnum::CountedStatic
                                  : DeterminacyEnum::Indeterminate;
  return t;
}
