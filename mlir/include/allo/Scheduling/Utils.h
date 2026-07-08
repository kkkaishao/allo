#ifndef ALLO_SCHEDULING_UTILS_H
#define ALLO_SCHEDULING_UTILS_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir::allo {
/// Dump a coarse cross-region dependence graph (analysis only) to a DOT file
/// (for visualization in Graphviz). The graph is keyed by region ID, with
/// edges labeled by dependence kind (RAW/WAR/WAW/SSA). The `funcName` argument
/// selects a single function to dump
FailureOr<std::string> dumpRegionDependenceAnaysis(ModuleOp module,
                                                   const std::string &funcName);

namespace sched {
/// Per-op: start time (cycle), relative to the op's scheduling region.
constexpr llvm::StringLiteral kStartTimeAttr = "allo.sched.t";
/// Per-op: sub-cycle start time (ns within the cycle), from the chaining solve.
constexpr llvm::StringLiteral kStartTimeInCycleAttr = "allo.sched.z";
/// Per-op: id of the scheduling region the op belongs to.
constexpr llvm::StringLiteral kRegionIdAttr = "allo.sched.region";
/// Func-level: array of per-region dictionaries (see the kRegionKey* fields).
constexpr llvm::StringLiteral kRegionsAttr = "allo.sched.regions";
/// Func-level: whole-kernel latency in cycles (sum of region latencies), set
/// only when every region's latency is known.
constexpr llvm::StringLiteral kLatencyAttr = "allo.sched.latency";
/// Func-level unit marker: the kLatencyAttr total is a worst-case bound (some
/// trip count came from an `allo.assume.ssa` range, not a constant).
constexpr llvm::StringLiteral kLatencyBoundAttr = "allo.sched.latency_is_bound";

/// Per-op scheduling *input* (not carrier output): the number of cycles a
/// non-pipelined limited op occupies its resource unit (= its latency). Absent
/// means fully pipelined (occupies its unit for a single cycle). Stamped by the
/// operator model; consumed by the resource-aware schedulers.
constexpr llvm::StringLiteral kResourceCyclesAttr = "allo.sched.rsrc_cycles";

/// Per-loop-op: this counted loop is a Phase B pipelined *level* -- its body's
/// child loops appear as nodes in one modulo problem. The value is the level's
/// scheduling region id. Set on the loop op itself (not its body ops), so the
/// reify can find and materialize the level loop into a `dcp.pipeline` even
/// when the level has no leaf ops of its own (all its children are loops).
constexpr llvm::StringLiteral kLevelAttr = "allo.sched.level";

// Keys of a per-region dictionary in the kRegionsAttr array.
constexpr llvm::StringLiteral kRegionKeyId = "id";
constexpr llvm::StringLiteral kRegionKeyKind = "kind"; // "cyclic" | "acyclic"
constexpr llvm::StringLiteral kRegionKeyII = "ii";     // omitted for acyclic
constexpr llvm::StringLiteral kRegionKeyLength =
    "length"; // number of cycle slots (single-iteration pipeline depth)
constexpr llvm::StringLiteral kRegionKeyLatency =
    "latency"; // cycles for the whole region (trip counts folded in)
constexpr llvm::StringLiteral kRegionKeyLatencyBound =
    "latency_is_bound"; // latency is a worst-case bound (assume-derived trip)
constexpr llvm::StringLiteral kRegionKeyParent =
    "parent"; // absorbed into a Phase B level (the level's region id); its
              // latency is already folded into the level, so it is excluded
              // from the top-level (program-order) latency composition
constexpr llvm::StringLiteral kRegionKeyParentStart =
    "parent_start"; // a child-loop node's start cycle within the level's II
} // namespace sched
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_UTILS_H
