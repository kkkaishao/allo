/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_REPORT_H
#define ALLO_MICROARCH_REPORT_H

#include "allo/Microarch/RegLedger.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo::uarch {

struct Datapath;

/// What the microarchitecture stage DECIDED, as data. A projection of the
/// `Datapath` taken once per emitted module, read back through JSON by Python.
///
/// Nothing here is re-derivable from the IR a consumer is handed, and nothing
/// duplicates the schedule report: op start cycles, region trip counts and
/// realizations are published there and joined on (func, region order). What is
/// only here is the BINDING and everything downstream of it, which is to say
/// what the emitter built rather than what the scheduler decided.
///
/// Write-only from the compiler's side. No pass may read it back; the moment
/// one does it is a second model of the datapath, and there is already one.

/// One functional-unit instance. `boundOps > 1` is exactly a sharing decision:
/// the trivial binding leaves every operation its own unit.
struct UnitReport {
  std::string identity; // `OperatorIdentity::key()`, the sharing equivalence
  std::string impl;     // the `dcp.operator` symbol; empty for a native unit
  std::string module;   // `operatorModuleName`; empty for a native unit
  unsigned width = 0;   // result width in bits
  unsigned latency = 0;
  unsigned boundOps = 1;
  bool comb = false; // native combinational, against an IP instance
  bool pipelined = true;
};

/// A class of multiplexer: `count` of them, each `fanin` sources wide at
/// `width` bits. Aggregated rather than enumerated because the cost of a mux is
/// a function of exactly those two numbers, and nothing downstream needs to
/// know WHICH ports a given mux feeds.
struct MuxClass {
  unsigned fanin = 0, width = 0, count = 0;
};

/// What the cost model needs of one array and no reader does: the ports the
/// model DEMANDED, and who drives them. Grouped apart because a reader asking
/// what an array became is answered by the fields above it.
struct MemCost {
  // `callReads`/`callWrites` are ports a CHILD drives, and `writingCalls`
  // counts the DISTINCT children among them: several ports of one child are
  // that child's own boundary, several children are genuinely concurrent
  // writers, and only the second is a banking problem.
  unsigned callReads = 0, callWrites = 0;
  unsigned writingCalls = 0, writingRegions = 0;
  // What `Datapath::portsNeeded` demanded of ONE bank, which is what the
  // storage had to be built to serve.
  unsigned portsNeededWrite = 0, portsNeededTotal = 0;
};

/// One array, and the storage decision taken for it.
struct MemReport {
  std::string owner;          // the name its ports are spelled from
  std::vector<int64_t> shape; // element shape
  unsigned width = 0;         // element bits
  unsigned banks = 1;
  std::string layout;  // "none", "cyclic", "block", "skew" or "complete"
  std::string storage; // the resolved `dcp.storage` realization
  unsigned depthWords = 0; // elements per bank
  unsigned readLatency = 0, writeLatency = 1;
  unsigned reads = 0, writes = 0; // accesses bound in this module
  MemCost cost;
  bool external = false, scattered = false, writesIndependent = false;
  bool rom = false, skewed = false;
  /// Whether the partition BOUGHT the bandwidth it costs: every access reaches
  /// one bank. An access the analysis could not fix takes a port on every bank,
  /// so a partition resolving none of them is N memories at the bandwidth of
  /// one. True for an unpartitioned array, which has nothing to resolve.
  bool partitionResolved = true;
};

/// One FIFO channel.
struct StreamReport {
  std::string owner;
  unsigned width = 0, depth = 0;
  bool crossesCall = false; // an end of it is a child port, not a local access
};

/// Sub-kernel invocations of one callee.
struct CallReport {
  std::string callee;
  unsigned count = 0;
  unsigned spawns = 0;            // of those, `await` spawns rather than calls
  std::optional<int64_t> latency; // the child's declared span, when static
};

/// What the cost model needs of one region and no reader does. Grouped apart
/// for the same reason as `MemCost`.
struct RegionCost {
  // Mux totals as the allocation charges them: inputs across every mux, and
  // 2:1-equivalent bits, since a k:1 mux costs about (k-1) 2:1 muxes per bit.
  unsigned muxInputs = 0, muxBits = 0;
  // The region's control plane: the iteration counter's width, and the address
  // strides riding beside it.
  unsigned counterWidth = 0, addrStrides = 0;
};

/// One region's allocation. `order` is the join key to the schedule report's
/// `RegionReport::order`: both are program order within the func.
struct RegionUarch {
  int64_t order = 0;
  std::string shape; // Leaf / Container / Guard / CallNode
  std::string kind;  // "cyclic" or "acyclic"
  std::optional<int64_t> interval;
  unsigned computeOps = 0; // operations bound to a unit in this region
  std::vector<UnitReport> units;
  std::vector<MuxClass> muxes;
  RegionCost cost;
};

/// One emitted module.
struct FuncUarch {
  std::string func;   // the `dcp` module symbol; joins to `FuncReport::name`
  std::string module; // the emitted `hw.module` name; joins to `Interfaces`
  bool top = false;
  std::vector<RegionUarch> regions;
  // Module-wide: a register run belongs to the value it carries, not to a
  // region, and the ledger counts it where it is BUILT.
  std::vector<RegClass> regs;
  std::vector<MemReport> mems;
  std::vector<StreamReport> streams;
  std::vector<CallReport> calls;
  unsigned readPorts = 0, writePorts = 0; // boundary port groups

  /// Project \p dp, plus the registers its emission built.
  FuncUarch(const Datapath &dp, llvm::StringRef symbol, llvm::StringRef module,
            const RegLedger &ledger);
  FuncUarch() = default;
};

/// One emission: every module it built, in emit order (callees before callers).
struct MicroarchReport {
  /// Bumped on a breaking rename. Its one purpose is that a baseline persisted
  /// to disk by a comparison tool is REFUSED rather than silently compared
  /// against a later schema; in-process the producer and consumer are the same
  /// build.
  static constexpr unsigned kVersion = 1;

  std::string binding;  // the sharing policy this emission ran under
  float cycleTime = 0;  // ns, the period the schedule was cut to
  std::vector<FuncUarch> funcs;

  /// The report as the JSON document Python parses. Absent optionals are
  /// OMITTED rather than null, as in the schedule report and the interface
  /// manifest, so a consumer tests for the field it needs.
  std::string toJSON() const;
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_REPORT_H
