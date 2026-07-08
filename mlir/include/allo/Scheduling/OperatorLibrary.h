/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORLIBRARY_H
#define ALLO_SCHEDULING_OPERATORLIBRARY_H

#include "allo/Scheduling/MemoryModel.h" // MemoryLibrary + populateMemoryResources
#include "allo/Scheduling/Scheduler.h"
#include "allo/Scheduling/Utils.h"

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm::yaml {
template <typename T> struct MappingTraits;
} // namespace llvm::yaml

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Abstract operator vocabulary (hardware-facing, no MLIR op names).
//===----------------------------------------------------------------------===//

/// The abstract operator kind a hardware engineer characterizes. `classify`
/// maps concrete IR ops onto these; the YAML library keys on them. The three
/// cast kinds are split because int-resize, int<->float conversion, and
/// float-resize have distinct hardware timing.
enum class OpKind {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Neg,
  Cmp,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  Select,
  ICastI, // integer resize (sext / zext / trunc / index_cast)
  FCastI, // int <-> float conversion (si/ui-to-fp, fp-to-si/ui)
  FCastF, // float resize (extf / truncf)
  MemRead,
  MemWrite,
  StreamRead,
  StreamWrite,
  Unknown // op the classifier does not recognize -> falls to `default`.
};

/// The abstract datatype family. Integers use `Int` as an umbrella that also
/// covers `UInt`; only ops the IR marks unsigned resolve to `UInt`.
enum class OpDType { Int, UInt, Half, BFloat16, Float, Double, None };

/// The abstract signature of a concrete operation: what a library row matches.
struct OpSignature {
  OpKind kind = OpKind::Unknown;
  OpDType dtype = OpDType::None;
  unsigned width = 0; // datapath width in the IR (0 = no numeric type).
};

/// Classify \p op into its abstract signature. This is the only place that
/// knows MLIR op names; everything above the C++ boundary speaks the
/// vocabulary.
OpSignature classify(Operation *op);

//===----------------------------------------------------------------------===//
// Realization (`impl`): how a characterized operator becomes RTL.
//
// `impl` is a single string: a reserved native keyword, or an IP module name.
//===----------------------------------------------------------------------===//

/// The reserved native-realization keywords. Any `impl` value that is NOT one
/// of these is an IP module name (instantiated as an `hw.module.extern`).
///   `comb`    — emit a CIRCT `comb` primitive directly.
///   `hwarith` — emit CIRCT `hwarith` (reserved; emission not yet implemented).
///   `builtin` — native, backend picks the dialect (= `comb` today).
bool isNativeImpl(llvm::StringRef impl);

/// How a realized IP operator participates in the datapath's stall protocol --
/// an axis orthogonal to latency (a scheduling property) and to which module
/// realizes it (`impl`). Any pipelined IP, whatever the vendor, presents one of
/// these interface classes; the datapath emitter builds the instance's ports
/// and connections from the contract instead of hardcoding a single shape.
enum class StallContract {
  /// `(data.., clk) -> data`: fixed latency, no external freeze -- the pipeline
  /// free-runs. Correct only where nothing stalls it (a stall-free region).
  FreeRunning,
  /// `(data.., clk, ce) -> data`: fixed latency; `ce == 0` freezes the internal
  /// pipeline in lockstep with the shell's shift chains. The canonical FPGA
  /// hard-IP contract (e.g. a float-operator `aclken`) and the default for IP.
  ClockEnable,
  /// `(s_data,s_valid,s_ready,clk) -> (m_data,m_valid,m_ready)`:
  /// self-backpressuring,
  /// *dynamic* latency (AXI-Stream-like). Reserved -- needs variable-latency
  /// scheduling, not yet supported.
  Elastic,
};

/// The stall contract of realization `impl`. Native impls are combinational and
/// stateless, so they have none (asserted). Every IP defaults to `ClockEnable`;
/// a future oplib `stall:` field would let a specific operator override this.
StallContract stallContract(llvm::StringRef impl);

//===----------------------------------------------------------------------===//
// Library entries
//===----------------------------------------------------------------------===//

/// A width predicate `{min, max}` on an op's datapath width (either bound
/// optional; an exact width is `min == max`). Omitted for float rows, whose
/// width is fixed by `dtype`.
struct WidthRange {
  std::optional<uint32_t> min;
  std::optional<uint32_t> max;
};

/// The `{in, out}` map form of a `delay_ns`.
struct DelayMap {
  double in = 0.0;
  double out = 0.0;
};

/// A datapath delay (ns). In YAML a scalar is the symmetric shorthand
/// (`in == out`); a `{in, out}` map gives the two independently. Exactly one of
/// `scalar`/`map` is engaged, per the YAML node shape (see PolymorphicTraits).
struct DelaySpec {
  std::optional<double> scalar;
  std::optional<DelayMap> map;
  double inNs() const { return scalar ? *scalar : (map ? map->in : 0.0); }
  double outNs() const { return scalar ? *scalar : (map ? map->out : 0.0); }
};

/// One row of the operator library. A primary (`operators:`) row matches an
/// abstract `(kind, dtype, width)` signature; an advanced
/// (`advanced_operators:`) row matches a raw MLIR op name (an escape hatch for
/// power users). Either way it carries the timing characterization to apply.
struct OperatorEntry {
  // Primary match predicate (abstract).
  std::optional<OpKind> kind;   // `op:`   — the operator kind.
  std::optional<OpDType> dtype; // `dtype:`— absent means "any datatype".

  // Advanced match predicate (raw MLIR op name, e.g. "allo.stream.get").
  std::string mlirOp; // `mlir_op:` — empty unless this is an advanced row.

  // Optional width predicate on the op's datapath width (see `classify`).
  std::optional<WidthRange> width; // `width: {min, max}`

  uint32_t latency = 0; // in cycles

  // Datapath delay in ns (`delay_ns:`, a scalar or `{in, out}`).
  std::optional<DelaySpec> delay;

  // Whether the operator is fully pipelined (accepts a new input every cycle).
  // A non-pipelined operator occupies its resource unit for its whole latency.
  // Default true (matches typical HLS units, e.g. a pipelined multiplier).
  bool pipelined = true;

  // Allocation pool this operator draws from (a key in OperatorLibrary::units).
  // Ops sharing a pool contend for a limited number of units -> a ResII bound.
  std::optional<std::string> unit;

  // Realization (`impl:`): a native keyword (`comb`/`hwarith`/`builtin`) or an
  // IP module name. Required on every compute row; absent on mem/stream rows.
  std::string impl;
};

/// The timing characterization resolved for a specific operation.
struct OperatorChar {
  std::string typeName; // stable: one Problem::OperatorType per matched entry
  uint32_t latency = 0;
  double inDelay = 0.0;
  double outDelay = 0.0;
  bool pipelined = true;
  std::string unit;       // allocation pool name (empty = unlimited)
  uint32_t unitLimit = 0; // number of units in that pool
  std::string impl;       // realization: native keyword or IP module name
};

/// A parsed operator library: advanced (raw-name) rows, then abstract rows,
/// plus a mandatory default. `lookup` returns the first matching row's
/// characterization (advanced rows first), else the default, guaranteeing every
/// op is characterized.
class OperatorLibrary {
public:
  /// Parse a library from YAML text / a file. The built-in default reproduces
  /// the pre-timing fixed model (comb=0, mul/div/rem/fp=3, mem/stream=1).
  static llvm::Expected<OperatorLibrary> parse(llvm::StringRef yaml);
  static llvm::Expected<OperatorLibrary> loadFile(llvm::StringRef path);
  static const OperatorLibrary &defaultLibrary();

  /// The library-declared cycle time (ns), if any: `cycle_time_ns` directly, or
  /// derived from `frequency_mhz`. A pass option overrides this.
  std::optional<double> cycleTime() const;

  /// Resolve the characterization for \p op: first matching row, else default.
  OperatorChar lookup(Operation *op) const;

  /// The `advanced_operators` `mlir_op` names that are not registered
  /// operations in \p ctx. A non-empty result means the library names an op we
  /// cannot express (a typo, or a dialect the pass does not load) -- the caller
  /// should fail loudly rather than silently ignore the row.
  std::vector<std::string> unregisteredAdvancedOps(MLIRContext &ctx) const;

  /// The storage-timing (`memory:`) view of the same device file. `lookup`
  /// characterizes memory/stream accesses through it, not the operator table.
  const MemoryLibrary &memoryLibrary() const { return memory; }

private:
  friend struct llvm::yaml::MappingTraits<OperatorLibrary>;

  /// Resolve \p e to a characterization, filling the allocation pool from
  /// `units`.
  OperatorChar resolveEntry(const OperatorEntry &e,
                            llvm::StringRef typeName) const;

  std::string device;                 // informational (the FPGA part).
  std::optional<double> frequencyMhz; // target clock; -> cycle time.
  std::optional<double> cycleTimeNs;  // explicit cycle time (overrides freq).
  std::map<std::string, uint32_t>
      units; // `units:` allocation pools (name->count).
  std::vector<OperatorEntry>
      advancedEntries;                // `advanced_operators:` (matched first).
  std::vector<OperatorEntry> entries; // `operators:` (abstract).
  OperatorEntry defaultEntry;
  MemoryLibrary
      memory; // `memory:` -- the storage dimension of the device file.
};

/// The compiled-in default library YAML (informational; parsed by
/// OperatorLibrary::defaultLibrary).
llvm::StringRef defaultLibraryYAML();

//===----------------------------------------------------------------------===//
// Operator model: apply a library to a scheduling problem.
//
// Beyond per-op latency/delay, this attaches the resources the resource-aware
// schedulers bind against: a per-memref memory-port resource (count from the
// array's partition/storage attributes) and a per-pool compute-allocation
// resource. It is the seam between the library (characterization) and CIRCT's
// Problem data model.
//===----------------------------------------------------------------------===//

/// Assign an operator type (latency + delays) to every op reached by \p walkFn,
/// sourced from \p lib. The problem type selects what extra properties apply:
///   - a `ChainingProblem` also receives per-type incoming/outgoing delays;
///   - a `SharedOperatorsProblem` also receives a limited resource: a
///   per-memref
///     port resource on memory accesses, or a named allocation-pool resource on
///     compute ops.
/// These branches are compiled only for problem types that support them.
template <class ProblemT, class WalkFn>
LogicalResult populateOperatorTypesImpl(ProblemT &problem, WalkFn walkFn,
                                        const OperatorLibrary &lib) {
  using namespace circt::scheduling;
  constexpr bool isChaining = std::is_base_of_v<ChainingProblem, ProblemT>;
  constexpr bool isShared = std::is_base_of_v<SharedOperatorsProblem, ProblemT>;

  walkFn([&](Operation *op) {
    OperatorChar c = lib.lookup(op);
    Problem::OperatorType opr = problem.getOrInsertOperatorType(c.typeName);
    problem.setLatency(opr, c.latency);
    if constexpr (isChaining) {
      problem.setIncomingDelay(opr, c.inDelay);
      problem.setOutgoingDelay(opr, c.outDelay);
    }
    problem.setLinkedOperatorType(op, opr);

    if constexpr (isShared) {
      // A compute op contends for its named allocation pool. A limited resource
      // requires a non-zero-latency op (CIRCT invariant): a combinational op
      // cannot contend for a cycle-long slot. (Memory-port resources are the
      // storage dimension, assigned separately by populateMemoryResources.)
      if (c.latency > 0 && !c.unit.empty() && c.unitLimit > 0) {
        Problem::ResourceType rsrc =
            problem.getOrInsertResourceType("unit_" + c.unit);
        problem.setLimit(rsrc, c.unitLimit);
        problem.setLinkedResourceTypes(
            op, SmallVector<Problem::ResourceType>{rsrc});
        // A non-pipelined multi-cycle unit holds its resource for its whole
        // latency; record that occupancy for the resource-aware schedulers.
        unsigned occ = (c.pipelined || c.latency <= 1) ? 1u : c.latency;
        if (occ > 1)
          op->setAttr(
              sched::kResourceCyclesAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), occ));
      }
    }
  });
  return success();
}

/// Populate operator types for every op reachable from \p body (a loop body).
template <class ProblemT>
LogicalResult populateOperatorTypes(Block &body, ProblemT &problem,
                                    const OperatorLibrary &lib) {
  return populateOperatorTypesImpl(
      problem, [&](auto handle) { body.walk(handle); }, lib);
}

/// Populate operator types over the (walked) top-level ops of a straight-line
/// region.
template <class ProblemT>
LogicalResult populateOperatorTypes(ArrayRef<Operation *> ops,
                                    ProblemT &problem,
                                    const OperatorLibrary &lib) {
  return populateOperatorTypesImpl(
      problem,
      [&](auto handle) {
        for (Operation *top : ops)
          top->walk(handle);
      },
      lib);
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORLIBRARY_H
