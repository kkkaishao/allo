/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The storage dimension of the scheduling model, the counterpart to the
// operator (compute) library. It provides:
//   - MemoryLibrary: read/write latency + delay for array and FIFO accesses,
//     from the device `memory:` section;
//   - characterize(): a memref's banking/port shape, from its `allo.part` /
//     `allo.bind.storage` attributes;
//   - MemoryBankModel / populateMemoryResources: the per-memref port resources
//     the resource-aware schedulers bind against.
//===----------------------------------------------------------------------===//

#ifndef ALLO_SCHEDULING_MEMORYMODEL_H
#define ALLO_SCHEDULING_MEMORYMODEL_H

#include "allo/IR/AlloAttrs.h"     // MemoryImplEnum (storage vocabulary)
#include "allo/Scheduling/Utils.h" // sched::kResourceCyclesAttr

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory timing library: the `memory:` section of the device file -- read/write
// latency + delay per storage *implementation* (register/LUTRAM/BRAM/URAM) for
// array accesses, plus one FIFO (stream) timing; the storage analog of the
// operator library. Access timing is a function of the accessed memref's
// implementation, so the scheduler distinguishes a 0-cycle register from a
// 1-cycle BRAM from a multi-cycle URAM.
//===----------------------------------------------------------------------===//

/// Read/write latencies (cycles) of one storage kind.
struct RWLatency {
  unsigned read = 0;
  unsigned write = 0;
};

/// Read/write combinational delays (ns) of one storage kind.
struct RWDelay {
  double read = 0.0;
  double write = 0.0;
};

/// Timing of one storage kind (a primitive implementation, or the FIFO):
/// latency and delay, each split by direction. Grouped by metric to match the
/// YAML.
struct MemKindTiming {
  RWLatency latency;
  RWDelay delay;
};

/// One row of the `primitives:` table: the read/write timing of a storage
/// implementation (register/LUTRAM/BRAM/URAM).
struct MemPrimitive {
  MemoryImplEnum impl = MemoryImplEnum::Auto;
  MemKindTiming timing;
};

/// The storage-timing library. `timing` characterizes a memory/stream access by
/// resolving the accessed memref's implementation and indexing `primitives`;
/// the fields are filled from the `memory:` YAML section (absent -> zero).
class MemoryLibrary {
public:
  struct Timing {
    unsigned latency = 0;
    double delay = 0.0;
    bool pipelined = true;
    // The accessed array's resolved storage implementation (Auto for a stream).
    // Accesses of different implementations must map to *different* operator
    // types -- otherwise they collapse onto one latency -- so it keys the type.
    MemoryImplEnum impl = MemoryImplEnum::Auto;
  };
  /// Timing for a memory/stream access op (load/store/stream get/put); a
  /// zero-latency, zero-delay result if \p op is not a memory access. An array
  /// access is timed by its memref's implementation (see `resolveImpl`).
  Timing timing(Operation *op) const;

  /// The timing of storage implementation \p impl, or a zero (combinational)
  /// timing if the library declares no such primitive.
  const MemKindTiming &forImpl(MemoryImplEnum impl) const;

  MemoryImplEnum defaultImpl = MemoryImplEnum::LUTRAM; // unbound on-chip arrays
  std::vector<MemPrimitive>
      primitives;     // `memory: primitives:` (per-impl timing)
  MemKindTiming fifo; // `memory: fifo:` (stream get & put)
};

//===----------------------------------------------------------------------===//
// Per-memref storage shape, derived from the array's `allo.part` /
// `allo.bind.storage` attributes -- the same partition/topology facts the
// resource-aware scheduler binds against (see MemoryBankModel), re-exposed for
// the microarch datapath (MemUnit) so both come from one model.
//===----------------------------------------------------------------------===//

struct MemoryChar {
  unsigned numBanks = 1;     // physical banks (block/cyclic partition factor)
  unsigned portsPerBank = 2; // concurrent ports per bank (from bind.storage)
  bool readOnly = false;     // ROM
  bool registers = false;    // complete partition -> scattered to registers
  MemoryImplEnum impl = MemoryImplEnum::LUTRAM; // resolved storage primitive
};

/// Characterize a memref's storage shape from its partition/storage attributes
/// (independent of any scheduling region -- a pure function of the attributes).
MemoryChar characterize(Value memref);

//===----------------------------------------------------------------------===//
// Partition / static-bank queries -- the banking facts a DCP banking pass
// reuses so it materializes the *same* banks the scheduler bound ResII against.
//===----------------------------------------------------------------------===//

/// A memref's `allo.part` partitioning, decoded.
struct PartitionInfo {
  unsigned factor = 1;    // product of block/cyclic factors (physical banks)
  bool unlimited = false; // complete partition -> registers
  bool hasBlock = false;  // >=1 block axis (defeats per-bank refinement)
  llvm::SmallVector<std::pair<unsigned, int64_t>>
      cyclicAxes; // (0-based dim, factor)
};

/// Decode a memref's `allo.part` attribute (an empty PartitionInfo when the
/// memref is unpartitioned).
PartitionInfo partitionOf(Value memRef);

/// The compile-time bank of an affine address \p map's \p dim-th result under
/// cyclic \p factor, or nullopt when it is not iteration-invariant modulo the
/// factor. The core predicate the per-bank ResII refinement and the banking
/// pass key on. The `Operation *` overload resolves the map from a memory
/// access (used pre-schedule, on affine/memref ops); the banking pass passes a
/// dcp op's map directly (post-schedule dcp.load/store are not recognized as
/// memory accesses).
std::optional<int64_t> staticBank(AffineMap map, unsigned dim, int64_t factor);
std::optional<int64_t> staticBank(Operation *op, unsigned dim, int64_t factor);

} // namespace mlir::allo

namespace mlir::allo::detail {

/// Per-bank memory-port model. `observe` every memory access in a scheduling
/// region, `finalize` the per-memref banking decision, then `resource` gives
/// the port resource (key + limit) for an access. Base ports (2, or 1 for a
/// single-port `allo.bind.storage`) come from the array; block/cyclic
/// `allo.part` factors scale the aggregate. When a partitioned array's accesses
/// are all *statically banked* -- their cyclic-partition subscripts are
/// iteration-invariant modulo the factor, so each hits a fixed bank -- every
/// bank is a separate limited resource; otherwise the array falls back to one
/// aggregate-port resource, so the refinement never under-counts.
class MemoryBankModel {
public:
  void observe(Operation *op);
  void finalize();
  /// {resource key, port limit} for a memory access (limit 0 = unlimited, a
  /// complete partition), or nullopt if \p op is not a memory access.
  std::optional<std::pair<std::string, unsigned>> resource(Operation *op) const;

private:
  struct MemInfo {
    bool unlimited = false;   // complete partition -> registers (no port bound)
    bool perBank = false;     // eligible for the per-bank refinement
    bool splitRW = false;     // dedicated read/write ports (SimpleDualPort)
    unsigned sharedPorts = 2; // per bank, shared R/W (Single/TrueDual/default)
    unsigned readPorts = 0;   // per bank, dedicated read  (splitRW)
    unsigned writePorts = 0;  // per bank, dedicated write (splitRW)
    unsigned partitionFactor = 1; // product of block/cyclic factors (aggregate)
    llvm::SmallVector<std::pair<unsigned, int64_t>>
        cyclicAxes; // (0-based dim, factor)
    llvm::SmallVector<Operation *> accesses;
  };
  llvm::DenseMap<Value, MemInfo> byMemref;
};

} // namespace mlir::allo::detail

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory resource model: apply the per-memref port/bank model to a scheduling
// problem. The storage twin of `populateOperatorTypes` -- it attaches the
// limited port resources (and multi-cycle occupancy) that memory accesses bind
// against. Occupancy comes from the `MemoryLibrary`; the port key + limit come
// from the array's `allo.part` / `allo.bind.storage` attributes. Only a
// `SharedOperatorsProblem` carries limited resources, so this compiles to a
// no-op for any other problem type.
//===----------------------------------------------------------------------===//

/// Assign per-memref memory-port resources to every memory access reached by
/// \p walkFn, sourced from \p memLib.
template <class ProblemT, class WalkFn>
LogicalResult populateMemoryResourcesImpl(ProblemT &problem, WalkFn walkFn,
                                          const MemoryLibrary &memLib) {
  using namespace circt::scheduling;
  if constexpr (!std::is_base_of_v<SharedOperatorsProblem, ProblemT>) {
    return success();
  } else {
    detail::MemoryBankModel banks;
    walkFn([&](Operation *op) { banks.observe(op); });
    banks.finalize();
    walkFn([&](Operation *op) {
      MemoryLibrary::Timing t = memLib.timing(op);
      // A limited resource requires a non-zero-latency op (CIRCT invariant): a
      // combinational access cannot contend for a cycle-long slot.
      if (t.latency == 0)
        return;
      std::optional<std::pair<std::string, unsigned>> port = banks.resource(op);
      if (!port || port->second == 0) // non-memory, or unlimited (registers)
        return;
      Problem::ResourceType rsrc = problem.getOrInsertResourceType(port->first);
      problem.setLimit(rsrc, port->second);
      problem.setLinkedResourceTypes(op,
                                     SmallVector<Problem::ResourceType>{rsrc});
      // A non-pipelined multi-cycle port holds its resource for its whole
      // latency; record that occupancy for the resource-aware schedulers.
      unsigned occ = (t.pipelined || t.latency <= 1) ? 1u : t.latency;
      if (occ > 1)
        op->setAttr(
            sched::kResourceCyclesAttr,
            IntegerAttr::get(IntegerType::get(op->getContext(), 64), occ));
    });
    return success();
  }
}

/// Populate memory-port resources for every access reachable from \p body.
template <class ProblemT>
LogicalResult populateMemoryResources(Block &body, ProblemT &problem,
                                      const MemoryLibrary &memLib) {
  return populateMemoryResourcesImpl(
      problem, [&](auto handle) { body.walk(handle); }, memLib);
}

/// Populate memory-port resources over the (walked) top-level ops of a
/// straight-line region.
template <class ProblemT>
LogicalResult populateMemoryResources(ArrayRef<Operation *> ops,
                                      ProblemT &problem,
                                      const MemoryLibrary &memLib) {
  return populateMemoryResourcesImpl(
      problem,
      [&](auto handle) {
        for (Operation *top : ops)
          top->walk(handle);
      },
      memLib);
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYMODEL_H
