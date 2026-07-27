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

#include "allo/IR/AlloAttrs.h"     // MemoryImplEnum
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
// Memory timing library: the `memory:` section of the device file. It holds
// read/write latency + delay per storage *implementation*
// (register/LUTRAM/BRAM/URAM) for array accesses, plus one FIFO (stream)
// timing. This is the storage analog of the operator library. Access timing is
// a function of the accessed memref's implementation, so the scheduler
// distinguishes a 0-cycle register from a 1-cycle BRAM from a multi-cycle URAM.
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
    // types, or they collapse onto one latency, so this keys the type.
    MemoryImplEnum impl = MemoryImplEnum::Auto;
  };
  /// Timing for a memory/stream access op (load/store/stream get/put); a
  /// zero-latency, zero-delay result if \p op is not a memory access. An array
  /// access is timed by its memref's implementation.
  Timing timing(Operation *op) const;

  /// The timing of storage implementation \p impl, or a zero (combinational)
  /// timing if the library declares no such primitive.
  MemKindTiming timing(MemoryImplEnum impl) const;

  /// The storage implementation an array access resolves to. A stream (timed
  /// by `fifo`) and a non-access both give `Auto`. Unlike `timing`, this does
  /// NOT consult the primitive table, so a caller can diagnose an
  /// implementation the device never declared *before* it falls to zero timing.
  MemoryImplEnum resolvedImpl(Operation *op) const;

  /// Whether the device declares timing for \p impl. The storage twin of
  /// `requiresUnmatchedIP`: an array resolving to an undeclared primitive would
  /// otherwise be scheduled at latency 0 and read before its data is valid.
  bool declares(MemoryImplEnum impl) const;

  MemoryImplEnum defaultImpl = MemoryImplEnum::LUTRAM; // unbound on-chip arrays
  std::vector<MemPrimitive>
      primitives;     // `memory: primitives:` (per-impl timing)
  MemKindTiming fifo; // `memory: fifo:` (stream get & put)
};

//===----------------------------------------------------------------------===//
// Per-memref storage shape, derived from the array's `allo.part` /
// `allo.bind.storage` attributes. These are the same partition/topology facts
// MemoryBankModel binds the resource-aware scheduler against, re-exposed for
// the microarch datapath (MemUnit) so both come from one model.
//===----------------------------------------------------------------------===//

struct MemoryChar {
  unsigned numBanks = 1;      // physical banks (block/cyclic partition factor)
  unsigned portsPerBank = 2;  // concurrent ports per bank (from bind.storage)
  bool readOnly = false;      // no write port needed (declared ROM, or by use)
  bool constantTable = false; // realized as a combinational constant array
  bool registers = false;     // complete partition -> scattered to registers
  MemoryImplEnum impl = MemoryImplEnum::LUTRAM; // resolved storage primitive
};

/// The `memref.global` initializer behind \p memRef, i.e. a constant table's
/// declared contents, or nullopt when it has none.
std::optional<Attribute> globalInitOf(Value memRef);

/// Whether \p memRef is a CONSTANT TABLE: it has a `memref.global` initializer
/// and nothing writes it. Read-only is a property of the USE, not of the
/// declaration, so an initialized array that is stored to even once is a real
/// memory that merely starts with contents, not a constant table.
///
/// This is the ONE definition the scheduler's port model and the emitter's ROM
/// realization must share. A constant table lowers to `hw.aggregate_constant`
/// read by one `hw.array_get` per access: combinational, no handshake, so
/// genuinely UNLIMITED-port. Billing it a 2-port RAM budget (which is what
/// deriving read-only from `allo.bind.storage` alone did) inflates II for free.
/// Note this is narrower than `MemoryChar::readOnly`: an explicit
/// `bind.storage type="rom_1p"` is a real memory whose ports the user chose.
///
/// Handing the array to a SUB-KERNEL also disqualifies it, whichever way the
/// child accesses it. A child MASTERS PORTS, driving addr/data/we into storage
/// the parent owns, and a constant table has none to master: it is a
/// combinational constant array, not a memory. Such an array therefore needs
/// real storage, keeping its declared values as power-on contents, and the port
/// model has to bill it.
bool isConstantTable(Value memRef);

/// Characterize a memref's storage shape from its partition/storage attributes
/// (a pure function of the attributes, independent of any scheduling region).
/// \p defaultImpl resolves an array with no explicit `allo.bind.storage impl=`;
/// pass the device's `MemoryLibrary::defaultImpl` so this agrees with the
/// implementation `MemoryLibrary::timing` resolved when it stamped the access
/// latencies (a hardcoded default here would silently disagree with the
/// schedule on any device whose `default_memory` is not that constant).
MemoryChar characterize(Value memref, MemoryImplEnum defaultImpl);

//===----------------------------------------------------------------------===//
// Partition and static-bank queries. A DCP banking pass reuses these facts so
// it materializes the *same* banks the scheduler bound its ResII against.
//===----------------------------------------------------------------------===//

/// The bank decomposition of a partitioned memref, in ELEMENT space: which bank
/// holds element `(i_0 .. i_{r-1})`, and where inside that bank it sits. This
/// is the single definition of "which bank", shared by the static split
/// (`dcp-resolve-banking`), the runtime crossbar (the emitter), and the
/// host-side layout (the interface manifest -> cosim), so all three materialize
/// the same banks the scheduler bound its ResII against.
///
/// A CYCLIC axis of factor F puts element `i_d` in bank `i_d mod F` at local
/// coordinate `i_d floordiv F`. A BLOCK axis puts it in bank
/// `i_d floordiv extent` at `i_d mod extent`, with `extent = ceil(S_d / F)`.
/// Several axes compose in mixed radix, in `allo.part` order, so the bank index
/// is `((b_1 * F_2) + b_2) * F_3 + ...`, exactly the fold the static split
/// performs. An axis with `dim == 0` in the attribute means *every* dimension,
/// which contributes one `Axis` each (so `numBanks` is `F^rank`, not `F`).
struct BankLayout {
  struct Axis {
    unsigned dim;   // 0-based memref dimension
    int64_t factor; // banks along this dimension
    bool block;     // block (contiguous chunks) vs cyclic (interleaved)
    int64_t extent; // per-bank extent of `dim` == ceil(shape[dim] / factor)
  };
  llvm::SmallVector<Axis, 2> axes; // mixed-radix order, most significant first
  llvm::SmallVector<int64_t, 4> bankShape; // per-bank extents, full memref rank
  unsigned numBanks = 1;                   // product of the axis factors
  bool registers = false;                  // complete partition: no banks

  /// Elements in one bank (the product of `bankShape`).
  int64_t bankWords() const;
};

/// Decode a memref's `allo.part` attribute into its element-space bank
/// decomposition (a single unpartitioned bank when there is no attribute).
BankLayout bankLayoutOf(Value memRef);

/// The compile-time bank of an access whose address map is \p map over a memref
/// of \p shape, or nullopt when the bank varies at runtime (a roaming access,
/// or any block axis whose subscript is not a constant). Generalizes
/// `staticBank` over every axis and both partition kinds; \p map may be in
/// element space (one result per dimension) or already linearized by
/// `dcp-flatten-memref` (one result), which is delinearized row-major before
/// the per-axis test.
std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    llvm::ArrayRef<int64_t> shape);

/// \p map rewritten as the single row-major linear element index it addresses,
/// simplified. The counterpart to `coordExpr`, which goes the other way, and
/// the ONE definition of the linear direction: `dcp-flatten-memref` rewrites an
/// access map with it, and the emitter evaluates the same expression to
/// hardware.
///
/// Doing this on the EXPRESSION rather than on emitted values is what makes the
/// delinearize/linearize pair of a coalesced nest cancel: `iv -> (iv floordiv
/// N, iv mod N)` composed with `(r, c) -> r*N + c` simplifies back to `iv`,
/// where the same round trip built out of `comb` ops is a divider, a modulo and
/// a multiplier that no later pass can fold.
///
/// A map that is already linear (one result over a rank>1 memref) is returned
/// unchanged, so this is total over both address-map forms.
AffineMap linearizeAccessMap(AffineMap map, llvm::ArrayRef<int64_t> shape);

/// A memref's `allo.part` partitioning, decoded. A projection of `BankLayout`
/// kept for the consumers that only need the aggregate facts (`factor` is
/// `BankLayout::numBanks`).
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

namespace mlir::allo {

/// Per-bank memory-port model. `observe` every memory access in a scheduling
/// region, `finalize` the per-memref banking decision, then `resource` gives
/// the port resource (key + limit) for an access. Base ports (2, or 1 for a
/// single-port `allo.bind.storage`) come from the array; block/cyclic
/// `allo.part` factors scale the aggregate. When a partitioned array's accesses
/// are all *statically banked*, every bank is a separate limited resource.
/// Statically banked means the cyclic-partition subscripts are
/// iteration-invariant modulo the factor, so each access hits a fixed bank.
/// Otherwise the array falls back to one aggregate-port resource, so the
/// refinement never under-counts.
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

} // namespace mlir::allo

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory resource model: apply the per-memref port/bank model to a scheduling
// problem. The storage twin of `populateOperatorTypes`. It attaches the
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
    MemoryBankModel banks;
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
