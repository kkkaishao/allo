/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MEMORYMODEL_H
#define ALLO_SCHEDULING_MEMORYMODEL_H

#include "allo/IR/AlloAttrs.h"         // MemoryImplEnum
#include "allo/Scheduling/Scheduler.h" // OccupancyProblem

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

/// The width an `index` value widens back to once read as an ordinary index,
/// even though a counter or address register may be BUILT narrower
/// (`RegionBlock::counterType`, `RegionBlock::AddrStride::width`) wherever its
/// value range allows. Named here because both the emitter (building the
/// register) and the scheduler (pricing an index-typed value's delay chain)
/// price against it.
inline constexpr unsigned kIndexWidth = 32;

//===----------------------------------------------------------------------===//
// Memory timing library: the `memory:` section of the device file. Holds
// read/write latency + delay per storage implementation
// (register/LUTRAM/BRAM/URAM), plus one FIFO (stream) timing. The storage
// analog of the operator library.
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
/// and nothing writes it. Read-only is a property of the USE, not the
/// declaration: an initialized array stored to even once is a real memory
/// that merely starts with contents, not a constant table.
///
/// The ONE definition the scheduler's port model and the emitter's ROM
/// realization share. A constant table lowers to `hw.aggregate_constant` read
/// by one `hw.array_get` per access: combinational, no handshake, genuinely
/// UNLIMITED-port; billing it a 2-port RAM budget inflates II for free. This
/// is narrower than `MemoryChar::readOnly`: an explicit `bind.storage
/// type="rom_1p"` is a real memory whose ports the user chose.
///
/// Handing the array to a SUB-KERNEL also disqualifies it: a child MASTERS
/// PORTS, driving addr/data/we into storage the parent owns, and a constant
/// table has none to master. Such an array needs real storage, keeping its
/// declared values as power-on contents, so the port model must bill it.
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

/// The bank decomposition of a partitioned memref, in ELEMENT space: which
/// bank holds element `(i_0 .. i_{r-1})`, and where inside that bank it sits.
/// The single definition of "which bank", shared by the port model
/// (`MemoryBankModel`), the static split (`dcp-resolve-banking`), the runtime
/// crossbar (the emitter), and the host-side layout (interface manifest ->
/// cosim).
///
/// A CYCLIC axis of factor F puts element `i_d` in bank `i_d mod F` at local
/// coordinate `i_d floordiv F`. A BLOCK axis puts it in bank
/// `i_d floordiv extent` at `i_d mod extent`, with `extent = ceil(S_d / F)`. A
/// SKEWED axis puts it in bank `(sum over all k of i_k) mod F`, keeping
/// `i_d floordiv F` on its distribution dimension `d` and every other
/// subscript whole. Several axes compose in mixed radix, in `allo.part`
/// order: `((b_1 * F_2) + b_2) * F_3 + ...`. An axis with `dim == 0` means
/// *every* dimension, contributing one `Axis` each (`numBanks` is `F^rank`,
/// not `F`); a skew is never spelled that way, since its bank already reads
/// every subscript.
///
/// The skew exists because block and cyclic are both functions of ONE
/// subscript, so no axis choice serves an array read both as `A[i][j]` and
/// `A[j][i]`. What a skew buys is CONFLICT FREEDOM, not a compile-time bank:
/// `A[i][Fj+k]` and `A[Fj+k][i]` each reach F distinct banks as `k` runs over
/// the factor, so an unrolled group takes one port per bank instead of F (see
/// `skewSlotOf`).
struct BankLayout {
  /// How an axis maps a subscript onto banks. `Cyclic` interleaves, `Block`
  /// chunks, `Skew` reads every subscript (see the type comment).
  enum class Kind { Cyclic, Block, Skew };
  struct Axis {
    unsigned dim; // 0-based memref dimension (the DISTRIBUTION dim for a skew)
    int64_t factor; // banks along this dimension
    Kind kind = Kind::Cyclic;
    int64_t extent; // per-bank extent of `dim` == ceil(shape[dim] / factor)
  };
  llvm::SmallVector<Axis, 2> axes; // mixed-radix order, most significant first
  llvm::SmallVector<int64_t, 4> bankShape; // per-bank extents, full memref rank
  unsigned numBanks = 1;                   // product of the axis factors
  bool registers = false;                  // complete partition: no banks

  /// Elements in one bank (the product of `bankShape`).
  int64_t bankWords() const;

  /// The single skewed axis, or null. At most one is allowed, because the slot
  /// analysis reasons about ONE rotation of the bank index and two skews
  /// compose into two independent ones.
  const Axis *skew() const;
};

/// `kind` as the interface manifest spells it, the name the host reproduces the
/// decomposition from.
llvm::StringRef bankKindName(BankLayout::Kind kind);

/// Decode a memref's `allo.part` attribute into its element-space bank
/// decomposition (a single unpartitioned bank when there is no attribute). THE
/// decoder of that attribute, as `assignedBankOf` is the one reader of a
/// decided bank: a consumer that wants only the bank count or only the
/// complete-partition flag reads them off here rather than parsing again.
BankLayout bankLayoutOf(Value memRef);

/// The canonical spelling of \p part for a memref of \p type: a COMPLETE
/// partition collapses to its one whole-array axis, a `dim == 0` block or
/// cyclic axis expands into one axis per dimension, and the axes are sorted by
/// dimension. Null canonicalizes to null.
///
/// `bankLayoutOf` folds the axes IN ORDER into a mixed-radix bank index, so an
/// attribute's spelling is part of the bank index function, not presentation:
/// two attributes describing the same banking must be spelled identically
/// before a caller and callee can be said to agree on one (a sub-kernel
/// masters port group `k` of exactly the caller's bank `k`).
PartitionAttr canonicalizePartition(PartitionAttr part, MemRefType type);

/// The coarsest banking of a memref of \p type that satisfies both \p a and
/// \p b, canonical; failure with \p why set when the two cannot be reconciled.
///
/// The order is REFINEMENT: `a` is below `b` when every pair of elements `a`
/// places in distinct banks `b` does too. A partition directive is a LOWER
/// BOUND on the bank-distinctness its kernel needs, so a kernel scheduled
/// against the join still sees every access group it asked to be
/// conflict-free. A complete partition is the top (every element its own
/// register) and an absent attribute the bottom (one bank).
///
/// Axes on DIFFERENT dimensions compose in mixed radix with no reconciling
/// needed. On ONE dimension the join must remain a SINGLE axis (`allo.part`
/// admits no duplicate dimension), so it exists only when one factor divides
/// the other (and, for a block axis, the finer chunk boundaries fall on the
/// coarser ones). A block axis against a cyclic axis has no common
/// single-axis refinement at all: the array-read-both-ways conflict `Skew`
/// exists to answer.
llvm::FailureOr<PartitionAttr> joinPartitions(PartitionAttr a, PartitionAttr b,
                                              MemRefType type,
                                              std::string &why);

/// An access's bank index and in-bank offset, as affine EXPRESSIONS over the
/// address map's operands: each partitioned axis contributes its mixed-radix
/// digit to `bank`, and what remains of the subscripts re-linearizes over the
/// per-bank shape into `offset`. \p map is in ELEMENT SPACE, one result per
/// memref dimension; linearizing happens at the point of use, never in the IR.
///
/// Deriving this on the EXPRESSION rather than on emitted values is what
/// makes common banked idioms free: `A[2*i]` under cyclic-2 has bank
/// `(2*i) mod 2` and offset `(2*i) floordiv 2`, which `simplifyAffineExpr`
/// folds to `0` and `i` (no hardware), where the same derivation on emitted
/// values leaves a multiply/mask/shift nothing downstream can fold away. The
/// flat address takes this same route (`linearizeAccessMap`); this is its
/// banked twin.
struct BankSplitExpr {
  AffineExpr bank;   // which of `layout`'s banks, mixed radix in axis order
  AffineExpr offset; // the element's row-major index inside that bank
  /// `offset` before it is linearized: the element's coordinate on each
  /// dimension of the per-bank shape. The static split rewrites an access map
  /// in element space and so needs these rather than their row-major fold.
  llvm::SmallVector<AffineExpr, 4> coords;
};
BankSplitExpr bankSplitOf(const BankLayout &layout, AffineMap map,
                          llvm::ArrayRef<int64_t> shape);

/// The values a map operand takes, when a caller knows them: inclusive bounds
/// on the dim standing for it. `known == false` is "anything", which is what an
/// operand the caller cannot bound gets.
struct DimRange {
  int64_t lo = 0, hi = 0;
  bool known = false;
};

/// The compile-time bank of an access whose address map is \p map over a memref
/// of \p shape, or nullopt when the bank varies at runtime (a roaming access,
/// or any block axis whose subscript is not a constant).
///
/// This is `bankSplitOf(...).bank` when that expression is ONE VALUE: the
/// bank a consumer routes to and the bank the port model bills are the same
/// expression asked two questions, so they cannot drift apart. A cyclic digit
/// is one value when every variable coefficient of its subscript vanishes
/// modulo the factor.
///
/// \p ranges bounds the dims (the map's own numbering), which a block digit
/// needs: `A[i]` under block-2 of an `i32[16]` is `i floordiv 8`, which folds
/// for no `i` but is CONSTANT for every `i` a loop over `[0,8)` produces, so
/// the standard idiom (a loop per block) resolves nothing without it. An
/// empty \p ranges asks the folding question alone.
std::optional<int64_t> staticBankOf(const BankLayout &layout, AffineMap map,
                                    llvm::ArrayRef<int64_t> shape,
                                    llvm::ArrayRef<DimRange> ranges = {});

/// A skewed access's bank, decomposed into the part it shares with the array's
/// other accesses and the part that distinguishes it.
///
/// A skewed bank is `(sum of the subscripts) mod F`, splitting into a runtime
/// `cls` plus a compile-time constant `slot`, so the bank is
/// `(cls + slot) mod F`. Two accesses whose `cls` agree reach the same bank
/// exactly when their slots do, at every runtime value of `cls` (the bank
/// index is one rotation of the slot index, a bijection).
///
/// A slot is billable the way a static bank is: `assign-banks` records it in
/// `kBankAttr`, the port model bills a port on it, and F accesses with F
/// distinct slots take one port per bank instead of a port on every bank. The
/// emitter must NOT route to it directly: the physical bank is the slot
/// rotated by `cls`, known only at run time. `BankLayout::skew()` tells the
/// two readings of `kBankAttr` apart.
struct SkewSlot {
  AffineExpr cls;    // the runtime part of the bank's linear form
  unsigned slot = 0; // its constant part, modulo the factor
};

/// \p map's `SkewSlot` over a skewed \p layout, or nullopt when the layout is
/// not skewed or the sum does not split (a non-affine or dynamic subscript).
/// The caller must check that every access to the array agrees on `cls` before
/// billing the slots, since accesses of DIFFERENT `cls` can collide.
std::optional<SkewSlot> skewSlotOf(const BankLayout &layout, AffineMap map,
                                   llvm::ArrayRef<int64_t> shape);

/// Where an access carries the bank `assign-banks` DECIDED for it, before the
/// schedule is reified. Afterwards the fact lives in the `dcp.load`/`dcp.store`
/// op's own `bank` attribute, which a rewrite cannot silently drop the way a
/// discardable one can.
constexpr llvm::StringLiteral kBankAttr = "allo.bank";

/// The bank \p op was assigned, or nullopt when it reaches EVERY bank of its
/// memref: a roaming subscript, a non-affine index, or an `assign-banks` that
/// never ran. Reads whichever carrier the IR layer uses, so the port model,
/// the static split and the emitter consult one recorded decision rather than
/// each re-deriving `staticBankOf`. Nullopt is the conservative answer
/// everywhere (bill, route and address through all the banks).
std::optional<unsigned> assignedBankOf(Operation *op);

/// \p map, in element space, rewritten as the single row-major linear element
/// index it addresses, simplified. The ONE definition of the linear
/// direction, applied AT THE POINT OF USE by everything needing a flat
/// address, so pricing, strength reduction and the emitter cannot disagree.
///
/// Nothing rewrites the IR with it, deliberately: element space carries
/// per-dimension structure the linear form cannot be simplified back into
/// (`(6i+j) floordiv 6` does not fold to `i` without knowing `j < 6`), and the
/// bank split needs that structure.
///
/// Doing this on the EXPRESSION rather than on emitted values is what makes
/// the delinearize/linearize pair of a coalesced nest cancel: `iv -> (iv
/// floordiv N, iv mod N)` composed with `(r, c) -> r*N + c` simplifies back
/// to `iv`, where the same round trip built out of `comb` ops is a divider, a
/// modulo and a multiplier that no later pass can fold.
AffineMap linearizeAccessMap(AffineMap map, llvm::ArrayRef<int64_t> shape);

} // namespace mlir::allo

namespace mlir::allo {

/// Per-bank memory-port model. `observe` every memory access in a scheduling
/// region, `finalize` the per-memref storage shape, then `resources` gives the
/// port resources one access holds. Base ports (2, or 1 for a single-port
/// `allo.bind.storage`) come from the array; each `allo.part` bank is then a
/// separate limited resource with those ports.
///
/// An access holds one port on EVERY bank it can reach: the bank
/// `assign-banks` assigned it, or all of them when assigned none. The latter
/// is not a conservative bound but the crossbar the emitter builds (read
/// every bank and mux the result, or drive every bank and demux the write
/// enable), so a partitioned array under a roaming access sustains
/// `portsPerBank` concurrent accesses, not `portsPerBank * numBanks`.
class MemoryBankModel {
public:
  void observe(Operation *op);
  void finalize();
  /// The port resources \p op holds at once, as {resource key, ports per bank}:
  /// one entry per bank it reaches. The limit repeats because it is a property
  /// of the bank, not of the access. Empty when \p op is not a memory access,
  /// or when its storage has no port to contend for (a constant table, a
  /// complete partition).
  llvm::SmallVector<std::pair<std::string, unsigned>>
  resources(Operation *op) const;

private:
  struct MemInfo {
    bool unlimited = false;   // no port to bind (registers / constant table)
    bool splitRW = false;     // dedicated read/write ports (SimpleDualPort)
    unsigned sharedPorts = 2; // per bank, shared R/W (Single/TrueDual/default)
    unsigned readPorts = 0;   // per bank, dedicated read  (splitRW)
    unsigned writePorts = 0;  // per bank, dedicated write (splitRW)
    BankLayout layout;        // element-space banks (one bank when unbanked)
  };
  llvm::DenseMap<Value, MemInfo> byMemref;
};

} // namespace mlir::allo

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory resource model: applies the per-memref port/bank model to a
// scheduling problem, the storage twin of `populateOperatorTypes`. The port
// key + limit come from the array's `allo.part` / `allo.bind.storage`
// attributes. A port is a one-cycle reservation whatever its latency
// (`getResourceCycles`'s default), so no occupancy window is set here. Only
// an `OccupancyProblem` carries limited resources, so this is a no-op for any
// other problem type.
//===----------------------------------------------------------------------===//

/// Assign per-memref memory-port resources to every memory access reached by
/// \p walkFn.
template <class ProblemT, class WalkFn>
LogicalResult populateMemoryResourcesImpl(ProblemT &problem, WalkFn walkFn) {
  using namespace circt::scheduling;
  if constexpr (!std::is_base_of_v<OccupancyProblem, ProblemT>) {
    return success();
  } else {
    MemoryBankModel banks;
    walkFn([&](Operation *op) { banks.observe(op); });
    banks.finalize();
    walkFn([&](Operation *op) {
      SmallVector<Problem::ResourceType> units;
      for (auto &[key, limit] : banks.resources(op)) {
        Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
        problem.setLimit(rsrc, limit);
        units.push_back(rsrc);
      }
      if (units.empty()) // non-memory, or storage with no port to contend for
        return;
      problem.setLinkedResourceTypes(op, units);
    });
    return success();
  }
}

/// Populate memory-port resources for every access reachable from \p body.
template <class ProblemT>
LogicalResult populateMemoryResources(Block &body, ProblemT &problem) {
  return populateMemoryResourcesImpl(problem,
                                     [&](auto handle) { body.walk(handle); });
}

/// Populate memory-port resources over the (walked) top-level ops of a
/// straight-line region.
template <class ProblemT>
LogicalResult populateMemoryResources(ArrayRef<Operation *> ops,
                                      ProblemT &problem) {
  return populateMemoryResourcesImpl(problem, [&](auto handle) {
    for (Operation *top : ops)
      top->walk(handle);
  });
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYMODEL_H
