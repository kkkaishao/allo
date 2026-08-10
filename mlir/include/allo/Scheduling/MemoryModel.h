/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MEMORYMODEL_H
#define ALLO_SCHEDULING_MEMORYMODEL_H

#include "allo/IR/AlloAttrs.h"         // MemoryPortEnum
#include "allo/Scheduling/Scheduler.h" // OccupancyProblem

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
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
/// (`RegionBlock::counterType`, `RegionBlock::AddrStride::width`). Both the
/// emitter and the scheduler price against it.
inline constexpr unsigned kIndexWidth = 32;

/// The bits a value of type \p t occupies in the datapath: `index` at
/// `kIndexWidth`, since it carries no width of its own; a float as its bit
/// pattern; an integer verbatim.
///
/// The single width rule: the model is value-typed while the emitted carrier is
/// a bit vector, so the operator pricing (`combParamWidth`), the emitter
/// (`uarch::datapathType`) and the boundary port model (`iface`) read one
/// answer.
unsigned datapathWidth(mlir::Type t);

//===----------------------------------------------------------------------===//
// Memory timing library: the `memory:` section of the device file. Read/write
// latency and delay per storage implementation, plus one FIFO (stream) timing.
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

/// Timing of one storage realization (or of the stream FIFO): latency and
/// delay, each split by direction.
struct MemKindTiming {
  RWLatency latency;
  RWDelay delay;
};

/// How many instances of its storage row one array may be held in. POLICY and
/// not hardware: a copy costs the row's area again and buys one instance's
/// reads, so where that line sits is the compiler's choice. One number here
/// until there is a reason to take it as an option.
constexpr unsigned kStorageCopies = 2;

/// The ports of ONE instance of a storage realization, per bank. Nullopt is no
/// limit on that axis. What an ARRAY may be given in a cycle is none of the
/// three but derived from them, which is what the scheduler reserves against
/// and the datapath binds against.
///
/// A block RAM instance's two ports each read or write, so two writers and a
/// concurrent reader take three of them and the pool is what says so. A row
/// whose directions are independent structures declares no pool, as a LUT RAM's
/// single write port against its one addressed read does.
struct StoragePorts {
  std::optional<unsigned> instReads;
  std::optional<unsigned> instWrites;
  std::optional<unsigned> instPool;
  /// Whether the counts above describe the WHOLE array rather than one instance
  /// of a structure the compiler may copy. An `allo.bind.storage type=`
  /// topology names the ports the array is to have and a stream has two ends,
  /// and neither leaves room for a copy the compiler adds on top.
  bool stated = false;

  /// Instances an array here may be spread over IN A CYCLE, which is what the
  /// scheduler is allowed to issue against. Not a bound on the copies BUILT:
  /// the port binding colours by what may share an address bus, so a schedule
  /// issuing two reads can still need three buses and each of those is a copy.
  unsigned copies() const { return stated ? 1 : kStorageCopies; }

  /// The tighter of the two budgets on every axis. A nullopt is no limit and
  /// yields to whatever the other side declares.
  StoragePorts meet(const StoragePorts &other) const;

  /// Whether this row can hold an array built with \p writes write ports over
  /// \p ports address buses, given AS MANY COPIES AS IT TAKES. The second is
  /// not the first plus the reads: where a port serves either direction, one
  /// bus may carry a read and a write that never issue together.
  ///
  /// Reads are not an argument because they never disqualify a row: a further
  /// read is a further copy, which is what the part does and what the emitter
  /// builds. A write does: every copy needs every write, so one instance's
  /// write ports are the ceiling however many copies there are. On a pooled row
  /// the writes also spend a port of every copy, so writes filling the pool
  /// leave a read nowhere to go.
  ///
  /// Where the ports are `stated` there is no copy to add and the whole array
  /// has to fit one instance.
  bool holds(unsigned writes, unsigned ports) const;

  /// Whether it can serve the topology \p want names, which is the same
  /// question asked of ports a directive requested rather than ones a binding
  /// built.
  bool holds(const StoragePorts &want) const;

  /// ONE INSTANCE's ports as one phrase for a diagnostic ("2 read / 1 write
  /// over 2 shared ports"), an unlimited axis spelled as such. What copies of
  /// it an array may be given is the sentence around this, not part of it.
  std::string describe() const;
};

/// One `dcp.storage` row: a structure the device can hold an array in, named by
/// the device's own vocabulary rather than by a case of a closed enum.
struct StorageRealization {
  std::string name;
  MemKindTiming timing;
  StoragePorts ports;
  /// The vendor attribute that pins an array to this structure, stamped on the
  /// emitted declaration. Empty where the part has no such attribute, and the
  /// synthesizer then chooses.
  std::string ramStyle;
  /// Whether the structure comes up holding contents. False for one that powers
  /// up undefined, as an UltraRAM does.
  bool canInit = true;
  /// Whether this is the row that is not a memory: one cell per element, no
  /// address, which is where a complete partition goes.
  bool scatter = false;
  /// What ONE instance spends over `(depth, width)`, the `uses` of the row
  /// verbatim. Held as the attribute rather than a number because the price is
  /// only meaningful at an array's own shape. Null where the device left the
  /// row unpriced.
  mlir::ArrayAttr uses;
};

/// The storage-timing library, filled from the `dcp.storage` and
/// `dcp.stream_timing` rows of the device.
class MemoryLibrary {
public:
  /// Build the library from a module's injected `dcp.device`: its
  /// `dcp.storage` rows and its `dcp.stream_timing`. A module with no
  /// `dcp.device` yields an empty (all-default) library.
  static MemoryLibrary fromModule(ModuleOp module);

  struct Timing {
    unsigned latency = 0;
    double delay = 0.0;
    // The accessed array's resolved storage realization, EMPTY for an access
    // with no storage axis: a stream is a FIFO timed by its own row. Accesses
    // of different realizations must map to *different* operator types, or they
    // collapse onto one latency, so this keys the type.
    std::string storage;
  };
  /// Timing for a memory/stream access op; zero latency and delay if \p op is
  /// not one. An array access is timed by its memref's storage realization.
  Timing timing(Operation *op) const;

  /// The device's row for the storage realization \p name, or null where it
  /// declares none. Everything the device states about a structure is read
  /// from here, so one lookup answers timing, ports and vendor attribute
  /// together.
  const StorageRealization *row(llvm::StringRef name) const;

  /// The timing of storage realization \p name. The device is required to
  /// declare every realization an array resolves to, which `PreVerification`
  /// enforces; an undeclared one asserts here and falls to a zero
  /// (combinational) timing.
  MemKindTiming timing(llvm::StringRef name) const;

  /// Whether \p storage is the row the device marked `scatter`: one cell per
  /// element, no address, no port limit. False for every row when the device
  /// marks none, which is what makes a complete partition unrealizable there
  /// rather than silently addressed.
  bool isScatter(llvm::StringRef storage) const {
    return !scatterStorage.empty() && storage == scatterStorage;
  }

  /// What one bank of \p words x \p width of \p storage spends, as a fraction
  /// of the part: the worst of its resources, which is the axis a design runs
  /// out on. Nullopt where the row is unpriced, where a cost is not measured at
  /// this shape, or where the device quotes no capacity for what it spends.
  /// ONE instance, the copies being a decision no one has taken this early.
  std::optional<double> fractionOfPart(llvm::StringRef storage, int64_t words,
                                       unsigned width) const;

  /// The row an unbound array of \p words x \p width takes: among the rows the
  /// device can PIN an array to, the cheapest by `fractionOfPart` of those at
  /// the least access latency. Latency first and without exception, since a
  /// row's latency is the contract the schedule is built on and trading a cycle
  /// for area is the user's call to make with `bind_storage`. \p needsInit
  /// excludes a row that powers up undefined. Empty where the device declares
  /// nothing it can both pin and price, and `defaultStorage` then stands.
  std::string rowFor(int64_t words, unsigned width, bool needsInit) const;

  // The `dcp.storage` marked `default`, EMPTY where the device marks none. A
  // device that marks one holds every unbound array there and the derivation
  // never runs; a device that marks none leaves the choice to `rowFor`. A name
  // rather than a handle, so replacing a row does not leave it dangling.
  std::string defaultStorage;
  // What a completely partitioned array resolves to: the `dcp.storage` marked
  // `scatter`, EMPTY when the device marks none. The compiler names no storage
  // of its own, so this is the one place the two axes meet.
  std::string scatterStorage;
  std::vector<StorageRealization> storage; // the `dcp.storage` rows
  MemKindTiming fifo;                      // `dcp.stream_timing`
  /// How much of each `dcp.resource` the part has, which is what turns a row's
  /// spend into a fraction and so makes two rows spending different primitives
  /// comparable at all.
  llvm::StringMap<int64_t> capacity;
};

//===----------------------------------------------------------------------===//
// Per-memref storage predicates, read off the array's `allo.part` /
// `allo.bind.storage` attributes.
//===----------------------------------------------------------------------===//

/// The `memref.global` initializer behind \p memRef, i.e. a constant table's
/// declared contents, or nullopt when it has none.
std::optional<Attribute> globalInitOf(Value memRef);

/// Whether \p memRef is a CONSTANT TABLE: it has a `memref.global` initializer
/// and nothing writes it. Read-only is a property of the USE, not the
/// declaration: an initialized array stored to even once is a real memory that
/// merely starts with contents.
///
/// A constant table lowers to `hw.aggregate_constant` read by one
/// `hw.array_get` per access: combinational, no handshake, genuinely
/// UNLIMITED-port. Narrower than `MemoryChar::readOnly`: an explicit
/// `bind.storage type="rom_1p"` is a real memory whose ports the user chose.
///
/// Handing the array to a SUB-KERNEL also disqualifies it: a child MASTERS
/// PORTS, driving addr/data/we into storage the parent owns, and a constant
/// table has none to master.
bool isConstantTable(Value memRef);

/// The `allo.bind.storage impl=` written on \p memref: what was ASKED for,
/// before `characterize` resolves it, and empty when nothing was. A complete
/// partition overrides an explicit choice, so this is what makes the two
/// directives comparable and their disagreement reportable.
llvm::StringRef boundStorageOf(Value memref);

/// The realization `recordArrayStorage` resolved \p memref to (`kStorageAttr`),
/// which is what it was ASKED for only where the user bound it. A lookup, so
/// two carriers of one array agree because there is one record, not because two
/// derivations were written the same way.
std::string resolvedStorageOf(Value memref);

/// The two orthogonal axes of one `allo.bind.storage` directive, mapped from
/// its `type` string (which port topology) and its `impl` string (which storage
/// realization). The RAM/ROM half of a `type` spelling is not an axis: read-only
/// is a property of the USE, which `isConstantTable` decides.
struct BindStorage {
  /// The topology asked for, empty where the directive names none. Absent is
  /// not the dual-port default: an array that asked for nothing takes whatever
  /// its realization has, and only an explicit topology narrows that.
  std::optional<MemoryPortEnum> port;
  llvm::StringRef storage; // empty: no explicit choice, not "no storage"
};

/// The axes \p bind states, all defaulted for a null dictionary.
BindStorage parseBindStorage(mlir::DictionaryAttr bind);

/// Whether topology \p a serves everything \p b asks for. The three form a
/// chain, `1p` under `s2p` under `t2p`, so two carriers of one array reconcile
/// by taking the one that covers the other.
bool topologyCovers(MemoryPortEnum a, MemoryPortEnum b);

//===----------------------------------------------------------------------===//
// Partition and static-bank queries. A DCP banking pass reuses these facts so
// it materializes the *same* banks the scheduler bound its ResII against.
//===----------------------------------------------------------------------===//

/// The bank decomposition of a partitioned memref, in ELEMENT space: which
/// bank holds element `(i_0 .. i_{r-1})`, and where inside that bank it sits.
/// The single definition of "which bank", shared by the port model, the static
/// split, the emitter's crossbar and the host-side layout.
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
/// A skew buys CONFLICT FREEDOM, not a compile-time bank, where block and
/// cyclic (both functions of ONE subscript) serve an array read only one way:
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

  /// The single skewed axis, or null. At most one is allowed: the slot analysis
  /// reasons about ONE rotation of the bank index.
  const Axis *skew() const;
};

/// `kind` as the interface manifest spells it, the name the host reproduces the
/// decomposition from.
llvm::StringRef bankKindName(BankLayout::Kind kind);

/// Decode a memref's `allo.part` attribute into its element-space bank
/// decomposition (a single unpartitioned bank when there is no attribute). THE
/// decoder of that attribute: a consumer wanting only the bank count or the
/// complete-partition flag reads it off here rather than parsing again.
BankLayout bankLayoutOf(Value memRef);

/// One array's storage shape: how it banks, what ports one bank has, and which
/// `dcp.storage` realization it resolves to. THE characterization, billed by
/// the scheduler's port model (`MemoryBankModel`) and built by the microarch
/// datapath (`MemUnit`), so what a schedule reserves and what the emitter wires
/// cannot drift apart.
struct MemoryChar {
  BankLayout layout; // element-space banks (one when unpartitioned)
  /// Ports ONE INSTANCE of the row holding one bank has: the resolved
  /// `storage` row's, narrowed by the `allo.bind.storage type=` topology. One
  /// budget for the scheduler and the emitter both.
  StoragePorts ports;
  bool constantTable = false; // realized as a combinational constant array
  /// The `dcp.storage` realization recorded for this array (`kStorageAttr`),
  /// read rather than re-resolved. EMPTY only for the one array that has
  /// nowhere to go, a complete partition on a device marking no `scatter` row,
  /// which `PreVerification` reports against the array.
  std::string storage;

  /// Whether there is no port here to contend for: a constant table is
  /// combinational, and a complete partition scattered the array into
  /// registers.
  bool unlimited() const { return layout.registers || constantTable; }
};

/// Where an array carries the `dcp.storage` realization it RESOLVED to. On the
/// array's carrier, so `dcp-resolve-banking` copies it onto every bank alloc
/// and a per-bank array answers what the whole one did.
///
/// Empty names the one array with nowhere to go, a complete partition on a
/// device marking no `scatter` row, which `PreVerification` reports.
constexpr llvm::StringLiteral kStorageAttr = "allo.storage";

/// Resolve every array of \p module to a `dcp.storage` realization and record
/// it under `kStorageAttr`. Runs ONCE, before any layer asks what an array was
/// realized as: the resolution is a cost model over the device, and re-running
/// it per consumer is how two layers come to disagree about one array.
void recordArrayStorage(ModuleOp module, const MemoryLibrary &lib);

/// Characterize a memref's storage shape from its partition attributes and the
/// realization `recordArrayStorage` resolved for it, independent of any
/// scheduling region. \p lib supplies what the device states about that
/// realization, and has to be the same device the access latencies were stamped
/// from, or the two disagree.
MemoryChar characterize(Value memref, const MemoryLibrary &lib);

/// The ports the `allo.bind.storage type=` topology on \p memref asks for, or
/// nullopt where it names none. A constraint rather than a budget: the array's
/// realization decides what it has and this only narrows it. `PreVerification`
/// reports a topology the row cannot meet.
std::optional<StoragePorts> requestedPortsOf(Value memref);

/// The canonical spelling of \p part for a memref of \p type: a COMPLETE
/// partition collapses to its one whole-array axis, a `dim == 0` block or
/// cyclic axis expands into one axis per dimension, and the axes are sorted by
/// dimension. Null canonicalizes to null.
///
/// `bankLayoutOf` folds the axes IN ORDER into a mixed-radix bank index, so a
/// spelling is part of the bank index function, not presentation: two
/// attributes describing the same banking must be spelled identically before a
/// caller and callee agree on one (a sub-kernel masters port group `k` of
/// exactly the caller's bank `k`).
PartitionAttr canonicalizePartition(PartitionAttr part, MemRefType type);

/// The coarsest banking of a memref of \p type that satisfies both \p a and
/// \p b, canonical; failure with \p why set when the two cannot be reconciled.
///
/// The order is REFINEMENT: `a` is below `b` when every pair of elements `a`
/// places in distinct banks `b` does too. A partition directive is a LOWER
/// BOUND on the bank-distinctness its kernel needs, so a kernel scheduled
/// against the join still sees every access group it asked to be
/// conflict-free. A complete partition is the top and an absent attribute the
/// bottom (one bank).
///
/// Axes on DIFFERENT dimensions compose in mixed radix with no reconciling
/// needed. On ONE dimension the join must remain a SINGLE axis (`allo.part`
/// admits no duplicate dimension), so it exists only when one factor divides
/// the other (and, for a block axis, the finer chunk boundaries fall on the
/// coarser ones). A block axis against a cyclic axis has no common single-axis
/// refinement at all.
llvm::FailureOr<PartitionAttr> joinPartitions(PartitionAttr a, PartitionAttr b,
                                              MemRefType type,
                                              std::string &why);

/// An access's bank index and in-bank offset, as affine EXPRESSIONS over the
/// address map's operands: each partitioned axis contributes its mixed-radix
/// digit to `bank`, and what remains of the subscripts re-linearizes over the
/// per-bank shape into `offset`. \p map is in ELEMENT SPACE, one result per
/// memref dimension; linearizing happens at the point of use, never in the IR.
///
/// Deriving this on the EXPRESSION rather than on emitted values is what makes
/// common banked idioms free: `A[2*i]` under cyclic-2 has bank `(2*i) mod 2`
/// and offset `(2*i) floordiv 2`, which fold to `0` and `i` (no hardware),
/// where the same derivation on emitted values leaves a multiply/mask/shift
/// nothing downstream can fold away.
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
/// of \p shape, or nullopt when the bank varies at runtime.
///
/// This is `bankSplitOf(...).bank` when that expression is ONE VALUE, so the
/// bank a consumer routes to and the bank the port model bills cannot drift
/// apart. A cyclic digit is one value when every variable coefficient of its
/// subscript vanishes modulo the factor.
///
/// \p ranges bounds the dims (the map's own numbering), which a block digit
/// needs: `A[i]` under block-2 of an `i32[16]` is `i floordiv 8`, which folds
/// for no `i` but is CONSTANT over every `i` a loop on `[0,8)` produces, so the
/// standard idiom (a loop per block) resolves nothing without it. An empty
/// \p ranges asks the folding question alone.
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
/// `kBankAttr` and the port model bills a port on it, so F accesses with F
/// distinct slots take one port per bank. The emitter must NOT route to it
/// directly: the physical bank is the slot rotated by `cls`, known only at run
/// time. `BankLayout::skew()` tells the two readings of `kBankAttr` apart.
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
/// never ran. Reads whichever carrier the IR layer uses, so every consumer sees
/// one recorded decision. Nullopt is the conservative answer everywhere (bill,
/// route and address through all the banks).
std::optional<unsigned> assignedBankOf(Operation *op);

/// \p map, in element space, rewritten as the single row-major linear element
/// index it addresses, simplified. Applied AT THE POINT OF USE by everything
/// needing a flat address, so pricing, strength reduction and the emitter
/// cannot disagree.
///
/// Nothing rewrites the IR with it, deliberately: element space carries
/// per-dimension structure the linear form cannot be simplified back into
/// (`(6i+j) floordiv 6` does not fold to `i` without knowing `j < 6`), and the
/// bank split needs that structure.
///
/// Working on the EXPRESSION is what cancels the delinearize/linearize pair of
/// a coalesced nest: `iv -> (iv floordiv N, iv mod N)` composed with
/// `(r, c) -> r*N + c` simplifies back to `iv`, where the same round trip built
/// out of `comb` ops is a divider, a modulo and a multiplier.
AffineMap linearizeAccessMap(AffineMap map, llvm::ArrayRef<int64_t> shape);

} // namespace mlir::allo

namespace mlir::allo {

/// Per-bank memory-port model. `observe` every memory access in a scheduling
/// region, `finalize` to `characterize` the arrays behind them, then
/// `resources` gives the port resources one access holds. Each `allo.part` bank
/// is a separate limited resource carrying the array's ports.
///
/// An access holds one port on EVERY bank it can reach: the bank `assign-banks`
/// assigned it, or all of them when assigned none. The latter is not a
/// conservative bound but the crossbar the emitter builds, so a partitioned
/// array under a roaming access sustains one bank's ports, not that times the
/// bank count.
class MemoryBankModel {
public:
  void observe(Operation *op);
  void finalize(const MemoryLibrary &lib);

  /// What one access holds: the port resources, as {resource key, slots per
  /// bank}, one entry per bank it reaches, and how many of those slots it takes
  /// on each. The limit repeats because it is a property of the bank, not of
  /// the access.
  struct PortDemand {
    llvm::SmallVector<std::pair<std::string, unsigned>> units;
    /// A read takes one slot, a write one of every copy the array is spread
    /// over: the copies all hold the same array and a write reaches all of
    /// them.
    unsigned slots = 1;
  };
  /// The ports \p op holds at once. Empty when \p op is not a memory access, or
  /// when its storage has no port to contend for (a constant table, a complete
  /// partition).
  PortDemand resources(Operation *op) const;

private:
  llvm::DenseMap<Value, MemoryChar> byMemref;
};

} // namespace mlir::allo

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Memory resource model: applies the per-memref port/bank model to a scheduling
// problem, the storage twin of `populateOperatorTypes`.
//===----------------------------------------------------------------------===//

/// Assign per-memref memory-port resources to every memory access \p problem
/// holds. A port is a one-cycle reservation whatever its latency
/// (`getResourceCycles`'s default), so no occupancy window is set.
///
/// Two passes over the same operations: the bank model has to see every access
/// of an array before it can say what one of them holds.
///
/// \p lib is what `characterize` resolves an array's storage row against. The
/// ports billed here do not depend on that row, but drawing them from the SAME
/// characterization the emitter builds from is what keeps the two in step.
template <class ProblemT>
void populateMemoryResources(ProblemT &problem, const MemoryLibrary &lib) {
  using namespace circt::scheduling;
  MemoryBankModel banks;
  for (Operation *op : problem.getOperations())
    banks.observe(op);
  banks.finalize(lib);
  for (Operation *op : problem.getOperations()) {
    MemoryBankModel::PortDemand held = banks.resources(op);
    SmallVector<Problem::ResourceType> units;
    for (auto &[key, limit] : held.units) {
      assert(held.slots <= limit &&
             "an access takes more slots than its own budget has, which no "
             "cycle can hold and the greedy placement would search forever "
             "for; every limit is one instance's ports once per copy and a "
             "write takes one of each, so it cannot outgrow the budget");
      Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
      problem.setLimit(rsrc, limit);
      units.push_back(rsrc);
    }
    if (units.empty()) // non-memory, or storage with no port to contend for
      continue;
    problem.setLinkedResourceTypes(op, units);
    problem.setResourceDemand(op, held.slots);
  }
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYMODEL_H
