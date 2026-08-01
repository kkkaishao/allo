/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_DATAPATH_H
#define ALLO_MICROARCH_DATAPATH_H

#include "allo/IR/AlloAttrs.h"           // MemoryImplEnum
#include "allo/IR/AlloOps.h"             // dcp::DCPathModuleOp
#include "allo/Scheduling/MemoryModel.h" // MemoryLibrary + BankLayout
#include "allo/Scheduling/RegionGraph.h" // RegionShape

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h" // function_ref
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace circt::hw {
class HWModuleOp;
} // namespace circt::hw
namespace mlir::allo::iface {
struct ModuleInterface;
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

struct BindingPolicy;

/// The already-emitted callees a rerouted container's leaf datapath needs to
/// lower a `dcp.instance` to a CallUnit: the child `hw.module`s to
/// instantiate + their port models (callee arg <-> addr/data/we names, read vs
/// write direction). Null for a plain leaf (no calls). Both maps are populated
/// bottom-up by the emit driver, so a callee is present before its caller.
struct CalleeCtx {
  const llvm::StringMap<circt::hw::HWModuleOp> &modules;
  const llvm::StringMap<iface::ModuleInterface> &ifaces;
};

/// The width a runtime `index` value is carried at. An `index` has no width of
/// its own in MLIR, so the datapath picks one, and everything that carries an
/// index reads it here: `hwWidth`, the operands an address cone is evaluated
/// over (`evalAffine`), and the boundary address ports the manifest publishes.
///
/// A counter or an address register may be BUILT narrower than this wherever
/// its own value range allows (`RegionBlock::counterType`,
/// `RegionBlock::AddrStride::width`); this is the width such a value widens
/// back to the moment it is read as an ordinary index. Named rather than
/// spelled at each of those points because it is a default, and a default is
/// the kind of thing that becomes a device or schedule option.
inline constexpr unsigned kIndexWidth = 32;

//===----------------------------------------------------------------------===//
// Identifiers. Cells are referenced by small integer ids (indices into the
// Datapath's vectors) rather than pointers, so the whole model stays trivially
// copyable and diffable across a rebind.
//===----------------------------------------------------------------------===//

using UnitId = unsigned;
using RegId = unsigned;
using MemId = unsigned;
using MuxId = unsigned;
using IOId = unsigned;
using ConstId = unsigned;
using RegionId = unsigned;
using StreamId = unsigned;
using CallId = unsigned;

//===----------------------------------------------------------------------===//
// A resolved driver of one input port. Exactly one Source feeds each input, so
// muxes (when sharing forces a choice) appear as their own cells whose output
// is the Source. `outPort` is overloaded per kind:
//   Unit    -> result index (single-result units use 0)
//   Reg     -> tap level to read (0 = chain head, i.e. the newest sample)
//   Mem     -> index of the read access whose loaded data this is
//   Mux      -> 0
//   IO       -> 0
//   Const    -> 0
//   Counter  -> 0 (id = the RegionBlock whose iteration counter this is)
//   Survivor -> which result of the producing region (id = the RegionBlock),
//               latched when that region completes and read by a sibling region
//   Stream   -> index of the get access whose loaded token this is
//               (id = the StreamChannel)
//   Call     -> which scalar result of a sub-kernel call (id = the CallUnit):
//               the child instance's result output, landing at start+latency
//   Scope    -> 0 (id = the ScopeUnit): a func-scope combinational cone,
//               stable from its inputs' producing regions onward
//===----------------------------------------------------------------------===//

struct Source {
  enum class Kind {
    None,
    Unit,
    Reg,
    Mem,
    Mux,
    IO,
    Const,
    Counter,
    Survivor,
    Stream,
    Call,
    Scope
  };
  Kind kind = Kind::None;
  unsigned id = 0;
  unsigned outPort = 0;
  /// A resolvable (non-None) source.
  explicit operator bool() const { return kind != Kind::None; }
};

//===----------------------------------------------------------------------===//
// Structural cells.
//===----------------------------------------------------------------------===//

/// A functional-unit instance (adder, multiplier, floating-point core, ...).
/// In the trivial binding every compute op gets its own unit, so `boundOps`
/// holds a single entry and no input needs a mux.
struct FuncUnit {
  UnitId id = 0;
  std::string opType;   // the operator mnemonic (comb: "addi"; IP: module name)
  std::string impl;     // IP module name (empty when combinational)
  bool comb = false;    // combinational (a `comb` primitive), not an IP module
  unsigned latency = 0; // result available `latency` cycles after issue
  bool pipelined = true; // accepts a new input every cycle
  // The IP's port/back-pressure contract (from its `dcp.operator`); unused for
  // a combinational unit. Clock-enable is the only contract the emitter builds.
  StallContractEnum stall = StallContractEnum::Ce;
  Type resultType; // value-typed (e.g. f32), not bit-blasted

  // Ops bound here, each with its issue cycle (residue mod II in a cyclic
  // region). Sharing puts several non-conflicting ops in this list.
  llvm::SmallVector<std::pair<Operation *, unsigned>, 1> boundOps;

  /// The representative bound op: the one whose operand types, arity and
  /// op-specific attributes characterize the unit. Every other op bound here is
  /// reservation-compatible with it (`Reservation.h`), so the choice of
  /// `front()` is arbitrary, but it must be the SAME everywhere: naming, port
  /// shape and timing all read it. Use this rather than `boundOps.front()`.
  Operation *repOp() const {
    assert(!boundOps.empty() &&
           "a unit with no bound op has no representative");
    return boundOps.front().first;
  }

  // One resolved driver per input operand port (post-binding). A *fused*
  // recurrence (II == latency, depth II-L == 0) has a self-referential input
  // (`Source::Unit{this.id}`): the IP's own pipeline is the accumulator
  // register.
  llvm::SmallVector<Source, 2> inputs;

  // Per-input reduction identity (parallel to `inputs`). A recurrence input is
  // the port that reads a loop-carried iter_arg, and it carries that iter_arg's
  // init here, so the emitter re-injects the identity on the port at the first
  // iteration and a retriggered reduction restarts each outer pass. None for
  // every non-recurrence input. This is the sole init mechanism; a register in
  // the recurrence cycle is a plain delay. It lives on the input port rather
  // than a register because the widened idiom trunc(add(ext(acc),ext(x))) reads
  // acc through a bare wire, not a tap.
  llvm::SmallVector<Source, 2> inputInits;

  // Per-input recurrence distance in iterations (parallel to `inputInits`); the
  // emitter re-injects `inputInits[k]` for the first `inputInitDist[k]` runs. 1
  // for an ordinary distance-1 recurrence; >1 for a chained carry (a 2nd-order
  // shift register `ym2 = ym1; ym1 = y` gives ym2 distance 2, so its init must
  // hold for the first two iterations, not just the first).
  llvm::SmallVector<unsigned, 2> inputInitDist;
};

/// A combinational cell sitting at FUNC SCOPE, outside every region: one op of
/// the arith cone the reifier leaves in the module body when a top-level loop's
/// induction bound or a top-level guard's predicate is an expression rather
/// than a value the datapath already carries (`for i in range(start, m+1)` in a
/// callee, `if k == 0` before a `dcp.select`).
///
/// It is NOT a `FuncUnit`. A unit belongs to one region, issues on that
/// region's pulse and holds a reservation slot; this belongs to none, issues on
/// nothing, and must be readable from every region that comes after its inputs
/// settle.
///
/// SSA dominance closes its input set: a value defined at func scope can only
/// read a scalar kernel argument, a literal, a top-level region's survivor, or
/// an earlier cone. All four are HELD, so the cone is a pure function of
/// settled registers with no clock, no register chain and no stall shell. Its
/// one timing obligation is a composition edge, and `recordSiblingDeps` carries
/// it by chasing through the cone to the regions its inputs come from.
struct ScopeUnit {
  unsigned id = 0;
  Operation *op = nullptr;             // the func-scope arith op
  std::string opType;                  // its comb mnemonic (`combKindOf`)
  Type resultType;                     // value-typed, like FuncUnit::resultType
  llvm::SmallVector<Source, 2> inputs; // one resolved driver per operand
};

/// A shift-register chain carrying one SSA value across cycle boundaries. Its
/// length is the largest delay any consumer needs; consumers read at their own
/// `tap` (see Source).
struct Register {
  RegId id = 0;
  Value value; // the L0 value being held (for provenance / debug)
  Type type;
  unsigned depth = 0; // chain length in cycles (>= 1 for a real register)
  Source input;       // driver of the chain head (the producing cell output)
};

/// A memref-backed memory with banks and ports. The storage primitive (register
/// / LUTRAM / BRAM / URAM) is resolved by the memory model; this model records
/// it, but physical selection (address decode, per-primitive ports) is left to
/// lowering.
struct MemUnit {
  MemId id = 0;
  Value memref;
  bool external = false;   // a func-argument memref (bare interface, no AXI)
  unsigned width = 0;      // element width in bits
  unsigned depthWords = 0; // elements per bank
  // The memref's `allo.part` decomposition: which axes are partitioned, by
  // what factor and kind, and the resulting per-bank shape. Decoded ONCE here
  // rather than re-parsed by each consumer that needs it (the boundary port
  // set, the address crossbar, the per-bank depth, the manifest layout), since
  // it is a fixed property of the memref. An unpartitioned memref decodes to a
  // single bank whose `bankShape` is the full shape, so this is total: no
  // consumer needs an "is it banked" guard before reading it.
  BankLayout layout;
  unsigned numBanks = 1; // == layout.numBanks (1 = unbanked or registers)
  /// This memory's banks are SKEWED and its accesses carry slots rather than
  /// banks (`Access::staticBank`), so they are read through lane-shared ports
  /// instead of routed. `layout.skew()` states the layout fact; this states
  /// that the slots were actually assigned, which `assign-banks` declines for
  /// an argument array and for one whose accesses do not share a bank class.
  /// False on a skewed memory that fell back, which then crossbars like any
  /// other.
  bool skewed = false;
  /// This argument crosses the TOP boundary completely partitioned, so it
  /// arrives as one port per ELEMENT rather than as an addressed interface.
  /// The one expression is `external && dp.atTop && layout.registers`, applied
  /// once by the builder.
  ///
  /// A complete partition commits the scheduler to unlimited combinational
  /// ports (`MemoryBankModel` bills none), and at a boundary the only structure
  /// that delivers that is every element present at once: an addressed port
  /// serves one element per cycle, which is exactly the resource the schedule
  /// was solved without. Below the top the question does not arise, because
  /// whoever owns the storage (a `seq.hlmem`, or a scattered argument's own
  /// input ports) serves an ordinary addressed port from it.
  bool scattered = false;
  /// The module ports holding one element of a `scattered` argument: the input
  /// it arrives on, and the output + write-enable it leaves on. A direction the
  /// kernel does not use has no port, so exactly the unused ones are empty, and
  /// the DIRECTION is what decides the names (`A_k` when only one is live,
  /// `A_k_in` / `A_k_out` when both are, which is the rule Vitis follows).
  struct ElemPort {
    std::string in, out, we;
  };
  /// One per element, flat row-major, when `scattered`. Composed once by
  /// `enumerateBoundaryPorts`, which is where the whole boundary naming lives;
  /// empty for every other memory.
  llvm::SmallVector<ElemPort> elemPorts;
  MemoryImplEnum impl = MemoryImplEnum::LUTRAM; // resolved storage primitive

  // Access latency of `impl`, read from the device memory model. These are the
  // SAME numbers the scheduler stamped onto this memref's
  // `dcp.load`/`dcp.store` (asserted per access in `bindResource`).
  //
  // The emitter must build its read / write ports at exactly these latencies:
  // the consumer's register depth was solved as `tY - (start + readLatency)`,
  // so a port built at any other latency samples the wrong cycle. A
  // call-mastered buffer the parent never touches has no access to read them
  // off, which is why they live here rather than being re-derived per site.
  unsigned readLatency = 0;
  unsigned writeLatency = 1;

  // `romInit` is the `initial_value` (a DenseElementsAttr) of the
  // `memref.global` this memref reads through, when it has one.
  //
  // `isRom` is the narrower property the emitter can realize: initialized and
  // never written. Such an array is a constant table, emitted as a
  // combinational `hw.aggregate_constant` indexed by `hw.array_get` (registered
  // to the read latency) with no writable hlmem. Read-only is a property of the
  // use, not of carrying an initializer: a mutable global with a power-on value
  // (`allo.lang.Stateful`) has `romInit` but is not a ROM.
  bool isRom = false;
  Attribute romInit;

  /// One bound access. A read's loaded data is referenced by
  /// Source{Mem, id, <index of this access>}; a write consumes `data`.
  struct Access {
    Operation *op = nullptr;
    bool isWrite = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned port = 0;
    /// This access's slot in the module's boundary port list: an index into
    /// `Datapath::readPorts` or `writePorts` by `isWrite`, and thus its port
    /// identity at the boundary. `kNoPort` for an access to an internal
    /// memory, which takes no module port.
    static constexpr unsigned kNoPort = ~0u;
    unsigned portIdx = kNoPort;
    /// The boundary port group's base name (`A_rd0`), from which every field
    /// port is composed (`A_rd0_addr`, ...); a data-dependent banked access
    /// additionally suffixes a bank (`A_rd0_b2`, see `extPorts`). Composed once
    /// with `portIdx`, since a NAME is as much the port's identity as its slot
    /// is, and it is the C++/Python manifest contract. Empty for an internal
    /// memory's access, which takes no module port.
    std::string portBase;
    /// Which bank this access routes to, when its memref is partitioned
    /// (`numBanks > 1`): the index `assign-banks` assigned it, or empty when it
    /// assigned none, in which case the access crossbars over all `numBanks`
    /// banks (boundary interfaces for an argument, `seq.hlmem`s for an internal
    /// buffer). 0 for an unbanked memref, which is the one bank there is.
    /// `externalBank` pairs it with the memref's bank count.
    ///
    /// READ, not derived (`assignedBankOf` in `allocateInputSlots`), which
    /// writes every access including the unbanked ones. It is the same recorded
    /// decision the scheduler's port model was billed against before the solve,
    /// so an access charged one bank's port cannot end up taking one on every
    /// bank here. The default is the CONSERVATIVE end: an `Access` built on
    /// some future path that never reaches that write crossbars, which is
    /// wasteful, where defaulting to bank 0 would silently route it to the
    /// wrong storage.
    ///
    /// Under a SKEWED layout it holds a SLOT, not a bank: the physical bank is
    /// the slot rotated by a runtime value shared with the array's other
    /// accesses, so it is billable (distinct slots are distinct banks at every
    /// rotation) but not routable. `MemUnit::skewed` is the flag that says
    /// which of the two readings applies, and every consumer that ROUTES must
    /// check it. The port model, which only counts, need not.
    std::optional<unsigned> staticBank;
    /// Which of a skewed memory's parallel port sets this access uses. Accesses
    /// of one lane hold distinct slots, so they reach distinct banks and can
    /// SHARE one port on each: the lane is read once per bank and its accesses
    /// select among those reads. Two accesses of the same slot always collide,
    /// so they land in different lanes and get a port each, which is exactly
    /// what the port model billed them. Always 0 off the skewed path.
    unsigned lane = 0;
    AffineMap addrMap; // index map over `addr` operands (identity when the
                       // subscript was not affine)
    llvm::SmallVector<Source, 2> addr; // address operand drivers (delayed IVs)
    Source data;                       // write data driver (writes only)
    /// One strength-reduced term of the address: a scaled counter its region
    /// carries (`RegionBlock::addrStrides`).
    struct ScaledTerm {
      unsigned region;
      unsigned slot; // index into that RegionBlock's `addrStrides`
    };
    /// ADDRESS STRENGTH REDUCTION. One expression this access's address
    /// hardware computes: `base` plus one register per term (a scaled counter
    /// or a digit of one that the controller advances, instead of arithmetic
    /// the datapath rebuilds every cycle) plus `residual` evaluated over
    /// `addr`. `planAddressGenerators` decides them together (`splitAddress`)
    /// and `buildAddr` builds exactly them.
    ///
    /// PARTIAL: a term reduces or does not on its own, so a data-dependent
    /// subscript does not cost the reduction to the row stride beside it. With
    /// nothing reduced `terms` is empty and the residual holds the whole
    /// expression, which is the arithmetic an unreduced address builds.
    struct Reduced {
      llvm::SmallVector<ScaledTerm, 3> terms;
      /// The expression's constant, and ZERO whenever a term exists: a register
      /// loads a constant at start anyway, so the first one that does not wrap
      /// absorbs it (`AddrStride::init`) rather than an adder carrying it.
      int64_t base = 0;
      AffineExpr residual; // null when the whole expression reduced
      /// Registers the RESIDUAL reads (`SplitAddress::reads`), in the order it
      /// names them: a digit the address does not sum but an operator on top of
      /// it wants cheap. Appended to the operand list `buildAddr` evaluates the
      /// residual over, so they land on the symbol positions it named.
      llvm::SmallVector<ScaledTerm, 2> reads;
    };
    /// The element index within the bank, and the bank digit when one is
    /// decoded at run time. Two cones off the same operands
    /// (`addressExprsOf`), reduced by the one definition and built by the one
    /// builder: a bank digit is `(counter floordiv D) mod F`, which is a
    /// register as much as a row stride is.
    Reduced offset;
    Reduced bank;
    /// How many cycles late this access needs the SCALED COUNTERS, i.e. the
    /// delay its counter operands would otherwise be tapped at. They run live,
    /// so their sum is delayed once rather than each operand separately, which
    /// is equivalent and costs less register. The residual's operands arrive
    /// already delayed, so it is added after the chain and this does not apply
    /// to it.
    unsigned addrDelay = 0;
  };
  llvm::SmallVector<Access, 2> accesses;
};

/// One bound access, referenced as (owning cell id, access index): a memory
/// access is `dp.mems[id].accesses[idx]`, a stream access
/// `dp.streams[id].accesses[idx]`. Used both for the module's boundary port
/// lists (`Datapath::readPorts` / `writePorts`) and for a region's own accesses
/// (`RegionBlock::memAccesses` / `streamAccesses`).
struct AccRef {
  unsigned id, idx;
};

/// A sub-kernel call as a multi-cycle datapath node. Built from a
/// `dcp.instance` and owned by the `RegionBlock` it sits in (a `dcp.sequential`
/// wrapping the call). The child instance *masters* the memory ports of its
/// memref operands (it drives their addr/data/we; the parent's `MemUnit`
/// supplies the storage), so a shared internal buffer becomes a
/// `seq.read`/`seq.write` the child addresses. Its scalar result lands at
/// `start + latency` as a survivor.
struct CallUnit {
  CallId id = 0;
  Operation *invoke = nullptr; // the dcp::DCPathInstanceOp
  RegionId region = 0;         // the RegionBlock (a dcp.sequential) it sits in
  std::string callee;          // callee symbol (key into CalleeCtx maps)
  // The invoke's `latency`: its start->done depth. Empty means the callee
  // publishes no whole-kernel latency at all, so it completes on its own
  // `done`.
  std::optional<int64_t> latency;
  unsigned start = 0; // region-relative issue cycle (the invoke `start`)
  /// Whether the child completes at a statically known cycle, so a consumer may
  /// be released by a static offset instead of its real `done`. This is the
  /// invoke's DECLARED `determinacy`, and it is deliberately not
  /// `latency.has_value()`: a dynamic-trip callee publishes a latency *bound*
  /// and is still indeterminate, so the two disagree exactly there.
  bool determinate = false;
  /// An `await` SPAWN rather than a scheduled call: it starts with its
  /// container and is ordered thereafter only by FIFO back-pressure, so it has
  /// no offset to be placed at and offers a consumer nothing to be
  /// time-triggered off. The `Concurrent` start policy at node granularity.
  bool async = false;

  /// One memory *port* the child drives for a mastered memref operand. A callee
  /// arg accessed at several points exposes several ports (a read-twice arg:
  /// two read ports; a read-modify-write accumulator: a read AND a write port),
  /// so there is one MemArg per child port, not per operand.
  struct MemArg {
    unsigned calleeArg;         // operand position == callee argument index
    MemId mem;                  // caller MemUnit backing this array
    bool isBoundary;            // a func BlockArgument vs an internal alloc
    bool isWrite;               // this port writes (vs reads)
    unsigned bank = 0;          // cyclic bank this port serves (0 unbanked)
    unsigned factor = 1;        // partition factor (1 unbanked)
    std::string addr, data, we; // child port names; `we` empty for a read
    std::string topBase; // top boundary port base (indexed); empty = internal
  };
  llvm::SmallVector<MemArg, 2> memArgs;

  /// A scalar operand the child consumes: its driver (an IO port, a sibling
  /// survivor, an enclosing counter, a same-region unit, or a constant, all
  /// resolved by `recordCallScalars`) plus the child port it feeds.
  struct ScalarArg {
    Source src;
    std::string port; // child scalar-input port name
    /// The port's width, so the wiring adapts the driver to the CHILD rather
    /// than the child's width propagating back into whatever produces it. It
    /// matters for one driver, an enclosing counter: an index has no width of
    /// its own, so caller and callee each pick one and only this says they
    /// agree.
    unsigned width = 0;
  };
  llvm::SmallVector<ScalarArg, 1> scalarIns;

  /// A stream (FIFO) operand: the child is one END of a channel, handshaking on
  /// three ports of its own. A channel crossing a call boundary is a
  /// back-pressured hand-off rather than a timed one, which is why the leaf
  /// datapath rejects a stream-operand call.
  struct StreamArg {
    unsigned calleeArg;             // operand position == callee argument index
    StreamId chan;                  // the channel this port binds
    bool isInput;                   // the child READS the channel
    unsigned depth = 2;             // the child's requested buffering
    std::string base;               // the child's port group
    std::string data, valid, ready; // its three port names
  };
  llvm::SmallVector<StreamArg, 1> streamArgs;

  /// The child result-output port per scalar result. The
  /// result's datapath Source is Source::Call{id, k} (registered in
  /// producerOf), captured into this region's survivor exactly like a compute
  /// result: a sibling reads it as Source::Survivor{region, k}.
  llvm::SmallVector<std::string, 1> resultPorts;

  /// An earlier sibling call this one must start after, and why. Composition
  /// predecessors at CALL granularity, the exact analogue of
  /// `RegionBlock::predecessors` on the instance substrate. Derived once by
  /// `DatapathBuilder::recordCallDeps`, by a rule that depends on how the
  /// owning region composes: a scheduled composition orders its children by
  /// their placed `start`, while a concurrent one has no schedule to order them
  /// by and must read the hazard directions.
  struct Pred {
    CallId call;
    /// The edge is a scalar RESULT hand-off, not a shared array. Such an edge
    /// can never be time-triggered: the producer's result port only holds from
    /// its `done`, so an exact-cycle release is not a safe contract even for a
    /// determinate producer.
    bool viaResult = false;
  };
  llvm::SmallVector<Pred, 2> predecessors;
};

/// A FIFO channel: a `!allo.stream` value, handshaked (valid/ready) rather than
/// addressed. A channel is either an *input* (the kernel reads it via
/// `allo.stream.get`) or an *output* (writes it via `allo.stream.put`); its
/// payload type and depth come from the stream type. A get's loaded token is
/// referenced by Source{Stream, id, <index of the get access>}; a put consumes
/// `data`. A channel carries exactly one access (single-producer /
/// single-consumer).
struct StreamChannel {
  StreamId id = 0;
  Value stream;         // the !allo.stream SSA value (a func block arg)
  Type payload;         // element type carried through the FIFO
  unsigned depth = 2;   // FIFO depth (from the stream type)
  bool isInput = false; // input (get) vs output (put)
  // A channel this kernel OWNS: defined by an `allo.stream.create` in its own
  // body rather than passed in, so both ends sit inside this module. It gets no
  // boundary port, since a `seq.fifo` in the module body carries it, and it is
  // the one channel that may be both read and written (a loop-carried delay
  // line), which leaves `isInput` meaningless for it.
  bool internal = false;
  /// Initial tokens (a `stream.create` initializer): the channel's history is
  /// `[init] ++ [produced]`, which is what breaks a feedback cycle's start
  /// dependence. Every process on an unseeded cycle would otherwise block
  /// reading an empty channel. Realized as a consumer-side prepend shim, not as
  /// tokens pushed into the FIFO. Null for an unseeded channel.
  Attribute init;

  /// A channel end that is a CHILD PORT rather than one of this module's own
  /// `get`/`put` accesses: `(call, index into that CallUnit's streamArgs)`. A
  /// container composes processes, so its channels are wired end-to-end between
  /// `hw.instance`s and it issues no access of its own; a leaf's channels have
  /// accesses and no call ends. A channel may have SEVERAL consumer ends, with
  /// the fan-out realized as one FIFO per reader pushed in lock-step, but only
  /// one producer end (`validateDatapath`: a merge has no deterministic token
  /// interleaving).
  struct CallEnd {
    CallId call;
    unsigned arg; // index into `dp.calls[call].streamArgs`
  };
  llvm::SmallVector<CallEnd, 2> callEnds;

  struct Access {
    Operation *op = nullptr; // the stream.get / stream.put op
    bool isPut = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned stage = 0;  // scheduled cycle within the region (dcpStart)
    Source data;         // put: the token's data driver (puts only)
    // A predicated access (an i1 `pred` operand from a masked `if`) consumes or
    // produces its token only where this holds. Delayed to `stage` like `data`;
    // None for an unconditional access.
    Source when;
  };
  llvm::SmallVector<Access, 1> accesses;
};

/// A multiplexer inserted where sharing makes several sources contend for one
/// sink input. Empty in the trivial binding; one per shared-unit input port
/// that sees different drivers across the ops bound to it.
struct Mux {
  MuxId id = 0;
  llvm::SmallVector<Source, 2> sources;
  // The op whose issue selects each source (parallel to `sources`): the source
  // is driven onto the shared unit's input on the cycle that op consumes it, so
  // the select is `delayValid(issue, dcpStart(op))`, the same per-op activation
  // pulse a store's write-enable uses. The MRT guarantees these are
  // mutually exclusive (disjoint residues), so the derived mux is a plain
  // priority chain.
  llvm::SmallVector<Operation *, 2> selectOps;
  RegionId region = 0; // region whose issue pulse times the selects
};

/// A top-level scalar INPUT port (a scalar kernel argument). Memref arguments
/// become external `MemUnit`s instead, and a scalar function result is a
/// `Result` (driven by a survivor), so every IOPort is an input by
/// construction.
struct IOPort {
  IOId id = 0;
  Value value;
  Type type;
};

/// A literal tied into the datapath.
struct ConstCell {
  ConstId id = 0;
  Attribute value;
  Type type;
};

/// A scalar function result, exposed as an output port driven by `source` (a
/// returning region's survivor, a passthrough scalar input, or a constant) and
/// valid when the function's `done` rises. An array (memref) result becomes a
/// trailing out-param (the buffer-results-to-out-params prepass) before emit,
/// so only scalars reach here.
struct Result {
  Source source;
  Type type;
  std::string name;
};

//===----------------------------------------------------------------------===//
// Regions. One RegionBlock per dcp region op (dcp.pipeline / dcp.sequential).
// Cyclic blocks are pipelined loops (constant trip, II-paced); acyclic blocks
// are straight-line. Blocks run in program order with no overlap, so a single
// sequential hand-off chains them.
//===----------------------------------------------------------------------===//

/// How a region produces one of its results. This is the ONE shape covering
/// every regime, so a consumer reads the same three fields whichever controller
/// runs.
/// A region result is always a *survivor register*: the value is latched when
/// it lands and held for whoever reads it (a sibling region, an enclosing
/// container's next iteration, the function's output port), which is why a
/// counted loop's final accumulator, a while's frozen recurrence and a guard's
/// muxed branch value are one concept and not three.
///
///   counted loop / while | `value` = the loop-carried next (the terminator's
///                        |   `dcp.uncondition` / `dcp.condition` operand),
///                        |   `init` = the matching `inits` operand. The two
///                        |   regimes differ only in the pulse the capture keys
///                        |   off, which is the controller's business, not the
///                        |   model's.
///   sequential           | `value` = the yielded value; no recurrence, so
///                        |   `init` is None (it lands exactly once).
///   guard (dcp.select)   | `value` = the THEN arm's yield, `elseValue` = the
///                        |   ELSE arm's; the survivor is `cond ? then : else`.
///
/// A `None` `value` is an untracked result: no survivor is built, and a
/// consumer that reads it fails at its own slot. A `None` `init` means the
/// result is not a loop-carried recurrence. Its survivor then powers on at zero
/// instead of being preloaded, which is only safe because such a result always
/// lands.
struct RegionResult {
  Source value;
  Source init;
  Source elseValue;
};

struct RegionBlock {
  RegionId id = 0;

  /// The `dcp.pipeline` / `dcp.sequential` / `dcp.select` this block models.
  /// Kept so a diagnostic about the region anchors on the loop the user wrote
  /// rather than on the enclosing function.
  Operation *op = nullptr;

  /// STRUCTURAL SHAPE, axis 1 of the controller discriminant. Which controller
  /// lowers a region is a function of (shape x termination class), and this is
  /// the axis the model must store. The termination axis is declared in the IR
  /// (`determinacy`), but the shape is a property of the region's *structure*,
  /// so it is derived once here instead of being recomputed from
  /// `children.empty()` / `guard` / `callUnits.empty()` at every consumer,
  /// including the validator, which would otherwise have to reproduce the
  /// emitter's own dispatch to know which timing rule applies.
  ///
  /// The populated cells:
  ///
  ///                | CountedStatic          | Conditional
  ///   -------------+------------------------+---------------------------
  ///   Leaf         | free-running / modulo  | flushing while
  ///   Container    | counted outer + child  | CHECK/RUN outer
  ///                | sequencer              |
  ///   Guard        | branch-pulse, run-once | (same: run-once either way)
  ///   CallNode     | fire + child `done`    | n/a
  ///
  /// Every other cell is a shape the frontend cannot produce; `emitRegion`
  /// rejects rather than falling through, so a newly reachable one is a
  /// deliberate extension and not an emergent special case.
  ///
  /// The four cells are spelled once in `RegionShape`, so the reifier (which
  /// charges each shape's boundary cost) and the emitter (which picks its
  /// controller) cannot come to different answers.
  using Shape = allo::RegionShape;
  /// Read off the region op by `dcpRegionShape` in
  /// `DatapathBuilder::deriveShapes`, which re-asks it of the BUILT model
  /// (parent/child edges linked, CallUnits bound) and asserts the two agree.
  Shape shape = Shape::Leaf;

  enum class Kind { Cyclic, Acyclic } kind = Kind::Acyclic;
  std::optional<unsigned> ii; // set iff Cyclic

  /// Whether at most ONE pass of this region is in flight. A cyclic region
  /// overlaps its iterations at `ii` by construction; every other family runs
  /// a pass to its `done` before the next is issued. Named here rather than
  /// re-derived at each consumer, because what reads it (`RegionTag`, so that
  /// `delayValid` may time a long delay with a counter rather than one
  /// flip-flop per cycle) is relying on the overlap rule, not on the kind.
  bool singlePass() const { return kind == Kind::Acyclic; }

  // Counted-loop induction: the IV runs `lb, lb+step, ...` up to (excluding)
  // `ub`. Each bound is an ordinary datapath `Source`, either a data-dependent
  // range start / count / stride or a literal `ConstCell` synthesized by
  // `recordRegionBounds`, so a bound reads exactly like every other operand in
  // the model and needs no "constant or Source" decode at its consumers.
  // Set for a Cyclic region, None for an Acyclic one (no counter).
  //
  // `ubSource` is the one exception, and `tripCount` is why: a constant trip
  // over a RUNTIME lb or step (the `for j in range(i, i+K)` window) has
  // `ub = lb + K*step`, DERIVED arithmetic rather than a value the datapath
  // already produces, so no cell can carry it. That case alone leaves
  // `ubSource` None and `terminatorOf` builds the expression; every other
  // counted region resolves its ub straight from the Source.
  std::optional<int64_t> tripCount; // constant trip iff Cyclic
  Source lbSource;                  // lower bound (counter init)
  Source ubSource;                  // upper bound; see `tripCount` above
  Source stepSource;                // step (counter increment)
  // The width the iteration counter is BUILT at, and therefore the width its
  // bounds are resolved to. It is i32 for every region whose counter stays
  // inside the module, and the callee's index-port width for a `CallNode`,
  // whose counter drives that port directly. Stored so the one place that
  // assembles the bounds (`terminatorOf`) can adapt there instead of a
  // controller rebuilding the whole `Terminator` after the fact. Null for an
  // Acyclic region. Derived by `DatapathBuilder::deriveCounterTypes`.
  Type counterType;
  // TERMINATION class as the emitter discriminates it, axis 2 of the pair
  // above. A while loop (a `dcp.condition` terminator) is a flushing pipeline
  // whose exit is data-dependent. The declared `determinacy` below is the same
  // axis read off the IR. The two agree in the direction that matters (a while
  // is always declared `Conditional`, asserted in `deriveShapes`) but NOT in
  // the converse: the reifier also stamps a `dcp.select` `Conditional`, while
  // `conditional` stays false for it since a guard is not a flushing loop.
  bool conditional = false;
  // The two raw structural flags `shape` is derived from, recorded by the
  // builder as it walks. Consumers should read `shape`.
  bool guard = false;      // this region op is a dcp.select
  bool container = false;  // nests another dcp region in EITHER arm, so a
                           // guard with children is `container` too and this
                           // is not the same as `shape == Container`
  std::string counterName; // source loop IV name (its NameLoc), for a readable
                           // iteration-counter wire; empty if the loop's IV
                           // carried no name (best-effort)
  /// A REGISTER this region carries beside its own counter, holding
  /// `coeff * digit` of it for a coefficient and a digit an access's address
  /// needs, tracked incrementally rather than rebuilt.
  ///
  /// This is address strength reduction. The constant multiply is the
  /// arithmetic that dominates an address (a row stride of 400 is a
  /// three-digit signed-digit network; the sum that follows it is one adder),
  /// and it is the part an induction variable makes unnecessary: consecutive
  /// iterations differ by a constant.
  ///
  /// A DIGIT of the counter rides the same register with two more constants.
  /// `(x floordiv D) mod K` advances by nothing on most iterations and by one
  /// on the ones where `x` crosses a multiple of `D`, so it is maintained by a
  /// carry from a companion register holding `x mod D` (itself a stride with
  /// `wrap = D`), and it wraps at `K` by subtracting once. A `floordiv` or a
  /// `mod` on the ADDRESS path pays every cycle, where this pays a comparator
  /// off it.
  ///
  /// So one update rule covers both:
  ///
  ///     raw  = cur + step + (carry fired ? bump : 0)
  ///     next = wrap && raw >= wrap ? raw - wrap : raw
  ///
  /// with a plain scaled counter at `bump = wrap = 0`. `step + bump <= wrap`
  /// holds by construction (`asDigit` refuses a step that could wrap twice), so
  /// the single subtract is exact.
  ///
  /// A DECREASING digit (`A[N-1-i]`) mirrors it: `step` and `bump` go negative
  /// and the wrap ADDS on borrow rather than subtracting on overflow. The
  /// borrow is `raw > cur` unsigned, since subtracting a positive amount can
  /// only raise the value by wrapping around zero.
  struct AddrStride {
    int64_t init;       // `coeff * lb`, the value the register loads at start
    int64_t step;       // `coeff * step`, added wherever the counter advances
    int64_t bump = 0;   // added when `carry`'s register wraps
    int64_t wrap = 0;   // subtracted on reaching it (0: a plain accumulator)
    unsigned carry = 0; // slot whose wrap gates `bump`; self means none
    bool hasCarry = false; // whether `carry` names one
    bool down = false;     // counts down, so `wrap` is added on borrow
    /// The width the register is BUILT at. Every field above is a compile-time
    /// constant, so the range it runs over is one too, and this is that range
    /// rounded up to bits: a wrapping digit needs `clog2` of its modulus and a
    /// row stride `clog2` of the array it walks, neither of which has anything
    /// to do with the counter's own width, which these borrowed before.
    /// `kIndexWidth` whenever the range is not bounded (`slotFor`).
    unsigned width = kIndexWidth;
  };
  /// Deduplicated, since two accesses down the same row share a stride. Some
  /// slots exist only to carry another (the `x mod D` companion of a quotient
  /// digit) and no access names them; a carry always precedes its consumer, so
  /// one pass emits them. Empty when no address follows this counter, or when
  /// its bounds are not constant, which is what makes the fields compile-time
  /// values at all.
  llvm::SmallVector<AddrStride> addrStrides;

  // Composition class, DERIVED by `dcpRegionTiming` in `addRegion`. The region
  // op carries it as an attribute too, but only as a report stamped from that
  // same function, so this reads the region and not the report.
  //
  // `deriveShapes` asserts the one cross-axis invariant, that `conditional`
  // implies `determinacy == conditional`, tying the emitter's termination
  // discriminant to the derived class.
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;

  // The TERMINAL cycle the latency model was composed off (`drain` on the
  // region op), against which `HWEmitter::emitRegion` checks the `drainStage`
  // it independently derives from the built datapath.
  //
  // The one place a model of the hardware meets the hardware; every other check
  // in the compiler compares one model against another. A leaf's `done` rises
  // `drainStage + 1` cycles after its last issue, so a divergence here is a
  // consumer placed at an offset the hardware does not honour.
  std::optional<int64_t> modelledDrain;

  // Composition predecessors: the earlier top-level sibling regions this one
  // must start after. Only top-level regions populate it, since container
  // children stay serial. A region depends on an earlier sibling iff they touch
  // a shared memref (a data hazard or a read-port conflict) or a cross-region
  // SSA edge (a scalar survivor). Functional units are auto-disjoint under
  // per-region binding, so shared *memory* is the only cross-region resource.
  // A region with no predecessors starts concurrently with the kernel; one with
  // predecessors starts on their joined `done`. Producers precede consumers in
  // program order, so the relation is a DAG. Set by `recordSiblingDeps`.
  llvm::SmallVector<RegionId, 2> predecessors;

  // Region nesting. A container region drives its `children` in its body; each
  // child's `parent` is the enclosing container. Top-level regions (no parent)
  // are the func-scope siblings chained by the sequencer; a container runs its
  // child `tripCount` times (hierarchical control, II_outer >= L_inner).
  std::optional<RegionId> parent;
  llvm::SmallVector<RegionId, 2> children;
  // A guard (dcp.select) with a non-empty `else` branch is a *dual* guard: its
  // `children` are the then-branch sub-schedule (run iff the predicate holds)
  // and `elseChildren` are the else-branch sub-schedule (run iff it does not).
  // Empty for a container loop and for a then-only guard; the two child lists
  // are the two mutually-exclusive arms.
  llvm::SmallVector<RegionId, 2> elseChildren;

  // Cells owned by this region (ids are Datapath-global; these record
  // membership and thus which counter drives them).
  llvm::SmallVector<UnitId, 4> units;
  llvm::SmallVector<RegId, 4> regs;
  llvm::SmallVector<MuxId, 2> muxes;
  llvm::SmallVector<CallId, 1> callUnits; // sub-kernel calls
  // The accesses this region ISSUES, driven by its controller and timed against
  // its schedule. Memories and streams are owned Datapath-wide, since a buffer
  // written in one region and read in another is one storage cell; membership
  // is a property of each ACCESS, and this is where it is recorded. Both lists
  // are in body program order (the order `bindResource` walks).
  llvm::SmallVector<AccRef, 2> memAccesses;
  llvm::SmallVector<AccRef, 1> streamAccesses;

  // The Sources this region's results come from, indexed by result number (see
  // RegionResult). Empty for a result-less region. For a loop this is also its
  // loop-carried recurrence, with `results[k]` being iter-arg k, since a
  // counted loop's k-th result IS the final value of its k-th iter-arg. The
  // emitter reads the same vector whether it captures once at the end (a leaf)
  // or advances a frozen register per outer iteration (a container).
  llvm::SmallVector<RegionResult, 1> results;

  // This region's control predicate, as a resolved i1 Source: a while's
  // per-iteration continue condition, or a guard's (dcp.select) run-once
  // predicate. None for a counted region, which terminates on its counter.
  //
  // A while's condition is a scheduled compute producer (cmpi/cmpf, a
  // Source::Unit): solved in-body for a leaf while, reified to a start-0
  // compute for a sequential wrapper. A guard's is that same combinational unit
  // over the enclosing counter (an affine guard `i > j`) or a scheduled
  // prologue region's survivor (a data-dependent `flag[j] > 0`). Either way it
  // is *held* for the run it gates: a guard start-gates its children by it, so
  // the not-taken arm's stores never fire structurally, with no per-store gate.
  Source condition;
};

//===----------------------------------------------------------------------===//
// The whole microarchitecture of one function.
//===----------------------------------------------------------------------===//

struct Datapath {
  dcp::DCPathModuleOp func;
  /// Whether this function is the TOP of the emitted design, i.e. whether its
  /// arguments name storage nobody in the design owns. A callee's array
  /// argument is a port it masters on its caller's storage, which is why the
  /// two answer `MemUnit::scattered` differently.
  bool atTop = false;

  // Derived structural cells.
  std::vector<FuncUnit> units;
  std::vector<ScopeUnit> scopeUnits; // func-scope cones, in block order
  std::vector<Register> regs;
  std::vector<MemUnit> mems;
  std::vector<StreamChannel> streams;
  std::vector<Mux> muxes;
  std::vector<IOPort> ios;
  std::vector<ConstCell> consts;
  std::vector<Result> results;      // scalar func results, in return order
  std::vector<CallUnit> calls;      // sub-kernel calls
  std::vector<RegionBlock> regions; // program order

  // The module's boundary memory ports: every access to an EXTERNAL (func
  // argument) memref, split by role and ordered by (memref, access). An
  // internal memref is on-chip `seq.hlmem` storage and takes no port, so it
  // appears in neither list. This is the ONE enumeration. The index of an
  // access here IS its port identity, mirrored back onto the access as
  // `MemUnit::Access::portIdx`, and read by the port declaration, the naming
  // layer, the manifest and the emitter alike.
  llvm::SmallVector<AccRef> readPorts, writePorts;

  // L1 binding decisions the policy writes; the structure above is derived from
  // these plus the schedule. (Memory port binding lives in MemUnit::accesses,
  // co-located with its memref.)
  llvm::DenseMap<Operation *, UnitId> opToUnit;

  /// Set when the builder hit a schedule it cannot realize and ALREADY emitted
  /// a diagnostic, namely a consumer placed before its producer's result is
  /// ready (`resolveOperand`). The build finishes with placeholder values so it
  /// stays bounded; `validateDatapath` turns this into a failure before any
  /// hardware is emitted.
  bool infeasible = false;

  Datapath() = default;
  /// \p memLib is the device's storage-timing view (the `dcp.device` `memory:`
  /// table the scheduler timed every access against); it resolves each
  /// MemUnit's implementation and access latency.
  Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
           const MemoryLibrary &memLib, const CalleeCtx *callees = nullptr,
           bool isTop = false);

  /// The dcp op whose execution produces \p s's value, or null when the Source
  /// has no producing op: a literal, the iteration counter, a kernel input
  /// port, a held value (Reg / Survivor) or a derived mux. The ONE definition
  /// of the Source -> op mapping, which `Source::outPort`'s per-kind overload
  /// would otherwise force every consumer to re-spell.
  Operation *producingOp(const Source &s) const;

  /// \p s's compile-time value, when it is an integer literal cell; empty for
  /// every Source whose value is only known at run time. Lets a consumer fold
  /// against a bound (an induction step, a recurrence distance) without the
  /// model carrying a second, constant-only spelling of it.
  std::optional<int64_t> constantOf(const Source &s) const;

  void dump(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// The model visitor. `Source`s are scattered across ~20 slots of the model, so
// any question of the form "for every driver in this datapath, ..." would
// otherwise be a hand-rolled sweep that has to be extended for each new slot.
// `forEachSource` is the one traversal; a new `Source` field is covered by
// adding it here, once.
//===----------------------------------------------------------------------===//

/// One `Source` slot in the model: what it drives, and whether being
/// unresolved (`Source::Kind::None`) is a defect there.
struct SourceSite {
  enum class Slot {
    UnitInput,        // a compute unit's operand port
    UnitInit,         // that port's reduction identity (absent => None)
    ScopeInput,       // a func-scope cone's operand
    RegisterInput,    // a shift chain's head driver
    MuxInput,         // one arm of a derived sharing mux
    MemAddress,       // an address operand of a memory access
    MemWriteData,     // a store's data (a load leaves it None)
    StreamData,       // a put's token data (a get leaves it None)
    StreamPredicate,  // a masked access's `pred` (absent => None)
    CallScalarIn,     // a scalar operand handed to a sub-kernel
    FuncResult,       // a scalar function result's driver
    RegionBound,      // a runtime lb / ub / step (compile-time => None)
    RegionResult,     // a region's yielded value / carried next (untracked
                      // => None)
    RegionResultInit, // that result's loop-carried identity (absent => None)
    RegionElseResult, // a dual guard's else-arm yield
    RegionCondition,  // a while's continue condition / a guard's predicate
                      // (a counted region has none => None)
  };
  Slot slot;
  /// Which port / operand / result of the owner this is.
  unsigned index = 0;
  /// The dcp op this slot belongs to, for a located diagnostic. Null for a slot
  /// owned by a region or by the function rather than by one op.
  Operation *op = nullptr;
  /// Whether an unresolved Source here is a DEFECT. False for the slots where
  /// `None` is the legitimate encoding of "absent" (see the comments above).
  bool required = true;

  /// A noun phrase naming this slot, for a diagnostic ("operand 1 of a compute
  /// unit"). Built only on failure.
  std::string describe() const;
};

/// Visit every `Source` slot in \p dp exactly once, in model order.
void forEachSource(
    const Datapath &dp,
    llvm::function_ref<void(const Source &, const SourceSite &)> fn);

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR. `readyCycleOf` is the single
// authority for "the cycle a producing op's result lands, relative to its
// issuing pulse": both the builder (register-depth derivation) and the emitter
// (result capture) consult it, so the latency model has one definition.
//===----------------------------------------------------------------------===//

/// Region-relative schedule cycle of a dcp compute/load/store op (its `start`).
unsigned dcpStart(Operation *op);
/// Result latency of a producing dcp op (0 if uncharacterized): a load's own
/// `latency`, or an IP compute's `latency` (stamped at emit from its operator).
unsigned dcpLatency(Operation *op);
/// The cycle a producing op's result is ready: `dcpStart + dcpLatency` (a
/// stream get is a combinational front-read, latency 0). Zero for an at-issue
/// value with no producing op (a constant, the iteration counter).
unsigned readyCycleOf(Operation *op);

/// The datapath's hardware width for a value type: `index` -> `kIndexWidth`, a
/// float carried as its bit pattern, an integer verbatim. This is the ONE width
/// rule. The model is value-typed but the emitted carrier is a bit vector, and
/// the emitter (`uarch::hwType`) and the boundary port model
/// (`iface::bitWidth`) must not disagree about how wide it is.
unsigned hwWidth(Type t);

/// The banking of an *external* (argument) memory access, so the boundary
/// presents one interface per bank. `factor == 1` is an unbanked memory
/// (`bank == 0`); a banked access is either statically routed (`bank` set) or
/// data-dependent (`bank` empty -> a crossbar over all `factor` interfaces).
///
/// Both halves are stored on the model (`MemUnit::numBanks` +
/// `Access::staticBank`), resolved once by the builder; this pairs them under
/// the name the consumers ask the question by. It lives here, not in an emitter
/// header, because the boundary/naming layer (`Naming.h`,
/// `iface::ModuleInterface`) needs it and must depend on L2 only.
struct ExternalBanking {
  unsigned factor = 1;          // physical banks (1 = unbanked)
  std::optional<unsigned> bank; // static bank, or empty = data-dependent
};
inline ExternalBanking externalBank(const MemUnit &m,
                                    const MemUnit::Access &acc) {
  return {m.numBanks, acc.staticBank};
}
// (`extPorts` in Naming.h names the resulting per-bank interfaces.)

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATH_H
