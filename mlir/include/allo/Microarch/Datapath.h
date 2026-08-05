/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_DATAPATH_H
#define ALLO_MICROARCH_DATAPATH_H

#include "allo/IR/AlloAttrs.h"                // MemoryPortEnum
#include "allo/IR/AlloOps.h"                  // dcp::DCPathModuleOp
#include "allo/Scheduling/MemoryModel.h"      // BankLayout
#include "allo/Scheduling/OperatorIdentity.h" // what one unit realizes
#include "allo/Scheduling/RegionGraph.h"      // RegionShape

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

/// The already-emitted callees a `dcp.instance` lowers against: the child
/// `hw.module`s to instantiate plus their port models. Null for a plain leaf.
/// Both maps are filled bottom-up by the emit driver, so a callee is present
/// before its caller.
struct CalleeCtx {
  const llvm::StringMap<circt::hw::HWModuleOp> &modules;
  const llvm::StringMap<iface::ModuleInterface> &ifaces;
};

//===----------------------------------------------------------------------===//
// Identifiers. Cells are referenced by ids indexing the Datapath's vectors
// rather than by pointers, so the model stays trivially copyable across a
// rebind.
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
// a mux forced by sharing is its own cell whose output is the Source.
//
// A Source names a WIRE. A shared unit's output carries a different bound op's
// result in each issue cycle, so a consumer asking WHEN its value is ready has
// to say which one it means. `outPort` is that index, overloaded per kind:
//   Unit    -> which bound op's result this is (index into `boundOps`; 0 under
//              the trivial allocation, where a unit has exactly one)
//   Reg     -> tap level to read (0 = chain head, i.e. the newest sample)
//   Mem     -> index of the read access whose loaded data this is
//   Mux      -> 0
//   IO       -> 0
//   Const    -> 0
//   Counter  -> 0 (id = the RegionBlock whose iteration counter this is)
//   Survivor -> which result of the producing region (id = the RegionBlock),
//               latched when that region completes
//   Stream   -> index of the get access whose loaded token this is
//               (id = the StreamChannel)
//   Call     -> which scalar result of a sub-kernel call (id = the CallUnit),
//               landing at start+latency
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
    Call
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
  // What this unit realizes, agreed on by every op bound here: two units may
  // be folded only if their identities are equal. Carries the realization the
  // emitter builds, the result type, and the fields the RTL module name is
  // spelled from.
  OperatorIdentity identity;
  unsigned latency = 0;  // result available `latency` cycles after issue
  bool pipelined = true; // accepts a new input every cycle
  // The IP's port/back-pressure contract (from its `dcp.operator`); unused for
  // a combinational unit. Clock-enable is the only contract the emitter builds.
  StallContractEnum stall = StallContractEnum::Ce;

  // Ops bound here, each with its issue cycle (residue mod II in a cyclic
  // region). Sharing puts several non-conflicting ops in this list. NEVER
  // empty: a unit exists because ops are bound to it.
  llvm::SmallVector<std::pair<Operation *, unsigned>, 1> boundOps;

  /// The representative bound op: the one whose operands shape the unit's
  /// input ports and whose location names it. The choice of `front()` is
  /// arbitrary but must be the same everywhere, so use this rather than
  /// `boundOps.front()`.
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

  // Per-input reduction identity (parallel to `inputs`). A recurrence input
  // carries the init of the loop-carried iter_arg it reads, which the emitter
  // re-injects on the port at the first iteration; None for every
  // non-recurrence input. This is the sole init mechanism, and it lives on the
  // input port rather than a register because the widened idiom
  // trunc(add(ext(acc),ext(x))) reads acc through a bare wire, not a tap.
  llvm::SmallVector<Source, 2> inputInits;

  // Per-input recurrence distance in iterations (parallel to `inputInits`); the
  // emitter re-injects `inputInits[k]` for the first `inputInitDist[k]` runs. 1
  // for an ordinary distance-1 recurrence; >1 for a chained carry (a 2nd-order
  // shift register `ym2 = ym1; ym1 = y` gives ym2 distance 2).
  llvm::SmallVector<unsigned, 2> inputInitDist;
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
  /// The cycle within the producing iteration at which `input` carries a fresh
  /// datum (`readyCycleOf` the producer, 0 for a held source): the other half
  /// of the depth formula, and the phase an II-folded chain captures on. Not
  /// re-derivable from `input`, whose shared unit names a representative op.
  unsigned ready = 0;
};

/// A memref-backed memory with banks and ports. Which `dcp.storage` the device
/// realizes it in is resolved by the memory model; this model records the name,
/// but physical selection (address decode, per-primitive ports) is left to
/// lowering.
struct MemUnit {
  MemId id = 0;
  Value memref;
  bool external = false;   // a func-argument memref (bare interface, no AXI)
  unsigned width = 0;      // element width in bits
  unsigned depthWords = 0; // elements per bank
  // The memref's `allo.part` decomposition: which axes are partitioned, by what
  // factor and kind, and the resulting per-bank shape. Decoded ONCE here rather
  // than re-parsed by each consumer. An unpartitioned memref decodes to a
  // single bank whose `bankShape` is the full shape, so this is total: no
  // consumer needs an "is it banked" guard before reading it.
  BankLayout layout;
  unsigned numBanks = 1; // == layout.numBanks (1 = unbanked or registers)
  /// This memory's banks are SKEWED and its accesses carry slots rather than
  /// banks (`Access::staticBank`), so they are read through lane-shared ports
  /// instead of routed. False on a skewed layout whose slots `assign-banks`
  /// declined to assign, which then crossbars like any other.
  bool skewed = false;
  /// This argument crosses the TOP boundary completely partitioned, arriving
  /// as one port per ELEMENT rather than an addressed interface
  /// (`external && dp.atTop && layout.registers`). A complete partition commits
  /// the scheduler to unlimited combinational ports, which only an
  /// element-per-port boundary delivers (an addressed port serves one element
  /// per cycle). Below the top the owning storage serves an ordinary addressed
  /// port.
  bool scattered = false;
  /// This memory's boundary WRITE port groups never collide: two of them may
  /// be enabled in one cycle, but only where the scheduler proved they address
  /// different words, so a consumer may place each in its own `always` block
  /// and infer a true dual port. False leaves them a priority chain, which is
  /// what a group per static store already was.
  ///
  /// It is exactly "`writePortColouring` accepted this array", since that is
  /// the condition the colouring refuses on, and the groups ARE its colours.
  bool writesIndependent = false;
  /// The module ports holding one element of a `scattered` argument: the input
  /// it arrives on, and the output + write-enable it leaves on. A direction the
  /// kernel does not use has no port, and the DIRECTION decides the names
  /// (`A_k` when only one is live, `A_k_in` / `A_k_out` when both are, which is
  /// the rule Vitis follows).
  struct ElemPort {
    std::string in, out, we;
  };
  /// One per element, flat row-major, when `scattered`; composed by
  /// `enumerateBoundaryPorts`. Empty for every other memory.
  llvm::SmallVector<ElemPort> elemPorts;
  std::string storage; // resolved `dcp.storage` realization

  // Access latency of `storage`, the same numbers the scheduler stamped onto
  // this memref's `dcp.load`/`dcp.store` (asserted per access in
  // `bindResource`). The consumer's register depth was solved as `tY - (start +
  // readLatency)`, so ports must be built at exactly these latencies.
  unsigned readLatency = 0;
  unsigned writeLatency = 1;

  // `romInit` is the `initial_value` (a DenseElementsAttr) of the
  // `memref.global` this memref reads through, when it has one. `isRom` is
  // the narrower, emitter-realizable property: initialized and never written,
  // so it becomes a combinational `hw.aggregate_constant` table with no
  // writable hlmem. Read-only is a property of the USE: a mutable global with
  // a power-on value (`allo.lang.Stateful`) has `romInit` but is not a ROM.
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
    /// additionally suffixes a bank (`A_rd0_b2`, see `extPorts`). It is as much
    /// the port's identity as `portIdx` is, and it is the C++/Python manifest
    /// contract. Empty for an internal memory's access.
    std::string portBase;
    /// Which bank this access routes to, when its memref is partitioned
    /// (`numBanks > 1`): the index `assign-banks` assigned it, or empty to
    /// crossbar over all `numBanks` banks. 0 for an unbanked memref.
    /// `externalBank` pairs it with the memref's bank count. Defaults to
    /// crossbar rather than bank 0, so an access on some unreached future path
    /// is merely wasteful instead of silently routed to the wrong storage.
    ///
    /// Under a SKEWED layout this holds a SLOT, not a bank: the physical bank
    /// is the slot rotated by a runtime value shared with the array's other
    /// accesses, so it is billable but not routable without first checking
    /// `MemUnit::skewed`.
    std::optional<unsigned> staticBank;
    /// Which of a skewed memory's parallel port sets this access uses.
    /// Accesses in one lane hold distinct slots, so they reach distinct banks
    /// and share one port per bank; two accesses of the same slot collide and
    /// must land in different lanes. Always 0 off the skewed path.
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
    /// ADDRESS STRENGTH REDUCTION. This access's address is `base` plus one
    /// register per term (a scaled counter, or a digit of one, that the
    /// controller advances instead of rebuilding arithmetic every cycle) plus
    /// `residual` evaluated over `addr`. PARTIAL: a term reduces or not on its
    /// own, so with nothing reduced `terms` is empty and `residual` holds the
    /// whole expression.
    struct Reduced {
      llvm::SmallVector<ScaledTerm, 3> terms;
      /// The expression's constant, and ZERO whenever a term exists: a register
      /// loads a constant at start anyway, so the first one that does not wrap
      /// absorbs it (`AddrStride::init`) rather than an adder carrying it.
      int64_t base = 0;
      AffineExpr residual; // null when the whole expression reduced
      /// Registers the RESIDUAL reads (`SplitAddress::reads`), in the order it
      /// names them. Appended to the operand list `buildAddr` evaluates the
      /// residual over, so they land on the symbol positions it named.
      llvm::SmallVector<ScaledTerm, 2> reads;
    };
    /// The element index within the bank, and the bank digit when one is
    /// decoded at run time: two cones off the same operands (`addressExprsOf`).
    /// A bank digit is `(counter floordiv D) mod F`, which is a register as
    /// much as a row stride is.
    Reduced offset;
    Reduced bank;
    /// How many cycles late this access needs the SCALED COUNTERS, i.e. the
    /// delay its counter operands would otherwise be tapped at. They run live,
    /// so their sum is delayed once rather than each operand separately, which
    /// is equivalent and costs less register. The residual's operands arrive
    /// already delayed, so this does not apply to it.
    unsigned addrDelay = 0;
  };
  llvm::SmallVector<Access, 2> accesses;
};

/// One bound access, referenced as (owning cell id, access index): a memory
/// access is `dp.mems[id].accesses[idx]`, a stream access
/// `dp.streams[id].accesses[idx]`.
struct AccRef {
  unsigned id, idx;
};

/// A sub-kernel call as a multi-cycle datapath node, built from a
/// `dcp.instance` and owned by the `RegionBlock` it sits in. The child instance
/// *masters* the memory ports of its memref operands (it drives their
/// addr/data/we; the parent's `MemUnit` supplies the storage), so a shared
/// internal buffer becomes a `seq.read`/`seq.write` the child addresses. Its
/// scalar result lands at `start + latency` as a survivor.
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
  /// invoke's DECLARED `determinacy`, deliberately not `latency.has_value()`:
  /// a dynamic-trip callee publishes a latency *bound* and is still
  /// indeterminate.
  bool determinate = false;
  /// An `await` SPAWN rather than a scheduled call: it starts with its
  /// container and is ordered thereafter only by FIFO back-pressure, so it has
  /// no offset to be placed at and offers a consumer nothing to be
  /// time-triggered off.
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
    /// The child says its write ports on this argument never collide, so the
    /// array backing them may give each its own `always` block
    /// (`MemUnit::writesIndependent`, `iface::Memory::independent`).
    bool independent = false;
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

  /// The child result-output port per scalar result. The result's datapath
  /// Source is Source::Call{id, k}, captured into this region's survivor
  /// exactly like a compute result: a sibling reads it as
  /// Source::Survivor{region, k}.
  llvm::SmallVector<std::string, 1> resultPorts;

  /// An earlier sibling call this one must start after, and why: composition
  /// predecessors at CALL granularity. Derived by `recordCallDeps` by a rule
  /// that depends on how the owning region composes, since a scheduled
  /// composition orders its children by their placed `start` while a concurrent
  /// one has no schedule to order them by and must read the hazard directions.
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
  /// dependence, since every process on an unseeded cycle would block reading
  /// an empty channel. Realized as a consumer-side prepend shim, not as tokens
  /// pushed into the FIFO. Null for an unseeded channel.
  Attribute init;

  /// A channel end that is a CHILD PORT rather than one of this module's own
  /// `get`/`put` accesses: `(call, index into that CallUnit's streamArgs)`. A
  /// container wires its channels end-to-end between `hw.instance`s and issues
  /// no access of its own; a leaf's channels have accesses and no call ends. A
  /// channel may have SEVERAL consumer ends, with the fan-out realized as one
  /// FIFO per reader pushed in lock-step, but only one producer end
  /// (`validateDatapath`: a merge has no deterministic token interleaving).
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
  // The op whose issue selects each source (parallel to `sources`): the select
  // is `delayValid(issue, dcpStart(op))`, the same per-op activation pulse a
  // store's write-enable uses. The MRT guarantees these are mutually exclusive
  // (disjoint residues), so the emitter builds a one-hot select.
  llvm::SmallVector<Operation *, 2> selectOps;
  RegionId region = 0; // region whose issue pulse times the selects
};

/// The combinational depth, in LUT levels, of the select a mux of \p sources
/// sources costs: `ceil(log2 k)`, since the emitter builds a one-hot AND-OR
/// reduction (`EmitContext::oneHotSelect`) and each level halves the term
/// count. Zero for a single source, which is a wire.
unsigned muxLevels(unsigned sources);

/// What one such level costs in ns: the device's OR row, since the select is
/// an AND-OR reduction rather than a chain of 2:1 selects.
double muxLevelDelay(const OperatorLibrary &lib);

/// The sub-cycle room \p u's bound ops have left, in ns: the smallest
/// `cycleTime - z(op) - inDelay(u)` over them, where `z` is the sub-cycle start
/// the scheduler solved and `inDelay` the row it priced the unit against. This
/// bounds the combinational delay binding may add in front of the unit. Never
/// negative on a schedule the chaining model accepted.
double unitSlack(const FuncUnit &u, float cycleTime,
                 const OperatorLibrary &lib);

/// A top-level scalar INPUT port (a scalar kernel argument). Memref arguments
/// become external `MemUnit`s instead and a scalar function result is a
/// `Result`, so every IOPort is an input by construction.
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
/// trailing out-param before emit, so only scalars reach here.
struct Result {
  Source source;
  Type type;
  std::string name;
};

//===----------------------------------------------------------------------===//
// Regions. One RegionBlock per dcp region op. Cyclic blocks are II-paced
// pipelined loops; acyclic blocks are straight-line, and blocks run in program
// order with no overlap.
//===----------------------------------------------------------------------===//

/// How a region produces one of its results. One shape covers every regime, so
/// a consumer reads the same three fields whichever controller runs. A region
/// result is always a *survivor register*: the value is latched when it lands
/// and held for whoever reads it (a sibling region, an enclosing container's
/// next iteration, the function's output port).
///
///   counted loop / while | `value` = the loop-carried next (the terminator's
///                        |   `dcp.uncondition` / `dcp.condition` operand),
///                        |   `init` = the matching `inits` operand. The two
///                        |   regimes differ only in the pulse the capture keys
///                        |   off.
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

  /// STRUCTURAL SHAPE, axis 1 of the controller discriminant (shape x
  /// termination class picks the controller).
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
  /// Every other cell is unreachable; `emitRegion` rejects rather than falling
  /// through. Spelled once in `RegionShape`, so the reifier (boundary cost)
  /// and the emitter (controller choice) cannot disagree.
  using Shape = allo::RegionShape;
  /// Read off the region op by `dcpRegionShape` in
  /// `DatapathBuilder::deriveShapes`, which re-asks it of the BUILT model
  /// (parent/child edges linked, CallUnits bound) and asserts the two agree.
  Shape shape = Shape::Leaf;

  enum class Kind { Cyclic, Acyclic } kind = Kind::Acyclic;
  std::optional<unsigned> ii; // set iff Cyclic

  /// Whether at most ONE pass of this region is in flight. A cyclic region
  /// overlaps its iterations at `ii` by construction; every other family runs
  /// a pass to its `done` before the next is issued. What reads it
  /// (`RegionTag`, so that `delayValid` may time a long delay with a counter
  /// rather than one flip-flop per cycle) relies on the overlap rule, not on
  /// the kind.
  bool singlePass() const { return kind == Kind::Acyclic; }

  // Counted-loop induction: the IV runs `lb, lb+step, ...` up to (excluding)
  // `ub`. Each bound is a datapath `Source`, either a data-dependent value or a
  // literal `ConstCell` synthesized by `recordRegionBounds`. Set for a Cyclic
  // region, None for an Acyclic one (no counter).
  //
  // `ubSource` is the one exception: a constant trip over a RUNTIME lb or step
  // (the `for j in range(i, i+K)` window) has `ub = lb + K*step`, DERIVED
  // arithmetic no cell can carry, so `ubSource` is None there and
  // `terminatorOf` builds the expression instead.
  std::optional<int64_t> tripCount; // constant trip iff Cyclic
  /// An UPPER BOUND on the trip of a loop that has no constant one, from the
  /// `allo.assume.ssa` range the scheduler distilled (`dcp.pipeline`'s
  /// `trip_bound`). Mutually exclusive with `tripCount`, which the op verifier
  /// enforces.
  std::optional<int64_t> tripBound;
  Source lbSource;   // lower bound (counter init)
  Source ubSource;   // upper bound; see `tripCount` above
  Source stepSource; // step (counter increment)
  // The width the iteration counter is BUILT at.
  Type counterType;
  // TERMINATION class as the emitter discriminates it, axis 2 of the pair
  // above. A while loop (a `dcp.condition` terminator) is a flushing pipeline
  // whose exit is data-dependent. The declared `determinacy` below agrees where
  // a while is always declared `Conditional` (asserted in `deriveShapes`), but
  // NOT conversely: the reifier also stamps a `dcp.select` `Conditional`, while
  // `conditional` stays false for it since a guard is not a flushing loop.
  bool conditional = false;
  // The two raw structural flags `shape` is derived from. Consumers should read
  // `shape`.
  bool guard = false;      // this region op is a dcp.select
  bool container = false;  // nests another dcp region in EITHER arm, so a
                           // guard with children is `container` too and this
                           // is not the same as `shape == Container`
  std::string counterName; // source loop IV name (its NameLoc), for a readable
                           // iteration-counter wire; empty when the IV carried
                           // no name (best-effort)
  /// A REGISTER this region carries beside its own counter, holding
  /// `coeff * digit` of it for a coefficient and a digit an access's address
  /// needs, tracked incrementally rather than rebuilt: an induction variable
  /// makes consecutive iterations differ by a constant, so the constant
  /// multiply that dominates an address is unnecessary.
  ///
  /// A DIGIT of the counter rides the same register with two more constants.
  /// `(x floordiv D) mod K` advances by nothing on most iterations and by one
  /// where `x` crosses a multiple of `D`, maintained by a carry from a
  /// companion register holding `x mod D` (itself a stride with `wrap = D`),
  /// wrapping at `K` by subtracting once. A `floordiv`/`mod` on the address
  /// path pays every cycle; this pays a comparator instead.
  ///
  /// One update rule covers both:
  ///
  ///     raw  = cur + step + (carry fired ? bump : 0)
  ///     next = wrap && raw >= wrap ? raw - wrap : raw
  ///
  /// A plain scaled counter is `bump = wrap = 0`. `step + bump <= wrap` holds
  /// by construction (`asDigit` refuses a step that could wrap twice), so the
  /// single subtract is exact.
  ///
  /// A DECREASING digit (`A[N-1-i]`) mirrors it: `step`/`bump` go negative and
  /// the wrap ADDS on borrow (`raw > cur` unsigned) instead of subtracting on
  /// overflow.
  struct AddrStride {
    int64_t init;       // `coeff * lb`, the value the register loads at start
    int64_t step;       // `coeff * step`, added wherever the counter advances
    int64_t bump = 0;   // added when `carry`'s register wraps
    int64_t wrap = 0;   // subtracted on reaching it (0: a plain accumulator)
    unsigned carry = 0; // slot whose wrap gates `bump`; self means none
    bool hasCarry = false; // whether `carry` names one
    bool down = false;     // counts down, so `wrap` is added on borrow
    /// The width the register is BUILT at: every field above is compile-time,
    /// so the range it runs over is too, rounded up to bits (a wrapping digit
    /// needs `clog2` of its modulus; a row stride, `clog2` of the array),
    /// independent of the counter's own width. `kIndexWidth` when the range is
    /// unbounded (`slotFor`).
    unsigned width = kIndexWidth;
  };
  /// Deduplicated, since two accesses down the same row share a stride. Some
  /// slots exist only to carry another (the `x mod D` companion of a quotient
  /// digit) and no access names them; a carry always precedes its consumer, so
  /// one pass emits them. Empty when no address follows this counter, or when
  /// its bounds are not constant, which is what makes the fields compile-time
  /// values at all.
  llvm::SmallVector<AddrStride> addrStrides;

  // Composition class, DERIVED by `dcpRegionTiming` in `addRegion` rather than
  // read back off the report the same function stamps on the region op.
  // `deriveShapes` asserts the one cross-axis invariant, that `conditional`
  // implies `determinacy == conditional`.
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;

  // The TERMINAL cycle the latency model was composed off (`drain` on the
  // region op), against which `HWEmitter::emitRegion` checks the `drainStage`
  // it independently derives from the built datapath. A leaf's `done` rises
  // `drainStage + 1` cycles after its last issue, so a divergence here is a
  // consumer placed at an offset the hardware does not honour.
  std::optional<int64_t> modelledDrain;

  // Composition predecessors: the earlier top-level sibling regions this one
  // must start after, set by `recordSiblingDeps`. Only top-level regions
  // populate it, since container children stay serial. A region depends on an
  // earlier sibling iff they touch a shared memref (a data hazard or a
  // read-port conflict) or a cross-region SSA edge (a scalar survivor);
  // functional units are auto-disjoint under per-region binding, so shared
  // *memory* is the only cross-region resource. A region with no predecessors
  // starts concurrently with the kernel; one with predecessors starts on their
  // joined `done`. Producers precede consumers in program order, so the
  // relation is a DAG.
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
  // Empty for a container loop and for a then-only guard.
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
  // counted loop's k-th result IS the final value of its k-th iter-arg.
  llvm::SmallVector<RegionResult, 1> results;

  // This region's control predicate, as a resolved i1 Source: a while's
  // per-iteration continue condition, or a guard's (dcp.select) run-once
  // predicate. None for a counted region, which terminates on its counter.
  //
  // A while's condition is a scheduled compute producer (cmpi/cmpf, a
  // Source::Unit); a guard's is that same combinational unit over the enclosing
  // counter (an affine guard `i > j`) or a scheduled prologue region's survivor
  // (a data-dependent `flag[j] > 0`). Either way it is *held* for the run it
  // gates: a guard start-gates its children by it, so the not-taken arm's
  // stores never fire structurally, with no per-store gate.
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
  /// How many write ports one array is worth spreading over, from the device's
  /// `max_writes`. A true dual port is what infers; past it the inference fails
  /// outright, so a further colour would buy nothing and still cost its address
  /// and data multiplexers. It bounds the module BOUNDARY for the same reason,
  /// since whatever backs the array upstream is the same RAM.
  unsigned maxWritePorts = 2;

  // Derived structural cells.
  std::vector<FuncUnit> units;
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
  // internal memref is on-chip `seq.hlmem` storage and takes no port. This is
  // the ONE enumeration: the index of an access here IS its port identity,
  // mirrored back onto the access as `MemUnit::Access::portIdx` and read by the
  // port declaration, the naming layer, the manifest and the emitter alike.
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
  /// \p lib is the device the scheduler priced this kernel against: its storage
  /// view resolves each MemUnit's implementation and access latency, its
  /// operator rows let \p policy price a fold's multiplexer against
  /// \p cycleTime, the period the schedule was cut to.
  Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
           const OperatorLibrary &lib, float cycleTime,
           const CalleeCtx *callees = nullptr, bool isTop = false);

  /// The dcp op whose execution produces \p s's value, or null when the Source
  /// has no producing op: a literal, the iteration counter, a kernel input
  /// port, a held value (Reg / Survivor) or a derived mux. The ONE definition
  /// of the Source -> op mapping.
  Operation *producingOp(const Source &s) const;

  /// \p s's compile-time value, when it is an integer literal cell; empty for
  /// every Source whose value is only known at run time.
  std::optional<int64_t> constantOf(const Source &s) const;

  void dump(llvm::raw_ostream &os) const;

  /// Log what the allocation cost: per region the compute ops, the units they
  /// were bound to and the muxes sharing grew, then per array its write ports.
  /// A diagnostic, not an IR attribute or a manifest field.
  void reportAllocation() const;

  /// The fewest ports ONE BANK of memory \p id can be built with: the largest
  /// set of its accesses that can issue in one cycle, counting a child's port
  /// as an access. Per bank, since a bank is its own `seq.hlmem` and accesses
  /// naming different ones never contend. With \p writesOnly, only writes are
  /// counted, which is the budget a RAM's write ports are checked against;
  /// otherwise every access counts, which is what a RAM PORT actually serves (a
  /// port reads OR writes in a cycle, so two writers plus a concurrent reader
  /// need three).
  ///
  /// Conservative in the safe direction. Only an ordering the model already
  /// proves separates a pair: two top-level regions touching one array are
  /// ordered by `recordSiblingDeps`, two calls by `recordCallDeps` unless a
  /// channel joins them in a concurrent container, and two region-local
  /// accesses at different modulo residues never share a cycle. Anything else
  /// counts as simultaneous, so this never under-states.
  unsigned portsNeeded(MemId id, bool writesOnly) const;

  /// Which write port each writer of \p id drives: a colouring of the very
  /// relation `portsNeeded` takes its clique over, so it uses exactly that many
  /// ports. This function's own accesses come first, indexed as
  /// `MemUnit::accesses` with `kNoWritePort` at a read, then the CALL-mastered
  /// writes at the slots `callPortSlot` names.
  ///
  /// Absent when the writes cannot be redistributed and each keeps its own
  /// port, for any of three reasons: `portGraph` declined to relate them, they
  /// need more than \p maxPorts, which the caller sets from what its device can
  /// build, or a simultaneous pair is not PROVEN to address different words.
  /// Writes on different ports must be, having no shared block to order them.
  /// Two pairs are proven: two accesses inside one region, where a memory
  /// dependence would have made the scheduler separate them by a cycle (a
  /// store's SDC row carries its write latency of 1), and two write ports of
  /// ONE child that declared them independent, which is that child asserting
  /// the same thing about its own accesses. Two DIFFERENT children, or a child
  /// and a local access, are related by nothing and refuse the colouring.
  std::optional<llvm::SmallVector<unsigned>>
  writePortColouring(MemId id, unsigned maxPorts) const;

  /// Where a CALL-mastered write of \p id sits in a `writePortColouring`
  /// result: after this function's accesses, the calls in order and each call's
  /// memory arguments in order, which is the order `portGraph` builds its
  /// vertices in.
  unsigned callPortSlot(MemId id, CallId call, unsigned arg) const;

  /// No write port applies: `writePortColouring`'s entry for an access that is
  /// not a write, and `portGraph`'s for a vertex that is a call's port rather
  /// than an access of this function.
  static constexpr unsigned kNoWritePort = ~0u;


  /// The accesses of \p id the port model counts and the "can issue in one
  /// cycle" relation over them, one adjacency bitset per access. \p accessOf
  /// maps a vertex back to its index in `MemUnit::accesses`, or `kNoWritePort`
  /// for a call. \p callerOf, when given, maps it to the call that masters it
  /// and whether that call declared its ports independent, or to `{-1, false}`
  /// for an access of this function. Shorter than \p accessOf when there are
  /// more than the 64 a bitset holds, where the relation is not built at all
  /// and every access counts as simultaneous.
  llvm::SmallVector<uint64_t>
  portGraph(MemId id, bool writesOnly,
            llvm::SmallVectorImpl<unsigned> &accessOf,
            llvm::SmallVectorImpl<std::pair<int, bool>> *callerOf = nullptr)
      const;
};

//===----------------------------------------------------------------------===//
// The model visitor. `Source`s are scattered across ~20 slots of the model, so
// `forEachSource` is the one traversal: a new `Source` field is covered by
// adding it here, once.
//===----------------------------------------------------------------------===//

/// One `Source` slot in the model: what it drives, and whether being
/// unresolved (`Source::Kind::None`) is a defect there.
struct SourceSite {
  enum class Slot {
    UnitInput,        // a compute unit's operand port
    UnitInit,         // that port's reduction identity (absent => None)
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
// authority for the cycle a producing op's result lands, relative to its
// issuing pulse.
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
/// rule: the model is value-typed but the emitted carrier is a bit vector, and
/// the emitter (`uarch::hwType`) and the boundary port model
/// (`iface::bitWidth`) must not disagree about how wide it is.
unsigned hwWidth(Type t);

/// The banking of an *external* (argument) memory access, so the boundary
/// presents one interface per bank. `factor == 1` is an unbanked memory
/// (`bank == 0`); a banked access is either statically routed (`bank` set) or
/// data-dependent (`bank` empty -> a crossbar over all `factor` interfaces).
///
/// Both halves are stored on the model (`MemUnit::numBanks` +
/// `Access::staticBank`); this pairs them under the name the consumers ask the
/// question by. It lives here, not in an emitter header, because the
/// boundary/naming layer (`Naming.h`, `iface::ModuleInterface`) needs it and
/// must depend on L2 only.
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
