/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The L2 microarchitecture layer: an in-memory, value-typed, technology-
// independent bound-datapath model that sits between the materialized schedule
// (`allo.dcp.*` ops) and structural RTL (hw/seq/comb). It is deliberately not
// an MLIR dialect: with a single consumer (the emitter) and no transforms, C++
// structs are cheaper to evolve than TableGen ops.
//
// Design invariant: the binder writes only the decision maps (op->unit,
// value->reg, access->port); the structural cells (units/regs/muxes) and their
// interconnect are *derived* from those decisions plus the schedule. Rebinding
// is therefore "edit the maps, re-derive", and the emitter depends only on the
// derived structure, never on which binding policy produced it.
//
// Register chains are modelled as one shift register with taps; memref
// arguments are bare external memory interfaces (no AXI).
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_DATAPATH_H
#define ALLO_MICROARCH_DATAPATH_H

#include "allo/IR/AlloAttrs.h" // MemoryImplEnum (storage primitive)

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo::uarch {

struct BindingPolicy;

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

//===----------------------------------------------------------------------===//
// Control: semi-abstract. A cell/mux-select/port access is active when an
// (abstract) per-region counter lands on one of `states`. Cyclic regions count
// modulo II; acyclic regions count a plain 0..length-1 sequence. Materializing
// this into a counter + decode logic is the emitter's job, not this model's.
//===----------------------------------------------------------------------===//

struct ControlPredicate {
  RegionId region = 0;
  llvm::SmallVector<unsigned, 2> states; // counter residues at which it fires
};

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
    Stream
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
  std::string opType;    // the operator mnemonic (e.g. "addi", "mulf")
  std::string impl;      // realization: native keyword or IP module name
  unsigned latency = 0;  // result available `latency` cycles after issue
  bool pipelined = true; // accepts a new input every cycle
  Type resultType;       // value-typed (e.g. f32), not bit-blasted

  // Ops bound here, each with its issue cycle (residue mod II in a cyclic
  // region). Sharing puts several non-conflicting ops in this list.
  llvm::SmallVector<std::pair<Operation *, unsigned>, 1> boundOps;

  // One resolved driver per input operand port (post-binding). A *fused*
  // recurrence (II == latency, depth II-L == 0) has a self-referential input
  // (`Source::Unit{this.id}`): the IP's own pipeline is the accumulator
  // register.
  llvm::SmallVector<Source, 2> inputs;

  // Per-input reduction identity (parallel to `inputs`). A recurrence input --
  // the port that reads a loop-carried iter_arg -- carries the iter_arg's init
  // here, so the emitter re-injects the identity on that port at the first
  // iteration and a retriggered reduction restarts each outer pass. None for
  // every non-recurrence input. This is the sole init mechanism; a register in
  // the recurrence cycle is a plain delay. It rides the input port (not a
  // register) because the widened idiom trunc(add(ext(acc),ext(x))) reads acc
  // through a bare wire, not a tap.
  llvm::SmallVector<Source, 2> inputInits;
};

/// A shift-register chain carrying one SSA value across cycle boundaries. Its
/// length is the largest delay any consumer needs; consumers read at their own
/// `tap` (see Source). Generalises the accumulator shift register from
/// rotate-reductions.
struct Register {
  RegId id = 0;
  Value value; // the L0 value being held (for provenance / debug)
  Type type;
  unsigned depth = 0; // chain length in cycles (>= 1 for a real register)
  Source input;       // driver of the chain head (the producing cell output)
  llvm::SmallVector<unsigned, 2> taps; // distinct tap levels actually read
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
  unsigned numBanks = 1;   // from MemoryBankModel (cyclic partition)
  unsigned portsPerBank = 2;
  MemoryImplEnum impl = MemoryImplEnum::LUTRAM; // resolved storage primitive

  /// One bound access. A read's loaded data is referenced by
  /// Source{Mem, id, <index of this access>}; a write consumes `data`.
  struct Access {
    Operation *op = nullptr;
    bool isWrite = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned bank = 0;
    unsigned port = 0;
    ControlPredicate when;
    AffineMap addrMap; // index map over `addr` operands (null: plain indices)
    llvm::SmallVector<Source, 2> addr; // address operand drivers (delayed IVs)
    Source data;                       // write data driver (writes only)
  };
  llvm::SmallVector<Access, 2> accesses;
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

  struct Access {
    Operation *op = nullptr; // the stream.get / stream.put op
    bool isPut = false;
    unsigned region = 0; // the RegionBlock this access is scheduled in
    unsigned stage = 0;  // scheduled cycle within the region (dcpStart)
    Source data;         // put: the token's data driver (puts only)
    // A predicated access (an i1 `pred` operand from a masked `if`) fires --
    // consumes / produces a token -- only where this holds. Delayed to `stage`
    // like `data`; None for an unconditional access.
    Source when;
  };
  llvm::SmallVector<Access, 1> accesses;
};

/// A multiplexer inserted where sharing makes several sources contend for one
/// sink input. Empty in the trivial binding; one per shared-unit input port
/// that sees different drivers across the ops bound to it.
struct Mux {
  MuxId id = 0;
  Type type;
  llvm::SmallVector<Source, 2> sources;
  llvm::SmallVector<ControlPredicate, 2>
      selects; // abstract predicate per source
  // The op whose issue selects each source (parallel to `sources`): the source
  // is driven onto the shared unit's input on the cycle that op consumes it, so
  // the select is `delayValid(issue, schedT(op))` -- the same per-op activation
  // pulse a store's write-enable uses. The MRT guarantees these are mutually
  // exclusive (disjoint residues), so the derived mux is a plain priority
  // chain.
  llvm::SmallVector<Operation *, 2> selectOps;
  RegionId region = 0; // region whose issue pulse times the selects
};

/// A top-level scalar/interface port (kernel argument or result). Memref
/// arguments become external `MemUnit`s instead.
struct IOPort {
  IOId id = 0;
  Value value;
  Type type;
  bool isInput = true;
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
// are straight-line. Blocks run
// in program order with no overlap (locked decision), so a single sequential
// hand-off chains them.
//===----------------------------------------------------------------------===//

struct RegionBlock {
  RegionId id = 0;
  enum class Kind { Cyclic, Acyclic } kind = Kind::Acyclic;
  std::optional<unsigned> ii; // set iff Cyclic
  unsigned length = 0;        // cycle slots (single-iteration depth)
  // Counted-loop induction: the IV runs `lb, lb+step, ...` up to (excluding) an
  // upper bound. Each of lb / ub / step is either compile-time -- the
  // `lb`/`step` integers (defaults 0/1) and the derived constant ub `lb +
  // tripCount*step` -- or a runtime value carried as a resolvable `Source` (a
  // data-dependent range start / count / stride). A runtime bound leaves
  // `tripCount` empty (no static trip); the two forms of one bound are mutually
  // exclusive.
  std::optional<int64_t> tripCount; // constant trip iff Cyclic
  int64_t lb = 0;                   // compile-time lower bound (counter init)
  int64_t step = 1;                 // compile-time step (counter increment)
  Source lbSource;          // runtime lower bound, else the `lb` constant
  Source ubSource;          // runtime upper bound, else `lb + tripCount*step`
  Source stepSource;        // runtime step, else the `step` constant
  bool conditional = false; // a while loop (dcp.condition terminator):
                            // flushing pipeline, exit is data-dependent
  bool guard = false;       // a dcp.select guard: its children run once
                            // iff the predicate holds (`guardCond`), else
                            // are skipped -- a predicated container
  bool container = false;   // nests another dcp region (a loop wrapper)
  std::string counterName;  // source loop IV name (its NameLoc), for a
                            // readable iteration-counter wire; empty if
  // the loop's IV carried no name (best-effort)

  // Region nesting. A container region drives its `children` in its body; each
  // child's `parent` is the enclosing container. Top-level regions (no parent)
  // are the func-scope siblings chained by the sequencer; a container runs its
  // child `tripCount` times (hierarchical control, II_outer >= L_inner).
  std::optional<RegionId> parent;
  llvm::SmallVector<RegionId, 2> children;

  // Cells owned by this region (ids are Datapath-global; these record
  // membership and thus which counter drives them).
  llvm::SmallVector<UnitId, 4> units;
  llvm::SmallVector<RegId, 4> regs;
  llvm::SmallVector<MuxId, 2> muxes;
};

//===----------------------------------------------------------------------===//
// The whole microarchitecture of one function.
//===----------------------------------------------------------------------===//

struct Datapath {
  func::FuncOp func;

  // Derived structural cells.
  std::vector<FuncUnit> units;
  std::vector<Register> regs;
  std::vector<MemUnit> mems;
  std::vector<StreamChannel> streams;
  std::vector<Mux> muxes;
  std::vector<IOPort> ios;
  std::vector<ConstCell> consts;
  std::vector<Result> results;      // scalar func results, in return order
  std::vector<RegionBlock> regions; // program order

  // L1 binding decisions the policy writes; the structure above is derived from
  // these plus the schedule. (Memory port binding lives in MemUnit::accesses,
  // co-located with its memref.)
  llvm::DenseMap<Operation *, UnitId> opToUnit;
  llvm::DenseMap<Value, RegId> valueToReg;

  // The Source of each of a region's results (its `uncondition` operands'
  // producers), for regions that yield one or more values, indexed by result
  // number. A sibling region consuming result k reads a
  // `Source::Survivor{producingRegion, k}`; the emitter latches each Source
  // when the producing region completes. Empty for result-less regions; a
  // `None` entry marks an untracked result (asserts only if a sibling reads
  // it).
  llvm::DenseMap<RegionId, llvm::SmallVector<Source>> regionResult;

  // The i1 predicate of a guard region (a dcp.select). The guard's children run
  // once iff it holds (emitGuard start-gates them); otherwise they never issue,
  // so their stores never fire -- the predicate reaches the store write-enable
  // structurally, not by a per-store gate. Like a while's condition, it is
  // either a scheduled value -- a preceding condition region's survivor, so
  // `condition` is set (a data-dependent guard, e.g. `flag[j] > 0`) -- or an
  // unscheduled raw combinational arith tree over the enclosing container's
  // counter, so `condition` is None and `condValue` is the root (an affine
  // guard `i > j` over a flattened nest, evaluated by evalRawArith). Present
  // only for a guard (RegionBlock::guard) region.
  struct GuardInfo {
    Source condition;
    Value condValue;
  };
  llvm::DenseMap<RegionId, GuardInfo> guardCond;

  // A container's loop-carried recurrence: per iter-arg, its init (loaded at
  // start) and next-value (advanced into the register when an outer iteration
  // drains) Sources. Each iter-arg becomes a frozen survivor register the
  // emitter latches, read by the children (their init reads
  // Source::Survivor{region, k}) and by a sibling of the final value. Recorded
  // for both regimes that need it -- a counted container carrying an
  // accumulator into an inner reduction, and a while (conditional) container /
  // leaf whose flushing controller also gates `running` on `condition`. The
  // `condition`/`condValue` fields are set only for a while (else None): a leaf
  // while's condition is a scheduled compute unit (`condition` is its Source);
  // a conditional container's is an unscheduled raw arith tree over the
  // iter-args
  // (`condValue` is its root, `condition` None). Such a region records no
  // `regionResult` (its results are these survivors).
  struct CarryInfo {
    Source condition;
    Value condValue;
    llvm::SmallVector<Source> inits;
    llvm::SmallVector<Source> nexts;
  };
  llvm::DenseMap<RegionId, CarryInfo> carryInfo;

  Datapath() = default;
  explicit Datapath(func::FuncOp f);
  Datapath(func::FuncOp func, const BindingPolicy &policy);

  void dump(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR. `readyCycleOf` is the single
// authority for "the cycle a producing op's result lands, relative to its
// issuing pulse": both the builder (register-depth derivation) and the emitter
// (result capture) consult it, so the latency model has one definition.
//===----------------------------------------------------------------------===//

/// The `dcp.operator` op characterizing a compute/load (null if none); returned
/// as a generic Operation to keep this header free of the dcp op headers.
Operation *dcpOperatorOp(Operation *op);
/// Region-relative schedule cycle of a dcp compute/load/store op (its `start`).
unsigned dcpStart(Operation *op);
/// Result latency of a producing dcp op (0 if uncharacterized).
unsigned dcpLatency(Operation *op);
/// The cycle a producing op's result is ready: `dcpStart + dcpLatency` (a
/// stream get is a combinational front-read, latency 0). Zero for an at-issue
/// value with no producing op (a constant, the iteration counter).
unsigned readyCycleOf(Operation *op);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_DATAPATH_H
