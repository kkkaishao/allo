/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The port-interface model: the single source of truth for a module's boundary
// port *names*.
//
// Every hardware boundary of a kernel is one typed interface: a `Stream`
// (FIFO handshake), a `Memory` (a read/write access to an argument array), a
// `Scalar` input, or a `Result` output. Each interface owns the *concrete*
// port-name strings the rest of the flow uses (`s_st_data`/`_valid`/`_ready`,
// `out_wr0_addr`/`_data`/`_we`, ...), all composed by `Naming.h`, so that the
// emitter (port declaration + body access), the dataflow-composition wiring,
// and the cosim harness never build a name independently and diverge.
//
// The structs carry only strings + ints (arg index, bank, factor, depth, bit
// width), so the whole model serializes to JSON and crosses the C++/Python seam
// verbatim: C++ authors every port name, Python only reads. That covers the
// module's own name, the fixed control ABI (clk/rst/start/done) and the extern
// operator modules it instantiates, i.e. everything a simulator needs to bind
// the design, so no consumer re-derives a name or reads the emitted IR back.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_INTERFACE_H
#define ALLO_MICROARCH_INTERFACE_H

#include "allo/Microarch/Naming.h" // uarch::Datapath + the naming vocabulary

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace mlir::allo::iface {

/// A FIFO channel interface. Input (a `get`): the module reads `data` when
/// `valid`, drives `ready`. Output (a `put`): the module drives `data`/`valid`,
/// reads `ready`.
struct FIFO {
  int arg;        // kernel block-argument index (-1 if not an argument)
  bool isInput;   // get (input) vs put (output)
  int depth;      // FIFO depth
  unsigned width; // payload bit width
  std::string base, data, valid, ready;
};

/// One physical interface to an argument array (a single bank of it when the
/// argument is cyclically partitioned). A read exposes `{addr(out), data(in)}`;
/// a write `{addr, data, we}` (all out, `we` empty for a read).
struct Memory {
  /// One partitioned axis of the argument, mirroring `allo::BankLayout::Axis`:
  /// the host needs the same element-space decomposition the RTL addresses
  /// with, which `bank`/`factor` alone cannot express (they give the bank's
  /// identity and the total count, not how elements map onto it).
  struct Axis {
    int dim;
    int64_t factor;
    bool block;
  };
  int arg;
  bool write;
  int bank, factor; // the bank this interface serves / total physical banks
  unsigned width;   // element bit width
  unsigned latency; // access latency
  std::string base, addr, data, we;
  std::vector<int64_t> shape; // the argument's element shape
  std::vector<Axis> axes;     // partitioned axes, mixed-radix order (empty when
                              // unbanked)
};

/// A scalar input argument (one port, no suffix).
struct Scalar {
  int arg;
  unsigned width;
  std::string name;
};

/// A scalar function result (one output port, driven at `done`).
struct Result {
  unsigned width;
  std::string name;
};

/// One extern operator module this module instantiates, with the port shape it
/// was declared with. Published for two reasons: the simulation-model generator
/// builds its behavioral module from the manifest rather than re-parsing the
/// emitted IR, and it joins to the device operator on `impl` + `predicate`
/// rather than guessing them back out of the module name.
struct Operator {
  /// What a port is FOR, so a consumer classifies structurally instead of
  /// matching the name `clk` / `ce` back out (the `ce` bit in particular
  /// decides whether the behavioral model gates on a clock enable).
  enum class Role { Data, Clk, Ce, Out };
  struct Port {
    std::string name;
    unsigned width;
    Role role;
    bool isInput() const { return role != Role::Out; }
  };
  std::string module;    // the extern module's RTL name
  std::string impl;      // the device operator's sym_name
  std::string predicate; // compare predicate; empty for everything else
  std::vector<Port> ports;
};

/// The whole boundary of one module. `reads`/`writes` group by access (an inner
/// vector is the access's per-bank interfaces: one entry unbanked, N when a
/// data-dependent access spans every bank).
struct ModuleInterface {
  // The emitted RTL module name and the MLIR symbol it came from; they differ
  // whenever the symbol needed legalizing (`top.child` -> `top_child`), and the
  // simulator only ever knows the former.
  std::string module, symbol;
  std::vector<Scalar> scalars;
  std::vector<FIFO> streams;
  std::vector<std::vector<Memory>> reads;
  std::vector<std::vector<Memory>> writes;
  std::vector<Result> results;
  std::vector<Operator> operators;

  ModuleInterface() = default;
  /// Build the boundary from \p dp, whose `readPorts` / `writePorts` are the
  /// one enumeration of its external memory accesses.
  explicit ModuleInterface(const uarch::Datapath &dp);

  /// Every memory interface of argument \p arg, reads before writes and flat
  /// across access groups. An argument accessed at several points has several
  /// port groups (read-twice -> two reads; an accumulator -> a read and a
  /// write), and a cyclically partitioned access has one interface per bank
  /// within its group; a caller wiring the argument needs all of them, which is
  /// why this flattens rather than preserving the grouping.
  llvm::SmallVector<const Memory *, 2> portsForArg(int arg) const;
  /// The scalar input port of argument \p arg, or null if \p arg is not one.
  const Scalar *scalarForArg(int arg) const;
  /// The stream interface of argument \p arg, or null if \p arg is not one. A
  /// stream argument is single-ended within a module (one `get` side or one
  /// `put` side), so unlike a memory it has exactly one interface.
  const FIFO *streamForArg(int arg) const;

  /// Serialize the model to a compact JSON object.
  std::string toJSON() const;
};

} // namespace mlir::allo::iface

#endif // ALLO_MICROARCH_INTERFACE_H
