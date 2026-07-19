/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The port-interface model: the single source of truth for a module's boundary
// port *names*.
//
// Every hardware boundary of a kernel is one typed interface -- a `Stream`
// (FIFO handshake), a `Memory` (a read/write access to an argument array), a
// `Scalar` input, or a `Result` output. Each interface owns the *concrete*
// port-name strings the rest of the flow uses (`s_data`/`s_valid`/`s_ready`,
// `out_wr_addr`/`_data`/`_we`, ...). The names are built here, once, from a
// base name plus the field-suffix vocabulary below -- so the emitter (port
// declaration + body access), the dataflow-composition wiring, and the cosim
// harness never re-append suffixes independently and diverge.
//
// The structs carry only strings + ints (arg index, bank, factor, depth, bit
// width), so the whole model serializes to JSON and crosses the C++/Python seam
// verbatim: C++ authors every port name, Python only reads.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_INTERFACE_H
#define ALLO_MICROARCH_INTERFACE_H

#include "allo/Microarch/HWEmitter.h" // uarch::AccRef and the base-name helpers

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>
#include <vector>

namespace mlir::allo::iface {

// The field-suffix vocabulary -- the ONE place these strings are defined.
constexpr llvm::StringLiteral kData = "_data";
constexpr llvm::StringLiteral kValid = "_valid";
constexpr llvm::StringLiteral kReady = "_ready";
constexpr llvm::StringLiteral kAddr = "_addr";
constexpr llvm::StringLiteral kWe = "_we";

// The field port name for a given interface base.
inline std::string
data_(llvm::StringRef base) { // avoid conflict with std::data()
  return base.str() + kData.str();
}
inline std::string valid(llvm::StringRef base) {
  return base.str() + kValid.str();
}
inline std::string ready(llvm::StringRef base) {
  return base.str() + kReady.str();
}
inline std::string addr(llvm::StringRef base) {
  return base.str() + kAddr.str();
}
inline std::string we(llvm::StringRef base) { return base.str() + kWe.str(); }

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
  int arg;
  bool write;
  int bank, factor; // cyclic bank this interface serves / the partition factor
  unsigned width;   // element bit width
  std::string base, addr, data, we;
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

/// The whole boundary of one module. `reads`/`writes` group by access (an inner
/// vector is the access's per-bank interfaces: one entry unbanked, N when a
/// data-dependent access spans every bank).
struct ModuleInterface {
  std::vector<Scalar> scalars;
  std::vector<FIFO> streams;
  std::vector<std::vector<Memory>> reads;
  std::vector<std::vector<Memory>> writes;
  std::vector<Result> results;

  ModuleInterface() = default;
  ModuleInterface(const uarch::Datapath &dp,
                  llvm::ArrayRef<uarch::AccRef> reads,
                  llvm::ArrayRef<uarch::AccRef> writes);

  /// Serialize the model to a compact JSON object.
  std::string toJSON() const;
};

} // namespace mlir::allo::iface

#endif // ALLO_MICROARCH_INTERFACE_H
