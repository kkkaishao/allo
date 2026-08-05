/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_REGLEDGER_H
#define ALLO_MICROARCH_REGLEDGER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <map>
#include <tuple>
#include <vector>

namespace mlir::allo::uarch {

/// Why a register exists. The emitter knows this where it BUILDS the register,
/// and nowhere later: a reader of the emitted design sees a `seq.compreg` and
/// can recover the reason only by parsing the name it happens to carry, which
/// is a second copy of a convention `Naming.h` owns.
enum class RegRole {
  Value,    // a value delay chain: one datum carried across cycle boundaries
  Pulse,    // an activation chain: a region's issue delayed to an op's stage
  Counted,  // the counter a deep pulse delay is built as instead of a chain
  Survivor, // a region result, or a loop-carried iter-arg latch
  Counter,  // an iteration counter, or one of its address strides
  Control,  // run / phase / pending / done, and the rest of the control plane
};

llvm::StringRef roleName(RegRole role);

/// One class of register RUN: `count` runs of `depth` registers in series,
/// `width` bits each, all built for the same reason. A lone register is a run
/// of depth 1, so every register belongs to exactly one class and the design's
/// flip-flop count is `sum(width * depth * count)`.
///
/// The run, not the register, is the cost unit. Past the synthesizer's
/// shift-register extraction threshold a run stops costing flip-flops per
/// stage, so a cost model handed only a register total cannot price it.
struct RegClass {
  RegRole role = RegRole::Control;
  unsigned width = 0, depth = 0, count = 0;
};

/// Every register one module's emission built. Filled at the one point that
/// creates a `seq.compreg` (`EmitContext::reg`) plus the chain builders, which
/// charge a whole run at once, so the total is a COUNT and not an estimate.
class RegLedger {
public:
  /// Charge one run of \p depth registers of \p width bits. A depth of zero is
  /// no run at all (a chain a consumer reads at tap 0 builds nothing).
  void add(RegRole role, unsigned width, unsigned depth) {
    assert(width && "a register holds at least one bit");
    if (depth)
      ++runs[{role, width, depth}];
  }

  /// Every class, in a deterministic order, so a report built from this does
  /// not reorder between two runs of the same compile.
  std::vector<RegClass> classes() const;

  /// Flip-flops across every class.
  unsigned bits() const;

  void dump(llvm::raw_ostream &os) const;

private:
  std::map<std::tuple<RegRole, unsigned, unsigned>, unsigned> runs;
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_REGLEDGER_H
