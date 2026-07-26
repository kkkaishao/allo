/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The envelope: everything a Datapath must satisfy before any hardware is
// built. Loud, never silent: a shape outside the subset is an `emitError` (a
// Python `RuntimeError`) or an `assert`, never a quiet drop.
//
// Three checks, in the order they run, split by WHOSE contract they enforce:
//
//   verifyDatapath       the MODEL's own well-formedness. Built on
//                        `forEachSource`, so a new Source slot in the model is
//                        covered by declaring it there, not by remembering to
//                        add a loop here.
//   checkDeviceCapability  what the DEVICE promises: storage access latency,
//                        an operator IP's stall contract. Violations are the
//                        device description disagreeing with what the emitted
//                        structure can realize.
//   checkEmitterSubset   what THIS EMITTER lowers: region shapes,
//                        stream protocol, condition timing, operator
//                        realizability. Mirrors `emitRegion`'s dispatch and
//                        reads the same stored `RegionBlock::shape`.
//
// Order matters for diagnostic quality, not correctness: an unresolved driver
// is reported at its own slot before a later check reports the same value as
// "not lowered".
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_VERIFY_H
#define ALLO_MICROARCH_VERIFY_H

#include "allo/Microarch/Datapath.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::allo::uarch {

/// Model well-formedness: a schedulable region set, a feasible schedule, and
/// every required `Source` slot resolved.
LogicalResult verifyDatapath(func::FuncOp func, const Datapath &dp);

/// Device-contract limits the emitted structure cannot realize.
LogicalResult checkDeviceCapability(func::FuncOp func, const Datapath &dp);

/// Shapes outside the subset this emitter lowers.
LogicalResult checkEmitterSubset(func::FuncOp func, const Datapath &dp);

/// The three above, in order. The one call the emit driver makes.
LogicalResult validateDatapath(func::FuncOp func, const Datapath &dp);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H
