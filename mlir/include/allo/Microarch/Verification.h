/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_VERIFY_H
#define ALLO_MICROARCH_VERIFY_H

#include "allo/Microarch/Datapath.h"
#include "allo/Scheduling/OperatorLibrary.h" // prices muxes and units

#include "mlir/Support/LLVM.h"

namespace mlir::allo::uarch {

// The three functions below are cut by who is at fault, which also picks the
// diagnostic each may raise.

/// What the design asks for and this device cannot give: reported against the
/// user, who can change the kernel, the schedule directives or the part.
/// `logging::error` and `logging::warn` only.
LogicalResult checkInputLegality(dcp::DCPathModuleOp func, const Datapath &dp);

/// Shapes this backend does not lower yet, including the one the clock rules
/// out: the schedule was cut against \p cycleTime (ns) over a datapath with no
/// sharing muxes, so a binding that grows them is held to that period here.
/// \p lib prices the muxes and the units they feed. `logging::unsupported`
/// only.
LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp,
                                 float cycleTime, const OperatorLibrary &lib);

/// Invariants an upstream pass owns, asserted at this seam. `assert` only, so
/// this compiles away in a release build.
void assertModelInvariants(const Datapath &dp);

/// The three above, in order. The one call the emit driver makes.
LogicalResult validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp,
                               float cycleTime, const OperatorLibrary &lib);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H
