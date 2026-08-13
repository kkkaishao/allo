/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_VERIFY_H
#define ALLO_MICROARCH_VERIFY_H

#include "allo/Microarch/Datapath.h"
#include "allo/Microarch/Report.h"           // TimingPath
#include "allo/Scheduling/OperatorLibrary.h" // prices muxes and units

#include "mlir/Support/LLVM.h"

#include <vector>

namespace mlir::allo::uarch {

// The three functions below are cut by who is at fault, which also picks the
// diagnostic each may raise.

/// How many of a module's worst paths the report keeps. More than one because a
/// compile is an iteration: the second and third findings are often the ones
/// the same fix also addresses.
inline constexpr unsigned kReportedPaths = 3;

/// What the design asks for and this device cannot give: reported against the
/// user, who can change the kernel, the schedule directives, the period or the
/// part. `logging::error` and `logging::warn` only.
LogicalResult checkInputLegality(dcp::DCPathModuleOp func, const Datapath &dp);

/// Shapes this backend does not lower yet, including the one the clock rules
/// out: the schedule was cut against \p cycleTime (ns) over a datapath with no
/// sharing muxes and no port selects, so what grows them is measured here, at
/// every capture point `forEachSource` enumerates, and appended to \p paths.
/// \p lib prices the muxes and the units they feed. `logging::unsupported`
/// where a binding can be withdrawn; elsewhere the path is reported and not
/// refused, missing a target period being a quality-of-result finding rather
/// than an illegal design. \p plannedBinding says the folds realize the
/// schedule solve's own allocation, which reserved headroom for every select
/// it bought, so a unit overrun is a broken invariant instead of a refusal.
LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp,
                                 float cycleTime, const OperatorLibrary &lib,
                                 bool plannedBinding,
                                 std::vector<TimingPath> &paths);

/// Invariants an upstream pass owns, asserted at this seam. `assert` only, so
/// this compiles away in a release build.
void assertModelInvariants(const Datapath &dp);

/// The three above, in order. The one call the emit driver makes. Returns this
/// module's `kReportedPaths` worst combinational paths, longest first, which
/// the report publishes and the QoR turns into a clock. Structures nobody
/// prices are not in them, so they estimate and never substitute for place and
/// route. Never empty: a module with no datapath still holds one register hop.
FailureOr<std::vector<TimingPath>>
validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp, float cycleTime,
                 const OperatorLibrary &lib, bool plannedBinding);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H
