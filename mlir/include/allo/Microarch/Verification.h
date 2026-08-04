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

/// Model well-formedness: a schedulable region set, a feasible schedule, and
/// every required `Source` slot resolved.
LogicalResult verifyDatapath(dcp::DCPathModuleOp func, const Datapath &dp);

/// Device-contract limits the emitted structure cannot realize, including the
/// clock: the schedule was cut against \p cycleTime (ns) over a datapath with
/// no sharing muxes and only the cells the solve saw, so a binding that grows
/// muxes and an expression the reifier synthesizes afterwards are both held to
/// it here. \p lib prices them and the units they feed.
LogicalResult checkDeviceCapability(dcp::DCPathModuleOp func,
                                    const Datapath &dp, float cycleTime,
                                    const OperatorLibrary &lib);

/// Shapes outside the subset this emitter lowers.
LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp);

/// The three above, in order. The one call the emit driver makes.
LogicalResult validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp,
                               float cycleTime, const OperatorLibrary &lib);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H
