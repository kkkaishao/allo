/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_VERIFY_H
#define ALLO_MICROARCH_VERIFY_H

#include "allo/Microarch/Datapath.h"

#include "mlir/Support/LLVM.h"

namespace mlir::allo::uarch {

/// Model well-formedness: a schedulable region set, a feasible schedule, and
/// every required `Source` slot resolved.
LogicalResult verifyDatapath(dcp::DCPathModuleOp func, const Datapath &dp);

/// Device-contract limits the emitted structure cannot realize.
LogicalResult checkDeviceCapability(dcp::DCPathModuleOp func,
                                    const Datapath &dp);

/// Shapes outside the subset this emitter lowers.
LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp);

/// The three above, in order. The one call the emit driver makes.
LogicalResult validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_VERIFY_H
