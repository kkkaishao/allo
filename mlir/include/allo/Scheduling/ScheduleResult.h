/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULERESULT_H
#define ALLO_SCHEDULING_SCHEDULERESULT_H

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::allo {

/// Write the solved schedule of one region onto the IR as attributes: each
/// registered op gets `allo.sched.t` (start time) and `allo.sched.region`, and a
/// per-region descriptor is appended to the func-level `allo.sched.regions`
/// array. This is the schedule "carrier"; nothing structural is materialized.
void annotateRegion(circt::scheduling::Problem &problem, func::FuncOp func,
                    int64_t regionId, llvm::StringRef kind,
                    std::optional<unsigned> ii, int64_t order);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULERESULT_H
