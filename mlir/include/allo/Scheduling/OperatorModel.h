/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORMODEL_H
#define ALLO_SCHEDULING_OPERATORMODEL_H

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Block.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::allo {

/// Assign operator types (Calyx-like latencies) and per-memory limited
/// resources to every op reachable from \p body (a loop body). Unclassified ops
/// default to zero-latency combinational (S4 refines this). Takes a
/// SharedOperatorsProblem so it serves both ModuloProblem (loops) and acyclic
/// resource problems (straight-line spans).
LogicalResult
populateOperatorTypes(Block &body,
                      circt::scheduling::SharedOperatorsProblem &problem);

/// Same, over the top-level ops of a straight-line region (each walked).
LogicalResult
populateOperatorTypes(ArrayRef<Operation *> ops,
                      circt::scheduling::SharedOperatorsProblem &problem);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORMODEL_H
