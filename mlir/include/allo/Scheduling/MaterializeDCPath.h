/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MATERIALIZEDCPATH_H
#define ALLO_SCHEDULING_MATERIALIZEDCPATH_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::allo {

class OperatorLibrary;

/// Reify the solved schedule of every scheduled func in \p module into
/// `allo.dcp.*` ops (dcp.operator declarations, dcp.pipeline/sequential
/// regions, dcp.compute/load/store), then strip the carrier. The schedule
/// reaches this step as transient `allo.sched.*` attributes, and operator
/// latencies come from \p lib.
/// This is the scheduler's reification step; it runs at module scope because
/// dcp.operator declarations are module-level symbols.
void materializeModuleToDCP(ModuleOp module, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MATERIALIZEDCPATH_H
