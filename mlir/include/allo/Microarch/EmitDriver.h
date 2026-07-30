/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_EMITDRIVER_H
#define ALLO_MICROARCH_EMITDRIVER_H

#include "allo/Microarch/Datapath.h"

#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::allo::iface {
struct ModuleInterface;
} // namespace mlir::allo::iface

namespace mlir::allo::uarch {

/// Lower the scheduled `func.func`s reachable from \p top to structural
/// `hw.module`s (leaf datapaths + dataflow/sequential tops), erasing the source
/// funcs. This is the free function behind the `allo-datapath-to-hw` pass.
/// Emission is rooted at \p top and runs bottom-up over the call DAG (callees
/// before callers), mirroring the scheduler. \p binding names the binding
/// policy. On success \p interfaces maps each emitted module's symbol name to
/// its port-interface JSON (the cosim manifest), so a caller gets the boundary
/// directly without reading any IR attribute.
LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top,
                               llvm::StringMap<std::string> &interfaces);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_EMITDRIVER_H
