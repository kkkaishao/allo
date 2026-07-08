/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_PASSES_H
#define ALLO_SCHEDULING_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::allo {
#define GEN_PASS_DECL
#include "allo/Scheduling/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "allo/Scheduling/Passes.h.inc"
} // namespace mlir::allo

#endif // ALLO_SCHEDULING_PASSES_H
