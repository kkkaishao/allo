/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMOPS_H
#define ALLO_TRANSFORMOPS_H

#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "allo/TransformOps/AlloTransformOps.h.inc"

namespace mlir::allo {
void registerTransformDialectExtension(DialectRegistry &registry);

constexpr StringLiteral OpIdentifier = "sym_name";
} // namespace mlir::allo

#endif // ALLO_TRANSFORMOPS_H
