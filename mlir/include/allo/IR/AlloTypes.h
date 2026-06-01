/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TYPES_H
#define ALLO_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// The ISA buffer element type interface must precede the typedef classes that
// implement it (ScalarBuffer). ISA typedefs are emitted into AlloTypes.h.inc
// because AlloTypes.td includes AlloISATypes.td.
#include "allo/IR/AlloTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "allo/IR/AlloTypes.h.inc"

#endif // ALLO_TYPES_H
