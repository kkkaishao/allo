/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_OPS_H
#define ALLO_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
// ISA relayout ops expose getReassociationIndices() (ReassociationIndices).
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "allo/IR/AlloDialect.h.inc"

#include "allo/IR/AlloOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.h.inc"

namespace mlir::allo {
constexpr llvm::StringLiteral kAlloSignedAttr = "allo.signed";
constexpr llvm::StringLiteral kAlloLazyAttr = "allo.lazy";
constexpr llvm::StringLiteral kAlloAsyncAttr = "allo.async";
constexpr llvm::StringLiteral kMemoryInitAttr = "allo.mem.init";
} // namespace mlir::allo

#endif // ALLO_OPS_H
