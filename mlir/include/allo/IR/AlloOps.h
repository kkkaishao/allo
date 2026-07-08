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

// ISA op interfaces (BufferAccessOpInterface references BufferTypeInterface,
// which is pulled in via AlloTypes.h above).
#include "allo/IR/AlloOpInterfaces.h.inc"

// Op interfaces defined in AlloOps.td (e.g. ScheduledOpInterface); must precede
// the op classes that implement them.
#include "allo/IR/AlloOpsInterfaces.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloISAOps.h.inc"

namespace mlir::allo {
constexpr llvm::StringLiteral kAlloSignedAttr = "allo.signed";
constexpr llvm::StringLiteral kAlloLazyAttr = "allo.lazy";
// A concurrent-spawn (`await`) marker carried onto the `func.call` that
// `convert-allo-to-func` produces from an `allo.invoke {async}` (func.call has
// no `async` field). Written by ConvertAlloToFunc and read by the dataflow
// composition lowering -- keep the two in sync through this one constant.
constexpr llvm::StringLiteral kAlloAsyncAttr = "allo.async";
} // namespace mlir::allo

#endif // ALLO_OPS_H
