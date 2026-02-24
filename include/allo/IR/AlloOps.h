#ifndef ALLO_OPS_H
#define ALLO_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloDialect.h.inc"
#include "allo/IR/AlloTypes.h"
#include "allo/IR/AlloInterfaces.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.h.inc"

namespace mlir::allo {
constexpr StringLiteral OpIdentifier = "sym_name";
constexpr StringLiteral VirtMapAttrName = "virtmap";
}

#endif // ALLO_OPS_H