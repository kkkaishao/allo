#ifndef ACT_OPS_H
#define ACT_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
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

#include "act/IR/ActAttrs.h"
#include "act/IR/ActTypes.h"

#include "act/IR/ActDialect.h.inc"

#include "act/IR/ActOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "act/IR/ActOps.h.inc"

#endif // ACT_OPS_H