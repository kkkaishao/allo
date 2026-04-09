#ifndef ACT_TYPES_H
#define ACT_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
class FlatSymbolRefAttr;
} // namespace mlir

#include "act/IR/ActTypesInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "act/IR/ActTypes.h.inc"

#endif // ACT_TYPES_H
