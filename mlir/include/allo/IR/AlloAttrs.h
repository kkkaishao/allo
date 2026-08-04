/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ATTRS_H
#define ALLO_ATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

#include "allo/IR/AlloEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.h.inc"

namespace mlir::allo {

/// What the resource vector \p uses spends at \p params: one entry per named
/// resource, each the PRODUCT of its factors with factor `i` evaluated at
/// `params[i]`, rounded to the nearest whole resource ONCE at the end.
///
/// \p uses is a `dcp.resource`-referencing `ResourceUseAttr` array, and
/// \p params is the parameter tuple of the realization's kind (an operator's
/// operand width; a multiplexer's fan-in and width). A null \p uses spends
/// nothing, which is what an undeclared cost means.
llvm::SmallVector<std::pair<mlir::SymbolRefAttr, int64_t>>
evaluateResourceUse(mlir::ArrayAttr uses, llvm::ArrayRef<int64_t> params);

} // namespace mlir::allo

#endif // ALLO_ATTRS_H
