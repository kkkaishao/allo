/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORIDENTITY_H
#define ALLO_SCHEDULING_OPERATORIDENTITY_H

#include "allo/IR/AlloOps.h" // dcp::DCPathComputeOp

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::allo {

class OperatorLibrary;

/// What one physical operator is: two operations share an identity exactly when
/// one unit can run both. The library's second key, finer than
/// `OperatorChar::typeName`, which names a timing row.
struct OperatorIdentity {
  /// The realization: a `dcp.operator` symbol (IP path) or a `CombOpKind`
  /// mnemonic (native path). Empty when no functional unit is built for the
  /// operation: a memory or stream access, a literal, a call.
  std::string realization;
  bool comb = false;                   // `realization` names a CombOpKind
  llvm::SmallVector<Type, 2> argTypes; // operand types, so width is in here
  Type resultType;
  Attribute predicate; // a compare's `predicate`; null otherwise
  Attribute map;       // an `affine.apply`'s `map`; null otherwise

  /// Whether an operation of this identity gets a functional unit.
  bool realized() const { return !realization.empty(); }

  bool operator==(const OperatorIdentity &o) const {
    return comb == o.comb && realization == o.realization &&
           llvm::ArrayRef<Type>(argTypes) == llvm::ArrayRef<Type>(o.argTypes) &&
           resultType == o.resultType && predicate == o.predicate &&
           map == o.map;
  }
  bool operator!=(const OperatorIdentity &o) const { return !(*this == o); }

  /// A stable string spelling, for map keys and reports. Not an RTL name:
  /// `Naming.h` owns those (`operatorModuleName`).
  std::string key() const;
};

/// The identity of a reified compute op, which carries its own realization.
OperatorIdentity operatorIdentity(dcp::DCPathComputeOp comp);

/// The identity \p lib resolves for \p op; empty when \p op has no realization.
/// Dispatches to the overload above for an already-reified op.
OperatorIdentity operatorIdentity(Operation *op, const OperatorLibrary &lib);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORIDENTITY_H
