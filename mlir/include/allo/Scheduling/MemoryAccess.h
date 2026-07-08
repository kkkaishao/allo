/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MEMORYACCESS_H
#define ALLO_SCHEDULING_MEMORYACCESS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::allo {

/// Whether an access targets an array (memref load/store) or a stream FIFO
/// (`allo.stream` get/put).
enum class AccessKind { Array, Stream };

/// A recognized memory access. `root` is the underlying buffer/stream SSA value
/// (view-like ops peeled), so distinct roots are distinct storage. `map` is the
/// affine subscript map (null for a non-affine `memref.load/store`); `indices`
/// are the subscript operands (array) or FIFO-select operands (stream).
struct MemAccess {
  Operation *op = nullptr;
  Value root;
  AccessKind kind = AccessKind::Array;
  bool isWrite = false;
  AffineMap map;
  llvm::SmallVector<Value, 4> indices;
};

/// Recognize \p op as a memory access (affine/memref load-store, or stream
/// get/put); nullopt if it is not one.
std::optional<MemAccess> asMemAccess(Operation *op);

/// Peel view-like ops (subview / cast / reinterpret_cast / view) to the
/// underlying buffer or stream root. Identity when \p v has no view-like def;
/// distinct roots are assumed non-aliasing (the Allo frontend has no pointers).
Value resolveRoot(Value v);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYACCESS_H
