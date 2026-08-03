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
/// (`resolveRoot`, in `Support/AliasAnalysis.h`), so distinct roots are
/// distinct storage. `map` is the element-space subscript map, one result per
/// memref dimension, and an ARRAY access always has one: a non-affine
/// `memref.load/store` carries the identity map over its indices, so every
/// consumer sees one encoding. `indices` are the subscript operands (array) or
/// FIFO-select operands (stream); a stream has no map. Whether an access is
/// AFFINE is a question about the op
/// (`affine::AffineReadOpInterface`), not about the map.
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

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYACCESS_H
