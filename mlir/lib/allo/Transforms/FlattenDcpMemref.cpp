/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::allo {
#define GEN_PASS_DEF_FLATTENDCPMEMREFPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

bool hasDivMod(AffineExpr e) {
  bool found = false;
  e.walk([&](AffineExpr sub) {
    switch (sub.getKind()) {
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Mod:
      found = true;
      break;
    default:
      break;
    }
  });
  return found;
}

// Compose the memref's row-major linearization into the (possibly delinearized)
// access map, yielding a single linear index. For a coalesced nest the map is
// `iv -> (iv floordiv N, iv mod N)`; the linearization `(r, c) -> r*N + c`
// composes to `(iv floordiv N)*N + iv mod N`, which `simplifyAffineExpr` folds
// back to `iv` (mod expands to `iv - (iv floordiv N)*N`, so the terms cancel).
AffineMap linearize(AffineMap map, ArrayRef<int64_t> shape) {
  unsigned rank = map.getNumResults();
  SmallVector<int64_t> stride(rank, 1);
  for (int k = static_cast<int>(rank) - 2; k >= 0; --k)
    stride[k] = stride[k + 1] * shape[k + 1];
  AffineExpr lin = getAffineConstantExpr(0, map.getContext());
  for (unsigned k = 0; k < rank; ++k)
    lin = lin + map.getResult(k) * stride[k];
  lin = simplifyAffineExpr(lin, map.getNumDims(), map.getNumSymbols());
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), lin,
                        map.getContext());
}

// Linearize one dcp memory access whose address map carries a floordiv/mod.
// The delinearize/linearize pair cancels to a plain index when the access spans
// the whole coalesced range (`A[iv floordiv N, iv mod N]` -> `A[iv]`); when it
// spans only a subset (`A[iv floordiv N, k]`) a power-of-two floordiv/mod
// survives, which the emitter lowers to a shift / mask (a non-power-of-two
// divisor is the banking pass's divide). A plain multi-dim map (no
// floordiv/mod) is left for the emitter to linearize.
template <class OpT> void flattenAccess(OpT op) {
  AffineMap map = op.getMap();
  if (llvm::none_of(map.getResults(), hasDivMod))
    return;
  auto shape = cast<MemRefType>(op.getMemref().getType()).getShape();
  op.setMapAttr(AffineMapAttr::get(linearize(map, shape)));
}

struct FlattenDcpMemrefPass
    : public allo::impl::FlattenDcpMemrefPassBase<FlattenDcpMemrefPass> {
  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
        flattenAccess(l);
      else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
        flattenAccess(s);
    });
  }
};

} // namespace
