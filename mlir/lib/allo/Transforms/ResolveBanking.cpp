/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr
#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h" // partitionOf, staticBank
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RESOLVEBANKINGPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// Address map of a dcp memory access.
AffineMap accessMap(Operation *op) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return l.getMap();
  return cast<dcp::DCPathStoreOp>(op).getMap();
}

void rewriteAccess(Operation *op, Value bank, AffineMap localMap) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    l.getMemrefMutable().assign(bank);
    l.setMapAttr(AffineMapAttr::get(localMap));
  } else {
    auto s = cast<dcp::DCPathStoreOp>(op);
    s.getMemrefMutable().assign(bank);
    s.setMapAttr(AffineMapAttr::get(localMap));
  }
}

// The static bank id (mixed-radix over the partition axes) of \p op, and its
// in-bank address map (each partitioned dim's subscript floordiv its factor).
// nullopt if any axis is not statically banked.
struct BankedAccess {
  unsigned bank;
  AffineMap localMap;
};
std::optional<BankedAccess>
resolve(Operation *op, ArrayRef<std::pair<unsigned, int64_t>> axes) {
  AffineMap map = accessMap(op);
  SmallVector<AffineExpr> results(map.getResults());
  unsigned bank = 0;
  for (auto [dim, factor] : axes) {
    std::optional<int64_t> b = staticBank(map, dim, factor);
    if (!b)
      return std::nullopt;
    bank = bank * factor + static_cast<unsigned>(*b);
    AffineExpr div =
        getAffineBinaryOpExpr(AffineExprKind::FloorDiv, results[dim],
                              getAffineConstantExpr(factor, map.getContext()));
    results[dim] =
        simplifyAffineExpr(div, map.getNumDims(), map.getNumSymbols());
  }
  return BankedAccess{bank,
                      AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                     results, map.getContext())};
}

// Split one internal partitioned alloc into per-bank allocs if all its accesses
// are static; return true if it was split.
bool splitAlloc(Operation *alloc) {
  Value memref = alloc->getResult(0);
  PartitionInfo p = partitionOf(memref);
  // Cyclic-only, refinable partition; block/complete fall back to the
  // aggregate model in the scheduler too (`MemoryBankModel::finalize`).
  if (p.unlimited || p.cyclicAxes.empty() || p.hasBlock)
    return false;

  SmallVector<Operation *> accesses;
  SmallVector<BankedAccess> banked;
  for (Operation *user : memref.getUsers()) {
    // A non-load/store use (e.g. the memref escaping) cannot be split safely.
    if (!isa<dcp::DCPathLoadOp, dcp::DCPathStoreOp>(user))
      return false;
    std::optional<BankedAccess> b = resolve(user, p.cyclicAxes);
    if (!b) {
      warn(Stage::Dcp, alloc) << "partitioned array has a data-dependent bank; "
                                 "left for the emitter "
                                 "crossbar";
      return false;
    }
    accesses.push_back(user);
    banked.push_back(*b);
  }

  auto mt = cast<MemRefType>(memref.getType());
  ArrayRef<int64_t> shape = mt.getShape();
  SmallVector<int64_t> bankShape(shape);
  for (auto [dim, factor] : p.cyclicAxes)
    bankShape[dim] = (shape[dim] + factor - 1) / factor; // ceil per bank
  auto bankType = MemRefType::get(bankShape, mt.getElementType());

  OpBuilder b(alloc);
  SmallVector<Value> banks;
  for (unsigned k = 0; k < p.factor; ++k) {
    Operation *bankAlloc =
        isa<memref::AllocaOp>(alloc)
            ? memref::AllocaOp::create(b, alloc->getLoc(), bankType)
                  .getOperation()
            : memref::AllocOp::create(b, alloc->getLoc(), bankType)
                  .getOperation();
    // Carry every attribute except the partition (a bank *is* one physical
    // memory); keeps bind.storage / the buffer NameLoc for emit naming.
    for (NamedAttribute attr : alloc->getAttrs())
      if (attr.getName() != kPartitionAttr)
        bankAlloc->setAttr(attr.getName(), attr.getValue());
    banks.push_back(bankAlloc->getResult(0));
  }

  for (auto [op, ba] : llvm::zip_equal(accesses, banked))
    rewriteAccess(op, banks[ba.bank], ba.localMap);
  alloc->erase();
  return true;
}

struct ResolveBankingPass
    : public allo::impl::ResolveBankingPassBase<ResolveBankingPass> {
  void runOnOperation() override {
    SmallVector<Operation *> allocs;
    getOperation().walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op) &&
          op->hasAttr(kPartitionAttr))
        allocs.push_back(op);
    });
    for (Operation *alloc : allocs)
      splitAlloc(alloc);
  }
};

} // namespace
