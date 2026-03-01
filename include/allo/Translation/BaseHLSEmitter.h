#ifndef ALLO_BASEHLSEMITTER_H
#define ALLO_BASEHLSEMITTER_H

#include "allo/IR/AlloOps.h"
#include "allo/Translation/BaseEmitter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::allo {

struct BaseHLSEmitter : public AlloBaseEmitter {
  using AlloBaseEmitter::AlloBaseEmitter;

  /// SCF statement emitters.
  virtual void emitScfFor(scf::ForOp op) = 0;
  virtual void emitScfIf(scf::IfOp op) = 0;
  virtual void emitScfWhile(scf::WhileOp op) = 0;
  virtual void emitScfCondition(scf::ConditionOp op) = 0;
  virtual void emitScfYield(scf::YieldOp op) = 0;
  virtual void emitScfParallel(scf::ParallelOp op) = 0;

  /// Affine statement emitters.
  virtual void emitAffineFor(affine::AffineForOp op) = 0;
  virtual void emitAffineIf(affine::AffineIfOp op) = 0;
  virtual void emitAffineParallel(affine::AffineParallelOp op) = 0;
  virtual void emitAffineApply(affine::AffineApplyOp op) = 0;
  virtual void emitAffineLoad(affine::AffineLoadOp op) = 0;
  virtual void emitAffineStore(affine::AffineStoreOp op) = 0;
  virtual void emitAffineYield(affine::AffineYieldOp op) = 0;

  /// MemRef-related statement emitters.
  virtual void emitLoad(memref::LoadOp op) = 0;
  virtual void emitStore(memref::StoreOp op) = 0;
  virtual void emitGetGlobal(memref::GetGlobalOp op) = 0;
  virtual void emitGlobal(memref::GlobalOp op) = 0;
  virtual void emitSubView(memref::SubViewOp op) = 0;
  virtual void emitReshape(memref::ReshapeOp op) = 0;

  /// Special operation emitters.
  virtual void emitCall(allo::CallOp op) {}
  virtual void emitSelect(arith::SelectOp op) {}
  virtual void emitConstant(arith::ConstantOp op) {}

  /// Top-level MLIR module emitter.
  virtual void emitModule(ModuleOp module) {}
};

} // namespace mlir::allo

#endif // ALLO_BASEHLSEMITTER_H
