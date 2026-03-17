/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_VIVADOHLSEMITTER_H
#define ALLO_TRANSLATION_VIVADOHLSEMITTER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Translation/EmitterState.h"

namespace mlir::allo {

struct VivadoHLSEmitter {
  explicit VivadoHLSEmitter(llvm::raw_ostream &os) : state(os) {}

  void emitFunction(func::FuncOp func);
  void emitCall(func::CallOp op);
  void emitReturn(func::ReturnOp op);

  void emitAffineFor(affine::AffineForOp op);
  void emitAffineLoad(affine::AffineLoadOp op);
  void emitAffineStore(affine::AffineStoreOp op);
  void emitAffineIf(affine::AffineIfOp op);
  void emitAffineYield(affine::AffineYieldOp op);
  void emitAffineApply(affine::AffineApplyOp op);

  void emitMemrefAlloc(memref::AllocOp op);
  void emitMemrefAlloca(memref::AllocaOp op);
  void emitMemrefLoad(memref::LoadOp op);
  void emitMemrefStore(memref::StoreOp op);
  void emitMemrefGlobal(memref::GlobalOp op);
  void emitMemrefGetGlobal(memref::GetGlobalOp op);

  void emitFor(scf::ForOp op);
  void emitIf(scf::IfOp op);
  void emitWhile(scf::WhileOp op);
  void emitCondition(scf::ConditionOp op);
  void emitSCFYield(scf::YieldOp op);

  void emitSelect(arith::SelectOp op);
  void emitConstant(arith::ConstantOp op);
  void emitCmpI(arith::CmpIOp op);
  void emitCmpF(arith::CmpFOp op);

  void emitModule(ModuleOp);

  EmitterState state;

private:
  void emitBlock(Block &block);
  void emitValue(Value val);
  void emitFunctionArguments(func::FuncOp func);
  void emitFunctionReturnType(func::FuncOp func);
  void emitFunctionDirectives(func::FuncOp func);
  void emitPartitionAttr(allo::PartitionAttr attr, Value value);
  void emitLoopDirectives(Operation *op);
  void emitBinaryOp(Operation *op, llvm::StringLiteral keyword);
  void emitMaxMinOp(Operation *op, llvm::StringLiteral keyword);
  void emitUnaryOp(Operation *op, llvm::StringLiteral keyword);
  void emitCastOp(Operation *op);
  std::string getPrimitiveTypeName(Type type);

  void dispatch(Operation *op);
};

struct AffineExprEmitter : public mlir::AffineExprVisitor<AffineExprEmitter> {
  explicit AffineExprEmitter(EmitterState &state, OperandRange operands)
      : state(state), operands(operands) {}

  void visitDimExpr(AffineDimExpr expr) {
    state.os << state.getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    state.os << state.getName(operands[expr.getPosition()]);
  }
  void visitConstantExpr(AffineConstantExpr expr) {
    state.os << expr.getValue();
  }
  void visitAddExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) { visitAffineBinExpr(expr, "%"); }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    visitAffineBinExpr(expr, "/");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    state.os << "((";
    visit(expr.getLHS());
    state.os << " + ";
    visit(expr.getRHS());
    state.os << " - 1) / ";
    visit(expr.getRHS());
    state.os << ")";
  }
  void emitAffineMap(AffineMap map) {
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      if (i > 0)
        state.os << ", ";
      visit(map.getResult(i));
    }
  }

private:
  EmitterState &state;
  OperandRange operands;
  void visitAffineBinExpr(AffineBinaryOpExpr expr, llvm::StringLiteral op) {
    state.os << "(";
    visit(expr.getLHS());
    state.os << " " << op << " ";
    visit(expr.getRHS());
    state.os << ")";
  }
};

void registerVivadoHLSTranslation();

} // namespace mlir::allo

#endif // ALLO_TRANSLATION_VIVADOHLSEMITTER_H
