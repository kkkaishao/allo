#include "allo/TransformOps/AlloTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::allo;

#define GET_OP_CLASSES
#include "allo/TransformOps/AlloTransformOps.cpp.inc"

namespace {
class AlloTransformDialectExtension
    : public transform::TransformDialectExtension<
          AlloTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlloTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<allo::AlloDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<affine::AffineDialect>();
    declareDependentDialect<arith::ArithDialect>();
    declareDependentDialect<scf::SCFDialect>();
    declareDependentDialect<tensor::TensorDialect>();
    declareDependentDialect<vector::VectorDialect>();
    declareDependentDialect<memref::MemRefDialect>();
    declareDependentDialect<bufferization::BufferizationDialect>();
    declareDependentDialect<math::MathDialect>();
    declareDependentDialect<LLVM::LLVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "allo/TransformOps/AlloTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void allo::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AlloTransformDialectExtension>();
}

///===----------------------------------------------------------------------===//
/// RenameOp implementation
///===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::RenameOp::applyToOne(transform::TransformRewriter &rewriter,
                                Operation *target,
                                transform::ApplyToEachResultList &results,
                                transform::TransformState &state) {
  if (isa<SymbolOpInterface>(target)) {
    Operation *symTableOp = SymbolTable::getNearestSymbolTable(target);
    if (!symTableOp)
      return emitSilenceableError() << "cannot find symbol table for target";

    SymbolTable symTable(symTableOp);
    if (failed(symTable.rename(target, getName()))) {
      return emitSilenceableError() << "failed to rename symbol";
    }
  } else {
    target->setAttr("sym_name", getNameAttr());
  }
  return DiagnosedSilenceableFailure::success();
}