set(LLVM_TARGET_DEFINITIONS AlloTransformOps.td)
mlir_tablegen(AlloTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AlloTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAlloTransformOpsIncGen)

add_dependencies(MLIRAlloIncGen MLIRAlloTransformOpsIncGen)