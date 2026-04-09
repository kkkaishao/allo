set(LLVM_TARGET_DEFINITIONS ActOps.td)
mlir_tablegen(ActDialect.h.inc -gen-dialect-decls -dialect=act)
mlir_tablegen(ActDialect.cpp.inc -gen-dialect-defs -dialect=act)
mlir_tablegen(ActOps.h.inc -gen-op-decls)
mlir_tablegen(ActOps.cpp.inc -gen-op-defs)
mlir_tablegen(ActOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ActOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRActOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ActTypes.td)
mlir_tablegen(ActTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ActTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(ActTypesInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(ActTypesInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRActTypesIncGen)

set(LLVM_TARGET_DEFINITIONS ActAttrs.td)
mlir_tablegen(ActAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(ActAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRActAttrsIncGen)

add_custom_target(MLIRActIncGen
    DEPENDS
    MLIRActOpsIncGen
    MLIRActTypesIncGen
    MLIRActAttrsIncGen
)