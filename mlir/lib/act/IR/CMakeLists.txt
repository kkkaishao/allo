add_mlir_dialect_library(MLIRAct
    ActOps.cpp
    ActTypes.cpp

    DEPENDS
    MLIRActIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRMemRefDialect
)