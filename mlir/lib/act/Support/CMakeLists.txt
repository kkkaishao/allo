add_mlir_library(MLIRActSupport
    SemanticMatching.cpp

    DEPENDS
    MLIRActIncGen

    LINK_LIBS PUBLIC
    MLIRAct
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRIR
)
