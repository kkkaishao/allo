add_mlir_library(MLIRActSupport
    CodeEmission.cpp
    Planning.cpp
    SemanticMatching.cpp
    SymbolicExpr.cpp
    TilingAnalysis.cpp

    DEPENDS
    MLIRActIncGen

    LINK_LIBS PUBLIC
    MLIRAct
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRIR
)
