add_mlir_library(MLIRActSupport
    CodeEmission.cpp
    SemanticMatching.cpp
    SymbolicExpr.cpp
    ParamSolving.cpp
    Planning.cpp

    DEPENDS
    MLIRActIncGen

    LINK_LIBS PUBLIC
    MLIRAct
    MLIRIR
)
