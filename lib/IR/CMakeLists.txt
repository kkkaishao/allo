add_mlir_dialect_library(MLIRAllo
    AlloOps.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)