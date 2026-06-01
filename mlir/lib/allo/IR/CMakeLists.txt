add_mlir_dialect_library(MLIRAllo
    AlloOps.cpp
    AlloISATypes.cpp
    AlloISAOps.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRAffineDialect
)