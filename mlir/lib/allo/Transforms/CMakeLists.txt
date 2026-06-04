add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    ReuseCleanup.cpp
    MaterializeApintWrapper.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass
)
