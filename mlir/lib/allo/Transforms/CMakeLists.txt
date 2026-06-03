add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    ReuseCleanup.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass
)
