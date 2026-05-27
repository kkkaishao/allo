add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
)
