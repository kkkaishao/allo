add_mlir_library(MLIRAlloSupport
    TopologyGraph.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAllo
    MLIRIR
)
