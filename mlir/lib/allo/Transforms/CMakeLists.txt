add_mlir_library(MLIRAlloTransforms
    Passes.cpp
    LocalizeGlobalStream.cpp

    DEPENDS
    MLIRAlloIncGen
)
