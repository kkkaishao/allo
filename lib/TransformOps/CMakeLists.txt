add_mlir_library(MLIRAlloTransformOps
    DialectExtension.cpp
    RaiseToAffine.cpp
    LoopTransformation.cpp
    Utils.cpp

    DEPENDS
    MLIRAlloIncGen
)