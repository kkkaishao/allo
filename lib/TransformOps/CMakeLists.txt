add_mlir_library(MLIRAlloTransformOps
    DialectExtension.cpp
    DataflowTransformation.cpp
    LoopTransformation.cpp
    RaiseToAffine.cpp
    Utils.cpp

    DEPENDS
    MLIRAlloIncGen
)