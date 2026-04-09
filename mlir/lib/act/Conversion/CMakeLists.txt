add_mlir_conversion_library(MLIRActConversion
    Passes.cpp
    ConvertActToCanonicalForm.cpp
    LowerToLLVM.cpp

    DEPENDS
    MLIRActIncGen
)
