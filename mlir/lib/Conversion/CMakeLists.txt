add_mlir_conversion_library(MLIRAlloConversion
    ConvertInstructionToCanonicalForm.cpp
    LowerToLLVM.cpp
    Passes.cpp

    DEPENDS
    MLIRAlloIncGen
)