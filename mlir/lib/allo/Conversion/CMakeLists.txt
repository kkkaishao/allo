add_mlir_conversion_library(MLIRAlloConversion
    ConvertAlloToFunc.cpp
    LowerToLLVM.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAllo
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRTransformUtils
)
