# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(MLIRAlloConversion
    ConvertAlloToFunc.cpp
    ConvertLoopToSchedule.cpp
    LowerInstructions.cpp
    LowerDataflow.cpp
    LowerToLLVM.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    CIRCTLoopSchedule
    CIRCTScheduling
    MLIRAllo
    MLIRAlloSupport
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRTransformUtils
)
