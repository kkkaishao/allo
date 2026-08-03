# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(MLIRAlloConversion
    ConvertAlloToFunc.cpp
    LowerDataflow.cpp
    LowerHLMem.cpp
    LowerToLLVM.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    CIRCTLoopSchedule
    CIRCTScheduling
    CIRCTHW
    CIRCTSV
    CIRCTSeq
    MLIRIR
    MLIRAllo
    MLIRAlloSupport
    MLIRBufferizationPipelines
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
    MLIRArithToLLVM
    MLIRTensorToLinalg
    MLIRControlFlowToLLVM
    MLIRBufferizationToMemRef
    MLIRReconcileUnrealizedCasts
)
