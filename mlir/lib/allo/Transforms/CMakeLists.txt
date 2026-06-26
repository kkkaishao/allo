# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    DataflowSpawn.cpp
    ReuseCleanup.cpp
    FoldConstantCalls.cpp
    GenerateApintWrapper.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass
)
