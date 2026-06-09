# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    DataflowSpawn.cpp
    ReuseCleanup.cpp
    MaterializeApintWrapper.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass
)
