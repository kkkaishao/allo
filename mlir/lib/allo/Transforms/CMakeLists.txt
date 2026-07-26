# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTransforms
    MaterializeTopology.cpp
    GridMapping.cpp
    DataflowSpawn.cpp
    ReuseCleanup.cpp
    FoldConstantCalls.cpp
    GenerateApintWrapper.cpp
    FoldIfStatements.cpp
    FlattenPerfectLoops.cpp
    FlattenDcpMemref.cpp
    ResolveBanking.cpp
    PropagatePartition.cpp
    OutlineLooseProcesses.cpp
    PerfectizeLoopNest.cpp
    RaiseCountedWhile.cpp
    UnrollUnderPipeline.cpp
    ReassociateReductions.cpp
    RotateReductions.cpp
    LegalizeArith.cpp
    SdcScheduling.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAlloSupport
    MLIRAnalysis
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRArithTransforms
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTransforms
    MLIRTransformUtils
    MLIRPass
)
