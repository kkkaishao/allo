# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloScheduling
    DependenceAnalysis.cpp
    OperatorModel.cpp
    ProblemBuilder.cpp
    RegionGraph.cpp
    ScheduleResult.cpp
    AlloSchedulePass.cpp

    DEPENDS
    MLIRAlloIncGen
    MLIRAlloSchedulingPassIncGen

    LINK_LIBS PUBLIC
    CIRCTScheduling
    MLIRAllo
    MLIRAlloSupport
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRAffineUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRIR
    MLIRTransformUtils
)
