# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloScheduling
    DependenceAnalysis.cpp
    Footprint.cpp
    HierarchicalDependence.cpp
    MemoryAccess.cpp
    MemoryModel.cpp
    OperatorLibrary.cpp
    ProblemBuilder.cpp
    RegionGraph.cpp
    Scheduler.cpp

    DEPENDS
    MLIRAlloIncGen

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
