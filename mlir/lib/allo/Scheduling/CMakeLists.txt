# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set(ALLO_SCHEDULING_LIBS
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
if(ALLO_ENABLE_ORTOOLS)
  list(APPEND ALLO_SCHEDULING_LIBS ortools::ortools)
  set_source_files_properties(CPSATScheduler.cpp
      PROPERTIES COMPILE_DEFINITIONS ALLO_ENABLE_ORTOOLS)
endif()

add_mlir_library(MLIRAlloScheduling
    AddressCost.cpp
    CPSATScheduler.cpp
    DependenceAnalysis.cpp
    Footprint.cpp
    HierarchicalDependence.cpp
    MemoryAccess.cpp
    MemoryModel.cpp
    OperatorLibrary.cpp
    ProblemBuilder.cpp
    RegionGraph.cpp
    Scheduler.cpp
    PreVerification.cpp
    SDC.cpp
    PostConversion.cpp

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    ${ALLO_SCHEDULING_LIBS}
)
