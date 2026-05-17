add_mlir_conversion_library(MLIRAlloConversion
    ConvertAlloToFunc.cpp
    LowerDataflow.cpp
    LowerToLLVM.cpp
    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAlloIncGen

    LINK_LIBS PUBLIC
    MLIRAllo
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIROpenMPDialect
    MLIROpenMPToLLVM
    MLIRTransformUtils
)

add_library(allo_dataflow_runtime SHARED
    DataflowRuntime.cpp
)

set_target_properties(allo_dataflow_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../allo/exp/_C"
)
