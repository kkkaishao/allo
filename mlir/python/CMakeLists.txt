# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

nanobind_add_module(_liballo
    NB_STATIC STABLE_ABI
    ir.cpp
    ops.cpp
    main.cpp
    transform.cpp
    passes.cpp
    execution_engine.cpp
)

set(NANOBIND_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/../allo/exp/_C")

# turn off -Wcast-qual for nanobind
target_compile_options(nanobind-static
    PUBLIC
    "-Wno-cast-qual"
    "-Wno-covered-switch-default"
)

set_target_properties(_liballo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${NANOBIND_OUTPUT_DIR}
    CXX_VISIBILITY_PRESET hidden
)

set(ALLO_PYTHON_MLIR_LIBS
    # Core IR / parser / pass infra used by python bindings.
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRAnalysis
    MLIRDialect
    MLIRDialectUtils
    MLIRBytecodeWriter

    # Dialects directly exposed in mlir/python.
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIndexDialect
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIROpenMPDialect
    MLIRSCFDialect
    MLIRShapeDialect
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRTransformDialect
    MLIRUBDialect
    MLIRVectorDialect

    # Transform dialect extensions used by load_transform_dialects().
    MLIRAffineTransformOps
    MLIRBufferizationTransformOps
    MLIRFuncTransformOps
    MLIRLinalgTransformOps
    MLIRMemRefTransformOps
    MLIRSCFTransformOps
    MLIRTensorTransformOps
    MLIRTransformLoopExtension
    MLIRTransformDialectInterfaces
    MLIRTransformDialectTransforms
    MLIRTransformDialectUtils
    MLIRTransformUtils
    MLIRVectorTransformOps

    # Conversions
    MLIRTensorToLinalg
    MLIRBufferizationToMemRef
    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRMathToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRTosaToLinalg
    MLIRIndexToLLVM
    MLIRSCFToControlFlow
    MLIRLinalgToStandard
    MLIRConvertToLLVMPass

    # Passes
    MLIRBufferizationPipelines
    MLIRTensorTransforms
    MLIRLLVMIRTransforms
    MLIRAffineTransforms
    MLIRTargetLLVMIRTransforms
    MLIRMathTransforms
    MLIROpenMPTransforms
    MLIRShapeOpsTransforms

    # Execution engine
    MLIRCAPIExecutionEngine
    MLIROpenMPToLLVMIRTranslation
)

target_link_libraries(_liballo
    PRIVATE
    ${ALLO_PYTHON_MLIR_LIBS}

    MLIRAllo
    MLIRAlloTransformOps
    MLIRAlloTranslation
)

target_link_options(_liballo PRIVATE ${LLVM_LDFLAGS})
