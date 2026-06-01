# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Assembles the `allo._mlir` Python package on top of the upstream MLIR Python
# bindings (single MLIR instance via the aggregate CAPI dylib). This is additive
# and lives alongside the legacy hand-written `_liballo` extension in ../python.

include(AddMLIRPython)

# Vendor our own relocatable MLIR instance under the `allo._mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=allo._mlir.")

set(ALLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/allo")
set(ALLO_MLIR_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/allo/Bindings")
# Assembled (symlinked in dev) into the source tree so `import allo._mlir` works
# from an editable install, mirroring how ../python emits into allo/exp/_C.
set(ALLO_MLIR_PYTHON_PACKAGE_DIR "${PROJECT_SOURCE_DIR}/../allo/_mlir")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AlloMLIRPythonSources)

declare_mlir_python_sources(AlloMLIRPythonSources.Core
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT AlloMLIRPythonSources
  SOURCES
    schedule.py
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AlloOps.td
  SOURCES
    dialects/allo.py
  DIALECT_NAME allo
)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AlloTransformOps.td
  SOURCES
    dialects/transform/allo.py
  DIALECT_NAME transform
  EXTENSION_NAME transform_allo_extension
)

################################################################################
# Extension: the slim CAPI driver module `_allo`
################################################################################

declare_mlir_python_extension(AlloMLIRPythonSources.Extension
  MODULE_NAME _allo
  ADD_TO_PARENT AlloMLIRPythonSources
  ROOT_DIR "${ALLO_MLIR_PYTHON_SOURCE_DIR}"
  SOURCES
    AlloModule.cpp
    AlloTypes.cpp
    AlloAttrs.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRAlloCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Common CAPI aggregate (single shared MLIR instance)
################################################################################

# RegisterEverything owns ALL upstream dialect/pass/extension registration (run
# at site-init) and provides the full CAPI breadth the core/dialect extensions
# need at runtime. MLIRAlloCAPI registration is therefore strictly ADDITIVE
# (allo dialect/extension/passes only) so the two never double-register.
set(_allo_declared_sources
  AlloMLIRPythonSources
  MLIRPythonSources
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(AlloMLIRAggregateCAPI
  INSTALL_COMPONENT AlloMLIRPythonModules
  INSTALL_DESTINATION "allo/_mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${ALLO_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES ${_allo_declared_sources}
)

################################################################################
# Instantiate the package
################################################################################

add_mlir_python_modules(AlloMLIRPythonModules
  ROOT_PREFIX "${ALLO_MLIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "allo/_mlir"
  MLIR_BINDINGS_PYTHON_NB_DOMAIN "allo"
  DECLARED_SOURCES ${_allo_declared_sources}
  COMMON_CAPI_LINK_LIBS
    AlloMLIRAggregateCAPI
)
