# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Locate the bundled LLVM/MLIR build. Honor an explicit -DLLVM_DIR/-DMLIR_DIR,
# else derive both from LLVM_BASE_DIR (the $LLVM_BASE_DIR env var, falling back
# to the in-tree externals build).
if(NOT DEFINED LLVM_DIR OR NOT DEFINED MLIR_DIR)
  set(LLVM_BASE_DIR "${PROJECT_SOURCE_DIR}/externals/llvm-project/build")
  if(NOT EXISTS "${LLVM_BASE_DIR}/lib/cmake/mlir")
    message(FATAL_ERROR "MLIR cmake package not found under '${LLVM_BASE_DIR}'. "
                        "Build externals/llvm-project or set -DLLVM_DIR/-DMLIR_DIR to the cmake package locations.")
  endif()
  set(LLVM_DIR "${LLVM_BASE_DIR}/lib/cmake/llvm")
  set(MLIR_DIR "${LLVM_BASE_DIR}/lib/cmake/mlir")
endif()

if(NOT DEFINED CIRCT_DIR)
  set(CIRCT_DIR "${PROJECT_SOURCE_DIR}/externals/circt/build/lib/cmake/circt")
  if(NOT EXISTS ${CIRCT_DIR})
    message(FATAL_ERROR "CIRCT not found under '${CIRCT_DIR}'. "
                        "Build externals/circt or set -DCIRCT_DIR / $CIRCT_DIR.")
  endif()
endif()

find_package(MLIR CONFIG REQUIRED)
find_package(CIRCT CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCIRCT)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

message(STATUS "Using Python binding")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# Python type stubs (.pyi) aid editor/IDE completion but add build time.
# TEMPORARILY DISABLED: bare-name stubgen of the `_allo` extension re-imports
# the core `_mlir` under its FQN and double-registers its enums. Re-enable once
# the CRTP-migration/stubgen interaction is resolved (was: default ON under
# Debug / RelWithDebInfo). Honors an explicit -DMLIR_PYTHON_STUBGEN_ENABLED.
if(NOT DEFINED MLIR_PYTHON_STUBGEN_ENABLED)
  set(MLIR_PYTHON_STUBGEN_ENABLED OFF)
endif()
message(STATUS "MLIR_PYTHON_STUBGEN_ENABLED: ${MLIR_PYTHON_STUBGEN_ENABLED}")

option(ALLO_STRIP_RELEASE_LIBS "Strip symbol tables from Release shared libs" ON)
if(ALLO_STRIP_RELEASE_LIBS AND
   (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-s")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
