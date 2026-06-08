# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
cmake_policy(SET CMP0116 NEW)

project(allo-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Locate the bundled LLVM/MLIR build. Honor an explicit -DLLVM_DIR/-DMLIR_DIR,
# else derive both from LLVM_BASE_DIR (the $LLVM_BASE_DIR env var, falling back
# to the in-tree externals build).
if(NOT DEFINED LLVM_DIR OR NOT DEFINED MLIR_DIR)
  if(NOT DEFINED LLVM_BASE_DIR)
    if(DEFINED ENV{LLVM_BASE_DIR})
      set(LLVM_BASE_DIR "$ENV{LLVM_BASE_DIR}")
    else()
      set(LLVM_BASE_DIR "${PROJECT_SOURCE_DIR}/../externals/llvm-project/build")
    endif()
  endif()
  if(NOT EXISTS "${LLVM_BASE_DIR}/lib/cmake/mlir")
    message(FATAL_ERROR "MLIR cmake package not found under '${LLVM_BASE_DIR}'. "
                        "Build externals/llvm-project or set -DLLVM_BASE_DIR / $LLVM_BASE_DIR.")
  endif()
  set(LLVM_DIR "${LLVM_BASE_DIR}/lib/cmake/llvm")
  set(MLIR_DIR "${LLVM_BASE_DIR}/lib/cmake/mlir")
endif()

find_package(MLIR CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

message(STATUS "Using Python binding")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
