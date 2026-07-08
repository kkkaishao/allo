# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
cmake_policy(SET CMP0116 NEW)

project(Allo LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# marl: lightweight fiber-based task scheduler that backs the CPU dataflow
# simulator runtime (PEs run as cooperatively-scheduled fibers). Built static
# and PIC; we only need the `marl` library target, so skip its aux targets.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/marl/CMakeLists.txt")
  message(FATAL_ERROR
    "externals/marl is missing. Run: git submodule update --init externals/marl")
endif()
set(MARL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MARL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MARL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/marl EXCLUDE_FROM_ALL)

# spdlog: use header-only mode to avoid rtti, exceptions, and shared lib dependencies
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/externals/spdlog/CMakeLists.txt")
  message(FATAL_ERROR
    "externals/spdlog is missing. Run: git submodule update --init externals/spdlog")
endif()
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/spdlog EXCLUDE_FROM_ALL)

add_subdirectory(mlir)
