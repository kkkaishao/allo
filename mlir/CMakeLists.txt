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

# OR-Tools backs the optional CP-SAT exact scheduler, and this find must come
# BEFORE find_package(CIRCT): a CIRCT built with OR-Tools names
# `ortools::ortools` in the link interface it exports for `CIRCTScheduling`, and
# CIRCTTargets.cmake fails outright if that imported target does not yet exist.
# So the find is unconditional rather than gated on the option below; the option
# only decides whether OUR exact scheduler is compiled. A genuinely
# OR-Tools-free distribution therefore needs an OR-Tools-free CIRCT beneath it.
#
# The option is OFF by default and never inferred from whether OR-Tools happens
# to be installed. What a distribution contains must not depend on the build
# machine, and a detected default is worse than useless here: `option()` caches
# its first value, so one configure that missed OR-Tools would silently pin the
# build tree to a scheduler-less wheel forever.
find_package(ortools CONFIG QUIET)
option(ALLO_ENABLE_ORTOOLS
       "Build the CP-SAT exact scheduler (requires OR-Tools)" OFF)
if(ALLO_ENABLE_ORTOOLS AND NOT ortools_FOUND)
  message(FATAL_ERROR
    "ALLO_ENABLE_ORTOOLS=ON but no OR-Tools cmake package was found. Point "
    "CMAKE_PREFIX_PATH at the install prefix holding lib/cmake/ortools (its "
    "config also resolves absl/protobuf from there), or set "
    "-DALLO_ENABLE_ORTOOLS=OFF.")
endif()

find_package(CIRCT CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
if(ALLO_ENABLE_ORTOOLS)
  message(STATUS "Using ortoolsConfig.cmake in: ${ortools_DIR}")
else()
  message(STATUS "OR-Tools disabled: the exact (CP-SAT) scheduler is not built")
endif()
if(ortools_FOUND)
  # Gated on the PACKAGE, not on ALLO_ENABLE_ORTOOLS: a CIRCT built with
  # OR-Tools links it into `CIRCTScheduling`, so the Python extensions pull a
  # shared libortools in transitively even when our exact scheduler is off, and
  # without its directory on their rpath they fail to dlopen. Turning our option
  # off does NOT give an OR-Tools-free build; only an OR-Tools-free CIRCT does.
  #
  # This is a DEVELOPER convenience either way: an absolute rpath into a local
  # prefix is not relocatable, so a build meant to be distributed has to link a
  # STATIC OR-Tools (or bundle the dylib) rather than rely on it.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

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

option(ALLO_STRIP_RELEASE_LIBS "Strip symbol tables from Release shared libs" ON)
if(ALLO_STRIP_RELEASE_LIBS AND
   (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-s")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)
