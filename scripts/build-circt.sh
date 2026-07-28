#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

CIRCT_DIR=${1:-"externals/circt"}
LLVM_DIR=${2:-"externals/llvm-project/build"}
BUILD_TYPE=${3:-"Release"}
CC=${3:-"clang"}
CXX=${4:-"clang++"}
EXTRA_ARGS=${@:6}

mkdir -p $CIRCT_DIR/build

cd $CIRCT_DIR
./utils/get-or-tools.sh

cd $CIRCT_DIR/build
cmake -G Ninja ../ \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_DIR/lib/cmake/mlir \
  -DLLVM_USE_LINKER=lld \
  -DCIRCT_INCLUDE_TESTS=OFF \
  -DCIRCT_INCLUDE_DOCS=OFF \
  -DCIRCT_INCLUDE_INTEGRATION_TESTS=OFF \
  -DCIRCT_BINDINGS_PYTHON_ENABLED=OFF \
  $EXTRA_ARGS

ninja -j $(nproc)
