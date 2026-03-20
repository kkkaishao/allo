# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
import os
import sys

nanobind_dir = os.environ.get("nanobind_DIR", None)
if nanobind_dir is None:
    try:
        import nanobind
    except ImportError:
        raise RuntimeError(
            "nanobind is required to build the project. Please install it with `pip install nanobind`."
        )
    nanobind_dir = nanobind.cmake_dir()

llvm_build_dir = os.environ.get("LLVM_BASE_DIR", None)
if llvm_build_dir is None:
    default_build_dir = os.path.join(
        os.path.dirname(__file__), "externals", "llvm-project", "build"
    )
    if os.path.exists(default_build_dir):
        llvm_build_dir = default_build_dir
    else:
        raise RuntimeError(
            "LLVM build directory is not specified. Please set the LLVM_BASE_DIR environment variable to the path of your LLVM build directory."
        )

cmake_args = [
    f"-Dnanobind_DIR={nanobind_dir}",
    f"-DLLVM_DIR={llvm_build_dir}/lib/cmake/llvm",
    f"-DMLIR_DIR={llvm_build_dir}/lib/cmake/mlir",
    f"-DPython3_EXECUTABLE={sys.executable}",
]

setup(cmake_source_dir="./mlir", cmake_args=cmake_args)
