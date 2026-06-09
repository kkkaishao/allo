# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar

import ml_dtypes
import numpy as np

from .utils import make_project_path, numpy_to_ctype

from ..lang.core import (
    APFloat,
    APInt,
    BufferType,
    DType,
    IndexType,
    StreamType,
    TypeBase,
    widen_apint_to_std,
)
from ..logging import stage, terminate_on_error
from .base import Backend, run_pipeline, set_top_llvm_c_wrapper
from ..lang.kernel import Kernel
from ..._mlir import ir
from ..._mlir.execution_engine import ExecutionEngine
from ..._mlir.runtime import (
    as_ctype,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)


@dataclass
class _CPUCompileCacheEntry:
    module: ir.Module
    engine: ExecutionEngine
    arg_types: list[TypeBase]
    res_types: list[TypeBase]


_DTYPE_TO_NP = {
    "bfloat16": ml_dtypes.bfloat16,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "index": np.int64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint1": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _find_first(paths: list[Path], stem: str) -> str | None:
    for lib_dir in paths:
        for suffix in (".dylib", ".so"):
            path = lib_dir / f"{stem}{suffix}"
            if path.exists():
                return str(path)
    return None


def _dataflow_runtime_lib() -> str:
    exp_dir = Path(__file__).resolve().parents[1]
    candidates = [
        exp_dir.parent / "_mlir" / "_mlir_libs",
        exp_dir.parents[1] / "build" / "lib",
    ]
    path = _find_first(candidates, "libAlloDataflowRuntime")
    if path is None:
        raise RuntimeError(
            "Cannot find liballo_dataflow_runtime. Rebuild Allo with `pip install -v -e .`."
        )
    return path


def _default_shared_libs() -> list[str]:
    llvm_base_dir = os.environ.get("LLVM_BASE_DIR")
    candidates = []
    if llvm_base_dir:
        candidates.append(Path(llvm_base_dir) / "lib")
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "externals"
        / "llvm-project"
        / "build"
        / "lib"
    )

    libs = []
    for lib_dir in candidates:
        found = []
        for stem in ("libmlir_runner_utils", "libmlir_c_runner_utils"):
            match = next(
                (
                    lib_dir / f"{stem}{suffix}"
                    for suffix in (".dylib", ".so")
                    if (lib_dir / f"{stem}{suffix}").exists()
                ),
                None,
            )
            if match is None:
                break
            found.append(str(match))
        if len(found) == 2:
            libs.extend(found)
            break
    return [*libs, _dataflow_runtime_lib()]


def _make_output_struct(memref_descriptors):
    fields = [
        (f"memref{i}", memref.__class__) for i, memref in enumerate(memref_descriptors)
    ]
    output_struct = type("OutputStruct", (ctypes.Structure,), {"_fields_": fields})()
    for i, memref in enumerate(memref_descriptors):
        setattr(output_struct, f"memref{i}", memref)
    return output_struct


def _pack_kernel_args(args, arg_types: list[TypeBase], res_types: list[TypeBase]):
    if len(args) != len(arg_types):
        raise ValueError(f"Expected {len(arg_types)} arguments, got {len(args)}")

    keepalive = []
    packed_args = []
    arg_arrays = []
    for arg, arg_type in zip(args, arg_types):
        ptr, obj, array = _pack_arg(arg, arg_type)
        packed_args.append(ptr)
        keepalive.append(obj)
        if array is not None:
            arg_arrays.append((arg, array))

    result_state = _pack_results(res_types)
    if result_state is None:
        return packed_args, keepalive, arg_arrays, None

    result_ptr, result_keepalive, result_decode = result_state
    keepalive.extend(result_keepalive)
    if len(res_types) == 1 and isinstance(res_types[0], DType):
        packed_args.append(result_ptr)
    else:
        packed_args.insert(0, result_ptr)
    return packed_args, keepalive, arg_arrays, result_decode


def _writeback_args(arg_arrays):
    for original, array in arg_arrays:
        if isinstance(original, np.ndarray) and original is not array:
            original[...] = _convert_back(array, original.dtype)


def _pack_arg(arg, arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        array = _as_array(arg, arg_type)
        desc = get_ranked_memref_descriptor(array)
        ptr = ctypes.pointer(ctypes.pointer(desc))
        return ptr, (array, desc, ptr), array

    if isinstance(arg_type, DType):
        value = _make_scalar(arg, arg_type)
        return value, value, None

    raise TypeError(f"Unsupported CPU argument type: {arg_type}")


def _pack_results(res_types: list[TypeBase]):
    if not res_types:
        return None

    if len(res_types) == 1 and isinstance(res_types[0], DType):
        scalar = _make_scalar(-1, res_types[0])
        return scalar, [scalar], lambda: scalar[0]

    descriptors = []
    keepalive = []
    for res_type in res_types:
        if not isinstance(res_type, BufferType):
            raise TypeError("Multiple CPU return values must be buffers")
        ctp = as_ctype(np.dtype(_numpy_dtype_for_dtype(res_type.dtype)))
        desc = make_nd_memref_descriptor(len(res_type.shape), ctp)()
        descriptors.append(desc)
        keepalive.append(desc)

    if len(descriptors) == 1:
        ptr = ctypes.pointer(ctypes.pointer(descriptors[0]))
        keepalive.append(ptr)
        return ptr, keepalive, lambda: ranked_memref_to_numpy(ptr[0])

    output = _make_output_struct(descriptors)
    ptr = ctypes.pointer(ctypes.pointer(output))
    keepalive.extend([output, ptr])
    return (
        ptr,
        keepalive,
        lambda: [
            ranked_memref_to_numpy(ctypes.pointer(getattr(ptr[0][0], f"memref{i}")))
            for i in range(len(descriptors))
        ],
    )


def _as_array(arg, buffer_type: BufferType):
    if not isinstance(arg, np.ndarray):
        raise TypeError("CPU buffer arguments must be numpy arrays")
    if tuple(arg.shape) != tuple(buffer_type.shape):
        raise ValueError(
            f"Expected buffer shape {tuple(buffer_type.shape)}, got {arg.shape}"
        )
    if not arg.flags["C_CONTIGUOUS"]:
        arg = np.ascontiguousarray(arg)

    np_dtype = _numpy_dtype_for_dtype(buffer_type.dtype)
    if arg.dtype != np_dtype:
        arg = arg.astype(np_dtype)
    return arg


def _make_scalar(value, dtype: DType):
    ctp = _ctype_for_dtype(dtype)
    if dtype.name == "float16":
        value = np.float16(value).view(np.int16)
    elif dtype.name == "bfloat16":
        value = ml_dtypes.bfloat16(value).view(np.int16)
    return (ctp * 1)(value)


def _numpy_dtype_for_dtype(dtype: DType):
    dtype = widen_apint_to_std(dtype)
    if dtype.name not in _DTYPE_TO_NP:
        _check_supported_dtype(dtype)
    return _DTYPE_TO_NP[dtype.name]


def _ctype_for_dtype(dtype: DType):
    return numpy_to_ctype(_numpy_dtype_for_dtype(dtype))


def _check_supported_dtype(dtype: DType):
    if isinstance(dtype, APInt) and dtype.primitive_width > 64:
        raise NotImplementedError("CPU backend does not support APInt > 64 bits yet")
    if isinstance(dtype, APFloat):
        raise NotImplementedError(f"CPU backend does not support {dtype.name}")
    if isinstance(dtype, IndexType):
        return
    raise TypeError(f"Unsupported CPU dtype: {dtype}")


def _convert_back(array, dtype):
    if dtype == np.dtype(np.float16):
        return array.view(np.float16)
    if dtype == ml_dtypes.bfloat16:
        return array.view(ml_dtypes.bfloat16)
    return array.astype(dtype, copy=False)


P = ParamSpec("P")
R = TypeVar("R")


class CPU(Backend, Generic[P, R]):
    """
    Backend for executing kernels on the CPU using LLVM's JIT compilation.

    This backend lowers the kernel to LLVMIR Dialect, compiles it using MLIR's ExecutionEngine (LLVM JIT),
    and executes it directly on the CPU. It supports buffer arguments as numpy arrays and scalar arguments
    as Python scalars.

    Currently the CPU backend does not support the tensor ABI, or arbitrary APInt/APFloat types
    """

    name = "cpu"

    def __init__(
        self,
        kernel: Kernel[P, R],
        *,
        opt_level: int = 2,
        shared_libs: list[str] = [],
    ):
        super().__init__(kernel)
        self.opt_level = opt_level
        self.shared_libs = _default_shared_libs()
        self.shared_libs.extend(shared_libs)
        self.engine: ExecutionEngine | None = None
        self.arg_types: list[TypeBase] = []
        self.res_types: list[TypeBase] = []

    @terminate_on_error
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.run(*args, **kwargs)

    @terminate_on_error
    def compile(self):
        if self.kernel.options.enable_tensor:
            raise NotImplementedError("CPU backend does not support tensor ABI yet")
        cache_key = self._cache_key(
            {
                "backend": self.name,
                "opt_level": self.opt_level,
                "shared_libs": self.shared_libs,
            }
        )
        cache = self._pcache_get("cpu.compile", cache_key)
        if cache is not None:
            self.module = cache.module
            self.engine = cache.engine
            self.arg_types = cache.arg_types
            self.res_types = cache.res_types
            return self.module
        else:
            cache = self._build_pcache(self.shared_libs)
            self._pcache_set("cpu.compile", cache_key, cache)
            self.module = cache.module
            self.engine = cache.engine
            self.arg_types = cache.arg_types
            self.res_types = cache.res_types
            return self.module

    def _build_pcache(self, shared_libs: list[str]) -> _CPUCompileCacheEntry:
        with stage("Compiling CPU Kernels"):
            arg_types = self.kernel.parse_argument_annotations()
            res_types = self.kernel.parse_return_annotation()
            if any(isinstance(ty, StreamType) for ty in arg_types):
                raise NotImplementedError(
                    "CPU backend does not support stream top-level arguments"
                )

            # Wrap a non-standard-width APInt boundary with a std-width interface
            # so the LLVM memref ABI is numpy-representable. No-op otherwise. Runs
            # before set_top_llvm_c_wrapper so the wrapper takes the public name.
            run_pipeline(
                self.module,
                "builtin.module(materialize-apint-wrapper{"
                f"top={self.kernel.func_name}}})",
            )
            if not set_top_llvm_c_wrapper(self.module, self.kernel.func_name):
                raise RuntimeError(
                    f"Cannot find top function '{self.kernel.func_name}'"
                )
            run_pipeline(self.module, "builtin.module(lower-to-llvm)")
            engine = ExecutionEngine(
                self.module,
                opt_level=self.opt_level,
                shared_libs=shared_libs,
            )
            return _CPUCompileCacheEntry(
                module=self.module,
                engine=engine,
                arg_types=arg_types,
                res_types=res_types,
            )

    @terminate_on_error
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self._ensure_compiled()
        packed_args, _, arg_arrays, result_decode = _pack_kernel_args(
            args, self.arg_types, self.res_types
        )
        with stage("Running CPU Kernels (JIT)"):
            assert self.engine is not None
            self.engine.invoke(self.kernel.func_name, *packed_args)
            _writeback_args(arg_arrays)
            if result_decode is None:
                return None  # type: ignore
            return result_decode()  # type: ignore

    @terminate_on_error
    def scaffold_project(
        self,
        project: str | None = None,
        *,
        exist_ok: bool = True,
    ) -> Path:
        project_path = make_project_path(project, self.kernel.func_name, exist_ok)
        self._ensure_compiled()
        assert self.module is not None
        (project_path / "lowered.mlir").write_text(str(self.module), encoding="utf-8")
        return project_path

    def _ensure_compiled(self):
        if self.engine is None:
            self.compile()
