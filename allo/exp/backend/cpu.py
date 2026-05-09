# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from .utils import make_project_path

from ..lang.core import APFloat, APInt, BufferType, DType, IndexType, TypeBase
from ..logging import stage, terminate_on_error
from .base import Backend


@dataclass
class _CPUCompileCacheEntry:
    module_owner: Any
    module: Any
    engine: Any
    arg_types: list[TypeBase]
    res_types: list[TypeBase]


class _F16(ctypes.Structure):
    _fields_ = [("f16", ctypes.c_int16)]


class _BF16(ctypes.Structure):
    _fields_ = [("bf16", ctypes.c_int16)]


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

_DTYPE_TO_CTYPE = {
    "bfloat16": ctypes.c_int16,
    "float16": ctypes.c_int16,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
    "index": ctypes.c_int64,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint1": ctypes.c_bool,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
}


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
    return libs


def _make_nd_memref_descriptor(rank: int, dtype):
    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_longlong * rank),
            ("strides", ctypes.c_longlong * rank),
        ]

    return MemRefDescriptor


def _get_ranked_memref_descriptor(array: np.ndarray):
    ctp = _as_ctype(array.dtype)
    desc = _make_nd_memref_descriptor(array.ndim, ctp)()
    desc.allocated = array.ctypes.data
    desc.aligned = array.ctypes.data_as(ctypes.POINTER(ctp))
    desc.offset = 0
    desc.shape = array.ctypes.shape
    strides_t = ctypes.c_longlong * array.ndim
    desc.strides = strides_t(*[stride // array.itemsize for stride in array.strides])
    return desc


def _ranked_memref_to_numpy(desc):
    content_ptr = ctypes.cast(
        ctypes.addressof(desc.aligned.contents)
        + desc.offset * ctypes.sizeof(desc.aligned.contents),
        type(desc.aligned),
    )
    array = np.ctypeslib.as_array(content_ptr, shape=desc.shape)
    strided = np.lib.stride_tricks.as_strided(
        array,
        np.ctypeslib.as_array(desc.shape),
        np.ctypeslib.as_array(desc.strides) * array.itemsize,
    )
    return _to_numpy(strided)


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
        desc = _get_ranked_memref_descriptor(array)
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
        ctp = _ctype_for_dtype(res_type.dtype)
        desc = _make_nd_memref_descriptor(len(res_type.shape), ctp)()
        descriptors.append(desc)
        keepalive.append(desc)

    if len(descriptors) == 1:
        ptr = ctypes.pointer(ctypes.pointer(descriptors[0]))
        keepalive.append(ptr)
        return ptr, keepalive, lambda: _ranked_memref_to_numpy(ptr[0][0])

    output = _make_output_struct(descriptors)
    ptr = ctypes.pointer(ctypes.pointer(output))
    keepalive.extend([output, ptr])
    return (
        ptr,
        keepalive,
        lambda: [
            _ranked_memref_to_numpy(getattr(ptr[0][0], f"memref{i}"))
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
    if dtype.name not in _DTYPE_TO_NP:
        _check_supported_dtype(dtype)
    return _DTYPE_TO_NP[dtype.name]


def _ctype_for_dtype(dtype: DType):
    if dtype.name not in _DTYPE_TO_CTYPE:
        _check_supported_dtype(dtype)
    return _DTYPE_TO_CTYPE[dtype.name]


def _check_supported_dtype(dtype: DType):
    if isinstance(dtype, APInt) and dtype.primitive_width > 64:
        raise NotImplementedError("CPU backend does not support APInt > 64 bits yet")
    if isinstance(dtype, APFloat):
        raise NotImplementedError(f"CPU backend does not support {dtype.name}")
    if isinstance(dtype, IndexType):
        return
    raise TypeError(f"Unsupported CPU dtype: {dtype}")


def _as_ctype(dtype):
    if dtype == np.dtype(np.float16):
        return _F16
    if dtype == ml_dtypes.bfloat16:
        return _BF16
    return np.ctypeslib.as_ctypes_type(dtype)


def _to_numpy(array):
    if array.dtype == _F16:
        return array.view("float16")
    if array.dtype == _BF16:
        return array.view("bfloat16")
    return array


def _convert_back(array, dtype):
    if dtype == np.dtype(np.float16):
        return array.view(np.float16)
    if dtype == ml_dtypes.bfloat16:
        return array.view(ml_dtypes.bfloat16)
    return array.astype(dtype, copy=False)


class CPU(Backend):
    name = "cpu"

    def __init__(
        self,
        kernel=None,
        *,
        opt_level: int = 2,
        shared_libs: list[str] | None = None,
    ):
        super().__init__(kernel)
        self.opt_level = opt_level
        self.shared_libs = shared_libs
        self.engine = None
        self.arg_types = None
        self.res_types = None

    def call_kernel(self, kernel, *args, **kwargs) -> Any:
        return CPU(
            kernel,
            opt_level=self.opt_level,
            shared_libs=self.shared_libs,
        ).run(*args, **kwargs)

    @terminate_on_error
    def compile(self):
        from .._C import execution_engine, ir, passes

        if self.engine is not None:
            assert self.module is not None
            return self.module
        if self.kernel.options.enable_tensor:
            raise NotImplementedError("CPU backend does not support tensor ABI yet")

        shared_libs = (
            _default_shared_libs() if self.shared_libs is None else self.shared_libs
        )
        cache_key = self._cache_key(
            {
                "backend": self.name,
                "opt_level": self.opt_level,
                "shared_libs": shared_libs,
                "version": 1,
            }
        )
        cached = self._process_cache_get("cpu.compile", cache_key)
        if cached is not None:
            with stage("Compiling CPU Kernels (Cache Hit)"):
                self._module_owner = cached.module_owner
                self.module = cached.module
                self.engine = cached.engine
                self.arg_types = cached.arg_types
                self.res_types = cached.res_types
            return self.module

        with stage("Compiling CPU Kernels"):
            module = self._get_working_module()
            arg_types = self.kernel.parse_argument_annotations()
            res_types = self.kernel.parse_return_annotation()

            top = module.lookup_func(self.kernel.func_name)
            if top is None:
                raise RuntimeError(
                    f"Cannot find top function '{self.kernel.func_name}'"
                )
            top.set_attr("llvm.emit_c_interface", ir.UnitAttr.get(module.get_context()))

            passes.lower_to_llvm(module, False)
            engine = execution_engine.ExecutionEngine(
                module,
                opt_level=self.opt_level,
                shared_libs=shared_libs,
            )
            self.arg_types = arg_types
            self.res_types = res_types
            self.engine = engine
            self._process_cache_set(
                "cpu.compile",
                cache_key,
                _CPUCompileCacheEntry(
                    module_owner=self._module_owner,
                    module=module,
                    engine=engine,
                    arg_types=arg_types,
                    res_types=res_types,
                ),
            )
            return module

    @terminate_on_error
    def run(self, *args, **kwargs) -> Any:
        self._ensure_compiled()
        return self.simulate(*args, **kwargs)

    @terminate_on_error
    def simulate(self, *args, **kwargs) -> Any:
        if kwargs:
            raise TypeError("CPU.simulate only accepts positional kernel arguments")
        if self.module is None:
            raise RuntimeError(
                "Kernel is not compiled yet. Run compilation before simulation."
            )
        assert self.engine is not None
        assert self.arg_types is not None
        assert self.res_types is not None

        packed_args, _keepalive, arg_arrays, result_decode = _pack_kernel_args(
            args, self.arg_types, self.res_types
        )
        with stage("Running CPU Kernels (JIT)"):
            self.engine.invoke(self.kernel.func_name, *packed_args)
            _writeback_args(arg_arrays)
            if result_decode is None:
                return None
            return result_decode()

    @terminate_on_error
    def scaffold_project(
        self,
        project: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        project_path = make_project_path(project, self.kernel.func_name, overwrite)
        self._ensure_compiled()
        assert self.module is not None
        (project_path / "lowered.mlir").write_text(str(self.module), encoding="utf-8")
        return project_path

    def _ensure_compiled(self):
        if self.engine is None:
            self.compile()
