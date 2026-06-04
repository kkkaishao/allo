from __future__ import annotations

import ctypes
import os
import subprocess

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .utils import _render_template
from ..utils import numpy_to_ctype
from ..base import write_text_if_changed
from ...lang.core import BufferType, DType, TypeBase, widen_apint_to_std
from ...logging import completed_output, log_debug, log_detail, run_command, stage

CSIM_MAKEFILE = "csim.mk"
CSIM_SHARED_LIBRARY = "libkernel.so"


_DTYPE_TO_NP = {
    "float32": np.float32,
    "float64": np.float64,
    "index": np.int32,
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


def _generate_csim_makefile(vitis_root: Path) -> str:
    return _render_template(
        "csim.mk",
        csim_shared_library=CSIM_SHARED_LIBRARY,
        vitis_root=os.fspath(vitis_root),
    )


def _prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
    old = env.get(name, "")
    env[name] = os.fspath(path) + (os.pathsep + old if old else "")


def _numpy_dtype_for_dtype(dtype: DType):
    dtype = widen_apint_to_std(dtype)
    if dtype.name not in _DTYPE_TO_NP:
        raise TypeError(f"Unsupported Vitis Python-native csim dtype: {dtype}")
    return _DTYPE_TO_NP[dtype.name]


def _ctype_for_dtype(dtype: DType):
    return numpy_to_ctype(_numpy_dtype_for_dtype(dtype))


def _as_csim_array(arg, buffer_type: BufferType):
    if not isinstance(arg, np.ndarray):
        raise TypeError(
            "Vitis Python-native csim buffer arguments must be numpy arrays"
        )
    if tuple(arg.shape) != tuple(buffer_type.shape):
        raise ValueError(
            f"Expected buffer shape {tuple(buffer_type.shape)}, got {arg.shape}"
        )

    np_dtype = _numpy_dtype_for_dtype(buffer_type.dtype)
    array = arg
    if array.dtype != np_dtype:
        array = array.astype(np_dtype)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def _writeback_csim_arrays(arg_arrays) -> None:
    for original, array in arg_arrays:
        if isinstance(original, np.ndarray) and original is not array:
            original[...] = array.astype(original.dtype, copy=False)


def _csim_argtype(arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        return np.ctypeslib.ndpointer(
            dtype=_numpy_dtype_for_dtype(arg_type.dtype),
            ndim=len(arg_type.shape),
            flags="C_CONTIGUOUS",
        )
    if isinstance(arg_type, DType):
        return _ctype_for_dtype(arg_type)
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _pack_csim_arg(arg, arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        array = _as_csim_array(arg, arg_type)
        return array, array
    if isinstance(arg_type, DType):
        return _ctype_for_dtype(arg_type)(arg), None
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _csim_return_type(res_types: list[TypeBase]):
    if not res_types:
        return None
    if len(res_types) == 1 and isinstance(res_types[0], DType):
        return _ctype_for_dtype(res_types[0])
    raise TypeError("Vitis Python-native csim only supports void or scalar return")


class PythonNativeCSimulator:
    def __init__(
        self,
        *,
        top: str,
        project_path: Path,
        vitis_root: Path,
        env: Mapping[str, str],
        arg_types: list[TypeBase],
        res_types: list[TypeBase],
        make_vars: Mapping[str, str] | None = None,
    ):
        self.top = top
        self.project_path = project_path
        self.vitis_root = vitis_root
        self.env = dict(env)
        self.arg_types = list(arg_types)
        self.res_types = list(res_types)
        self.make_vars = dict(make_vars or {})
        self.library_path = self._resolve_project_path(
            self.make_vars.get("OUT", CSIM_SHARED_LIBRARY)
        )
        self._library: ctypes.CDLL | None = None
        self._function = None

    def run(self, *args, exist_ok: bool = True) -> Any:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"Expected {len(self.arg_types)} arguments, got {len(args)}"
            )
        self.build(exist_ok=exist_ok)
        func = self._get_function()
        packed_args = []
        arg_arrays = []
        for arg, arg_type in zip(args, self.arg_types):
            packed, array = _pack_csim_arg(arg, arg_type)
            packed_args.append(packed)
            if array is not None:
                arg_arrays.append((arg, array))

        with stage("Running Vitis C Simulation"):
            result = func(*packed_args)
            _writeback_csim_arrays(arg_arrays)
            return result

    def build(self, *, exist_ok: bool = True) -> Path:
        write_text_if_changed(
            self.project_path / CSIM_MAKEFILE,
            _generate_csim_makefile(self.vitis_root),
        )
        if self.library_path.exists() and exist_ok:
            log_debug(
                f"Building Vitis C Simulation Shared Library: {self.library_path} (cache hit)"
            )
            return self.library_path

        self._library = None
        self._function = None
        env = self._make_env()
        with stage("Building Vitis C Simulation Shared Library"):
            dry_run = run_command(
                self._make_command(dry_run=True),
                cwd=self.project_path,
                env=env,
            )
            self._log_make_commands(dry_run)
            run_command(self._make_command(), cwd=self.project_path, env=env)
        return self.library_path

    def _make_env(self) -> dict[str, str]:
        env = dict(self.env)
        vitis_root = Path(self.make_vars.get("VITIS_ROOT", os.fspath(self.vitis_root)))
        _prepend_env_path(env, "LD_LIBRARY_PATH", vitis_root / "lib" / "lnx64.o")
        return env

    def _make_command(self, *, dry_run: bool = False) -> list[str]:
        cmd = ["make"]
        if dry_run:
            cmd.append("-n")
        cmd.extend(
            [
                "-f",
                CSIM_MAKEFILE,
                f"TOP={self.top}",
                *[f"{key}={value}" for key, value in self.make_vars.items()],
            ]
        )
        return cmd

    def _log_make_commands(self, result: subprocess.CompletedProcess[str]) -> None:
        output = completed_output(result)
        if output:
            log_detail(f"Make command:\n{output}")

    def _resolve_project_path(self, path: object) -> Path:
        resolved = Path(os.fspath(path) if isinstance(path, os.PathLike) else str(path))
        if resolved.is_absolute():
            return resolved
        return self.project_path / resolved

    def _get_function(self):
        if self._function is not None:
            return self._function
        if self._library is None:
            self._library = ctypes.CDLL(os.fspath(self.library_path))
        func = getattr(self._library, self.top)
        func.argtypes = [_csim_argtype(arg_type) for arg_type in self.arg_types]
        func.restype = _csim_return_type(self.res_types)
        self._function = func
        return func
