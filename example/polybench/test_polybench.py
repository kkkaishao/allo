# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from allo.lang.core import BufferType, DType
from allo.lang.kernel import Kernel

_DTYPE_TO_NP = {
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

_RTOL = 1e-5
_ATOL = 1e-5
_SEED = 42


def _numpy_dtype(dtype: DType):
    if dtype.name not in _DTYPE_TO_NP:
        raise TypeError(f"Unsupported polybench test dtype: {dtype}")
    return _DTYPE_TO_NP[dtype.name]


def _make_arg(arg_type, rng: np.random.Generator):
    if isinstance(arg_type, BufferType):
        dtype = _numpy_dtype(arg_type.dtype)
        shape = tuple(arg_type.shape)
        if np.issubdtype(dtype, np.floating):
            return rng.uniform(0.01, 0.25, size=shape).astype(dtype)
        if dtype == np.bool_:
            return rng.integers(0, 2, size=shape).astype(dtype)
        return rng.integers(0, 4, size=shape).astype(dtype)
    if isinstance(arg_type, DType):
        dtype = _numpy_dtype(arg_type)
        if np.issubdtype(dtype, np.floating):
            return np.dtype(dtype).type(rng.uniform(0.01, 0.25))
        if dtype == np.bool_:
            return np.dtype(dtype).type(rng.integers(0, 2))
        return np.dtype(dtype).type(rng.integers(0, 4))
    raise TypeError(f"Unsupported polybench test argument type: {arg_type}")


def _make_spd_matrix(arg_type: BufferType, rng: np.random.Generator):
    assert len(arg_type.shape) == 2 and arg_type.shape[0] == arg_type.shape[1]
    dtype = _numpy_dtype(arg_type.dtype)
    n = arg_type.shape[0]
    base = rng.uniform(-0.05, 0.05, size=(n, n)).astype(dtype)
    matrix = base @ base.T
    matrix += np.eye(n, dtype=dtype)
    return matrix.astype(dtype, copy=False)


def _make_diagonally_dominant_matrix(arg_type: BufferType, rng: np.random.Generator):
    assert len(arg_type.shape) == 2 and arg_type.shape[0] == arg_type.shape[1]
    dtype = _numpy_dtype(arg_type.dtype)
    n = arg_type.shape[0]
    matrix = rng.uniform(-0.01, 0.01, size=(n, n)).astype(dtype)
    matrix += np.eye(n, dtype=dtype) * np.dtype(dtype).type(2.0)
    return matrix.astype(dtype, copy=False)


def _make_lower_triangular_matrix(arg_type: BufferType, rng: np.random.Generator):
    matrix = _make_diagonally_dominant_matrix(arg_type, rng)
    return np.tril(matrix).astype(matrix.dtype, copy=False)


def _make_args(module_name: str, arg_types: list[BufferType | DType]):
    rng = np.random.default_rng(
        _SEED + sum((i + 1) * ord(c) for i, c in enumerate(module_name))
    )
    args = [_make_arg(arg_type, rng) for arg_type in arg_types]

    if module_name == "cholesky":
        args[0] = _make_spd_matrix(arg_types[0], rng)
    elif module_name in {"lu", "ludcmp"}:
        args[0] = _make_diagonally_dominant_matrix(arg_types[0], rng)
    elif module_name == "trisolv":
        args[0] = _make_lower_triangular_matrix(arg_types[0], rng)
    elif module_name == "durbin":
        dtype = _numpy_dtype(arg_types[0].dtype)
        args[0] = rng.uniform(-0.001, 0.001, size=arg_types[0].shape).astype(dtype)
    elif module_name == "nussinov":
        dtype = _numpy_dtype(arg_types[0].dtype)
        args[0] = rng.integers(1, 3, size=arg_types[0].shape).astype(dtype)
        args[1].fill(0)
    return args


def _copy_args(args):
    return [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _assert_close(name: str, label: str, actual, expected):
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{name} {label}",
    )


def _check_result(
    name: str, kernel_args, reference_args, kernel_result, reference_result
):
    for i, (kernel_arg, reference_arg) in enumerate(zip(kernel_args, reference_args)):
        if isinstance(kernel_arg, np.ndarray):
            _assert_close(name, f"argument {i}", kernel_arg, reference_arg)

    kernel_results = _as_tuple(kernel_result)
    if not kernel_results:
        return

    reference_results = _as_tuple(reference_result)
    assert len(kernel_results) == len(reference_results)
    for i, (actual, expected) in enumerate(zip(kernel_results, reference_results)):
        if isinstance(actual, np.ndarray):
            _assert_close(name, f"return {i}", actual, expected)
        else:
            assert actual == expected


def _iter_top_kernels():
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("test_"):
            continue
        module_name = path.stem
        module = importlib.import_module(f"{__package__}.{module_name}")
        kernel = getattr(module, module_name)
        assert isinstance(kernel, Kernel)
        reference = getattr(module, f"np_{module_name}")
        assert callable(reference)
        yield module_name, kernel, reference


def _make_test(module_name: str, name: str, kernel: Kernel, reference):
    def test_kernel():
        assert callable(reference)
        args = _make_args(module_name, kernel.parse_argument_annotations())
        kernel_args = _copy_args(args)
        reference_args = _copy_args(args)
        kernel_result = kernel(*kernel_args)
        reference_result = reference(*reference_args)
        _check_result(
            module_name, kernel_args, reference_args, kernel_result, reference_result
        )

    test_kernel.__name__ = name
    test_kernel.__qualname__ = name
    return test_kernel


for _module_name, _kernel, _reference in _iter_top_kernels():
    _test_name = f"test_{_module_name}"
    globals()[_test_name] = _make_test(_module_name, _test_name, _kernel, _reference)
