# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math

import numpy as np

from allo.exp.lang.core import BufferType, DType
from allo.exp.lang.kernel import Kernel

RTOL = 1e-5
ATOL = 1e-5
SEED = 42

_DTYPE_TO_NP = {
    "float32": np.float32,
    "float64": np.float64,
    "index": np.int64,
    "int32": np.int32,
    "uint8": np.uint8,
}


def _numpy_dtype(dtype: DType):
    if dtype.name not in _DTYPE_TO_NP:
        raise TypeError(f"Unsupported machsuite test dtype: {dtype}")
    return _DTYPE_TO_NP[dtype.name]


def make_machsuite_args(kernel: Kernel, case: str):
    rng = np.random.default_rng(
        SEED + sum((i + 1) * ord(c) for i, c in enumerate(case))
    )
    arg_types = kernel.parse_argument_annotations()
    args = [_make_arg(arg_type, rng) for arg_type in arg_types]

    if case in {"bfs_bulk", "bfs_queue"}:
        _set_bfs_args(args)
    elif case == "kmp":
        args[2].fill(0)
        args[3].fill(0)
    elif case.startswith("md_grid_"):
        max_points = args[1].shape[-1]
        args[0][...] = rng.integers(
            1, max_points + 1, size=args[0].shape, dtype=np.int32
        )
        for arg in args[1:]:
            arg[...] = rng.uniform(0.0, 20.0, size=arg.shape)
    elif case.startswith("md_knn_"):
        for arg in args[:3]:
            arg[...] = rng.uniform(0.0, 20.0, size=arg.shape)
        args[3][...] = rng.integers(0, args[0].shape[0], size=args[3].shape)
    elif case == "fft_strided":
        n = args[0].shape[0]
        for i in range(args[2].shape[0]):
            angle = 2.0 * math.pi * i / n
            args[2][i] = args[2].dtype.type(math.cos(angle))
            args[3][i] = args[3].dtype.type(math.sin(angle))
    elif case == "spmv_crs":
        _set_crs_args(args, rng)
    elif case == "radixsort":
        args[0][...] = rng.integers(0, 100000, size=args[0].shape, dtype=np.int32)
    elif case == "viterbi":
        args[0][...] = rng.integers(0, args[3].shape[1], size=args[0].shape)
    return args


def run_machsuite_kernel(kernel: Kernel, case: str):
    reference = _get_reference(kernel)
    args = make_machsuite_args(kernel, case)
    kernel_args = _copy_args(args)
    reference_args = _copy_args(args)
    kernel_result = kernel(*kernel_args)
    reference_result = reference(*reference_args)
    _check_result(case, kernel_args, reference_args, kernel_result, reference_result)
    return kernel_result


def _make_arg(arg_type, rng: np.random.Generator):
    if isinstance(arg_type, BufferType):
        dtype = _numpy_dtype(arg_type.dtype)
        shape = tuple(arg_type.shape)
        if np.issubdtype(dtype, np.floating):
            return rng.uniform(0.01, 0.25, size=shape).astype(dtype)
        if dtype == np.uint8:
            return rng.integers(0, 256, size=shape, dtype=dtype)
        return rng.integers(0, 4, size=shape, dtype=dtype)
    if isinstance(arg_type, DType):
        dtype = _numpy_dtype(arg_type)
        if np.issubdtype(dtype, np.floating):
            return np.dtype(dtype).type(rng.uniform(0.01, 0.25))
        return np.dtype(dtype).type(rng.integers(0, 4))
    raise TypeError(f"Unsupported machsuite test argument type: {arg_type}")


def _set_bfs_args(args):
    nodes, edges = args[0], args[1]
    n_nodes = nodes.shape[0] // 2
    fanout = min(n_nodes - 1, edges.shape[0])
    nodes.fill(fanout)
    edges.fill(0)
    nodes[0] = 0
    nodes[1] = fanout
    edges[:fanout] = np.arange(1, fanout + 1, dtype=edges.dtype)
    for node in range(1, n_nodes):
        nodes[2 * node] = fanout
        nodes[2 * node + 1] = fanout
    args[2] = np.int32(0)


def _set_crs_args(args, rng: np.random.Generator):
    val, cols, row, vec = args
    nnz = val.shape[0]
    n = vec.shape[0]
    row[...] = np.linspace(0, nnz, n + 1, dtype=np.int32)
    row[-1] = nnz
    cols[...] = rng.integers(0, n, size=cols.shape, dtype=np.int32)


def _get_reference(kernel: Kernel):
    module = importlib.import_module(kernel.fn.__module__)
    reference = getattr(module, f"np_{kernel.func_name}")
    assert callable(reference)
    return reference


def _copy_args(args):
    return [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]


def _check_result(
    case: str, kernel_args, reference_args, kernel_result, reference_result
):
    for i, (actual, expected) in enumerate(zip(kernel_args, reference_args)):
        if isinstance(actual, np.ndarray):
            _assert_close(case, f"argument {i}", actual, expected)

    kernel_results = _as_tuple(kernel_result)
    if not kernel_results:
        return

    reference_results = _as_tuple(reference_result)
    assert len(kernel_results) == len(reference_results)
    for i, (actual, expected) in enumerate(zip(kernel_results, reference_results)):
        _assert_close(case, f"return {i}", actual, expected)


def _assert_close(case: str, label: str, actual, expected):
    if isinstance(actual, np.ndarray):
        if np.issubdtype(actual.dtype, np.floating):
            np.testing.assert_allclose(
                actual, expected, rtol=RTOL, atol=ATOL, err_msg=f"{case} {label}"
            )
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=f"{case} {label}")
    elif isinstance(actual, np.floating):
        np.testing.assert_allclose(
            actual, expected, rtol=RTOL, atol=ATOL, err_msg=f"{case} {label}"
        )
    else:
        assert actual == expected


def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)
