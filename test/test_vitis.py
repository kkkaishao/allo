# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import re
import tempfile

import numpy as np
import pytest

from allo.exp._C import passes
from allo.exp._C.passes import emit_vivado_hls
from allo.exp.backend.vitis.core import (
    DEFAULT_VITIS_SETTINGS,
    HLS_PREPARE_PIPELINE,
    Vitis,
)
from allo.exp.backend.vitis.utils import detect_vitis_tool
from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.lang.core import i32, range as allo_range
from allo.exp.lang.kernel import kernel


def _emit_vitis_cpp(fn) -> str:
    module = compile_kernel(fn)
    passes.run(HLS_PREPARE_PIPELINE, module.get_operation())
    code = emit_vivado_hls(module)
    assert code is not None
    return code


def _assert_contains(code: str, *patterns: str):
    for pattern in patterns:
        assert pattern in code


def _assert_regex(code: str, *patterns: str):
    for pattern in patterns:
        assert re.search(pattern, code), pattern


@functools.cache
def _has_vitis() -> bool:
    try:
        detect_vitis_tool(DEFAULT_VITIS_SETTINGS)
    except (RuntimeError, SystemExit):
        return False
    return True


def test_vitis_basic_kernel():
    @kernel
    def top(x: i32, out: "i32[1]"):
        out[0] = x + 1

    code = _emit_vitis_cpp(top)
    _assert_contains(
        code,
        "#include <hls_stream.h>",
        "#include <hls_streamofblocks.h>",
        "void top(",
        "#pragma HLS inline off",
        "return;",
    )


def test_vitis_loop():
    @kernel
    def top(x: "i32[4]", out: "i32[4]"):
        for i in allo_range(4):
            out[i] = x[i] + 1

    code = _emit_vitis_cpp(top)
    _assert_contains(code, "void top(", "for (", " += ", "[")


def test_vitis_nested_calls():
    @kernel
    def top(x: i32, out: "i32[1]"):
        @kernel
        def worker(v: i32, dst: "i32[1]"):
            dst[0] = v + 1

        worker(x, out)

    code = _emit_vitis_cpp(top)
    _assert_contains(code, "top_worker")
    _assert_regex(code, r"void top_worker\(uint32_t v\d+, uint32_t v\d+\[1\]\);")


def test_vitis_scalar_stream():
    @kernel
    def top(x: i32, out: "i32[1]"):
        fifo: "Stream[i32][2,2]"
        fifo[0, 1].put(x)
        out[0] = fifo[0, 1].get()

    code = _emit_vitis_cpp(top)
    _assert_regex(
        code,
        r"hls::stream<uint32_t> v\d+\[2\]\[2\];",
        r"#pragma HLS stream variable=v\d+ depth=2",
    )
    _assert_contains(code, ".write(", ".read();")


def test_vitis_stream_parameter():
    @kernel
    def top(x: i32, out: "i32[1]"):
        fifo: "Stream[i32][2,2]"

        @kernel
        def worker(s: "Stream[i32][2,2]", v: i32):
            s[0, 1].put(v)

        worker(fifo, x)
        out[0] = fifo[0, 1].get()

    code = _emit_vitis_cpp(top)
    _assert_contains(code, "top_worker")
    _assert_regex(
        code,
        r"void top_worker\(hls::stream<uint32_t> &v\d+, uint32_t v\d+\);",
        r"#pragma HLS stream variable=v\d+ depth=2",
    )


def test_vitis_block_stream():
    @kernel
    def top(out: "i32[1]"):
        fifo: "Stream[i32[2,2]]"
        buf: "i32[2,2]"
        buf[0, 0] = 7
        fifo.put(buf)
        recv = fifo.get()
        out[0] = recv[0, 0]

    code = _emit_vitis_cpp(top)
    _assert_contains(
        code,
        "hls::stream_of_blocks<uint32_t[2][2], 2>",
        "hls::write_lock<uint32_t[2][2]>",
        "hls::read_lock<uint32_t[2][2]>",
    )
    _assert_regex(code, r"for \(int32_t i\d+ = 0; i\d+ < 2; \+\+i\d+\)")
    assert ".write(" not in code
    assert ".read()" not in code


def test_vitis_csim_scalar_stream():
    if not _has_vitis():
        pytest.skip("Vitis HLS toolchain is not available")

    @kernel
    def top(x: i32, out: "i32[1]"):
        fifo: "Stream[i32]"
        fifo.put(x)
        out[0] = fifo.get()

    out = np.zeros((1,), dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = Vitis(top, project_path=tmpdir)
        backend.csim(7, out)

    np.testing.assert_array_equal(out, np.array([7], dtype=np.int32))
