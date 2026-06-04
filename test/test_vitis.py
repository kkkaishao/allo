# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vitis HLS backend regression tests.

Two layers:

* **Codegen** tests drive a kernel through the schedule interface
  (``kernel.schedule() -> s.<transform>() -> s.export("vitis").hls_code``) and
  assert on the emitted C++. They need no toolchain and run everywhere.
* **Synthesis / simulation** tests are gated on ``is_vitis_available()`` and
  invoke real Vitis HLS via ``s.export("vitis", part=...).synth()`` / csim.

Tests always go through the schedule interface, never a hand-built ``Vitis``.
"""

import re
import tempfile

import numpy as np
import pytest

from allo.exp.lang.core import range as arange, i32, f32, APInt, Stream, Template
from allo.exp.lang.kernel import kernel
from allo.exp.backend.vitis.core import is_vitis_available

u32 = APInt(32, signed=False)
u256 = APInt(256, signed=False)

PART = "xcvu9p-flga2104-2-i"
requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)


def _hls(schedule, **export_kwargs) -> str:
    """Emit the HLS C++ for a scheduled kernel (no toolchain required)."""
    return schedule.export("vitis", **export_kwargs).hls_code


def _contains(code: str, *needles: str):
    for needle in needles:
        assert needle in code, f"expected to find {needle!r} in:\n{code}"


def _regex(code: str, *patterns: str):
    for pattern in patterns:
        assert re.search(pattern, code), f"no match for {pattern!r} in:\n{code}"


# ===========================================================================
# Codegen-text tests (no toolchain)
# ===========================================================================


def test_codegen_vadd_pipeline():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, 'extern "C" void vadd(float ', "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \+ v\d+;")


def test_codegen_vadd2_tile():
    @kernel
    def vadd2(A: f32[8, 8], B: f32[8, 8], C: f32[8, 8]):
        for i in arange(8, name="i"):
            for j in arange(8, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = vadd2.schedule()
    i, j = s.affine(s.loops("i", "j"))
    s.tile((i, j), factors=[4, 4])
    code = _hls(s)
    _contains(code, "void vadd2(float v0[8][8]")
    # 8 split by 4 -> a 2-iteration outer band over a 4-iteration inner band.
    _regex(code, r"< 2;", r"< 4;")
    assert code.count("for (") >= 4


def test_codegen_gemm_reorder_pipeline():
    M, K, N = Template("M"), Template("K"), Template("N")

    @kernel(M, K, N)
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in arange(M, name="i"):
            for j in arange(N, name="j"):
                for k in arange(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm[16, 16, 16].schedule()
    s.affine(s.loops("i", "j", "k"))
    s.reorder((s.loop("k"), s.loop("j")))
    s.pipeline(s.loop("j"), ii=1)
    code = _hls(s)
    _contains(code, "void gemm(float v0[16][16]", "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \* v\d+;")
    assert code.count("for (") >= 3


def test_codegen_reduction():
    @kernel
    def vsum(A: f32[16], out: f32[1]):
        for i in arange(16, name="i"):
            out[0] += A[i]

    s = vsum.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, "void vsum(float v0[16], float v1[1])", "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \+ v\d+;")


def test_codegen_stencil():
    @kernel
    def stencil(A: f32[18], B: f32[16]):
        for i in arange(16, name="i"):
            B[i] = A[i] + A[i + 1] + A[i + 2]

    s = stencil.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, "void stencil(float v0[18], float v1[16])", "#pragma HLS pipeline")
    # three taps summed -> at least two additions
    assert len(re.findall(r"= v\d+ \+ v\d+;", code)) >= 2


def test_codegen_wide_integer():
    @kernel
    def copy256(A: u256[8], B: u256[8]):
        for i in arange(8, name="i"):
            B[i] = A[i]

    code = _hls(copy256.schedule())
    _contains(code, "void copy256(ap_uint<256> v0[8], ap_uint<256> v1[8])")


def test_codegen_apint_csim_wrapper():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def vadd5(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    backend = vadd5.schedule().export("vitis")
    # The synthesizable interface keeps the real ap_int boundary.
    _contains(
        backend.hls_code,
        "void vadd5(ap_int<5> v0[8], ap_uint<5> v1[8], ap_int<5> v2[8])",
    )
    # C simulation wraps it with a std-width interface around the renamed kernel,
    # so ctypes can call it (signedness preserved per operand).
    csim_cpp = backend._compile_for_csim().kernel_cpp
    _contains(
        csim_cpp,
        'extern "C" void vadd5(int8_t v0[8], uint8_t v1[8], int8_t v2[8])',
        "void vadd5__impl(ap_int<5>",
        "ap_int<5> v",  # signed temp matches the callee parameter
        "ap_uint<5> v",  # unsigned temp
    )


def test_codegen_bit_slice():
    @kernel
    def bits(x: u32, out: u32[1]):
        y: u32 = x
        y[0:4] = 5
        out[0] = y[4:8]

    code = _hls(bits.schedule())
    _contains(code, "& ~(0xfULL <<", "static_cast<uint32_t>", "ap_uint<4>")
    _regex(code, r">> v\d+\) & 0xfULL")


def test_codegen_block_stream_datamover():
    @kernel
    def dmover(inp: i32[4, 4], out: i32[1]):
        fifo: Stream[i32[4, 4]]

        @kernel
        def load(src: i32[4, 4], strm: Stream[i32[4, 4]]):
            strm.put(src)

        @kernel
        def compute(strm: Stream[i32[4, 4]], dst: i32[1]):
            blk = strm.get()
            dst[0] = blk[0, 0]

        load(inp, fifo)
        compute(fifo, out)

    code = _hls(dmover.schedule())
    # Block payload streams element-by-element through a scalar FIFO whose depth
    # is scaled by the block size (2 blocks x 4x4 = 32), not via stream_of_blocks.
    _contains(
        code,
        "hls::stream<uint32_t>",
        ".write(",
        ".read()",
        "dmover_load",
        "dmover_compute",
    )
    _regex(code, r"#pragma HLS stream variable=v\d+ depth=32")
    assert "stream_of_blocks" not in code
    assert "read_lock" not in code


def test_codegen_maxi_interface():
    @kernel
    def axicopy(A: i32[64], B: i32[64]):
        for i in arange(64, name="i"):
            B[i] = A[i] + 1

    backend = axicopy.schedule().export("vitis", part=PART)
    backend.set_axi(0, offset="slave", bundle="gmem")
    backend.set_axi(1, offset="slave", bundle="gmem")
    code = backend.hls_code
    _contains(
        code,
        "#pragma HLS INTERFACE mode=m_axi port=v0 offset=slave bundle=gmem",
        "#pragma HLS INTERFACE mode=m_axi port=v1 offset=slave bundle=gmem",
    )


def test_synth_requires_part():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    # Exporting without a part is fine for codegen, but synth must reject it.
    # The backend funnels errors through terminate_on_error -> SystemExit.
    backend = vadd.schedule().export("vitis")
    assert backend.hls_code  # codegen works without a part
    with pytest.raises(SystemExit):
        backend.synth()


# ===========================================================================
# Synthesis / simulation tests (gated on a real Vitis HLS toolchain)
# ===========================================================================


@requires_vitis
def test_synth_gemm_tile_pipeline():
    M = N = K = 16

    @kernel
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in arange(M, name="i"):
            for j in arange(N, name="j"):
                for k in arange(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm.schedule()
    i, j, k = s.affine(s.loops("i", "j", "k"))
    s.tile((i, j), factors=[4, 4])
    s.pipeline(s.loop("k"), ii=1)
    with tempfile.TemporaryDirectory() as project:
        report = s.export("vitis", part=PART, project_path=project).synth()
        assert report.xml_path.exists()


@requires_vitis
def test_synth_block_stream_datamover():
    @kernel
    def dmover(inp: i32[4, 4], out: i32[1]):
        fifo: Stream[i32[4, 4]]

        @kernel
        def load(src: i32[4, 4], strm: Stream[i32[4, 4]]):
            strm.put(src)

        @kernel
        def compute(strm: Stream[i32[4, 4]], dst: i32[1]):
            blk = strm.get()
            dst[0] = blk[0, 0]

        load(inp, fifo)
        compute(fifo, out)

    with tempfile.TemporaryDirectory() as project:
        report = (
            dmover.schedule().export("vitis", part=PART, project_path=project).synth()
        )
        assert report.xml_path.exists()


@requires_vitis
def test_csim_vadd():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=1)
    a = np.random.rand(16).astype(np.float32)
    b = np.random.rand(16).astype(np.float32)
    c = np.zeros(16, dtype=np.float32)
    with tempfile.TemporaryDirectory() as project:
        backend = s.export("vitis", project_path=project)
        backend(a, b, c)
    np.testing.assert_allclose(c, a + b, rtol=1e-5)


@requires_vitis
def test_csim_apint():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def addsub(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    a = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    c = np.zeros(8, dtype=np.int8)
    with tempfile.TemporaryDirectory() as project:
        backend = addsub.schedule().export("vitis", project_path=project)
        backend(a, b, c)
    # i5 result wraps modulo 2**5 with sign extension back to int8.
    expected = ((a.astype(np.int16) + b + 16) % 32 - 16).astype(np.int8)
    np.testing.assert_array_equal(c, expected)
