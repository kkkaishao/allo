# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Systolic GEMM arrays: a 2D output-stationary array and a 1D row array.

Functional checks run through both the CPU dataflow simulator and Vitis csim.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import f32, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"


# ===========================================================================
# 2D output-stationary systolic GEMM on an (M+2) x (N+2) PE grid
# ===========================================================================

M, N, K = 2, 2, 2
P0, P1 = M + 2, N + 2


@kernel
def systolic_2d(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    fifo_A: Stream[f32][P0, P1]
    fifo_B: Stream[f32][P0, P1]

    @kernel(mapping=[P0, P1])
    def pe(
        A: f32[M, K],
        B: f32[K, N],
        C: f32[M, N],
        fifo_A: Stream[f32][P0, P1],
        fifo_B: Stream[f32][P0, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        if (i == 0 or i == M + 1) and (j == 0 or j == N + 1):
            pass
        elif j == 0:
            for k in range(K):
                fifo_A[i, j + 1].put(A[i - 1, k])
        elif i == 0:
            for k in range(K):
                fifo_B[i + 1, j].put(B[k, j - 1])
        elif i == M + 1:
            for k in range(K):
                b: f32 = fifo_B[i, j].get()
        elif j == N + 1:
            for k in range(K):
                a: f32 = fifo_A[i, j].get()
        else:
            c: f32 = 0
            for k in range(K):
                a: f32 = fifo_A[i, j].get()
                b: f32 = fifo_B[i, j].get()
                c += a * b
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
            C[i - 1, j - 1] = c

    pe(A, B, C, fifo_A, fifo_B)


def test_2d_cpu_sim():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    systolic_2d.schedule().export("cpu")(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


def test_2d_codegen():
    code = systolic_2d.schedule().export("vitis").hls_code
    # Each PE is specialized into its own function; the inner compute PE keeps
    # both A and B FIFOs while the idle corner PEs are pruned entirely.
    assert "void systolic_2d_pe_1_1(" in code
    assert "systolic_2d_pe_0_0" not in code
    assert "systolic_2d_pe_0_1(" in code


@requires_vitis
def test_2d_csim():
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    with tempfile.TemporaryDirectory() as proj:
        systolic_2d.schedule().export("vitis", project_path=proj)(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)


@requires_vitis
def test_2d_synth():
    with tempfile.TemporaryDirectory() as proj:
        report = (
            systolic_2d.schedule().export("vitis", part=PART, project_path=proj).synth()
        )
        assert report.exists()


# ===========================================================================
# 1D (row) systolic GEMM: row 0 streams B down, row-1 PEs pass A along and MAC
# ===========================================================================

M_1D, N_1D, K_1D = 16, 16, 16
P_1D = K_1D + 2


@kernel
def systolic_1d(A: f32[M_1D, K_1D], B: f32[K_1D, N_1D], C: f32[M_1D, N_1D]):
    fifo_A: Stream[f32][2, P_1D]
    fifo_B: Stream[f32][2, P_1D]

    @kernel(mapping=[2, P_1D])
    def pe(
        A: f32[M_1D, K_1D],
        B: f32[K_1D, N_1D],
        C: f32[M_1D, N_1D],
        fifo_A: Stream[f32][2, P_1D],
        fifo_B: Stream[f32][2, P_1D],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        if i == 0 and (j == 0 or j == P_1D - 1):
            pass
        elif i == 0:
            for _ in range(N_1D):
                for k in range(K_1D):
                    fifo_B[i + 1, j].put(B[k, j - 1])
        elif j == 0:
            for m in range(M_1D):
                for k in range(K_1D):
                    fifo_A[i, j + 1].put(A[m, k])
        elif j == P_1D - 1:
            for m in range(M_1D):
                for _ in range(K_1D):
                    a: f32 = fifo_A[i, j].get()
        else:
            for m in range(M_1D):
                c: f32 = 0
                for _ in range(K_1D):
                    a: f32 = fifo_A[i, j].get()
                    b: f32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                C[m, j - 1] = c

    pe(A, B, C, fifo_A, fifo_B)


def test_1d_codegen():
    code = systolic_1d.schedule().export("vitis").hls_code
    assert "void systolic_1d_pe_1_1(" in code
    # the two idle corners of the feeder row are pruned
    assert "systolic_1d_pe_0_0" not in code


def test_1d_cpu_sim():
    A = np.random.rand(M_1D, K_1D).astype(np.float32)
    B = np.random.rand(K_1D, N_1D).astype(np.float32)
    C = np.zeros((M_1D, N_1D), dtype=np.float32)
    systolic_1d.schedule().export("cpu")(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-3)


@requires_vitis
def test_1d_csim():
    A = np.random.rand(M_1D, K_1D).astype(np.float32)
    B = np.random.rand(K_1D, N_1D).astype(np.float32)
    C = np.zeros((M_1D, N_1D), dtype=np.float32)
    with tempfile.TemporaryDirectory() as proj:
        systolic_1d.schedule().export("vitis", project_path=proj)(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-3)


@requires_vitis
def test_1d_synth():
    with tempfile.TemporaryDirectory() as proj:
        report = (
            systolic_1d.schedule().export("vitis", part=PART, project_path=proj).synth()
        )
        assert report.exists()
