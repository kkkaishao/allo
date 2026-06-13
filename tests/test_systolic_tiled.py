# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiled 2D systolic GEMM (int32): a Mt x Nt PE array sweeps over
(M/Mt) x (N/Nt) output tiles.

Functional checks run through Vitis csim; the CPU dataflow simulator currently
deadlocks on multi-PE arrays.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import i32, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.core import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"

M, N, K = 16, 16, 16
Mt, Nt = 4, 4
P0, P1 = Mt + 2, Nt + 2


@kernel
def top(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
    fifo_A: Stream[i32][P0, P1]
    fifo_B: Stream[i32][P0, P1]

    @kernel(mapping=[P0, P1])
    def gemm(
        A: i32[M, K],
        B: i32[K, N],
        C: i32[M, N],
        fifo_A: Stream[i32][P0, P1],
        fifo_B: Stream[i32][P0, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        for m in range(M // Mt):
            for n in range(N // Nt):
                if (i == 0 or i == Mt + 1) and (j == 0 or j == Nt + 1):
                    pass
                elif j == 0:
                    for k in range(K):
                        fifo_A[i, j + 1].put(A[m * Mt + i - 1, k])
                elif i == 0:
                    for k in range(K):
                        fifo_B[i + 1, j].put(B[k, n * Nt + j - 1])
                elif i == Mt + 1:
                    for k in range(K):
                        b: i32 = fifo_B[i, j].get()
                elif j == Nt + 1:
                    for k in range(K):
                        a: i32 = fifo_A[i, j].get()
                else:
                    c: i32 = 0
                    for k in range(K):
                        a: i32 = fifo_A[i, j].get()
                        b: i32 = fifo_B[i, j].get()
                        c += a * b
                        fifo_A[i, j + 1].put(a)
                        fifo_B[i + 1, j].put(b)
                    C[m * Mt + i - 1, n * Nt + j - 1] = c

    gemm(A, B, C, fifo_A, fifo_B)


def test_codegen():
    code = top.schedule().export("vitis").hls_code
    assert "void top_gemm_1_1(" in code
    assert "top_gemm_0_0" not in code


def test_cpu_sim():
    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    top.schedule().export("cpu")(A, B, C)
    np.testing.assert_array_equal(C, A @ B)


@requires_vitis
def test_csim():
    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    with tempfile.TemporaryDirectory() as proj:
        top.schedule().export("vitis", project_path=proj)(A, B, C)
    np.testing.assert_array_equal(C, A @ B)
