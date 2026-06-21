# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bit-packed int8 systolic GEMM: PP int8 operands are packed into one i32 word
and unpacked per-lane with bit slicing inside the PE.

Note the bit-slice widths are written ``x[p*8 : p*8 + 8]`` (not ``(p+1)*8``) so
the static width cancels to a compile-time constant.

Functional checks run through Vitis csim; the CPU dataflow simulator currently
deadlocks on multi-PE arrays.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import i8, i32, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"

M, N, K = 8, 8, 4
PP = 4
P0, P1 = M // PP + 2, N + 2


@kernel
def top(X: i32[M, K // PP], W: i32[K // PP, N], Z: i32[M // PP, N]):
    fifo_A: Stream[i32][P0, P1]
    fifo_B: Stream[i32][P0, P1]

    @kernel(mapping=[P0, P1])
    def gemm(
        X: i32[M, K // PP],
        W: i32[K // PP, N],
        Z: i32[M // PP, N],
        fifo_A: Stream[i32][P0, P1],
        fifo_B: Stream[i32][P0, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        if (i == 0 or i == M // PP + 1) and (j == 0 or j == N + 1):
            pass
        elif j == 0:
            for k in range(K):
                fifo_A[i, j + 1].put(X[(i - 1) * PP, k])
        elif i == 0:
            for k in range(K):
                fifo_B[i + 1, j].put(W[k // PP, j - 1])
        elif i == M // PP + 1:
            for k in range(K):
                b: i32 = fifo_B[i, j].get()
        elif j == N + 1:
            for k in range(K):
                a: i32 = fifo_A[i, j].get()
        else:
            Z_elm: i32 = Z[i - 1, j - 1]
            for k in range(K):
                c: i32 = 0
                a: i32 = fifo_A[i, j].get()
                b: i32 = fifo_B[i, j].get()
                for p in range(PP):
                    a_unpacked: i8 = a[p * 8 : p * 8 + 8]
                    b_unpacked: i8 = b[p * 8 : p * 8 + 8]
                    c += a_unpacked * b_unpacked
                fifo_A[i, j + 1].put(a)
                fifo_B[i + 1, j].put(b)
                Z_elm[k * 8 : k * 8 + 8] += c
            Z[i - 1, j - 1] = Z_elm

    gemm(X, W, Z, fifo_A, fifo_B)


def test_codegen():
    code = top.schedule().export("vitis").hls_code
    assert "void top_gemm_1_1(" in code
    assert "top_gemm_0_0" not in code
    # the per-lane int8 unpack lowers to an 8-bit shift/mask, and the signed
    # product casts the lane back to int8
    assert "& 0xffULL" in code
    assert "static_cast<int8_t>" in code


def test_cpu_sim():
    np_type = np.int32
    X = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    W = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    packed_X = np.ascontiguousarray(np.ascontiguousarray(X).view(np_type))
    packed_W = np.ascontiguousarray(
        np.ascontiguousarray(W.transpose()).view(np_type).transpose()
    )
    Z = np.zeros((M // PP, N), dtype=np_type)
    top.schedule().export("cpu")(packed_X, packed_W, Z)
    np_C = X @ W
    np_C_packed = np.ascontiguousarray(
        np.ascontiguousarray(np_C.transpose()).view(np_type).transpose()
    )
    np.testing.assert_allclose(Z, np_C_packed, atol=1e-3)


@requires_vitis
def test_csim():
    np_type = np.int32
    X = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    W = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    packed_X = np.ascontiguousarray(np.ascontiguousarray(X).view(np_type))
    packed_W = np.ascontiguousarray(
        np.ascontiguousarray(W.transpose()).view(np_type).transpose()
    )
    Z = np.zeros((M // PP, N), dtype=np_type)
    with tempfile.TemporaryDirectory() as proj:
        top.schedule().export("vitis", project_path=proj)(packed_X, packed_W, Z)
    np_C = X @ W
    np_C_packed = np.ascontiguousarray(
        np.ascontiguousarray(np_C.transpose()).view(np_type).transpose()
    )
    np.testing.assert_allclose(Z, np_C_packed, atol=1e-3)
