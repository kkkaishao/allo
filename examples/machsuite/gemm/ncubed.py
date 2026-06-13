# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, grid, kernel
from .. import run_machsuite_kernel
import numpy as np


@kernel
def gemm(A: "f32[64, 64]", B: "f32[64, 64]") -> "f32[64, 64]":
    C: "f32[64, 64]" = 0.0
    for i, j in grid(64, 64):
        for k in range(64):
            C[i, j] += A[i, k] * B[k, j]
    return C


def np_gemm(A, B):
    return (A @ B).astype(np.float32)


def test_gemm():
    run_machsuite_kernel(gemm, "gemm_ncubed")
