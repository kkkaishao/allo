# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, grid, kernel


@kernel
def gemm(A: "f32[64, 64]", B: "f32[64, 64]") -> "f32[64, 64]":
    C: "f32[64, 64]" = 0.0
    for i, j in grid(64, 64):
        for k in range(64):
            C[i, j] += A[i, k] * B[k, j]
    return C
