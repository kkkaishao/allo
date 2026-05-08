# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

TSTEPS = 40
N = 120


def np_jacobi_1d(A, B):
    for m in range(TSTEPS):
        for i0 in range(1, N - 1):
            B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])

        for i1 in range(1, N - 1):
            A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])
    return A, B


@kernel
def jacobi_1d(A: "f32[N]", B: "f32[N]"):
    for m in range(TSTEPS):
        for i0 in range(1, N - 1):
            B[i0] = 0.33333 * (A[i0 - 1] + A[i0] + A[i0 + 1])

        for i1 in range(1, N - 1):
            A[i1] = 0.33333 * (B[i1 - 1] + B[i1] + B[i1 + 1])
