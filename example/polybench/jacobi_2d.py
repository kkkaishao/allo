# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

TSTEPS = 40
N = 90


def np_jacobi_2d(A, B):
    for m in range(TSTEPS):
        for i0 in range(N - 2):
            for j0 in range(N - 2):
                B[i0 + 1, j0 + 1] = 0.2 * (
                    A[i0, j0 + 1]
                    + A[i0 + 1, j0]
                    + A[i0 + 1, j0 + 1]
                    + A[i0 + 1, j0 + 2]
                    + A[i0 + 2, j0 + 1]
                )

        for i1 in range(N - 2):
            for j1 in range(N - 2):
                A[i1 + 1, j1 + 1] = 0.2 * (
                    B[i1, j1 + 1]
                    + B[i1 + 1, j1]
                    + B[i1 + 1, j1 + 1]
                    + B[i1 + 1, j1 + 2]
                    + B[i1 + 2, j1 + 1]
                )
    return A, B


@kernel
def compute_A(A0: "f32[N, N]", B0: "f32[N, N]"):
    for i0 in range(N - 2):
        for j0 in range(N - 2):
            B0[i0 + 1, j0 + 1] = 0.2 * (
                A0[i0, j0 + 1]
                + A0[i0 + 1, j0]
                + A0[i0 + 1, j0 + 1]
                + A0[i0 + 1, j0 + 2]
                + A0[i0 + 2, j0 + 1]
            )


@kernel
def compute_B(B1: "f32[N, N]", A1: "f32[N, N]"):
    for i1 in range(N - 2):
        for j1 in range(N - 2):
            A1[i1 + 1, j1 + 1] = 0.2 * (
                B1[i1, j1 + 1]
                + B1[i1 + 1, j1]
                + B1[i1 + 1, j1 + 1]
                + B1[i1 + 1, j1 + 2]
                + B1[i1 + 2, j1 + 1]
            )


@kernel
def jacobi_2d(A: "f32[N, N]", B: "f32[N, N]"):
    for m in range(TSTEPS):
        compute_A(A, B)
        compute_B(B, A)
