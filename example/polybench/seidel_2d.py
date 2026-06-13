# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, kernel

TSTEPS = 40
N = 120


def np_seidel_2d(A):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A[i, j] = (
                    A[i - 1, j - 1]
                    + A[i - 1, j]
                    + A[i - 1, j + 1]
                    + A[i, j - 1]
                    + A[i, j]
                    + A[i, j + 1]
                    + A[i + 1, j - 1]
                    + A[i + 1, j]
                    + A[i + 1, j + 1]
                ) / 9.0
    return A


@kernel
def seidel_2d(A: "f32[N, N]"):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A[i, j] = (
                    A[i - 1, j - 1]
                    + A[i - 1, j]
                    + A[i - 1, j + 1]
                    + A[i, j - 1]
                    + A[i, j]
                    + A[i, j + 1]
                    + A[i + 1, j - 1]
                    + A[i + 1, j]
                    + A[i + 1, j + 1]
                ) / 9.0
