# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

N = 120


def np_lu(A):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]

        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]
    return A


@kernel
def lu(A: "f32[N, N]"):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]

        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]
