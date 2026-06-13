# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, kernel

M = 60
N = 80


def np_gramschmidt(A, Q, R):
    for k in range(N):
        nrm = 0.0
        for i in range(M):
            nrm += A[i, k] * A[i, k]
        R[k, k] = nrm

        for i in range(M):
            Q[i, k] = A[i, k] / R[k, k]

        for j in range(k + 1, N):
            R[k, j] = 0.0
            for i in range(M):
                R[k, j] += Q[i, k] * A[i, j]

            for i in range(M):
                A[i, j] -= Q[i, k] * R[k, j]
    return A, Q, R


@kernel
def gramschmidt(A: "f32[M, N]", Q: "f32[M, N]", R: "f32[N, N]"):
    for k in range(N):
        nrm: f32 = 0.0
        for i in range(M):
            nrm += A[i, k] * A[i, k]
        R[k, k] = nrm

        for i in range(M):
            Q[i, k] = A[i, k] / R[k, k]

        for j in range(k + 1, N):
            R[k, j] = 0.0
            for i in range(M):
                R[k, j] += Q[i, k] * A[i, j]

            for i in range(M):
                A[i, j] -= Q[i, k] * R[k, j]
