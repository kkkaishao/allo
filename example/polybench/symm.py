# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, kernel

M = 60
N = 80
alpha = 1.5
beta = 1.2


def np_symm(A0, A1, B0, B1, C):
    summ = C * 0.0
    for i1 in range(M):
        for j1 in range(N):
            for k1 in range(M):
                if k1 < i1:
                    summ[i1, j1] += B0[k1, j1] * A0[i1, k1]

    for i in range(M):
        for k in range(i):
            for j in range(N):
                C[k, j] = C[k, j] + alpha * B1[i, j] * A1[i, k]
        for j1 in range(N):
            C[i, j1] = (
                beta * C[i, j1] + alpha * B1[i, j1] * A1[i, i] + alpha * summ[i, j1]
            )
    return C


@kernel
def compute_sum(A: "f32[M, M]", B: "f32[M, N]", summ: "f32[M, N]"):
    for i1 in range(M):
        for j1 in range(N):
            for k1 in range(M):
                if k1 < i1:
                    summ[i1, j1] += B[k1, j1] * A[i1, k1]


@kernel
def update_C(
    A: "f32[M, M]",
    B: "f32[M, N]",
    summ: "f32[M, N]",
    C: "f32[M, N]",
):
    for i in range(M):
        for k in range(i):
            for j in range(N):
                C[k, j] = C[k, j] + alpha * B[i, j] * A[i, k]
        for j1 in range(N):
            C[i, j1] = (
                beta * C[i, j1] + alpha * B[i, j1] * A[i, i] + alpha * summ[i, j1]
            )


@kernel
def symm(
    A0: "f32[M, M]",
    A1: "f32[M, M]",
    B0: "f32[M, N]",
    B1: "f32[M, N]",
    C: "f32[M, N]",
):
    summ: "f32[M, N]" = 0.0
    compute_sum(A0, B0, summ)
    update_C(A1, B1, summ, C)
