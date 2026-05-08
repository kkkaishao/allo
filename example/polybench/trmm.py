# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

M = 60
N = 80
alpha = 1.5


def np_trmm(A, B):
    for i in range(M):
        for j in range(N):
            for k in range(M):
                if k > i:
                    B[i, j] += A[k, i] * B[k, j]

    B[:, :] = B * alpha
    return B


@kernel
def S0(A: "f32[M, M]", B: "f32[M, N]"):
    for i1 in range(M):
        for j1 in range(N):
            for k1 in range(M):
                if k1 > i1:
                    B[i1, j1] += A[k1, i1] * B[k1, j1]


@kernel
def S1(B: "f32[M, N]"):
    for i0 in range(M):
        for j0 in range(N):
            B[i0, j0] = B[i0, j0] * alpha


@kernel
def trmm(A: "f32[M, M]", B: "f32[M, N]"):
    S0(A, B)
    S1(B)
