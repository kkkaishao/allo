# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang import f32, kernel

M = 116
N = 124


def np_atax(A, x, y):
    out_Ax = np.dot(A, x)
    y += np.dot(A.T, out_Ax)
    return y


@kernel
def stage_M(A: "f32[M, N]", x: "f32[N]", out_Ax: "f32[M]"):
    for m in range(M):
        for r in range(N):
            out_Ax[m] += A[m, r] * x[r]


@kernel
def stage_N(A: "f32[M, N]", out_Ax: "f32[M]", y: "f32[N]"):
    for n in range(N):
        for k in range(M):
            y[n] += A[k, n] * out_Ax[k]


@kernel
def atax(A: "f32[M, N]", x: "f32[N]", y: "f32[N]"):
    out_Ax: "f32[M]" = 0.0
    stage_M(A, x, out_Ax)
    stage_N(A, out_Ax, y)
