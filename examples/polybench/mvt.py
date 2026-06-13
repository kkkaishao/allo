# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang import f32, kernel

N = 120


def np_mvt(A, A_copy, y1, y2, x1, x2, x1_out, x2_out):
    x1_out[:] = x1 + np.dot(A, y1)
    x2_out[:] = x2 + np.dot(A_copy.T, y2)
    return x1_out, x2_out


@kernel
def stageA(x1_in: "f32[N]", x1_out: "f32[N]", A: "f32[N, N]", y1: "f32[N]"):
    for i0 in range(N):
        x: f32 = x1_in[i0]
        for j0 in range(N):
            x += A[i0, j0] * y1[j0]
        x1_out[i0] = x


@kernel
def stageB(x2_in: "f32[N]", x2_out: "f32[N]", A: "f32[N, N]", y2: "f32[N]"):
    for i1 in range(N):
        x: f32 = x2_in[i1]
        for j1 in range(N):
            x += A[j1, i1] * y2[j1]
        x2_out[i1] = x


@kernel
def mvt(
    A: "f32[N, N]",
    A_copy: "f32[N, N]",
    y1: "f32[N]",
    y2: "f32[N]",
    x1: "f32[N]",
    x2: "f32[N]",
    x1_out: "f32[N]",
    x2_out: "f32[N]",
):
    stageA(x1, x1_out, A, y1)
    stageB(x2, x2_out, A_copy, y2)
