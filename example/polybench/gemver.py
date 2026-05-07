# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

N = 120
alpha = 0.1
beta = 0.1


@kernel
def gemver(
    A: "f32[N, N]",
    u1: "f32[N]",
    u2: "f32[N]",
    v1: "f32[N]",
    v2: "f32[N]",
    x: "f32[N]",
    y: "f32[N]",
    w: "f32[N]",
    z: "f32[N]",
):
    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in range(N):
        for j in range(N):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in range(N):
        x[i] = x[i] + z[i]

    for i in range(N):
        for j in range(N):
            w[i] = w[i] + alpha * A[i, j] * x[j]
