# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, index, kernel

N = 120


@kernel
def ludcmp(A: "f32[N, N]", b: "f32[N]", x: "f32[N]", y: "f32[N]"):
    for i in range(N):
        for j in range(i):
            w_lower: f32 = A[i, j]
            for k in range(j):
                w_lower -= A[i, k] * A[k, j]
            A[i, j] = w_lower / A[j, j]

        for j in range(i, N):
            w_upper: f32 = A[i, j]
            for k in range(i):
                w_upper -= A[i, k] * A[k, j]
            A[i, j] = w_upper

    for i in range(N):
        alpha_y: f32 = b[i]
        for j in range(i):
            alpha_y -= A[i, j] * y[j]
        y[i] = alpha_y

    for i_inv in range(N):
        i: index = N - 1 - i_inv
        alpha_x: f32 = y[i]
        for j in range(i + 1, N):
            alpha_x -= A[i, j] * x[j]
        x[i] = alpha_x / A[i, i]
