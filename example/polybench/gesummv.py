# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

N = 90
alpha = 0.1
beta = 0.1


@kernel
def compute_tmp(
    y_in: "f32[N]",
    y_out: "f32[N]",
    A: "f32[N, N]",
    B: "f32[N, N]",
    x: "f32[N]",
    tmp: "f32[N]",
):
    tt: "f32[N]" = 0.0
    yy: "f32[N]"
    for i0 in range(N):
        yy[i0] = y_in[i0]

    for i in range(N):
        for j in range(N):
            tt[i] += A[i, j] * x[j]
            yy[i] += B[i, j] * x[j]

    for i1 in range(N):
        tmp[i1] = tt[i1]
        y_out[i1] = yy[i1]


@kernel
def compute_y(y_in: "f32[N]", y_out: "f32[N]", tmp: "f32[N]"):
    for i0 in range(N):
        y_out[i0] = alpha * tmp[i0] + beta * y_in[i0]


@kernel
def gesummv(A: "f32[N, N]", B: "f32[N, N]", x: "f32[N]", y: "f32[N]"):
    y_init: "f32[N]" = 0.0
    y_fifo: "f32[N]"
    tmp: "f32[N]"
    compute_tmp(y_init, y_fifo, A, B, x, tmp)
    compute_y(y_fifo, y, tmp)
