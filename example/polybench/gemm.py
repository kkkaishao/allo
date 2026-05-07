# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

P = 60
R = 70
Q = 80
beta = 0.1


@kernel
def mm1(A: "f32[P, Q]", B: "f32[Q, R]", out_AB: "f32[P, R]"):
    for i0 in range(P):
        for j0 in range(R):
            for k0 in range(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


@kernel
def ele_add(out_AB: "f32[P, R]", C: "f32[P, R]", output: "f32[P, R]"):
    for i2 in range(P):
        for j2 in range(R):
            output[i2, j2] = beta * C[i2, j2] + out_AB[i2, j2]


@kernel
def gemm(
    A: "f32[P, Q]",
    B: "f32[Q, R]",
    C: "f32[P, R]",
    output: "f32[P, R]",
):
    out_AB: "f32[P, R]" = 0.0
    mm1(A, B, out_AB)
    ele_add(out_AB, C, output)
