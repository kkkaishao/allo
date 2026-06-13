# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang import f32, kernel

P = 40
R = 50
Q = 70
S = 80
alpha = 0.1
beta = 0.5


def np_two_mm(A, B, C, D):
    out_AB = np.dot(A, B)
    out_ABC = np.dot(out_AB, C)
    return out_ABC * beta + D * alpha


@kernel
def mm1(A: "f32[P, Q]", B: "f32[Q, R]", out_AB: "f32[P, R]"):
    for i0 in range(P):
        for j0 in range(R):
            for k0 in range(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


@kernel
def mm2(out_AB: "f32[P, R]", C: "f32[R, S]", out_ABC: "f32[P, S]"):
    for i1 in range(P):
        for j1 in range(S):
            for k1 in range(R):
                out_ABC[i1, j1] += out_AB[i1, k1] * C[k1, j1]


@kernel
def ele_add(out_ABC: "f32[P, S]", D: "f32[P, S]", output: "f32[P, S]"):
    for i2 in range(P):
        for j2 in range(S):
            output[i2, j2] = out_ABC[i2, j2] * beta + D[i2, j2] * alpha


@kernel
def two_mm(
    A: "f32[P, Q]",
    B: "f32[Q, R]",
    C: "f32[R, S]",
    D: "f32[P, S]",
) -> "f32[P, S]":
    out_AB: "f32[P, R]" = 0.0
    out_ABC: "f32[P, S]" = 0.0
    output: "f32[P, S]"
    mm1(A, B, out_AB)
    mm2(out_AB, C, out_ABC)
    ele_add(out_ABC, D, output)
    return output
