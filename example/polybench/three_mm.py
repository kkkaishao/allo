# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang import f32, kernel

P = 40
R = 50
Q = 60
T = 70
S = 80


def np_three_mm(A, B, C, D):
    return np.dot(np.dot(A, B), np.dot(C, D))


@kernel
def mm1(A: "f32[P, Q]", B: "f32[Q, R]", out_AB: "f32[P, R]"):
    for i0 in range(P):
        for j0 in range(R):
            for k0 in range(Q):
                out_AB[i0, j0] += A[i0, k0] * B[k0, j0]


@kernel
def mm2(C: "f32[R, S]", D: "f32[S, T]", out_CD: "f32[R, T]"):
    for i1 in range(R):
        for j1 in range(T):
            for k1 in range(S):
                out_CD[i1, j1] += C[i1, k1] * D[k1, j1]


@kernel
def mm3(out_AB: "f32[P, R]", out_CD: "f32[R, T]", out_ABC: "f32[P, T]"):
    for i2 in range(P):
        for j2 in range(T):
            for k2 in range(R):
                out_ABC[i2, j2] += out_AB[i2, k2] * out_CD[k2, j2]


@kernel
def three_mm(
    A: "f32[P, Q]",
    B: "f32[Q, R]",
    C: "f32[R, S]",
    D: "f32[S, T]",
) -> "f32[P, T]":
    out_AB: "f32[P, R]" = 0.0
    out_CD: "f32[R, T]" = 0.0
    output: "f32[P, T]" = 0.0
    mm1(A, B, out_AB)
    mm2(C, D, out_CD)
    mm3(out_AB, out_CD, output)
    return output
