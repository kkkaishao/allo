# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, kernel

M = 60
N = 80
alpha = 1.5
beta = 1.2


def np_syr2k(A, A_copy, B, B_copy, Cin, Cout):
    buffer = Cin.copy()
    for i in range(N):
        for j in range(N):
            if j <= i:
                buffer[i, j] = beta * Cin[i, j]

    for i in range(N):
        for k in range(M):
            for j in range(N):
                if j <= i:
                    buffer[i, j] += (
                        A[j, k] * alpha * B[i, k] + B_copy[j, k] * alpha * A_copy[i, k]
                    )

    Cout[:, :] = buffer
    return Cout


@kernel
def update_C(Cin: "f32[N, N]", Cout: "f32[N, N]"):
    for i0 in range(N):
        for j0 in range(N):
            if j0 <= i0:
                Cout[i0, j0] = beta * Cin[i0, j0]
            else:
                Cout[i0, j0] = Cin[i0, j0]


@kernel
def compute_sum(
    A: "f32[N, M]",
    A_copy: "f32[N, M]",
    B: "f32[N, M]",
    B_copy: "f32[N, M]",
    Cin: "f32[N, N]",
    Cout: "f32[N, N]",
):
    buffer: "f32[N, N]" = 0.0
    for i0 in range(N):
        for j0 in range(N):
            buffer[i0, j0] = Cin[i0, j0]

    for i1 in range(N):
        for k1 in range(M):
            for j1 in range(N):
                if j1 <= i1:
                    buffer[i1, j1] += (
                        A[j1, k1] * alpha * B[i1, k1]
                        + B_copy[j1, k1] * alpha * A_copy[i1, k1]
                    )

    for i2 in range(N):
        for j2 in range(N):
            Cout[i2, j2] = buffer[i2, j2]


@kernel
def syr2k(
    A: "f32[N, M]",
    A_copy: "f32[N, M]",
    B: "f32[N, M]",
    B_copy: "f32[N, M]",
    Cin: "f32[N, N]",
    Cout: "f32[N, N]",
):
    C: "f32[N, N]" = 0.0
    update_C(Cin, C)
    compute_sum(A, A_copy, B, B_copy, C, Cout)
