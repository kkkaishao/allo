# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

M = 116
N = 124


@kernel
def stageS(A: "f32[N, M]", r: "f32[N]", s: "f32[M]"):
    for i0 in range(N):
        local_r: f32 = r[i0]
        for j0 in range(M):
            s[j0] += local_r * A[i0, j0]


@kernel
def stageQ(A: "f32[N, M]", p: "f32[M]", q: "f32[N]"):
    for i1 in range(N):
        for j1 in range(M):
            q[i1] += A[i1, j1] * p[j1]


@kernel
def bicg(
    A: "f32[N, M]",
    A_copy: "f32[N, M]",
    p: "f32[M]",
    r: "f32[N]",
    q: "f32[N]",
    s: "f32[M]",
):
    stageS(A, r, s)
    stageQ(A_copy, p, q)
