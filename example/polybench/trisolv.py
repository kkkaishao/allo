# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

N = 120


@kernel
def trisolv(L: "f32[N, N]", b: "f32[N]", x: "f32[N]"):
    for i in range(N):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
