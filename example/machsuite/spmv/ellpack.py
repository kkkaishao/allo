# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f64, i32, kernel

N = 494
L = 10


@kernel
def ellpack(NZ: "f64[N * L]", cols: "i32[N * L]", vec: "f64[N]") -> "f64[N]":
    out: "f64[N]" = 0.0

    for i in range(N):
        for j in range(L):
            idx: i32 = j + i * L
            if cols[idx] != -1:
                out[i] += NZ[idx] * vec[cols[idx]]

    return out
