# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f64, i32, kernel

N = 494
NNZ = 1666


@kernel
def crs(
    val: "f64[NNZ]", cols: "i32[NNZ]", row: "i32[N + 1]", vec: "f64[N]"
) -> "f64[N]":
    out: "f64[N]" = 0.0

    for i in range(N):
        tmp_begin: i32 = row[i]
        tmp_end: i32 = row[i + 1]

        for j in range(tmp_begin, tmp_end):
            out[i] += val[j] * vec[cols[j]]

    return out
