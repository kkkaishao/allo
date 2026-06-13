# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f64, i32, kernel
from .. import run_machsuite_kernel
import numpy as np

N = 64
NNZ = 192


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


def np_crs(val, cols, row, vec):
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(row[i], row[i + 1]):
            out[i] += val[j] * vec[cols[j]]
    return out


def test_crs():
    run_machsuite_kernel(crs, "spmv_crs")
