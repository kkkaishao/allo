# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f64, i32, kernel
from .. import run_machsuite_kernel
import numpy as np

N = 64
L = 4


@kernel
def ellpack(NZ: "f64[N * L]", cols: "i32[N * L]", vec: "f64[N]") -> "f64[N]":
    out: "f64[N]" = 0.0

    for i in range(N):
        for j in range(L):
            idx: i32 = j + i * L
            if cols[idx] != -1:
                out[i] += NZ[idx] * vec[cols[idx]]

    return out


def np_ellpack(NZ, cols, vec):
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(L):
            idx = j + i * L
            if cols[idx] != -1:
                out[i] += NZ[idx] * vec[cols[idx]]
    return out


def test_ellpack():
    run_machsuite_kernel(ellpack, "spmv_ellpack")
