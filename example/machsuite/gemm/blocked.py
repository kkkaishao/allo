# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import i32, kernel
from .. import run_machsuite_kernel
import numpy as np

M, N, K = 64, 64, 64
S = 8


@kernel
def bbgemm(A: "i32[M, K]", B: "i32[K, N]") -> "i32[M, N]":
    C: "i32[M, N]" = 0

    i_max: i32 = 0
    j_max: i32 = 0
    k_max: i32 = 0
    sum_value: i32 = 0

    for i in range(0, M, S):
        i_max = i + S if i + S < M else M
        for j in range(0, N, S):
            j_max = j + S if j + S < N else N
            for k in range(0, K, S):
                k_max = k + S if k + S < K else K
                for ii in range(i, i_max):
                    for jj in range(j, j_max):
                        sum_value = 0
                        for kk in range(k, k_max):
                            sum_value += A[ii, kk] * B[kk, jj]
                        C[ii, jj] += sum_value
    return C


def np_bbgemm(A, B):
    return (A.astype(np.int64) @ B.astype(np.int64)).astype(np.int32)


def test_bbgemm():
    run_machsuite_kernel(bbgemm, "gemm_blocked")
