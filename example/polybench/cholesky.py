# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang import f32, kernel
from allo.operators import math as amath

N = 120


def np_cholesky(A):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A[i, j] = A[i, j] / A[j, j]
        for k in range(i):
            A[i, i] = A[i, i] - A[i, k] * A[i, k]
        A[i, i] = np.sqrt(A[i, i] * 1.0)
    return A


@kernel
def cholesky(A: "f32[N, N]"):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A[i, j] = A[i, j] / A[j, j]
        for k in range(i):
            A[i, i] = A[i, i] - A[i, k] * A[i, k]
        A[i, i] = amath.sqrt(A[i, i] * 1.0)
