# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

TSTEPS = 40
N = 20


@kernel
def heat_3d(A: "f32[N, N, N]", B: "f32[N, N, N]"):
    const0: f32 = 0.125
    const1: f32 = 2.0

    for m in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        const0 * (A[i + 1, j, k] - const1 * A[i, j, k] + A[i - 1, j, k])
                        + const0
                        * (A[i, j + 1, k] - const1 * A[i, j, k] + A[i, j - 1, k])
                        + const0
                        * (A[i, j, k + 1] - const1 * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )

                    A[i, j, k] = (
                        const0 * (B[i + 1, j, k] - const1 * B[i, j, k] + B[i - 1, j, k])
                        + const0
                        * (B[i, j + 1, k] - const1 * B[i, j, k] + B[i, j - 1, k])
                        + const0
                        * (B[i, j, k + 1] - const1 * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )
