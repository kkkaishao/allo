# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import grid, i32, kernel

height_size = 32
col_size = 32
row_size = 16


@kernel
def stencil3d(
    C: "i32[2]", orig: "i32[row_size, col_size, height_size]"
) -> "i32[row_size, col_size, height_size]":
    sol: "i32[row_size, col_size, height_size]" = 0

    for j, k in grid(col_size, row_size):
        sol[k, j, 0] = orig[k, j, 0]
        sol[k, j, height_size - 1] = orig[k, j, height_size - 1]

    for i, k in grid(height_size - 1, row_size):
        sol[k, 0, i + 1] = orig[k, 0, i + 1]
        sol[k, col_size - 1, i + 1] = orig[k, col_size - 1, i + 1]

    for j, i in grid(col_size - 2, height_size - 2):
        sol[0, j + 1, i + 1] = orig[0, j + 1, i + 1]
        sol[row_size - 1, j + 1, i + 1] = orig[row_size - 1, j + 1, i + 1]

    for i, j, k in grid(height_size - 2, col_size - 2, row_size - 2):
        sum0: i32 = orig[k + 1, j + 1, i + 1]
        sum1: i32 = (
            orig[k + 1, j + 1, i + 2]
            + orig[k + 1, j + 1, i]
            + orig[k + 1, j + 2, i + 1]
            + orig[k + 1, j, i + 1]
            + orig[k + 2, j + 1, i + 1]
            + orig[k, j + 1, i + 1]
        )
        mul0: i32 = sum0 * C[0]
        mul1: i32 = sum1 * C[1]
        sol[k + 1, j + 1, i + 1] = mul0 + mul1

    return sol
