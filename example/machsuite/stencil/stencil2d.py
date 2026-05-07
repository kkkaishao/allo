# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import grid, i32, kernel

col_size = 64
row_size = 128
f_size = 9


@kernel
def stencil2d(
    orig: "i32[row_size, col_size]", filter: "i32[f_size]"
) -> "i32[row_size, col_size]":
    sol: "i32[row_size, col_size]" = 0

    for i, j in grid(row_size - 2, col_size - 2):
        temp: i32 = 0
        for m in range(3):
            for n in range(3):
                mul: i32 = filter[m * 3 + n] * orig[i + m, j + n]
                temp += mul
        sol[i, j] = temp

    return sol
