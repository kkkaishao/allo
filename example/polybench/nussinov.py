# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, index, kernel

N = 180


def np_nussinov(seq, table):
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                if table[i, j] < table[i, j - 1]:
                    table[i, j] = table[i, j - 1]

            if i + 1 < N:
                if table[i, j] < table[i + 1, j]:
                    table[i, j] = table[i + 1, j]

            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    w = seq[i] + seq[j]

                    match = 0.0
                    if w == 3.0:
                        match = 1.0

                    s2 = table[i + 1, j - 1] + match
                    if table[i, j] < s2:
                        table[i, j] = s2
                else:
                    if table[i, j] < table[i + 1, j - 1]:
                        table[i, j] = table[i + 1, j - 1]

            for k in range(i + 1, j):
                s3 = table[i, k] + table[k + 1, j]
                if table[i, j] < s3:
                    table[i, j] = s3
    return table


@kernel
def nussinov(seq: "f32[N]", table: "f32[N, N]"):
    for i_inv in range(N):
        i: index = N - 1 - i_inv
        for j in range(i + 1, N):
            if j - 1 >= 0:
                if table[i, j] < table[i, j - 1]:
                    table[i, j] = table[i, j - 1]

            if i + 1 < N:
                if table[i, j] < table[i + 1, j]:
                    table[i, j] = table[i + 1, j]

            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    w: f32 = seq[i] + seq[j]

                    match: f32 = 0.0
                    if w == 3.0:
                        match = 1.0

                    s2: f32 = table[i + 1, j - 1] + match
                    if table[i, j] < s2:
                        table[i, j] = s2
                else:
                    if table[i, j] < table[i + 1, j - 1]:
                        table[i, j] = table[i + 1, j - 1]

            for k in range(i + 1, j):
                s3: f32 = table[i, k] + table[k + 1, j]
                if table[i, j] < s3:
                    table[i, j] = s3
