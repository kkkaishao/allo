# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

N = 180


@kernel
def floyd_warshall(path: "f32[N, N]"):
    for k in range(N):
        for i in range(N):
            for j in range(N):
                path_: f32 = path[i, k] + path[k, j]
                if path[i, j] >= path_:
                    path[i, j] = path_
