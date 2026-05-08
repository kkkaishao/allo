# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, index, kernel

TSTEPS = 4
N = 5

DX = 1.0 / N
DY = 1.0 / N
DT = 1.0 / TSTEPS
B1 = 2.0
B2 = 1.0
mul1 = B1 * DT / (DX * DX)
mul2 = B2 * DT / (DY * DY)

a = -mul1 / 2.0
b = 1.0 + mul1
c = a
d = -mul2 / 2.0
e = 1.0 + mul2
f = d


def np_adi(u, v, p, q):
    for t in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            v[0, i] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = v[0, i]
            for j in range(1, N - 1):
                p[i, j] = -c / (a * p[i, j - 1] + b)
                q[i, j] = (
                    -d * u[j, i - 1]
                    + (1.0 + 2.0 * d) * u[j, i]
                    - f * u[j, i + 1]
                    - a * q[i, j - 1]
                ) / (a * p[i, j - 1] + b)

            v[N - 1, i] = 1.0
            for j in range(N - 2, -1, -1):
                v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]

        for i in range(1, N - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = u[i, 0]
            for j in range(1, N - 1):
                p[i, j] = -f / (d * p[i, j - 1] + e)
                q[i, j] = (
                    -a * v[i - 1, j]
                    + (1.0 + 2.0 * a) * v[i, j]
                    - c * v[i + 1, j]
                    - d * q[i, j - 1]
                ) / (d * p[i, j - 1] + e)
            u[i, N - 1] = 1.0
            for j in range(N - 2, -1, -1):
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]

    return u, v, p, q


@kernel
def adi(
    u: "f32[N, N]",
    v: "f32[N, N]",
    p: "f32[N, N]",
    q: "f32[N, N]",
):
    for t in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            v[0, i] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = v[0, i]
            for j in range(1, N - 1):
                p[i, j] = -c / (a * p[i, j - 1] + b)
                q[i, j] = (
                    -d * u[j, i - 1]
                    + (1.0 + 2.0 * d) * u[j, i]
                    - f * u[j, i + 1]
                    - a * q[i, j - 1]
                ) / (a * p[i, j - 1] + b)

            v[N - 1, i] = 1.0
            for j_rev in range(N - 1):
                j: index = N - 2 - j_rev
                v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]
        for i in range(1, N - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = u[i, 0]
            for j in range(1, N - 1):
                p[i, j] = -f / (d * p[i, j - 1] + e)
                q[i, j] = (
                    -a * v[i - 1, j]
                    + (1.0 + 2.0 * a) * v[i, j]
                    - c * v[i + 1, j]
                    - d * q[i, j - 1]
                ) / (d * p[i, j - 1] + e)
            u[i, N - 1] = 1.0
            for j_rev in range(N - 1):
                j: index = N - 2 - j_rev
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]
