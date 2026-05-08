# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

Tmax = 40
Nx = 60
Ny = 80


def np_fdtd_2d(ex, ey, hz, fict):
    for m in range(Tmax):
        for j in range(Ny):
            ey[0, j] = fict[m]

        for i in range(1, Nx):
            for j in range(Ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])

        for i in range(Nx):
            for j in range(1, Ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])

        for i in range(Nx - 1):
            for j in range(Ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                )
    return ex, ey, hz


@kernel
def fdtd_2d(
    ex: "f32[Nx, Ny]",
    ey: "f32[Nx, Ny]",
    hz: "f32[Nx, Ny]",
    fict: "f32[Tmax]",
):
    for m in range(Tmax):
        for j in range(Ny):
            ey[0, j] = fict[m]

        for i in range(1, Nx):
            for j in range(Ny):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])

        for i in range(Nx):
            for j in range(1, Ny):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])

        for i in range(Nx - 1):
            for j in range(Ny - 1):
                hz[i, j] = hz[i, j] - 0.7 * (
                    ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                )
