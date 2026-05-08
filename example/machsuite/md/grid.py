# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f64, i32, grid, kernel
from .. import run_machsuite_kernel
import numpy as np

nAtoms = 64
domainEdge = 20.0
blockSide = 3
nBlocks = blockSide * blockSide * blockSide
blockEdge = domainEdge / blockSide
densityFactor = 4
lj1 = 1.5
lj2 = 2.0


@kernel
def md_x(
    n_points: "i32[blockSide, blockSide, blockSide]",
    position_x: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_y: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_z: "f64[blockSide, blockSide, blockSide, densityFactor]",
) -> "f64[blockSide, blockSide, blockSide, densityFactor]":
    force_x: "f64[blockSide, blockSide, blockSide, densityFactor]" = 0.0

    for b0_x, b0_y, b0_z in grid(blockSide, blockSide, blockSide):
        base_q_x: "f64[densityFactor]" = 0.0
        base_q_y: "f64[densityFactor]" = 0.0
        base_q_z: "f64[densityFactor]" = 0.0

        for b1_x in range(max(0, b0_x - 1), min(blockSide, b0_x + 2)):
            for b1_y in range(max(0, b0_y - 1), min(blockSide, b0_y + 2)):
                for b1_z in range(max(0, b0_z - 1), min(blockSide, b0_z + 2)):
                    q_idx_range: i32 = n_points[b1_x, b1_y, b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x, b1_y, b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x, b1_y, b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x, b1_y, b1_z, q_idx]
                    for p_idx in range(n_points[b0_x, b0_y, b0_z]):
                        p_x: f64 = position_x[b0_x, b0_y, b0_z, p_idx]
                        p_y: f64 = position_y[b0_x, b0_y, b0_z, p_idx]
                        p_z: f64 = position_z[b0_x, b0_y, b0_z, p_idx]
                        sum_x: f64 = force_x[b0_x, b0_y, b0_z, p_idx]

                        for q_idx in range(q_idx_range):
                            q_x: f64 = base_q_x[q_idx]
                            q_y: f64 = base_q_y[q_idx]
                            q_z: f64 = base_q_z[q_idx]
                            if q_x != p_x or q_y != p_y or q_z != p_z:
                                dx: f64 = p_x - q_x
                                dy: f64 = p_y - q_y
                                dz: f64 = p_z - q_z
                                r2inv: f64 = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv: f64 = r2inv * r2inv * r2inv
                                potential: f64 = r6inv * (lj1 * r6inv - lj2)
                                f: f64 = r2inv * potential
                                sum_x += f * dx

                        force_x[b0_x, b0_y, b0_z, p_idx] = sum_x
    return force_x


@kernel
def md_y(
    n_points: "i32[blockSide, blockSide, blockSide]",
    position_x: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_y: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_z: "f64[blockSide, blockSide, blockSide, densityFactor]",
) -> "f64[blockSide, blockSide, blockSide, densityFactor]":
    force_y: "f64[blockSide, blockSide, blockSide, densityFactor]" = 0.0

    for b0_x, b0_y, b0_z in grid(blockSide, blockSide, blockSide):
        base_q_x: "f64[densityFactor]" = 0.0
        base_q_y: "f64[densityFactor]" = 0.0
        base_q_z: "f64[densityFactor]" = 0.0

        for b1_x in range(max(0, b0_x - 1), min(blockSide, b0_x + 2)):
            for b1_y in range(max(0, b0_y - 1), min(blockSide, b0_y + 2)):
                for b1_z in range(max(0, b0_z - 1), min(blockSide, b0_z + 2)):
                    q_idx_range: i32 = n_points[b1_x, b1_y, b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x, b1_y, b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x, b1_y, b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x, b1_y, b1_z, q_idx]
                    for p_idx in range(n_points[b0_x, b0_y, b0_z]):
                        p_x: f64 = position_x[b0_x, b0_y, b0_z, p_idx]
                        p_y: f64 = position_y[b0_x, b0_y, b0_z, p_idx]
                        p_z: f64 = position_z[b0_x, b0_y, b0_z, p_idx]
                        sum_y: f64 = force_y[b0_x, b0_y, b0_z, p_idx]

                        for q_idx in range(q_idx_range):
                            q_x: f64 = base_q_x[q_idx]
                            q_y: f64 = base_q_y[q_idx]
                            q_z: f64 = base_q_z[q_idx]
                            if q_x != p_x or q_y != p_y or q_z != p_z:
                                dx: f64 = p_x - q_x
                                dy: f64 = p_y - q_y
                                dz: f64 = p_z - q_z
                                r2inv: f64 = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv: f64 = r2inv * r2inv * r2inv
                                potential: f64 = r6inv * (lj1 * r6inv - lj2)
                                f: f64 = r2inv * potential
                                sum_y += f * dy

                        force_y[b0_x, b0_y, b0_z, p_idx] = sum_y
    return force_y


@kernel
def md_z(
    n_points: "i32[blockSide, blockSide, blockSide]",
    position_x: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_y: "f64[blockSide, blockSide, blockSide, densityFactor]",
    position_z: "f64[blockSide, blockSide, blockSide, densityFactor]",
) -> "f64[blockSide, blockSide, blockSide, densityFactor]":
    force_z: "f64[blockSide, blockSide, blockSide, densityFactor]" = 0.0

    for b0_x, b0_y, b0_z in grid(blockSide, blockSide, blockSide):
        base_q_x: "f64[densityFactor]" = 0.0
        base_q_y: "f64[densityFactor]" = 0.0
        base_q_z: "f64[densityFactor]" = 0.0

        for b1_x in range(max(0, b0_x - 1), min(blockSide, b0_x + 2)):
            for b1_y in range(max(0, b0_y - 1), min(blockSide, b0_y + 2)):
                for b1_z in range(max(0, b0_z - 1), min(blockSide, b0_z + 2)):
                    q_idx_range: i32 = n_points[b1_x, b1_y, b1_z]
                    for q_idx in range(densityFactor):
                        base_q_x[q_idx] = position_x[b1_x, b1_y, b1_z, q_idx]
                        base_q_y[q_idx] = position_y[b1_x, b1_y, b1_z, q_idx]
                        base_q_z[q_idx] = position_z[b1_x, b1_y, b1_z, q_idx]
                    for p_idx in range(n_points[b0_x, b0_y, b0_z]):
                        p_x: f64 = position_x[b0_x, b0_y, b0_z, p_idx]
                        p_y: f64 = position_y[b0_x, b0_y, b0_z, p_idx]
                        p_z: f64 = position_z[b0_x, b0_y, b0_z, p_idx]
                        sum_z: f64 = force_z[b0_x, b0_y, b0_z, p_idx]

                        for q_idx in range(q_idx_range):
                            q_x: f64 = base_q_x[q_idx]
                            q_y: f64 = base_q_y[q_idx]
                            q_z: f64 = base_q_z[q_idx]
                            if q_x != p_x or q_y != p_y or q_z != p_z:
                                dx: f64 = p_x - q_x
                                dy: f64 = p_y - q_y
                                dz: f64 = p_z - q_z
                                r2inv: f64 = 1.0 / (dx * dx + dy * dy + dz * dz)
                                r6inv: f64 = r2inv * r2inv * r2inv
                                potential: f64 = r6inv * (lj1 * r6inv - lj2)
                                f: f64 = r2inv * potential
                                sum_z += f * dz

                        force_z[b0_x, b0_y, b0_z, p_idx] = sum_z
    return force_z


def np_md_x(n_points, position_x, position_y, position_z):
    return _np_md(n_points, position_x, position_y, position_z)[0]


def np_md_y(n_points, position_x, position_y, position_z):
    return _np_md(n_points, position_x, position_y, position_z)[1]


def np_md_z(n_points, position_x, position_y, position_z):
    return _np_md(n_points, position_x, position_y, position_z)[2]


def _np_md(n_points, position_x, position_y, position_z):
    force_x = np.zeros(
        (blockSide, blockSide, blockSide, densityFactor), dtype=np.float64
    )
    force_y = np.zeros_like(force_x)
    force_z = np.zeros_like(force_x)

    for b0_x in range(blockSide):
        for b0_y in range(blockSide):
            for b0_z in range(blockSide):
                for b1_x in range(max(0, b0_x - 1), min(blockSide, b0_x + 2)):
                    for b1_y in range(max(0, b0_y - 1), min(blockSide, b0_y + 2)):
                        for b1_z in range(max(0, b0_z - 1), min(blockSide, b0_z + 2)):
                            q_idx_range = n_points[b1_x, b1_y, b1_z]
                            for p_idx in range(n_points[b0_x, b0_y, b0_z]):
                                p_x = position_x[b0_x, b0_y, b0_z, p_idx]
                                p_y = position_y[b0_x, b0_y, b0_z, p_idx]
                                p_z = position_z[b0_x, b0_y, b0_z, p_idx]
                                sum_x = force_x[b0_x, b0_y, b0_z, p_idx]
                                sum_y = force_y[b0_x, b0_y, b0_z, p_idx]
                                sum_z = force_z[b0_x, b0_y, b0_z, p_idx]

                                for q_idx in range(q_idx_range):
                                    q_x = position_x[b1_x, b1_y, b1_z, q_idx]
                                    q_y = position_y[b1_x, b1_y, b1_z, q_idx]
                                    q_z = position_z[b1_x, b1_y, b1_z, q_idx]
                                    if q_x != p_x or q_y != p_y or q_z != p_z:
                                        dx = p_x - q_x
                                        dy = p_y - q_y
                                        dz = p_z - q_z
                                        r2inv = 1.0 / (dx * dx + dy * dy + dz * dz)
                                        r6inv = r2inv * r2inv * r2inv
                                        potential = r6inv * (lj1 * r6inv - lj2)
                                        f = r2inv * potential
                                        sum_x += f * dx
                                        sum_y += f * dy
                                        sum_z += f * dz

                                force_x[b0_x, b0_y, b0_z, p_idx] = sum_x
                                force_y[b0_x, b0_y, b0_z, p_idx] = sum_y
                                force_z[b0_x, b0_y, b0_z, p_idx] = sum_z
    return force_x, force_y, force_z


def test_md_x():
    run_machsuite_kernel(md_x, "md_grid_x")


def test_md_y():
    run_machsuite_kernel(md_y, "md_grid_y")


def test_md_z():
    run_machsuite_kernel(md_z, "md_grid_z")
