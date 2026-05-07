# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f64, i32, kernel

nAtoms = 256
maxNeighbors = 16
lj1 = 1.5
lj2 = 2.0
domainEdge = 20.0


@kernel
def md_x(
    position_x: "f64[nAtoms]",
    position_y: "f64[nAtoms]",
    position_z: "f64[nAtoms]",
    NL: "i32[nAtoms * maxNeighbors]",
) -> "f64[nAtoms]":
    i_x: f64 = 0.0
    i_y: f64 = 0.0
    i_z: f64 = 0.0
    jidx: i32 = 0
    j_x: f64 = 0.0
    j_y: f64 = 0.0
    j_z: f64 = 0.0
    delx: f64 = 0.0
    dely: f64 = 0.0
    delz: f64 = 0.0
    r2inv: f64 = 0.0
    r6inv: f64 = 0.0
    potential: f64 = 0.0
    force: f64 = 0.0
    fx: f64 = 0.0
    force_x: "f64[nAtoms]" = 0.0

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fx = 0.0

        for j in range(maxNeighbors):
            jidx = NL[i * maxNeighbors + j]
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            force = r2inv * potential
            fx = fx + delx * force
        force_x[i] = fx
    return force_x


@kernel
def md_y(
    position_x: "f64[nAtoms]",
    position_y: "f64[nAtoms]",
    position_z: "f64[nAtoms]",
    NL: "i32[nAtoms * maxNeighbors]",
) -> "f64[nAtoms]":
    i_x: f64 = 0.0
    i_y: f64 = 0.0
    i_z: f64 = 0.0
    jidx: i32 = 0
    j_x: f64 = 0.0
    j_y: f64 = 0.0
    j_z: f64 = 0.0
    delx: f64 = 0.0
    dely: f64 = 0.0
    delz: f64 = 0.0
    r2inv: f64 = 0.0
    r6inv: f64 = 0.0
    potential: f64 = 0.0
    force: f64 = 0.0
    fy: f64 = 0.0
    force_y: "f64[nAtoms]"

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fy = 0.0

        for j in range(maxNeighbors):
            jidx = NL[i * maxNeighbors + j]
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            force = r2inv * potential
            fy = fy + dely * force
        force_y[i] = fy
    return force_y


@kernel
def md_z(
    position_x: "f64[nAtoms]",
    position_y: "f64[nAtoms]",
    position_z: "f64[nAtoms]",
    NL: "i32[nAtoms * maxNeighbors]",
) -> "f64[nAtoms]":
    i_x: f64 = 0.0
    i_y: f64 = 0.0
    i_z: f64 = 0.0
    jidx: i32 = 0
    j_x: f64 = 0.0
    j_y: f64 = 0.0
    j_z: f64 = 0.0
    delx: f64 = 0.0
    dely: f64 = 0.0
    delz: f64 = 0.0
    r2inv: f64 = 0.0
    r6inv: f64 = 0.0
    potential: f64 = 0.0
    force: f64 = 0.0
    fz: f64 = 0.0
    force_z: "f64[nAtoms]"

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fz = 0.0

        for j in range(maxNeighbors):
            jidx = NL[i * maxNeighbors + j]
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            force = r2inv * potential
            fz = fz + delz * force
        force_z[i] = fz
    return force_z
