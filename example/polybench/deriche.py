# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from allo.exp.lang import f32, index, kernel

W = 192
H = 128

alpha = 0.25
k = (
    (1.0 - math.exp(-alpha))
    * (1.0 - math.exp(-alpha))
    / (1.0 + 2.0 * alpha * math.exp(-alpha) - math.exp(2.0 * alpha))
)
a1 = k
a2 = k * math.exp(-alpha) * (alpha - 1.0)
a3 = k * math.exp(-alpha) * (alpha + 1.0)
a4 = -k * math.exp(-2.0 * alpha)
a5 = k
a6 = k * math.exp(-alpha) * (alpha - 1.0)
a7 = k * math.exp(-alpha) * (alpha + 1.0)
a8 = -k * math.exp(-2.0 * alpha)
b1 = 2.0 ** (-alpha)
b2 = -math.exp(-2.0 * alpha)
c1 = 1.0
c2 = 1.0


@kernel
def deriche(
    imgIn: "f32[W, H]",
    imgOut: "f32[W, H]",
    y1: "f32[W, H]",
    y2: "f32[W, H]",
):
    for i in range(W):
        ym1: f32 = 0.0
        ym2: f32 = 0.0
        xm1: f32 = 0.0
        for j in range(H):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]

    for i in range(W):
        yp1: f32 = 0.0
        yp2: f32 = 0.0
        xp1: f32 = 0.0
        xp2: f32 = 0.0
        for j_inv in range(H):
            j: index = H - 1 - j_inv
            y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i, j]
            yp2 = yp1
            yp1 = y2[i, j]

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])

    for j in range(H):
        tm1: f32 = 0.0
        ym1_col: f32 = 0.0
        ym2_col: f32 = 0.0
        for i in range(W):
            y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1_col + b2 * ym2_col
            tm1 = imgOut[i, j]
            ym2_col = ym1_col
            ym1_col = y1[i, j]

    for j in range(H):
        tp1: f32 = 0.0
        tp2: f32 = 0.0
        yp1_col: f32 = 0.0
        yp2_col: f32 = 0.0
        for i_inv in range(W):
            i: index = W - 1 - i_inv
            y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1_col + b2 * yp2_col
            tp2 = tp1
            tp1 = imgOut[i, j]
            yp2_col = yp1_col
            yp1_col = y2[i, j]

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])
