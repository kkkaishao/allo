# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang import f32, kernel

N = 120


def np_durbin(r, y):
    y[0] = -r[0]
    beta = 1.0
    alpha = -r[0]

    for k in range(1, N):
        beta = (1.0 - alpha * alpha) * beta
        sum_ = 0.0
        z = np.zeros_like(y)
        for i in range(k):
            sum_ = sum_ + r[k - i - 1] * y[i]

        alpha = -1.0 * (r[k] + sum_)

        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]

        for i in range(k):
            y[i] = z[i]

        y[k] = alpha
    return y


@kernel
def durbin(r: "f32[N]", y: "f32[N]"):
    y[0] = -r[0]
    beta: f32 = 1.0
    alpha: f32 = -r[0]

    for k in range(1, N):
        beta = (1.0 - alpha * alpha) * beta
        sum_: f32 = 0.0

        z: "f32[N]" = 0.0
        for i in range(k):
            sum_ = sum_ + r[k - i - 1] * y[i]

        alpha = -1.0 * (r[k] + sum_)

        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]

        for i in range(k):
            y[i] = z[i]

        y[k] = alpha
