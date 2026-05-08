# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang import f32, kernel

M = 80
N = 100


def np_covariance(data, mean, cov):
    for x in range(M):
        mean[x] = np.sum(data[:, x]) / N

    for i in range(M):
        for j in range(M):
            covariance = 0.0
            for p in range(N):
                covariance += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
            cov[i, j] = covariance / (N - 1)
    return mean, cov


@kernel
def covariance(data: "f32[N, M]", mean: "f32[M]", cov: "f32[M, M]"):
    for x in range(M):
        total: f32 = 0.0
        for k in range(N):
            total += data[k, x]
        mean[x] = total / N

    for i in range(M):
        for j in range(M):
            covariance: f32 = 0.0
            for p in range(N):
                covariance += (data[p, i] - mean[i]) * (data[p, j] - mean[j])
            cov[i, j] = covariance / (N - 1)
