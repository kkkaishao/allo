# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang import f32, kernel
from allo.exp.operators import math as amath

M = 80
N = 100
N_float = 100.0
epsilon = 1e-5


def np_correlation(data_mean, data_stddev, data_for_center, corr):
    mean = np.zeros((M,), dtype=data_mean.dtype)
    stddev = np.zeros((M,), dtype=data_mean.dtype)
    data_centered = np.zeros((N, M), dtype=data_mean.dtype)

    for x in range(M):
        mean[x] = np.sum(data_mean[:, x]) / N

    for x in range(M):
        variance = 0.0
        for m in range(N):
            variance += (data_stddev[m, x] - mean[x]) * (data_stddev[m, x] - mean[x])
        stddev[x] = np.sqrt(variance / N_float)
        if stddev[x] <= epsilon:
            stddev[x] = 1.0

    for x in range(N):
        for y in range(M):
            data_centered[x, y] = (data_for_center[x, y] - mean[y]) / (
                np.sqrt(N_float) * stddev[y]
            )

    for i in range(M - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, M):
            corr_v = 0.0
            for k in range(N):
                corr_v += data_centered[k, i] * data_centered[k, j]
            corr[j, i] = corr_v
            corr[i, j] = corr_v

    corr[M - 1, M - 1] = 1.0
    return corr


@kernel
def compute_mean(data: "f32[N, M]", mean: "f32[M]"):
    for x in range(M):
        total: f32 = 0.0
        for k in range(N):
            total += data[k, x]
        mean[x] = total / N


@kernel
def compute_stddev(
    data: "f32[N, M]",
    mean: "f32[M]",
    mean_passed_on: "f32[M]",
    stddev: "f32[M]",
):
    for x in range(M):
        variance: f32 = 0.0
        for m in range(N):
            variance += (data[m, x] - mean[x]) * (data[m, x] - mean[x])
        stddev[x] = amath.sqrt(variance / N_float)
        mean_passed_on[x] = mean[x]
        if stddev[x] <= epsilon:
            stddev[x] = 1.0


@kernel
def center_reduce(
    data: "f32[N, M]",
    data_out: "f32[N, M]",
    mean: "f32[M]",
    stddev: "f32[M]",
):
    for x in range(N):
        for y in range(M):
            d: f32 = data[x, y]
            d -= mean[y]
            d /= amath.sqrt(N_float) * stddev[y]
            data_out[x, y] = d


@kernel
def compute_corr(data: "f32[N, M]", corr: "f32[M, M]"):
    for i in range(M - 1):
        corr[i, i] = 1.0
        for j in range(M):
            if j > i:
                corr_v: f32 = 0.0
                for k in range(N):
                    corr_v += data[k, i] * data[k, j]
                corr[j, i] = corr_v
                corr[i, j] = corr_v

    corr[M - 1, M - 1] = 1.0


@kernel
def correlation(
    data_mean: "f32[N, M]",
    data_stddev: "f32[N, M]",
    data_for_center: "f32[N, M]",
    corr: "f32[M, M]",
):
    mean: "f32[M]" = 0.0
    mean_passed_on: "f32[M]" = 0.0
    stddev: "f32[M]" = 0.0
    compute_mean(data_mean, mean)
    compute_stddev(data_stddev, mean, mean_passed_on, stddev)
    data_centered: "f32[N, M]" = 0.0
    center_reduce(data_for_center, data_centered, mean_passed_on, stddev)
    compute_corr(data_centered, corr)
