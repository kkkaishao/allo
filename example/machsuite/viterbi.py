# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import f32, i32, kernel
from . import run_machsuite_kernel
import numpy as np

N_OBS = 32
N_STATES = 16
N_TOKENS = 16


@kernel
def viterbi(
    obs: "i32[N_OBS]",
    init: "f32[N_STATES]",
    transition: "f32[N_STATES, N_STATES]",
    emission: "f32[N_STATES, N_TOKENS]",
) -> "i32[N_OBS]":
    llike: "f32[N_OBS, N_STATES]"

    for s in range(N_STATES):
        llike[0, s] = init[s] + emission[s, obs[0]]

    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            min_p: f32 = llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p: f32 = (
                    llike[t - 1, prev] + transition[prev, curr] + emission[curr, obs[t]]
                )
                if p < min_p:
                    min_p = p
            llike[t, curr] = min_p

    min_s: i32 = 0
    min_p: f32 = llike[N_OBS - 1, 0]
    for s in range(1, N_STATES):
        p: f32 = llike[N_OBS - 1, s]
        if p < min_p:
            min_p = p
            min_s = s

    path: "i32[N_OBS]"
    path[N_OBS - 1] = min_s

    for t in range(N_OBS - 1):
        actual_t: i32 = N_OBS - 2 - t
        min_s = 0
        min_p = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
        for s in range(1, N_STATES):
            p: f32 = llike[actual_t, s] + transition[s, path[actual_t + 1]]
            if p < min_p:
                min_p = p
                min_s = s
        path[actual_t] = min_s

    return path


def np_viterbi(obs, init, transition, emission):
    llike = np.zeros((N_OBS, N_STATES), dtype=np.float32)

    for s in range(N_STATES):
        llike[0, s] = init[s] + emission[s, obs[0]]

    for t in range(1, N_OBS):
        for curr in range(N_STATES):
            min_p = llike[t - 1, 0] + transition[0, curr] + emission[curr, obs[t]]
            for prev in range(1, N_STATES):
                p = llike[t - 1, prev] + transition[prev, curr] + emission[curr, obs[t]]
                if p < min_p:
                    min_p = p
            llike[t, curr] = min_p

    min_s = 0
    min_p = llike[N_OBS - 1, 0]
    for s in range(1, N_STATES):
        p = llike[N_OBS - 1, s]
        if p < min_p:
            min_p = p
            min_s = s

    path = np.zeros(N_OBS, dtype=np.int32)
    path[N_OBS - 1] = min_s

    for t in range(N_OBS - 1):
        actual_t = N_OBS - 2 - t
        min_s = 0
        min_p = llike[actual_t, 0] + transition[0, path[actual_t + 1]]
        for s in range(1, N_STATES):
            p = llike[actual_t, s] + transition[s, path[actual_t + 1]]
            if p < min_p:
                min_p = p
                min_s = s
        path[actual_t] = min_s
    return path


def test_viterbi():
    run_machsuite_kernel(viterbi, "viterbi")
