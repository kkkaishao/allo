# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import i32, kernel
from .. import run_machsuite_kernel
import numpy as np

N_NODES = 32
N_NODES_2 = N_NODES * 2
N_EDGES = 128
N_LEVELS = 6
MAX_LEVEL = 999999


@kernel
def bfs_bulk(
    nodes: "i32[N_NODES_2]", edges: "i32[N_EDGES]", starting_node: i32
) -> ("i32[N_NODES]", "i32[N_LEVELS]"):
    level: "i32[N_NODES]" = MAX_LEVEL
    level_counts: "i32[N_LEVELS]" = 0
    level[starting_node] = 0
    level_counts[0] = 1

    for horizon in range(N_LEVELS):
        cnt: i32 = 0
        horizon_i32: i32 = horizon
        for n in range(N_NODES):
            if level[n] == horizon_i32:
                tmp_begin: i32 = nodes[2 * n]
                tmp_end: i32 = nodes[2 * n + 1]
                for e in range(tmp_begin, tmp_end):
                    tmp_dst: i32 = edges[e]
                    tmp_level: i32 = level[tmp_dst]

                    if tmp_level == MAX_LEVEL:
                        level[tmp_dst] = horizon_i32 + 1
                        cnt += 1

        if cnt != 0:
            level_counts[horizon + 1] = cnt

    return level, level_counts


def np_bfs_bulk(nodes, edges, starting_node):
    level = np.full(N_NODES, MAX_LEVEL, dtype=np.int32)
    level_counts = np.zeros(N_LEVELS, dtype=np.int32)

    level[starting_node] = 0
    level_counts[0] = 1

    for horizon in range(N_LEVELS):
        cnt = 0
        for n in range(N_NODES):
            if level[n] == horizon:
                for e in range(nodes[2 * n], nodes[2 * n + 1]):
                    dst = edges[e]
                    if level[dst] == MAX_LEVEL:
                        level[dst] = horizon + 1
                        cnt += 1
        if cnt != 0:
            level_counts[horizon + 1] = cnt
    return level, level_counts


def test_bfs_bulk():
    run_machsuite_kernel(bfs_bulk, "bfs_bulk")
