# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import i32, kernel

N_NODES = 256
N_NODES_2 = 512
N_EDGES = 4096
N_LEVELS = 10
MAX_LEVEL = 999999


@kernel
def bfs_queue(
    nodes: "i32[N_NODES_2]", edges: "i32[N_EDGES]", starting_node: i32
) -> ("i32[N_NODES]", "i32[N_LEVELS]"):
    level: "i32[N_NODES]" = MAX_LEVEL
    level_counts: "i32[N_LEVELS]" = 0
    queue: "i32[N_NODES]" = 0
    front: i32 = 0
    rear: i32 = 0

    level[starting_node] = 0
    level_counts[0] = 1
    queue[rear] = starting_node
    rear = (rear + 1) % N_NODES

    while front != rear:
        n: i32 = queue[front]
        front = (front + 1) % N_NODES
        tmp_begin: i32 = nodes[2 * n]
        tmp_end: i32 = nodes[2 * n + 1]
        for e in range(tmp_begin, tmp_end):
            tmp_dst: i32 = edges[e]
            tmp_level: i32 = level[tmp_dst]

            if tmp_level == MAX_LEVEL:
                tmp_level = level[n] + 1
                level[tmp_dst] = tmp_level
                level_counts[tmp_level] += 1
                queue[rear] = tmp_dst
                rear = (rear + 1) % N_NODES

    return level, level_counts
