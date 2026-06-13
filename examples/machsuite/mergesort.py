# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import i32, kernel
from . import run_machsuite_kernel
import numpy as np

N = 256


@kernel
def merge(a: "i32[N]", start: i32, m: i32, stop: i32):
    temp: "i32[N]"

    tmp_j: i32 = 0
    tmp_i: i32 = 0

    i: i32 = start
    j: i32 = stop

    for index in range(start, m + 1):
        temp[index] = a[index]

    for index in range(m + 1, stop + 1):
        temp[m + 1 + stop - index] = a[index]

    for k in range(start, stop + 1):
        tmp_j = temp[j]
        tmp_i = temp[i]

        if tmp_j < tmp_i:
            a[k] = tmp_j
            j -= 1
        else:
            a[k] = tmp_i
            i += 1


@kernel
def merge_sort(a: "i32[N]") -> "i32[N]":
    start: i32 = 0
    stop: i32 = N - 1

    f: i32 = 0
    m: i32 = 1
    mid: i32 = 0
    to: i32 = 0

    while m < stop - start + 1:
        for ii in range(start, stop, m + m):
            f = ii

            mid = ii + m - 1

            to = ii + m + m - 1
            if to <= stop:
                merge(a, f, mid, to)
            else:
                merge(a, f, mid, stop)

        m += m

    return a


def np_merge_sort(a):
    a[...] = np.sort(a)
    return a


def test_merge_sort():
    run_machsuite_kernel(merge_sort, "mergesort")
