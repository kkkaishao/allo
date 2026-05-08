# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import index, kernel, u8
from . import run_machsuite_kernel

S = 32
P = 16


@kernel
def kmp(pattern: "u8[P]", input_str: "u8[S]", kmp_next: "u8[P]", matches: "u8[1]"):
    k: index = 0
    x: index = 1

    for i in range(P - 1):
        while k > 0 and pattern[k] != pattern[x]:
            k = kmp_next[k - 1]

        if pattern[k] == pattern[x]:
            k += 1
        kmp_next[x] = k
        x += 1

    q: index = 0
    for i in range(S):
        while q > 0 and pattern[q] != input_str[i]:
            q = kmp_next[q - 1]

        if pattern[q] == input_str[i]:
            q += 1

        if q >= P:
            matches[0] += 1
            q = kmp_next[q - 1]


def np_kmp(pattern, input_str, kmp_next, matches):
    k = 0
    x = 1

    for _ in range(P - 1):
        while k > 0 and pattern[k] != pattern[x]:
            k = int(kmp_next[k - 1])

        if pattern[k] == pattern[x]:
            k += 1
        kmp_next[x] = k
        x += 1

    q = 0
    for i in range(S):
        while q > 0 and pattern[q] != input_str[i]:
            q = int(kmp_next[q - 1])

        if pattern[q] == input_str[i]:
            q += 1

        if q >= P:
            matches[0] += 1
            q = int(kmp_next[q - 1])


def test_kmp():
    run_machsuite_kernel(kmp, "kmp")
