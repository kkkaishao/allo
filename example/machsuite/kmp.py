# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import index, kernel, u8

S = 100
P = 100


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
