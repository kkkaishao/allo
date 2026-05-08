# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp.lang import f32, kernel

Q = 20
R = 25
P = 30
S = 30


def np_doitgen(A, x, sum_):
    for r in range(R):
        for q in range(Q):
            for p in range(P):
                sum_[p] = 0.0
                for s in range(P):
                    sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
            for p1 in range(P):
                A[r, q, p1] = sum_[p1]
    return A


@kernel
def doitgen(A: "f32[R, Q, S]", x: "f32[P, S]", sum_: "f32[P]"):
    for r in range(R):
        for q in range(Q):
            for p in range(P):
                sum_[p] = 0.0
                for s in range(P):
                    sum_[p] = sum_[p] + A[r, q, s] * x[s, p]
            for p1 in range(P):
                A[r, q, p1] = sum_[p1]
