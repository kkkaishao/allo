# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.lang import i32, kernel
from . import run_machsuite_kernel
import numpy as np

ALEN = 32
BLEN = 32
RESULT_LEN = ALEN + BLEN
MATRIX_SIZE = (ALEN + 1) * (BLEN + 1)

MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_SCORE = -1

ALIGN_VAL = 1
SKIPA_VAL = 2
SKIPB_VAL = 3


@kernel
def needwun(SEQA: "i32[ALEN]", SEQB: "i32[BLEN]") -> "i32[2, RESULT_LEN]":
    M: "i32[MATRIX_SIZE]" = 0
    ptr: "i32[MATRIX_SIZE]" = 0
    result: "i32[2, RESULT_LEN]" = 0

    score: i32 = 0
    row_up: i32 = 0
    row: i32 = 0
    up_left: i32 = 0
    up: i32 = 0
    left: i32 = 0
    max_val: i32 = 0

    for i in range(ALEN + 1):
        M[i] = i * GAP_SCORE

    for j in range(BLEN + 1):
        M[j * (ALEN + 1)] = j * GAP_SCORE

    for bi in range(1, BLEN + 1):
        for ai in range(1, ALEN + 1):
            if SEQA[ai - 1] == SEQB[bi - 1]:
                score = MATCH_SCORE
            else:
                score = MISMATCH_SCORE

            row_up = (bi - 1) * (ALEN + 1)
            row = bi * (ALEN + 1)

            up_left = M[row_up + (ai - 1)] + score
            up = M[row_up + ai] + GAP_SCORE
            left = M[row + (ai - 1)] + GAP_SCORE

            max_val = up_left
            if up > max_val:
                max_val = up
            if left > max_val:
                max_val = left

            M[row + ai] = max_val
            if max_val == left:
                ptr[row + ai] = SKIPB_VAL
            elif max_val == up:
                ptr[row + ai] = SKIPA_VAL
            else:
                ptr[row + ai] = ALIGN_VAL

    a_idx: i32 = ALEN
    b_idx: i32 = BLEN
    a_str_idx: i32 = 0
    b_str_idx: i32 = 0
    r: i32 = 0

    for step in range(ALEN + BLEN):
        if a_idx > 0 or b_idx > 0:
            if a_idx == 0:
                result[0, a_str_idx] = 45
                result[1, b_str_idx] = SEQB[b_idx - 1]
                a_str_idx = a_str_idx + 1
                b_str_idx = b_str_idx + 1
                b_idx = b_idx - 1
            elif b_idx == 0:
                result[0, a_str_idx] = SEQA[a_idx - 1]
                result[1, b_str_idx] = 45
                a_str_idx = a_str_idx + 1
                b_str_idx = b_str_idx + 1
                a_idx = a_idx - 1
            else:
                r = b_idx * (ALEN + 1)
                if ptr[r + a_idx] == ALIGN_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                    b_idx = b_idx - 1
                elif ptr[r + a_idx] == SKIPB_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = 45
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    a_idx = a_idx - 1
                else:
                    result[0, a_str_idx] = 45
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx = a_str_idx + 1
                    b_str_idx = b_str_idx + 1
                    b_idx = b_idx - 1

    for idx in range(RESULT_LEN):
        if result[0, idx] == 0:
            result[0, idx] = 95
        if result[1, idx] == 0:
            result[1, idx] = 95

    return result


def np_needwun(SEQA, SEQB):
    M = np.zeros(MATRIX_SIZE, dtype=np.int32)
    ptr = np.zeros(MATRIX_SIZE, dtype=np.int32)
    result = np.zeros((2, RESULT_LEN), dtype=np.int32)

    for i in range(ALEN + 1):
        M[i] = i * GAP_SCORE

    for j in range(BLEN + 1):
        M[j * (ALEN + 1)] = j * GAP_SCORE

    for bi in range(1, BLEN + 1):
        for ai in range(1, ALEN + 1):
            score = MATCH_SCORE if SEQA[ai - 1] == SEQB[bi - 1] else MISMATCH_SCORE
            row_up = (bi - 1) * (ALEN + 1)
            row = bi * (ALEN + 1)

            up_left = M[row_up + (ai - 1)] + score
            up = M[row_up + ai] + GAP_SCORE
            left = M[row + (ai - 1)] + GAP_SCORE

            max_val = up_left
            if up > max_val:
                max_val = up
            if left > max_val:
                max_val = left

            M[row + ai] = max_val
            if max_val == left:
                ptr[row + ai] = SKIPB_VAL
            elif max_val == up:
                ptr[row + ai] = SKIPA_VAL
            else:
                ptr[row + ai] = ALIGN_VAL

    a_idx = ALEN
    b_idx = BLEN
    a_str_idx = 0
    b_str_idx = 0

    for _ in range(ALEN + BLEN):
        if a_idx > 0 or b_idx > 0:
            if a_idx == 0:
                result[0, a_str_idx] = 45
                result[1, b_str_idx] = SEQB[b_idx - 1]
                a_str_idx += 1
                b_str_idx += 1
                b_idx -= 1
            elif b_idx == 0:
                result[0, a_str_idx] = SEQA[a_idx - 1]
                result[1, b_str_idx] = 45
                a_str_idx += 1
                b_str_idx += 1
                a_idx -= 1
            else:
                r = b_idx * (ALEN + 1)
                if ptr[r + a_idx] == ALIGN_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx += 1
                    b_str_idx += 1
                    a_idx -= 1
                    b_idx -= 1
                elif ptr[r + a_idx] == SKIPB_VAL:
                    result[0, a_str_idx] = SEQA[a_idx - 1]
                    result[1, b_str_idx] = 45
                    a_str_idx += 1
                    b_str_idx += 1
                    a_idx -= 1
                else:
                    result[0, a_str_idx] = 45
                    result[1, b_str_idx] = SEQB[b_idx - 1]
                    a_str_idx += 1
                    b_str_idx += 1
                    b_idx -= 1

    for idx in range(RESULT_LEN):
        if result[0, idx] == 0:
            result[0, idx] = 95
        if result[1, idx] == 0:
            result[1, idx] = 95
    return result


def test_needwun():
    run_machsuite_kernel(needwun, "nw")
