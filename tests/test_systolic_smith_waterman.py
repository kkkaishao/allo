# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smith-Waterman alignment-scoring systolic array: a non-GEMM wavefront with
three diagonally-flowing FIFOs; exercises ``max`` on signed values.

Functional checks run through Vitis csim; the CPU dataflow simulator currently
deadlocks on multi-PE arrays.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import i8, i32, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"

GAP = 2  # linear gap penalty
M, N = 4, 4
P0, P1 = M + 2, N + 2
SIM = 3
MIS = -3


@kernel
def top(A: i8[M], B: i8[N], S: i32[P0 - 1, P1 - 1]):
    fifo_A: Stream[i32][P0, P1]
    fifo_B: Stream[i32][P0, P1]
    fifo_C: Stream[i32][P0, P1]

    @kernel(mapping=[P0, P1])
    def sw(
        A: i8[M],
        B: i8[N],
        S: i32[P0 - 1, P1 - 1],
        fifo_A: Stream[i32][P0, P1],
        fifo_B: Stream[i32][P0, P1],
        fifo_C: Stream[i32][P0, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        if (i == 0 and j == P1 - 1) or (i == P0 - 1 and j == 0):
            pass
        elif i == 0 and j == 0:
            fifo_C[i + 1, j + 1].put(0)
        elif i == 0:
            fifo_B[i + 1, j].put(0)
            fifo_C[i + 1, j + 1].put(0)
        elif j == 0:
            fifo_A[i, j + 1].put(0)
            fifo_C[i + 1, j + 1].put(0)
        elif i == P0 - 1 and j == P1 - 1:
            fifo_C[i, j].get()
        elif i == P0 - 1:
            fifo_B[i, j].get()
            fifo_C[i, j].get()
        elif j == P1 - 1:
            fifo_A[i, j].get()
            fifo_C[i, j].get()
        else:
            a: i32 = fifo_A[i, j].get()
            b: i32 = fifo_B[i, j].get()
            c: i32 = fifo_C[i, j].get()
            # ternary needs a non-constexpr branch to deduce its type
            sim_v: i32 = SIM
            mis_v: i32 = MIS
            aligning: i32 = c + (sim_v if A[i - 1] == B[j - 1] else mis_v)
            gap_A: i32 = a - GAP
            gap_B: i32 = b - GAP
            score: i32 = max(max(0, aligning), max(gap_A, gap_B))
            S[i, j] = score
            fifo_A[i, j + 1].put(max(gap_A, score))
            fifo_B[i + 1, j].put(max(gap_B, score))
            fifo_C[i + 1, j + 1].put(score)

    sw(A, B, S, fifo_A, fifo_B, fifo_C)


def _ref(seqA, seqB):
    sm = np.zeros((len(seqA) + 1, len(seqB) + 1), dtype=int)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            if i == 0 or j == 0:
                sm[i][j] = 0
            else:
                sim = SIM if seqA[i - 1] == seqB[j - 1] else MIS
                sm[i][j] = max(
                    0,
                    sm[i - 1][j - 1] + sim,
                    max([sm[a, j] - GAP * (i - a) for a in range(i)]),
                    max([sm[i, b] - GAP * (j - b) for b in range(j)]),
                )
    return sm


def test_codegen():
    code = top.schedule().export("vitis").hls_code
    assert "void top_sw_1_1(" in code
    # signed max must cast operands so negatives compare correctly
    assert "std::max(static_cast<int32_t>" in code


def test_cpu_sim():
    chars = np.array(["A", "C", "G", "T"], dtype="c")
    A = np.random.choice(chars, size=M).view(np.int8)
    B = np.random.choice(chars, size=N).view(np.int8)
    S = np.zeros((P0 - 1, P1 - 1), dtype=np.int32)
    top.schedule().export("cpu")(A, B, S)
    np.testing.assert_equal(S[1:, 1:], _ref(A, B)[1:, 1:])


@requires_vitis
def test_csim():
    chars = np.array(["A", "C", "G", "T"], dtype="c")
    A = np.random.choice(chars, size=M).view(np.int8)
    B = np.random.choice(chars, size=N).view(np.int8)
    S = np.zeros((P0 - 1, P1 - 1), dtype=np.int32)
    with tempfile.TemporaryDirectory() as proj:
        top.schedule().export("vitis", project_path=proj)(A, B, S)
    np.testing.assert_equal(S[1:, 1:], _ref(A, B)[1:, 1:])
