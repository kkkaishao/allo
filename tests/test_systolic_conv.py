# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D convolution systolic array: one compute PE per output pixel; mixes
compile-time PE-role branches (``get_wid``) with a runtime ``if`` that maps the
linear PE row back to an (output_row, output_col) coordinate.

Functional checks run through both the CPU dataflow simulator and Vitis csim.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import f32, i32, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"

IR, IC = 6, 6
FR, FC = 3, 3
OR, OC = IR - FR + 1, IC - FC + 1
P0, P1 = OR * OC + 2, FR


@kernel
def top(A: f32[IR, IC], B: f32[FR, FC], C: f32[OR, OC]):
    fifo_A: Stream[f32][P0, P1]
    fifo_B: Stream[f32][P0, P1]

    @kernel(mapping=[P0, P1])
    def conv(
        A: f32[IR, IC],
        B: f32[FR, FC],
        C: f32[OR, OC],
        fifo_A: Stream[f32][P0, P1],
        fifo_B: Stream[f32][P0, P1],
    ):
        pi = allo.get_wid(0)
        pj = allo.get_wid(1)
        if (pi == 0 or pi == P0 - 1) and (pj == 0 or pj == P1 - 1):
            pass
        elif pj == 0:
            output_row: i32 = 0
            output_col: i32 = 0
            for r in range(OR):
                if pi > r * OC and pi <= (r + 1) * OC:
                    output_row = r
                    output_col = pi - r * OC - 1
            for row in range(FR):
                for col in range(FC):
                    fifo_A[pi, pj + 1].put(A[row + output_row, col + output_col])
        elif pi == 0:
            for row in range(FR):
                for col in range(FC):
                    fifo_B[pi + 1, pj].put(B[FR - row - 1, FC - col - 1])
        elif pi == P0 - 1:
            for _ in range(FR * FC):
                drain_B: f32 = fifo_B[pi, pj].get()
        elif pj == P1 - 1:
            for _ in range(FR * FC):
                drain_A: f32 = fifo_A[pi, pj].get()
        else:
            partial_sum: f32 = 0
            for k in range(FR * FC):
                a: f32 = fifo_A[pi, pj].get()
                b: f32 = fifo_B[pi, pj].get()
                partial_sum += a * b
                fifo_A[pi, pj + 1].put(partial_sum)
                fifo_B[pi + 1, pj].put(b)
            out_row: i32 = 0
            out_col: i32 = 0
            for r in range(OR):
                if pi > r * OC and pi <= (r + 1) * OC:
                    out_row = r
                    out_col = pi - r * OC - 1
            C[out_row, out_col] += partial_sum

    conv(A, B, C, fifo_A, fifo_B)


def _ref(A, B):
    C = np.zeros((OR, OC), dtype=np.float32)
    for y in range(OR):
        for x in range(OC):
            v = 0.0
            for r in range(FR):
                for c in range(FC):
                    v += A[y + r, x + c] * B[FR - 1 - r, FC - 1 - c]
            C[y, x] = v
    return C


def test_codegen():
    code = top.schedule().export("vitis").hls_code
    # a compute PE exists; the four idle corners are pruned
    assert "void top_conv_1_1(" in code
    assert "top_conv_0_0" not in code


def test_cpu_sim():
    A = np.random.rand(IR, IC).astype(np.float32)
    B = np.random.rand(FR, FC).astype(np.float32)
    C = np.zeros((OR, OC), dtype=np.float32)
    top.schedule().export("cpu")(A, B, C)
    np.testing.assert_allclose(C, _ref(A, B), atol=1e-5)


@requires_vitis
def test_csim():
    A = np.random.rand(IR, IC).astype(np.float32)
    B = np.random.rand(FR, FC).astype(np.float32)
    C = np.zeros((OR, OC), dtype=np.float32)
    with tempfile.TemporaryDirectory() as proj:
        top.schedule().export("vitis", project_path=proj)(A, B, C)
    np.testing.assert_allclose(C, _ref(A, B), atol=1e-3)


@requires_vitis
def test_synth():
    with tempfile.TemporaryDirectory() as proj:
        report = top.schedule().export("vitis", part=PART, project_path=proj).synth()
        assert report.exists()
