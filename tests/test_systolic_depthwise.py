# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-stationary systolic depthwise conv2d (``SystolicDepthwise``).

Depthwise has no cross-channel reduction (K = KH*KW); channels map to PE rows
(parallel, per-channel filters flow east), pixels to columns. Functional checks
run through the CPU dataflow simulator and Vitis csim.
"""

import tempfile

import numpy as np
import pytest

from allo.lang.core import f32, i8, i32
from allo.library.systolic.conv import SystolicDepthwise, conv2d_output_shape
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"


def _ref_depthwise(X, W, stride, pad):
    """NHWC reference: X[IH,IW,C], W[KH,KW,C] -> Y[OH,OW,C]."""
    IH, IW, C = X.shape
    KH, KW, _ = W.shape
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, stride, pad)
    Xp = np.pad(X, ((pad, pad), (pad, pad), (0, 0)))
    Y = np.zeros((OH, OW, C), dtype=W.dtype)
    for oh in range(OH):
        for ow in range(OW):
            for kh in range(KH):
                for kw in range(KW):
                    Y[oh, ow, :] += (
                        W[kh, kw, :] * Xp[oh * stride + kh, ow * stride + kw, :]
                    )
    return Y


def _operands(C, IH, IW, KH, KW, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        X = rng.integers(-4, 4, size=(IH, IW, C)).astype(dtype)
        W = rng.integers(-4, 4, size=(KH, KW, C)).astype(dtype)
    else:
        X = rng.random((IH, IW, C)).astype(dtype)
        W = rng.random((KH, KW, C)).astype(dtype)
    return X, W


# (name, C, IH, IW, KH, KW, stride, pad, array=(Ct, Pt))
CASES = [
    ("dw_same_3x3", 16, 8, 8, 3, 3, 1, 1, (8, 8)),
    ("dw_valid_3x3", 16, 8, 8, 3, 3, 1, 0, (8, 4)),  # OH=OW=6, N=36
    ("dw_stride2", 16, 9, 9, 3, 3, 2, 0, (8, 4)),  # OH=OW=4, N=16
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_cpu_sim(case):
    _, C, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(C, IH, IW, KH, KW, np.float32)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, C), np.float32)
    m = SystolicDepthwise(
        f32, f32, f32, C, IH, IW, KH, KW, stride=S, pad=P, array=array
    )
    m.schedule.export("cpu")(W, X, Y)
    np.testing.assert_allclose(Y, _ref_depthwise(X, W, S, P), atol=1e-4)


def test_int_cpu_sim():
    C, IH, IW, KH, KW, S, P, array = 16, 8, 8, 3, 3, 1, 1, (8, 8)
    X, W = _operands(C, IH, IW, KH, KW, np.int8)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, C), np.int32)
    m = SystolicDepthwise(i8, i32, i32, C, IH, IW, KH, KW, stride=S, pad=P, array=array)
    m.schedule.export("cpu")(W, X, Y)
    np.testing.assert_array_equal(
        Y, _ref_depthwise(X.astype(np.int32), W.astype(np.int32), S, P)
    )


@requires_vitis
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_csim(case):
    _, C, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(C, IH, IW, KH, KW, np.float32, seed=2)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, C), np.float32)
    m = SystolicDepthwise(
        f32, f32, f32, C, IH, IW, KH, KW, stride=S, pad=P, array=array
    )
    with tempfile.TemporaryDirectory() as proj:
        m.schedule.export("vitis", project_path=proj)(W, X, Y)
    np.testing.assert_allclose(Y, _ref_depthwise(X, W, S, P), atol=1e-3)


# buffered depthwise: Pt must divide OW
BUF_CASES = [
    ("dwbuf_same_3x3", 16, 8, 8, 3, 3, 1, 1, (8, 8)),
    ("dwbuf_stride2", 16, 9, 9, 3, 3, 2, 0, (8, 4)),  # OW=4
]


@pytest.mark.parametrize("case", BUF_CASES, ids=[c[0] for c in BUF_CASES])
def test_buffered_cpu_sim(case):
    _, C, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(C, IH, IW, KH, KW, np.float32, seed=3)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, C), np.float32)
    m = SystolicDepthwise(
        f32,
        f32,
        f32,
        C,
        IH,
        IW,
        KH,
        KW,
        stride=S,
        pad=P,
        array=array,
        variant="buffered",
    )
    m.schedule.export("cpu")(W, X, Y)
    np.testing.assert_allclose(Y, _ref_depthwise(X, W, S, P), atol=1e-4)


@requires_vitis
@pytest.mark.parametrize("case", BUF_CASES, ids=[c[0] for c in BUF_CASES])
def test_buffered_csim(case):
    _, C, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(C, IH, IW, KH, KW, np.float32, seed=4)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, C), np.float32)
    m = SystolicDepthwise(
        f32,
        f32,
        f32,
        C,
        IH,
        IW,
        KH,
        KW,
        stride=S,
        pad=P,
        array=array,
        variant="buffered",
    )
    with tempfile.TemporaryDirectory() as proj:
        m.schedule.export("vitis", project_path=proj)(W, X, Y)
    np.testing.assert_allclose(Y, _ref_depthwise(X, W, S, P), atol=1e-3)


@requires_vitis
@pytest.mark.parametrize("variant", ["direct", "buffered"])
def test_synth(variant):
    m = SystolicDepthwise(
        f32, f32, f32, 16, 8, 8, 3, 3, stride=1, pad=1, array=(8, 8), variant=variant
    )
    with tempfile.TemporaryDirectory() as proj:
        mod = m.schedule.export("vitis", part=PART, project_path=proj)
        mod.set_axi(0, bundle="gmem0")
        mod.set_axi(1, bundle="gmem1")
        mod.set_axi(2, bundle="gmem2")
        mod.synth()
        assert mod.synth_report.exists()
