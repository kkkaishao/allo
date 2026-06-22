# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-stationary systolic conv2d library (``allo.library.systolic.conv``).

Convolution lowers to an OS systolic GEMM (K = KH*KW*Ci); the PE array and store
are reused from ``mm.os`` and only the input loader does im2col. Functional
checks run through the CPU dataflow simulator and Vitis csim.
"""

import tempfile

import numpy as np
import pytest

from allo.lang.core import f32, i8, i32
from allo.library.systolic.conv import (
    SystolicConv2D,
    conv2d_output_shape,
    flatten_weights,
)
from allo.backend.vitis.utils import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"


def _ref_conv2d(X, W, stride, pad):
    """NHWC reference: X[IH,IW,Ci], W[KH,KW,Ci,Co] -> Y[OH,OW,Co]."""
    IH, IW, Ci = X.shape
    KH, KW, _, Co = W.shape
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, stride, pad)
    Xp = np.pad(X, ((pad, pad), (pad, pad), (0, 0)))
    Y = np.zeros((OH, OW, Co), dtype=W.dtype)
    for oh in range(OH):
        for ow in range(OW):
            for kh in range(KH):
                for kw in range(KW):
                    Y[oh, ow, :] += (
                        Xp[oh * stride + kh, ow * stride + kw, :, None]
                        * W[kh, kw, :, :]
                    ).sum(axis=0)
    return Y


def _operands(Co, Ci, IH, IW, KH, KW, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        X = rng.integers(-4, 4, size=(IH, IW, Ci)).astype(dtype)
        W = rng.integers(-4, 4, size=(KH, KW, Ci, Co)).astype(dtype)
    else:
        X = rng.random((IH, IW, Ci)).astype(dtype)
        W = rng.random((KH, KW, Ci, Co)).astype(dtype)
    return X, W


# (name, Co, Ci, IH, IW, KH, KW, stride, pad, array)
CASES = [
    ("valid_3x3", 8, 4, 6, 6, 3, 3, 1, 0, (4, 4)),
    ("same_3x3", 8, 4, 8, 8, 3, 3, 1, 1, (4, 8)),
    ("stride2_3x3", 8, 4, 9, 9, 3, 3, 2, 0, (4, 4)),
    ("pointwise_1x1", 16, 8, 8, 8, 1, 1, 1, 0, (8, 8)),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_cpu_sim(case):
    _, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(Co, Ci, IH, IW, KH, KW, np.float32)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.float32)
    m = SystolicConv2D(
        f32, f32, f32, Co, Ci, IH, IW, KH, KW, stride=S, pad=P, array=array
    )
    m.schedule.export("cpu")(flatten_weights(W), X, Y)
    np.testing.assert_allclose(Y, _ref_conv2d(X, W, S, P), atol=1e-4)


def test_int_cpu_sim():
    Co, Ci, IH, IW, KH, KW, S, P, array = 8, 4, 8, 8, 3, 3, 1, 1, (4, 8)
    X, W = _operands(Co, Ci, IH, IW, KH, KW, np.int8)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.int32)
    m = SystolicConv2D(
        i8, i32, i32, Co, Ci, IH, IW, KH, KW, stride=S, pad=P, array=array
    )
    m.schedule.export("cpu")(flatten_weights(W).astype(np.int8), X, Y)
    ref = _ref_conv2d(X.astype(np.int32), W.astype(np.int32), S, P)
    np.testing.assert_array_equal(Y, ref)


def test_codegen():
    m = SystolicConv2D(f32, f32, f32, 8, 4, 8, 8, 3, 3, stride=1, pad=1, array=(4, 8))
    code = m.schedule.export("vitis").hls_code
    assert "void " in code and "conv" in code


@requires_vitis
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_csim(case):
    _, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(Co, Ci, IH, IW, KH, KW, np.float32, seed=2)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.float32)
    m = SystolicConv2D(
        f32, f32, f32, Co, Ci, IH, IW, KH, KW, stride=S, pad=P, array=array
    )
    with tempfile.TemporaryDirectory() as proj:
        m.schedule.export("vitis", project_path=proj)(flatten_weights(W), X, Y)
    np.testing.assert_allclose(Y, _ref_conv2d(X, W, S, P), atol=1e-3)


# buffered: cols must divide OW (column-tile = output-row segment)
BUF_CASES = [
    ("buf_same_3x3", 8, 4, 8, 8, 3, 3, 1, 1, (4, 8)),
    ("buf_same_nt4", 8, 4, 8, 8, 3, 3, 1, 1, (4, 4)),
    ("buf_valid_3x3", 8, 4, 8, 8, 3, 3, 1, 0, (4, 3)),  # OW=6, Nt=3
    ("buf_stride2", 8, 4, 9, 9, 3, 3, 2, 0, (4, 4)),  # OW=4, Nt=4
]


@pytest.mark.parametrize("case", BUF_CASES, ids=[c[0] for c in BUF_CASES])
def test_buffered_cpu_sim(case):
    _, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(Co, Ci, IH, IW, KH, KW, np.float32, seed=5)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.float32)
    m = SystolicConv2D(
        f32,
        f32,
        f32,
        Co,
        Ci,
        IH,
        IW,
        KH,
        KW,
        stride=S,
        pad=P,
        array=array,
        variant="buffered",
    )
    m.schedule.export("cpu")(flatten_weights(W), X, Y)
    np.testing.assert_allclose(Y, _ref_conv2d(X, W, S, P), atol=1e-4)


@requires_vitis
@pytest.mark.parametrize("case", BUF_CASES, ids=[c[0] for c in BUF_CASES])
def test_buffered_csim(case):
    _, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _operands(Co, Ci, IH, IW, KH, KW, np.float32, seed=6)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.float32)
    m = SystolicConv2D(
        f32,
        f32,
        f32,
        Co,
        Ci,
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
        m.schedule.export("vitis", project_path=proj)(flatten_weights(W), X, Y)
    np.testing.assert_allclose(Y, _ref_conv2d(X, W, S, P), atol=1e-3)


# packed: low-bit int DSP packing (cols even, cols | OW)
PACK_CASES = [
    ("pack_i8_same", i8, 18, 8, 4, 8, 8, 3, 3, 1, 1, (4, 8)),
    ("pack_i8_valid", i8, 18, 8, 4, 8, 8, 3, 3, 1, 0, (4, 6)),  # OW=6
]


def _int_operands(Co, Ci, IH, IW, KH, KW, lo, hi, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(lo, hi, size=(IH, IW, Ci)).astype(np.int8)
    W = rng.integers(lo, hi, size=(KH, KW, Ci, Co)).astype(np.int8)
    return X, W


@pytest.mark.parametrize("case", PACK_CASES, ids=[c[0] for c in PACK_CASES])
def test_packed_cpu_sim(case):
    _, Tin, _G, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _int_operands(Co, Ci, IH, IW, KH, KW, -8, 8, seed=4)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.int32)
    m = SystolicConv2D(
        Tin,
        i32,
        i32,
        Co,
        Ci,
        IH,
        IW,
        KH,
        KW,
        stride=S,
        pad=P,
        array=array,
        variant="packed",
    )
    m.schedule.export("cpu")(flatten_weights(W).astype(np.int8), X, Y)
    ref = _ref_conv2d(X.astype(np.int32), W.astype(np.int32), S, P)
    np.testing.assert_array_equal(Y, ref)


@requires_vitis
@pytest.mark.parametrize("case", PACK_CASES, ids=[c[0] for c in PACK_CASES])
def test_packed_csim(case):
    _, Tin, _G, Co, Ci, IH, IW, KH, KW, S, P, array = case
    X, W = _int_operands(Co, Ci, IH, IW, KH, KW, -8, 8, seed=5)
    OH, OW = conv2d_output_shape(IH, IW, KH, KW, S, P)
    Y = np.zeros((OH, OW, Co), np.int32)
    m = SystolicConv2D(
        Tin,
        i32,
        i32,
        Co,
        Ci,
        IH,
        IW,
        KH,
        KW,
        stride=S,
        pad=P,
        array=array,
        variant="packed",
    )
    with tempfile.TemporaryDirectory() as proj:
        m.schedule.export("vitis", project_path=proj)(
            flatten_weights(W).astype(np.int8), X, Y
        )
    ref = _ref_conv2d(X.astype(np.int32), W.astype(np.int32), S, P)
    np.testing.assert_array_equal(Y, ref)


@requires_vitis
@pytest.mark.parametrize("variant", ["direct", "buffered"])
def test_synth(variant):
    array = (16, 16) if variant == "direct" else (16, 8)  # buffered: 8 | OW=8
    m = SystolicConv2D(
        f32, f32, f32, 16, 16, 8, 8, 3, 3, stride=1, pad=1, array=array, variant=variant
    )
    with tempfile.TemporaryDirectory() as proj:
        mod = m.schedule.export("vitis", part=PART, project_path=proj)
        mod.set_axi(0, bundle="gmem0")
        mod.set_axi(1, bundle="gmem1")
        mod.set_axi(2, bundle="gmem2")
        assert mod.synth().exists()
