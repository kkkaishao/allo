# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Systolic 2D-convolution library.

One entry point, :func:`make`, builds a scheduled conv and returns the
``Kernel`` and its ``Schedule`` for further composition. A standard conv is run
as a systolic GEMM over ``K = IC*KH*KW`` with on-the-fly im2col (the PE array is
the weight-stationary GEMM array); depthwise has no channel reduction and uses
the array as independent per-channel FIR lanes. 1x1 (pointwise) conv *is* a GEMM
-- use :func:`pointwise`, which delegates to the GEMM library.

    from allo.library.systolic.conv import conv  # the package

    top, top_s = conv.make(i8, i32, i32, IC=32, OC=32, IH=16, IW=16, KH=3, KW=3,
                           kind="conv2d", variant="direct", pad=1)
    v = top_s.export("vitis", part="xcu280-fsvh2892-2L-e", freq_mhz=300.0)

Choices (introspectable as module constants):

* ``kind`` (:data:`KINDS`)
    - ``"conv2d"`` standard conv (covers strided / dilated / grouped via params).
    - ``"depthwise"`` per-channel conv (MobileNet/EfficientNet); ``OC`` is
      ignored (out-channels == ``IC``). ``variant`` ``"direct"`` (memory-bound --
      re-reads the window) or ``"buffered"`` (line buffer -- reads input once).
* ``variant`` (:data:`VARIANTS`, conv2d only)
    - ``"direct"`` stream the input from DRAM (re-read); minimal on-chip memory.
    - ``"buffered"`` line-buffer the input rows (read once); ``block`` is the
      ``OHb`` output-row block (defaults to the whole plane).
    - ``"packed"`` low-bitwidth-int DSP packing (i8/i4 only): P=2 output channels
      share one DSP multiply -> ~2x fewer DSPs for int8 (int4 -> LUTs, 0 DSP).
* ``config`` (:data:`CONFIGS`) or ``array=(rows, cols)``: PE-array shape. For
  ``conv2d`` ``rows`` tiles ``IC`` and ``cols`` tiles ``OC``; for ``depthwise``
  the ``rows*cols`` PEs are channel lanes tiling ``C``.

TENSOR LAYOUTS are **NHWC** (channels innermost) so the array lanes read
contiguously; conv2d weights are reordered to ``[KH,KW,IC,OC]``, depthwise to
``[KH,KW,C]``. See each factory's docstring.
"""

from typing import Literal
from ....lang.core import i4, i8, i16, i32, f16, f32
from .conv2d_ws import make_direct_weight_stationary_conv2d
from .conv2d_ws_buffered import make_buffered_weight_stationary_conv2d
from .conv2d_ws_packed import make_packed_weight_stationary_conv2d
from .depthwise import make_weight_stationary_depthwise
from .depthwise_buffered import make_buffered_weight_stationary_depthwise
from ..mm import make as _gemm_make

# PE-array presets: name -> (rows, cols).
CONFIGS: dict[str, tuple] = {
    "balanced": (8, 8),  # 64 PEs
    "performance": (16, 16),  # 256 PEs
}

KINDS = ("conv2d", "depthwise")
VARIANTS = ("direct", "buffered", "packed")

_FACTORY = {
    ("conv2d", "direct"): make_direct_weight_stationary_conv2d,
    ("conv2d", "buffered"): make_buffered_weight_stationary_conv2d,
    ("conv2d", "packed"): make_packed_weight_stationary_conv2d,
}


def make(
    Tin,
    Tacc,
    Tout,
    IC,
    OC,
    IH,
    IW,
    KH,
    KW,
    *,
    kind: Literal["conv2d", "depthwise"] = "conv2d",
    variant: Literal["direct", "buffered", "packed"] = "direct",
    stride=1,
    pad=0,
    dil=1,
    config: Literal["balanced", "performance"] = "performance",
    array: tuple[int, ...] | None = None,
    block: int | None = None,
    depth=2,
    ii=1,
):
    """Build a scheduled systolic conv for the given configuration; returns the
    ``Kernel`` and its ``Schedule``.

    See the module docstring for the meaning of each argument, the config
    presets, and the (NHWC) tensor layouts. ``block`` bounds the buffered
    variant's on-chip line buffer (the ``OHb`` output-row block; ``None`` buffers
    the whole plane). ``array`` overrides ``config`` as an explicit (rows, cols).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {list(KINDS)}")
    if array is None:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; choose from {list(CONFIGS)}")
        array = CONFIGS[config]
    rows, cols = array

    if kind == "depthwise":
        if variant == "direct":
            return make_weight_stationary_depthwise(
                Tin,
                Tacc,
                Tout,
                IC,
                IH,
                IW,
                KH,
                KW,
                rows,
                cols,
                stride=stride,
                pad=pad,
                dil=dil,
                depth=depth,
                ii=ii,
            )
        if variant == "buffered":
            dkw = {"stride": stride, "pad": pad, "dil": dil, "depth": depth, "ii": ii}
            if block is not None:
                dkw["OHb"] = block
            return make_buffered_weight_stationary_depthwise(
                Tin, Tacc, Tout, IC, IH, IW, KH, KW, rows, cols, **dkw
            )
        raise ValueError("depthwise supports variant='direct' or 'buffered'")

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    kwargs = {"stride": stride, "pad": pad, "dil": dil, "depth": depth, "ii": ii}
    if block is not None:
        if variant != "buffered":
            raise ValueError("`block` is only valid for variant='buffered'")
        kwargs["OHb"] = block
    if variant == "packed":
        # DSP packing pairs 2 output channels/DSP; only int8/int4 benefit (wider
        # products exceed the 27-bit DSP port). gap > product width: 2*w+2.
        if not (Tin.is_int() and Tin.primitive_width <= 8):
            raise ValueError(
                "variant='packed' supports only low-bitwidth int (i8/i4); "
                "use 'direct' for int16/float"
            )
        kwargs["P"] = 2
        kwargs["G"] = 2 * Tin.primitive_width + 2
    return _FACTORY[(kind, variant)](
        Tin, Tacc, Tout, IC, OC, IH, IW, KH, KW, rows, cols, **kwargs
    )


def pointwise(
    Tin,
    Tacc,
    Tout,
    IC,
    OC,
    IH,
    IW,
    *,
    dataflow: Literal["os", "ws"] = "ws",
    variant: Literal["direct", "buffered", "packed"] = "direct",
    config: Literal["balanced", "performance"] = "performance",
    array: tuple[int, ...] | None = None,
    depth=2,
    ii=1,
):
    """Build a 1x1 (pointwise) conv -- which **is** a GEMM over the ``IH*IW``
    spatial positions: ``out[ihw, oc] = sum_ic inp[ihw, ic] * w[ic, oc]``.

    Delegates to the GEMM library (:func:`...mm.make`) with ``M = IH*IW``,
    ``N = OC``, ``K = IC``; returns its ``(top, top_s)``. The operands are the
    GEMM operands (``ws``: ``A [IH*IW, IC]``, ``B [IC, OC]``, ``C [IH*IW, OC]``)
    -- reshape the NHWC activation ``[IH,IW,IC] -> [IH*IW, IC]`` (a no-op view).
    All GEMM variants/precisions apply, including ``packed`` for int8/int4.
    """
    return _gemm_make(
        Tin,
        Tacc,
        Tout,
        IH * IW,
        OC,
        IC,
        dataflow=dataflow,
        variant=variant,
        config=config,
        array=array,
        depth=depth,
        ii=ii,
    )


__all__ = [
    "CONFIGS",
    "KINDS",
    "VARIANTS",
    "make",
    "pointwise",
    "make_direct_weight_stationary_conv2d",
    "make_buffered_weight_stationary_conv2d",
    "make_packed_weight_stationary_conv2d",
    "make_weight_stationary_depthwise",
    "make_buffered_weight_stationary_depthwise",
]
