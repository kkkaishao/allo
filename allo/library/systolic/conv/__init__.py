# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Systolic 2D-convolution library.

Convolution lowers to an output-stationary systolic **GEMM**: the reduction
window ``K = KH*KW*Ci`` is the contraction axis, output channels ``Co`` index one
PE-array axis and output pixels ``OH*OW`` the other. The PE array and output
store are reused verbatim from :mod:`allo.library.systolic.mm.os`; only the input
loader is convolution-specific (im2col on the fly). Feature maps are **NHWC**
(``X[IH,IW,Ci]`` / ``Y[OH,OW,Co]``); weights arrive pre-flattened and
pre-transposed as ``Wt[K, Co]`` -- a plain ``W[KH,KW,Ci,Co].reshape(K, Co)``.
"""

from typing import Literal
from allo.lang import Module
from allo.lang.core import DType
from .os.direct import make_direct_output_stationary_conv2d
from .os.buffered import make_buffered_output_stationary_conv2d
from .os.packed import make_packed_output_stationary_conv2d
from .os.depthwise import (
    make_direct_output_stationary_depthwise,
    make_buffered_output_stationary_depthwise,
)

# PE-array presets: name -> (rows, cols). rows tiles Co; cols tiles OH*OW.
CONFIGS: dict[str, tuple] = {
    "balanced": (8, 8),  # 64 PEs, fits small parts
    "performance": (16, 16),  # 256 PEs, maximizes throughput
}

VARIANTS = ("direct", "buffered", "packed")


def conv2d_output_shape(IH, IW, KH, KW, stride, pad):
    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1
    return OH, OW


def flatten_weights(W):
    """Reshape an NHWC-style weight tensor ``W[KH,KW,Ci,Co]`` into the
    pre-transposed ``Wt[K, Co]`` the loaders expect (``K = KH*KW*Ci``)."""
    KH, KW, Ci, Co = W.shape
    return W.reshape(KH * KW * Ci, Co).copy()


class SystolicConv2D(Module):
    """**Output-stationary systolic conv2d** (NHWC).

    Signature: ``conv(Wt: Tin[K, Co], X: Tin[IH, IW, Ci], Y: Tout[OH, OW, Co])``
    where ``K = KH*KW*Ci`` and ``Wt = W[KH,KW,Ci,Co].reshape(K, Co)`` (use
    :func:`flatten_weights`). Computes a standard 2D convolution with the given
    ``stride`` and symmetric zero-``pad``. A ``1x1`` kernel is an ordinary
    pointwise conv (degenerates to a GEMM over the channel axis).

    PARAMETERS
    ----------
    variant (default ``direct``):

    * ``direct``   -- no on-chip buffer; im2col gathered straight from DRAM. The
      data-dependent gather can't burst, so it is DRAM-bound (II~16); simplest,
      and fine when bandwidth-rich or memory is tight.
    * ``buffered`` -- stage the feature map in on-chip BRAM once, then im2col from
      BRAM at II=1, leaving the design PE-bound at the compute floor. Requires
      ``cols | OW`` (column-tile = output-row segment). The realistic conv choice.
    * ``packed``   -- like ``buffered`` but packs ``P=2`` adjacent output pixels'
      activations into one DSP multiply (signed borrow-chain unpack); low-bit int
      only (i8/i4), ``cols`` even. Halves (i8) / removes (i4) the multiply DSPs.

    ``array`` overrides ``config`` to set the PE-array ``(rows, cols)`` directly
    (``rows`` tiles ``Co``, ``cols`` tiles ``OH*OW``); both must divide evenly.

    ``depth`` / ``ii`` are the systolic FIFO depth and pipeline initiation
    interval (``depth=2, ii=1`` is the high-performance default; float reduction
    is fadd-recurrence bound to II~4 regardless).
    """

    def __init__(
        self,
        Tin,
        Tacc,
        Tout,
        Co,
        Ci,
        IH,
        IW,
        KH,
        KW,
        *,
        stride: int = 1,
        pad: int = 0,
        variant: Literal["direct", "buffered", "packed"] = "direct",
        config: Literal["balanced", "performance"] = "performance",
        array: tuple[int, ...] | None = None,
        depth=2,
        ii=1,
    ):
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise ValueError("Tin/Tacc/Tout must be Allo DType")
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {list(VARIANTS)}"
            )
        if array is None:
            if config not in CONFIGS:
                raise ValueError(
                    f"unknown config {config!r}; choose from {list(CONFIGS)}"
                )
            array = CONFIGS[config]
        rows, cols = array

        OH, OW = conv2d_output_shape(IH, IW, KH, KW, stride, pad)
        N = OH * OW
        if Co % rows != 0 or N % cols != 0:
            raise ValueError(f"array {array} must tile Co={Co} and OH*OW={N} evenly")

        if variant == "direct":
            top, s = make_direct_output_stationary_conv2d(
                Tin,
                Tacc,
                Tout,
                Co,
                Ci,
                IH,
                IW,
                KH,
                KW,
                rows,
                cols,
                stride=stride,
                pad=pad,
                depth=depth,
                ii=ii,
            )
        elif variant == "buffered":
            if OW % cols != 0:
                raise ValueError(
                    f"variant='buffered' requires cols | OW (column-tile = "
                    f"output-row segment); cols={cols}, OW={OW}"
                )
            top, s = make_buffered_output_stationary_conv2d(
                Tin,
                Tacc,
                Tout,
                Co,
                Ci,
                IH,
                IW,
                KH,
                KW,
                rows,
                cols,
                stride=stride,
                pad=pad,
                depth=depth,
                ii=ii,
            )
        elif variant == "packed":
            if OW % cols != 0:
                raise ValueError(
                    f"variant='packed' requires cols | OW (column-tile = "
                    f"output-row segment); cols={cols}, OW={OW}"
                )
            if not (Tin.is_int() and Tin.primitive_width <= 8):
                raise ValueError(
                    "variant='packed' supports only low-bitwidth int (i8/i4); "
                    "use 'buffered' for int16/float"
                )
            if cols % 2 != 0:
                raise ValueError("variant='packed' needs cols even (P=2 packing)")
            G = 2 * Tin.primitive_width + 2  # i8->18, i4->10
            top, s = make_packed_output_stationary_conv2d(
                Tin,
                Tacc,
                Tout,
                Co,
                Ci,
                IH,
                IW,
                KH,
                KW,
                rows,
                cols,
                P=2,
                G=G,
                stride=stride,
                pad=pad,
                depth=depth,
                ii=ii,
            )
        else:
            raise NotImplementedError(f"variant={variant} not implemented")

        name = f"SystolicConv2D_{variant}_{Co}x{Ci}x{KH}x{KW}_s{stride}p{pad}_{rows}x{cols}"
        super().__init__(name, top, s)


class SystolicDepthwise(Module):
    """**Output-stationary systolic depthwise conv2d** (NHWC).

    Signature: ``dwconv(W: Tin[KH, KW, C], X: Tin[IH, IW, C], Y: Tout[OH, OW, C])``
    -- each channel convolved with its own ``KH x KW`` filter, **no cross-channel
    reduction** (``K = KH*KW``). The PE array maps **channels to rows** and output
    **pixels to columns**: a channel's filter flows east (shared by every pixel in
    its row), while activations are channel-specific and fed per-PE.

    ``array`` overrides ``config`` to set ``(Ct, Pt)`` directly (``Ct`` tiles the
    channels ``C``, ``Pt`` tiles ``OH*OW``); both must divide evenly.
    """

    def __init__(
        self,
        Tin,
        Tacc,
        Tout,
        C,
        IH,
        IW,
        KH,
        KW,
        *,
        stride: int = 1,
        pad: int = 0,
        variant: Literal["direct", "buffered"] = "direct",
        config: Literal["balanced", "performance"] = "performance",
        array: tuple[int, ...] | None = None,
        depth=2,
        ii=1,
    ):
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise ValueError("Tin/Tacc/Tout must be Allo DType")
        if array is None:
            if config not in CONFIGS:
                raise ValueError(
                    f"unknown config {config!r}; choose from {list(CONFIGS)}"
                )
            array = CONFIGS[config]
        Ct, Pt = array

        OH, OW = conv2d_output_shape(IH, IW, KH, KW, stride, pad)
        N = OH * OW
        if C % Ct != 0 or N % Pt != 0:
            raise ValueError(f"array {array} must tile C={C} and OH*OW={N} evenly")

        if variant == "direct":
            top, s = make_direct_output_stationary_depthwise(
                Tin,
                Tacc,
                Tout,
                C,
                IH,
                IW,
                KH,
                KW,
                Ct,
                Pt,
                stride=stride,
                pad=pad,
                depth=depth,
                ii=ii,
            )
        elif variant == "buffered":
            if OW % Pt != 0:
                raise ValueError(
                    f"variant='buffered' requires Pt | OW; Pt={Pt}, OW={OW}"
                )
            top, s = make_buffered_output_stationary_depthwise(
                Tin,
                Tacc,
                Tout,
                C,
                IH,
                IW,
                KH,
                KW,
                Ct,
                Pt,
                stride=stride,
                pad=pad,
                depth=depth,
                ii=ii,
            )
        else:
            raise NotImplementedError(f"variant={variant} not implemented")
        name = f"SystolicDepthwise_{variant}_{C}x{KH}x{KW}_s{stride}p{pad}_{Ct}x{Pt}"
        super().__init__(name, top, s)
