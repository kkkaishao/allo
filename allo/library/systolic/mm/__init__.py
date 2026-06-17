# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Systolic matrix-multiply (GEMM) library."""

from typing import Literal
from allo.lang import Module
from allo.lang.core import DType
from .os.direct import make_direct_output_stationary_gemm
from .os.buffered import make_buffered_output_stationary_gemm
from .os.packed import make_packed_output_stationary_gemm

# PE-array presets: name -> (rows, cols). rows tiles M (os) / K (ws); cols tiles N.
CONFIGS: dict[str, tuple] = {
    "balanced": (8, 8),  # 64 PEs, fits small parts
    "performance": (16, 16),  # 256 PEs, maximizes throughput
}

DATAFLOWS = ("os", "ws")
VARIANTS = ("direct", "buffered", "packed")


class SystolicGEMM(Module):
    """**Systolic Array-based GEMM**

    Signature:

    * OS Style: ``gemm(A: Tin[K, M], B: Tin[K, N], C: Tout[M, N])``,
    computes ``C = A^T @ B`` (``A`` pre-transposed to [K,M] for better
    reuse in the PE array).
    * WS Style: ``gemm(A: Tin[M, K], B: Tin[K, N], C: Tout[M, N])``,
    computes ``C = A @ B``.

    PARAMETERS
    ----------
    ``dataflow`` is either output-stationary ("os") or weight-stationary ("ws")

    variant: ``direct`` / ``buffered`` / ``packed`` / ``dequant``, default "direct". See below.

    * ``direct``            -- no on-chip buffering, re-read from DRAM each tile; DSP-heavy,
      but minimal on-chip storage and simplest control.
    * ``buffered``          -- add on-chip buffers to reuse DRAM tiles across multiple
      PE array passes; buffer size vs DRAM traffic tradeoff tunable by ``block``
      (``Nc`` for OS, ``Mc`` for WS).
    * ``packed``            -- like direct, but pack multiple columns' worth of products
      into each DSP to save resources; only for low-bitwidth integer (i8/i4) where the
      products fit in the DSP ports.

    ``block`` (buffered variants only) bounds the on-chip operand buffer: it is the
    ``Nc`` column-block for ``os`` and the ``Mc`` row-block for ``ws``; ``None``
    buffers the whole operand.

    ``array`` is an alternative to ``config`` for specifying the PE-array shape as
    a (rows, cols) tuple. If both are given, ``array`` takes precedence over the
    preset specified by ``config``.

    ``depth`` and ``ii`` are the systolic array's pipeline depth and initiation interval,
    respectively. The default of ``depth=2, ii=1`` is a common choice for high performance.
    """

    def __init__(
        self,
        Tin,
        Tacc,
        Tout,
        M,
        N,
        K,
        *,
        dataflow: Literal["os", "ws"] = "os",
        variant: Literal["direct", "buffered", "packed"] = "direct",
        config: Literal["balanced", "performance"] = "performance",
        array: tuple[int, ...] | None = None,
        block: int | None = None,
        depth=2,
        ii=1,
    ):
        if (
            not isinstance(Tin, DType)
            or not isinstance(Tacc, DType)
            or not isinstance(Tout, DType)
        ):
            raise ValueError("Tin/Tacc/Tout must be Allo DType")

        if dataflow == "ws":
            raise ValueError(
                "Cuurently ws implementations are broken (deadlock in cosim)"
            )
        # verify configuration args
        if dataflow not in DATAFLOWS:
            raise ValueError(
                f"unknown dataflow {dataflow!r}; choose from {list(DATAFLOWS)}"
            )
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; choose from {list(VARIANTS)}"
            )
        if array is None:
            if config == "balanced":
                array = (8, 8)
            elif config == "performance":
                array = (16, 16)
            else:
                raise ValueError(
                    f"unknown config {config!r}; choose from 'balanced' or 'performance'"
                )
        rows, cols = array

        if dataflow == "os":
            # preconditions for all OS variants
            if M % rows != 0 or N % cols != 0:
                raise ValueError("array must tile the matrix evenly")

        # route to right factory function
        if (dataflow, variant) == ("os", "direct"):
            top, s = make_direct_output_stationary_gemm(
                Tin, Tacc, Tout, M, N, K, rows, cols, depth=depth, ii=ii
            )
        elif (dataflow, variant) == ("os", "buffered"):
            block = block or N  # default to buffering all of B
            if N % block != 0 or block % cols != 0:
                raise ValueError("`block` must divide N and be a multiple of cols")
            top, s = make_buffered_output_stationary_gemm(
                Tin, Tacc, Tout, M, N, K, rows, cols, Nc=block, depth=depth, ii=ii
            )
        elif (dataflow, variant) == ("os", "packed"):
            if M % rows != 0 or N % cols != 0:
                raise ValueError("array must tile the matrix evenly")
            if not Tin.is_int() or not Tacc.is_int():
                raise ValueError(
                    "variant='packed' is integer-only; use 'direct' for float"
                )
            # DSP48E2's input width is 27 bits,
            # but pack=4 doesn't show benefits over pack=2 in real synthesis results.
            # only pack=2 is supported.
            P = 2
            G = 2 * Tin.primitive_width + 2
            if not (Tin.is_int() and Tin.primitive_width <= 8):
                raise ValueError(
                    "variant='packed' supports only low-bitwidth int (i8/i4); use 'direct' for int16/float"
                )
            if N % P != 0:
                raise ValueError("N must be even for P=2 packing")
            top, s = make_packed_output_stationary_gemm(
                Tin, Tacc, Tout, M, N, K, rows, cols, P=P, G=G, depth=depth, ii=ii
            )
        else:
            raise NotImplementedError(
                f"dataflow={dataflow} variant={variant} not implemented"
            )
        name = f"SystolicGEMM_{dataflow}_{variant}_{rows}x{cols}"
        super().__init__(name, top, s)
