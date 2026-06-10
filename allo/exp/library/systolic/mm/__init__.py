# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Systolic matrix-multiply (GEMM) library.

One entry point, :func:`make`, builds a optimized GEMM and its schedule for
any (dataflow x buffering x precision x array) combination and returns the
``Kernel`` and its ``Schedule`` for further composition. For example:

    from allo.exp.library.systolic import mm

    top = mm.make(f16, f32, f16, 128, 128, 128,
                  dataflow="ws", config="performance")
    v = top.export("vitis", part="xcu280-fsvh2892-2L-e", freq_mhz=300.0)
    v.set_axi(0, bundle="gA")   # arg 0 = A,  1 = B,  2 = C
    v.set_axi(1, bundle="gB")
    v.set_axi(2, bundle="gC")
    report = v.synth()          # or  v(A, B, C)  for csim

Choices (all introspectable as module constants):

* ``dataflow`` (:data:`DATAFLOWS`)
    - ``"os"`` output-stationary: C is accumulated in the PE. **Expects A
      PRE-TRANSPOSED ``[K,M]``** (contiguous lane reads).
    - ``"ws"`` weight-stationary: B is resident in the PE. Expects A standard
      ``[M,K]``. ~4x faster than OS for **float** (II=1 vs II=4 fadd recurrence),
      at parity for integer -- prefer it for float / latency-critical GEMM.
* ``variant`` (:data:`VARIANTS`)
    - ``"direct"`` stream operands from DRAM (re-read); minimal on-chip memory.
    - ``"buffered"`` on-chip operand buffer, read-once; ``block`` bounds its size
      (``Nc`` columns for os / ``Mc`` rows for ws; defaults to the whole operand).
    - ``"packed"`` low-bitwidth-int DSP packing (i8/i4 only): 2 columns share one
      DSP multiply -> ~2x fewer DSPs for int8 (int4 maps to LUTs -> 0 DSP).
* ``precision`` (:data:`PRECISIONS`): input->accumulate->output dtype combo.
* ``config`` (:data:`CONFIGS`) or ``array=(rows, cols)``: PE-array shape. ``rows``
  tiles M (os) or K (ws); ``cols`` tiles N. So ``M % rows == 0`` (os) /
  ``K % rows == 0`` (ws) and ``N % cols == 0`` are required.

INPUT LAYOUT (see :data:`INPUT_LAYOUT`): operand ``A`` differs by dataflow -- os
takes ``A_T [K,M]`` (pre-transposed), ws takes ``A [M,K]``. ``B`` is ``[K,N]`` and
``C`` is ``[M,N]`` for both.

For full control, the underlying factories are re-exported (they return
``(top, top_s)`` and take explicit Allo ``DType`` objects).
"""

from typing import Literal
from ....lang.core import i4, i8, i16, i32, f16, f32
from .os_direct import make_direct_output_stationary_gemm
from .os_buffered import make_buffered_output_stationary_gemm
from .os_packed import make_packed_output_stationary_gemm
from .ws_direct import make_direct_weight_stationary_gemm
from .ws_buffered import make_buffered_weight_stationary_gemm
from .ws_packed import make_packed_weight_stationary_gemm

# PE-array presets: name -> (rows, cols). rows tiles M (os) / K (ws); cols tiles N.
CONFIGS: dict[str, tuple] = {
    "balanced": (8, 8),  # 64 PEs, fits small parts
    "performance": (16, 16),  # 256 PEs, maximizes throughput
}

DATAFLOWS = ("os", "ws")
VARIANTS = ("direct", "buffered", "packed")

# Operand-A layout each dataflow expects (B is [K,N], C is [M,N] for both).
INPUT_LAYOUT = {
    "os": "A_T [K, M] (pre-transposed)",
    "ws": "A [M, K]",
}

_FACTORY = {
    ("os", "direct"): make_direct_output_stationary_gemm,
    ("os", "buffered"): make_buffered_output_stationary_gemm,
    ("os", "packed"): make_packed_output_stationary_gemm,
    ("ws", "direct"): make_direct_weight_stationary_gemm,
    ("ws", "buffered"): make_buffered_weight_stationary_gemm,
    ("ws", "packed"): make_packed_weight_stationary_gemm,
}
# The buffered block-size keyword each dataflow's factory takes.
_BLOCK_KW = {"os": "Nc", "ws": "Mc"}


def make(
    Tin,
    Tacc,
    Tout,
    M,
    N,
    K,
    *,
    dataflow: Literal["os", "ws"] = "os",
    variant: Literal["direct", "buffered"] = "direct",
    config: Literal["balanced", "performance"] = "performance",
    array: tuple[int, ...] | None = None,
    block: int | None = None,
    depth=2,
    ii=1,
):
    """Build a scheduled systolic GEMM for the given configuration.
    Returns the original ``Kernel`` and its ``Schedule`` for further composition.

    See the module docstring for the meaning of each
    argument, the precision/config presets, and -- importantly -- the per-dataflow
    ``A`` input layout (:data:`INPUT_LAYOUT`).

    ``block`` (buffered variants only) bounds the on-chip operand buffer: it is the
    ``Nc`` column-block for ``os`` and the ``Mc`` row-block for ``ws``; ``None``
    buffers the whole operand.

    ``array`` is an alternative to ``config`` for specifying the PE-array shape as
    a (rows, cols) tuple. If both are given, ``array`` takes precedence over the
    preset specified by ``config``.

    ``depth`` and ``ii`` are the systolic array's pipeline depth and initiation interval,
    respectively. The default of ``depth=2, ii=1`` is a common choice for high performance.
    """
    if dataflow not in DATAFLOWS:
        raise ValueError(
            f"unknown dataflow {dataflow!r}; choose from {list(DATAFLOWS)}"
        )
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    if array is None:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; choose from {list(CONFIGS)}")
        array = CONFIGS[config]
    rows, cols = array

    kwargs = {"depth": depth, "ii": ii}
    if block is not None:
        if variant != "buffered":
            raise ValueError("`block` is only valid for variant='buffered'")
        kwargs[_BLOCK_KW[dataflow]] = block
    if variant == "packed":
        # DSP packing pairs 2 columns/DSP; only int8/int4 benefit (wider products
        # exceed the 27-bit DSP port -> no saving). gap > product width: 2*w+2.
        if not (Tin.is_int() and Tin.primitive_width <= 8):
            raise ValueError(
                "variant='packed' supports only low-bitwidth int (i8/i4); "
                "use 'direct' for int16/float"
            )
        kwargs["P"] = 2
        kwargs["G"] = 2 * Tin.primitive_width + 2

    _top, top_s = _FACTORY[(dataflow, variant)](
        Tin, Tacc, Tout, M, N, K, rows, cols, **kwargs
    )
    return _top, top_s


__all__ = [
    "CONFIGS",
    "DATAFLOWS",
    "VARIANTS",
    "INPUT_LAYOUT",
    "make",
    "make_direct_output_stationary_gemm",
    "make_buffered_output_stationary_gemm",
    "make_packed_output_stationary_gemm",
    "make_direct_weight_stationary_gemm",
    "make_buffered_weight_stationary_gemm",
    "make_packed_weight_stationary_gemm",
]
