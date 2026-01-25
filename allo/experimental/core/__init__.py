# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import Any

from .types import (
    constexpr,
    int1,
    int2,
    int3,
    int4,
    int5,
    int6,
    int7,
    int8,
    int9,
    int10,
    int11,
    int12,
    int13,
    int14,
    int15,
    int16,
    int32,
    int64,
    uint1,
    uint2,
    uint3,
    uint4,
    uint5,
    uint6,
    uint7,
    uint8,
    uint9,
    uint10,
    uint11,
    uint12,
    uint13,
    uint14,
    uint15,
    uint16,
    uint32,
    uint64,
    APInt,
    fp16,
    fp32,
    fp64,
    bf16,
    APFloat,
    tensor,
    array,
    Stream,
    Channel,
    channel,
    tuple,
    tuple_type,
    Array,
    slice_type,
    scalar_type,
    slice,
    range,
    index,
    grid,
)

from .library import *


def __getattr__(name: str) -> Any:
    if name in ("kernel", "consteval"):
        module = import_module("allo.experimental.core.kernel")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
