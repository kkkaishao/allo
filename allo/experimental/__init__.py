# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import Any

from . import settings as settings

_PUBLIC_MODULES = ("allo.experimental.core", "allo.experimental.dsl")
_CORE_EXPORTS = (
    "constexpr",
    "int1",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "int8",
    "int9",
    "int10",
    "int11",
    "int12",
    "int13",
    "int14",
    "int15",
    "int16",
    "int32",
    "int64",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "uint8",
    "uint9",
    "uint10",
    "uint11",
    "uint12",
    "uint13",
    "uint14",
    "uint15",
    "uint16",
    "uint32",
    "uint64",
    "APInt",
    "fp16",
    "fp32",
    "fp64",
    "bf16",
    "APFloat",
    "tensor",
    "array",
    "Stream",
    "Channel",
    "channel",
    "tuple",
    "tuple_type",
    "Array",
    "slice_type",
    "scalar_type",
    "slice",
    "range",
    "index",
    "grid",
    "CmpPred",
    "Operation",
    "BoundOperation",
    "operation",
    "resolve_operation",
    "as_member_function",
    "kernel",
    "consteval",
)
_DSL_EXPORTS = (
    "load",
    "store",
    "add",
    "sub",
    "mul",
    "div",
    "floordiv",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "pos",
    "neg",
    "invert",
    "logical_and",
    "logical_not",
    "logical_or",
    "cast",
    "max",
    "min",
    "acq_buf",
    "rel_buf",
    "get_pid",
    "get_num_progs",
    "to_stream",
    "put",
    "get",
    "sqrt",
)
__all__ = [*_CORE_EXPORTS, *_DSL_EXPORTS, "settings"]


def __getattr__(name: str) -> Any:
    for module_name in _PUBLIC_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
