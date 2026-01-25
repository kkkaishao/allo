from importlib import import_module
from typing import Any

_PUBLIC_MODULES = ("allo.experimental.dsl.builtins", "allo.experimental.dsl.math")
__all__ = [
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
]


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
