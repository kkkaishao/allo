# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-self

from typing import TYPE_CHECKING
import importlib
import sys

from . import _liballo

__all__ = [
    "ir",
    "arith",
    "math",
    "scf",
    "cf",
    "ub",
    "func",
    "affine",
    "tensor",
    "memref",
    "linalg",
    "allo",
    "passes",
    "transform",
    "execution_engine",
]

_EAGER_SUBMODULES = (
    "ir",
    "arith",
    "math",
    "scf",
    "cf",
    "ub",
    "func",
    "affine",
    "tensor",
    "memref",
    "linalg",
    "allo",
    "passes",
)

if TYPE_CHECKING:
    from . import (
        ir,
        arith,
        math,
        scf,
        cf,
        ub,
        func,
        affine,
        tensor,
        memref,
        linalg,
        allo,
        passes,
        transform,
        execution_engine,
    )
else:
    for _name in _EAGER_SUBMODULES:
        _mod = getattr(_liballo, _name)
        globals()[_name] = _mod
        sys.modules[f"{__name__}.{_name}"] = _mod
    del _mod
    del _name

    def __getattr__(name: str):
        if name in {"transform", "execution_engine"}:
            mod = importlib.import_module(f".{name}", __name__)
            globals()[name] = mod
            return mod
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__():
        return sorted(set(globals()) | set(__all__))
