from typing import TYPE_CHECKING
from . import _liballo as _C

__all__ = [
    "ir",
    "utils",
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
    "transform",
]

_LAZY_SUBMODULES = frozenset(
    {
        "utils",
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
        "transform",
    }
)

if TYPE_CHECKING:
    from . import (
        ir,
        utils,
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
        transform,
    )
else:
    ir = _C.ir

    def _load(name: str):
        mod = _C._load_submodule(name)
        globals()[name] = mod
        return mod

    def __getattr__(name: str):
        if name == "ir":
            return ir
        if name in _LAZY_SUBMODULES:
            return _load(name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__():
        return sorted(set(globals()) | set(__all__))
