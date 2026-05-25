# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=import-self

import sys

from . import _liballo

_NATIVE_SUBMODULES = (
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
    "schedule",
    "transform",
)

for _name in _NATIVE_SUBMODULES:
    _mod = getattr(_liballo, _name)
    globals()[_name] = _mod
    sys.modules[f"{__name__}.{_name}"] = _mod

from . import execution_engine as execution_engine
