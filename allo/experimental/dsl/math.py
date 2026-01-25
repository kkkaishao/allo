# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Union

from .. import core, settings
from .._C.liballo import linalg as linalg_d, math as math_d
from ..compiler.builder import AlloOpBuilder
from ..core.library import NO_FOLD, operation
from ..core.types import constexpr, tensor
from .ops_common import _ScalarOrArray, _unary_op_checks, _unary_op_create


@operation
def sqrt(x: _ScalarOrArray) -> _ScalarOrArray:
    pass


@sqrt.validate
def _validate_sqrt(x) -> str:
    return _unary_op_checks(x, "sqrt")


@sqrt.const_fold
def _fold_sqrt(x):
    if isinstance(x, constexpr):
        return constexpr(x.value**0.5)
    return NO_FOLD


@sqrt.lower
def _lower_sqrt(builder: AlloOpBuilder, x: Union[tensor, constexpr]):
    def create_sqrt(val: tensor) -> tensor:
        if isinstance(val.type, core.Array) and settings.USE_TENSOR:
            ret = linalg_d.SqrtOp.create(builder, val.handle, val.handle).get_result_at(
                0
            )
        else:
            ret = math_d.SqrtOp.create(builder, val.handle)
        return tensor(ret, val.type)

    return _unary_op_create(builder, x, "sqrt", create_sqrt)
