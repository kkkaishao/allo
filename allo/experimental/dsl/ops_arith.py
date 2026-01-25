# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins

from .. import core
from ..compiler.builder import AlloOpBuilder
from ..core.library import NO_FOLD, CmpPred, operation
from ..core.types import constexpr
from .ops_common import (
    _ScalarOrArray,
    _binary_op_checks,
    _binary_op_create,
    _binary_op_create_signed,
    _cmp_op_create,
    _unary_op_checks,
    _unary_op_create,
)


@operation
def add(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@add.validate
def _validate_add(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "add")


@add.const_fold
def _fold_add(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value + rhs.value)
    return NO_FOLD


@add.lower
def _lower_add(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "add", builder.create_add)


@operation
def sub(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@sub.validate
def _validate_sub(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "sub")


@sub.const_fold
def _fold_sub(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value - rhs.value)
    return NO_FOLD


@sub.lower
def _lower_sub(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "sub", builder.create_sub)


@operation
def mul(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@mul.validate
def _validate_mul(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "mul")


@mul.const_fold
def _fold_mul(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value * rhs.value)
    return NO_FOLD


@mul.lower
def _lower_mul(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "mul", builder.create_mul)


@operation
def div(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, signed=constexpr(False)
) -> _ScalarOrArray:
    pass


@div.validate
def _validate_div(lhs, rhs, signed=constexpr(False)) -> str:
    if not isinstance(signed, constexpr) or not isinstance(signed.value, bool):
        return (
            f"div operation requires 'signed' to be a boolean constexpr, got {signed}."
        )
    return _binary_op_checks(lhs, rhs, "div")


@div.const_fold
def _fold_div(lhs, rhs, signed=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value / rhs.value)
    return NO_FOLD


@div.lower
def _lower_div(builder: AlloOpBuilder, lhs, rhs, signed=constexpr(False)):
    return _binary_op_create_signed(
        builder, lhs, rhs, "div", builder.create_div, signed.value
    )


@operation
def floordiv(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, signed=constexpr(False)
) -> _ScalarOrArray:
    pass


@floordiv.validate
def _validate_floordiv(lhs, rhs, signed=constexpr(False)) -> str:
    if not isinstance(signed, constexpr) or not isinstance(signed.value, bool):
        return (
            "floordiv operation requires 'signed' to be a boolean constexpr, "
            f"got {signed}."
        )
    return _binary_op_checks(lhs, rhs, "floordiv")


@floordiv.const_fold
def _fold_floordiv(lhs, rhs, signed=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value // rhs.value)
    return NO_FOLD


@floordiv.lower
def _lower_floordiv(builder: AlloOpBuilder, lhs, rhs, signed=constexpr(False)):
    return _binary_op_create_signed(
        builder, lhs, rhs, "floordiv", builder.create_floordiv, signed.value
    )


@operation
def mod(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, signed=constexpr(False)
) -> _ScalarOrArray:
    pass


@mod.validate
def _validate_mod(lhs, rhs, signed=constexpr(False)) -> str:
    if not isinstance(signed, constexpr) or not isinstance(signed.value, bool):
        return (
            f"mod operation requires 'signed' to be a boolean constexpr, got {signed}."
        )
    return _binary_op_checks(lhs, rhs, "mod")


@mod.const_fold
def _fold_mod(lhs, rhs, signed=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value % rhs.value)
    return NO_FOLD


@mod.lower
def _lower_mod(builder: AlloOpBuilder, lhs, rhs, signed=constexpr(False)):
    return _binary_op_create_signed(
        builder, lhs, rhs, "mod", builder.create_mod, signed.value
    )


@operation
def pow(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@pow.validate
def _validate_pow(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "pow")


@pow.const_fold
def _fold_pow(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value**rhs.value)
    return NO_FOLD


@pow.lower
def _lower_pow(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "pow", builder.create_pow)


@operation
def lshift(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@lshift.validate
def _validate_lshift(lhs, rhs) -> str:
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "lshift operation does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return _binary_op_checks(lhs, rhs, "lshift")


@lshift.const_fold
def _fold_lshift(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value << rhs.value)
    return NO_FOLD


@lshift.lower
def _lower_lshift(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "lshift", builder.create_lshift)


@operation
def rshift(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, signed=constexpr(False)
) -> _ScalarOrArray:
    pass


@rshift.validate
def _validate_rshift(lhs, rhs, signed=constexpr(False)) -> str:
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "rshift operation does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    if not isinstance(signed, constexpr) or not isinstance(signed.value, bool):
        return (
            "rshift operation requires 'signed' argument to be a boolean constexpr, "
            f"got {signed}."
        )
    return _binary_op_checks(lhs, rhs, "rshift")


@rshift.const_fold
def _fold_rshift(lhs, rhs, signed: constexpr):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        if signed.value:
            return constexpr(lhs.value >> rhs.value)
        return constexpr((lhs.value & ((1 << 64) - 1)) >> rhs.value)
    return NO_FOLD


@rshift.lower
def _lower_rshift(builder: AlloOpBuilder, lhs, rhs, signed: constexpr):
    return _binary_op_create_signed(
        builder, lhs, rhs, "rshift", builder.create_rshift, signed.value
    )


@operation
def bitwise_and(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@bitwise_and.validate
def _validate_bitwise_and(lhs, rhs) -> str:
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_and operation does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return _binary_op_checks(lhs, rhs, "bitwise_and")


@bitwise_and.const_fold
def _fold_bitwise_and(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value & rhs.value)
    return NO_FOLD


@bitwise_and.lower
def _lower_bitwise_and(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(
        builder, lhs, rhs, "bitwise_and", builder.create_bitwise_and
    )


@operation
def bitwise_or(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@bitwise_or.validate
def _validate_bitwise_or(lhs, rhs) -> str:
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_or operation does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return _binary_op_checks(lhs, rhs, "bitwise_or")


@bitwise_or.const_fold
def _fold_bitwise_or(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value | rhs.value)
    return NO_FOLD


@bitwise_or.lower
def _lower_bitwise_or(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "bitwise_or", builder.create_bitwise_or)


@operation
def bitwise_xor(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@bitwise_xor.validate
def _validate_bitwise_xor(lhs, rhs) -> str:
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_xor operation does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return _binary_op_checks(lhs, rhs, "bitwise_xor")


@bitwise_xor.const_fold
def _fold_bitwise_xor(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value ^ rhs.value)
    return NO_FOLD


@bitwise_xor.lower
def _lower_bitwise_xor(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(
        builder, lhs, rhs, "bitwise_xor", builder.create_bitwise_xor
    )


def _validate_cmp_ordered_arg(ordered) -> str:
    if not isinstance(ordered, constexpr) or not isinstance(ordered.value, bool):
        return f"comparison requires 'ordered' to be boolean constexpr, got {ordered}."
    return ""


@operation
def eq(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@eq.validate
def _validate_eq(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "eq")


@eq.const_fold
def _fold_eq(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value == rhs.value)
    return NO_FOLD


@eq.lower
def _lower_eq(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.EQ, "eq", ordered.value)


@operation
def ne(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@ne.validate
def _validate_ne(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "ne")


@ne.const_fold
def _fold_ne(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value != rhs.value)
    return NO_FOLD


@ne.lower
def _lower_ne(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.NE, "ne", ordered.value)


@operation
def lt(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@lt.validate
def _validate_lt(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "lt")


@lt.const_fold
def _fold_lt(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value < rhs.value)
    return NO_FOLD


@lt.lower
def _lower_lt(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.LT, "lt", ordered.value)


@operation
def le(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@le.validate
def _validate_le(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "le")


@le.const_fold
def _fold_le(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value <= rhs.value)
    return NO_FOLD


@le.lower
def _lower_le(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.LE, "le", ordered.value)


@operation
def gt(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@gt.validate
def _validate_gt(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "gt")


@gt.const_fold
def _fold_gt(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value > rhs.value)
    return NO_FOLD


@gt.lower
def _lower_gt(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.GT, "gt", ordered.value)


@operation
def ge(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, ordered=constexpr(False)
) -> _ScalarOrArray:
    pass


@ge.validate
def _validate_ge(lhs, rhs, ordered=constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else _binary_op_checks(lhs, rhs, "ge")


@ge.const_fold
def _fold_ge(lhs, rhs, ordered=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(lhs.value >= rhs.value)
    return NO_FOLD


@ge.lower
def _lower_ge(builder: AlloOpBuilder, lhs, rhs, ordered=constexpr(False)):
    return _cmp_op_create(builder, lhs, rhs, CmpPred.GE, "ge", ordered.value)


@operation
def neg(lhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@neg.validate
def _validate_neg(lhs) -> str:
    return _unary_op_checks(lhs, "neg")


@neg.const_fold
def _fold_neg(lhs):
    if isinstance(lhs, constexpr):
        return constexpr(-lhs.value)
    return NO_FOLD


@neg.lower
def _lower_neg(builder: AlloOpBuilder, lhs):
    return _unary_op_create(builder, lhs, "neg", builder.create_neg)


@operation
def invert(lhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@invert.validate
def _validate_invert(lhs) -> str:
    if not isinstance(lhs, constexpr) and lhs.dtype.is_float():
        return (
            f"invert operation does not support floating point types, got {lhs.dtype}."
        )
    return _unary_op_checks(lhs, "invert")


@invert.const_fold
def _fold_invert(lhs):
    if isinstance(lhs, constexpr):
        return constexpr(~lhs.value)
    return NO_FOLD


@invert.lower
def _lower_invert(builder: AlloOpBuilder, lhs):
    return _unary_op_create(builder, lhs, "invert", builder.create_invert)


@operation
def pos(lhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@pos.const_fold
def _fold_pos(lhs):
    if isinstance(lhs, constexpr):
        return lhs
    return NO_FOLD


@pos.lower
def _lower_pos(builder: AlloOpBuilder, lhs):  # noqa: ARG001
    return lhs


@operation
def logical_and(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@logical_and.validate
def _validate_logical_and(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "logical_and")


@logical_and.const_fold
def _fold_logical_and(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(bool(lhs.value) and bool(rhs.value))
    return NO_FOLD


@logical_and.lower
def _lower_logical_and(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(
        builder, lhs, rhs, "logical_and", builder.create_logical_and
    )


@operation
def logical_or(lhs: _ScalarOrArray, rhs: _ScalarOrArray) -> _ScalarOrArray:
    pass


@logical_or.validate
def _validate_logical_or(lhs, rhs) -> str:
    return _binary_op_checks(lhs, rhs, "logical_or")


@logical_or.const_fold
def _fold_logical_or(lhs, rhs):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(bool(lhs.value) or bool(rhs.value))
    return NO_FOLD


@logical_or.lower
def _lower_logical_or(builder: AlloOpBuilder, lhs, rhs):
    return _binary_op_create(builder, lhs, rhs, "logical_or", builder.create_logical_or)


@operation
def logical_not(operand: _ScalarOrArray) -> _ScalarOrArray:
    pass


@logical_not.validate
def _validate_logical_not(operand) -> str:
    return _unary_op_checks(operand, "logical_not")


@logical_not.const_fold
def _fold_logical_not(operand):
    if isinstance(operand, constexpr):
        return constexpr(not bool(operand.value))
    return NO_FOLD


@logical_not.lower
def _lower_logical_not(builder: AlloOpBuilder, operand):
    return _unary_op_create(builder, operand, "logical_not", builder.create_logical_not)


@operation
def max(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, propagate_nan=False
) -> _ScalarOrArray:
    pass


@max.validate
def _validate_max(lhs, rhs, propagate_nan=constexpr(False)) -> str:
    if not isinstance(propagate_nan, constexpr) or not isinstance(
        propagate_nan.value, bool
    ):
        return (
            "max operation requires 'propagate_nan' argument to be boolean constexpr, "
            f"got {propagate_nan}."
        )
    msg = _binary_op_checks(lhs, rhs, "max")
    if msg:
        return msg
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if (lhs.dtype.is_uint() and rhs.dtype.is_int()) or (
            lhs.dtype.is_int() and rhs.dtype.is_uint()
        ):
            return (
                "max operation does not allow implicit mixing of signed/unsigned integers, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return ""


@max.const_fold
def _fold_max(lhs, rhs, propagate_nan=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(builtins.max(lhs.value, rhs.value))
    return NO_FOLD


@max.lower
def _lower_max(builder: AlloOpBuilder, lhs, rhs, propagate_nan=constexpr(False)):
    lhs_rhs = _binary_op_create(builder, lhs, rhs, "max", lambda a, b: (a, b))
    if lhs_rhs is NO_FOLD:
        builder.compile_error("max constexpr folding failed unexpectedly")
    lhs, rhs = lhs_rhs
    return builder.create_max(
        lhs, rhs, signed=lhs.dtype.is_int(), propagate_nan=propagate_nan.value
    )


@operation
def min(
    lhs: _ScalarOrArray, rhs: _ScalarOrArray, propagate_nan=False
) -> _ScalarOrArray:
    pass


@min.validate
def _validate_min(lhs, rhs, propagate_nan=constexpr(False)) -> str:
    if not isinstance(propagate_nan, constexpr) or not isinstance(
        propagate_nan.value, bool
    ):
        return (
            "min operation requires 'propagate_nan' argument to be boolean constexpr, "
            f"got {propagate_nan}."
        )
    msg = _binary_op_checks(lhs, rhs, "min")
    if msg:
        return msg
    if not isinstance(lhs, constexpr) and not isinstance(rhs, constexpr):
        if (lhs.dtype.is_uint() and rhs.dtype.is_int()) or (
            lhs.dtype.is_int() and rhs.dtype.is_uint()
        ):
            return (
                "min operation does not allow implicit mixing of signed/unsigned integers, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return ""


@min.const_fold
def _fold_min(lhs, rhs, propagate_nan=constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return constexpr(builtins.min(lhs.value, rhs.value))
    return NO_FOLD


@min.lower
def _lower_min(builder: AlloOpBuilder, lhs, rhs, propagate_nan=constexpr(False)):
    lhs_rhs = _binary_op_create(builder, lhs, rhs, "min", lambda a, b: (a, b))
    if lhs_rhs is NO_FOLD:
        builder.compile_error("min constexpr folding failed unexpectedly")
    lhs, rhs = lhs_rhs
    # Intentionally call create_min (bugfix from old path that called create_max).
    return builder.create_min(
        lhs, rhs, signed=lhs.dtype.is_int(), propagate_nans=propagate_nan.value
    )
