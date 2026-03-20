# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins

from ..compiler.builder import AlloOpBuilder, CmpPred
from ..core.library import NO_FOLD, operator
from ..core.types import Constexpr, DType, ShapedType, BaseType
from .ops_common import (
    binary_op_checks,
    cmp_op_create,
    lower_binary_op,
    lower_unary_op,
    unary_op_checks,
)


@operator
def add(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@add.validate
def _validate_add(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "add")


@add.const_fold
def _fold_add(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value + rhs.value)
    return NO_FOLD


@add.lower
def _lower_add(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "add", builder.create_add)


@operator
def sub(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@sub.validate
def _validate_sub(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "sub")


@sub.const_fold
def _fold_sub(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value - rhs.value)
    return NO_FOLD


@sub.lower
def _lower_sub(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "sub", builder.create_sub)


@operator
def mul(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@mul.validate
def _validate_mul(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "mul")


@mul.const_fold
def _fold_mul(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value * rhs.value)
    return NO_FOLD


@mul.lower
def _lower_mul(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "mul", builder.create_mul)


@operator
def div(
    lhs: DType | ShapedType, rhs: DType | ShapedType, signed=Constexpr(False)
) -> DType | ShapedType:
    pass


@div.validate
def _validate_div(lhs, rhs, signed=Constexpr(False)) -> str:
    if not isinstance(signed, Constexpr) or not isinstance(signed.value, bool):
        return (
            f"div operator requires 'signed' to be a boolean Constexpr, got {signed}."
        )
    return binary_op_checks(lhs, rhs, "div")


@div.const_fold
def _fold_div(lhs, rhs, signed=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value / rhs.value)
    return NO_FOLD


@div.lower
def _lower_div(builder: AlloOpBuilder, lhs, rhs, signed=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "div",
        builder.create_div,
        signed=signed.value,
    )


@operator
def floordiv(
    lhs: DType | ShapedType, rhs: DType | ShapedType, signed=Constexpr(False)
) -> DType | ShapedType:
    pass


@floordiv.validate
def _validate_floordiv(lhs, rhs, signed=Constexpr(False)) -> str:
    if not isinstance(signed, Constexpr) or not isinstance(signed.value, bool):
        return (
            "floordiv operator requires 'signed' to be a boolean Constexpr, "
            f"got {signed}."
        )
    return binary_op_checks(lhs, rhs, "floordiv")


@floordiv.const_fold
def _fold_floordiv(lhs, rhs, signed=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value // rhs.value)
    return NO_FOLD


@floordiv.lower
def _lower_floordiv(builder: AlloOpBuilder, lhs, rhs, signed=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "floordiv",
        builder.create_floordiv,
        signed=signed.value,
    )


@operator
def mod(
    lhs: DType | ShapedType, rhs: DType | ShapedType, signed=Constexpr(False)
) -> DType | ShapedType:
    pass


@mod.validate
def _validate_mod(lhs, rhs, signed=Constexpr(False)) -> str:
    if not isinstance(signed, Constexpr) or not isinstance(signed.value, bool):
        return (
            f"mod operator requires 'signed' to be a boolean Constexpr, got {signed}."
        )
    return binary_op_checks(lhs, rhs, "mod")


@mod.const_fold
def _fold_mod(lhs, rhs, signed=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value % rhs.value)
    return NO_FOLD


@mod.lower
def _lower_mod(builder: AlloOpBuilder, lhs, rhs, signed=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "mod",
        builder.create_mod,
        signed=signed.value,
    )


@operator
def pow(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@pow.validate
def _validate_pow(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "pow")


@pow.const_fold
def _fold_pow(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value**rhs.value)
    return NO_FOLD


@pow.lower
def _lower_pow(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "pow", builder.create_pow)


@operator
def lshift(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@lshift.validate
def _validate_lshift(lhs, rhs) -> str:
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "lshift operator does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return binary_op_checks(lhs, rhs, "lshift")


@lshift.const_fold
def _fold_lshift(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value << rhs.value)
    return NO_FOLD


@lshift.lower
def _lower_lshift(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "lshift", builder.create_lshift)


@operator
def rshift(
    lhs: DType | ShapedType, rhs: DType | ShapedType, signed=Constexpr(False)
) -> DType | ShapedType:
    pass


@rshift.validate
def _validate_rshift(lhs, rhs, signed=Constexpr(False)) -> str:
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "rshift operator does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    if not isinstance(signed, Constexpr) or not isinstance(signed.value, bool):
        return (
            "rshift operator requires 'signed' argument to be a boolean Constexpr, "
            f"got {signed}."
        )
    return binary_op_checks(lhs, rhs, "rshift")


@rshift.const_fold
def _fold_rshift(lhs, rhs, signed=Constexpr(False)):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        if signed.value:
            return Constexpr(lhs.value >> rhs.value)
        return Constexpr((lhs.value & ((1 << 64) - 1)) >> rhs.value)
    return NO_FOLD


@rshift.lower
def _lower_rshift(builder: AlloOpBuilder, lhs, rhs, signed=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "rshift",
        builder.create_rshift,
        signed=signed.value,
    )


@operator
def bitwise_and(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@bitwise_and.validate
def _validate_bitwise_and(lhs, rhs) -> str:
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_and operator does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return binary_op_checks(lhs, rhs, "bitwise_and")


@bitwise_and.const_fold
def _fold_bitwise_and(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value & rhs.value)
    return NO_FOLD


@bitwise_and.lower
def _lower_bitwise_and(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "bitwise_and", builder.create_and)


@operator
def bitwise_or(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@bitwise_or.validate
def _validate_bitwise_or(lhs, rhs) -> str:
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_or operator does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return binary_op_checks(lhs, rhs, "bitwise_or")


@bitwise_or.const_fold
def _fold_bitwise_or(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value | rhs.value)
    return NO_FOLD


@bitwise_or.lower
def _lower_bitwise_or(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "bitwise_or", builder.create_or)


@operator
def bitwise_xor(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@bitwise_xor.validate
def _validate_bitwise_xor(lhs, rhs) -> str:
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if lhs.dtype.is_float() or rhs.dtype.is_float():
            return (
                "bitwise_xor operator does not support floating point types, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return binary_op_checks(lhs, rhs, "bitwise_xor")


@bitwise_xor.const_fold
def _fold_bitwise_xor(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value ^ rhs.value)
    return NO_FOLD


@bitwise_xor.lower
def _lower_bitwise_xor(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "bitwise_xor", builder.create_xor)


def _validate_cmp_ordered_arg(ordered) -> str:
    if not isinstance(ordered, Constexpr) or not isinstance(ordered.value, bool):
        return f"comparison requires 'ordered' to be boolean Constexpr, got {ordered}."
    return ""


@operator
def eq(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@eq.validate
def _validate_eq(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    if isinstance(lhs, BaseType) and isinstance(rhs, BaseType):
        return msg if msg else ""
    return msg if msg else binary_op_checks(lhs, rhs, "eq")


@eq.const_fold
def _fold_eq(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value == rhs.value)
    if isinstance(lhs, BaseType) and isinstance(rhs, BaseType):
        return Constexpr(lhs == rhs)
    return NO_FOLD


@eq.lower
def _lower_eq(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.EQ, "eq", ordered.value)


@operator
def ne(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@ne.validate
def _validate_ne(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    if isinstance(lhs, BaseType) and isinstance(rhs, BaseType):
        return msg if msg else ""
    return msg if msg else binary_op_checks(lhs, rhs, "ne")


@ne.const_fold
def _fold_ne(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value != rhs.value)
    if isinstance(lhs, BaseType) and isinstance(rhs, BaseType):
        return Constexpr(lhs != rhs)
    return NO_FOLD


@ne.lower
def _lower_ne(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.NE, "ne", ordered.value)


@operator
def lt(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@lt.validate
def _validate_lt(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else binary_op_checks(lhs, rhs, "lt")


@lt.const_fold
def _fold_lt(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value < rhs.value)
    return NO_FOLD


@lt.lower
def _lower_lt(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.LT, "lt", ordered.value)


@operator
def le(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@le.validate
def _validate_le(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else binary_op_checks(lhs, rhs, "le")


@le.const_fold
def _fold_le(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value <= rhs.value)
    return NO_FOLD


@le.lower
def _lower_le(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.LE, "le", ordered.value)


@operator
def gt(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@gt.validate
def _validate_gt(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else binary_op_checks(lhs, rhs, "gt")


@gt.const_fold
def _fold_gt(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value > rhs.value)
    return NO_FOLD


@gt.lower
def _lower_gt(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.GT, "gt", ordered.value)


@operator
def ge(
    lhs: DType | ShapedType, rhs: DType | ShapedType, ordered=Constexpr(False)
) -> DType | ShapedType:
    pass


@ge.validate
def _validate_ge(lhs, rhs, ordered=Constexpr(False)) -> str:
    msg = _validate_cmp_ordered_arg(ordered)
    return msg if msg else binary_op_checks(lhs, rhs, "ge")


@ge.const_fold
def _fold_ge(lhs, rhs, ordered=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(lhs.value >= rhs.value)
    return NO_FOLD


@ge.lower
def _lower_ge(builder: AlloOpBuilder, lhs, rhs, ordered=Constexpr(False)):
    return cmp_op_create(builder, lhs, rhs, CmpPred.GE, "ge", ordered.value)


@operator
def neg(lhs: DType | ShapedType) -> DType | ShapedType:
    pass


@neg.validate
def _validate_neg(lhs) -> str:
    return unary_op_checks(lhs, "neg")


@neg.const_fold
def _fold_neg(lhs):
    if isinstance(lhs, Constexpr):
        return Constexpr(-lhs.value)
    return NO_FOLD


@neg.lower
def _lower_neg(builder: AlloOpBuilder, lhs):
    return lower_unary_op(builder, lhs, "neg", builder.create_neg)


@operator
def invert(lhs: DType | ShapedType) -> DType | ShapedType:
    pass


@invert.validate
def _validate_invert(lhs) -> str:
    if not isinstance(lhs, Constexpr) and lhs.dtype.is_float():
        return (
            f"invert operator does not support floating point types, got {lhs.dtype}."
        )
    return unary_op_checks(lhs, "invert")


@invert.const_fold
def _fold_invert(lhs):
    if isinstance(lhs, Constexpr):
        return Constexpr(~lhs.value)
    return NO_FOLD


@invert.lower
def _lower_invert(builder: AlloOpBuilder, lhs):
    return lower_unary_op(builder, lhs, "invert", builder.create_invert)


@operator
def pos(lhs: DType | ShapedType) -> DType | ShapedType:
    pass


@pos.const_fold
def _fold_pos(lhs):
    if isinstance(lhs, Constexpr):
        return lhs
    return NO_FOLD


@pos.lower
def _lower_pos(builder: AlloOpBuilder, lhs):  # noqa: ARG001
    return lhs


@operator
def logical_and(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@logical_and.validate
def _validate_logical_and(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "logical_and")


@logical_and.const_fold
def _fold_logical_and(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(bool(lhs.value) and bool(rhs.value))
    return NO_FOLD


@logical_and.lower
def _lower_logical_and(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "logical_and", builder.create_logical_and)


@operator
def logical_or(lhs: DType | ShapedType, rhs: DType | ShapedType) -> DType | ShapedType:
    pass


@logical_or.validate
def _validate_logical_or(lhs, rhs) -> str:
    return binary_op_checks(lhs, rhs, "logical_or")


@logical_or.const_fold
def _fold_logical_or(lhs, rhs):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(bool(lhs.value) or bool(rhs.value))
    return NO_FOLD


@logical_or.lower
def _lower_logical_or(builder: AlloOpBuilder, lhs, rhs):
    return lower_binary_op(builder, lhs, rhs, "logical_or", builder.create_logical_or)


@operator
def logical_not(operand: DType | ShapedType) -> DType | ShapedType:
    pass


@logical_not.validate
def _validate_logical_not(operand) -> str:
    return unary_op_checks(operand, "logical_not")


@logical_not.const_fold
def _fold_logical_not(operand):
    if isinstance(operand, Constexpr):
        return Constexpr(not bool(operand.value))
    return NO_FOLD


@logical_not.lower
def _lower_logical_not(builder: AlloOpBuilder, operand):
    return lower_unary_op(
        builder,
        operand,
        "logical_not",
        builder.create_logical_not,
        promote=False,
    )


@operator
def max(
    lhs: DType | ShapedType, rhs: DType | ShapedType, propagate_nan=False
) -> DType | ShapedType:
    pass


@max.validate
def _validate_max(lhs, rhs, propagate_nan=Constexpr(False)) -> str:
    if not isinstance(propagate_nan, Constexpr) or not isinstance(
        propagate_nan.value, bool
    ):
        return (
            "max operator requires 'propagate_nan' argument to be boolean Constexpr, "
            f"got {propagate_nan}."
        )
    msg = binary_op_checks(lhs, rhs, "max")
    if msg:
        return msg
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if (lhs.dtype.is_uint() and rhs.dtype.is_int()) or (
            lhs.dtype.is_int() and rhs.dtype.is_uint()
        ):
            return (
                "max operator does not allow implicit mixing of signed/unsigned integers, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return ""


@max.const_fold
def _fold_max(lhs, rhs, propagate_nan=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(builtins.max(lhs.value, rhs.value))
    return NO_FOLD


@max.lower
def _lower_max(builder: AlloOpBuilder, lhs, rhs, propagate_nan=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "max",
        builder.create_max,
        signed=lhs.dtype.is_int(),
        floating=lhs.dtype.is_float(),
        extra_kwargs={"propagate_nan": propagate_nan.value},
    )


@operator
def min(
    lhs: DType | ShapedType, rhs: DType | ShapedType, propagate_nan=False
) -> DType | ShapedType:
    pass


@min.validate
def _validate_min(lhs, rhs, propagate_nan=Constexpr(False)) -> str:
    if not isinstance(propagate_nan, Constexpr) or not isinstance(
        propagate_nan.value, bool
    ):
        return (
            "min operator requires 'propagate_nan' argument to be boolean Constexpr, "
            f"got {propagate_nan}."
        )
    msg = binary_op_checks(lhs, rhs, "min")
    if msg:
        return msg
    if not isinstance(lhs, Constexpr) and not isinstance(rhs, Constexpr):
        if (lhs.dtype.is_uint() and rhs.dtype.is_int()) or (
            lhs.dtype.is_int() and rhs.dtype.is_uint()
        ):
            return (
                "min operator does not allow implicit mixing of signed/unsigned integers, "
                f"got {lhs.dtype} and {rhs.dtype}."
            )
    return ""


@min.const_fold
def _fold_min(lhs, rhs, propagate_nan=Constexpr(False)):  # noqa: ARG001
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return Constexpr(builtins.min(lhs.value, rhs.value))
    return NO_FOLD


@min.lower
def _lower_min(builder: AlloOpBuilder, lhs, rhs, propagate_nan=Constexpr(False)):
    return lower_binary_op(
        builder,
        lhs,
        rhs,
        "min",
        builder.create_min,
        signed=lhs.dtype.is_int(),
        floating=lhs.dtype.is_float(),
        extra_kwargs={"propagate_nan": propagate_nan.value},
    )
