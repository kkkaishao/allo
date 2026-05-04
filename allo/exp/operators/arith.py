from __future__ import annotations

import builtins

from ..compiler.builder import AlloOpBuilder, CmpPred
from ..lang.core import AlloValue, ConstexprValue, TypeBase
from ..lang.operator import NO_FOLD, operator
from .utils import (
    cmp_op_create,
    lower_binary_op,
    lower_unary_op,
    operator_body_unreachable,
)


def _fold_binary(lhs, rhs, fn):
    if not (isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue)):
        return NO_FOLD
    try:
        return ConstexprValue(fn(lhs.value, rhs.value))
    except Exception:
        return NO_FOLD


def _fold_unary(operand, fn):
    if not isinstance(operand, ConstexprValue):
        return NO_FOLD
    try:
        return ConstexprValue(fn(operand.value))
    except Exception:
        return NO_FOLD


def _fold_const_bool(value):
    if isinstance(value, ConstexprValue) and isinstance(value.value, bool):
        return value.value
    return None


def _const_bool(value, name: str) -> bool:
    assert isinstance(value, ConstexprValue) and isinstance(
        value.value, bool
    ), f"'{name}' must be a boolean constexpr"
    return value.value


@operator
def add(x, y):
    operator_body_unreachable()


@add.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs + rhs)


@add.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "add", builder.create_add)


@operator
def sub(x, y):
    operator_body_unreachable()


@sub.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs - rhs)


@sub.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "sub", builder.create_sub)


@operator
def mul(x, y):
    operator_body_unreachable()


@mul.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs * rhs)


@mul.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "mul", builder.create_mul)


@operator
def div(x, y, signed=ConstexprValue(False)):
    operator_body_unreachable()


@div.fold
def _(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs / rhs)


@div.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)):
    return lower_binary_op(
        builder,
        x,
        y,
        "div",
        builder.create_div,
        signed=_const_bool(signed, "signed"),
    )


@operator
def floordiv(x, y, signed=ConstexprValue(False)):
    operator_body_unreachable()


@floordiv.fold
def _(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs // rhs)


@floordiv.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)):
    return lower_binary_op(
        builder,
        x,
        y,
        "floordiv",
        builder.create_floordiv,
        signed=_const_bool(signed, "signed"),
    )


@operator
def mod(x, y, signed=ConstexprValue(False)):
    operator_body_unreachable()


@mod.fold
def _(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs % rhs)


@mod.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)):
    return lower_binary_op(
        builder,
        x,
        y,
        "mod",
        builder.create_mod,
        signed=_const_bool(signed, "signed"),
    )


@operator
def pow(x, y):
    operator_body_unreachable()


@pow.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs**rhs)


@pow.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "pow", builder.create_pow)


@operator
def lshift(x, y):
    operator_body_unreachable()


@lshift.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs << rhs)


@lshift.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "lshift", builder.create_lshift)


@operator
def rshift(x, y, signed=ConstexprValue(False)):
    operator_body_unreachable()


@rshift.fold
def _(x, y, signed=ConstexprValue(False)):
    signed_value = _fold_const_bool(signed)
    if signed_value is None:
        return NO_FOLD
    if signed_value:
        return _fold_binary(x, y, lambda lhs, rhs: lhs >> rhs)
    return _fold_binary(x, y, lambda lhs, rhs: (lhs & ((1 << 64) - 1)) >> rhs)


@rshift.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)):
    return lower_binary_op(
        builder,
        x,
        y,
        "rshift",
        builder.create_rshift,
        signed=_const_bool(signed, "signed"),
    )


@operator
def bitwise_and(x, y):
    operator_body_unreachable()


@bitwise_and.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs & rhs)


@bitwise_and.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_and", builder.create_bitwise_and)


@operator
def bitwise_or(x, y):
    operator_body_unreachable()


@bitwise_or.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs | rhs)


@bitwise_or.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_or", builder.create_bitwise_or)


@operator
def bitwise_xor(x, y):
    operator_body_unreachable()


@bitwise_xor.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs ^ rhs)


@bitwise_xor.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_xor", builder.create_bitwise_xor)


@operator
def eq(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@eq.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x == y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs == rhs)


@eq.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.EQ, "eq", _const_bool(ordered, "ordered")
    )


@operator
def ne(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@ne.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x != y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs != rhs)


@ne.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.NE, "ne", _const_bool(ordered, "ordered")
    )


@operator
def lt(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@lt.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs < rhs)


@lt.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.LT, "lt", _const_bool(ordered, "ordered")
    )


@operator
def le(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@le.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs <= rhs)


@le.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.LE, "le", _const_bool(ordered, "ordered")
    )


@operator
def gt(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@gt.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs > rhs)


@gt.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.GT, "gt", _const_bool(ordered, "ordered")
    )


@operator
def ge(x, y, ordered=ConstexprValue(False)):
    operator_body_unreachable()


@ge.fold
def _(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs >= rhs)


@ge.build
def _(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.GE, "ge", _const_bool(ordered, "ordered")
    )


@operator
def pos(x):
    operator_body_unreachable()


@pos.fold
def _(x):
    if isinstance(x, ConstexprValue):
        return x
    return NO_FOLD


@pos.build
def _(builder: AlloOpBuilder, x):  # noqa: ARG001
    return x


@operator
def neg(x):
    operator_body_unreachable()


@neg.fold
def _(x):
    return _fold_unary(x, lambda operand: -operand)


@neg.build
def _(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "neg", builder.create_neg)


@operator
def invert(x):
    operator_body_unreachable()


@invert.fold
def _(x):
    return _fold_unary(x, lambda operand: ~operand)


@invert.build
def _(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "invert", builder.create_invert)


@operator
def logical_and(x, y):
    operator_body_unreachable()


@logical_and.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) and bool(rhs))


@logical_and.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "logical_and", builder.create_logical_and)


@operator
def logical_or(x, y):
    operator_body_unreachable()


@logical_or.fold
def _(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) or bool(rhs))


@logical_or.build
def _(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "logical_or", builder.create_logical_or)


@operator
def logical_not(x):
    operator_body_unreachable()


@logical_not.fold
def _(x):
    return _fold_unary(x, lambda operand: not bool(operand))


@logical_not.build
def _(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "logical_not", builder.create_logical_not)


@operator
def max(x, y, propagate_nan=ConstexprValue(False)):
    operator_body_unreachable()


@max.fold
def _(x, y, propagate_nan=ConstexprValue(False)):
    if _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.max)


@max.build
def _(
    builder: AlloOpBuilder,
    x: AlloValue,
    y: AlloValue,
    propagate_nan=ConstexprValue(False),
):
    return lower_binary_op(
        builder,
        x,
        y,
        "max",
        builder.create_max,
        extra_kwargs={"propagate_nan": _const_bool(propagate_nan, "propagate_nan")},
    )


@operator
def min(x, y, propagate_nan=ConstexprValue(False)):
    operator_body_unreachable()


@min.fold
def _(x, y, propagate_nan=ConstexprValue(False)):
    if _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.min)


@min.build
def _(
    builder: AlloOpBuilder,
    x: AlloValue,
    y: AlloValue,
    propagate_nan=ConstexprValue(False),
):
    return lower_binary_op(
        builder,
        x,
        y,
        "min",
        builder.create_min,
        extra_kwargs={"propagate_nan": _const_bool(propagate_nan, "propagate_nan")},
    )


@operator
def cast(x, dst_type):
    operator_body_unreachable()


@cast.build
def _(builder: AlloOpBuilder, x: AlloValue | ConstexprValue, dst_type: TypeBase):
    assert isinstance(dst_type, TypeBase)
    return builder.cast(x, dst_type)
