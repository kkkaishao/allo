from __future__ import annotations

import builtins

from ..compiler.builder import AlloOpBuilder, CmpPred
from ..lang.core import AlloValue, ConstexprValue, TypeBase
from ..lang.operator import NO_FOLD, operator
from .utils import cmp_op_create, lower_binary_op, lower_unary_op


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
    pass


@add.fold
def _add_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs + rhs)


@add.build
def _add_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "add", builder.create_add)


@operator
def sub(x, y):
    pass


@sub.fold
def _sub_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs - rhs)


@sub.build
def _sub_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "sub", builder.create_sub)


@operator
def mul(x, y):
    pass


@mul.fold
def _mul_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs * rhs)


@mul.build
def _mul_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "mul", builder.create_mul)


@operator
def div(x, y, signed=ConstexprValue(False)):
    pass


@div.fold
def _div_fold(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs / rhs)


@div.build
def _div_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)
):
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
    pass


@floordiv.fold
def _floordiv_fold(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs // rhs)


@floordiv.build
def _floordiv_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)
):
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
    pass


@mod.fold
def _mod_fold(x, y, signed=ConstexprValue(False)):
    if _fold_const_bool(signed) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs % rhs)


@mod.build
def _mod_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)
):
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
    pass


@pow.fold
def _pow_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs**rhs)


@pow.build
def _pow_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "pow", builder.create_pow)


@operator
def lshift(x, y):
    pass


@lshift.fold
def _lshift_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs << rhs)


@lshift.build
def _lshift_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "lshift", builder.create_lshift)


@operator
def rshift(x, y, signed=ConstexprValue(False)):
    pass


@rshift.fold
def _rshift_fold(x, y, signed=ConstexprValue(False)):
    signed_value = _fold_const_bool(signed)
    if signed_value is None:
        return NO_FOLD
    if signed_value:
        return _fold_binary(x, y, lambda lhs, rhs: lhs >> rhs)
    return _fold_binary(x, y, lambda lhs, rhs: (lhs & ((1 << 64) - 1)) >> rhs)


@rshift.build
def _rshift_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, signed=ConstexprValue(False)
):
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
    pass


@bitwise_and.fold
def _bitwise_and_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs & rhs)


@bitwise_and.build
def _bitwise_and_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_and", builder.create_bitwise_and)


@operator
def bitwise_or(x, y):
    pass


@bitwise_or.fold
def _bitwise_or_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs | rhs)


@bitwise_or.build
def _bitwise_or_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_or", builder.create_bitwise_or)


@operator
def bitwise_xor(x, y):
    pass


@bitwise_xor.fold
def _bitwise_xor_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: lhs ^ rhs)


@bitwise_xor.build
def _bitwise_xor_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "bitwise_xor", builder.create_bitwise_xor)


@operator
def eq(x, y, ordered=ConstexprValue(False)):
    pass


@eq.fold
def _eq_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x == y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs == rhs)


@eq.build
def _eq_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.EQ, "eq", _const_bool(ordered, "ordered")
    )


@operator
def ne(x, y, ordered=ConstexprValue(False)):
    pass


@ne.fold
def _ne_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    if isinstance(x, TypeBase) and isinstance(y, TypeBase):
        return ConstexprValue(x != y)
    return _fold_binary(x, y, lambda lhs, rhs: lhs != rhs)


@ne.build
def _ne_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.NE, "ne", _const_bool(ordered, "ordered")
    )


@operator
def lt(x, y, ordered=ConstexprValue(False)):
    pass


@lt.fold
def _lt_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs < rhs)


@lt.build
def _lt_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.LT, "lt", _const_bool(ordered, "ordered")
    )


@operator
def le(x, y, ordered=ConstexprValue(False)):
    pass


@le.fold
def _le_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs <= rhs)


@le.build
def _le_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.LE, "le", _const_bool(ordered, "ordered")
    )


@operator
def gt(x, y, ordered=ConstexprValue(False)):
    pass


@gt.fold
def _gt_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs > rhs)


@gt.build
def _gt_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.GT, "gt", _const_bool(ordered, "ordered")
    )


@operator
def ge(x, y, ordered=ConstexprValue(False)):
    pass


@ge.fold
def _ge_fold(x, y, ordered=ConstexprValue(False)):
    if _fold_const_bool(ordered) is None:
        return NO_FOLD
    return _fold_binary(x, y, lambda lhs, rhs: lhs >= rhs)


@ge.build
def _ge_build(
    builder: AlloOpBuilder, x: AlloValue, y: AlloValue, ordered=ConstexprValue(False)
):
    return cmp_op_create(
        builder, x, y, CmpPred.GE, "ge", _const_bool(ordered, "ordered")
    )


@operator
def pos(x):
    pass


@pos.fold
def _pos_fold(x):
    if isinstance(x, ConstexprValue):
        return x
    return NO_FOLD


@pos.build
def _pos_build(builder: AlloOpBuilder, x):  # noqa: ARG001
    return x


@operator
def neg(x):
    pass


@neg.fold
def _neg_fold(x):
    return _fold_unary(x, lambda operand: -operand)


@neg.build
def _neg_build(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "neg", builder.create_neg)


@operator
def invert(x):
    pass


@invert.fold
def _invert_fold(x):
    return _fold_unary(x, lambda operand: ~operand)


@invert.build
def _invert_build(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "invert", builder.create_invert)


@operator
def logical_and(x, y):
    pass


@logical_and.fold
def _logical_and_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) and bool(rhs))


@logical_and.build
def _logical_and_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "logical_and", builder.create_logical_and)


@operator
def logical_or(x, y):
    pass


@logical_or.fold
def _logical_or_fold(x, y):
    return _fold_binary(x, y, lambda lhs, rhs: bool(lhs) or bool(rhs))


@logical_or.build
def _logical_or_build(builder: AlloOpBuilder, x: AlloValue, y: AlloValue):
    return lower_binary_op(builder, x, y, "logical_or", builder.create_logical_or)


@operator
def logical_not(x):
    pass


@logical_not.fold
def _logical_not_fold(x):
    return _fold_unary(x, lambda operand: not bool(operand))


@logical_not.build
def _logical_not_build(builder: AlloOpBuilder, x: AlloValue):
    return lower_unary_op(builder, x, "logical_not", builder.create_logical_not)


@operator
def max(x, y, propagate_nan=ConstexprValue(False)):
    pass


@max.fold
def _max_fold(x, y, propagate_nan=ConstexprValue(False)):
    if _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.max)


@max.build
def _max_build(
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
    pass


@min.fold
def _min_fold(x, y, propagate_nan=ConstexprValue(False)):
    if _fold_const_bool(propagate_nan) is None:
        return NO_FOLD
    return _fold_binary(x, y, builtins.min)


@min.build
def _min_build(
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
    pass


@cast.build
def _cast_build(
    builder: AlloOpBuilder, x: AlloValue | ConstexprValue, dst_type: TypeBase
):
    assert isinstance(dst_type, TypeBase)
    return builder.cast(x, dst_type)
