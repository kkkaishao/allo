# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Sequence

from .. import core
from ..compiler.builder import AlloOpBuilder
from ..core.library import NO_FOLD
from ..core.types import constexpr, scalar_type, tensor, tuple


def _is_const_int_tuple(t) -> bool:
    if not isinstance(t, core.tuple) or not t.is_constexpr:
        return False
    for val in t.values:
        if not isinstance(val, constexpr) or not isinstance(val.value, int):
            return False
    return True


def _is_const_positive_int(t, allow_zero=False) -> bool:
    return (
        isinstance(t, constexpr)
        and isinstance(t.value, int)
        and (t.value > 0 if not allow_zero else t.value >= 0)
    )


def _is_const_of_type(t, expected_type) -> bool:
    return isinstance(t, constexpr) and isinstance(t.value, expected_type)


class _ScalarOrArray:
    """Type-hint placeholder."""


def _to_index_tensor(builder: AlloOpBuilder, value) -> tensor:
    if isinstance(value, constexpr):
        return builder.create_const_index(value.value)
    if isinstance(value, tensor):
        if not isinstance(value.type, scalar_type):
            builder.compile_error(
                f"Every index must be scalar, got value type {value.type}."
            )
        return builder.scalar_cast(value, core.index)
    builder.compile_error(
        f"Invalid index type: {type(value)}. Expected constexpr or scalar tensor."
    )


def _prepare_tuple_indices(builder: AlloOpBuilder, slices: tuple) -> List[tensor]:
    indices: List[tensor] = []
    for val in slices.values:
        if isinstance(val, tuple):
            builder.compile_error("Nested tuples are not supported in slices.")
        indices.append(_to_index_tensor(builder, val))
    return indices


def _binary_op_checks(lhs, rhs, op_name="") -> str:
    lhs_ok = isinstance(lhs.type, (core.scalar_type, core.types.constexpr_type)) or (
        isinstance(lhs.type, core.Array) and isinstance(lhs.dtype, core.scalar_type)
    )
    rhs_ok = isinstance(rhs.type, (core.scalar_type, core.types.constexpr_type)) or (
        isinstance(rhs.type, core.Array) and isinstance(rhs.dtype, core.scalar_type)
    )
    if not (lhs_ok and rhs_ok):
        return f"{op_name} requires operands to be scalars or arrays of scalars, got {lhs.type} and {rhs.type}."
    return ""


def _materialize_constexpr_pair(
    builder: AlloOpBuilder, lhs: tensor | constexpr, rhs: tensor | constexpr
):
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return lhs, rhs
    if isinstance(lhs, constexpr):
        assert isinstance(rhs, tensor)
        lhs = builder.make_scalar(lhs.value, rhs.dtype)
    if isinstance(rhs, constexpr):
        assert isinstance(lhs, tensor)
        rhs = builder.make_scalar(rhs.value, lhs.dtype)
    assert isinstance(lhs, tensor) and isinstance(rhs, tensor)
    return lhs, rhs


def _prepare_binary_operands(
    builder: AlloOpBuilder,
    lhs: tensor | constexpr,
    rhs: tensor | constexpr,
    op_name: str,
):
    lhs, rhs = _materialize_constexpr_pair(builder, lhs, rhs)
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return lhs, rhs

    assert isinstance(lhs, tensor) and isinstance(rhs, tensor)
    dst_ty = builder.get_promoted_dtype(lhs.dtype, rhs.dtype, op_name)
    operands = []
    for operand in [lhs, rhs]:
        if isinstance(operand.type, core.scalar_type):
            operands.append(builder.scalar_cast(operand, dst_ty))
        else:
            operands.append(builder.tensor_cast(operand, dst_ty))
    lhs, rhs = builder.create_broadcast(operands[0], operands[1])
    return lhs, rhs


def _binary_op_create(
    builder: AlloOpBuilder,
    lhs: tensor | constexpr,
    rhs: tensor | constexpr,
    op_name: str,
    create_fn,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return NO_FOLD
    return create_fn(lhs, rhs)


def _binary_op_create_signed(
    builder: AlloOpBuilder,
    lhs: tensor | constexpr,
    rhs: tensor | constexpr,
    op_name: str,
    create_fn,
    signed=True,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    if isinstance(lhs, constexpr) and isinstance(rhs, constexpr):
        return NO_FOLD
    signed = signed or not (lhs.dtype.is_uint() and rhs.dtype.is_uint())
    return create_fn(lhs, rhs, signed)


def _cmp_op_create(
    builder: AlloOpBuilder,
    lhs: tensor | constexpr,
    rhs: tensor | constexpr,
    pred,
    op_name="",
    ordered=True,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    assert isinstance(lhs, tensor) and isinstance(rhs, tensor)
    if lhs.dtype.is_float():
        return builder.create_cmpf(lhs, rhs, pred, ordered)
    return builder.create_cmpi(lhs, rhs, pred, lhs.dtype.is_int())


def _unary_op_checks(operand, op_name="") -> str:
    ok = isinstance(operand.type, (core.scalar_type, core.types.constexpr_type)) or (
        isinstance(operand.type, core.Array)
        and isinstance(operand.dtype, core.scalar_type)
    )
    if not ok:
        return (
            f"{op_name} requires the operand to be a scalar or an array of scalars, "
            f"got {operand.type}."
        )
    return ""


def _unary_op_create(
    builder: AlloOpBuilder, operand: tensor, op_name: str, create_fn
) -> tensor:
    dst_ty = builder.get_promoted_dtype(operand.dtype, None, op_name)
    operand = builder.cast(operand, dst_ty)
    return create_fn(operand)
