# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

from __future__ import annotations

from ..compiler.builder import AlloOpBuilder
from ..lang.core import AlloValue, ConstexprValue, ShapedType
from ..lang.operator import operator
from .utils import (
    _linalg_generic_result_types,
    linalg_op_result,
    operator_body_unreachable,
    resolve_linalg_output,
    shaped_type_like,
)
from .._mlir.dialects import linalg as linalg_d


def _expect_shaped(builder: AlloOpBuilder, value: AlloValue, op_name: str):
    assert isinstance(value, AlloValue)
    if not isinstance(value.type, ShapedType):
        return builder.compile_error(f"Operator '{op_name}' expects shaped operands")
    return value


def _promote_pair(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue):
    result_dtype = builder.get_promoted_dtype_nary("mul", [lhs.dtype, rhs.dtype])
    return (
        builder.cast_to_dtype(lhs, result_dtype),
        builder.cast_to_dtype(rhs, result_dtype),
        result_dtype,
    )


@operator
def matmul(lhs, rhs, acc=ConstexprValue(None)):
    operator_body_unreachable()


@matmul.build
def _(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue, acc=ConstexprValue(None)):
    lhs = _expect_shaped(builder, lhs, "matmul")
    rhs = _expect_shaped(builder, rhs, "matmul")
    assert type(lhs.type) is type(rhs.type)

    if lhs.type.rank != 2 or rhs.type.rank != 2:
        return builder.compile_error("Operator 'matmul' expects rank-2 operands")
    if lhs.type.shape[1] != rhs.type.shape[0]:
        return builder.compile_error(
            "Operator 'matmul' has incompatible contraction dimensions"
        )

    lhs, rhs, result_dtype = _promote_pair(builder, lhs, rhs)
    result_type = shaped_type_like(
        lhs, (lhs.type.shape[0], rhs.type.shape[1]), result_dtype
    )
    output = resolve_linalg_output(builder, result_type, acc, "matmul")
    ip, loc = builder.get_insertion_point_and_loc()
    op = linalg_d.MatmulOp(
        _linalg_generic_result_types(builder, output.type),
        [lhs.handle, rhs.handle],
        [output.handle],
        ip=ip,
        loc=loc,
    )
    linalg_d.fill_builtin_region(op.operation)
    return linalg_op_result(op, output)


@operator
def dot(lhs, rhs, acc=ConstexprValue(None)):
    operator_body_unreachable()


@dot.build
def _(builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue, acc=ConstexprValue(None)):
    lhs = _expect_shaped(builder, lhs, "dot")
    rhs = _expect_shaped(builder, rhs, "dot")
    assert type(lhs.type) is type(rhs.type)

    if lhs.type.rank != 1 or rhs.type.rank != 1:
        return builder.compile_error("Operator 'dot' expects rank-1 operands")
    if lhs.type.shape[0] != rhs.type.shape[0]:
        return builder.compile_error("Operator 'dot' expects equal-length operands")

    lhs, rhs, result_dtype = _promote_pair(builder, lhs, rhs)
    result_type = shaped_type_like(lhs, (), result_dtype)
    output = resolve_linalg_output(builder, result_type, acc, "dot")
    ip, loc = builder.get_insertion_point_and_loc()
    op = linalg_d.DotOp(
        _linalg_generic_result_types(builder, output.type),
        [lhs.handle, rhs.handle],
        [output.handle],
        ip=ip,
        loc=loc,
    )
    linalg_d.fill_builtin_region(op.operation)
    return linalg_op_result(op, output)
