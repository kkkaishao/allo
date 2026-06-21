# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence, Callable
from dataclasses import dataclass
from typing import NoReturn, cast
from ..compiler.builder import AlloOpBuilder
from ..lang.core import (
    AlloValue,
    BufferType,
    ConstexprValue,
    DType,
    ShapedType,
    TensorType,
)
from .._mlir import ir
from .._mlir.dialects import linalg


@dataclass
class BitSlice:
    """A lowered ``x[lo:hi]`` bit slice.

    ``lo``/``hi`` are the (possibly dynamic) bound values; ``width`` is the
    statically inferred bit count ``hi - lo``, or ``None`` when it is not a
    compile-time constant (e.g. the offset is dynamic but the width is not).
    """

    lo: object
    hi: object
    width: int | None


def shaped_type_with_dtype(src_type: ShapedType, dtype: DType) -> ShapedType:
    if isinstance(src_type, TensorType):
        return TensorType(src_type.shape, dtype)
    assert isinstance(src_type, BufferType)
    return BufferType(src_type.shape, dtype)


def shaped_type_like(value: AlloValue, shape, dtype: DType) -> ShapedType:
    if isinstance(value.type, TensorType):
        return TensorType(shape, dtype)
    assert isinstance(value.type, BufferType)
    return BufferType(shape, dtype)


def make_linalg_output(builder: AlloOpBuilder, result_type: ShapedType) -> AlloValue:
    return builder.make_buffer(result_type)


def is_default_acc(acc) -> bool:
    return isinstance(acc, ConstexprValue) and acc.value is None


def resolve_linalg_output(
    builder: AlloOpBuilder, result_type: ShapedType, acc, op_name: str
) -> AlloValue:
    if is_default_acc(acc):
        if isinstance(result_type, BufferType):
            return builder.compile_error(
                f"Operator '{op_name}' requires acc for memref output"
            )
        return make_linalg_output(builder, result_type)

    if not isinstance(acc, AlloValue) or not isinstance(acc.type, ShapedType):
        return builder.compile_error(f"Operator '{op_name}' acc must be a shaped value")
    if type(acc.type) is not type(result_type):
        return builder.compile_error(
            f"Operator '{op_name}' acc type must match output storage kind"
        )
    if tuple(acc.type.shape) != tuple(result_type.shape):
        return builder.compile_error(
            f"Operator '{op_name}' acc shape {acc.type.shape} does not match "
            f"result shape {result_type.shape}"
        )
    if acc.dtype != result_type.dtype:
        return builder.compile_error(
            f"Operator '{op_name}' acc dtype {acc.dtype} does not match "
            f"result dtype {result_type.dtype}"
        )
    return acc


def linalg_op_result(op, output: AlloValue) -> AlloValue:
    assert isinstance(output.type, ShapedType)
    if isinstance(output.type, TensorType):
        assert len(op.results) == 1
        return AlloValue(op.results[0], output.type)
    assert isinstance(output.type, BufferType)
    assert len(op.results) == 0
    return output


def emit_linalg_named_unary(
    builder: AlloOpBuilder,
    operand: AlloValue,
    output: AlloValue,
    op_cls,
) -> AlloValue:
    assert isinstance(output.type, ShapedType)
    ip, loc = builder.get_insertion_point_and_loc()
    op = op_cls(
        _linalg_generic_result_types(builder, output.type),
        [operand.handle],
        [output.handle],
        ip=ip,
        loc=loc,
    )
    # Upstream named-linalg OpView constructors leave the body region empty;
    # populate it from the op's registered region builder.
    linalg.fill_builtin_region(op.operation)
    return linalg_op_result(op, output)


def emit_linalg_named_binary(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    output: AlloValue,
    op_cls,
) -> AlloValue:
    assert isinstance(output.type, ShapedType)
    ip, loc = builder.get_insertion_point_and_loc()
    op = op_cls(
        _linalg_generic_result_types(builder, output.type),
        [lhs.handle, rhs.handle],
        [output.handle],
        ip=ip,
        loc=loc,
    )
    # Upstream named-linalg OpView constructors leave the body region empty;
    # populate it from the op's registered region builder.
    linalg.fill_builtin_region(op.operation)
    return linalg_op_result(op, output)


def _linalg_generic_result_types(builder: AlloOpBuilder, result_type: ShapedType):
    if isinstance(result_type, TensorType):
        return [result_type.materialize(builder.context)]
    assert isinstance(result_type, BufferType)
    return []


def _yield_value(value):
    return value.handle if isinstance(value, AlloValue) else value


def _identity_maps(builder: AlloOpBuilder, rank: int, count: int):
    ident = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank, builder.context))
    return ir.ArrayAttr.get([ident] * count)


def _iterator_types(builder: AlloOpBuilder, rank: int):
    par = ir.Attribute.parse("#linalg.iterator_type<parallel>", builder.context)
    return ir.ArrayAttr.get([par] * rank)


def emit_linalg_generic_unary(
    builder: AlloOpBuilder,
    operand: AlloValue,
    output: AlloValue,
    build_fn: Callable[[AlloValue], object],
) -> AlloValue:
    assert isinstance(operand.type, ShapedType)
    assert isinstance(output.type, ShapedType)
    result_type = output.type
    maps = _identity_maps(builder, result_type.rank, 2)
    iters = _iterator_types(builder, result_type.rank)
    ip, loc = builder.get_insertion_point_and_loc()
    op = linalg.GenericOp(
        _linalg_generic_result_types(builder, result_type),
        [operand.handle],
        [output.handle],
        maps,
        iters,
        ip=ip,
        loc=loc,
    )
    body = op.regions[0].blocks.append(
        operand.dtype.materialize(builder.context),
        output.dtype.materialize(builder.context),
    )
    region_arg = AlloValue(body.arguments[0], operand.dtype)
    with builder.at_block_end(body):
        linalg.YieldOp(
            [_yield_value(build_fn(region_arg))], ip=builder._ip, loc=builder._loc
        )
    return linalg_op_result(op, output)


def emit_linalg_generic_binary(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    output: AlloValue,
    build_fn: Callable[[AlloValue, AlloValue], object],
) -> AlloValue:
    assert isinstance(lhs.type, ShapedType) and isinstance(rhs.type, ShapedType)
    assert tuple(lhs.type.shape) == tuple(rhs.type.shape)
    assert isinstance(output.type, ShapedType)
    result_type = output.type
    maps = _identity_maps(builder, result_type.rank, 3)
    iters = _iterator_types(builder, result_type.rank)
    ip, loc = builder.get_insertion_point_and_loc()
    op = linalg.GenericOp(
        _linalg_generic_result_types(builder, result_type),
        [lhs.handle, rhs.handle],
        [output.handle],
        maps,
        iters,
        ip=ip,
        loc=loc,
    )
    body = op.regions[0].blocks.append(
        lhs.dtype.materialize(builder.context),
        rhs.dtype.materialize(builder.context),
        output.dtype.materialize(builder.context),
    )
    lhs_arg = AlloValue(body.arguments[0], lhs.dtype)
    rhs_arg = AlloValue(body.arguments[1], rhs.dtype)
    with builder.at_block_end(body):
        linalg.YieldOp(
            [_yield_value(build_fn(lhs_arg, rhs_arg))], ip=builder._ip, loc=builder._loc
        )
    return linalg_op_result(op, output)


def _broadcast_failure(shape, lhs_shape, rhs_shape) -> bool:
    return not shape and (lhs_shape or rhs_shape)


def _infer_elementwise_shape(
    builder: AlloOpBuilder, values: Sequence[AlloValue], op_name: str
):
    assert values
    shape = list(cast(ShapedType, values[0].type).shape)
    for value in values[1:]:
        next_shape = cast(ShapedType, value.type).shape
        prev_shape = shape
        shape, _, _ = builder.infer_broadcast_shape(prev_shape, next_shape)
        if _broadcast_failure(shape, prev_shape, next_shape):
            return builder.compile_error(
                f"Operator '{op_name}' operands with shapes {prev_shape} and "
                f"{next_shape} are not broadcastable"
            )
    return shape


def _broadcast_operand_to_shape(
    builder: AlloOpBuilder,
    operand: AlloValue,
    result_type: ShapedType,
    op_name: str,
) -> AlloValue:
    if isinstance(operand.type, ShapedType):
        if type(operand.type) is not type(result_type):
            return builder.compile_error(
                f"Operator '{op_name}' operands must use the same storage kind"
            )
        shape, indices, _ = builder.infer_broadcast_shape(
            operand.type.shape, result_type.shape
        )
        if _broadcast_failure(shape, operand.type.shape, result_type.shape) or tuple(
            shape
        ) != tuple(result_type.shape):
            return builder.compile_error(
                f"Operator '{op_name}' operand shape {operand.type.shape} is not "
                f"broadcastable to result shape {result_type.shape}"
            )
        if indices:
            return _broadcast_shaped_operand(builder, operand, shape, indices)
        return operand

    dst_type = shaped_type_with_dtype(result_type, operand.dtype)
    return builder.cast(operand, dst_type)


def prepare_linalg_elementwise_operands(
    builder: AlloOpBuilder,
    operands: Sequence[AlloValue],
    result_dtype: DType,
    acc,
    op_name: str,
) -> tuple[list[AlloValue], AlloValue]:
    acc_value = None if is_default_acc(acc) else acc
    shaped_operands = [
        operand for operand in operands if isinstance(operand.type, ShapedType)
    ]
    if not shaped_operands:
        assert not is_default_acc(acc)
        return builder.compile_error(
            f"Operator '{op_name}' acc requires at least one shaped operand"
        )
    if acc_value is not None:
        if not isinstance(acc_value, AlloValue) or not isinstance(
            acc_value.type, ShapedType
        ):
            return builder.compile_error(
                f"Operator '{op_name}' acc must be a shaped value"
            )
        result_type = shaped_type_with_dtype(acc_value.type, result_dtype)
    else:
        kind = type(shaped_operands[0].type)
        for operand in shaped_operands[1:]:
            if type(operand.type) is not kind:
                return builder.compile_error(
                    f"Operator '{op_name}' operands must use the same storage kind"
                )
        shape = _infer_elementwise_shape(builder, shaped_operands, op_name)
        result_type = shaped_type_like(shaped_operands[0], shape, result_dtype)

    output = resolve_linalg_output(builder, result_type, acc, op_name)
    prepared = [
        _broadcast_operand_to_shape(builder, operand, output.type, op_name)
        for operand in operands
    ]
    return prepared, output


def emit_linalg_unary(
    builder: AlloOpBuilder,
    operand: AlloValue,
    result_dtype: DType,
    build_fn: Callable[[AlloValue], object],
    *,
    named_op_cls=None,
    acc=ConstexprValue(None),
    op_name: str = "operator",
) -> AlloValue:
    operands, output = prepare_linalg_elementwise_operands(
        builder, [operand], result_dtype, acc, op_name
    )
    operand = operands[0]
    if named_op_cls is not None:
        return emit_linalg_named_unary(builder, operand, output, named_op_cls)
    return emit_linalg_generic_unary(builder, operand, output, build_fn)


def emit_linalg_binary(
    builder: AlloOpBuilder,
    lhs: AlloValue,
    rhs: AlloValue,
    result_dtype: DType,
    build_fn: Callable[[AlloValue, AlloValue], object],
    *,
    named_op_cls=None,
    acc=ConstexprValue(None),
    op_name: str = "operator",
) -> AlloValue:
    operands, output = prepare_linalg_elementwise_operands(
        builder, [lhs, rhs], result_dtype, acc, op_name
    )
    lhs, rhs = operands
    if named_op_cls is not None:
        return emit_linalg_named_binary(builder, lhs, rhs, output, named_op_cls)
    return emit_linalg_generic_binary(builder, lhs, rhs, output, build_fn)


def _broadcast_shaped_operand(
    builder: AlloOpBuilder, value: AlloValue, shape, indices
) -> AlloValue:
    result_type = shaped_type_like(value, shape, value.dtype)
    output = make_linalg_output(builder, result_type)
    ip, loc = builder.get_insertion_point_and_loc()
    op = linalg.BroadcastOp(
        _linalg_generic_result_types(builder, result_type),
        value.handle,
        output.handle,
        indices,
        ip=ip,
        loc=loc,
    )
    linalg.fill_builtin_region(op.operation)
    return linalg_op_result(op, output)


def broadcast_pair_by_shape(
    builder: AlloOpBuilder, lhs: AlloValue, rhs: AlloValue
) -> tuple[AlloValue, AlloValue]:
    lhs_is_shaped = isinstance(lhs.type, ShapedType)
    rhs_is_shaped = isinstance(rhs.type, ShapedType)

    if lhs_is_shaped and rhs_is_shaped:
        assert type(lhs.type) is type(rhs.type)
        lhs_shape = cast(ShapedType, lhs.type).shape
        rhs_shape = cast(ShapedType, rhs.type).shape
        shape, indices_lhs, indices_rhs = builder.infer_broadcast_shape(
            lhs_shape, rhs_shape
        )
        if _broadcast_failure(shape, lhs_shape, rhs_shape):
            return builder.compile_error(
                f"Shapes {lhs_shape} and {rhs_shape} are not broadcastable"
            )
        if indices_lhs:
            lhs = _broadcast_shaped_operand(builder, lhs, shape, indices_lhs)
        if indices_rhs:
            rhs = _broadcast_shaped_operand(builder, rhs, shape, indices_rhs)
        return lhs, rhs

    if not lhs_is_shaped and not rhs_is_shaped:
        return lhs, rhs

    if lhs_is_shaped:
        dst_type = shaped_type_like(lhs, cast(ShapedType, lhs.type).shape, rhs.dtype)
        return lhs, builder.cast(rhs, dst_type)

    dst_type = shaped_type_like(rhs, cast(ShapedType, rhs.type).shape, lhs.dtype)
    return builder.cast(lhs, dst_type), rhs


def operator_body_unreachable() -> NoReturn:
    raise RuntimeError("Allo operator declarations are not directly executed.")
