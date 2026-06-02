# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..lang.operator import operator
from ..lang.core import (
    DType,
    ShapedType,
    StreamType,
    u1,
    ConstexprValue,
    AlloValue,
)
from ..compiler.builder import AlloOpBuilder
from .utils import operator_body_unreachable


def _normalize_stream_indices(
    builder: AlloOpBuilder, stream_type: StreamType, slices, context: str
):
    assert isinstance(stream_type, StreamType)
    if not isinstance(slices, tuple):
        return builder.compile_error(
            f"{context} indices must be a tuple of scalar index expressions."
        )
    if len(slices) != stream_type.rank:
        return builder.compile_error(
            f"{context} expects {stream_type.rank} indices, got {len(slices)}."
        )
    for idx, dim in zip(slices, stream_type.shape):
        if not isinstance(idx, ConstexprValue):
            continue
        if type(idx.value) is not int:
            return builder.compile_error(
                f"{context} constexpr indices must be integers."
            )
        if idx.value < 0 or idx.value >= dim:
            return builder.compile_error(
                f"{context} index {idx.value} is out of bounds for dimension size {dim}."
            )
    return builder.normalize_indices(
        slices,
        expected_len=stream_type.rank,
        context=context,
    )


def _load_stream_value(builder: AlloOpBuilder, stream: AlloValue, slices):
    assert isinstance(stream.type, StreamType)
    if stream.is_indexed:
        return builder.compile_error(
            "Cannot index a specific stream, Use get() or put(value) on the specific stream."
        )
    indices = _normalize_stream_indices(builder, stream.type, slices, "Stream")
    ref = AlloValue(stream.handle, stream.type)
    ref.indices = tuple(indices)
    return ref


@operator
def load(lhs, slices):
    operator_body_unreachable()


@load.build
def _(builder: AlloOpBuilder, lhs, slices: slice | tuple):
    if isinstance(lhs, AlloValue) and isinstance(lhs.type, StreamType):
        return _load_stream_value(builder, lhs, slices)

    if isinstance(slices, tuple):
        indices = builder.normalize_indices(slices)
        if isinstance(lhs.type, ShapedType):
            if len(indices) != lhs.type.rank:
                return builder.compile_error(
                    f"Load with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {lhs.type.rank}."
                )
            return builder.create_load(lhs, indices)
        if isinstance(lhs.type, DType):
            if len(indices) != 1:
                return builder.compile_error(
                    f"Bit extraction with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            return builder.create_bit_extract(lhs, indices[0])

    elif isinstance(slices, slice):
        raise NotImplementedError("Slice indices are not supported yet.")

    return builder.compile_error(
        f"Unsupported load operation: lhs of type {lhs.type} with indices of type {type(slices)}"
    )


@operator
def store(dst, slices, value):
    operator_body_unreachable()


@store.build
def _(builder: AlloOpBuilder, dst, slices: slice | tuple, value):
    if isinstance(dst, AlloValue) and isinstance(dst.type, StreamType):
        return builder.compile_error(
            "Cannot assign to a stream. Use put(value) on the stream reference."
        )

    if isinstance(slices, tuple):
        indices = builder.normalize_indices(slices)
        if isinstance(dst.type, ShapedType):
            if len(indices) != dst.type.rank:
                builder.compile_error(
                    f"Store with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {dst.type.rank}."
                )
            val = builder.cast(value, dst.dtype)
            return builder.create_store(val, dst, indices)
        if isinstance(dst.type, DType):
            if len(indices) != 1:
                return builder.compile_error(
                    f"Bit insertion with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            val = builder.cast(value, u1)
            return builder.create_bit_insert(val, dst, indices[0])

    elif isinstance(slices, slice):
        raise NotImplementedError("Slice indices are not supported yet.")

    raise builder.compile_error(
        f"Unsupported store operation: dst of type {dst.type} with indices of type {type(slices)}"
    )
