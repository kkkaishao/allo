# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

from ..lang.operator import operator
from ..lang.core import (
    APInt,
    DType,
    ShapedType,
    StreamType,
    u1,
    ConstexprValue,
    AlloValue,
)
from ..compiler.builder import AlloOpBuilder
from .utils import operator_body_unreachable, BitSlice


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


def _bit_slice(builder: AlloOpBuilder, value: AlloValue, slc: BitSlice):
    # The result width ``hi - lo`` must be statically known (the offset may be
    # dynamic); the codegen infers it affinely and leaves ``width`` as ``None``
    # when it is not a compile-time constant.
    if not isinstance(value, AlloValue) or not isinstance(value.dtype, APInt):
        return builder.compile_error(
            "Bit slicing is only supported on signless integer scalars."
        )
    if slc.lo is None or slc.hi is None:
        return builder.compile_error(
            "Bit slice requires explicit lower and upper bounds, e.g. 'x[lo:hi]'."
        )
    if slc.width is None:
        return builder.compile_error(
            "Bit slice width 'hi - lo' must be a compile-time constant; "
            "only the offset may be dynamic."
        )
    if slc.width <= 0:
        return builder.compile_error(
            "Bit slice upper bound must be greater than the lower bound."
        )
    return slc.lo, slc.hi, APInt(slc.width, signed=False)


@operator
def load(lhs, slices):
    operator_body_unreachable()


@load.build
def _(builder: AlloOpBuilder, lhs, slices: BitSlice | tuple):
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

    elif isinstance(slices, BitSlice):
        lo, hi, result_dtype = _bit_slice(builder, lhs, slices)
        return builder.create_bit_get_slice(lhs, lo, hi, result_dtype)

    return builder.compile_error(
        f"Unsupported load operation: lhs of type {lhs.type} with indices of type {type(slices)}"
    )


@operator
def store(dst, slices, value):
    operator_body_unreachable()


@store.build
def _(builder: AlloOpBuilder, dst, slices: BitSlice | tuple, value):
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

    elif isinstance(slices, BitSlice):
        lo, hi, slice_dtype = _bit_slice(builder, dst, slices)
        val = builder.cast(value, slice_dtype)
        return builder.create_bit_set_slice(dst, lo, hi, val)

    raise builder.compile_error(
        f"Unsupported store operation: dst of type {dst.type} with indices of type {type(slices)}"
    )
