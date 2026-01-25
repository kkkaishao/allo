# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Tuple, Union

from .. import core
from .. import settings
from ..compiler.builder import AlloOpBuilder
from ..core.library import operation
from ..core.types import (
    constexpr,
    slice,
    tensor,
    tuple,
    _is_array_of_scalar_tensor,
    _is_scalar_tensor,
)
from .ops_common import _is_const_positive_int, _prepare_tuple_indices, _to_index_tensor


@operation
def load(buffer: tensor, slices: Union[slice, Tuple]) -> tensor:
    pass


@load.lower
def _lower_load(builder: AlloOpBuilder, buffer: tensor, slices: Union[slice, tuple]):
    if isinstance(slices, tuple):
        indices = _prepare_tuple_indices(builder, slices)
        if isinstance(buffer.type, core.Channel):
            if len(buffer.shape) != len(indices):
                builder.compile_error(
                    "Number of indices does not match channel group dimensions."
                )
            buffer.type.last_indices = indices
            return buffer
        if _is_array_of_scalar_tensor(buffer):
            if len(buffer.shape) != len(indices):
                builder.compile_error("Number of indices does not match buffer rank.")
            return builder.create_load(buffer, indices, use_tensor=settings.USE_TENSOR)
        if _is_scalar_tensor(buffer):
            if len(slices.values) != 1:
                builder.compile_error(
                    "Bit extract with tuple index requires exactly one index."
                )
            idx = _to_index_tensor(builder, slices.values[0])
            return builder.create_bit_extract_op(buffer, idx, 1)

    if isinstance(slices, core.slice) and _is_scalar_tensor(buffer):
        if slices.stop is not None:
            builder.compile_error("Stop index is not supported for bit extract.")
        start_idx = _to_index_tensor(builder, slices.start)
        if _is_const_positive_int(slices.step):
            width = slices.step.value
        else:
            builder.compile_error(
                "Only constant positive integer step is supported for bit extract."
            )
        return builder.create_bit_extract_op(buffer, start_idx, width)

    builder.compile_error(
        f"Unsupported combination of buffer type {buffer.type} and slice type {type(slices)} in load operation."
    )


@operation
def store(buffer: tensor, slices: Union[slice, Tuple], value) -> tensor | None:
    pass


@store.lower
def _lower_store(
    builder: AlloOpBuilder, buffer: tensor, slices: Union[slice, tuple], value
):
    if isinstance(buffer.type, core.Channel):
        builder.compile_error("Store operation is not supported on channels.")

    if isinstance(slices, tuple):
        if _is_array_of_scalar_tensor(buffer):
            indices = _prepare_tuple_indices(builder, slices)
            if isinstance(value, constexpr):
                value = builder.make_scalar(value.value, buffer.dtype)
            else:
                value = builder.scalar_cast(value, buffer.dtype)
            return builder.create_store(
                buffer, value, indices, use_tensor=settings.USE_TENSOR
            )

        if _is_scalar_tensor(buffer):
            if len(slices.values) != 1:
                builder.compile_error(
                    "Bit insert with tuple index requires exactly one index."
                )
            idx = _to_index_tensor(builder, slices.values[0])
            if isinstance(value, constexpr):
                value = builder.make_scalar(value.value, core.uint1)
            else:
                value = builder.scalar_cast(value, core.uint1)
            return builder.create_bit_insert_op(buffer, value, idx, 1)

    if isinstance(slices, core.slice) and _is_scalar_tensor(buffer):
        if slices.stop is not None:
            builder.compile_error("Stop index is not supported for bit insert.")
        start_idx = _to_index_tensor(builder, slices.start)
        if _is_const_positive_int(slices.step):
            width = slices.step.value
        else:
            builder.compile_error(
                "Only constant positive integer step is supported for bit insert."
            )
        if isinstance(value, constexpr):
            value = builder.make_scalar(value.value, core.APInt(width, signed=False))
        else:
            value = builder.scalar_cast(value, core.APInt(width, signed=False))
        return builder.create_bit_insert_op(buffer, value, start_idx, width)

    builder.compile_error(
        f"Unsupported combination of buffer type {buffer.type} and slice type {type(slices)} in store operation."
    )
