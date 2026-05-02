# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..lang.operator import operator
from ..lang.core import DType, AlloValue, ShapedType, u1
from ..compiler.builder import AlloOpBuilder


@operator
def load(lhs, slices):
    pass


@load.build
def _load_build(builder: AlloOpBuilder, lhs: AlloValue, slices: slice | tuple):
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
    pass


@store.build
def _store_build(
    builder: AlloOpBuilder, dst: AlloValue, slices: slice | tuple, value: AlloValue
):
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
