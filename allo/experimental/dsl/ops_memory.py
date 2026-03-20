# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..core.library import operator
from ..core.types import DType, Proxy, ShapedType, uint1
from ..compiler.builder import AlloOpBuilder
from .ops_common import prepare_tuple_indices


@operator
def load(lhs: Proxy, slices: slice | tuple) -> Proxy:
    pass


@load.lower
def _lower_load(builder: AlloOpBuilder, lhs: Proxy, slices: slice | tuple):
    if isinstance(slices, tuple):
        indices = prepare_tuple_indices(builder, slices)
        if isinstance(lhs.type, ShapedType):
            if len(indices) != lhs.type.rank:
                builder.compile_error(
                    f"Load with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {lhs.type.rank}."
                )
            return builder.create_load(lhs, indices)
        if isinstance(lhs.type, DType):
            if len(indices) != 1:
                builder.compile_error(
                    f"Bit extraction with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            return builder.create_bit_extract(lhs, indices[0])

    elif isinstance(slices, slice):
        raise NotImplementedError("Slice indices are not supported yet.")

    raise builder.compile_error(
        f"Unsupported index type: {type(slices)}. Indices must be a slice or a tuple of indices."
    )


@operator
def store(dst: Proxy, slices: slice | tuple, val: Proxy) -> None:
    pass


@store.lower
def _lower_store(builder: AlloOpBuilder, dst: Proxy, slices: slice | tuple, val: Proxy):
    if isinstance(slices, tuple):
        indices = prepare_tuple_indices(builder, slices)
        if isinstance(dst.type, ShapedType):
            if len(indices) != dst.type.rank:
                builder.compile_error(
                    f"Store with tuple indices must have the same number of indices as the rank of the array, got {len(indices)} indices for an array of rank {dst.type.rank}."
                )
            val = builder.make_or_cast_scalar(val, dst.dtype)
            return builder.create_store(val, dst, indices)
        if isinstance(dst.type, DType):
            if len(indices) != 1:
                builder.compile_error(
                    f"Bit insertion with tuple indices must have exactly one index for scalar types, got {len(indices)}."
                )
            val = builder.make_or_cast_scalar(val, uint1)
            return builder.create_bit_insert(val, dst, indices[0])

    elif isinstance(slices, slice):
        raise NotImplementedError("Slice indices are not supported yet.")

    raise builder.compile_error(
        f"Unsupported index type: {type(slices)}. Indices must be a slice or a tuple of indices."
    )
