# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..compiler.builder import AlloOpBuilder
from ..core.library import operator, as_member_function
from ..core.types import Constexpr, Proxy, index, Stream, StreamRef
from .._C import allo


@operator
def get_wid(axis: Constexpr) -> Proxy:
    """Get the worker ID along the given axis."""
    pass


@get_wid.validate
def _validate_get_wid(axis: Constexpr) -> str:
    if not isinstance(axis, Constexpr):
        return f"Expected axis to be a Constexpr, got {type(axis).__name__}."
    if axis.value < 0:
        return f"Axis must be non-negative, got {axis.value}."
    return ""


@get_wid.lower
def _lower_get_wid(builder: AlloOpBuilder, axis: Constexpr):
    wid = allo.GetWorkerIdOp(builder, axis.value)
    return Proxy(wid, index)


@operator
def get_num_workers(axis: Constexpr) -> Proxy:
    """Get the number of workers along the given axis."""
    pass


@get_num_workers.validate
def _validate_get_num_wids(axis: Constexpr) -> str:
    if not isinstance(axis, Constexpr):
        return f"Expected axis to be a Constexpr, got {type(axis).__name__}."
    if axis.value < 0:
        return f"Axis must be non-negative, got {axis.value}."
    return ""


@get_num_workers.lower
def _lower_get_num_wids(builder: AlloOpBuilder, axis: Constexpr):
    num_wids = allo.GetNumWorkersOp(builder, axis.value)
    return Proxy(num_wids, index)


@as_member_function
@operator
def get(stream: Proxy) -> Proxy:
    """Get a value from a stream at the given indices."""
    pass


@get.validate
def _validate_get(stream: Proxy) -> str:
    if not isinstance(stream.type, Stream):
        return f"Expected stream to a reference to a Stream, got {stream.type}."
    if (
        isinstance(stream.type, StreamRef)
        and len(stream.type.indices) != stream.type.rank
    ):
        return f"Expected {stream.type.rank} indices for stream of rank {stream.type.rank}, got {len(stream.type.indices)}."
    return ""


@get.lower
def _lower_get(builder: AlloOpBuilder, stream: Proxy) -> Proxy:
    if isinstance(stream.type, StreamRef):
        indices = builder.normalize_indices(stream.type.indices)
    else:
        indices = []
    return builder.create_stream_get(stream, indices)


@as_member_function
@operator
def put(stream: Proxy, value: Proxy):
    """Put a value into a stream at the given indices."""
    pass


@put.validate
def _validate_put(stream: Proxy, value: Proxy) -> str:
    if not isinstance(stream.type, Stream):
        return f"Expected stream to be of type Stream, got {stream.type}."
    if (
        isinstance(stream.type, StreamRef)
        and len(stream.type.indices) != stream.type.rank
    ):
        return f"Expected {stream.type.rank} indices for stream of rank {stream.type.rank}, got {len(stream.type.indices)}."
    return ""


@put.lower
def _lower_put(builder: AlloOpBuilder, stream: Proxy, value: Proxy) -> None:
    if isinstance(stream.type, StreamRef):
        indices = builder.normalize_indices(stream.type.indices)
    else:
        indices = []
    value = builder.cast(value, stream.type.base_type)
    builder.create_stream_put(stream, indices, value)
