# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .. import core
from .._C.liballo import allo as allo_d
from ..compiler.builder import AlloOpBuilder
from ..core.library import as_member_function, operation
from ..core.types import constexpr, tensor, _is_scalar_tensor
from .ops_common import _is_const_positive_int


@operation
def get_pid(dim: int) -> int:
    pass


@get_pid.validate
def _validate_get_pid(dim: constexpr) -> str:
    if not _is_const_positive_int(dim, allow_zero=True):
        return (
            "get_pid operation requires 'dim' to be a non-negative integer constexpr, "
            f"got {dim}."
        )
    return ""


@get_pid.lower
def _lower_get_pid(builder: AlloOpBuilder, dim: constexpr) -> tensor:
    return builder.create_get_pid_op(dim.value)


@operation
def get_num_progs(dim: int) -> int:
    pass


@get_num_progs.validate
def _validate_get_num_progs(dim: constexpr) -> str:
    if not _is_const_positive_int(dim, allow_zero=True):
        return (
            "get_num_progs operation requires 'dim' to be a non-negative integer constexpr, "
            f"got {dim}."
        )
    return ""


@get_num_progs.lower
def _lower_get_num_progs(builder: AlloOpBuilder, dim: constexpr) -> tensor:
    return builder.create_get_n_progs_op(dim.value)


@as_member_function
@operation
def to_stream(chan: tensor, depth: int = 2) -> tensor:
    pass


@to_stream.validate
def _validate_to_stream(chan: tensor, depth=constexpr(2)) -> str:
    if not isinstance(chan.type, core.Channel):
        return (
            "to_stream operation requires 'chan' argument to be Channel, "
            f"got {chan.type}."
        )
    if not _is_const_positive_int(depth):
        return (
            "to_stream operation requires 'depth' to be a positive integer constexpr, "
            f"got {depth}."
        )
    return ""


@to_stream.lower
def _lower_to_stream(builder: AlloOpBuilder, chan: tensor, depth=constexpr(2)):
    depth_v = depth.value
    name_v = chan.type.name
    indices = [i.handle for i in chan.type.last_indices]
    dtype = chan.dtype
    stream_ty = allo_d.StreamType.get(builder.context, dtype.to_ir(builder), depth_v)
    rank = 1 if not chan.type.dshape else len(chan.type.dshape)
    affine_map = builder.get_identity_map(rank)
    ret = allo_d.ChanToStreamOp.create(builder, name_v, indices, stream_ty, affine_map)
    return tensor(ret, core.types.Stream(dtype, depth_v))


@as_member_function
@operation
def acq_buf(name: str, size: int = 1) -> tensor:
    pass


@acq_buf.validate
def _validate_acq_buf(chan: tensor, size=constexpr(1)) -> str:
    if not isinstance(chan.type, core.Channel):
        return (
            "acq_buf operation requires 'chan' argument to be Channel, "
            f"got {chan.type}."
        )
    if not _is_const_positive_int(size):
        return (
            "acq_buf operation requires 'size' to be a positive integer constexpr, "
            f"got {size}."
        )
    return ""


@acq_buf.lower
def _lower_acq_buf(builder: AlloOpBuilder, chan: tensor, size=constexpr(1)):
    size_v = size.value
    indices_v = []
    chan_ty = chan.type
    for val in chan_ty.last_indices:
        if _is_const_positive_int(val, allow_zero=True):
            indices_v.append(builder.create_const_index(val.value).handle)
        elif _is_scalar_tensor(val):
            indices_v.append(builder.scalar_cast(val, core.index).handle)
        else:
            builder.compile_error(
                f"Invalid index value in channel indices: {val}. "
                "Expected integer constexpr or scalar tensor."
            )
    elem_ty = chan_ty.data_ty
    ac_op = allo_d.ChanAcquireBufferOp.create(
        builder, chan_ty.name, indices_v, elem_ty.to_ir(builder), size_v
    )
    tensors = [tensor(ac_op.get_result_at(i), elem_ty) for i in range(size_v)]
    return core.tuple(tensors) if size_v > 1 else tensors[0]


@as_member_function
@operation
def rel_buf(chan: tensor, *buffers) -> None:
    pass


@rel_buf.validate
def _validate_rel_buf(chan: tensor, *buffers) -> str:
    if not isinstance(chan.type, core.Channel):
        return (
            "rel_buf operation requires 'chan' argument to be Channel, "
            f"got {chan.type}."
        )
    for buf in buffers:
        if not isinstance(buf.type, core.Array):
            return (
                "rel_buf operation requires all buffers to be arrays, "
                f"got {buf.type}."
            )
    return ""


@rel_buf.lower
def _lower_rel_buf(builder: AlloOpBuilder, chan: tensor, *buffers):
    name_v = chan.type.name
    indices = [i.handle for i in chan.type.last_indices]
    buffer_handles = [buf.handle for buf in buffers]
    allo_d.ChanReleaseBufferOp.create(builder, name_v, indices, buffer_handles)


@as_member_function
@operation
def put(stream, value) -> None:
    pass


@put.validate
def _validate_put(stream: tensor, value: tensor | constexpr) -> str:
    if not isinstance(stream.type, core.types.Stream):
        return (
            "put operation requires 'stream' argument to be Stream, "
            f"got {stream.type}."
        )
    if not _is_scalar_tensor(value) and not isinstance(value, constexpr):
        return (
            "put operation requires value to be scalar tensor or constexpr, "
            f"got {value}."
        )
    return ""


@put.lower
def _lower_put(builder: AlloOpBuilder, stream: tensor, value: tensor | constexpr):
    if isinstance(value, constexpr):
        value = builder.make_scalar(value.value, stream.dtype)
    else:
        value = builder.scalar_cast(value, stream.dtype)
    allo_d.StreamPutOp.create(builder, value.handle, stream.handle)


@as_member_function
@operation
def get(stream) -> tensor:
    pass


@get.validate
def _validate_get(stream: tensor) -> str:
    if not isinstance(stream.type, core.types.Stream):
        return (
            "get operation requires 'stream' argument to be Stream, "
            f"got {stream.type}."
        )
    return ""


@get.lower
def _lower_get(builder: AlloOpBuilder, stream: tensor) -> tensor:
    ret = allo_d.StreamGetOp.create(builder, stream.handle)
    return tensor(ret, stream.dtype)
