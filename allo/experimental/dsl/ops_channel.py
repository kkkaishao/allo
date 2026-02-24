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


def _gen_channel_indices(builder: AlloOpBuilder, chan_ty: core.Channel):
    indices_v = []
    assert (
        len(chan_ty.last_indices) > 0
    ), "Channel type must have at least one index for codegen."
    for val in chan_ty.last_indices.pop():
        if _is_const_positive_int(val, allow_zero=True):
            indices_v.append(builder.create_const_index(val.value).handle)
        elif _is_scalar_tensor(val):
            indices_v.append(builder.scalar_cast(val, core.index).handle)
        else:
            builder.compile_error(
                f"Invalid index value in channel indices: {val}. "
                "Expected integer constexpr or scalar tensor."
            )
    return indices_v


@as_member_function
@operation
def acquire(name: str, size: int = 1) -> tensor:
    pass


@acquire.validate
def _validate_acquire(chan: tensor, size=constexpr(1)) -> str:
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


@acquire.lower
def _lower_acquire(builder: AlloOpBuilder, chan: tensor, size=constexpr(1)):
    size_v = size.value
    indices_v = []
    chan_ty = chan.type
    indices_v = _gen_channel_indices(builder, chan_ty)
    elem_ty = chan_ty.data_ty
    ac_op = allo_d.ChanAcquireOp.create(
        builder, chan_ty.name, indices_v, elem_ty.to_ir(builder), size_v
    )
    tensors = [tensor(ac_op.get_result_at(i), elem_ty) for i in range(size_v)]
    return core.tuple(tensors) if size_v > 1 else tensors[0]


@as_member_function
@operation
def release(chan: tensor, *buffers) -> None:
    pass


@release.validate
def _validate_release(chan: tensor, *buffers) -> str:
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


@release.lower
def _lower_release(builder: AlloOpBuilder, chan: tensor, *buffers):
    name_v = chan.type.name
    indices_v = _gen_channel_indices(builder, chan.type)
    buffer_handles = [buf.handle for buf in buffers]
    allo_d.ChanReleaseOp.create(builder, name_v, indices_v, buffer_handles)


@as_member_function
@operation
def put(stream, value, blocking=False) -> None:
    pass


@put.validate
def _validate_put(
    chan: tensor, value: tensor | constexpr, blocking=constexpr(False)
) -> str:
    if not isinstance(chan.type, core.types.Channel):
        return (
            "put operation requires 'chan' argument to be Channel, " f"got {chan.type}."
        )
    if not _is_scalar_tensor(value) and not isinstance(value, constexpr):
        return (
            "put operation requires value to be scalar tensor or constexpr, "
            f"got {value}."
        )
    if not isinstance(blocking, constexpr):
        return "put operation requires `blocking` argument to be constexpr"
    return ""


@put.lower
def _lower_put(
    builder: AlloOpBuilder,
    chan: tensor,
    value: tensor | constexpr,
    blocking=constexpr(False),
):
    chan_ty = chan.type
    indices_v = _gen_channel_indices(builder, chan_ty)
    elem_ty = chan_ty.data_ty
    value = builder.make_or_cast(value, elem_ty)
    blocking_v = blocking.value
    allo_d.ChanPutOp.create(builder, chan_ty.name, indices_v, value.handle, blocking_v)


@as_member_function
@operation
def get(stream, blocking=False) -> tensor:
    pass


@get.validate
def _validate_get(chan: tensor, blocking=constexpr(False)) -> str:
    if not isinstance(chan.type, core.types.Channel):
        return (
            "get operation requires 'chan' argument to be Channel, " f"got {chan.type}."
        )
    if not isinstance(chan.type.data_ty, core.scalar_type):
        return (
            "get operation requires channel data type to be scalar, "
            f"got {chan.type.data_ty}."
        )
    if not isinstance(blocking, constexpr):
        return "get operation requires `blocking` argument to be constexpr"
    return ""


@get.lower
def _lower_get(
    builder: AlloOpBuilder, chan: tensor, blocking=constexpr(False)
) -> tensor:
    chan_ty = chan.type
    indices_v = _gen_channel_indices(builder, chan_ty)
    elt_ty = chan.dtype.to_ir(builder)
    blocking_v = blocking.value
    ret = allo_d.ChanGetOp.create(builder, elt_ty, chan_ty.name, indices_v, blocking_v)
    return tensor(ret, chan.dtype)
