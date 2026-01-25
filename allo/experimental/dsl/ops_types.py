# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

from .. import core, settings
from ..compiler.builder import AlloOpBuilder
from ..core.library import operation
from ..core.types import constexpr, scalar_type, tensor
from .ops_common import _is_const_int_tuple, _is_const_of_type, _is_const_positive_int


@operation
def _apfloat_call(elt: scalar_type, value: constexpr, ann=None):
    pass


@_apfloat_call.validate
def _validate_apfloat_call(
    elt: scalar_type, value: constexpr, ann=None
):  # noqa: ARG001
    if not isinstance(value, constexpr) or not isinstance(value.value, (float, int)):
        return f"APFloat requires a floating point or integer constexpr value, got {value}."
    return ""


@_apfloat_call.lower
def _lower_apfloat_call(
    builder: AlloOpBuilder, elt: scalar_type, value: constexpr, ann=None
) -> tensor:
    ret = builder.make_scalar(value.value, elt)
    if ann is not None:
        builder.annotate(ret, ann)
    return ret


@operation
def _apint_call(elt: scalar_type, value: constexpr, ann=None):
    pass


@_apint_call.validate
def _validate_apint_call(elt: scalar_type, value: constexpr, ann=None):  # noqa: ARG001
    if not isinstance(value, constexpr) or not isinstance(value.value, int):
        return f"APInt requires an integer constexpr value, got {value}."
    return ""


@_apint_call.lower
def _lower_apint_call(
    builder: AlloOpBuilder, elt: scalar_type, value: constexpr, ann=None
) -> tensor:
    ret = builder.make_scalar(value.value, elt)
    if ann is not None:
        builder.annotate(ret, ann)
    return ret


@operation
def _array_call(
    base_ty: scalar_type, shape: Sequence[int], init=None, ann=None
) -> tensor:
    pass


@_array_call.validate
def _validate_array_call(
    base_ty: scalar_type, shape: Sequence[int], init=None, ann=None  # noqa: ARG001
) -> str:
    if not isinstance(base_ty, scalar_type):
        return f"Array dtype must be a scalar type, got {base_ty}."
    if isinstance(shape, constexpr):
        shape = [shape]
    if not all(
        isinstance(s, constexpr) and isinstance(s.value, int) and s.value > 0
        for s in shape
    ):
        return (
            "Array shape must be a sequence of positive integer constexpr, "
            f"got {shape}."
        )
    return ""


@_array_call.lower
def _lower_array_call(
    builder: AlloOpBuilder,
    base_ty: scalar_type,
    shape: Sequence[constexpr],
    init=None,
    ann=None,
) -> tensor:
    shape_vals = (
        [shape.value] if isinstance(shape, constexpr) else [s.value for s in shape]
    )
    ret = builder.make_array(base_ty, shape_vals, init, use_tensor=settings.USE_TENSOR)
    if ann is not None:
        builder.annotate(ret, ann)
    return ret


@operation
def _channel_call(
    name: str,
    data_ty: scalar_type | core.Array,
    capacity=2,
    shape: core.tuple | None = None,
) -> tensor:
    pass


@_channel_call.validate
def _validate_channel_call(
    name: constexpr,
    data_ty,
    capacity=constexpr(2),
    shape: core.tuple | None = None,
) -> str:
    if not _is_const_of_type(name, str):
        return (
            "channel operation requires 'name' argument to be string constexpr, "
            f"got {name}."
        )
    if not isinstance(data_ty, core.scalar_type) and not (
        isinstance(data_ty, core.Array)
        and isinstance(data_ty.base_ty, core.scalar_type)
    ):
        return (
            "channel operation requires 'data_ty' to be scalar type or array of scalar type, "
            f"got {data_ty}."
        )
    if not _is_const_positive_int(capacity):
        return (
            "channel operation requires 'capacity' to be positive integer constexpr, "
            f"got {capacity}."
        )
    if shape is not None and not _is_const_int_tuple(shape):
        return (
            "channel operation requires 'shape' to be tuple of integer constexpr, "
            f"got {shape}."
        )
    return ""


@_channel_call.lower
def _lower_channel_call(
    builder: AlloOpBuilder,
    name: constexpr,
    data_ty,
    capacity=constexpr(2),
    shape: core.tuple | None = None,
):
    shape_v = [s.value for s in shape.values] if shape is not None else [1]
    return builder.make_channel(name.value, data_ty, capacity.value, shape_v)


@operation
def cast(value: tensor, dst_ty: scalar_type) -> tensor:
    pass


@cast.validate
def _validate_cast(value: tensor, dst_ty: scalar_type) -> str:
    if not isinstance(dst_ty, scalar_type):
        return f"cast operation requires destination type to be scalar, got {dst_ty}."
    if not (
        isinstance(value.type, core.scalar_type)
        or (
            isinstance(value.type, core.Array)
            and isinstance(value.dtype, core.scalar_type)
        )
    ):
        return f"cast operation requires scalar/array scalar input, got {value.type}."
    return ""


@cast.lower
def _lower_cast(builder: AlloOpBuilder, value: tensor, dst_ty: scalar_type) -> tensor:
    if isinstance(value.type, core.scalar_type):
        return builder.scalar_cast(value, dst_ty)
    return builder.tensor_cast(value, dst_ty)


# Bind constructors to core types.
core.APFloat.__call__ = _apfloat_call
core.APInt.__call__ = _apint_call
core.Array.__call__ = staticmethod(_array_call)
core.Channel.__call__ = staticmethod(_channel_call)
