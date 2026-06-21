# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

from .utils import operator_body_unreachable
from ..lang.core import (
    AlloValue,
    ConstexprValue,
    StreamType,
    index,
)
from ..lang.operator import operator
from ..compiler.builder import AlloOpBuilder
from .._mlir.dialects import allo


@operator
def get_wid(axis):
    operator_body_unreachable()


@get_wid.build
def _(builder: AlloOpBuilder, axis: ConstexprValue):
    if (
        not isinstance(axis, ConstexprValue)
        or not isinstance(axis.value, int)
        or axis.value < 0
    ):
        return builder.compile_error(
            "The axis of allo.get_wid must be a constant non-negative int"
        )
    wid = allo.GetWorkerIdOp(
        axis.value, ip=builder.save_insertion_point(), loc=builder.get_loc()
    )
    return AlloValue(wid.result, index)


@operator
def get_nw(axis):
    operator_body_unreachable()


@get_nw.build
def _(builder: AlloOpBuilder, axis: ConstexprValue):
    if (
        not isinstance(axis, ConstexprValue)
        or not isinstance(axis.value, int)
        or axis.value < 0
    ):
        return builder.compile_error(
            "The axis of allo.get_num_workers must be a constant non-negative int"
        )
    nw = allo.GetNumWorkersOp(
        axis.value, ip=builder.save_insertion_point(), loc=builder.get_loc()
    )
    return AlloValue(nw.result, index)


def _materialize_stream(builder: AlloOpBuilder, stream):
    if isinstance(stream, AlloValue) and isinstance(stream.type, StreamType):
        if stream.is_indexed:
            return stream
        if stream.type.rank != 0:
            return builder.compile_error(
                f"Stream has rank {stream.type.rank}; index it before calling get/put."
            )
        ref = AlloValue(stream.handle, stream.type)
        ref.indices = ()
        return ref
    stream_type = getattr(stream, "type", type(stream).__name__)
    return builder.compile_error(
        f"Stream get/put expects a stream value, got '{stream_type}'."
    )


@operator(cls=AlloValue)
def get(stream):
    operator_body_unreachable()


@get.build
def _(builder: AlloOpBuilder, stream):
    stream = _materialize_stream(builder, stream)
    return builder.create_stream_get(stream)


@operator(cls=AlloValue)
def put(stream, value):
    operator_body_unreachable()


@put.build
def _(builder: AlloOpBuilder, stream, value):
    stream = _materialize_stream(builder, stream)
    if not isinstance(value, (AlloValue, ConstexprValue)):
        return builder.compile_error(
            f"Stream put expects a runtime value or constexpr literal, got '{type(value).__name__}'."
        )
    return builder.create_stream_put(stream, value)
