# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NoReturn

from ..compiler.errors import ActError
from ..lang.act import (
    ActTensorType,
    TensorProxy,
    dtype_to_mlir,
    primitive,
)
from ..lang.core import DType


def _expect_tensor(value, name: str) -> TensorProxy:
    if not isinstance(value, TensorProxy):
        raise ActError(f"Primitive '{name}' expects tensor operands.")
    return value


def _identity_map(rank: int) -> str:
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_map<({dims}) -> ({dims})>"


def _cast_op(src: ActTensorType, dst: ActTensorType) -> str:
    if src.dtype.primitive_width > dst.dtype.primitive_width:
        return "arith.truncf"
    return "arith.extf"


def _primitive_body_unreachable() -> NoReturn:
    raise RuntimeError("Primitive declarations are not directly executed.")


@primitive
def identity(x: TensorProxy) -> TensorProxy:
    _primitive_body_unreachable()


@identity.infer
def _(x: TensorProxy) -> ActTensorType:
    x = _expect_tensor(x, "identity")
    return x.type


@identity.build
def _(x: TensorProxy) -> TensorProxy:
    assert identity.infer_impl is not None
    result_type = identity.infer_impl(x)
    return identity.create([x], {}, result_type)


@identity.lower
def _(emitter, node, final_dest):
    return emitter.get_value(node.inputs[0])


@primitive
def matmul(
    a: TensorProxy, b: TensorProxy, *, dtype: DType | None = None
) -> TensorProxy:
    _primitive_body_unreachable()


@matmul.infer
def _(a: TensorProxy, b: TensorProxy, *, dtype: DType | None = None) -> ActTensorType:
    a = _expect_tensor(a, "matmul")
    b = _expect_tensor(b, "matmul")
    if dtype is not None and not isinstance(dtype, DType):
        raise ActError("Primitive 'matmul' dtype must be an Allo DType.")
    if a.type.rank != 2 or b.type.rank != 2:
        raise ActError("Primitive 'matmul' expects rank-2 tensors.")
    if (
        a.type.shape[1] is not None
        and b.type.shape[0] is not None
        and a.type.shape[1] != b.type.shape[0]
    ):
        raise ActError("Primitive 'matmul' has incompatible contraction dimensions.")
    return ActTensorType(dtype or a.type.dtype, (a.type.shape[0], b.type.shape[1]))


@matmul.build
def _(a: TensorProxy, b: TensorProxy, *, dtype: DType | None = None) -> TensorProxy:
    assert matmul.infer_impl is not None
    return matmul.create([a, b], {}, matmul.infer_impl(a, b, dtype=dtype))


@matmul.lower
def _(emitter, node, final_dest):
    lhs, rhs = node.inputs
    lhs_value = emitter.get_value(lhs)
    rhs_value = emitter.get_value(rhs)
    out_value = emitter.output_value(node.result_type, final_dest)
    result = emitter.value()
    emitter.emit(
        f"{result} = linalg.matmul ins({lhs_value}, {rhs_value} : {lhs.type.mlir()}, {rhs.type.mlir()}) "
        f"outs({out_value} : {node.result_type.mlir()}) -> {node.result_type.mlir()}"
    )
    return result


@primitive
def cast(x: TensorProxy, *, dtype: DType) -> TensorProxy:
    _primitive_body_unreachable()


@cast.infer
def _(x: TensorProxy, *, dtype: DType) -> ActTensorType:
    x = _expect_tensor(x, "cast")
    if not isinstance(dtype, DType):
        raise ActError("Primitive 'cast' dtype must be an Allo DType.")
    if not x.type.dtype.is_float() or not dtype.is_float():
        raise ActError("MVP primitive 'cast' only supports floating-point casts.")
    return ActTensorType(dtype, x.type.shape)


@cast.build
def _(x: TensorProxy, *, dtype: DType) -> TensorProxy:
    assert cast.infer_impl is not None
    result_type = cast.infer_impl(x, dtype=dtype)
    if x.type == result_type:
        return identity(x)
    return cast.create([x], {}, result_type)


@cast.lower
def _(emitter, node, final_dest):
    source = node.inputs[0]
    source_value = emitter.get_value(source)
    out_value = emitter.output_value(node.result_type, final_dest)
    result = emitter.value()
    map_attr = _identity_map(node.result_type.rank)
    iterators = ", ".join('"parallel"' for _ in range(node.result_type.rank))
    src_dtype = dtype_to_mlir(source.type.dtype)
    dst_dtype = dtype_to_mlir(node.result_type.dtype)
    cast_value = emitter.value()
    emitter.emit(
        f"{result} = linalg.generic {{indexing_maps = [{map_attr}, {map_attr}], "
        f"iterator_types = [{iterators}]}} ins({source_value}: {source.type.mlir()}) "
        f"outs({out_value}: {node.result_type.mlir()}) {{"
    )
    emitter.indent()
    emitter.emit(f"^bb0(%in: {src_dtype}, %out: {dst_dtype}):")
    emitter.indent()
    emitter.emit(
        f"{cast_value} = {_cast_op(source.type, node.result_type)} %in : "
        f"{src_dtype} to {dst_dtype}"
    )
    emitter.emit(f"linalg.yield {cast_value} : {dst_dtype}")
    emitter.dedent()
    emitter.dedent()
    emitter.emit(f"}} -> {node.result_type.mlir()}")
    return result


@primitive
def softmax(x: TensorProxy, *, dim: int) -> TensorProxy:
    _primitive_body_unreachable()


@softmax.infer
def _(x: TensorProxy, *, dim: int) -> ActTensorType:
    x = _expect_tensor(x, "softmax")
    if not isinstance(dim, int):
        raise ActError("Primitive 'softmax' dimension must be an integer.")
    if dim < 0 or dim >= x.type.rank:
        raise ActError(
            f"Primitive 'softmax' dimension {dim} is out of range for rank {x.type.rank}."
        )
    return x.type


@softmax.build
def _(x: TensorProxy, *, dim: int) -> TensorProxy:
    assert softmax.infer_impl is not None
    return softmax.create([x], {"dim": dim}, softmax.infer_impl(x, dim=dim))


@softmax.lower
def _(emitter, node, final_dest):
    source = node.inputs[0]
    source_value = emitter.get_value(source)
    out_value = emitter.output_value(node.result_type, final_dest)
    result = emitter.value()
    dim = node.attrs["dim"]
    emitter.emit(
        f"{result} = linalg.softmax dimension({dim}) ins({source_value} : {source.type.mlir()}) "
        f"outs({out_value} : {node.result_type.mlir()}) -> {node.result_type.mlir()}"
    )
    return result
