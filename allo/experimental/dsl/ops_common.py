# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from ..core.types import (
    Constexpr,
    ConstexprType,
    DType,
    ShapedType,
    BufferType,
    Proxy,
)
from ..compiler.builder import AlloOpBuilder
from ..core.library import NO_FOLD


def binary_op_checks(lhs, rhs, op_name="") -> str:
    if not isinstance(lhs, (Proxy, Constexpr)) or not isinstance(
        rhs, (Proxy, Constexpr)
    ):
        return f"{op_name} can only be applied to Proxies or Constexprs, got {type(lhs)} and {type(rhs)}."
    lhs_ok = isinstance(lhs.type, (DType, ConstexprType)) or (
        isinstance(lhs.type, ShapedType) and isinstance(lhs.dtype, DType)
    )
    rhs_ok = isinstance(rhs.type, (DType, ConstexprType)) or (
        isinstance(rhs.type, ShapedType) and isinstance(rhs.dtype, DType)
    )
    if not (lhs_ok and rhs_ok):
        return f"{op_name} requires operands to be scalars or arrays of scalars, got {lhs.type} and {rhs.type}."
    if isinstance(lhs.type, BufferType) and isinstance(rhs.type, BufferType):
        return f"Cannot perform {op_name} on two buffers. Buffer operations must be performed through load and store."
    return ""


def _invoke_with_supported_kwargs(fn, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)

    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kw:
        return fn(*args, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def lower_binary_op(
    builder: AlloOpBuilder,
    lhs: Proxy | Constexpr,
    rhs: Proxy | Constexpr,
    op_name: str,
    create_fn,
    *,
    signed: bool | None = None,
    floating: bool | None = None,
    base_floating: bool | None = None,
    exp_floating: bool | None = None,
    extra_kwargs: dict | None = None,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return NO_FOLD

    assert isinstance(lhs, Proxy) and isinstance(rhs, Proxy)
    lhs_is_float = lhs.dtype.is_float()
    rhs_is_float = rhs.dtype.is_float()
    if floating is None:
        floating = lhs_is_float and rhs_is_float
    if signed is None:
        signed = not (lhs.dtype.is_uint() and rhs.dtype.is_uint())
    if base_floating is None:
        base_floating = lhs_is_float
    if exp_floating is None:
        exp_floating = rhs_is_float
    if floating:
        signed = False

    kwargs = {
        "signed": signed,
        "floating": floating,
        "base_floating": base_floating,
        "exp_floating": exp_floating,
    }
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return _invoke_with_supported_kwargs(create_fn, lhs, rhs, **kwargs)


def _materialize_Constexpr_pair(
    builder: AlloOpBuilder, lhs: Proxy | Constexpr, rhs: Proxy | Constexpr
):
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return lhs, rhs
    if isinstance(lhs, Constexpr):
        lhs = builder.cast(lhs, rhs.dtype)
    if isinstance(rhs, Constexpr):
        rhs = builder.cast(rhs, lhs.dtype)
    return lhs, rhs


def _prepare_binary_operands(
    builder: AlloOpBuilder,
    lhs: Proxy | Constexpr,
    rhs: Proxy | Constexpr,
    op_name: str,
):
    lhs, rhs = _materialize_Constexpr_pair(builder, lhs, rhs)
    if isinstance(lhs, Constexpr) and isinstance(rhs, Constexpr):
        return lhs, rhs

    assert isinstance(lhs, Proxy) and isinstance(rhs, Proxy)
    term_signs = [1, -1] if op_name == "sub" else None
    dst_ty = builder.get_promoted_dtype_nary(
        op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
    )
    operands = [builder.cast(lhs, dst_ty), builder.cast(rhs, dst_ty)]
    lhs, rhs = builder.create_broadcast(operands[0], operands[1])
    return lhs, rhs


def cmp_op_create(
    builder: AlloOpBuilder,
    lhs: Proxy | Constexpr,
    rhs: Proxy | Constexpr,
    pred,
    op_name="",
    ordered=True,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    assert isinstance(lhs, Proxy) and isinstance(rhs, Proxy)
    if lhs.dtype.is_float():
        return builder.create_cmpf(lhs, rhs, pred, ordered=ordered)
    return builder.create_cmpi(lhs, rhs, pred, signed=lhs.dtype.is_int())


def unary_op_checks(operand, op_name="") -> str:
    ok = isinstance(operand.type, (DType, ConstexprType)) or (
        isinstance(operand.type, ShapedType) and isinstance(operand.dtype, DType)
    )
    if not ok:
        return (
            f"{op_name} requires the operand to be a scalar or an array of scalars, "
            f"got {operand.type}."
        )
    return ""


def lower_unary_op(
    builder: AlloOpBuilder,
    operand: Proxy,
    op_name: str,
    create_fn,
    *,
    promote: bool = True,
    floating: bool | None = None,
    extra_kwargs: dict | None = None,
) -> Proxy:
    if promote:
        dst_ty = builder.get_promoted_dtype_nary(op_name, [operand.dtype])
        operand = builder.cast(operand, dst_ty)
    if floating is None:
        floating = operand.dtype.is_float()

    kwargs = {
        "floating": floating,
    }
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return _invoke_with_supported_kwargs(create_fn, operand, **kwargs)
