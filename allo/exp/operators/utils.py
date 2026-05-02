from __future__ import annotations

import inspect

from ..compiler.builder import AlloOpBuilder, CmpPred
from ..lang.core import AlloValue, ConstexprValue


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


def _materialize_constexpr_pair(
    builder: AlloOpBuilder,
    lhs: AlloValue | ConstexprValue,
    rhs: AlloValue | ConstexprValue,
):
    if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
        return lhs, rhs
    if isinstance(lhs, ConstexprValue):
        assert isinstance(rhs, AlloValue)
        lhs = builder.cast(lhs, rhs.dtype)
    if isinstance(rhs, ConstexprValue):
        assert isinstance(lhs, AlloValue)
        rhs = builder.cast(rhs, lhs.dtype)
    return lhs, rhs


def _prepare_binary_operands(
    builder: AlloOpBuilder,
    lhs: AlloValue | ConstexprValue,
    rhs: AlloValue | ConstexprValue,
    op_name: str,
):
    lhs, rhs = _materialize_constexpr_pair(builder, lhs, rhs)
    assert isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)
    term_signs = [1, -1] if op_name == "sub" else None
    dst_ty = builder.get_promoted_dtype_nary(
        op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
    )
    operands = [builder.cast_to_dtype(lhs, dst_ty), builder.cast_to_dtype(rhs, dst_ty)]
    lhs, rhs = builder.broadcast_pair(operands[0], operands[1])
    return lhs, rhs


def lower_binary_op(
    builder: AlloOpBuilder,
    lhs: AlloValue | ConstexprValue,
    rhs: AlloValue | ConstexprValue,
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


def cmp_op_create(
    builder: AlloOpBuilder,
    lhs: AlloValue | ConstexprValue,
    rhs: AlloValue | ConstexprValue,
    pred: CmpPred,
    op_name: str,
    ordered: bool,
):
    lhs, rhs = _prepare_binary_operands(builder, lhs, rhs, op_name)
    if lhs.dtype.is_float():
        return builder.create_cmpf(lhs, rhs, pred, ordered=ordered)
    return builder.create_cmpi(lhs, rhs, pred, signed=lhs.dtype.is_int())


def lower_unary_op(
    builder: AlloOpBuilder,
    operand: AlloValue | ConstexprValue,
    op_name: str,
    create_fn,
    *,
    promote: bool = True,
    floating: bool | None = None,
    extra_kwargs: dict | None = None,
):
    assert isinstance(operand, AlloValue)
    if promote:
        dst_ty = builder.get_promoted_dtype_nary(op_name, [operand.dtype])
        operand = builder.cast_to_dtype(operand, dst_ty)
    if floating is None:
        floating = operand.dtype.is_float()

    kwargs = {"floating": floating}
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return _invoke_with_supported_kwargs(create_fn, operand, **kwargs)
