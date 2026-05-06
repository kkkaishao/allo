# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, NoReturn, Sequence, cast

from ...compiler.errors import ActError
from ...lang.act import (
    BufferSpec,
    IndexExpr,
    PatternExpr,
    _as_tuple,
    as_index_expr,
    pattern,
)


def _pattern_body_unreachable() -> NoReturn:
    raise RuntimeError("Pattern declarations are not directly executed.")


def _expect_pattern(value, name: str) -> PatternExpr:
    if not isinstance(value, PatternExpr):
        raise ActError(f"Pattern '{name}' expects a source pattern.")
    return value


def _basis(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    return cast(tuple[IndexExpr, ...], expr.attrs["basis"])


def _counts(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    return cast(tuple[IndexExpr, ...], expr.attrs["counts"])


def _strides(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    return cast(tuple[IndexExpr, ...], expr.attrs["strides"])


def _reassociation(expr: PatternExpr) -> tuple[tuple[int, ...], ...]:
    return cast(tuple[tuple[int, ...], ...], expr.attrs["reassociation"])


def _output_shape(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    return cast(tuple[IndexExpr, ...], expr.attrs["output_shape"])


def _permutation(expr: PatternExpr) -> tuple[int, ...]:
    return cast(tuple[int, ...], expr.attrs["permutation"])


def _mul_shape_dims(dims: Sequence[IndexExpr]) -> IndexExpr:
    assert len(dims) > 0
    if all(dim.is_static for dim in dims):
        value = 1
        for dim in dims:
            value *= dim.static_value
        return IndexExpr("const", value=value)
    result = dims[0]
    for dim in dims[1:]:
        result = result * dim
    return result


def _static_product(dims: Sequence[IndexExpr]) -> int | None:
    if not all(dim.is_static for dim in dims):
        return None
    value = 1
    for dim in dims:
        value *= dim.static_value
    return value


def _check_static_index(expr: IndexExpr, pred: Callable[[int], bool], message: str):
    if expr.is_static and not pred(expr.static_value):
        raise ActError(message)


def _check_static_dim(expr: IndexExpr, where: str):
    _check_static_index(
        expr, lambda value: value > 0, f"{where} dimensions must be positive."
    )


def _check_reassociation_covers(
    reassociation: tuple[tuple[int, ...], ...], rank: int, where: str
):
    if len(reassociation) == 0:
        raise ActError(f"{where} reassociation must not be empty.")
    if any(len(group) == 0 for group in reassociation):
        raise ActError(f"{where} reassociation groups must not be empty.")
    flat = [idx for group in reassociation for idx in group]
    if flat != list(range(rank)):
        raise ActError(
            f"{where} reassociation must cover dimensions 0..{rank - 1} in order."
        )


def _check_expand_shape_static_consistency(
    source_shape: tuple[IndexExpr, ...],
    reassociation: tuple[tuple[int, ...], ...],
    output_shape: tuple[IndexExpr, ...],
):
    for source_dim, group in zip(source_shape, reassociation):
        output_product = _static_product([output_shape[i] for i in group])
        if (
            source_dim.is_static
            and output_product is not None
            and source_dim.static_value != output_product
        ):
            raise ActError(
                "expand_shape static shape mismatch: "
                f"source dimension is {source_dim.static_value}, "
                f"but reassociation group {group} has product {output_product}."
            )


@pattern
def strided(buffer: BufferSpec, *, basis, counts, strides) -> PatternExpr:
    _pattern_body_unreachable()


@strided.build
def _strided_build(buffer: BufferSpec, *, basis, counts, strides) -> PatternExpr:
    if not isinstance(buffer, BufferSpec):
        raise ActError("strided expects a buffer.")
    return strided.create(
        buffer=buffer,
        attrs={
            "basis": tuple(as_index_expr(v) for v in _as_tuple(basis)),
            "counts": tuple(as_index_expr(v) for v in _as_tuple(counts)),
            "strides": tuple(as_index_expr(v) for v in _as_tuple(strides)),
        },
    )


@strided.verify
def _strided_verify(expr: PatternExpr):
    assert expr.buffer is not None
    basis = _basis(expr)
    counts = _counts(expr)
    strides = _strides(expr)
    if not (len(basis) == len(counts) == len(strides)):
        raise ActError("strided basis, counts, and strides must have the same rank.")
    if expr.buffer.kind == "hbm" and len(counts) != len(expr.buffer.shape):
        raise ActError(
            f"HBM strided access expects rank {len(expr.buffer.shape)}, got {len(counts)}."
        )
    if expr.buffer.kind != "hbm" and len(counts) != 1:
        raise ActError("MVP frontend only supports 1-D on-chip strided accesses.")
    for dim in basis:
        _check_static_index(
            dim, lambda value: value >= 0, "strided basis must be non-negative."
        )
    for dim in counts:
        _check_static_index(
            dim, lambda value: value > 0, "strided counts must be positive."
        )
    for dim in strides:
        _check_static_index(
            dim, lambda value: value > 0, "strided strides must be positive."
        )

    limits = expr.buffer.shape if expr.buffer.kind == "hbm" else (expr.buffer.slots,)
    for i, (base, count, stride, limit) in enumerate(
        zip(basis, counts, strides, limits)
    ):
        if base.is_static and count.is_static and stride.is_static:
            max_index = base.static_value + stride.static_value * (
                count.static_value - 1
            )
            if max_index >= limit:
                raise ActError(
                    f"strided access for buffer '{expr.buffer.name}' is out of bounds in dimension {i}."
                )


@strided.shape
def _strided_shape(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    assert expr.buffer is not None
    return expr.buffer.visible_shape_for_counts(_counts(expr))


@strided.lower
def _strided_lower(emitter, expr: PatternExpr) -> str:
    result = emitter.value()
    emitter.emit(
        f"{result} = act.strided basis{emitter.format_paren_list(_basis(expr))} "
        f"counts{emitter.format_paren_list(_counts(expr))} "
        f"strides{emitter.format_paren_list(_strides(expr))}"
    )
    return result


@pattern
def expand(
    source: PatternExpr, reassociation: Sequence[Sequence[int]], *, shape: Sequence
) -> PatternExpr:
    _pattern_body_unreachable()


@expand.build
def _expand_build(
    source: PatternExpr, reassociation: Sequence[Sequence[int]], *, shape: Sequence
) -> PatternExpr:
    source = _expect_pattern(source, "expand_shape")
    return expand.create(
        source=source,
        attrs={
            "reassociation": tuple(tuple(group) for group in reassociation),
            "output_shape": tuple(as_index_expr(dim) for dim in shape),
        },
    )


@expand.verify
def _expand_verify(expr: PatternExpr):
    assert expr.source is not None
    reassociation = _reassociation(expr)
    output_shape = _output_shape(expr)
    if len(reassociation) != len(expr.source.visible_shape()):
        raise ActError(
            "expand_shape reassociation must have one group per source dimension."
        )
    _check_reassociation_covers(reassociation, len(output_shape), "expand_shape")
    for dim in output_shape:
        _check_static_dim(dim, "expand_shape output")
    _check_expand_shape_static_consistency(
        expr.source.visible_shape(), reassociation, output_shape
    )


@expand.shape
def _expand_shape(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    return _output_shape(expr)


@expand.lower
def _expand_lower(emitter, expr: PatternExpr) -> str:
    assert expr.source is not None
    result = emitter.value()
    source = emitter.emit_pattern(expr.source)
    reassociation = _reassociation(expr)
    output_shape = _output_shape(expr)
    emitter.emit(
        f"{result} = act.expand_shape {source} "
        f"{emitter.format_reassociation(reassociation)} "
        f"output_shape {emitter.format_square_list(output_shape)}"
    )
    return result


@pattern
def collapse(
    source: PatternExpr, reassociation: Sequence[Sequence[int]]
) -> PatternExpr:
    _pattern_body_unreachable()


@collapse.build
def _collapse_build(
    source: PatternExpr, reassociation: Sequence[Sequence[int]]
) -> PatternExpr:
    source = _expect_pattern(source, "collapse_shape")
    return collapse.create(
        source=source,
        attrs={"reassociation": tuple(tuple(group) for group in reassociation)},
    )


@collapse.verify
def _collapse_verify(expr: PatternExpr):
    assert expr.source is not None
    _check_reassociation_covers(
        _reassociation(expr), len(expr.source.visible_shape()), "collapse_shape"
    )


@collapse.shape
def _collapse_shape(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    assert expr.source is not None
    source_shape = expr.source.visible_shape()
    return tuple(
        _mul_shape_dims([source_shape[i] for i in group])
        for group in _reassociation(expr)
    )


@collapse.lower
def _collapse_lower(emitter, expr: PatternExpr) -> str:
    assert expr.source is not None
    result = emitter.value()
    source = emitter.emit_pattern(expr.source)
    emitter.emit(
        f"{result} = act.collapse_shape {source} "
        f"{emitter.format_reassociation(_reassociation(expr))}"
    )
    return result


@pattern
def transpose(source: PatternExpr, permutation: Sequence[int]) -> PatternExpr:
    _pattern_body_unreachable()


@transpose.build
def _transpose_build(source: PatternExpr, permutation: Sequence[int]) -> PatternExpr:
    source = _expect_pattern(source, "transpose")
    return transpose.create(source=source, attrs={"permutation": tuple(permutation)})


@transpose.verify
def _transpose_verify(expr: PatternExpr):
    assert expr.source is not None
    permutation = _permutation(expr)
    shape_rank = len(expr.source.visible_shape())
    if sorted(permutation) != list(range(shape_rank)):
        raise ActError(
            f"transpose permutation {permutation} does not match rank {shape_rank}."
        )


@transpose.shape
def _transpose_shape(expr: PatternExpr) -> tuple[IndexExpr, ...]:
    assert expr.source is not None
    source_shape = expr.source.visible_shape()
    return tuple(source_shape[i] for i in _permutation(expr))


@transpose.lower
def _transpose_lower(emitter, expr: PatternExpr) -> str:
    assert expr.source is not None
    result = emitter.value()
    source = emitter.emit_pattern(expr.source)
    permutation = "[" + ", ".join(str(v) for v in _permutation(expr)) + "]"
    emitter.emit(f"{result} = act.transpose {source} permutation = {permutation}")
    return result
