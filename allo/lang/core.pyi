# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the Allo frontend type/value system.

The dtype singletons (``i32``, ``f32`` ...), ``Stream``, ``Stateful`` and
``constexpr`` are declared as ``TypeAlias = Any`` so they work as annotation
heads -- both bare (``a: f32``) and subscripted (``a: f32[M, K]``). Only the
*values* inside the brackets (the shape dimensions) remain flagged by the type
checker; that is inherent to writing runtime values in a type-expression slot
and cannot be fixed by a stub.
"""
from builtins import bool as _bool
from collections.abc import Iterator, Sequence
from typing import Any, TypeAlias

# The native ``_mlir`` bindings carry no stubs, so IR handles are typed ``Any``.
Context: TypeAlias = Any
Type: TypeAlias = Any
Value: TypeAlias = Any

# ==========================================================================#
# dtype singletons -- usable as bare types and as subscript heads
# ==========================================================================#
i2: TypeAlias = Any
i3: TypeAlias = Any
i4: TypeAlias = Any
i5: TypeAlias = Any
i6: TypeAlias = Any
i7: TypeAlias = Any
i8: TypeAlias = Any
i9: TypeAlias = Any
i10: TypeAlias = Any
i11: TypeAlias = Any
i12: TypeAlias = Any
i13: TypeAlias = Any
i14: TypeAlias = Any
i15: TypeAlias = Any
i16: TypeAlias = Any
i32: TypeAlias = Any
i64: TypeAlias = Any
i128: TypeAlias = Any
i256: TypeAlias = Any
u1: TypeAlias = Any
u2: TypeAlias = Any
u3: TypeAlias = Any
u4: TypeAlias = Any
u5: TypeAlias = Any
u6: TypeAlias = Any
u7: TypeAlias = Any
u8: TypeAlias = Any
u9: TypeAlias = Any
u10: TypeAlias = Any
u11: TypeAlias = Any
u12: TypeAlias = Any
u13: TypeAlias = Any
u14: TypeAlias = Any
u15: TypeAlias = Any
u16: TypeAlias = Any
u32: TypeAlias = Any
u64: TypeAlias = Any
u128: TypeAlias = Any
u256: TypeAlias = Any
f16: TypeAlias = Any
f32: TypeAlias = Any
f64: TypeAlias = Any
bf16: TypeAlias = Any
index: TypeAlias = Any
bool: TypeAlias = Any
constexpr: TypeAlias = Any
Stream: TypeAlias = Any
Stateful: TypeAlias = Any

DEFAULT_STREAM_DEPTH: int
torch_dtype_map: dict[str, DType]

# ==========================================================================#
# Frontend type system
# ==========================================================================#
class TypeBase:
    name: str
    def __init__(self, name: str) -> None: ...
    def __eq__(self, value: object, /) -> _bool: ...
    def __ne__(self, value: object, /) -> _bool: ...
    def materialize(self, context: Context, /) -> Type: ...

class Template:
    name: str
    def __init__(self, name: str) -> None: ...
    def __getitem__(self, shape: Any) -> ShapeExpr: ...

class ConstexprType(TypeBase):
    def __init__(self) -> None: ...

class DType(TypeBase):
    primitive_width: int
    def __init__(self, name: str, primitive_width: int) -> None: ...
    def __hash__(self) -> int: ...
    def is_int(self) -> _bool: ...
    def is_intn(self, n: int) -> _bool: ...
    def is_uint(self) -> _bool: ...
    def is_uintn(self, n: int) -> _bool: ...
    def is_int_signless(self) -> _bool: ...
    def is_fp16(self) -> _bool: ...
    def is_fp32(self) -> _bool: ...
    def is_fp64(self) -> _bool: ...
    def is_bf16(self) -> _bool: ...
    def is_float(self) -> _bool: ...
    def is_index(self) -> _bool: ...
    def __getitem__(self, shape: Any) -> ShapeExpr: ...

class APInt(DType):
    signed: _bool
    def __init__(self, bit_width: int, signed: _bool = ...) -> None: ...

class APFloat(DType):
    def __init__(self, exp_width: int, sig_width: int) -> None: ...

class IndexType(DType):
    def __init__(self) -> None: ...

apint = APInt
apfloat = APFloat

class ShapedType(TypeBase):
    shape: Sequence[int]
    dtype: DType
    rank: int
    def __init__(self, name: str, shape: Sequence[int], dtype: DType) -> None: ...

class TensorType(ShapedType):
    def __init__(self, shape: Sequence[int], dtype: DType) -> None: ...

class BufferType(ShapedType):
    def __init__(self, shape: Sequence[int], dtype: DType) -> None: ...

class StreamType(TypeBase):
    base_type: DType | ShapedType
    depth: int
    shape: tuple[int, ...]
    rank: int
    def __init__(
        self,
        base_type: DType | ShapedType,
        depth: int = ...,
        shape: Sequence[int] = ...,
    ) -> None: ...

class StatefulType(TypeBase):
    inner: DType | BufferType
    def __init__(self, inner: DType | BufferType) -> None: ...

def widen_apint_to_std(dtype: DType) -> DType: ...

# ==========================================================================#
# Deferred type annotations (unresolved descriptors)
# ==========================================================================#
class ShapeExpr:
    dtype: Any
    shape: tuple[Any, ...]
    def __init__(self, dtype: Any, shape: Any) -> None: ...

class StreamExpr:
    base: Any
    depth: Any
    shape: tuple[Any, ...]
    def __init__(
        self, base: Any, depth: Any = ..., shape: Sequence[Any] = ...
    ) -> None: ...
    def __getitem__(self, key: Any) -> StreamExpr: ...

class StatefulExpr:
    base: Any
    def __init__(self, base: Any) -> None: ...

# ==========================================================================#
# Frontend value system
# ==========================================================================#
class ValueBase:
    type: TypeBase
    @property
    def handle(self) -> Any: ...

class ConstexprValue(ValueBase):
    value: Any
    def __init__(self, value: Any) -> None: ...

class AlloValue(ValueBase):
    dtype: Any
    shape: Any
    rank: int
    indices: tuple[AlloValue, ...] | None
    def __init__(self, handle: Value, type: TypeBase) -> None: ...
    @property
    def handle(self) -> Value: ...
    @property
    def is_indexed(self) -> _bool: ...

class StatefulValue(ValueBase):
    storage: AlloValue
    def __init__(self, storage: AlloValue, value_type: TypeBase) -> None: ...
    @property
    def handle(self) -> Value: ...
    @property
    def is_scalar(self) -> _bool: ...

def unwrap_if_constexpr(o: Any) -> Any: ...

# ==========================================================================#
# Loop iteration spaces (typed so loop variables come out as ints)
# ==========================================================================#
class Range:
    def __init__(
        self, start: Any, stop: Any = ..., step: Any = ..., *, name: Any = ...
    ) -> None: ...
    def __iter__(self) -> Iterator[int]: ...

class Grid:
    def __init__(self, *ranges: Any, name: Any = ...) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...

range = Range
grid = Grid
