# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
from collections.abc import Sequence


from dataclasses import dataclass

from ..._mlir import ir
from ..._mlir.ir import Context, Type, Value
from ..._mlir.dialects.allo import StreamType as MlirStreamType

# ==========================================================================#
# Frontend type system
# ==========================================================================#


class TypeBase:
    """
    Represents a frontend type in the Allo compiler.

    The frontend type should be able to compare itself with other frontend types,
    and generate a corresponding underlying MLIR type.

    Every concrete frontend type should have a unique name, which is used for type comparison and debugging purposes.
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: object, /):
        return isinstance(value, TypeBase) and self.name == value.name

    def __ne__(self, value: object, /):
        return not self.__eq__(value)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def materialize(self, context: Context, /) -> Type:
        raise NotImplementedError()


class Template:
    def __init__(self, name: str):
        assert (
            isinstance(name, str) and name.isidentifier()
        ), f"invalid template name: {name}"
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class ConstexprType(TypeBase):
    """
    Represents a constexpr type in the Allo compiler.

    A constexpr type is a frontend-only type that will not be materialized into MLIR.
    Assert out if some logic tries to materialize a constexpr type, as it should not happen.
    """

    def __init__(self) -> None:
        super().__init__("constexpr")

    def materialize(self, context: Context, /) -> Type:
        assert False, "constexpr type should not be materialized"


constexpr = ConstexprType()  # singleton instance for constexpr type


class DType(TypeBase):
    """
    Represents a primitive data type in the Allo compiler, such as int32, float64, etc.

    Every concrete DType should have unique name.
    """

    def __init__(self, name: str, primitive_width: int):
        super().__init__(name)
        self.primitive_width = primitive_width

    def __hash__(self) -> int:
        return hash((self.name, self.primitive_width))

    def materialize(self, context: Context, /) -> Type:
        raise NotImplementedError(
            "Every concrete DType should implement its own materialization logic"
        )

    def is_int(self):
        return self.name.startswith("int")

    def is_intn(self, n: int):
        return self.is_int() and self.primitive_width == n

    def is_uint(self):
        return self.name.startswith("uint")

    def is_uintn(self, n: int):
        return self.is_uint() and self.primitive_width == n

    def is_int_signless(self):
        return self.is_int() or self.is_uint()

    def is_fp16(self):
        return self.name == "float16"

    def is_fp32(self):
        return self.name == "float32"

    def is_fp64(self):
        return self.name == "float64"

    def is_bf16(self):
        return self.name == "bfloat16"

    def is_float(self):
        return self.is_fp16() or self.is_fp32() or self.is_fp64() or self.is_bf16()

    def is_index(self):
        return self.name == "index"


class APInt(DType):
    """
    Represents an arbitrary precision integer type in the Allo compiler.

    The APInt type is parameterized by its bit width, which can be any positive integer.
    """

    def __init__(self, bit_width: int, signed=False):
        if bit_width <= 0:
            raise ValueError("bit_width must be a positive integer")
        name = f"int{bit_width}" if signed else f"uint{bit_width}"
        super().__init__(name, bit_width)
        self.signed = signed

    def materialize(self, context: Context, /) -> Type:
        return ir.IntegerType.get_signless(self.primitive_width, context)


apint = APInt  # name alias for easier usage

### make some commonly used DType for easier usage
# i1 = APInt(1, signed=True) # use u1 instead
i2 = APInt(2, signed=True)
i3 = APInt(3, signed=True)
i4 = APInt(4, signed=True)
i5 = APInt(5, signed=True)
i6 = APInt(6, signed=True)
i7 = APInt(7, signed=True)
i8 = APInt(8, signed=True)
i9 = APInt(9, signed=True)
i10 = APInt(10, signed=True)
i11 = APInt(11, signed=True)
i12 = APInt(12, signed=True)
i13 = APInt(13, signed=True)
i14 = APInt(14, signed=True)
i15 = APInt(15, signed=True)
i16 = APInt(16, signed=True)
i32 = APInt(32, signed=True)
i64 = APInt(64, signed=True)
i128 = APInt(128, signed=True)
i256 = APInt(256, signed=True)

u1 = APInt(1, signed=False)  # also used as boolean type
u2 = APInt(2, signed=False)
u3 = APInt(3, signed=False)
u4 = APInt(4, signed=False)
u5 = APInt(5, signed=False)
u6 = APInt(6, signed=False)
u7 = APInt(7, signed=False)
u8 = APInt(8, signed=False)
u9 = APInt(9, signed=False)
u10 = APInt(10, signed=False)
u11 = APInt(11, signed=False)
u12 = APInt(12, signed=False)
u13 = APInt(13, signed=False)
u14 = APInt(14, signed=False)
u15 = APInt(15, signed=False)
u16 = APInt(16, signed=False)
u32 = APInt(32, signed=False)
u64 = APInt(64, signed=False)
u128 = APInt(128, signed=False)
u256 = APInt(256, signed=False)

bool = u1


class APFloat(DType):
    """
    Represents an arbitrary precision floating-point type in the Allo compiler.

    The APFloat type is parameterized by its bit width, which can be any positive integer.

    TODO: maybe support real arbitrary precision floating-point types in the future
    """

    def __init__(self, exp_width: int, sig_width: int):
        if exp_width <= 0 or sig_width <= 0:
            raise ValueError("exp_width and sig_width must be positive integers")
        width = 1 + exp_width + sig_width  # 1 bit for sign
        if (exp_width, sig_width) == (5, 10):
            name = "float16"
        elif (exp_width, sig_width) == (8, 23):
            name = "float32"
        elif (exp_width, sig_width) == (11, 52):
            name = "float64"
        elif (exp_width, sig_width) == (8, 7):
            name = "bfloat16"
        else:
            raise NotImplementedError(
                "only fp16, fp32, fp64 and bf16 are supported for now"
            )
        super().__init__(name, width)

    def materialize(self, context: Context, /) -> Type:
        if self.name == "float16":
            return ir.F16Type.get(context)
        elif self.name == "float32":
            return ir.F32Type.get(context)
        elif self.name == "float64":
            return ir.F64Type.get(context)
        elif self.name == "bfloat16":
            return ir.BF16Type.get(context)
        else:
            assert False, f"unsupported floating-point type: {self.name}"


apfloat = APFloat  # name alias for easier usage

### make some commonly used APFloat for easier usage
f16 = APFloat(5, 10)
f32 = APFloat(8, 23)
f64 = APFloat(11, 52)
bf16 = APFloat(8, 7)


class IndexType(DType):
    """
    Represents an index type in the Allo compiler.

    The index type is a special type used for indexing and loop bounds,
    its an opaque type that does not have a fixed bit width
    """

    def __init__(self):
        super().__init__("index", 2**32 - 1)

    def materialize(self, context: Context, /) -> Type:
        return ir.IndexType.get(context)


index = IndexType()  # singleton instance for index type


class ShapedType(TypeBase):
    """
    Represents a shaped type in the Allo compiler, such as tensor, memref, etc.
    It's an abstract base class for all shaped types.

    The ShapedType is parameterized by its shape and element type.
    """

    def __init__(self, name: str, shape: Sequence[int], dtype: DType):
        super().__init__(name)
        self.shape = shape
        self.dtype = dtype
        self.rank = len(shape)


class TensorType(ShapedType):
    """
    Represents a tensor type in the Allo compiler.

    The TensorType is a concrete shaped type that represents a multi-dimensional array of elements.
    """

    def __init__(self, shape: Sequence[int], dtype: DType):
        prefix = "x".join(map(str, shape))
        name = f"tensor<{prefix + 'x' if prefix else ''}{dtype.name}>"
        super().__init__(name, shape, dtype)

    def materialize(self, context: Context, /) -> Type:
        mlir_dtype = self.dtype.materialize(context)
        return ir.RankedTensorType.get(list(self.shape), mlir_dtype)


class BufferType(ShapedType):
    """
    Represents a memref type in the Allo compiler.

    The MemRefType is a concrete shaped type that represents a multi-dimensional array of elements with a specific memory layout.
    """

    def __init__(self, shape: Sequence[int], dtype: DType):
        prefix = "x".join(map(str, shape))
        name = f"memref<{prefix + 'x' if prefix else ''}{dtype.name}>"
        super().__init__(name, shape, dtype)

    def materialize(self, context: Context, /) -> Type:
        mlir_dtype = self.dtype.materialize(context)
        # `loc` is only used for diagnostics during layout verification.
        return ir.MemRefType.get(
            list(self.shape), mlir_dtype, loc=ir.Location.unknown(context)
        )


DEFAULT_STREAM_DEPTH = 2


class StreamType(TypeBase):
    """
    Represents an Allo stream type.

    `base_type` is the transmission unit. It can be either a scalar dtype or a
    shaped buffer type. `shape` describes an array of streams; the empty shape is
    a single rank-0 stream.
    """

    def __init__(
        self,
        base_type: DType | ShapedType,
        depth: int = DEFAULT_STREAM_DEPTH,
        shape: Sequence[int] = (),
        *,
        is_global: builtins.bool = False,
    ):
        assert isinstance(base_type, (DType, ShapedType))
        assert isinstance(depth, int) and depth > 0
        shape = tuple(shape)
        assert all(isinstance(dim, int) and dim >= 0 for dim in shape)
        prefix = "GStream" if is_global else "Stream"
        shape_suffix = "[" + ",".join(str(dim) for dim in shape) + "]" if shape else ""
        super().__init__(f"{prefix}[{base_type}]{shape_suffix}")
        self.base_type = base_type
        self.depth = depth
        self.shape = shape
        self.rank = len(shape)
        self.is_global = is_global

    def __getitem__(self, key):
        if self.shape:
            raise TypeError(f"Stream type '{self}' already has a shape")
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 0:
            prefix = "GStream" if self.is_global else "Stream"
            raise TypeError(f"{prefix}[Ty][] is invalid; use {prefix}[Ty] instead")
        if not all(type(dim) is int and dim >= 0 for dim in key):
            raise TypeError("Stream shape dimensions must be non-negative integers")
        return StreamType(self.base_type, self.depth, key, is_global=self.is_global)

    def materialize(self, context: Context, /) -> Type:
        base = self.base_type.materialize(context)
        return MlirStreamType.get(base, self.depth, list(self.shape))


class _StreamFactory:
    def __init__(self, prefix: str, *, is_global: builtins.bool):
        self.prefix = prefix
        self.is_global = is_global

    def __getitem__(self, base_type):
        if not isinstance(base_type, (DType, ShapedType)):
            raise TypeError(f"{self.prefix} base type must be a scalar or buffer type")
        return StreamType(base_type, DEFAULT_STREAM_DEPTH, (), is_global=self.is_global)

    def __repr__(self) -> str:
        return self.prefix

    __str__ = __repr__


Stream = _StreamFactory("Stream", is_global=False)
GStream = _StreamFactory("GStream", is_global=True)


# =========================================================================#
# Frontend value system
# =========================================================================#


@dataclass
class ValueBase:
    """
    Represents a frontend value in the Allo compiler.

    The frontend value should should hold its frontend type,
    and its underlying MLIR value handle if any.
    """

    type: TypeBase

    @property
    def handle(self):
        raise NotImplementedError()


class ConstexprValue(ValueBase):
    """
    Represents a constexpr value in the Allo compiler.

    A constexpr value is a frontend-only value that does not have a corresponding MLIR value handle.
    """

    def __init__(self, value):
        # peel out nested constexpr value
        while isinstance(value, ConstexprValue):
            value = value.value
        self.value = value
        self.type = ConstexprType()

    def __str__(self) -> str:
        return f"constexpr({self.value})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def handle(self):
        return None


class AlloValue(ValueBase):
    """
    Proxy value class for all non-constexpr values in the Allo compiler.

    An Allo value should always have a corresponding MLIR value handle, as it represents a value that will be materialized into MLIR.
    """

    def __init__(self, handle: Value, type: TypeBase):
        assert handle is not None, "handle cannot be None for AlloValue"
        self._handle = handle
        self.type = type
        self.dtype = type.dtype if isinstance(type, ShapedType) else type
        # wrap the shape to frontend values
        self.shape = (
            [ConstexprValue(s) for s in type.shape]
            if isinstance(type, (ShapedType, StreamType))
            else ()
        )
        self.rank = len(self.shape)
        self.indices: tuple[AlloValue, ...] | None = None

    def __str__(self) -> str:
        return f"AlloValue<{self.type}>"

    def __repr__(self) -> str:
        return f"AlloValue<{self.type}>({self.handle})"

    @property
    def handle(self) -> Value:
        return self._handle

    @property
    def is_indexed(self) -> builtins.bool:
        return self.indices is not None


class AlloSymbolRef(ValueBase):
    """Frontend proxy for a global stream symbol, optionally with stream indices."""

    def __init__(
        self,
        name: str,
        type: StreamType,
        indices: Sequence[AlloValue] | None = None,
    ):
        assert isinstance(name, str) and name
        assert isinstance(type, StreamType)
        assert type.is_global
        assert indices is None or all(isinstance(idx, AlloValue) for idx in indices)
        self.name = name
        self.type = type
        self.indices = None if indices is None else tuple(indices)
        self.shape = [ConstexprValue(s) for s in type.shape]
        self.rank = type.rank

    @property
    def is_indexed(self) -> builtins.bool:
        return self.indices is not None

    def __str__(self) -> str:
        return f"AlloSymbolRef<{self.type}>(@{self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def handle(self):
        return None


# map from PyTorch dtype string to Allo DType, for easier interop with PyTorch/NumPy
torch_dtype_map: dict[str, DType] = {
    "bool": u1,
    "int8": i8,
    "int16": i16,
    "short": i16,
    "int32": i32,
    "int": i32,
    "int64": i64,
    "intp": i64,
    "uint8": u8,
    "uint16": u16,
    "uint32": u32,
    "uint64": u64,
    "uintp": u64,
    "float16": f16,
    "half": f16,
    "float32": f32,
    "float": f32,
    "float64": f64,
    "double": f64,
    "bfloat16": bf16,
}


def unwrap_if_constexpr(o):
    """
    Helper function to unwrap the value from a Constexpr wrapper if needed.
    If the input value is not a Constexpr, return it as is.
    """
    if isinstance(o, list):
        return [unwrap_if_constexpr(v) for v in o]
    if isinstance(o, tuple):
        return tuple(unwrap_if_constexpr(v) for v in o)
    return o.value if isinstance(o, ConstexprValue) else o


class Range:
    def __init__(
        self,
        start,
        stop=None,
        step=None,
        *,
        name: ConstexprValue = ConstexprValue(""),
    ):
        self.name = unwrap_if_constexpr(name)
        self.step = step if step is not None else ConstexprValue(1)
        if stop is None:
            self.start = ConstexprValue(0)
            self.stop = start
        else:
            self.start = start
            self.stop = stop

    def __iter__(self) -> Range:
        raise RuntimeError("allo.range can only be used within allo kernels")

    def __next__(self) -> AlloValue:
        raise RuntimeError("allo.range can only be used within allo kernels")


range = Range  # name alias for easier usage


class Grid:
    def __init__(self, *ranges: tuple, name: ConstexprValue = ConstexprValue("")):
        self.name = name.value
        self.starts = []
        self.stops = []
        self.steps = []

        # canonicalize expressions
        for r in ranges:
            if isinstance(r, (ConstexprValue, AlloValue)):
                self.starts.append(ConstexprValue(0))
                self.stops.append(r)
                self.steps.append(ConstexprValue(1))
            elif len(r) == 1:
                self.starts.append(ConstexprValue(0))
                self.stops.append(r[0])
                self.steps.append(ConstexprValue(1))
            elif len(r) == 2:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(ConstexprValue(1))
            elif len(r) == 3:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(r[2])
            else:
                raise ValueError(
                    f"invalid range specification {r} in grid, expected 1, 2 or 3 elements"
                )

    def __iter__(self) -> Grid:
        raise RuntimeError("allo.grid can only be used within allo kernels")

    def __next__(self) -> tuple[AlloValue, ...]:
        raise RuntimeError("allo.grid can only be used within allo kernels")


grid = Grid  # name alias for easier usage
