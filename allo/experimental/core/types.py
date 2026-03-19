# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import re
from collections.abc import Sequence

from .._C.ir import (
    Value,
    Type,
    NoneType,
    Context,
    MemRefType,
    F16Type,
    F32Type,
    F64Type,
    BF16Type,
    IndexType as MLIRIndexType,
    IntegerType,
    AffineMap,
    RankedTensorType,
    OpState,
)

from typing import cast

ALLO_BUILTIN_ATTR = "__allo_builtin_attribute__"

##############################
# Base frontend type/value IR
##############################


class BaseType:
    def __eq__(self, value: object, /) -> bool:
        raise NotImplementedError

    def __ne__(self, value: object, /) -> bool:
        eq = self.__eq__(value)
        return not eq

    def to_frontend(
        self, handles: Sequence[Value], cursor: int
    ) -> builtins.tuple[BaseValue, int]:
        raise NotImplementedError

    def to_mlir(self, context: Context) -> Type:
        raise NotImplementedError


class BaseValue:
    type: BaseType

    @property
    def handle(self) -> Value | None:
        raise NotImplementedError


class ConstexprType(BaseType):
    """
    allo constexpr type, represents a compile-time constant value
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, ConstexprType) and self.value == value.value

    def to_frontend(
        self, handles: Sequence[Value], cursor: int
    ) -> builtins.tuple[BaseValue, int]:
        return Constexpr(self.value), cursor

    def to_mlir(self, context: Context) -> Type:
        # constexprs will not occur in the IR
        return NoneType.get(context)


class Constexpr(BaseValue):
    """
    allo constexpr value, represents a compile-time constant value
    """

    def __init__(self, value):
        # peel out nested constexprs
        while isinstance(value, Constexpr):
            value = value.value
        self.value = value
        self.type = ConstexprType(value)

    @property
    def handle(self) -> Value | None:
        # constexprs will not occur in the IR
        return None

    def __str__(self) -> str:
        return f"constexpr({self.value})"

    def __repr__(self) -> str:
        return self.__str__()


constexpr = Constexpr  # alias for easier usage


def unwrap_if_constexpr(o: object) -> object:
    if isinstance(o, list):
        return [unwrap_if_constexpr(x) for x in o]
    return o.value if isinstance(o, Constexpr) else o


class DType(BaseType):
    """
    base class for allo primitive data types
    """

    def __init__(self, name: str, primitive_width: int):
        self.name = name
        self.primitive_width = primitive_width

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, DType) and self.name == value.name

    def __str__(self):
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.primitive_width))

    def to_frontend(
        self, handles: Sequence[Value], cursor: int
    ) -> builtins.tuple[BaseValue, int]:
        return Proxy(handles[cursor], self), cursor + 1

    def to_mlir(self, context: Context) -> Type:
        raise NotImplementedError(
            f"to_mlir not implemented for current type {self.__class__}"
        )

    def is_int(self) -> bool:
        return self.name.startswith("int")

    def is_int_n(self, n: int) -> bool:
        return self.is_int() and self.primitive_width == n

    def is_uint(self) -> bool:
        return self.name.startswith("uint")

    def is_uint_n(self, n: int) -> bool:
        return self.is_uint() and self.primitive_width == n

    def is_int_signless(self) -> bool:
        return self.is_int() or self.is_uint()

    def is_int_signless_n(self, n: int) -> bool:
        return self.is_int_signless() and self.primitive_width == n

    def is_float(self) -> bool:
        return self.name.startswith("float") or self.name.startswith("bfloat")

    def is_fp16(self) -> bool:
        return self.name == "float16"

    def is_fp32(self) -> bool:
        return self.name == "float32"

    def is_fp64(self) -> bool:
        return self.name == "float64"

    def is_bf16(self) -> bool:
        return self.name == "bfloat16"

    def is_index(self) -> bool:
        return self.name == "index"


class APInt(DType):
    """
    arbitrary precision integer type
    """

    def __init__(self, bitwidth: int, signed: bool = True):
        self.signed = signed
        name = f"int{bitwidth}" if signed else f"uint{bitwidth}"
        super().__init__(name, bitwidth)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, APInt)
            and self.primitive_width == value.primitive_width
            and self.signed == value.signed
        )

    def to_mlir(self, context: Context) -> Type:
        return IntegerType.get(self.primitive_width, context)


class APFloat(DType):
    """
    arbitrary precision floating point type
    """

    def __init__(self, exp_width: int, mantissa_width: int):
        self.exp_width = exp_width
        self.mantissa_width = mantissa_width
        width = 1 + exp_width + mantissa_width  # sign bit + exponent + mantissa
        if (exp_width, mantissa_width) == (5, 10):
            name = "float16"
        elif (exp_width, mantissa_width) == (8, 23):
            name = "float32"
        elif (exp_width, mantissa_width) == (11, 52):
            name = "float64"
        elif (exp_width, mantissa_width) == (8, 7):
            name = "bfloat16"
        else:
            raise NotImplementedError("Unsupported floating point format")
        super().__init__(name, width)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, APFloat)
            and self.exp_width == value.exp_width
            and self.mantissa_width == value.mantissa_width
        )

    def to_mlir(self, context: Context) -> Type:
        if self.name == "float16":
            return F16Type.get(context)
        elif self.name == "float32":
            return F32Type.get(context)
        elif self.name == "float64":
            return F64Type.get(context)
        elif self.name == "bfloat16":
            return BF16Type.get(context)
        else:
            raise NotImplementedError("Unsupported floating point format")


class IndexType(DType):
    """
    allo index type, represents a machine integer type used for indexing and loop bounds
    """

    def __init__(self, bitwidth: int = 32):
        super().__init__("index", bitwidth)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, IndexType)
            and self.primitive_width == value.primitive_width
        )

    def to_mlir(self, context: Context) -> Type:
        return MLIRIndexType.get(context)


index = IndexType()  # default index type

# we have no bool type
# use int1 instead
int1 = APInt(1, True)
int2 = APInt(2, True)
int3 = APInt(3, True)
int4 = APInt(4, True)
int5 = APInt(5, True)
int6 = APInt(6, True)
int7 = APInt(7, True)
int8 = APInt(8, True)
int9 = APInt(9, True)
int10 = APInt(10, True)
int11 = APInt(11, True)
int12 = APInt(12, True)
int13 = APInt(13, True)
int14 = APInt(14, True)
int15 = APInt(15, True)
int16 = APInt(16, True)
int32 = APInt(32, True)
int64 = APInt(64, True)

uint1 = APInt(1, False)
uint2 = APInt(2, False)
uint3 = APInt(3, False)
uint4 = APInt(4, False)
uint5 = APInt(5, False)
uint6 = APInt(6, False)
uint7 = APInt(7, False)
uint8 = APInt(8, False)
uint9 = APInt(9, False)
uint10 = APInt(10, False)
uint11 = APInt(11, False)
uint12 = APInt(12, False)
uint13 = APInt(13, False)
uint14 = APInt(14, False)
uint15 = APInt(15, False)
uint16 = APInt(16, False)
uint32 = APInt(32, False)
uint64 = APInt(64, False)

# TODO: support real 'arbitrary precision' floating point
fp16 = APFloat(5, 10)
fp32 = APFloat(8, 23)
fp64 = APFloat(11, 52)
bf16 = APFloat(8, 7)


class ShapedType(BaseType):
    """
    base class for allo shaped types (buffers and tensors)
    """

    dtype: DType
    shape: Sequence[int]
    rank: int

    def __init__(self, dtype: DType, shape: Sequence[int]):
        self.dtype = dtype
        self.shape = [shape] if isinstance(shape, int) else shape
        self.rank = len(shape)


class BufferType(ShapedType):
    """
    allo buffer type, represents a block of memory with a certain shape and element type
    """

    def __init__(self, dtype: DType, shape: Sequence[int]):
        super().__init__(dtype, shape)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, BufferType)
            and self.dtype == value.dtype
            and self.shape == value.shape
        )

    def __str__(self):
        return f"buffer<{"x".join(str(s) for s in self.shape)}x{self.dtype}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.dtype, builtins.tuple(self.shape)))

    def to_frontend(
        self, handles: Sequence[Value], cursor: int
    ) -> builtins.tuple[BaseValue, int]:
        return Proxy(handles[cursor], self), cursor + 1

    def to_mlir(self, context: Context) -> MemRefType:
        identity = AffineMap.get_identity(len(self.shape), context)
        return MemRefType.get(self.shape, self.dtype.to_mlir(context), identity)


class TensorType(ShapedType):
    """
    allo tensor type, used when enabling tensor-based IR generation.
    Represents a multi-dimensional array with a certain shape and element type.
    """

    def __init__(self, dtype: DType, shape: Sequence[int]):
        super().__init__(dtype, shape)

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, TensorType)
            and self.dtype == value.dtype
            and self.shape == value.shape
        )

    def __str__(self):
        return f"tensor<{"x".join(str(s) for s in self.shape)}x{self.dtype}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.dtype, builtins.tuple(self.shape)))

    def to_frontend(
        self, handles: Sequence[Value], cursor: int
    ) -> builtins.tuple[BaseValue, int]:
        return Proxy(handles[cursor], self), cursor + 1

    def to_mlir(self, context: Context) -> RankedTensorType:
        return RankedTensorType.get(self.shape, self.dtype.to_mlir(context))


class Proxy(BaseValue):
    """
    allo frontend proxy value. This is the value type that is actually used in the frontend IR. It contains a handle to the underlying MLIR value, as well as metadata about the type and shape of the value for use in the frontend.
    """

    def __init__(self, handle: Value | OpState, type: ShapedType | DType):
        if isinstance(handle, OpState):
            if handle.get_num_results() != 1:
                raise ValueError(
                    "Proxy can only be created from an OpState with a single result"
                )
            handle = handle.get_result_at(0)
        self._handle = cast(Value, handle)
        self.type = type
        self.dtype = type.dtype if isinstance(type, ShapedType) else type
        self.shape = tuple(type.shape) if isinstance(type, ShapedType) else ()
        if len(self.shape) > 0:
            self.shape = tuple(Constexpr(s) for s in self.shape)

    def __hash__(self) -> int:
        return hash(self._handle)

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(str(s) for s in self.shape)}]"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def handle(self) -> Value:
        return self._handle


torch_types_to_core_types_map: dict[str, DType] = {
    "bool": int1,
    "int8": int8,
    "int16": int16,
    "short": int16,
    "int32": int32,
    "int": int32,
    "int64": int64,
    "intp": int64,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
    "uintp": uint64,
    "float16": fp16,
    "half": fp16,
    "float32": fp32,
    "float": fp32,
    "float64": fp64,
    "double": fp64,
    "bfloat16": bf16,
}


class range:
    def __init__(
        self,
        start,
        stop=None,
        step=None,
        *,
        name: constexpr = constexpr(""),
    ):
        self.name = unwrap_if_constexpr(name)
        self.step = step if step is not None else constexpr(1)
        if stop is None:
            self.start = constexpr(0)
            self.stop = start
        else:
            self.start = start
            self.stop = stop

    def __iter__(self) -> range:
        raise RuntimeError("allo.range can only be used within allo kernels")

    def __next__(self) -> int:
        raise RuntimeError("allo.range can only be used within allo kernels")


class grid:
    def __init__(self, *ranges: tuple, name: constexpr = constexpr("")):
        self.name = name.value
        self.starts = []
        self.stops = []
        self.steps = []

        # canonicalize expressions
        for r in ranges:
            if isinstance(r, (constexpr, Proxy)):
                self.starts.append(constexpr(0))
                self.stops.append(r)
                self.steps.append(constexpr(1))
            elif len(r) == 1:
                self.starts.append(constexpr(0))
                self.stops.append(r[0])
                self.steps.append(constexpr(1))
            elif len(r) == 2:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(constexpr(1))
            elif len(r) == 3:
                self.starts.append(r[0])
                self.stops.append(r[1])
                self.steps.append(r[2])
            else:
                raise ValueError(
                    f"invalid range specification {r} in grid, expected 1, 2 or 3 elements"
                )

    def __iter__(self) -> grid:
        raise RuntimeError("allo.grid can only be used within allo kernels")

    def __next__(self) -> tuple[int, ...]:
        raise RuntimeError("allo.grid can only be used within allo kernels")
