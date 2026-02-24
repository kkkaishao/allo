# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
from typing import (
    List,
    Tuple,
    Sequence,
    Optional,
    Union,
    Any,
    Dict,
)
from warnings import warn
from functools import cached_property

from .._C.liballo import ir, allo as allo_d
from .._utils import _tuple_create, validate_block_shape
from .. import settings


##############################
# Base frontend type/value IR
##############################


class base_type:
    """
    base class for allo types
    """

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def gen_proxy_value(
        self, handles: Sequence[ir.Value], cursor: int
    ) -> Tuple[base_value, int]:
        """
        Given a list of MLIR values, build its corresponding frontend value.
        Cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError(
            f"_unflatten_ir not implemented for current type {self.__class__}"
        )

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        """
        Append the MLIR types corresponding to this type into out.
        """
        raise NotImplementedError(
            f"append_ir_types not implemented for current type {self.__class__}"
        )

    def mangle(self) -> str:
        raise NotImplementedError(
            f"mangle not implemented for current type {self.__class__}"
        )

    @staticmethod
    def __call__(*args, **kwargs):
        """
        provide a type hint for LSP when the type is called as a constructor,
        e.g. `allo.fp32(1.0)` or `allo.array(allo.int32, [4, 8])`
        """
        raise RuntimeError(
            f"Allo type constructors should not be called directly and only used for type hinting."
        )


class base_value:
    """
    base class for allo values
    """

    type: base_type

    def append_ir_values(self, out: List[ir.Value]) -> None:
        """
        Append the MLIR values corresponding to this value into out.
        """
        raise NotImplementedError(
            f"append_ir_values not implemented for current value {self.__class__}"
        )


######################
# Constexpr type/value
######################


class constexpr_type(base_type):
    """
    base class for allo constexpr types
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, constexpr_type) and self.value == other.value

    def __repr__(self):
        return f"constexpr_type({self.value})"

    def mangle(self) -> str:
        if hasattr(self.value, "mangle"):
            val = self.value.mangle()
        else:
            val = repr(self.value)
        return f"c{val}"

    def gen_proxy_value(
        self, handles: List[ir.Value], cursor: int
    ) -> Tuple[base_value, int]:
        return constexpr(self.value), cursor

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        # constexprs will not occur in the IR
        pass


class constexpr(base_value):
    """
    base class for allo constexpr values
    """

    def __init__(self, value):
        # peel out nested constexprs
        while isinstance(value, constexpr):
            value = value.value
        self.value = value
        self.type = constexpr_type(value)

    def __repr__(self):
        return f"constexpr({self.value})"

    def append_ir_values(self, out: List[ir.Value]) -> None:
        # constexprs will not occur in the IR
        pass

    def __eq__(self, other):
        return constexpr(self.value == _unwrap_if_constexpr(other))

    def __ne__(self, other):
        return constexpr(self.value != _unwrap_if_constexpr(other))

    def __bool__(self):
        return bool(self.value)

    def __index__(self) -> int:
        return int(self.value)

    def __add__(self, other) -> Any:
        return self.value + _unwrap_if_constexpr(other)

    def __sub__(self, other) -> Any:
        return self.value - _unwrap_if_constexpr(other)

    def __mul__(self, other) -> Any:
        return self.value * _unwrap_if_constexpr(other)

    def __lt__(self, other):
        return constexpr(self.value < _unwrap_if_constexpr(other))

    def __le__(self, other):
        return constexpr(self.value <= _unwrap_if_constexpr(other))

    def __gt__(self, other):
        return constexpr(self.value > _unwrap_if_constexpr(other))

    def __ge__(self, other):
        return constexpr(self.value >= _unwrap_if_constexpr(other))


def _unwrap_if_constexpr(o: Any):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return _tuple_create(o, [_unwrap_if_constexpr(x) for x in o])
    if isinstance(o, tuple):
        return tuple([_unwrap_if_constexpr(x) for x in o], o.type)
    return o.value if isinstance(o, constexpr) else o


def _is_allo_non_scalar_tensor(o: Any) -> bool:
    return isinstance(o, tensor) and isinstance(o.type, Array) and o.type.numel != 1


def _is_scalar_tensor(o: Any) -> bool:
    return isinstance(o, tensor) and isinstance(o.type, scalar_type)


def _is_array_of_scalar_tensor(o: Any) -> bool:
    return (
        isinstance(o, tensor)
        and isinstance(o.type, Array)
        and isinstance(o.dtype, scalar_type)
    )


##############
# Scalar types
##############


class scalar_type(base_type):
    """
    base class for allo scalar types
    """

    def __init__(self, name: str, width: int):
        self.name = name
        self.primitive_width = width

    def __eq__(self, other):
        other = _unwrap_if_constexpr(other)
        return isinstance(other, scalar_type) and self.name == other.name

    @property
    def scalar(self):
        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"allo.core.{self.name}"

    def __hash__(self):
        return hash((self.name, self.primitive_width))

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        out.append(self.to_ir(builder))

    def gen_proxy_value(
        self, handles: List[ir.Value], cursor: int
    ) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def to_ir(self, builder: ir.OpBuilder):
        raise NotImplementedError(
            f"to_ir not implemented for current type {self.__class__}"
        )

    def is_int_signless(self):
        return self.name.startswith("int") or self.name.startswith("uint")

    def is_int(self):
        return self.name.startswith("int")

    def is_intn(self, n: int):
        return self.is_int() and self.primitive_width == n

    def is_uint(self):
        return self.name.startswith("uint")

    def is_uintn(self, n: int):
        return self.is_uint() and self.primitive_width == n

    def is_float(self):
        return self.name.startswith("float") or self.name.startswith("bfloat")

    def is_fp16(self):
        return self.name == "float16"

    def is_bf16(self):
        return self.name == "bfloat16"

    def is_fp32(self):
        return self.name == "float32"

    def is_fp64(self):
        return self.name == "float64"

    def is_index(self):
        return self.name == "index"


type_placeholder = scalar_type("placeholder", 64)


class APInt(scalar_type):
    """
    arbitrary precision integer type
    """

    def __init__(self, bitwidth: int, signed: bool = True):
        self.signed = signed
        name = f"int{bitwidth}" if signed else f"uint{bitwidth}"
        super().__init__(name, bitwidth)

    def __eq__(self, other):
        return (
            isinstance(other, APInt)
            and self.primitive_width == other.primitive_width
            and self.signed == other.signed
        )

    def __hash__(self):
        return hash((self.primitive_width, self.signed))

    def to_ir(self, builder: ir.OpBuilder):
        return ir.IntegerType.get(self.primitive_width, builder.context)

    def mangle(self) -> str:
        prefix = "i" if self.signed else "u"
        return f"{prefix}{self.primitive_width}"

    @staticmethod
    def __call__(value, *, ann: Optional[str] = None) -> tensor:
        """
        Placeholder for operation binding in dsl/ops_types.py.
        """
        return base_type.__call__(value, ann)


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


class APFloat(scalar_type):
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

    def __eq__(self, other):
        return (
            isinstance(other, APFloat)
            and self.exp_width == other.exp_width
            and self.mantissa_width == other.mantissa_width
        )

    def __hash__(self):
        return hash((self.exp_width, self.mantissa_width))

    def to_ir(self, builder: ir.OpBuilder):
        if self.name == "float16":
            return ir.F16Type.get(builder.context)
        elif self.name == "float32":
            return ir.F32Type.get(builder.context)
        elif self.name == "float64":
            return ir.F64Type.get(builder.context)
        elif self.name == "bfloat16":
            return ir.BF16Type.get(builder.context)
        else:
            raise NotImplementedError("Unsupported floating point format")

    def mangle(self) -> str:
        return self.name

    @staticmethod
    def __call__(value: float, *, ann: Optional[str] = None) -> tensor:
        """
        Placeholder for operation binding in dsl/ops_types.py.
        """
        return base_type.__call__(value, ann)


# TODO: support real 'arbitrary precision' floating point
fp16 = APFloat(5, 10)
fp32 = APFloat(8, 23)
fp64 = APFloat(11, 52)
bf16 = APFloat(8, 7)


class index_type(scalar_type):
    """
    allo index type
    """

    def __init__(self, width: int = 32):
        super().__init__("index", width)

    def to_ir(self, builder: ir.OpBuilder):
        return ir.IndexType.get(builder.context)

    def mangle(self) -> str:
        return "index"


# default to index type of 32 bits
index = index_type()


###############
# Composite types
###############


def _unwrap_shape(shape) -> List:
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


class Array(base_type):
    """
    base class for allo buffer types
    """

    def __init__(self, base_ty: scalar_type, shape: Sequence):
        self.base_ty = base_ty
        if not shape:
            raise ValueError("array must have non-zero rank")
        self.shape = shape
        self.numel = validate_block_shape(self.shape)
        self.name = f"array({self.shape}, {self.base_ty})"

    def __eq__(self, other):
        other = _unwrap_if_constexpr(other)
        return (
            isinstance(other, Array)
            and self.base_ty == other.base_ty
            and self.shape == other.shape
        )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.base_ty, tuple(self.shape)))

    def to_ir(self, builder: ir.OpBuilder):
        if settings.USE_TENSOR:
            return ir.RankedTensorType.get(self.shape, self.base_ty.to_ir(builder))
        else:
            identity = ir.AffineMap.get_identity(len(self.shape), builder.context)
            return ir.MemRefType.get(self.shape, self.base_ty.to_ir(builder), identity)

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        out.append(self.to_ir(builder))

    def gen_proxy_value(
        self, handles: List[ir.Value], cursor: int
    ) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        elt = self.base_ty.mangle()
        shape = "x".join([str(s) for s in self.shape])
        return f"b{shape}x{elt}"

    @staticmethod
    def __call__(
        base_ty: scalar_type,
        shape: Sequence[int],
        init: Optional[tensor | constexpr] = None,
        ann: Optional[str] = None,
    ) -> tensor:
        """
        Placeholder for operation binding in dsl/ops_types.py.
        """
        return base_type.__call__(base_ty, shape, init, ann)


# export to user interface
# capitalized is for type hinting
# lowercase is for value construction
array = Array(type_placeholder, [0])


class slice_type(base_type):

    def __init__(self):
        self.name = "slice_type"


class tuple_type(base_type):
    def __init__(self, types: Sequence[base_type], fields=None):
        self.types = types
        self.fields = fields

    @cached_property
    def name(self):
        if self.fields is None:
            return "[" + ",".join(str(v) for v in self.types) + "]"
        return (
            "[" + ",".join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + "]"
        )

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.types == other.types
            and self.fields == other.fields
        )

    def __getitem__(self, index: int):
        return self.types[index]

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        for ty in self.types:
            ty.append_ir_types(builder, out)

    def gen_proxy_value(
        self, handles: List[ir.Value], cursor: int
    ) -> Tuple[base_value, int]:
        values = []
        for ty in self.types:
            val, cursor = ty.gen_proxy_value(handles, cursor)
            values.append(val)
        return tuple(values, self), cursor

    def mangle(self) -> str:
        inner = "+".join([ty.mangle() for ty in self.types])
        return f"t{inner}"


class Channel(base_type):
    last_indices: List[List[tensor]] = []

    def __init__(
        self,
        name: str,
        data_ty: Union[scalar_type, Array],
        shape: Sequence[int],
        capacity: int = 2,
    ):
        self.name = name
        self.data_ty = data_ty
        self.capacity = capacity
        self.shape = shape
        self.dshape = data_ty.shape if isinstance(data_ty, Array) else []

    def __eq__(self, other):
        return (
            isinstance(other, Channel)
            and self.name == other.name
            and self.data_ty == other.data_ty
            and self.capacity == other.capacity
            and self.shape == other.shape
        )

    def __str__(self):
        return f"Channel<{self.name}, {self.data_ty}, capacity={self.capacity}, shape={self.shape}>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(
            (self.name, self.data_ty, self.capacity, builtins.tuple(self.shape))
        )

    def append_ir_types(self, builder: ir.OpBuilder, out: List[ir.Type]) -> None:
        # channel types will not occur directly in the IR
        pass

    def mangle(self) -> str:
        base_mangle = self.data_ty.mangle()
        return f"chan{self.capacity}b{base_mangle}"

    @staticmethod
    def __call__(
        name: str,
        data_ty: scalar_type | Array,
        capacity=2,
        shape: builtins.tuple[constexpr, ...] | None = None,
    ) -> tensor:
        """
        Placeholder for operation binding in dsl/ops_types.py.
        """
        return base_type.__call__(name, data_ty, capacity, shape)


# export to user interface
channel = Channel("placeholder", type_placeholder, [], 2)


def _type_for_tuple_values(values, fields=None):
    types = []
    for v in values:
        if isinstance(v, (int, float, scalar_type, Array, slice_type)):
            types.append(constexpr_type(v))
        elif isinstance(v, base_value):
            types.append(v.type)
        else:
            raise TypeError(f"cannot infer type for tuple value {v}")
    return tuple_type(types, fields)


def _normalize_tuple(t):
    normalized_tuple = _unwrap_if_constexpr(t)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


class tuple(base_value):
    def __init__(
        self,
        values: Sequence[base_value],
        type: Optional[tuple_type] = None,
        is_constexpr=False,
    ):
        self.values = [i for i in values]
        if isinstance(type, tuple_type):
            # annotation for LSP support
            self.type: tuple_type = type
        elif isinstance(type, (list, tuple)):
            self.type = tuple_type(type)
        else:
            self.type = _type_for_tuple_values(values)
        self.is_constexpr = is_constexpr

    def __getitem__(self, index):
        if isinstance(index, int):
            index = constexpr(index)
        if isinstance(index, constexpr):
            return self.values[index.value]
        else:
            assert isinstance(index, (slice, builtins.slice))
            return tuple(self.values[index.start : index.stop : index.step])

    def __setitem__(self, key, value):
        key = _unwrap_if_constexpr(key)
        assert isinstance(key, int)
        self.values[key] = value
        # recompute type
        self.type = _type_for_tuple_values(self.values, self.type.fields)

    def __getattr__(self, name: str):
        fields = self.type.fields
        if fields is None or name not in fields:
            raise AttributeError(f"tuple has no attribute '{name}'")
        return self.values[fields.index(name)]

    def __add__(self, other):
        other = _normalize_tuple(other)
        return tuple(self.values + other.values)

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        other = _normalize_tuple(other)
        return constexpr(self.values == other.values)

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def append_ir_values(self, out: List[ir.Value]) -> None:
        for v in self.values:
            v.append_ir_values(out)

    def __repr__(self):
        return f"tuple({', '.join(repr(v) for v in self.values)})"

    def __len__(self):
        return len(self.values)

    def __hash__(self):
        return hash(builtins.tuple(self.values))


class slice:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        self.type = slice_type()


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.dtype.primitive_width
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


class tensor(base_value):
    """
    base class for allo frontend values

    In Allo, all the frontend non-constexpr values are represented as tensors,
    scalars are 0D tensors like int32[]
    block buffers are nD tensors like int32[4, 8]
    """

    def __init__(self, handle: ir.Value, type: base_type):
        """
        not called directly
        """
        if isinstance(handle, ir.OpState):
            raise ValueError(
                "don't pass OpState to tensor, did you forget to call get_result()?"
            )
        super().__init__()
        # handle to the underlying MLIR value
        self.handle = handle
        # block shape
        if isinstance(type, Array):
            self.shape = type.shape
        elif isinstance(type, Channel):
            self.shape = type.shape
        else:
            self.shape = []
        self.numel = type.numel if isinstance(type, Array) else 1
        self.type = type  # the underlying frontend type
        # we use dtype to refer to the primitive scalar type
        if isinstance(type, Array):
            self.dtype = type.base_ty
        elif isinstance(type, Channel):
            self.dtype = (
                type.data_ty.base_ty
                if isinstance(type.data_ty, Array)
                else type.data_ty
            )
        else:
            self.dtype = type

    def __str__(self):
        return str(self.dtype) + "[" + ", ".join(str(s) for s in self.shape) + "]"

    def __repr__(self):
        return self.__str__()

    def append_ir_values(self, out: List[ir.Value]) -> None:
        out.append(self.handle)

    def __hash__(self):
        return hash((self.handle, self.type))


class range:
    def __init__(
        self,
        start,
        stop=None,
        step=None,
        *,
        # used for lsp hints
        unroll=None,
        pipeline=None,
        name: constexpr = constexpr(""),
    ):
        self.name = name.value
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
            if isinstance(r, (constexpr, tensor)):
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

    def __next__(self) -> Tuple[int, ...]:
        raise RuntimeError("allo.grid can only be used within allo kernels")


torch_types_to_core_types_map: Dict[str, scalar_type] = {
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
