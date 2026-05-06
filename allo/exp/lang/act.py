# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, overload

from ..compiler.errors import (
    ActError,
    DiagnosticLocation,
    callable_diagnostic_location,
    capture_act_location,
)
from . import core as _core
from .core import DType

_BARE_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_CURRENT_ACCESS_BUILDER: ContextVar["InstructionBuilder | None"] = ContextVar(
    "_CURRENT_ACCESS_BUILDER", default=None
)


def _check_symbol_name(name: str, kind: str):
    if not isinstance(name, str) or _BARE_SYMBOL_RE.fullmatch(name) is None:
        raise ActError(f"Invalid {kind} name '{name}'.")


def _as_tuple(value) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def dtype_to_mlir(dtype: DType) -> str:
    if dtype.name == "bfloat16":
        return "bf16"
    if dtype.name == "float16":
        return "f16"
    if dtype.name == "float32":
        return "f32"
    if dtype.name == "float64":
        return "f64"
    if dtype.name == "index":
        return "index"
    if dtype.name.startswith("int") or dtype.name.startswith("uint"):
        return f"i{dtype.primitive_width}"
    raise ActError(f"Unsupported dtype '{dtype}'.")


_DTYPES = {
    "bf16": _core.bf16,
    "bfloat16": _core.bf16,
    "f16": _core.f16,
    "float16": _core.f16,
    "f32": _core.f32,
    "float32": _core.f32,
    "f64": _core.f64,
    "float64": _core.f64,
    "index": _core.index,
}
for _prefix in ("i", "u"):
    for _width in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        32,
        64,
        128,
        256,
    ):
        _name = f"{_prefix}{_width}"
        if hasattr(_core, _name):
            _DTYPES[_name] = getattr(_core, _name)


@dataclass(frozen=True)
class ActTensorType:
    dtype: DType
    shape: tuple[int | None, ...]

    def mlir(self) -> str:
        dims = "x".join("?" if dim is None else str(dim) for dim in self.shape)
        return f"tensor<{dims}x{dtype_to_mlir(self.dtype)}>"

    @property
    def rank(self) -> int:
        return len(self.shape)


def parse_tensor_annotation(annotation) -> ActTensorType:
    if isinstance(annotation, ActTensorType):
        return annotation
    if annotation is inspect.Parameter.empty:
        raise ActError("Compute arguments require tensor type annotations.")
    if not isinstance(annotation, str):
        raise ActError(f"Unsupported compute annotation '{annotation}'.")

    text = annotation.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    match = re.fullmatch(r"([A-Za-z_]\w*)\s*\[(.*)\]", text)
    if match is None:
        raise ActError(f"Unsupported compute annotation '{annotation}'.")
    dtype_name, shape_text = match.groups()
    dtype = _DTYPES.get(dtype_name)
    if dtype is None:
        raise ActError(f"Unknown dtype '{dtype_name}'.")

    dims = []
    for raw_dim in shape_text.split(","):
        dim = raw_dim.strip()
        if dim == "?":
            dims.append(None)
        elif re.fullmatch(r"\d+", dim):
            dims.append(int(dim))
        else:
            raise ActError(f"Unsupported tensor dimension '{dim}'.")
    if len(dims) == 0:
        raise ActError(f"Tensor annotation '{annotation}' has empty shape.")
    return ActTensorType(dtype, tuple(dims))


@dataclass(frozen=True)
class IndexExpr:
    kind: str
    value: int | str | None = None
    lhs: "IndexExpr | None" = None
    rhs: "IndexExpr | None" = None

    def __add__(self, other) -> "IndexExpr":
        return IndexExpr("add", lhs=self, rhs=as_index_expr(other))

    def __radd__(self, other) -> "IndexExpr":
        return IndexExpr("add", lhs=as_index_expr(other), rhs=self)

    def __mul__(self, other) -> "IndexExpr":
        return IndexExpr("mul", lhs=self, rhs=as_index_expr(other))

    def __rmul__(self, other) -> "IndexExpr":
        return IndexExpr("mul", lhs=as_index_expr(other), rhs=self)

    @property
    def is_static(self) -> bool:
        return self.kind == "const"

    @property
    def static_value(self) -> int:
        assert self.kind == "const"
        assert isinstance(self.value, int)
        return self.value

    def collect_params(self) -> set[str]:
        if self.kind == "param":
            assert isinstance(self.value, str)
            return {self.value}
        if self.kind == "const":
            return set()
        assert self.lhs is not None and self.rhs is not None
        return self.lhs.collect_params() | self.rhs.collect_params()


def as_index_expr(value) -> IndexExpr:
    if isinstance(value, IndexExpr):
        return value
    if isinstance(value, int):
        return IndexExpr("const", value=value)
    raise ActError(f"Expected an index expression, got '{value}'.")


def _operand_role(inst: "InstructionSpec", index: int) -> str:
    if index < len(inst.sources):
        return f"src[{index}] buffer '{inst.sources[index].name}'"
    dst_index = index - len(inst.sources)
    return f"dst[{dst_index}] buffer '{inst.destinations[dst_index].name}'"


def _compute_arg_role(inst: "InstructionSpec", index: int, arg_name: str | None) -> str:
    if arg_name is None:
        return f"compute argument for {_operand_role(inst, index)}"
    return f"compute argument '{arg_name}' for {_operand_role(inst, index)}"


@dataclass
class BufferSpec:
    isa: "ISA"
    name: str
    kind: str
    slots: int
    shape: tuple[int, ...]
    dtype: DType

    def type_mlir(self) -> str:
        dtype = dtype_to_mlir(self.dtype)
        if self.kind == "scalar":
            return f"!act.scalar<{dtype}>"
        dims = "x".join(str(dim) for dim in self.shape)
        return f"!act.{self.kind}<{dims}x{dtype}>"

    def visible_shape_for_counts(
        self, counts: Sequence[IndexExpr]
    ) -> tuple[IndexExpr, ...]:
        shape = tuple(counts)
        if self.kind != "hbm":
            shape = shape + tuple(as_index_expr(dim) for dim in self.shape)
            if len(shape) > 1 and shape[0].is_static and shape[0].static_value == 1:
                shape = shape[1:]
        return shape


class PatternOp:
    def __init__(self, fn: Callable[..., Any]):
        self.name = fn.__name__
        self.build_impl: Callable[..., "PatternExpr"] | None = None
        self.shape_impl: Callable[["PatternExpr"], tuple[IndexExpr, ...]] | None = None
        self.verify_impl: Callable[["PatternExpr"], None] | None = None
        self.lower_impl: Callable[..., str] | None = None
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> "PatternExpr":
        location = capture_act_location()
        builder = _require_access_builder(f"Pattern '{self.name}'")
        if self.build_impl is None:
            raise ActError(
                f"Pattern '{self.name}' does not define trace construction.",
                location=location,
            )
        try:
            result = self.build_impl(*args, **kwargs)
            builder._check_pattern(result)
        except ActError as error:
            error.attach_location(location)
            raise
        result.location = location
        return result

    def build(self, fn: Callable[..., "PatternExpr"]) -> Callable[..., "PatternExpr"]:
        assert self.build_impl is None
        assert callable(fn), "Pattern build function must be callable."
        self.build_impl = fn
        return fn

    def shape(
        self, fn: Callable[["PatternExpr"], tuple[IndexExpr, ...]]
    ) -> Callable[["PatternExpr"], tuple[IndexExpr, ...]]:
        assert self.shape_impl is None
        assert callable(fn), "Pattern shape function must be callable."
        self.shape_impl = fn
        return fn

    def verify(
        self, fn: Callable[["PatternExpr"], None]
    ) -> Callable[["PatternExpr"], None]:
        assert self.verify_impl is None
        assert callable(fn), "Pattern verify function must be callable."
        self.verify_impl = fn
        return fn

    def lower(self, fn: Callable[..., str]) -> Callable[..., str]:
        assert self.lower_impl is None
        assert callable(fn), "Pattern lower function must be callable."
        self.lower_impl = fn
        return fn

    def create(
        self,
        *,
        attrs: dict[str, Any] | None = None,
        source: "PatternExpr | None" = None,
        buffer: BufferSpec | None = None,
        location: DiagnosticLocation | None = None,
    ) -> "PatternExpr":
        expr = PatternExpr(
            self, attrs or {}, source=source, buffer=buffer, location=location
        )
        if self.verify_impl is not None:
            self.verify_impl(expr)
        return expr


def pattern(fn: Callable[..., Any]) -> PatternOp:
    return PatternOp(fn)


def _collect_index_params(value) -> set[str]:
    if isinstance(value, IndexExpr):
        return value.collect_params()
    if isinstance(value, (tuple, list)):
        params = set()
        for item in value:
            params |= _collect_index_params(item)
        return params
    return set()


@contextmanager
def _access_context(builder: "InstructionBuilder"):
    token = _CURRENT_ACCESS_BUILDER.set(builder)
    try:
        yield
    finally:
        _CURRENT_ACCESS_BUILDER.reset(token)


def _require_access_builder(where: str) -> "InstructionBuilder":
    builder = _CURRENT_ACCESS_BUILDER.get()
    if builder is None:
        raise ActError(f"{where} can only be used inside an access region.")
    return builder


@dataclass
class PatternExpr:
    pattern: PatternOp
    attrs: dict[str, Any] = field(default_factory=dict)
    source: "PatternExpr | None" = None
    buffer: BufferSpec | None = None
    location: DiagnosticLocation | None = None

    @property
    def kind(self) -> str:
        return self.pattern.name

    def visible_shape(self) -> tuple[IndexExpr, ...]:
        shape = self.pattern.shape_impl
        assert shape is not None, f"Pattern '{self.kind}' does not define shape."
        return shape(self)

    def collect_params(self) -> set[str]:
        params = self.source.collect_params() if self.source is not None else set()
        for value in self.attrs.values():
            params |= _collect_index_params(value)
        return params

    def base_buffer(self) -> BufferSpec:
        if self.buffer is not None:
            return self.buffer
        assert self.source is not None
        return self.source.base_buffer()


@dataclass(eq=False)
class TensorProxy:
    name: str
    type: ActTensorType
    _ctx: "ComputeContext"
    producer: "PrimitiveNode | None" = None


class Primitive:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        legal_regions: Sequence[str] = ("compute",),
    ):
        self.name = fn.__name__
        self.legal_regions = tuple(legal_regions)
        self.infer_impl: Callable[..., ActTensorType] | None = None
        self.build_impl: Callable[..., TensorProxy] | None = None
        self.lower_impl: Callable[..., str] | None = None
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> TensorProxy:
        location = capture_act_location()
        if self.build_impl is None:
            raise ActError(
                f"Primitive '{self.name}' does not define trace construction.",
                location=location,
            )
        try:
            result = self.build_impl(*args, **kwargs)
        except ActError as error:
            error.attach_location(location)
            raise
        if result.producer is not None:
            result.producer.location = location
        return result

    def infer(self, fn: Callable[..., ActTensorType]) -> Callable[..., ActTensorType]:
        assert self.infer_impl is None
        assert callable(fn), "Primitive inference function must be callable."
        self.infer_impl = fn
        return fn

    def build(self, fn: Callable[..., TensorProxy]) -> Callable[..., TensorProxy]:
        assert self.build_impl is None
        assert callable(fn), "Primitive build function must be callable."
        self.build_impl = fn
        return fn

    def lower(self, fn: Callable[..., str]) -> Callable[..., str]:
        assert self.lower_impl is None
        assert callable(fn), "Primitive lower function must be callable."
        self.lower_impl = fn
        return fn

    def create(
        self,
        inputs: Sequence[TensorProxy],
        attrs: dict,
        result_type: ActTensorType,
    ) -> TensorProxy:
        if len(inputs) == 0:
            raise ActError(f"Primitive '{self.name}' requires inputs.")
        for inp in inputs:
            if not isinstance(inp, TensorProxy):
                raise ActError(f"Primitive '{self.name}' expects tensor operands.")
        ctx = inputs[0]._ctx
        if ctx.region_kind not in self.legal_regions:
            raise ActError(
                f"Primitive '{self.name}' is not legal in {ctx.region_kind} region."
            )
        return ctx.add_node(self, inputs, attrs, result_type)


@overload
def primitive(fn: Callable[..., Any]) -> Primitive: ...


@overload
def primitive(
    *, legal_regions: Sequence[str] = ("compute",)
) -> Callable[[Callable[..., Any]], Primitive]: ...


def primitive(
    fn: Callable[..., Any] | None = None,
    *,
    legal_regions: Sequence[str] = ("compute",),
) -> Primitive | Callable[[Callable[..., Any]], Primitive]:
    def decorator(fn: Callable[..., Any]) -> Primitive:
        return Primitive(fn, legal_regions=legal_regions)

    if fn is not None:
        return decorator(fn)
    return decorator


@dataclass(eq=False)
class PrimitiveNode:
    primitive: Primitive
    inputs: tuple[TensorProxy, ...]
    attrs: dict
    result_type: ActTensorType
    location: DiagnosticLocation | None = None
    result: TensorProxy | None = None

    @property
    def kind(self) -> str:
        return self.primitive.name


@dataclass
class ComputeArg:
    name: str
    type: ActTensorType
    proxy: TensorProxy


@dataclass
class ComputeSpec:
    args: list[ComputeArg]
    nodes: list[PrimitiveNode]
    returns: list[TensorProxy]
    location: DiagnosticLocation | None = None


class ComputeContext:
    def __init__(
        self,
        instruction: "InstructionSpec",
        arg_types: Sequence[ActTensorType],
        arg_names: Sequence[str],
        location: DiagnosticLocation | None = None,
    ):
        self.instruction = instruction
        self.location = location
        self.region_kind = "compute"
        self.nodes: list[PrimitiveNode] = []
        self.args: list[ComputeArg] = []
        for name, ty in zip(arg_names, arg_types):
            proxy = TensorProxy(name, ty, self)
            self.args.append(ComputeArg(name, ty, proxy))

    def add_node(
        self,
        primitive: Primitive,
        inputs: Sequence[TensorProxy],
        attrs: dict,
        result_type: ActTensorType,
    ) -> TensorProxy:
        assert isinstance(primitive, Primitive)
        assert isinstance(result_type, ActTensorType)
        for inp in inputs:
            if not isinstance(inp, TensorProxy):
                raise ActError(f"Primitive '{primitive.name}' expects tensor operands.")
            if inp._ctx is not self:
                raise ActError(
                    "Primitive inputs must belong to the same compute region."
                )
        node = PrimitiveNode(primitive, tuple(inputs), attrs, result_type)
        result = TensorProxy(
            f"{primitive.name}{len(self.nodes)}", result_type, self, producer=node
        )
        node.result = result
        self.nodes.append(node)
        return result

    def finish(self, returns) -> ComputeSpec:
        if not isinstance(returns, tuple):
            returns = (returns,)
        out = []
        for value in returns:
            if not isinstance(value, TensorProxy):
                raise ActError("Compute functions must return primitive tensor values.")
            if value._ctx is not self:
                raise ActError("Compute returned a tensor from another compute region.")
            if value.producer is None:
                raise ActError("Compute returns must be produced by primitives.")
            out.append(value)
        expected = len(self.instruction.destinations)
        if len(out) != expected:
            raise ActError(
                f"Compute for '{self.instruction.name}' must return {expected} value(s), got {len(out)}."
            )
        for i, (value, dest_arg) in enumerate(zip(out, self.args[-expected:])):
            role = _operand_role(self.instruction, len(self.instruction.sources) + i)
            _check_tensor_compatible(
                value.type,
                dest_arg.type,
                f"return value for {role}",
            )
        return ComputeSpec(self.args, self.nodes, out, self.location)


def _check_tensor_compatible(
    actual: ActTensorType, expected: ActTensorType, where: str
):
    if actual.dtype != expected.dtype:
        raise ActError(f"{where} expects dtype {expected.dtype}, got {actual.dtype}.")
    if actual.rank != expected.rank:
        raise ActError(f"{where} expects rank {expected.rank}, got {actual.rank}.")
    for i, (actual_dim, expected_dim) in enumerate(zip(actual.shape, expected.shape)):
        if (
            actual_dim is not None
            and expected_dim is not None
            and actual_dim != expected_dim
        ):
            raise ActError(
                f"{where} dimension {i} expects {expected_dim}, got {actual_dim}."
            )


@dataclass
class InstructionSpec:
    isa: "ISA"
    name: str
    sources: list[BufferSpec]
    destinations: list[BufferSpec]
    location: DiagnosticLocation | None = None
    addr_params: list[str] = field(default_factory=list)
    patterns: list[PatternExpr] = field(default_factory=list)
    compute_spec: ComputeSpec | None = None

    def validate_complete(self):
        if len(self.addr_params) == 0:
            raise ActError(
                f"Instruction '{self.name}' does not declare address parameters.",
                location=self.location,
            )
        if len(self.patterns) != len(self.sources) + len(self.destinations):
            raise ActError(
                f"Instruction '{self.name}' does not define access patterns.",
                location=self.location,
            )
        if self.compute_spec is None:
            raise ActError(
                f"Instruction '{self.name}' does not define compute semantics.",
                location=self.location,
            )


class InstructionBuilder:
    def __init__(self, spec: InstructionSpec):
        self.spec = spec

    def access(self, fn: Callable):
        location = callable_diagnostic_location(fn, marker=".access")
        if len(self.spec.patterns) != 0:
            raise ActError(
                f"Instruction '{self.spec.name}' already defines access patterns.",
                location=location,
            )

        try:
            signature = inspect.signature(fn)
            params = list(signature.parameters.values())
            args = self._make_access_args(params)
            with _access_context(self):
                result = fn(*args)
        except ActError as error:
            error.attach_location(location or self.spec.location)
            raise

        try:
            self._set_access_patterns(self._normalize_access_return(result))
        except ActError as error:
            error.attach_location(location or self.spec.location, override=True)
            raise
        return fn

    def _make_access_args(self, params: Sequence[inspect.Parameter]):
        if len(params) == 0:
            raise ActError(
                f"Access for '{self.spec.name}' must declare address parameters."
            )
        for param in params:
            self._check_access_param(param)
        names = [param.name for param in params]
        if len(set(names)) != len(names):
            raise ActError(
                f"Instruction '{self.spec.name}' has duplicate address parameters."
            )
        self.spec.addr_params = names
        return [IndexExpr("param", value=name) for name in names]

    def _check_access_param(self, param: inspect.Parameter):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise ActError(f"Access for '{self.spec.name}' does not support *args.")
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ActError(f"Access for '{self.spec.name}' does not support **kwargs.")
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ActError(
                f"Access for '{self.spec.name}' only supports positional address parameters."
            )
        if param.default is not inspect.Parameter.empty:
            raise ActError(
                f"Access parameter '{param.name}' must not have a default value."
            )
        _check_symbol_name(param.name, "address parameter")

    def _normalize_access_return(self, result) -> tuple[PatternExpr, ...]:
        if isinstance(result, PatternExpr):
            values = (result,)
        elif isinstance(result, (tuple, list)):
            values = tuple(result)
        else:
            raise ActError(
                f"Access for '{self.spec.name}' must return access pattern(s)."
            )
        for value in values:
            if not isinstance(value, PatternExpr):
                raise ActError(
                    f"Access for '{self.spec.name}' must return access pattern(s)."
                )
        return values

    def _set_access_patterns(self, patterns: Sequence[PatternExpr]):
        expected = len(self.spec.sources) + len(self.spec.destinations)
        if len(patterns) != expected:
            raise ActError(
                f"Instruction '{self.spec.name}' expects {expected} access patterns "
                f"({len(self.spec.sources)} src + {len(self.spec.destinations)} dst), got {len(patterns)}."
            )
        known_params = set(self.spec.addr_params)
        for index, (pattern, buffer) in enumerate(
            zip(patterns, self.spec.sources + self.spec.destinations)
        ):
            self._check_pattern(pattern)
            if pattern.base_buffer() is not buffer:
                raise ActError(
                    f"Instruction '{self.spec.name}' access pattern for "
                    f"{_operand_role(self.spec, index)} is based on buffer "
                    f"'{pattern.base_buffer().name}'."
                )
            unknown = pattern.collect_params() - known_params
            if unknown:
                raise ActError(
                    f"Instruction '{self.spec.name}' uses unknown address parameter(s): {sorted(unknown)}."
                )
        self.spec.patterns = list(patterns)
        if self.spec.compute_spec is not None:
            self._check_compute_args(
                [arg.type for arg in self.spec.compute_spec.args],
                [arg.name for arg in self.spec.compute_spec.args],
            )

    def compute(self, fn: Callable):
        location = callable_diagnostic_location(fn, marker=".compute")
        if self.spec.compute_spec is not None:
            raise ActError(
                f"Instruction '{self.spec.name}' already defines compute semantics.",
                location=location,
            )
        try:
            signature = inspect.signature(fn)
            params = list(signature.parameters.values())
            expected = len(self.spec.sources) + len(self.spec.destinations)
            if len(params) != expected:
                raise ActError(
                    f"Compute for '{self.spec.name}' expects {expected} arguments "
                    f"({len(self.spec.sources)} src + {len(self.spec.destinations)} dst), got {len(params)}."
                )
            for param in params:
                self._check_compute_param(param)
            arg_types = [parse_tensor_annotation(param.annotation) for param in params]
            arg_names = [param.name for param in params]
            self._check_compute_args(arg_types, arg_names)
            ctx = ComputeContext(self.spec, arg_types, arg_names, location)
            self.spec.compute_spec = ctx.finish(fn(*[arg.proxy for arg in ctx.args]))
        except ActError as error:
            error.attach_location(location or self.spec.location)
            raise
        return fn

    def _check_buffer(self, buffer: BufferSpec):
        if not isinstance(buffer, BufferSpec) or buffer.isa is not self.spec.isa:
            raise ActError(
                f"Instruction '{self.spec.name}' references a buffer outside ISA '{self.spec.isa.name}'."
            )

    def _check_pattern(self, pattern: PatternExpr):
        if not isinstance(pattern, PatternExpr):
            raise ActError(
                f"Instruction '{self.spec.name}' access expects access patterns."
            )
        if pattern.buffer is not None:
            self._check_buffer(pattern.buffer)
        if pattern.source is not None:
            self._check_pattern(pattern.source)

    def _check_compute_param(self, param: inspect.Parameter):
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ActError(
                f"Compute for '{self.spec.name}' only supports positional tensor arguments."
            )
        if param.default is not inspect.Parameter.empty:
            raise ActError(
                f"Compute argument '{param.name}' must not have a default value."
            )
        _check_symbol_name(param.name, "compute argument")

    def _check_compute_args(
        self,
        arg_types: Sequence[ActTensorType],
        arg_names: Sequence[str] | None = None,
    ):
        expected_buffers = self.spec.sources + self.spec.destinations
        for index, (ty, buffer) in enumerate(zip(arg_types, expected_buffers)):
            label = _compute_arg_role(
                self.spec, index, None if arg_names is None else arg_names[index]
            )
            if ty.dtype != buffer.dtype:
                raise ActError(f"{label} expects dtype {buffer.dtype}, got {ty.dtype}.")
        if len(self.spec.patterns) == 0:
            return
        for index, (ty, pattern) in enumerate(zip(arg_types, self.spec.patterns)):
            label = _compute_arg_role(
                self.spec, index, None if arg_names is None else arg_names[index]
            )
            shape = pattern.visible_shape()
            if len(shape) != ty.rank:
                raise ActError(
                    f"{label} expects rank {len(shape)} from access pattern, got {ty.rank}."
                )
            for i, (dim_expr, dim) in enumerate(zip(shape, ty.shape)):
                if (
                    dim_expr.is_static
                    and dim is not None
                    and dim_expr.static_value != dim
                ):
                    raise ActError(
                        f"{label} dimension {i} expects {dim_expr.static_value}, got {dim}."
                    )
                if not dim_expr.is_static and dim is not None:
                    raise ActError(
                        f"{label} dimension {i} must be '?' for dynamic access shape."
                    )


class ISA:
    def __init__(self, name: str):
        _check_symbol_name(name, "ISA")
        self.name = name
        self.buffers: list[BufferSpec] = []
        self.instructions: list[InstructionSpec] = []
        self._symbol_names: set[str] = set()

    def hbm(self, name: str, *, shape: Sequence[int] | int, dtype: DType) -> BufferSpec:
        return self._add_buffer(name, "hbm", slots=1, shape=shape, dtype=dtype)

    def vector(
        self, name: str, *, slots: int, shape: Sequence[int] | int, dtype: DType
    ) -> BufferSpec:
        return self._add_buffer(name, "vector", slots=slots, shape=shape, dtype=dtype)

    def scalar(self, name: str, *, slots: int, dtype: DType) -> BufferSpec:
        return self._add_buffer(name, "scalar", slots=slots, shape=(), dtype=dtype)

    def tile(
        self, name: str, *, slots: int, shape: Sequence[int] | int, dtype: DType
    ) -> BufferSpec:
        return self._add_buffer(name, "tile", slots=slots, shape=shape, dtype=dtype)

    def _add_buffer(
        self,
        name: str,
        kind: str,
        *,
        slots: int,
        shape: Sequence[int] | int,
        dtype: DType,
    ) -> BufferSpec:
        self._check_new_symbol(name, "buffer")
        if not isinstance(dtype, DType):
            raise ActError("Buffer dtype must be an Allo DType.")
        if not isinstance(slots, int) or slots <= 0:
            raise ActError("Buffer slots must be positive.")
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        if kind != "scalar" and len(shape_tuple) == 0:
            raise ActError(f"{kind} buffer requires a non-empty shape.")
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape_tuple):
            raise ActError("Buffer shape dimensions must be positive.")
        spec = BufferSpec(self, name, kind, slots, shape_tuple, dtype)
        self.buffers.append(spec)
        self._symbol_names.add(name)
        return spec

    def instruction(
        self,
        name: str,
        *,
        src: Sequence[BufferSpec] | BufferSpec,
        dst: Sequence[BufferSpec] | BufferSpec,
    ):
        self._check_new_symbol(name, "instruction")
        sources = list(_as_tuple(src))
        destinations = list(_as_tuple(dst))
        if len(sources) == 0 or len(destinations) == 0:
            raise ActError(
                "Instructions require at least one source and one destination."
            )
        for buffer in sources + destinations:
            if not isinstance(buffer, BufferSpec) or buffer.isa is not self:
                raise ActError(
                    f"Instruction '{name}' references a buffer outside ISA '{self.name}'."
                )

        def decorator(fn: Callable):
            location = callable_diagnostic_location(fn, marker=".instruction")
            try:
                self._check_new_symbol(name, "instruction")
                spec = InstructionSpec(
                    self, name, sources, destinations, location=location
                )
                builder = InstructionBuilder(spec)
                fn(builder)
                spec.validate_complete()
            except ActError as error:
                error.attach_location(location)
                raise
            self.instructions.append(spec)
            self._symbol_names.add(name)
            return spec

        return decorator

    def _check_new_symbol(self, name: str, kind: str):
        _check_symbol_name(name, kind)
        if name in self._symbol_names:
            raise ActError(f"Duplicate symbol '{name}'.")

    def emit_mlir(self, path: str | Path | None = None) -> str:
        from ..compiler.act_codegen import emit_isa

        text = emit_isa(self)
        if path is not None:
            Path(path).write_text(text)
        return text
