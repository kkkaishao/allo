import ast
import inspect
import textwrap
import warnings
import re
from collections.abc import Sequence
from typing import Literal, ParamSpec, Generic, TypeVar, Callable, overload
from dataclasses import dataclass

from ..lang.core import (
    constexpr,
    TypeBase,
    Template,
    TensorType,
    BufferType,
    DType,
    ShapedType,
    StreamType,
    DEFAULT_STREAM_DEPTH,
    unwrap_if_constexpr,
)
from ..logging import log_fatal

from .._C.ir import ModuleOp, Context

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def _eval_shape_dim(node: ast.AST, scope: dict[str, object]) -> int:
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.Name):
        if node.id not in scope:
            raise TypeError(f"Unknown shape variable '{node.id}'")
        value = unwrap_if_constexpr(scope[node.id])
    elif isinstance(node, ast.UnaryOp):
        value = _eval_shape_dim(node.operand, scope)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise TypeError(f"Unsupported shape expression: {ast.unparse(node)}")
    elif isinstance(node, ast.BinOp):
        lhs = _eval_shape_dim(node.left, scope)
        rhs = _eval_shape_dim(node.right, scope)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.FloorDiv):
            return lhs // rhs
        raise TypeError(f"Unsupported shape expression: {ast.unparse(node)}")
    else:
        raise TypeError(f"Unsupported shape expression: {ast.unparse(node)}")

    if type(value) is not int:
        raise TypeError(f"Shape expression '{ast.unparse(node)}' must be constexpr int")
    return value


def _parse_shape_dims(content: str, scope: dict[str, object]) -> list[int]:
    raw = content.strip()
    if raw == "":
        return []
    expr = ast.parse(raw, mode="eval").body
    dim_exprs = expr.elts if isinstance(expr, ast.Tuple) else [expr]
    dims = [_eval_shape_dim(dim, scope) for dim in dim_exprs]
    if any(dim < 0 for dim in dims):
        raise TypeError(f"Shape dimensions must be non-negative: [{content}]")
    return dims


def _split_annotation_groups(annotation: str) -> tuple[str, list[str]]:
    text = annotation.strip()
    match = re.match(r"[A-Za-z_]\w*", text)
    if match is None:
        raise TypeError(f"Unsupported type annotation: {annotation}")
    head = match.group(0)
    i = match.end()
    groups = []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        if text[i] != "[":
            raise TypeError(f"Unsupported type annotation: {annotation}")
        start = i + 1
        depth = 0
        while i < len(text):
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    groups.append(text[start:i].strip())
                    i += 1
                    break
                if depth < 0:
                    raise TypeError(f"Unsupported type annotation: {annotation}")
            i += 1
        if depth != 0:
            raise TypeError(f"Unsupported type annotation: {annotation}")
    return head, groups


def _split_tuple_annotation(annotation: str) -> list[str] | None:
    text = annotation.strip()
    if not text.startswith("(") or not text.endswith(")"):
        return None

    inner = text[1:-1].strip()
    if inner == "":
        return []

    parts = []
    start = 0
    bracket_depth = 0
    paren_depth = 0
    for i, ch in enumerate(inner):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "," and bracket_depth == 0 and paren_depth == 0:
            part = inner[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1

    part = inner[start:].strip()
    if part:
        parts.append(part)
    return parts if parts else None


@dataclass
class KernelOptions:
    enable_tensor: bool = False
    typing_style: Literal["cpp", "hls"] = "hls"
    fast_math: bool = False


class Kernel(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        *,
        mapping: Sequence[int | Template],
        options: KernelOptions,
        template: Sequence[Template] = (),
        template_bindings: dict[str, object] | None = None,
        definition_scope: dict[str, object] | None = None,
    ):
        assert all(isinstance(arg, Template) for arg in template)
        template_names = [arg.name for arg in template]
        if len(template_names) != len(set(template_names)):
            log_fatal("Template arguments must be unique")
        # verify the mappings
        if mapping and not all(isinstance(m, (int, Template)) for m in mapping):
            log_fatal(
                "Every mapping argument should be either a const int or a template variable"
            )
        self.fn = fn
        self.file_name = fn.__code__.co_filename
        self.func_name = fn.__name__
        self.signature = inspect.signature(fn)
        self.mapping = mapping
        self.options = options
        self.template = tuple(template)
        self.template_bindings = (
            {} if template_bindings is None else template_bindings.copy()
        )
        assert set(self.template_bindings).issubset(set(template_names))
        self.definition_scope = (
            {} if definition_scope is None else definition_scope.copy()
        )
        self.module: ModuleOp | None = None
        self.context: Context | None = None

        try:
            raw_src, begin_line = inspect.getsourcelines(fn)
        except OSError:
            warnings.warn(
                f"Could not retrieve source code for function {fn.__name__} defined in {self.file_name}. "
                "This may be due to the function being defined in an interactive environment or a dynamically generated function. "
                "Line number information may be inaccurate.",
                RuntimeWarning,
            )
            raw_src = ""
            begin_line = 1

        src = textwrap.dedent("".join(raw_src))
        match = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
        if match:
            start_pos = match.start()
            offset = src[:start_pos].count("\n")
            self.begin_line = begin_line + offset
            self.src = src[start_pos:]
        else:
            self.begin_line = begin_line
            self.src = src

        # save metadata
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__
        self.capture_scope = self._build_capture_scope()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the kernel with the active backend context, or CPU by default."""
        from ..backend.base import current_backend

        backend = current_backend()
        if backend is not None:
            return backend.call_kernel(self, *args, **kwargs)

        from ..backend import CPU

        return CPU(self).run(*args, **kwargs)

    def __getitem__(self, bindings):
        if not self.template:
            raise TypeError(f"Kernel '{self.func_name}' has no template arguments")
        if self.template_bindings:
            raise TypeError(f"Kernel '{self.func_name}' is already specialized")
        if not isinstance(bindings, tuple):
            bindings = (bindings,)
        if len(bindings) != len(self.template):
            raise TypeError(
                f"Kernel '{self.func_name}' expects {len(self.template)} template arguments, "
                f"got {len(bindings)}"
            )
        assert len(bindings) == len(self.template), (
            f"Kernel {self.func_name} expects {len(self.template)} template arguments, "
            f"got {len(bindings)}"
        )

        template_bindings = {}
        for template, value in zip(self.template, bindings):
            value = unwrap_if_constexpr(value)
            if not isinstance(value, (TypeBase, int, float)):
                raise TypeError(
                    f"Unsupported template binding for '{template.name}' in kernel "
                    f"'{self.func_name}': {type(value).__name__}"
                )
            template_bindings[template.name] = value

        # apply the mappings here
        mapping = []
        for idx, m in enumerate(self.mapping):
            if isinstance(m, int):
                mapping.append(m)
            else:
                val = template_bindings.get(m.name)
                if isinstance(val, float):
                    log_fatal(
                        f"Mapping argument ({idx}) should bind to a constant int value, but got a float"
                    )

        return Kernel(
            self.fn,
            mapping=mapping,
            options=self.options,
            template=self.template,
            template_bindings=template_bindings,
            definition_scope=self.definition_scope,
        )

    def check_templates_bounded(self):
        if not self.template:
            return
        expected = {arg.name for arg in self.template}
        if set(self.template_bindings) != expected:
            missing = ", ".join(sorted(expected - set(self.template_bindings)))
            raise TypeError(
                f"Templated kernel '{self.func_name}' must be specialized before compilation"
                + (f": missing {missing}" if missing else "")
            )

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree.body[0]

    def _build_capture_scope(self):
        fn = self.fn
        scope = self.__globals__ | self.definition_scope
        if fn.__closure__ is None:
            return scope | self.template_bindings
        nonlocals = {
            name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
        }
        return scope | nonlocals | self.template_bindings

    def get_capture_scope(self):
        return self.capture_scope

    def parse_type_annotation(
        self, annotation: object, scope: dict[str, object] | None = None
    ) -> TypeBase:
        if scope is None:
            scope = self.get_capture_scope()
        annotation = unwrap_if_constexpr(annotation)
        if annotation is constexpr:
            return constexpr
        if isinstance(annotation, Template):
            if annotation.name not in scope:
                raise TypeError(f"Template '{annotation.name}' is not bound")
            bound = unwrap_if_constexpr(scope[annotation.name])
            if not isinstance(bound, TypeBase):
                raise TypeError(
                    f"Template '{annotation.name}' must bind to a type in type annotations"
                )
            return bound
        if isinstance(annotation, TypeBase):
            if isinstance(annotation, StreamType) and annotation.is_global:
                raise TypeError(f"Unsupported type annotation: {annotation}")
            return annotation
        if isinstance(annotation, str):
            annotation = annotation.strip()
            # Case 1: direct type name, e.g. "int32"
            primitive_type = unwrap_if_constexpr(scope.get(annotation, None))
            if primitive_type is not None and isinstance(primitive_type, TypeBase):
                return primitive_type
            head, groups = _split_annotation_groups(annotation)
            if head == "Stream":
                return self._parse_stream_annotation(annotation, groups, scope)
            if head in scope:
                # Case 2: shaped type, e.g. "int32[4, 8]"
                head_value = unwrap_if_constexpr(scope[head])
                if isinstance(head_value, DType) and len(groups) == 1:
                    dtype = head_value
                    shape = _parse_shape_dims(groups[0], scope)
                    if self.options.enable_tensor:
                        return TensorType(dtype=dtype, shape=shape)
                    return BufferType(dtype=dtype, shape=shape)
        raise TypeError(f"Unsupported type annotation: {annotation}")

    def _parse_stream_base_type(
        self, annotation: str, scope: dict[str, object], prefix: str
    ) -> DType | ShapedType:
        annotation = annotation.strip()
        if not annotation:
            raise TypeError(f"{prefix} base type cannot be empty")

        scoped = unwrap_if_constexpr(scope.get(annotation, None))
        if isinstance(scoped, (DType, ShapedType)):
            return scoped

        head, groups = _split_annotation_groups(annotation)
        if head not in scope:
            raise TypeError(f"Unknown {prefix} base type '{head}'")

        head_value = unwrap_if_constexpr(scope[head])
        if isinstance(head_value, DType) and len(groups) == 0:
            return head_value
        if isinstance(head_value, DType) and len(groups) == 1:
            return BufferType(
                dtype=head_value, shape=_parse_shape_dims(groups[0], scope)
            )
        if isinstance(head_value, ShapedType) and len(groups) == 0:
            return head_value

        raise TypeError(
            f"{prefix} base type must be a scalar or buffer type, got '{annotation}'"
        )

    def _parse_stream_annotation(
        self,
        annotation: str,
        groups: list[str],
        scope: dict[str, object],
    ) -> StreamType:
        if len(groups) == 1:
            shape = []
        elif len(groups) == 2:
            if groups[1].strip() == "":
                raise TypeError("Stream[Ty][] is invalid; use Stream[Ty] instead")
            shape = _parse_shape_dims(groups[1], scope)
        else:
            raise TypeError(
                f"Unsupported Stream annotation '{annotation}', expected Stream[Ty] or Stream[Ty][shape]"
            )

        base_type = self._parse_stream_base_type(groups[0], scope, "Stream")
        return StreamType(base_type, DEFAULT_STREAM_DEPTH, shape)

    def parse_argument_annotations(self) -> list[TypeBase]:
        arg_types = []
        scope = self.get_capture_scope()
        for param in self.signature.parameters.values():
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param.name}' is missing a type annotation. Please provide an explicit type annotation for all parameters."
                )
            ty = self.parse_type_annotation(annotation, scope=scope)
            assert not (isinstance(ty, StreamType) and ty.is_global)
            arg_types.append(ty)
        return arg_types

    def parse_return_annotation(self) -> list[TypeBase]:
        annotation = self.signature.return_annotation
        annotation = unwrap_if_constexpr(annotation)
        if annotation is inspect.Signature.empty or annotation is None:
            return []
        if isinstance(annotation, str) and annotation.strip() == "None":
            return []
        scope = self.get_capture_scope()
        if isinstance(annotation, tuple):
            res_types = [
                self.parse_type_annotation(elt, scope=scope) for elt in annotation
            ]
        elif (
            isinstance(annotation, str)
            and (tuple_annotations := _split_tuple_annotation(annotation)) is not None
        ):
            res_types = [
                self.parse_type_annotation(elt, scope=scope)
                for elt in tuple_annotations
            ]
        else:
            res_types = [self.parse_type_annotation(annotation, scope=scope)]
        for ty in res_types:
            assert not (isinstance(ty, StreamType) and ty.is_global)
            if isinstance(ty, StreamType):
                raise TypeError("Stream is not allowed as a kernel return type.")
        return res_types

    def compile(self):
        if self.module is not None:
            return self.module

        from ..compiler.mlir_codegen import compile

        self.check_templates_bounded()
        arg_types = self.parse_argument_annotations()
        res_types = self.parse_return_annotation()
        return compile(self, arg_types, res_types, options=self.options)


@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def kernel(
    *template: Template,
    mapping: Sequence = (),
    options: KernelOptions = KernelOptions(),
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def kernel(
    *args,
    mapping: Sequence = (),
    options: KernelOptions = KernelOptions(),
) -> Kernel[P, R] | Callable[[Callable[P, R]], Kernel[P, R]]:
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    definition_scope = frame.f_back.f_locals.copy()
    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], Template):
        fn = args[0]
        template = ()
    else:
        fn = None
        template = args
        assert all(isinstance(arg, Template) for arg in template)

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(
            fn
        ), "The @kernel decorator can only be applied to callable objects"
        return Kernel(
            fn,
            mapping=mapping,
            options=options,
            template=template,
            definition_scope=definition_scope,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


class ConstevalFunction(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        from ..lang.core import unwrap_if_constexpr

        args = [unwrap_if_constexpr(arg) for arg in args]  # type: ignore
        kwargs = {k: unwrap_if_constexpr(v) for k, v in kwargs.items()}  # type: ignore
        return self.fn(*args, **kwargs)


def consteval(fn: Callable[P, R]) -> ConstevalFunction[P, R]:
    assert callable(
        fn
    ), "The @consteval decorator can only be applied to callable objects"
    return ConstevalFunction(fn)
