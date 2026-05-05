import ast
import inspect
import textwrap
import warnings
import re
from collections.abc import Sequence
from typing import Literal, ParamSpec, Generic, TypeVar, Callable, overload
from dataclasses import dataclass

from allo.exp.lang.core import (
    constexpr,
    TypeBase,
    TensorType,
    BufferType,
    DType,
    unwrap_if_constexpr,
)

from .._C.ir import ModuleOp, Context

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def _parse_shape_dims(content: str) -> list[int]:
    raw = content.strip()
    if raw == "":
        return []
    dims = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not re.fullmatch(r"\d+", tok):
            raise TypeError(f"Unsupported type annotation: [{content}]")
        dims.append(int(tok))
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


@dataclass
class KernelOptions:
    enable_tensor: bool = False
    typing_style: Literal["cpp", "hls"] = "hls"
    fast_math: bool = False


class Kernel(Generic[P, R]):
    def __init__(
        self, fn: Callable[P, R], *, mapping: Sequence, options: KernelOptions
    ):
        self.fn = fn
        self.file_name = fn.__code__.co_filename
        self.func_name = fn.__name__
        self.signature = inspect.signature(fn)
        self.mapping = mapping
        self.options = options
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

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree.body[0]

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {
            name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
        }
        return self.__globals__ | nonlocals

    def parse_type_annotation(self, annotation: object) -> TypeBase:
        annotation = unwrap_if_constexpr(annotation)
        if annotation is constexpr:
            return constexpr
        if isinstance(annotation, TypeBase):
            return annotation
        if isinstance(annotation, str):
            annotation = annotation.strip()
            # Case 1: direct type name, e.g. "int32"
            primitive_type = self.__globals__.get(annotation, None)
            if primitive_type is not None and isinstance(primitive_type, TypeBase):
                return primitive_type
            head, groups = _split_annotation_groups(annotation)
            if head in self.__globals__:
                # Case 2: shaped type, e.g. "int32[4, 8]"
                if isinstance(self.__globals__[head], DType) and len(groups) == 1:
                    dtype = self.__globals__[head]
                    shape = _parse_shape_dims(groups[0])
                    if self.options.enable_tensor:
                        return TensorType(dtype=dtype, shape=shape)
                    return BufferType(dtype=dtype, shape=shape)
                composite_type = self.__globals__[head]
        raise TypeError(f"Unsupported type annotation: {annotation}")

    def parse_argument_annotations(self) -> list[TypeBase]:
        arg_types = []
        for param in self.signature.parameters.values():
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param.name}' is missing a type annotation. Please provide an explicit type annotation for all parameters."
                )
            arg_types.append(self.parse_type_annotation(annotation))
        return arg_types

    def parse_return_annotation(self) -> list[TypeBase]:
        annotation = self.signature.return_annotation
        annotation = unwrap_if_constexpr(annotation)
        if annotation is inspect.Signature.empty or annotation is None:
            return []
        if isinstance(annotation, tuple):
            return [self.parse_type_annotation(elt) for elt in annotation]
        return [self.parse_type_annotation(annotation)]

    def compile(self):
        if self.module is not None:
            return self.module

        from ..compiler.mlir_codegen import compile

        arg_types = self.parse_argument_annotations()
        res_types = self.parse_return_annotation()
        return compile(self, arg_types, res_types, options=self.options)


@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def kernel(
    *, mapping: Sequence = (), options: KernelOptions = KernelOptions()
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def kernel(
    fn: Callable[P, R] | None = None,
    *,
    mapping: Sequence = (),
    options: KernelOptions = KernelOptions(),
) -> Kernel[P, R] | Callable[[Callable[P, R]], Kernel[P, R]]:

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(
            fn
        ), "The @kernel decorator can only be applied to callable objects"
        return Kernel(fn, mapping=mapping, options=options)

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
