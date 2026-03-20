# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap
import re
import warnings
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar, Generic, Callable, overload, Optional, Any
from collections.abc import Sequence
from ..core.types import (
    BaseType,
    Constexpr,
    int32,
    fp32,
    int1,
    int8,
    int16,
    int64,
    BufferType,
    TensorType,
    BaseValue,
    DType,
    unwrap_if_constexpr,
    torch_types_to_core_types_map,
)
from .._C.ir import ModuleOp

P = ParamSpec("P")
R = TypeVar("R")


def _infer_value_type(val, enable_tensor: bool = True) -> BaseType:
    if isinstance(val, BaseValue):
        return val.type

    if isinstance(val, int):
        if -(2**7) <= val < 2**7:
            return int8
        elif -(2**15) <= val < 2**15:
            return int16
        elif -(2**31) <= val < 2**31:
            return int32
        else:
            return int64
    elif isinstance(val, float):
        return fp32
    elif isinstance(val, bool):
        return int1
    elif hasattr(val, "dtype") and hasattr(val, "shape"):
        if not isinstance(val.shape, Sequence):
            raise TypeError(
                f"Unsupported shape type: {type(val.shape)}. Shape must be a sequence of integers."
            )
        dtype = torch_types_to_core_types_map.get(val.dtype, None)
        if dtype is None:
            raise TypeError(f"Unsupported tensor dtype: {val.dtype}")
        if enable_tensor:
            return TensorType(dtype=dtype, shape=val.shape)
        else:
            return BufferType(dtype=dtype, shape=val.shape)
    else:
        raise TypeError(f"Unsupported value type: {type(val)}")


@dataclass(frozen=True)
class CompileOptions:
    allow_implicit_type_infer: bool = False
    enable_tensor: bool = False
    module_map: dict = field(default_factory=dict)


class Kernel(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        *,
        mapping: Optional[Sequence],
        attr: dict,
        is_top: bool = False,
        options: CompileOptions = CompileOptions(),
    ):
        # setup basic fields
        self.fn = fn
        self.file_name = fn.__code__.co_filename
        self.func_name = fn.__name__
        self.signature = inspect.signature(fn)
        self.options = options

        try:
            raw_src, starting_line_number = inspect.getsourcelines(fn)
        except OSError:
            warnings.warn(
                f"Could not retrieve source code for function {fn.__name__} defined in {self.file_name}. "
                "This may be due to the function being defined in an interactive environment or a dynamically generated function. "
                "Line number information may be inaccurate.",
                RuntimeWarning,
            )
            raw_src = ""
            starting_line_number = 1
        src = textwrap.dedent("".join(raw_src))
        match = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
        if match:
            start_pos = match.start()
            offset = src[:start_pos].count("\n")
            self.begin_line = starting_line_number + offset
            self.src = src[start_pos:]
        else:
            self.begin_line = starting_line_number
            self.src = src

        self.attr = attr
        self.mapping = mapping
        self.is_top = is_top
        self.module = None
        self.context = None

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

    def __repr__(self) -> str:
        out = ""
        if self.is_top:
            out += "@accelerator\n"
        else:
            out += "@kernel\n"
        out += f"<{self.func_name} at {self.file_name}:{self.begin_line}>"
        return out

    def emit_vivado_hls(self):
        """Emit Vivado HLS compatible C++ code. Kernel must be compiled before calling this method."""
        if self.module is None:
            raise RuntimeError(
                f"Kernel {self.func_name} has not been compiled. Please compile the kernel before emitting Vivado HLS code."
            )
        else:
            from .._C.passes import emit_vivado_hls

            code = emit_vivado_hls(self.module)
            if code is None:
                raise RuntimeError(
                    f"Failed to emit Vivado HLS code for kernel {self.func_name}. Please check if the kernel is compatible with Vivado HLS and if the module was generated correctly."
                )
            return code

    def schedule(self):
        from ..schedule import Schedule

        """Get the schedule object for this kernel. Kernel must be compiled before calling this method."""
        if self.module is None:
            raise RuntimeError(
                f"Kernel {self.func_name} has not been compiled. Please compile the kernel before scheduling."
            )
        else:
            return Schedule.from_module(self.module)

    def compile(
        self,
        arg_types: Sequence[BaseType | str] = [],
        res_types: Sequence[BaseType | str] = [],
    ):
        """Compile the kernel with explicitly provided argument and return types."""
        from ..compiler.codegen import compile

        self.module, self.context = compile(
            self, arg_types=arg_types, res_types=res_types
        )
        return self.module

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        """Compile the kernel with argument types inferred at callsite and return types from annotations"""
        from ..compiler.codegen import compile

        arg_types = self.specialize_arg_types(*args, **kwargs)
        res_types = self.parse_return_annotation(self.signature.return_annotation)
        module = compile(self, arg_types, res_types)
        return module

    def specialize_arg_types(self, *args, **kwargs) -> Sequence[BaseType]:
        """Parse of infer argument types to get a list of frontend types for specialization."""
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments

        arg_types = []
        for k, v in params.items():
            annotation = self.signature.parameters[k].annotation
            if annotation is inspect.Parameter.empty:
                # TODO: try infer
                try:
                    inferred = _infer_value_type(v, self.options.enable_tensor)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to infer type for parameter '{k}' with value '{v}': {e}. Please provide an explicit type annotation or use a value with an inferrable type."
                    )
                arg_types.append(inferred)
            elif isinstance(annotation, str):
                parsed = self.parse_type_annotation(annotation)
                arg_types.append(parsed)
            elif isinstance(annotation, BaseType) or annotation == Constexpr:
                arg_types.append(annotation)
            else:
                msg = textwrap.dedent(
                    f"""
                    Unsupported type annotation for parameter '{k}': {annotation}.
                    For builtin types, such as int, float, bool, please use the corresponding allo types (e.g., int32, fp32, int1).
                    """
                )
                raise TypeError(msg)
        return arg_types

    def parse_type_annotation(self, annotation: object) -> BaseType:
        annotation = unwrap_if_constexpr(annotation)
        if annotation == Constexpr:
            return Constexpr
        if isinstance(annotation, BaseType):
            return annotation
        if isinstance(annotation, str):
            # Case 1: direct type name, e.g. "int32"
            if annotation in globals() and isinstance(globals()[annotation], BaseType):
                return globals()[annotation]
            # Case 2: buffer types, e.g. "int32[4][8]"
            buffer_match = re.fullmatch(r"([A-Za-z_]\w*)((?:\[\d+\])+$)", annotation)
            if buffer_match:
                dtype_str = buffer_match.group(1)
                shape = [
                    int(x) for x in re.findall(r"\[(\d+)\]", buffer_match.group(2))
                ]
                if dtype_str in globals() and isinstance(globals()[dtype_str], DType):
                    dtype = globals()[dtype_str]
                    if self.options.enable_tensor:
                        return TensorType(dtype=dtype, shape=shape)
                    else:
                        return BufferType(dtype=dtype, shape=shape)
        raise TypeError(f"Unsupported type annotation: {annotation}")

    def parse_return_annotation(self, annotation: object) -> list[BaseType]:
        annotation = unwrap_if_constexpr(annotation)
        if annotation is inspect.Signature.empty or annotation is None:
            return []
        if isinstance(annotation, tuple):
            return [self.parse_type_annotation(elt) for elt in annotation]
        return [self.parse_type_annotation(annotation)]


def schedule(k: Kernel):
    """Get the schedule object for a kernel. Kernel must be compiled before calling this function."""
    from ..schedule import Schedule

    if isinstance(k, Kernel):
        if k.module is None:
            raise RuntimeError(
                f"Kernel {k.func_name} has not been compiled. Please compile the kernel before scheduling."
            )
        else:
            return Schedule.from_module(k.module)
    elif isinstance(k, ModuleOp):
        return Schedule.from_module(k)
    else:
        raise TypeError(
            f"Unsupported type for scheduling: {type(k)}. Expected Kernel or ModuleOp."
        )


class ConstevalFunction(Generic[P, R]):
    def __init__(self, fn: Callable[P, R]):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        from .types import unwrap_if_constexpr

        args = [unwrap_if_constexpr(arg) for arg in args]
        kwargs = {k: unwrap_if_constexpr(v) for k, v in kwargs.items()}
        return self.fn(*args, **kwargs)


################
# Decorators
################


@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def kernel(
    *,
    mapping: Optional[Sequence] = None,
    attr: dict = {},
    options: CompileOptions = CompileOptions(),
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def kernel(
    fn: Optional[Callable[P, R]] = None,
    *,
    mapping: Optional[Sequence] = None,
    attr: dict[str, Any] = {},
    options: CompileOptions = CompileOptions(),
) -> Kernel[P, R] | Callable[[Callable[P, R]], Kernel[P, R]]:

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(fn)
        return Kernel(fn, mapping=mapping, attr=attr, options=options)

    if fn is not None:
        return decorator(fn)

    return decorator


@overload
def accelerator(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def accelerator(
    *, attr: dict = {}, options: CompileOptions = CompileOptions()
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def accelerator(
    fn: Optional[Callable[P, R]] = None,
    *,
    attr: dict[str, Any] = {},
    options: CompileOptions = CompileOptions(),
) -> Kernel[P, R] | Callable[[Callable[P, R]], Kernel[P, R]]:

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(fn)
        return Kernel(fn, mapping=None, attr=attr, options=options, is_top=True)

    if fn is not None:
        return decorator(fn)

    return decorator


def consteval(fn: Callable[P, R]) -> ConstevalFunction[P, R]:
    assert callable(fn)
    return ConstevalFunction(fn)
