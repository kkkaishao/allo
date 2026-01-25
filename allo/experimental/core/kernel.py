# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import abc
import copy
import inspect
import re
import textwrap
import builtins
from collections import OrderedDict as _OrderedDict
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Generic,
    Callable,
    TypeVar,
    overload,
    Optional,
    ParamSpec,
    Sequence,
    Any,
    TYPE_CHECKING,
    OrderedDict,
)

from .._C.liballo import ir
from .library import Operation, BoundOperation
from .types import (
    int1,
    int8,
    int16,
    int32,
    int64,
    uint64,
    fp32,
    tuple_type,
    torch_types_to_core_types_map,
    Array,
    base_type,
    constexpr,
    constexpr_type,
    _unwrap_if_constexpr,
)

if TYPE_CHECKING:
    from .schedule import Schedule


def _build_eval_scope(scope: dict[str, Any]) -> dict[str, Any]:
    eval_scope = dict(scope)
    eval_scope.setdefault("__builtins__", builtins)
    return eval_scope


def _eval_ast_expr(expr: ast.AST, scope: dict[str, Any], file_name: str, line: int):
    code = compile(ast.Expression(copy.deepcopy(expr)), file_name, "eval")
    return eval(code, _build_eval_scope(scope), {})


class KernelParam:
    """Normalized parameter metadata used by kernel call binding/specialization."""

    def __init__(self, number, param: inspect.Parameter):
        self.number = number
        self.name = param.name
        self.param = param
        self.annotation = param.annotation
        if self.annotation is not inspect.Parameter.empty:
            if (
                not isinstance(self.annotation, base_type)
                and self.annotation is not constexpr
            ):
                raise TypeError(
                    f"Unsupported parameter type annotation: {self.annotation}. "
                    "Only core types and constexpr are supported as parameter annotations."
                )
        self.is_constexpr = self.annotation is constexpr


@dataclass
class KernelSpecialization:
    """Monomorphized call signature selected at one call site."""

    kernel: "KernelLike"
    arg_types: list[base_type]
    ret_types: list[base_type]
    bound_args: "OrderedDict[str, Any]"
    key: tuple

    @property
    def attrs(self):
        return self.kernel.attrs

    @property
    def mapping(self):
        return self.kernel.mapping


class KernelLike(abc.ABC):
    """Common protocol for global and nested kernels consumed by the code generator."""

    fn_name: str
    file_name: str
    begin_line: int
    mapping: Sequence[int]
    attrs: dict
    src: str

    @property
    @abc.abstractmethod
    def ret_types(self) -> list[base_type]:
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self) -> ast.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def get_capture_scope(self):
        raise NotImplementedError

    @abc.abstractmethod
    def bind_call_args(self, args, kwargs) -> "OrderedDict[str, Any]":
        raise NotImplementedError

    @abc.abstractmethod
    def specialize_from_call(
        self,
        bound_args: "OrderedDict[str, Any]",
    ) -> KernelSpecialization:
        raise NotImplementedError

    @property
    def identity(self) -> str:
        return f"{self.file_name}:{self.begin_line}:{self.fn_name}"

    @staticmethod
    def normalize_annotation_to_types(
        annotation,
        *,
        location: str,
        allow_constexpr: bool = False,
    ) -> list[base_type]:
        if annotation in (inspect.Signature.empty, None):
            return []
        annotations: list[Any]
        if isinstance(annotation, builtins.tuple):
            annotations = list(annotation)
        elif isinstance(annotation, list):
            annotations = list(annotation)
        else:
            annotations = [annotation]

        normalized = []
        for ann in annotations:
            if isinstance(ann, base_type):
                normalized.append(ann)
                continue
            if allow_constexpr and ann is constexpr:
                normalized.append(constexpr_type(None))
                continue
            raise TypeError(
                f"Unsupported annotation '{ann}' in {location}. "
                "Only core types are supported."
            )
        return normalized

    @staticmethod
    def _annotation_for_parameter(annotation, *, location: str):
        if annotation is inspect.Signature.empty:
            return annotation
        if annotation is constexpr:
            return constexpr
        if not isinstance(annotation, base_type):
            raise TypeError(
                f"Unsupported annotation '{annotation}' in {location}. "
                "Only core types and constexpr are supported."
            )
        return annotation

    @staticmethod
    def _type_key(ty: base_type) -> str:
        if hasattr(ty, "mangle"):
            return ty.mangle()
        return repr(ty)

    def _specialization_key(
        self,
        arg_types: Sequence[base_type],
        ret_types: Sequence[base_type],
    ) -> tuple:
        return (
            self.identity,
            builtins.tuple(self._type_key(ty) for ty in arg_types),
            builtins.tuple(self._type_key(ty) for ty in ret_types),
            builtins.tuple(self.mapping),
        )


P = ParamSpec("P")
R = TypeVar("R")


def _validate_mapping(mapping: Sequence[int]):
    for idx in mapping:
        if not isinstance(idx, int) or idx <= 0:
            raise ValueError(
                f"Mapping indices must be positive integers, but got {idx}"
            )


def _infer_arg_types(arg):
    if isinstance(arg, bool):
        return int1
    if isinstance(arg, int):
        if -(2**7) <= arg < 2**7:
            return int8
        if -(2**15) <= arg < 2**15:
            return int16
        if -(2**31) <= arg < 2**31:
            return int32
        if -(2**63) <= arg < 2**63:
            return int64
        return uint64
    if isinstance(arg, float):
        return fp32
    if isinstance(arg, builtins.tuple):
        return tuple_type([_infer_arg_types(elem) for elem in arg])
    if hasattr(arg, "dtype") and hasattr(arg, "shape"):
        shape = [s for s in arg.shape]
        dtype = torch_types_to_core_types_map.get(str(arg.dtype), None)
        if dtype is None:
            raise TypeError(f"Unsupported tensor dtype: {arg.dtype}")
        return Array(dtype, shape)
    raise TypeError(f"Unsupported argument type: {type(arg)}")


class AlloKernel:
    def __init__(self, context: ir.Context, module: ir.ModuleOp):
        self.context = context
        self.module = module

    def customize(self) -> "Schedule":
        from .schedule import Schedule

        return Schedule(self.module, self.context)

    def simulate(self):
        raise NotImplementedError()

    def synthesis(self):
        raise NotImplementedError()


class Kernel(KernelLike, Generic[P, R]):
    def __init__(
        self, fn: Callable[P, R], mapping: Optional[Sequence[int]] = None, attrs=None
    ):
        self.fn = fn
        self.file_name = fn.__code__.co_filename
        self.fn_name = fn.__name__
        self.signature = inspect.signature(fn)
        self.params = [
            KernelParam(i, p) for i, p in enumerate(self.signature.parameters.values())
        ]
        try:
            raw_src, starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise RuntimeError(
                "Failed to get source lines for kernel function. "
                "Make sure the function is not defined in an interactive environment like REPL or Jupyter Notebook."
            ) from e
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

        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__

        self.attrs = attrs or {}
        self.mapping = mapping or []
        _validate_mapping(self.mapping)
        self.arg_types = [None] * len(self.params)
        self._parsed_template: ast.Module | None = None
        self._ret_types = KernelLike.normalize_annotation_to_types(
            self.signature.return_annotation,
            location=f"return annotation of {self.fn_name}",
        )

    def parse(self):
        if self._parsed_template is None:
            tree = ast.parse(self.src)
            assert isinstance(tree, ast.Module)
            assert len(tree.body) == 1
            assert isinstance(tree.body[0], ast.FunctionDef)
            self._parsed_template = tree
        # Return a fresh copy so lowering passes can safely mutate AST nodes.
        return copy.deepcopy(self._parsed_template)

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {
            name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
        }
        return self.__globals__ | nonlocals

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AlloKernel:
        return self.make_ir(*args, **kwargs)

    @property
    def ret_type(self):
        return self._ret_types

    @ret_type.setter
    def ret_type(self, value):
        self._ret_types = value

    @property
    def ret_types(self):
        return self._ret_types

    def bind_call_args(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    def specialize_from_call(self, bound_args) -> KernelSpecialization:
        arg_types = [None] * len(self.params)
        kp_map = {kp.name: kp for kp in self.params}
        for name, value in bound_args.items():
            kp = kp_map[name]
            value = _unwrap_if_constexpr(value)
            if kp.annotation is not inspect.Parameter.empty:
                if kp.annotation is constexpr:
                    arg_types[kp.number] = constexpr_type(value)
                else:
                    arg_types[kp.number] = kp.annotation
            else:
                arg_types[kp.number] = _infer_arg_types(value)

        return KernelSpecialization(
            kernel=self,
            arg_types=arg_types,
            ret_types=self.ret_types,
            bound_args=bound_args,
            key=self._specialization_key(arg_types, self.ret_types),
        )

    def _prepare_func_type(self, *args, **kwargs):
        bound_args = self.bind_call_args(args, kwargs)
        spec = self.specialize_from_call(bound_args)
        self.arg_types = list(spec.arg_types)
        self.ret_type = list(spec.ret_types)

    def make_ir(self, *args: P.args, **kwargs: P.kwargs) -> AlloKernel:
        self._prepare_func_type(*args, **kwargs)
        from ..compiler.generator import ast_to_allo_ir

        context, module = ast_to_allo_ir(self, {})
        return AlloKernel(context, module)


class _NestedKernelParam:
    def __init__(
        self,
        number: int,
        name: str,
        annotation: ast.expr | None,
        default: ast.expr | None,
    ):
        self.number = number
        self.name = name
        self.annotation = annotation
        self.default = default


class _NestedFreeVarCollector(ast.NodeVisitor):
    def __init__(self):
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def _mark_target(self, target):
        if isinstance(target, ast.Name):
            self.stores.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._mark_target(elt)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._mark_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._mark_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._mark_target(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For):
        self._mark_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit_For(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._mark_target(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.stores.add(node.name)
        # Nested definitions have their own capture scope.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.stores.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.stores.add(node.name)


def _collect_nested_freevars(func_def: ast.FunctionDef) -> set[str]:
    """Collect lexical freevars referenced by a nested kernel body."""

    collector = _NestedFreeVarCollector()
    for arg in (
        list(func_def.args.args)
        + list(func_def.args.kwonlyargs)
        + ([] if func_def.args.vararg is None else [func_def.args.vararg])
        + ([] if func_def.args.kwarg is None else [func_def.args.kwarg])
    ):
        collector.stores.add(arg.arg)

    for default in func_def.args.defaults:
        collector.visit(default)
    for default in func_def.args.kw_defaults:
        if default is not None:
            collector.visit(default)
    for stmt in func_def.body:
        collector.visit(stmt)
    return collector.loads - collector.stores


def _validate_nested_capture(name: str, value):
    """Allow only compile-time-safe capture objects for nested kernels."""

    if isinstance(value, (constexpr, base_type, KernelLike, Operation, BoundOperation)):
        return
    if isinstance(value, ModuleType):
        return
    raise TypeError(
        f"invalid nested capture '{name}' of type {type(value).__name__}; "
        "only constexpr/type/kernel/op values are allowed. "
        "Pass runtime values explicitly as parameters."
    )


def _parse_kernel_decorator_from_ast(
    func_def: ast.FunctionDef,
    *,
    file_name: str,
    scope_provider: Callable[[], dict[str, Any]],
):
    """Parse `@kernel` / `@kernel(...)` decorators from a nested function AST node."""

    mapping: Sequence[int] = []
    attrs: dict | None = None
    found = False

    for dec in func_def.decorator_list:
        scope = scope_provider()
        if isinstance(dec, ast.Call):
            decorator = _eval_ast_expr(
                dec.func, scope, file_name, getattr(dec, "lineno", 1)
            )
            if decorator is not kernel:
                continue
            if found:
                raise TypeError(f"duplicate @kernel decorators on '{func_def.name}'")
            found = True
            if dec.args:
                raise TypeError(
                    f"@kernel on nested function '{func_def.name}' does not accept positional arguments"
                )
            kws = {
                kw.arg: _eval_ast_expr(
                    kw.value, scope, file_name, getattr(kw, "lineno", 1)
                )
                for kw in dec.keywords
            }
            mapping = kws.get("mapping", [])
            attrs = kws.get("attrs", None)
        else:
            decorator = _eval_ast_expr(dec, scope, file_name, getattr(dec, "lineno", 1))
            if decorator is kernel:
                if found:
                    raise TypeError(
                        f"duplicate @kernel decorators on '{func_def.name}'"
                    )
                found = True
                mapping = []
                attrs = None

    if not found:
        return False, [], {}
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise TypeError(
            f"@kernel attrs for nested function '{func_def.name}' must be a dict"
        )
    _validate_mapping(mapping)
    return True, list(mapping), attrs


class NestedKernel(KernelLike):
    """AST-backed kernel used for hoisted nested `@allo.kernel` definitions."""

    def __init__(
        self,
        func_def: ast.FunctionDef,
        *,
        file_name: str,
        begin_line: int,
        mapping: Optional[Sequence[int]] = None,
        attrs=None,
        scope_provider: Callable[[], dict[str, Any]],
        name_resolver: Callable[[str], Any],
    ):
        self.func_def = func_def
        self.fn_name = func_def.name
        self.file_name = file_name
        self.begin_line = begin_line
        self.mapping = list(mapping or [])
        self.attrs = attrs or {}
        self.scope_provider = scope_provider
        self.name_resolver = name_resolver
        self.src = ast.unparse(func_def)

        defaults = [None] * (len(func_def.args.args) - len(func_def.args.defaults))
        defaults += list(func_def.args.defaults)
        self._params = [
            _NestedKernelParam(i, arg.arg, arg.annotation, defaults[i])
            for i, arg in enumerate(func_def.args.args)
        ]
        self._ret_types = self._resolve_return_types()
        self.freevars = _collect_nested_freevars(func_def)

    @classmethod
    def from_ast(
        cls,
        func_def: ast.FunctionDef,
        *,
        file_name: str,
        begin_line_offset: int,
        scope_provider: Callable[[], dict[str, Any]],
        name_resolver: Callable[[str], Any],
    ):
        is_kernel, mapping, attrs = _parse_kernel_decorator_from_ast(
            func_def, file_name=file_name, scope_provider=scope_provider
        )
        if not is_kernel:
            raise TypeError(
                f"unsupported non-kernel nested function '{func_def.name}'. "
                "Only @allo.kernel nested definitions are supported."
            )
        return cls(
            func_def,
            file_name=file_name,
            begin_line=begin_line_offset + func_def.lineno - 1,
            mapping=mapping,
            attrs=attrs,
            scope_provider=scope_provider,
            name_resolver=name_resolver,
        )

    @property
    def params(self):
        return self._params

    @property
    def ret_types(self):
        return self._ret_types

    def _resolve_annotation(self, annotation):
        if annotation is None:
            return inspect.Signature.empty
        resolved = _eval_ast_expr(annotation, self.scope_provider(), self.file_name, 1)
        return self._annotation_for_parameter(
            resolved, location=f"parameter annotation of nested kernel {self.fn_name}"
        )

    def _resolve_return_types(self):
        if self.func_def.returns is None:
            return []
        resolved = _eval_ast_expr(
            self.func_def.returns,
            self.scope_provider(),
            self.file_name,
            self.begin_line,
        )
        return self.normalize_annotation_to_types(
            resolved,
            location=f"return annotation of nested kernel {self.fn_name}",
        )

    def parse(self):
        return ast.Module(body=[copy.deepcopy(self.func_def)], type_ignores=[])

    def get_capture_scope(self):
        # Resolve and validate captures at specialization time so diagnostics point
        # to the caller context instead of Python closure internals.
        scope = dict(self.scope_provider())
        for name in self.freevars:
            value = self.name_resolver(name)
            _validate_nested_capture(name, value)
            scope[name] = value
        return scope

    def bind_call_args(self, args, kwargs):
        # Match Python's call binding rules for positional/keyword/default args.
        if (
            self.func_def.args.vararg is not None
            or self.func_def.args.kwarg is not None
        ):
            raise TypeError(
                f"varargs are not supported in nested kernel '{self.fn_name}'"
            )
        ordered: "OrderedDict[str, Any]" = _OrderedDict()
        param_names = [p.name for p in self._params]
        if len(args) > len(param_names):
            raise TypeError(
                f"{self.fn_name}() takes {len(param_names)} positional arguments "
                f"but {len(args)} were given"
            )
        for i, arg in enumerate(args):
            ordered[param_names[i]] = arg
        for name, value in kwargs.items():
            if name not in param_names:
                raise TypeError(
                    f"{self.fn_name}() got an unexpected keyword argument '{name}'"
                )
            if name in ordered:
                raise TypeError(
                    f"{self.fn_name}() got multiple values for argument '{name}'"
                )
            ordered[name] = value
        scope = self.get_capture_scope()
        for p in self._params:
            if p.name in ordered:
                continue
            if p.default is None:
                raise TypeError(
                    f"{self.fn_name}() missing required argument '{p.name}'"
                )
            ordered[p.name] = _eval_ast_expr(p.default, scope, self.file_name, 1)
        return ordered

    def specialize_from_call(self, bound_args) -> KernelSpecialization:
        scope = self.get_capture_scope()
        arg_types = [None] * len(self._params)
        for p in self._params:
            value = _unwrap_if_constexpr(bound_args[p.name])
            ann = self._resolve_annotation(p.annotation)
            if ann is constexpr:
                arg_types[p.number] = constexpr_type(value)
            elif ann is inspect.Signature.empty:
                arg_types[p.number] = _infer_arg_types(value)
            else:
                arg_types[p.number] = ann
        ret_types = self.ret_types
        _ = scope  # keep parity with dynamic captures during specialization
        return KernelSpecialization(
            kernel=self,
            arg_types=arg_types,
            ret_types=ret_types,
            bound_args=bound_args,
            key=self._specialization_key(arg_types, ret_types),
        )


################
# Decorators
################
@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def kernel(
    *, mapping: Optional[Sequence[int]] = None, attrs=None
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def kernel(
    fn: Optional[Callable[P, R]] = None,
    *,
    mapping: Optional[Sequence[int]] = None,
    attrs=None,
) -> Callable[[Callable[P, R]], Kernel[P, R]] | Kernel[P, R]:

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(fn)
        return Kernel(fn, mapping=mapping, attrs=attrs)

    if fn is not None:
        return decorator(fn)

    return decorator


class ConstevalFunction(Generic[P, R]):
    """Callable wrapper executed at compile time inside kernel lowering."""

    def __init__(self, fn: Callable[P, R]):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        from .types import _unwrap_if_constexpr

        # Consteval always sees plain Python values; runtime-dependent checks are
        # performed in the generator where AST location is available.
        args = [_unwrap_if_constexpr(arg) for arg in args]
        kwargs = {k: _unwrap_if_constexpr(v) for k, v in kwargs.items()}
        return self.fn(*args, **kwargs)


def consteval(fn: Callable[P, R]) -> ConstevalFunction[P, R]:
    assert callable(fn)
    return ConstevalFunction(fn)
