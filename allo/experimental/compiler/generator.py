# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import builtins
import contextlib
import copy
import textwrap
import hashlib
from functools import cached_property

from dataclasses import dataclass, field
from types import ModuleType
from typing import (
    Dict,
    Callable,
    Any,
    List,
    Type,
    Iterable,
    Set,
    Union,
    Sequence,
)

from .errors import CompilationError, CompileTimeAssertionFailure, VerificationError
from .. import core, dsl
from .._C.liballo import ir, scf as scf_d, cf as cf_d, ub as ub_d, allo as allo_d
from .._utils import apply_with_path, set_iterable_path
from ..core.library import BoundOperation, NO_FOLD, Operation, resolve_operation
from ..core.kernel import (
    KernelLike,
    KernelSpecialization,
    NestedKernel,
    ConstevalFunction,
)
from ..core.types import (
    base_type,
    base_value,
    constexpr,
    constexpr_type,
    tensor,
    _unwrap_if_constexpr,
    _is_allo_non_scalar_tensor,
)
from .builder import AlloOpBuilder


_condition_types = {
    bool,
    int,
    type(None),
}


class ScopeState:
    """
    Mutable frontend scope while lowering one kernel body.
    """

    def __init__(self):
        self.lscope: Dict[str, base_value] = {}
        self.local_defs: Dict[str, base_value] = {}

    def set_value(self, name: str, value):
        self.lscope[name] = value
        self.local_defs[name] = value

    @contextlib.contextmanager
    def enter_subregion(self, builder):
        liveins = self.lscope.copy()
        prev_defs = self.local_defs.copy()
        self.local_defs = {}
        insert_point = builder.save_insertion_point()
        try:
            yield liveins, prev_defs
        finally:
            builder.restore_insertion_point(insert_point)
            self.lscope = liveins
            self.local_defs = prev_defs


def flatten_values_to_ir(values: Iterable[base_value]):
    handles = []
    for value in values:
        value.append_ir_values(handles)
    return handles


def unflatten_ir_values(handles: Sequence[ir.Value], types: List[base_type]):
    cursor = 0
    for ty in types:
        value, cursor = ty.gen_proxy_value(handles, cursor)
        yield value
    assert cursor == len(handles)


def deserialize_structured_args(arg_types, handles: Sequence[ir.Value]):
    """
    Reconstruct structured frontend values from flattened MLIR function arguments.
    """

    def make_template(ty):
        if isinstance(ty, (list, builtins.tuple, core.tuple_type)):
            return core.tuple([make_template(x) for x in ty], ty)
        return core.constexpr(None)

    values = make_template(arg_types)
    cursor = 0

    def build_value(path, ty: base_type):
        nonlocal cursor
        value, cursor = ty.gen_proxy_value(handles, cursor)
        set_iterable_path(values, path, value)

    apply_with_path(arg_types, build_value)
    return values


class ContainsReturnChecker(ast.NodeVisitor):
    def __init__(self, gscope):
        self.gscope = gscope

    def _visit_stmts(self, body) -> bool:
        return any(self.visit(s) for s in body)

    def _visit_function(self, fn) -> bool:
        # No need to check within nested functions as they do not cause early return.
        return False

    def generic_visit(self, node) -> bool:
        ret = False
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ret = ret or self.visit(item)
            elif isinstance(value, ast.AST):
                ret = ret or self.visit(value)
        return ret

    def visit_Attribute(self, node: ast.Attribute) -> bool:
        if isinstance(node.value, ast.Name):
            if node.value.id in self.gscope:
                value = self.gscope[node.value.id]
                fn = getattr(value, node.attr)
                return self._visit_function(fn)
            return False
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> bool:
        if type(node.ctx) is ast.Store:
            return False
        if node.id in self.gscope:
            fn = self.gscope[node.id]
            return self._visit_function(fn)
        return False

    def visit_Return(self, node: ast.Return) -> bool:
        return True

    def visit_Assign(self, node: ast.Assign) -> bool:
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> bool:
        return False

    def visit_Module(self, node: ast.Module) -> bool:
        return self._visit_stmts(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> bool:
        return False

    def visit_If(self, node: ast.If) -> bool:
        ret = self._visit_stmts(node.body)
        if node.orelse:
            ret = ret or self._visit_stmts(node.orelse)
        return ret

    def visit_IfExp(self, node: ast.IfExp) -> bool:
        return self.visit(node.body) or self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> bool:
        return self.visit(node.func)


class ControlFlowLowering:
    """
    Delegation boundary for control-flow lowering. It keeps `CodeGenerator`
    surface thinner while preserving existing lowering logic.
    """

    def __init__(self, generator):
        self.generator = generator

    def visit_if(self, node: ast.If):
        return self.generator._visit_if_impl(node)

    def visit_grid(self, node: ast.For, grid):
        return self.generator._visit_grid_impl(node, grid)

    def visit_for(self, node: ast.For):
        return self.generator._visit_for_impl(node)

    def visit_while(self, node: ast.While):
        return self.generator._visit_while_impl(node)


class ASTFunction:
    def __init__(self, arg_types, ret_types, attrs):
        self.arg_types = arg_types
        self.ret_types = ret_types
        self.attrs = attrs

    def gen_mlir_types(
        self, builder: ir.OpBuilder, types: List[base_type]
    ) -> List[ir.Type]:
        ir_types = []
        for ty in types:
            if ty is None:
                continue
            ty.append_ir_types(builder, ir_types)
        return ir_types

    def serialize(self, builder: ir.OpBuilder) -> ir.FunctionType:
        # constexprs will not occur in function signatures
        # but constexprs will occur in proxy values (in deserialize)
        arg_ir_types = self.gen_mlir_types(builder, self.arg_types)
        ret_ir_types = self.gen_mlir_types(builder, self.ret_types)
        return ir.FunctionType.get(arg_ir_types, ret_ir_types, builder.context)

    def deserialize(self, handles: Sequence[ir.Value]):
        return deserialize_structured_args(self.arg_types, handles)


@dataclass
class _KernelCompileState:
    """Cross-generator cache and recursion guard for kernel monomorphization."""

    compiled_kernels: Dict[tuple, allo_d.KernelOp] = field(default_factory=dict)
    inflight_kernels: Set[tuple] = field(default_factory=set)


@dataclass
class _FunctionLoweringState:
    """Per-function state used by structured return lowering."""

    exit_block: ir.Block | None
    ret_types: List[base_type]
    ret_ir_types: List[ir.Type]


@dataclass
class _CarryAnalysisCacheEntry:
    """Cached loop-carry analysis result for one loop node + livein signature."""

    names: tuple[str, ...]
    carried_type_keys: tuple[str, ...]
    validation_signature: tuple[tuple[str, str, str | None], ...]


class CodeGenerator(ast.NodeVisitor):
    """
    Lower one kernel AST into Allo/MLIR with explicit scope + CFG bookkeeping.
    """

    def __init__(
        self,
        context: ir.Context,
        module: ir.ModuleOp,
        builder: AlloOpBuilder,
        prototype: ASTFunction,
        gscope,
        fn: KernelLike,
        fn_name: str,
        file_name: str,
        begin_line: int,
        options,
        module_map,
        shared_state: _KernelCompileState | None = None,
    ):
        # setup basic fields and context
        self.context = context
        self.module = module
        self.builder = builder
        self.prototype = prototype
        self.fn = fn
        self.fn_name = fn_name
        self.file_name = file_name
        self.begin_line = begin_line
        self.options = options

        # trackers
        self.scope = ScopeState()
        self.gscope = {}  # global scope
        for k, v in gscope.items():
            if isinstance(v, ModuleType):
                # module remap
                # e.g import numpy as np
                self.gscope[k] = module_map.get(v.__name__, v)
                continue
            module_name = getattr(v, "__module__", None)
            if module_name in module_map:
                # func remap
                # e.g from numpy.random import randn
                self.gscope[k] = getattr(module_map[module_name], v.__name__)
            else:
                self.gscope[k] = v
        self.module_map = module_map
        self.scf_stack = []  # stack to track scf constructs
        self.curr_fn = None
        # current AST node being visited
        self.builder.curr_node = None
        # are we visiting a consteval function?
        self.visiting_consteval_fn = False
        # are we visiting default arg expressions?
        self.visiting_default_args = False
        # find a value by name
        self.dereference_name: Callable[[str], Any] = self._define_name_lookup()
        self.loop_counters = [0]
        self.loop_name_stack = []
        self.fn_state: _FunctionLoweringState | None = None
        self.shared_state = shared_state or _KernelCompileState()
        self.hoisted_nested_kernels: Dict[str, NestedKernel] = {}
        self.last_lowered_kernel: allo_d.KernelOp | None = None
        self._stop_statement_visit = False
        self._carry_analysis_cache: Dict[tuple, _CarryAnalysisCacheEntry] = {}
        self._operation_resolve_cache: Dict[int, Operation | None] = {}
        self._global_name_allow_cache: Dict[str, bool] = {}
        self._contains_return_cache: Dict[int, bool] = {}
        self._contains_return_checker = ContainsReturnChecker(self.gscope)

        self.name_loc_as_prefix = None
        self.compile_error = self.builder.compile_error
        self.control_flow = ControlFlowLowering(self)

    @property
    def lscope(self) -> Dict[str, base_value]:
        return self.scope.lscope

    @lscope.setter
    def lscope(self, value: Dict[str, base_value]):
        self.scope.lscope = value

    @property
    def local_defs(self) -> Dict[str, base_value]:
        return self.scope.local_defs

    @local_defs.setter
    def local_defs(self, value: Dict[str, base_value]):
        self.scope.local_defs = value

    def _is_global_constexpr(self, name: str):
        marker = object()
        val = self.gscope.get(name, marker)
        if val is marker:
            return False
        return isinstance(val, constexpr) or isinstance(val, base_type)

    builtin_namespace = {
        max.__name__: dsl.max,
        min.__name__: dsl.min,
        range.__name__: core.types.range,
    }

    def _define_name_lookup(self):
        """Resolve names with kernel safety rules for globals and compile-time values."""

        def local_lookup(name: str, absent):
            return self.lscope.get(name, absent)

        def global_lookup(name: str, absent):
            val = self.gscope.get(name, absent)
            if self._is_allowed_global_name(name, val, absent):
                return val
            self.compile_error(
                textwrap.dedent(
                    f"""
                Cannot access global variable {name} from within allo kernel
                function. Allo kernels can only access global variables that
                are instantiated as constexpr (`x = allo.constexpr(42)`). Note that this is different from
                annotating a variable as constexpr (`x: allo.constexpr = 42`), which is not supported."""
                ).replace("\n", " "),
            )

        absent_marker = object()

        def name_lookup(name: str) -> Any:
            absent = absent_marker
            for lookup_function in (
                local_lookup,
                global_lookup,
                self.builtin_namespace.get,
            ):
                val = lookup_function(name, absent)
                if val is not absent:
                    return val
            raise NameError(f"name '{name}' is not defined")

        return name_lookup

    def _resolve_operation_cached(self, value):
        key = id(value)
        if key not in self._operation_resolve_cache:
            self._operation_resolve_cache[key] = resolve_operation(value)
        return self._operation_resolve_cache[key]

    def _is_allowed_global_name(self, name: str, value, absent) -> bool:
        if value is absent:
            return True
        if name in self.builtin_namespace:
            return True
        # These two flags are context-sensitive and must stay dynamic.
        if self.visiting_default_args or self.visiting_consteval_fn:
            return True
        cached = self._global_name_allow_cache.get(name, None)
        if cached is not None:
            return cached
        allowed = (
            type(value) is ModuleType
            or isinstance(value, KernelLike)
            or isinstance(value, ConstevalFunction)
            or self._resolve_operation_cached(value) is not None
            or getattr(value, "__module__", "").startswith(
                "allo.experimental.core"  # TODO: change to allo.core when merged
            )
            or isinstance(value, base_type)
            or self._is_global_constexpr(name)
        )
        self._global_name_allow_cache[name] = allowed
        return allowed

    @staticmethod
    def _type_key_for_cache(ty) -> str:
        if hasattr(ty, "mangle"):
            return ty.mangle()
        return repr(ty)

    def _build_carry_validation_signature(
        self, liveins: Dict[str, base_value], ignore: Set[str]
    ) -> tuple[tuple[str, str, str | None], ...]:
        signature: list[tuple[str, str, str | None]] = []
        for name, value in liveins.items():
            if name in ignore:
                continue
            if isinstance(value, base_value):
                signature.append(
                    (
                        name,
                        type(value).__name__,
                        self._type_key_for_cache(getattr(value, "type", None)),
                    )
                )
            else:
                signature.append((name, type(value).__name__, None))
        return tuple(signature)

    def _contains_return(self, node: ast.If) -> bool:
        key = id(node)
        cached = self._contains_return_cache.get(key, None)
        if cached is not None:
            return cached
        has_ret = self._contains_return_checker.visit(node)
        self._contains_return_cache[key] = has_ret
        return has_ret

    def _maybe_set_loc_to_name(self, val, name):
        if isinstance(val, (ir.BlockArgument, ir.Value)):
            named_loc = ir.Location(val.loc, name, self.context)
            val.loc = named_loc
        elif isinstance(val, base_value):
            handles = []
            val.append_ir_values(handles)
            for h in handles:
                named_loc = ir.Location(h.loc, name, self.context)
                h.loc = named_loc

    def set_value(self, name: str, val):
        self.scope.set_value(name, val)

    @contextlib.contextmanager
    def _name_loc_prefix(self, prefix):
        self.name_loc_as_prefix = prefix
        yield
        self.name_loc_as_prefix = None

    def _capture_scope_for_nested_kernel(self):
        """Visible symbols for nested-kernel decorator/default-expression evaluation."""

        scope = dict(self.gscope)
        scope.update(self.lscope)
        return scope

    def _register_nested_kernel_symbol(self, nested_kernel: NestedKernel):
        existing = self.hoisted_nested_kernels.get(nested_kernel.fn_name)
        if existing is not None and existing.func_def is not nested_kernel.func_def:
            self.compile_error(
                f"duplicate nested kernel definition '{nested_kernel.fn_name}'"
            )
        self.hoisted_nested_kernels[nested_kernel.fn_name] = nested_kernel
        self.set_value(nested_kernel.fn_name, nested_kernel)

    def _collect_and_hoist_nested_kernels(self, fn_def: ast.FunctionDef):
        """Collect nested `@allo.kernel` definitions from any block and hoist symbols."""

        def try_register(fn_node: ast.FunctionDef):
            try:
                nested = NestedKernel.from_ast(
                    fn_node,
                    file_name=self.file_name,
                    begin_line_offset=self.begin_line,
                    scope_provider=self._capture_scope_for_nested_kernel,
                    name_resolver=self.dereference_name,
                )
            except Exception as e:
                raise CompilationError(fn_node, str(e)) from e
            self._register_nested_kernel_symbol(nested)

        def walk(node: ast.AST):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef):
                    try_register(child)
                    continue
                walk(child)

        for stmt in fn_def.body:
            if isinstance(stmt, ast.FunctionDef):
                try_register(stmt)
                continue
            walk(stmt)

    def _is_hoisted_nested_function_node(self, node: ast.FunctionDef) -> bool:
        nested = self.hoisted_nested_kernels.get(node.name)
        return nested is not None and nested.func_def is node

    @cached_property
    def _consteval_mlir_runtime_types(self):
        """Best-effort MLIR runtime proxy type set used by consteval safety checks."""
        runtime_types = []
        for name in [
            "Value",
            "BlockArgument",
            "OpResult",
            "Operation",
            "OpState",
            "Block",
            "Region",
            "Context",
            "Type",
        ]:
            ty = getattr(ir, name, None)
            if isinstance(ty, type):
                runtime_types.append(ty)
        return tuple(runtime_types)

    def _find_runtime_dependent_consteval_value(self, value, path: str):
        """Return first runtime-dependent object found in `value`, or None."""

        if isinstance(value, constexpr):
            return None
        if isinstance(value, tensor):
            return path, value
        if isinstance(value, core.tuple):
            for i, v in enumerate(value.values):
                found = self._find_runtime_dependent_consteval_value(v, f"{path}[{i}]")
                if found is not None:
                    return found
            return None
        if isinstance(value, base_value):
            # Reject non-constexpr frontend values by default.
            return path, value
        if isinstance(value, self._consteval_mlir_runtime_types):
            return path, value
        if isinstance(value, dict):
            for k, v in value.items():
                found = self._find_runtime_dependent_consteval_value(k, f"{path}<key>")
                if found is not None:
                    return found
                found = self._find_runtime_dependent_consteval_value(
                    v, f"{path}[{repr(k)}]"
                )
                if found is not None:
                    return found
            return None
        if isinstance(value, (list, builtins.tuple, set, frozenset)):
            for i, v in enumerate(value):
                found = self._find_runtime_dependent_consteval_value(v, f"{path}[{i}]")
                if found is not None:
                    return found
            return None
        return None

    def _validate_consteval_args(self, args, kws):
        """Reject consteval arguments that may depend on runtime execution state."""

        for i, arg in enumerate(args):
            found = self._find_runtime_dependent_consteval_value(arg, f"args[{i}]")
            if found is not None:
                path, bad = found
                self.compile_error(
                    "consteval arguments must be runtime-independent; "
                    f"got {type(bad).__name__} at {path}"
                )
        for key, value in kws.items():
            found = self._find_runtime_dependent_consteval_value(
                value, f"kwargs[{key!r}]"
            )
            if found is not None:
                path, bad = found
                self.compile_error(
                    "consteval arguments must be runtime-independent; "
                    f"got {type(bad).__name__} at {path}"
                )

    def _normalize_consteval_result(self, value):
        """Mixed-wrap consteval return values into frontend compile-time objects."""

        found = self._find_runtime_dependent_consteval_value(value, "return")
        if found is not None:
            path, bad = found
            self.compile_error(
                "consteval return value must be runtime-independent; "
                f"got {type(bad).__name__} at {path}"
            )
        if isinstance(
            value,
            (
                constexpr,
                base_type,
                KernelLike,
                Operation,
                BoundOperation,
                ModuleType,
                ConstevalFunction,
            ),
        ):
            return value
        return constexpr(value)

    def _coerce_value_for_type(
        self,
        value,
        expected_type: base_type,
        *,
        context: str,
    ):
        if not isinstance(value, base_value):
            value = constexpr(value)
        if isinstance(expected_type, constexpr_type):
            return constexpr(_unwrap_if_constexpr(value))
        if isinstance(value, constexpr):
            if isinstance(expected_type, core.scalar_type):
                return self.builder.make_scalar(value.value, expected_type)
            self.compile_error(
                f"{context} expected value of type {expected_type}, "
                f"but got constexpr({value.value})"
            )
        if not isinstance(value, base_value):
            self.compile_error(
                f"{context} expected an Allo value, but got {type(value).__name__}"
            )
        if isinstance(value, tensor):
            if value.type == expected_type:
                return value
            if isinstance(expected_type, core.scalar_type) and isinstance(
                value.type, core.scalar_type
            ):
                return self.builder.scalar_cast(value, expected_type)
            self.compile_error(
                f"{context} expected {expected_type}, but got {value.type}"
            )
        if value.type != expected_type:
            self.compile_error(
                f"{context} expected {expected_type}, but got {value.type}"
            )
        return value

    def _prepare_return_operands(self, node: ast.Return, ret_value):
        """Type-check and flatten a `return` value against function annotations."""

        assert self.fn_state is not None
        expected_types = self.fn_state.ret_types
        if ret_value is None:
            actual_values = []
        elif isinstance(ret_value, core.tuple):
            actual_values = list(ret_value.values)
        else:
            actual_values = [ret_value]

        if len(expected_types) == 0:
            if ret_value is not None:
                self.compile_error(
                    "return value requires an explicit function return annotation"
                )
            return []
        if ret_value is None:
            raise CompilationError(
                node,
                f"missing return value, expected {len(expected_types)} value(s)",
            )
        if len(actual_values) != len(expected_types):
            raise CompilationError(
                node,
                f"return arity mismatch: expected {len(expected_types)} value(s), "
                f"got {len(actual_values)}",
            )
        coerced = [
            self._coerce_value_for_type(v, ty, context="return")
            for v, ty in zip(actual_values, expected_types)
        ]
        return flatten_values_to_ir(coerced)

    def _kernel_symbol_for_spec(self, spec: KernelSpecialization) -> str:
        """Stable symbol name for a monomorphized kernel specialization."""

        digest = hashlib.sha1(repr(spec.key).encode("utf-8")).hexdigest()[:12]
        return f"{spec.kernel.fn_name}_{digest}"

    def _enter_kernel_lowering(self):
        """Guard against accidental re-entrant lowering in one generator instance."""

        if self.curr_fn is not None:
            self.compile_error("nested lowering state corruption detected")

    def _finalize_kernel_lowering(self):
        """Emit final terminators and function-level return convergence logic."""

        assert self.fn_state is not None
        curr_block = self.builder.save_insertion_point().block
        if len(self.fn_state.ret_ir_types) == 0:
            if not curr_block.has_terminator():
                allo_d.ReturnOp.create(self.builder, [])
            return

        # Non-void kernels converge through a dedicated exit block carrying
        # flattened return operands as block arguments.
        if not curr_block.has_terminator():
            if len(self.fn_state.ret_ir_types) > 0:
                self.compile_error(
                    f"missing return in kernel '{self.fn_name}', "
                    "all reachable paths must return a value"
                )
            assert self.fn_state.exit_block is not None
            cf_d.BranchOp.create(self.builder, self.fn_state.exit_block, [])

        assert self.fn_state.exit_block is not None
        self.builder.set_insertion_point_to_end(self.fn_state.exit_block)
        ret_vals = [
            self.fn_state.exit_block.get_arg_at(i)
            for i in range(len(self.fn_state.ret_ir_types))
        ]
        allo_d.ReturnOp.create(self.builder, ret_vals)

    def _materialize_kernel_call_args(self, spec: KernelSpecialization):
        """Materialize runtime call operands, skipping constexpr-specialized params."""

        runtime_args = []
        for value, ty in zip(spec.bound_args.values(), spec.arg_types):
            if isinstance(ty, constexpr_type):
                continue
            runtime_args.append(
                self._coerce_value_for_type(
                    value, ty, context=f"call '{spec.kernel.fn_name}'"
                )
            )
        return runtime_args

    def _ensure_kernel_compiled(self, spec: KernelSpecialization):
        """Compile callee specialization once, with recursion detection and caching."""

        existing = self.shared_state.compiled_kernels.get(spec.key)
        if existing is not None:
            return existing
        if spec.key in self.shared_state.inflight_kernels:
            self.compile_error(
                f"recursion detected while compiling kernel '{spec.kernel.fn_name}'"
            )
        self.shared_state.inflight_kernels.add(spec.key)
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        prev_src = self.builder.src
        try:
            callee_name = self._kernel_symbol_for_spec(spec)
            callee_prototype = ASTFunction(spec.arg_types, spec.ret_types, spec.attrs)
            # Always create callee ops at module top-level to keep insertion stable
            # and avoid duplicate-attachment assertions in nested compilation.
            self.builder.set_insertion_point_to_end(self.module.body)
            self.builder.loc = ir.Location(
                spec.kernel.file_name,
                spec.kernel.begin_line,
                1,
                self.context,
            )
            self.builder.src = spec.kernel.src
            callee_generator = CodeGenerator(
                context=self.context,
                module=self.module,
                builder=self.builder,
                prototype=callee_prototype,
                gscope=spec.kernel.get_capture_scope(),
                fn=spec.kernel,
                fn_name=callee_name,
                file_name=spec.kernel.file_name,
                begin_line=spec.kernel.begin_line,
                options=self.options,
                module_map=self.module_map,
                shared_state=self.shared_state,
            )
            callee_generator.visit(spec.kernel.parse())
            callee_op = callee_generator.last_lowered_kernel
            if callee_op is None:
                raise RuntimeError(
                    f"internal error: callee '{callee_name}' was not lowered"
                )
            self.shared_state.compiled_kernels[spec.key] = callee_op
            return callee_op
        except Exception as e:
            raise CompilationError(
                self.builder.curr_node,
                f"In function '{self.fn_name}', failed to compile callee "
                f"'{spec.kernel.fn_name}': {e}",
            ) from e
        finally:
            self.builder.src = prev_src
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            self.shared_state.inflight_kernels.discard(spec.key)

    def _emit_kernel_call_and_decode_results(
        self,
        spec: KernelSpecialization,
        callee_op: allo_d.KernelOp,
    ):
        """Emit call op and decode flattened results back to frontend values."""

        runtime_args = self._materialize_kernel_call_args(spec)
        arg_values = flatten_values_to_ir(runtime_args)
        call_op = allo_d.CallOp.create(self.builder, callee_op, arg_values)
        if len(spec.ret_types) == 0:
            return None
        ret_handles = [call_op.get_result_at(i) for i in range(call_op.num_results)]
        ret_vals = list(unflatten_ir_values(ret_handles, spec.ret_types))
        if len(ret_vals) == 1:
            return ret_vals[0]
        return core.tuple(ret_vals, type=core.tuple_type(spec.ret_types))

    #################
    # AST Visitors
    #################

    def visit_compound_statements(self, stmts):
        if not isinstance(stmts, (builtins.list, builtins.tuple)):
            stmts = [stmts]
        prev_stop = self._stop_statement_visit
        self._stop_statement_visit = False
        try:
            for stmt in stmts:
                self.visit(stmt)
                # stop after explicit control-flow termination in current statement list.
                if self._stop_statement_visit:
                    break
        finally:
            self._stop_statement_visit = prev_stop

    def visit(self, node):
        if node is None:
            return
        # save last location to restore later
        last_node = self.builder.curr_node
        last_loc = self.builder.loc
        # update current node
        self.builder.curr_node = node
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            here_loc = ir.Location(
                self.file_name,
                node.lineno + self.begin_line - 1,
                node.col_offset + 1,
                self.context,
            )
            if self.name_loc_as_prefix is not None:
                here_loc = ir.Location(here_loc, self.name_loc_as_prefix, self.context)
            self.builder.loc = here_loc
        # visit node
        try:
            ret = super().visit(node)
        except CompilationError as e:
            if e.src is None and hasattr(self.fn, "src"):
                e.src = self.fn.src
                e.msg = e._format_message()
            raise e
        except Exception as e:
            raise CompilationError(node, str(e)) from e
        # restore last location
        self.builder.curr_node = last_node
        self.builder.loc = last_loc
        return ret

    def generic_visit(self, node):
        self.compile_error(
            f"Unsupported syntax '{type(node).__name__}' in allo kernel function '{self.fn_name}'",
        )

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        if self.curr_fn is not None:
            if self._is_hoisted_nested_function_node(node):
                return
            self.compile_error(
                f"unsupported non-kernel nested function '{node.name}'. "
                "Only nested @allo.kernel definitions are supported."
            )

        self._enter_kernel_lowering()
        arg_names, _ = self.visit(node.args)
        fn_ty = self.prototype.serialize(self.builder)
        curr_fn = allo_d.KernelOp.create(
            self.builder, self.fn_name, fn_ty, [], self.fn.mapping
        )
        entry_block = curr_fn.add_entry_block()

        ret_ir_types = self.prototype.gen_mlir_types(
            self.builder, self.prototype.ret_types
        )
        exit_block = None
        if len(ret_ir_types) > 0:
            self.builder.set_insertion_point_to_end(entry_block)
            # For non-void kernels, all returns branch to this block.
            exit_block = self.builder.create_block()
            for ty in ret_ir_types:
                exit_block.add_arg(ty)

        arg_handles = [curr_fn.get_arg_at(i) for i in range(curr_fn.num_args)]
        arg_values = self.prototype.deserialize(arg_handles)
        for name, val in zip(arg_names, arg_values):
            self._maybe_set_loc_to_name(val, name)
            self.set_value(name, val)

        prev_hoisted = self.hoisted_nested_kernels
        self.hoisted_nested_kernels = {}
        self._collect_and_hoist_nested_kernels(node)

        self.curr_fn = curr_fn
        self.fn_state = _FunctionLoweringState(
            exit_block=exit_block,
            ret_types=list(self.prototype.ret_types),
            ret_ir_types=ret_ir_types,
        )
        self.builder.set_insertion_point_to_start(entry_block)
        self.visit_compound_statements(node.body)
        self._finalize_kernel_lowering()

        self.builder.set_insertion_point_after(curr_fn.operation)
        self.last_lowered_kernel = curr_fn
        self.curr_fn = None
        self.fn_state = None
        self.hoisted_nested_kernels = prev_hoisted

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        kwargs_names = self.visit(node.kwarg)
        return arg_names, kwargs_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_keyword(self, node):
        return node.arg, self.visit(node.value)

    def visit_Constant(self, node):
        return constexpr(node.value)

    def visit_Expr(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return core.slice(lower, upper, step)

    def visit_Subscript(self, node):
        return self.visit_Subscript_Load(node)

    def visit_Subscript_Load(self, node):
        assert isinstance(node.ctx, ast.Load)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if isinstance(lhs, core.tuple) and isinstance(slices, constexpr):
            # access shape dims. e.g x.shape[0]
            if not isinstance(slices.value, int):
                self.compile_error("invalid index for shape access, must be an integer")
            return lhs[slices.value]
        slices = (
            core.tuple([slices]) if isinstance(slices, (tensor, constexpr)) else slices
        )
        return self.call_operation(dsl.load, [lhs, slices], {})

    def visit_Subscript_Store(self, node, value):
        assert isinstance(node.ctx, ast.Store)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        slices = (
            core.tuple([slices]) if isinstance(slices, (tensor, constexpr)) else slices
        )
        return self.call_operation(dsl.store, [lhs, slices, value], {})

    def visit_Pass(self, node):
        pass

    ############################
    # Comparisons and Operators
    ############################

    def visit_Compare(self, node):
        if not (len(node.comparators) == 1 and len(node.ops) == 1):
            self.compile_error("simultaneous multi-way comparisons are not supported")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        library_op = self._available_comparison_methods.get(type(node.ops[0]), None)
        if library_op is None:
            self.compile_error(
                f"Unsupported comparison operator '{type(node.ops[0]).__name__}' in allo kernel functions",
            )
        return self._apply_binary_method(library_op, lhs, rhs)

    _available_comparison_methods: Dict[Type[ast.cmpop], Operation] = {
        ast.Eq: dsl.eq,
        ast.NotEq: dsl.ne,
        ast.Lt: dsl.lt,
        ast.LtE: dsl.le,
        ast.Gt: dsl.gt,
        ast.GtE: dsl.ge,
    }

    def _apply_binary_method(self, library_op, lhs, rhs):
        return self.call_operation(library_op, [lhs, rhs], {})

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        library_op = self._available_binary_methods.get(type(node.op), None)
        if library_op is None:
            raise CompilationError(
                node,
                f"Unsupported binary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self._apply_binary_method(library_op, lhs, rhs)

    _available_binary_methods: Dict[Type[ast.operator], Operation] = {
        ast.Add: dsl.add,
        ast.Sub: dsl.sub,
        ast.Mult: dsl.mul,
        ast.Div: dsl.div,
        ast.FloorDiv: dsl.floordiv,
        ast.Mod: dsl.mod,
        ast.Pow: dsl.pow,
        ast.LShift: dsl.lshift,
        ast.RShift: dsl.rshift,
        ast.BitAnd: dsl.bitwise_and,
        ast.BitOr: dsl.bitwise_or,
        ast.BitXor: dsl.bitwise_xor,
    }

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        fn = self._available_unary_methods.get(type(node.op), None)
        if fn is None:
            raise CompilationError(
                node,
                f"Unsupported unary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operation(fn, [operand], {})

    _available_unary_methods: Dict[Type[ast.unaryop], Operation] = {
        ast.UAdd: dsl.pos,
        ast.USub: dsl.neg,
        ast.Not: dsl.logical_not,
        ast.Invert: dsl.invert,
    }

    def visit_BoolOp(self, node):
        library_op = self._available_boolop_methods.get(type(node.op), None)
        if library_op is None:
            raise CompilationError(
                node,
                f"Unsupported boolean operator '{type(node.op).__name__}' in allo kernel functions",
            )
        nontrivial_values = []

        for subnode in node.values:
            value = self.visit(subnode)
            if isinstance(value, constexpr):
                # constant folding
                bv = bool(_unwrap_if_constexpr(value))
                if (bv is False) and (library_op is dsl.logical_and):
                    return constexpr(False)
                if (bv is True) and (library_op is dsl.logical_or):
                    return constexpr(True)
                # otherwise constexpr has no effect, so can be skipped
            elif isinstance(value, tensor) and isinstance(value.type, core.Array):
                raise CompilationError(
                    node, "non-scalar values are not supported in boolean operations"
                )
            else:
                nontrivial_values.append(value)

        if len(nontrivial_values) == 0:
            # all values are constant folded
            if library_op == dsl.logical_and:
                return constexpr(True)
            else:
                return constexpr(False)

        while len(nontrivial_values) >= 2:
            # reduce from left to right
            rhs = nontrivial_values.pop()
            lhs = nontrivial_values.pop()
            res = self._apply_binary_method(library_op, lhs, rhs)
            nontrivial_values.append(res)

        assert len(nontrivial_values) == 1
        return nontrivial_values[0]

    _available_boolop_methods: Dict[Type[ast.boolop], Operation] = {
        ast.And: dsl.logical_and,
        ast.Or: dsl.logical_or,
    }

    ##########################
    # Control Flow Statements
    ##########################

    def visit_Break(self, node):
        raise CompilationError(
            node, "'break' statement is not supported in allo kernel functions"
        )

    def visit_Continue(self, node):
        raise CompilationError(
            node, "'continue' statement is not supported in allo kernel functions"
        )

    def visit_If(self, node):
        return self.control_flow.visit_if(node)

    def _visit_if_impl(self, node):
        cond = self.visit(node.test)

        if isinstance(cond, tensor):
            if isinstance(cond.type, core.Array):
                raise CompilationError(
                    node,
                    "'if' condition must be a scalar, please call unsplat on the condition tensor before using it in 'if' statements",
                )
            cond = self.builder.scalar_cast(cond, core.int1)
            if self._contains_return(node):
                if self.scf_stack:
                    raise CompilationError(
                        node,
                        "'return' statements inside nested `scf` constructs are not supported",
                    )
                self.visit_if_with_return(cond, node)
            else:
                self.visit_if_without_return(cond, node)
        else:
            # if constexpr branch
            assert isinstance(cond, constexpr)
            cond = _unwrap_if_constexpr(cond)
            if type(cond) not in _condition_types:
                raise CompilationError(
                    node,
                    "`if` conditionals can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in _condition_types),
                        type(cond).__name__,
                    ),
                )
            selected = node.body if cond else node.orelse
            self.visit_compound_statements(selected)

    def visit_if_with_return(self, cond: tensor, node: ast.If):
        with EnterSubRegion(self):
            curr_block = self.builder.save_insertion_point().block
            # create block will move insertion point to the new block
            # so we need to save curr_block first
            then_block = self.builder.create_block()
            else_block = self.builder.create_block()

            self.builder.set_insertion_point_to_end(curr_block)
            cf_d.CondBranchOp.create(
                self.builder,
                cond.handle,
                then_block,
                else_block,
            )
            # visit then and else blocks
            liveins = self.lscope.copy()
            then_defs, else_defs, phi_names = self.visit_then_else_block(
                node, liveins, then_block, else_block
            )
            end_if = self.builder.create_block()
            branch_types = [then_defs[name].type for name in phi_names]
            for ty in branch_types:
                tys = []
                ty.append_ir_types(self.builder, tys)
                for ir_ty in tys:
                    end_if.add_arg(ir_ty)

            branch_to_end = []
            if not then_block.has_terminator():
                self.builder.set_insertion_point_to_end(then_block)
                then_handles = flatten_values_to_ir(
                    then_defs[name] for name in phi_names
                )
                cf_d.BranchOp.create(self.builder, end_if, then_handles)
                branch_to_end.append(("then", then_handles))
            if not else_block.has_terminator():
                self.builder.set_insertion_point_to_end(else_block)
                else_handles = flatten_values_to_ir(
                    else_defs[name] for name in phi_names
                )
                cf_d.BranchOp.create(self.builder, end_if, else_handles)
                branch_to_end.append(("else", else_handles))

        if len(branch_to_end) == 0:
            self._stop_statement_visit = True
            return

        self.builder.set_insertion_point_to_start(end_if)
        res_handles = [end_if.get_arg_at(i) for i in range(end_if.num_args)]
        phi_values = unflatten_ir_values(res_handles, branch_types)
        for name, handle in zip(phi_names, phi_values):
            self.set_value(name, handle)

    def visit_if_without_return(self, cond: tensor, node: ast.If):
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            then_block = self.builder.create_block()
            else_block = self.builder.create_block() if node.orelse else None
            liveins = self.lscope.copy()
            then_defs, else_defs, phi_names = self.visit_then_else_block(
                node, liveins, then_block, else_block
            )
            then_handles = flatten_values_to_ir(then_defs[name] for name in phi_names)
            for name, val in zip(phi_names, then_handles):
                self._maybe_set_loc_to_name(val, name)

            self.builder.set_insertion_point_and_loc(ip, last_loc)
            # If there are returned values (phi nodes), scf.if MUST have an else block.
            has_else = else_block is not None or len(phi_names) > 0
            if_op = scf_d.IfOp.create(
                self.builder,
                [h.type for h in then_handles],
                cond.handle,
                has_else,
            )
            then_block.merge_before(if_op.then_block)
            self.builder.set_insertion_point_to_end(if_op.then_block)
            if_op.then_block.remove_terminator()  # remove default terminator
            # create yield
            scf_d.YieldOp.create(self.builder, then_handles)
            # create else block if exists
            if has_else:
                if node.orelse:
                    else_block.merge_before(if_op.else_block)
                self.builder.set_insertion_point_to_end(if_op.else_block)
                # create yield
                else_handles = flatten_values_to_ir(
                    else_defs[name] for name in phi_names
                )
                for name, val in zip(phi_names, else_handles):
                    self._maybe_set_loc_to_name(val, name)
                if_op.else_block.remove_terminator()  # remove default terminator
                scf_d.YieldOp.create(self.builder, else_handles)

        # update lscope with phi nodes
        res_handles = [if_op.get_result_at(i) for i in range(len(then_handles))]
        types = [then_defs[name].type for name in phi_names]
        phi_values = unflatten_ir_values(res_handles, types)
        for name, handle in zip(phi_names, phi_values):
            self.set_value(name, handle)

    def visit_then_else_block(
        self,
        node,
        liveins: Dict[str, base_value],
        then_block: ir.Block,
        else_block: ir.Block,
    ):
        # then block
        self.builder.set_insertion_point_to_start(then_block)
        self.visit_compound_statements(node.body)
        then_defs = self.local_defs.copy()  # capture ssa defs in then block
        then_vals = self.lscope.copy()  # capture what can be referenced in then block

        # else block
        if node.orelse:
            self.builder.set_insertion_point_to_start(else_block)
            # reset lscope and local_defs for else block
            self.lscope = liveins.copy()
            self.local_defs = {}
            self.visit_compound_statements(node.orelse)
            else_defs = self.local_defs.copy()  # capture ssa defs in else block
            else_vals = (
                self.lscope.copy()
            )  # capture what can be referenced in else block
        else:
            else_defs = {}
            else_vals = liveins.copy()

        # update block arguments
        names = []
        for name, value in liveins.items():
            if not isinstance(value, base_value):
                continue
            then_handles = flatten_values_to_ir([then_vals[name]])
            else_handles = flatten_values_to_ir([else_vals[name]])
            # see if then and else block refer to the same handles
            if then_handles == else_handles:
                continue
            names.append(name)
            then_defs[name] = then_vals[name]
            else_defs[name] = else_vals[name]
            # check type
            for defs, block_name in [(then_defs, "then"), (else_defs, "else")]:
                type_equal = type(defs[name]) == type(value)  # noqa: E721
                if not (type_equal and defs[name].type == value.type):
                    raise CompilationError(
                        node,
                        f"initial value for `{name}` is of type {value}, "
                        f"but the {block_name} block redefines it as {defs[name]}",
                    )
        # we don't support variables defined both in then and else blocks
        # but not in liveins. e.g.
        # if cond:
        #    z = 1
        # else:
        #    z = 2
        return then_defs, else_defs, names

    def _compute_loop_name(self):
        # Calculate hierarchical name
        current_idx = self.loop_counters[-1]
        if not self.loop_name_stack:
            loop_name = f"loop_{current_idx}"
        else:
            loop_name = f"{self.loop_name_stack[-1]}_{current_idx}"
        self.loop_counters[-1] += 1
        return loop_name

    def visit_Grid(self, node: ast.For, grid: core.grid):
        return self.control_flow.visit_grid(node, grid)

    def _visit_grid_impl(self, node: ast.For, grid: core.grid):
        if not isinstance(node.target, ast.Tuple):
            self.compile_error("loop target must be a tuple in 'grid' loops")
        if len(node.target.elts) != len(grid.starts):
            self.compile_error(
                "loop target arity must match the number of loop dimensions in 'grid' loops"
            )
        lbs = grid.starts
        ubs = grid.stops
        steps = grid.steps
        if not all(isinstance(s, constexpr) and s.value > 0 for s in steps):
            self.compile_error("loop steps must be positive integers in 'grid' loops")

        lb_tensors = []
        ub_tensors = []
        step_tensors = []

        def create_or_cast(x):
            if isinstance(x, constexpr):
                return self.builder.create_const_index(x.value)
            else:
                return self.builder.scalar_cast(x, core.index)

        for lb, ub, step in zip(lbs, ubs, steps):
            lb_tensors.append(create_or_cast(lb))
            ub_tensors.append(create_or_cast(ub))
            step_tensors.append(create_or_cast(step))

        # compute hierarchical loop name
        loop_name = self._compute_loop_name()
        self.loop_counters.append(0)
        self.loop_name_stack.append(loop_name)

        with EnterSubRegion(self):
            # create placeholders for the loop variables
            index_type = ir.IndexType.get(self.context)
            iv_placeholders = [
                ub_d.PoisonOp.create(self.builder, index_type) for _ in lbs
            ]
            for i, name in enumerate(node.target.elts):
                if not isinstance(name, ast.Name):
                    self.compile_error(
                        "loop target must be a tuple of variables in 'grid' loops"
                    )
                self.set_value(name.id, tensor(iv_placeholders[i], core.index))
            liveins = self.lscope.copy()
            names, init_handles, init_types = self._find_carries(
                node, liveins, ignore={name.id for name in node.target.elts}
            )
            if len(init_handles) > 0:
                raise NotImplementedError(
                    "reductions in 'grid' loops are not supported yet"
                )
            # create grid op
            par_op = scf_d.ParallelOp.create(
                self.builder,
                [lb.handle for lb in lb_tensors],
                [ub.handle for ub in ub_tensors],
                [step.handle for step in step_tensors],
                init_handles,
            )
            self.scf_stack.append(node)
            self.builder.set_insertion_point_to_start(par_op.body)
            # scf.parallel does not have iter_args like scf.for
            # no need to update block arguments.

            # visit loop body
            self.visit_compound_statements(node.body)
            self.scf_stack.pop()
            # create yield
            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)
            self.builder.set_insertion_point_to_end(par_op.body)
            # remove default terminator
            # par_op.body.remove_terminator()
            # scf.parallel uses scf.reduce as terminator
            # see: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfparallel-scfparallelop
            # scf_d.ReduceOp.create(self.builder, [])

            ivs = par_op.induction_vars
            for iv, placeholder in zip(ivs, iv_placeholders):
                placeholder.replace_all_uses_with(iv)
                placeholder.erase()
            for i, name in enumerate(node.target.elts):
                iv = ivs[i]
                self.set_value(name.id, tensor(iv, core.index))
                self._maybe_set_loc_to_name(iv, name.id)

        # update lscope with carried variables
        res_handles = [par_op.get_result_at(i) for i in range(len(init_handles))]
        res_values = unflatten_ir_values(res_handles, init_types)

        for name, handle in zip(names, res_values):
            self.set_value(name, handle)
            self._maybe_set_loc_to_name(handle, name)

        if grid.name:
            loop_name = grid.name
        par_op.set_attr("sym_name", self.builder.get_string_attr(loop_name))
        # Pop from stacks
        self.loop_counters.pop()
        self.loop_name_stack.pop()

    def visit_For(self, node):
        return self.control_flow.visit_for(node)

    def _visit_for_impl(self, node):
        if node.orelse:
            self.compile_error("'else' blocks in 'for' loops are not supported")
        if not isinstance(node.iter, ast.Call):
            self.compile_error("only 'for' loops over iterators are supported")
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = dict(self.visit(kw) for kw in node.iter.keywords)
        if IteratorClass is core.range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            lb = iterator.start
            ub = iterator.stop
            step = iterator.step
        elif IteratorClass is core.grid:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            return self.visit_Grid(node, iterator)
        else:
            self.compile_error(
                "unsupported iterator in 'for' loops, only 'range' and 'grid' are supported"
            )

        if not isinstance(node.target, ast.Name):
            self.compile_error("loop target must be a single variable in 'for' loops")

        if not (isinstance(step, constexpr) and step.value > 0):
            self.compile_error("loop step must be a positive integer in 'for' loops")

        params = []
        for x in [lb, ub, step]:
            if isinstance(x, constexpr):
                x = self.builder.create_const_index(x.value)
            else:
                x = self.builder.scalar_cast(x, core.index)
            params.append(x)
        lb, ub, step = params

        # compute hierarchical loop name
        loop_name = self._compute_loop_name()
        self.loop_counters.append(0)
        self.loop_name_stack.append(loop_name)

        with EnterSubRegion(self):
            # create placeholder for the loop variable
            index_type = ir.IndexType.get(self.context)
            iv_placeholder = ub_d.PoisonOp.create(self.builder, index_type)
            self.set_value(node.target.id, tensor(iv_placeholder, core.index))

            liveins = self.lscope.copy()
            names, init_handles, init_types = self._find_carries(
                node, liveins, ignore={node.target.id}
            )
            # create for op
            for_op = scf_d.ForOp.create(
                self.builder,
                lb.handle,
                ub.handle,
                step.handle,
                init_handles,
            )
            self.scf_stack.append(node)
            for_op_body = for_op.body
            self.builder.set_insertion_point_to_start(for_op_body)
            block_handles = [
                # the first argument is the loop index variable, skip it
                for_op_body.get_arg_at(i + 1)
                for i in range(len(init_handles))
            ]
            block_args = unflatten_ir_values(block_handles, init_types)
            for name, val in zip(names, block_args):
                self._maybe_set_loc_to_name(val, name)
                self.set_value(name, val)
            # visit loop body
            self.visit_compound_statements(node.body)
            self.scf_stack.pop()
            # create yield
            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)
            self.builder.set_insertion_point_to_end(for_op_body)
            # MLIR will auto-add a terminator if missing,
            # but actually we don't want that here
            for_op_body.remove_terminator()
            scf_d.YieldOp.create(self.builder, yield_handles)
            assert for_op_body.parent_region.size() == 1

            # update induction variable to the actual one
            iv = for_op.induction_var
            iv_placeholder.replace_all_uses_with(iv)
            iv_placeholder.erase()
            self.set_value(node.target.id, tensor(iv, core.index))
            self._maybe_set_loc_to_name(iv, node.target.id)

        # update lscope with carried variables
        res_handles = [for_op.get_result_at(i) for i in range(len(init_handles))]
        res_values = unflatten_ir_values(res_handles, init_types)
        for name, handle in zip(names, res_values):
            self.set_value(name, handle)
            self._maybe_set_loc_to_name(handle, name)

        if iterator.name:
            loop_name = iterator.name
        for_op.set_attr("sym_name", self.builder.get_string_attr(loop_name))
        # Pop from stacks
        self.loop_counters.pop()
        self.loop_name_stack.pop()

    def _find_carries(self, node, liveins: Dict[str, base_value], ignore: Set[str]):
        ignore_frozen = frozenset(ignore)
        cache_key = (id(node), ignore_frozen, id(node.body))
        validation_signature = self._build_carry_validation_signature(liveins, ignore)
        cached = self._carry_analysis_cache.get(cache_key)
        if cached is not None and cached.validation_signature == validation_signature:
            init_types = []
            init_handles = []
            names = list(cached.names)
            current_type_keys = []
            cache_valid = True
            for name in names:
                livein_val = liveins.get(name)
                if not isinstance(livein_val, base_value):
                    cache_valid = False
                    break
                current_type_keys.append(self._type_key_for_cache(livein_val.type))
                init_types.append(livein_val.type)
                init_handles.extend(flatten_values_to_ir([livein_val]))
            if cache_valid and tuple(current_type_keys) == cached.carried_type_keys:
                self.lscope = liveins.copy()
                self.local_defs = {}
                return names, init_handles, init_types

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        # create a dummy block to analyze carries
        block = self.builder.create_block()
        self.builder.set_insertion_point_to_start(block)
        # dry visit
        old_loop_counters = self.loop_counters[:]
        old_loop_name_stack = self.loop_name_stack[:]
        self.scf_stack.append(node)
        self.visit_compound_statements(node.body)
        self.scf_stack.pop()
        # restore states
        self.loop_counters = old_loop_counters
        self.loop_name_stack = old_loop_name_stack
        block.erase()
        self.builder.set_insertion_point_and_loc(ip, last_loc)

        init_types = []
        init_handles = []
        names = []

        for name, livein_val in liveins.items():
            if name in ignore:
                continue
            if isinstance(livein_val, base_value):
                loop_val = self.lscope[name]
                self._verify_loop_carried_variable(node, name, livein_val, loop_val)

                livein_handles = flatten_values_to_ir([livein_val])
                loop_handles = flatten_values_to_ir([loop_val])
                # if they point to different values, we need to carry it
                if livein_handles != loop_handles:
                    names.append(name)
                    init_types.append(livein_val.type)
                    init_handles.extend(livein_handles)
            else:
                assert name not in self.local_defs

        # restore local state
        self.lscope = liveins.copy()
        self.local_defs = {}
        carried_type_keys = tuple(self._type_key_for_cache(ty) for ty in init_types)
        self._carry_analysis_cache[cache_key] = _CarryAnalysisCacheEntry(
            names=tuple(names),
            carried_type_keys=carried_type_keys,
            validation_signature=validation_signature,
        )
        return names, init_handles, init_types

    def visit_While(self, node):
        return self.control_flow.visit_while(node)

    def _visit_while_impl(self, node):
        loop_name = self._compute_loop_name()
        # Push to stacks for children
        self.loop_counters.append(0)
        self.loop_name_stack.append(loop_name)

        with EnterSubRegion(self):
            liveins = self.lscope.copy()
            names, init_handles, init_frontend_types = self._find_carries(
                node, liveins, ignore=set()
            )
            init_types = [h.type for h in init_handles]
            # create while op
            while_op = scf_d.WhileOp.create(self.builder, init_types, init_handles)

            # create before block for condition
            before_block = self.builder.create_block_in_region(
                while_op.before,
                init_types,
            )
            self.builder.set_insertion_point_to_start(before_block)
            block_args = [before_block.get_arg_at(i) for i in range(len(init_types))]
            condition_args = unflatten_ir_values(block_args, init_frontend_types)
            for name, val in zip(names, condition_args):
                self._maybe_set_loc_to_name(val, name)
                self.set_value(name, val)

            # visit condition
            cond = self.visit(node.test)
            self.builder.set_insertion_point_to_end(before_block)
            assert isinstance(cond, tensor)
            # create condition op
            scf_d.ConditionOp.create(self.builder, cond.handle, block_args)

            # create after block for loop body
            after_block = self.builder.create_block_in_region(
                while_op.after, init_types
            )
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = [after_block.get_arg_at(i) for i in range(len(init_types))]
            body_args = unflatten_ir_values(body_handles, init_frontend_types)
            for name, val in zip(names, body_args):
                self._maybe_set_loc_to_name(val, name)
                self.set_value(name, val)
            # visit loop body
            self.scf_stack.append(node)
            self.visit_compound_statements(node.body)
            self.scf_stack.pop()

            # create yield
            yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)
            scf_d.YieldOp.create(self.builder, yield_handles)

        # update lscope with carried variables
        res_handles = [while_op.get_result_at(i) for i in range(len(init_types))]
        res_values = unflatten_ir_values(res_handles, init_frontend_types)
        for name, handle in zip(names, res_values):
            self.set_value(name, handle)
            self._maybe_set_loc_to_name(handle, name)

        while_op.set_attr("sym_name", self.builder.get_string_attr(loop_name))
        # Pop from stacks
        self.loop_counters.pop()
        self.loop_name_stack.pop()

        if node.orelse:
            raise CompilationError(
                node, "'else' clauses on 'while' loops are not supported"
            )

    def visit_IfExp(self, node):
        cond = self.visit(node.test)

        if isinstance(cond, tensor):
            if _is_allo_non_scalar_tensor(cond):
                raise CompilationError(
                    node,
                    "non-scalar tensor conditions are not supported in conditional expressions",
                )
            cond = self.builder.scalar_cast(cond, core.int1)
            with EnterSubRegion(self):
                # note that if exp cannot define new variables
                ip, last_loc = self.builder.get_insertion_point_and_loc()

                then_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(then_block)
                then_val = self.visit(node.body)

                else_block = self.builder.create_block()
                self.builder.set_insertion_point_to_start(else_block)
                else_val = self.visit(node.orelse)

                def normalize_branch(branch, peer):
                    if isinstance(branch, tensor):
                        return branch
                    if isinstance(branch, constexpr):
                        if isinstance(peer, tensor) and isinstance(
                            peer.type, core.scalar_type
                        ):
                            return self.builder.make_scalar(branch.value, peer.type)
                        self.compile_error(
                            "dynamic conditional expressions require tensor-valued branches"
                        )
                    self.compile_error(
                        f"unsupported branch value type in conditional expression: {type(branch)}"
                    )

                then_val = normalize_branch(then_val, else_val)
                else_val = normalize_branch(else_val, then_val)

                self.builder.set_insertion_point_and_loc(ip, last_loc)
                if then_val.type != else_val.type:
                    raise CompilationError(
                        node,
                        "the two branches of conditional expression must have the same type",
                    )
                res_types = []
                then_val.type.append_ir_types(self.builder, res_types)
                if_op = scf_d.IfOp.create(self.builder, res_types, cond.handle, True)
                then_block.merge_before(if_op.then_block)
                self.builder.set_insertion_point_to_end(if_op.then_block)
                scf_d.YieldOp.create(self.builder, [then_val.handle])

                else_block.merge_before(if_op.else_block)
                self.builder.set_insertion_point_to_end(if_op.else_block)
                scf_d.YieldOp.create(self.builder, [else_val.handle])
                return tensor(if_op.get_result_at(0), then_val.type)
        else:
            # if constexpr branch
            cond = _unwrap_if_constexpr(cond)
            if type(cond) not in _condition_types:
                raise CompilationError(
                    node,
                    "conditional expressions can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in _condition_types),
                        type(cond).__name__,
                    ),
                )
            if cond:
                return self.visit(node.body)
            else:
                return self.visit(node.orelse)

    @staticmethod
    def _verify_loop_carried_variable(
        node, name: str, livein_val: base_value, loop_val: base_value
    ):
        if type(loop_val) is not type(livein_val):
            raise CompilationError(
                node,
                f"Loop carried variable {name} changed type, was {type(loop_val)} but is now {type(livein_val)}",
            )
        if isinstance(loop_val, tensor) and loop_val.type != livein_val.type:
            raise CompilationError(
                node,
                f"Loop-carried variable {name} has initial type {livein_val.type} "
                f"but is re-assigned to {loop_val.type} in loop! "
                f"Please make sure that the type stays consistent.",
            )

    #####################
    # Assignments
    #####################

    def visit_AnnAssign(self, node):
        annotation = self.visit(node.annotation)
        target = self.visit(node.target)
        value = self.visit(node.value) if node.value else None

        if annotation == constexpr:
            if target in self.lscope:
                raise CompilationError(
                    node, f"redefinition of constexpr variable '{target}'"
                )
            value = constexpr(value)
            self.lscope[target] = value
            return self.lscope[target]

        # default behavior: treat as normal assignment
        return self.visit_Assign(node)

    def visit_Assign(self, node: Union[ast.Assign, ast.AnnAssign]):
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        if len(targets) != 1:
            raise CompilationError(
                node, "multiple assignment targets are not supported"
            )
        target = targets[0]
        if isinstance(target, ast.Name):
            with self._name_loc_prefix(target.id):
                value = self.visit(node.value)
        else:
            value = self.visit(node.value)
        self._do_assignment(target, value)

    def _do_assignment(self, target, value):
        assert isinstance(target.ctx, ast.Store)
        if isinstance(target, ast.Subscript):
            return self.visit_Subscript_Store(target, value)
        if isinstance(target, ast.Tuple):
            assert isinstance(value, core.tuple)
            for i, elt in enumerate(target.elts):
                self._do_assignment(elt, value.values[i])
            return None
        if isinstance(target, ast.Attribute):
            self.compile_error(
                "assignment to attributes is not supported, use a function call instead"
            )
        assert isinstance(target, ast.Name)
        if isinstance(value, constexpr):
            target_val = self.lscope.get(target.id, None)
            if target_val is not None:
                assert isinstance(target_val, tensor)
                if isinstance(target_val.type, core.scalar_type):
                    value = self.builder.make_scalar(value.value, target_val.type)
                if isinstance(target_val.type, core.Array):
                    if isinstance(value, core.tuple):
                        raise NotImplementedError()
                    ret = self.builder.fill_array(target_val, value)
                    value = ret if ret is not None else target_val

        self.set_value(self.visit(target), value)
        return None

    def visit_AugAssign(self, node):
        lhs = self._load_lvalue_as_rvalue(node.target)
        rhs = self.visit(node.value)
        library_op = self._available_binary_methods.get(type(node.op), None)
        if library_op is None:
            raise CompilationError(
                node,
                f"Unsupported binary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        result = self._apply_binary_method(library_op, lhs, rhs)
        self._do_assignment(node.target, result)
        return result

    def _load_lvalue_as_rvalue(self, target):
        if isinstance(target, ast.Name):
            return self.dereference_name(target.id)
        if isinstance(target, ast.Subscript):
            load_target = copy.copy(target)
            load_target.ctx = ast.Load()
            return self.visit_Subscript_Load(load_target)
        if isinstance(target, ast.Attribute):
            self.compile_error(
                "assignment to attributes is not supported, use a function call instead"
            )
        if isinstance(target, ast.Tuple):
            self.compile_error("augmented assignment on tuple targets is not supported")
        self.compile_error(
            f"unsupported assignment target '{type(target).__name__}' in augmented assignment"
        )

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        elts = [self.visit(e) for e in node.elts]
        is_constexpr = all(
            isinstance(elt, constexpr)
            or (isinstance(elt, core.tuple) and elt.is_constexpr)
            for elt in elts
        )
        return core.tuple(elts, is_constexpr=is_constexpr)

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id
        return self.dereference_name(node.id)

    def visit_List(self, node):
        ctx = self.visit(node.ctx)
        assert ctx is None
        elts = core.tuple([self.visit(e) for e in node.elts])
        return elts

    def visit_ListComp(self, node):
        if len(node.generators) != 1:
            raise CompilationError(node, "nested list comprehensions are not supported")
        comp = node.generators[0]
        iter = self.visit(comp.iter)
        if not isinstance(iter, core.tuple):
            raise CompilationError(
                node, "only tuple iteration is supported in list comprehensions"
            )

        results = []
        for item in iter:
            if not isinstance(comp.target, ast.Name):
                raise CompilationError(
                    node,
                    "only simple variable targets are supported in list comprehensions",
                )
            self.set_value(comp.target.id, item)
            results.append(self.visit(node.elt))
        return core.tuple(results)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if isinstance(lhs, ModuleType):
            # follow module_map until reaching fixed-point:
            while (name := lhs.__name__) in self.module_map:
                lhs = self.module_map[name]
                if lhs.__name__ == name:
                    break
        if isinstance(lhs, tensor) and node.attr == "shape":
            # shape is stored without tuple type
            # we need to wrap it in a tuple to make it work with unpacking and indexing
            return core.tuple([constexpr(s) for s in lhs.shape], is_constexpr=True)
        return getattr(lhs, node.attr)

    def visit_Return(self, node):
        if self.fn_state is None:
            self.compile_error("return is only allowed inside allo kernels")
        if self.scf_stack:
            raise CompilationError(
                node,
                "unsupported early-return in loops (`for`/`while`/`grid`); "
                "return is only supported at function scope and in `if` branches",
            )
        ret_value = self.visit(node.value) if node.value is not None else None
        ret_operands = self._prepare_return_operands(node, ret_value)
        self.builder.set_insertion_point_to_end(
            self.builder.save_insertion_point().block
        )
        if self.fn_state.exit_block is None:
            # void kernel: return directly, no synthetic exit block.
            allo_d.ReturnOp.create(self.builder, [])
        else:
            cf_d.BranchOp.create(self.builder, self.fn_state.exit_block, ret_operands)
        # Stop visiting remaining statements in this lexical list to avoid
        # inserting dead extra branches/terminators after an explicit return.
        self._stop_statement_visit = True

    #################
    # Function Call
    #################
    def visit_Call(self, node):
        fn = _unwrap_if_constexpr(self.visit(node.func))
        static_fn = self.statically_implemented_functions.get(fn, None)
        if static_fn is not None:
            return static_fn(self, node)

        self.visiting_consteval_fn = isinstance(fn, ConstevalFunction)
        try:
            # build kwargs and args
            kws = dict(self.visit(kw) for kw in node.keywords)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    arg = self.visit(arg.value)
                    assert isinstance(arg, core.tuple)
                    args.extend(arg.values)
                else:
                    ret = self.visit(arg)
                    args.append(ret)
        finally:
            self.visiting_consteval_fn = False
        return self.call_function(fn, args, kws)

    def call_function(self, fn, args, kws):
        """Dispatch callable targets across kernel/op/type/consteval frontends."""

        if isinstance(fn, KernelLike):
            return self.call_kernel(fn, args, kws)
        op = self._resolve_operation_cached(fn)
        if op is not None:
            return self.call_operation(op, args, kws)
        if isinstance(fn, base_type):
            return self.call_function(fn.__call__, args, kws)
        if isinstance(fn, ConstevalFunction):
            # TODO: pass more target info to consteval functions
            try:
                self._validate_consteval_args(args, kws)
                ret = fn(*args, **kws)
                return self._normalize_consteval_result(ret)
            except CompilationError:
                raise
            except Exception as e:
                self.compile_error(
                    f"error when calling consteval function '{fn.__name__}': {e}"
                )
        fn_mod = getattr(fn, "__module__", type(fn).__module__)
        fn_name = getattr(fn, "__name__", type(fn).__name__)
        self.compile_error(
            f"only allo kernel functions, operations, and consteval functions can be called in allo kernel functions, but got {fn_mod}.{fn_name}"
        )

    def call_operation(
        self, fn: Operation | BoundOperation, args, kws, ty=None
    ):  # noqa: ARG002
        if isinstance(fn, BoundOperation):
            args = fn.bind_args(args)
            fn = fn.op
        err_msg = fn.run_validate(*args, **kws)
        if err_msg:
            raise CompilationError(
                self.builder.curr_node,
                f"Preconditions not met for operation '{fn.__name__}': {err_msg}",
            )
        folded = fn.run_const_fold(*args, **kws)
        if folded is not NO_FOLD:
            return folded
        # save states
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        ret = fn.run_lower(self.builder, *args, **kws)
        # restore states
        self.builder.set_insertion_point_and_loc(ip, last_loc)
        return ret

    def call_kernel(self, fn: KernelLike, args, kws):
        """Lower/call a kernel specialization and decode structured return values."""

        try:
            bound_args = fn.bind_call_args(args, kws)
            spec = fn.specialize_from_call(bound_args)
        except Exception as e:
            raise CompilationError(
                self.builder.curr_node,
                f"failed to bind kernel call '{fn.fn_name}': {e}",
            ) from e
        callee_op = self._ensure_kernel_compiled(spec)
        return self._emit_kernel_call_and_decode_results(spec, callee_op)

    ##################
    # Builtins
    ##################

    @staticmethod
    def static_executor(python_fn):
        def ret(self, node: ast.Call):
            kws = {
                name: _unwrap_if_constexpr(value)
                for name, value in (self.visit(keyword) for keyword in node.keywords)
            }
            args = [_unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
            return constexpr(python_fn(*args, **kws))

        return ret

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError(
                "`static_assert` requires one or two positional arguments only"
            )

        passed = _unwrap_if_constexpr(self.visit(node.args[0]))
        if not isinstance(passed, bool):
            raise NotImplementedError(
                "Assertion condition could not be determined at compile-time. Make sure that it depends only on `constexpr` values"
            )
        if not passed:
            if arg_count == 1:
                message = ""
            else:
                try:
                    message = self.visit(node.args[1])
                except Exception as e:
                    message = "<failed to evaluate assertion message: " + repr(e) + ">"

            raise CompileTimeAssertionFailure(
                node, _unwrap_if_constexpr(message), self.fn.src
            )
        return None

    statically_implemented_functions = {
        # dsl.static_assert: execute_static_assert,
        print: static_executor(print),
        int: static_executor(int),
        len: static_executor(len),
    }


class EnterSubRegion:
    """Scoped helper that snapshots/restores frontend symbol state + insertion point."""

    def __init__(self, generator: CodeGenerator):
        self.generator = generator

    def __enter__(self):
        self.ctx = self.generator.scope.enter_subregion(self.generator.builder)
        self.ctx.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.ctx.__exit__(*args, **kwargs)


def ast_to_allo_ir(fn: KernelLike, module_map=None, options=None):
    """Entry point: lower one kernel-like object to a verified Allo IR module."""

    if module_map is None:
        module_map = {}
    prototype = ASTFunction(fn.arg_types, fn.ret_types, fn.attrs)
    # prepare context
    context = ir.Context()
    context.load_dialects()
    # initialize builder
    builder = AlloOpBuilder(context)
    builder.src = fn.src
    builder.loc = ir.Location(fn.file_name, fn.begin_line, 1, context)
    module = ir.ModuleOp.create(builder)
    module.set_attr("allo.module", builder.get_unit_attr())
    builder.module = module
    builder.set_insertion_point_to_end(module.body)

    # start code generation
    shared_state = _KernelCompileState()
    generator = CodeGenerator(
        context=context,
        module=module,
        builder=builder,
        prototype=prototype,
        gscope=fn.get_capture_scope(),
        file_name=fn.file_name,
        begin_line=fn.begin_line,
        fn=fn,
        fn_name=fn.fn_name,
        module_map=module_map,
        options=options,
        shared_state=shared_state,
    )
    generator.visit(fn.parse())
    module = generator.module
    if not module.verify():
        print(module)
        raise VerificationError(
            f"In function '{fn.fn_name}': generated Allo IR module verification failed."
        )
    # transfer ownership of context to the caller
    return context, module
