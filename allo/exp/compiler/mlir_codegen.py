import ast

import builtins
import copy
from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Type
from types import ModuleType

from .._C.ir import Context, ModuleOp, Location, Value, FunctionType, Block
from .._C.func import FuncOp, ReturnOp, CallOp
from .._C.cf import BranchOp, CondBranchOp
from .._C.scf import (
    IfOp,
    ForOp,
    YieldOp as SCFYieldOp,
    WhileOp,
    ConditionOp,
    ParallelOp,
)
from .._C.arith import SelectOp
from .._C.ub import PoisonOp
from .builder import AlloOpBuilder
from ..lang.kernel import ConstevalFunction, Kernel, KernelOptions
from ..lang.kernel import kernel as kernel_decorator
from ..lang.core import (
    ConstexprValue,
    AlloValue,
    TypeBase,
    DType,
    ShapedType,
    Range,
    Grid,
    ConstexprType,
    constexpr,
    unwrap_if_constexpr,
    index,
    bool,
)
from ..lang.operator import Operator, BoundOperator, NO_FOLD
from ..operators import arith as arith_ops, memory as mem_ops
from .errors import CompilationError, StaticAssertionError


def generate_function_type(
    context: Context, arg_types: Sequence[TypeBase], res_types: Sequence[TypeBase]
) -> FunctionType:
    mlir_arg_types = []
    for ty in arg_types:
        if isinstance(ty, ConstexprType):
            continue
        mlir_arg_types.append(ty.materialize(context))
    mlir_res_types = []
    for ty in res_types:
        if isinstance(ty, ConstexprType):
            continue
        mlir_res_types.append(ty.materialize(context))
    return FunctionType.get(mlir_arg_types, mlir_res_types, context)


class ReturnPlacementChecker(ast.NodeVisitor):
    def __init__(self, src: str, file_name: str, begin_line: int):
        self.src = src
        self.file_name = file_name
        self.begin_line = begin_line
        self.function_depth = 0
        self.loop_depth = 0
        self.if_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.function_depth > 0:
            return
        self.function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.function_depth -= 1

    def visit_Return(self, node: ast.Return):
        if self.loop_depth > 0:
            raise CompilationError(
                self.src,
                "'return' is not supported inside loops (for/grid/while).",
                node,
                file_name=self.file_name,
                begin_line=self.begin_line,
            )
        if self.if_depth > 1:
            raise CompilationError(
                self.src,
                "'return' is not supported inside nested 'if' statements.",
                node,
                file_name=self.file_name,
                begin_line=self.begin_line,
            )

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_If(self, node: ast.If):
        self.if_depth += 1
        self.generic_visit(node)
        self.if_depth -= 1


@dataclass(frozen=True)
class NestedKernelSymbol:
    name: str
    node: ast.FunctionDef
    owner_func_name: str


class MLIRCodeGenerator(ast.NodeVisitor):
    def __init__(
        self,
        context: Context,
        module: ModuleOp,
        builder: AlloOpBuilder,
        kernel: Kernel,
        arg_types: Sequence[TypeBase],
        res_types: Sequence[TypeBase],
        func_name: str,
        file_name: str,
        begin_line: int,
        options: KernelOptions,
        gscope: dict,
        callee_context: dict[str, object] | None = None,
        fscope: dict[str, object] | None = None,
        closure_scope: dict[str, object] | None = None,
        forbidden_closure_scope: dict[str, object] | None = None,
        active_kernel_calls: list[str] | None = None,
    ):
        # setup basic info
        self.context = context
        self.module = module
        self.builder = builder
        self.func_name = func_name
        self.file_name = file_name
        self.begin_line = begin_line
        self.options = options
        self.kernel = kernel
        self.arg_types = arg_types
        self.res_types = res_types

        # trackers
        self.gscope = gscope
        self.lscope: dict[str, object] = (
            {} if callee_context is None else callee_context.copy()
        )
        self.fscope: dict[str, object] = {} if fscope is None else fscope.copy()
        self.closure_scope = {} if closure_scope is None else closure_scope.copy()
        self.forbidden_closure_scope = (
            {} if forbidden_closure_scope is None else forbidden_closure_scope.copy()
        )
        self._active_kernel_calls = (
            [] if active_kernel_calls is None else active_kernel_calls
        )
        self._kernel_call_counter = 0
        self._kernel_base_names: set[str] = set()
        self._entry_function_visited = False
        self.scf_stack = []  # control flow stack
        self.curr_func = None
        self.generated_func = None
        self.name_loc_prefix = None
        self.lookup = self._define_name_lookup()
        self.visiting_consteval_fn = False
        self.visiting_default_args = False
        self.dry_run_loop_analysis = False
        self.block_terminated = False
        self.has_explicit_return_annotation = False

        self.compile_error = self.builder.compile_error

    builtin_namespace = {
        "range": Range,
        "max": arith_ops.max,
        "min": arith_ops.min,
    }

    def _define_name_lookup(self):
        def local_lookup(name: str, absent):
            val = self.lscope.get(name, absent)
            if val is not absent:
                return val
            val = self.fscope.get(name, absent)
            if val is not absent:
                return val
            return absent

        def closure_lookup(name: str, absent):
            val = self.closure_scope.get(name, absent)
            if val is not absent:
                return val
            val = self.fscope.get(name, absent)
            if val is not absent:
                return val
            if name in self.forbidden_closure_scope:
                captured = self.forbidden_closure_scope[name]
                captured_ty = (
                    captured.type
                    if isinstance(captured, AlloValue)
                    else type(captured).__name__
                )
                return self.compile_error(
                    f"Invalid closure capture '{name}' in kernel '{self.func_name}'. "
                    "Only constexpr values, kernels, types, consteval functions, operators, and modules can be captured from outer scope, "
                    f"but got '{captured_ty}'."
                )
            return absent

        def global_lookup(name: str, absent):
            val = self.gscope.get(name, absent)
            if self._is_allowed_global_var(name, val, absent):
                if self._is_python_scalar_const(val):
                    return ConstexprValue(val)
                return val
            return absent

        absent = object()

        def lookup(name: str):
            for lookup_fn in (
                local_lookup,
                closure_lookup,
                global_lookup,
                self.builtin_namespace.get,
            ):
                val = lookup_fn(name, absent)
                if val is not absent:
                    return val
            return self.compile_error(
                f"Name '{name}' is not defined in the current scope"
            )

        return lookup

    def _is_global_constexpr(self, name: str):
        marker = object()
        val = self.gscope.get(name, marker)
        if val is marker:
            return False
        return isinstance(val, ConstexprValue)

    @staticmethod
    def _is_python_scalar_const(val: object):
        return isinstance(val, (builtins.int, builtins.float))

    def _is_allowed_static_value(self, name: str, val: object):
        return (
            name in self.builtin_namespace
            or isinstance(val, ModuleType)
            or isinstance(val, Kernel)
            or isinstance(val, NestedKernelSymbol)
            or isinstance(val, (Operator, BoundOperator))
            or val in (Range, Grid)
            or isinstance(val, TypeBase)
            or isinstance(val, ConstexprValue)
            or isinstance(val, ConstevalFunction)
        )

    def _is_allowed_global_var(self, name: str, val: object, absent):
        if val is absent:
            return False
        if name in self.builtin_namespace:
            return True
        if self.visiting_consteval_fn or self.visiting_default_args:
            # allow all global names when visiting default argument values, since we don't have good way to track the usage of default argument values and enforce the restriction only on used ones. This is a bit unsound but should be fine in practice since default argument values are usually simple and unlikely to have side effects.
            return True

        return (
            self._is_allowed_static_value(name, val)
            or self._is_global_constexpr(name)
            or self._is_python_scalar_const(val)
        )

    @contextmanager
    def _name_loc_prefix(self, prefix):
        self.name_loc_prefix = prefix
        yield
        self.name_loc_prefix = None

    def _set_value(self, name: str, value: object):
        self.lscope[name] = value

    def _maybe_set_loc_to_name(self, name, value):
        if isinstance(value, AlloValue):
            name_loc = Location(value.handle.get_loc(), name, self.context)
            value.handle.set_loc(name_loc)
        elif isinstance(value, Value):
            name_loc = Location(value.get_loc(), name, self.context)
            value.set_loc(name_loc)
        else:
            assert False, "invalid call to _maybe_set_loc_to_name"

    def visit(self, node: ast.AST):
        if node is None:
            return

        last_node = self.builder.curr_node
        last_loc = self.builder.get_loc()
        last_src = self.builder.src
        last_file_name = self.builder.file_name
        last_begin_line = self.builder.begin_line

        # recursive visit
        self.builder.src = self.kernel.src
        self.builder.file_name = self.file_name
        self.builder.begin_line = self.begin_line
        self.builder.curr_node = node
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            loc = Location(
                self.file_name,
                node.lineno + self.begin_line - 1,  # type: ignore
                node.col_offset,  # type: ignore
                self.context,
            )
            if self.name_loc_prefix is not None:
                loc = Location(
                    loc,
                    self.name_loc_prefix,
                    self.context,
                )
            self.builder.set_loc(loc)
        try:
            return super().visit(node)
        finally:
            # restore the builder state
            self.builder.curr_node = last_node
            self.builder.src = last_src
            self.builder.file_name = last_file_name
            self.builder.begin_line = last_begin_line
            self.builder.set_loc(last_loc)

    def generic_visit(self, node: ast.AST):
        return self.compile_error(f"Unsupported syntax: {ast.unparse(node)}")

    def visit_compound_stmts(self, stmts, allow_nested_kernel_def: bool = False):
        if not isinstance(stmts, list):
            stmts = [stmts]
        for stmt in stmts:
            if self.block_terminated:
                break
            if isinstance(stmt, ast.FunctionDef):
                if not allow_nested_kernel_def:
                    return self.compile_error(
                        "Nested kernel definitions are only supported at the top level of a kernel body."
                    )
                self.visit(stmt)
                continue
            self.visit(stmt)

    def visit_Module(self, node: ast.Module):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self._entry_function_visited:
            self._entry_function_visited = True
            return self._visit_entry_function_def(node)
        return self._register_nested_kernel_def(node)

    def _visit_entry_function_def(self, node: ast.FunctionDef):
        self._precheck_return_placement(node)
        self.block_terminated = False
        self.has_explicit_return_annotation = node.returns is not None

        arg_names, _ = self.visit(node.args)
        for i, default in enumerate(node.args.defaults[::-1]):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            if name in self.lscope:
                continue
            # construct a fake assignment node to visit the default argument value
            target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                return self.compile_error(
                    "Default arguments must have type annotations"
                )
            init_node = ast.AnnAssign(
                target=target,
                annotation=annotation,
                value=default,
                simple=1,
            )
            try:
                self.visiting_default_args = True
                self.visit(init_node)
            finally:
                self.visiting_default_args = False

        fn_ty: FunctionType = generate_function_type(
            self.context, self.arg_types, self.res_types
        )
        fn_op = FuncOp(self.builder, self.func_name, fn_ty)
        self.curr_func = fn_op
        self.generated_func = fn_op

        entry_block = fn_op.add_entry_block()
        arg_handles = fn_op.get_args()

        arg_idx = 0
        for name, ty in zip(arg_names, self.arg_types):
            if isinstance(ty, ConstexprType):
                if not isinstance(self.lscope.get(name), ConstexprValue):
                    return self.compile_error(
                        f"Missing constexpr argument binding for parameter '{name}' in function '{self.func_name}'."
                    )
                continue
            assert arg_idx < len(arg_handles)
            handle = arg_handles[arg_idx]
            arg_idx += 1
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(name, proxy)
        assert arg_idx == len(arg_handles)

        # visit the function body
        self.builder.set_insertion_point_to_start(entry_block)
        self.visit_compound_stmts(node.body, allow_nested_kernel_def=True)

        # restore the function context
        self.curr_func = None
        if not self.block_terminated:
            if len(self.res_types) > 0:
                return self.compile_error(
                    "Missing return statement for non-void function. Please add a top-level return statement matching the declared return type."
                )
            ip, _ = self.builder.get_insertion_point_and_loc()
            self.builder.set_insertion_point_to_end(ip.get_block())
            ReturnOp(self.builder, [])
        self.builder.set_insertion_point_after(fn_op.get_operation())

    def _resolve_kernel_decorator(self, decorator: ast.AST):
        if isinstance(decorator, ast.Name):
            return self.gscope.get(decorator.id, self.closure_scope.get(decorator.id))
        if isinstance(decorator, ast.Attribute):
            base = unwrap_if_constexpr(self.visit(decorator.value))
            return getattr(base, decorator.attr)
        return None

    def _register_nested_kernel_def(self, node: ast.FunctionDef):
        if len(node.decorator_list) != 1:
            return self.compile_error(
                f"Nested function '{node.name}' must use exactly one '@kernel' decorator."
            )

        decorator = node.decorator_list[0]
        if isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if kw.arg != "mapping":
                    return self.compile_error(
                        f"Nested kernel '{node.name}' does not support decorator keyword argument '{kw.arg}'."
                    )
            decorator = decorator.func

        if self._resolve_kernel_decorator(decorator) is not kernel_decorator:
            return self.compile_error(
                f"Nested function '{node.name}' is not allowed. Only allo kernels are supported for nested definitions."
            )

        if node.name in self.lscope or node.name in self.fscope:
            return self.compile_error(
                f"Nested kernel name '{node.name}' conflicts with an existing local symbol."
            )
        self.fscope[node.name] = NestedKernelSymbol(
            name=node.name,
            node=node,
            owner_func_name=self.func_name,
        )

    def _precheck_return_placement(self, node: ast.FunctionDef):
        ReturnPlacementChecker(self.kernel.src, self.file_name, self.begin_line).visit(
            node
        )

    def visit_arguments(self, node: ast.arguments):
        args_names = [self.visit(arg) for arg in node.args]
        kwargs_names = self.visit(node.kwarg)  # type: ignore
        return args_names, kwargs_names

    def visit_arg(self, node: ast.arg):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_keyword(self, node: ast.keyword):
        return node.arg, self.visit(node.value)

    def visit_Constant(self, node: ast.Constant):
        return ConstexprValue(node.value)

    def visit_Expr(self, node: ast.Expr):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return builtins.slice(lower, upper, step)

    def visit_Compare(self, node: ast.Compare):
        if not (len(node.ops) == 1 and len(node.comparators) == 1):
            return self.compile_error(
                "simultaneous multi-way comparisons are not supported"
            )
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        library_op = self._available_comparison_methods.get(type(node.ops[0]), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported comparison operator '{type(node.ops[0]).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    _available_comparison_methods: dict[Type[ast.cmpop], Operator] = {
        ast.Eq: arith_ops.eq,
        ast.NotEq: arith_ops.ne,
        ast.Lt: arith_ops.lt,
        ast.LtE: arith_ops.le,
        ast.Gt: arith_ops.gt,
        ast.GtE: arith_ops.ge,
    }

    def _ast_expr_may_be_float(self, node: ast.AST) -> builtins.bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, float)

        if isinstance(node, ast.Name):
            val = unwrap_if_constexpr(self.lookup(node.id))
            if isinstance(val, AlloValue):
                return isinstance(val.dtype, DType) and val.dtype.is_float()
            return isinstance(val, float)

        if isinstance(node, ast.Subscript):
            return self._ast_expr_may_be_float(node.value)

        if isinstance(node, ast.UnaryOp):
            return self._ast_expr_may_be_float(node.operand)

        if isinstance(node, ast.BinOp):
            return self._ast_expr_may_be_float(
                node.left
            ) or self._ast_expr_may_be_float(node.right)

        if isinstance(node, ast.Call):
            return True

        return False

    def _materialize_constexpr_pair(self, lhs, rhs):
        if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
            return lhs, rhs
        if isinstance(lhs, ConstexprValue):
            assert isinstance(rhs, AlloValue)
            lhs = self.builder.cast(lhs, rhs.dtype)
        if isinstance(rhs, ConstexprValue):
            assert isinstance(lhs, AlloValue)
            rhs = self.builder.cast(rhs, lhs.dtype)
        return lhs, rhs

    def _prepare_binary_operands(
        self, lhs: AlloValue, rhs: AlloValue, op_name: str
    ) -> tuple[AlloValue, AlloValue]:
        assert isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)
        term_signs = [1, -1] if op_name == "sub" else None
        dst_ty = self.builder.get_promoted_dtype_nary(
            op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
        )
        lhs = self.builder.cast_to_dtype(lhs, dst_ty)
        rhs = self.builder.cast_to_dtype(rhs, dst_ty)
        return self.builder.broadcast_pair(lhs, rhs)

    def _lower_direct_binary(self, op_name: str, lhs, rhs):
        if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
            if op_name == "add":
                return ConstexprValue(lhs.value + rhs.value)
            if op_name == "sub":
                return ConstexprValue(lhs.value - rhs.value)
            if op_name == "mul":
                return ConstexprValue(lhs.value * rhs.value)
            assert False, f"Unsupported direct binary operator: {op_name}"

        lhs, rhs = self._materialize_constexpr_pair(lhs, rhs)
        if not (isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)):
            return self.compile_error(
                f"Binary operator '{op_name}' expects runtime values to be AlloValues"
            )

        if isinstance(lhs.type, ShapedType) or isinstance(rhs.type, ShapedType):
            if op_name == "add":
                return self.call_operator(arith_ops.add, [lhs, rhs])
            if op_name == "sub":
                return self.call_operator(arith_ops.sub, [lhs, rhs])
            if op_name == "mul":
                return self.call_operator(arith_ops.mul, [lhs, rhs])
            assert False, f"Unsupported direct binary operator: {op_name}"

        lhs, rhs = self._prepare_binary_operands(lhs, rhs, op_name)
        floating = lhs.dtype.is_float()
        if op_name == "add":
            return self.builder.create_add(lhs, rhs, floating=floating)
        if op_name == "sub":
            return self.builder.create_sub(lhs, rhs, floating=floating)
        if op_name == "mul":
            return self.builder.create_mul(lhs, rhs, floating=floating)
        assert False, f"Unsupported direct binary operator: {op_name}"

    def _lower_binary_values(self, op: ast.operator, lhs, rhs):
        if isinstance(op, ast.Add):
            return self._lower_direct_binary("add", lhs, rhs)
        if isinstance(op, ast.Sub):
            return self._lower_direct_binary("sub", lhs, rhs)
        if isinstance(op, ast.Mult):
            return self._lower_direct_binary("mul", lhs, rhs)

        library_op = self._available_binary_methods.get(type(op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported binary operator '{type(op).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    def _lower_binop_tree(self, node: ast.BinOp):
        def lower_expr(expr):
            if isinstance(expr, ast.BinOp):
                lhs = lower_expr(expr.left)
                rhs = lower_expr(expr.right)
                return self._lower_binary_values(expr.op, lhs, rhs)
            return self.visit(expr)

        return lower_expr(node)

    def _collect_add_sub_terms(self, node: ast.AST, sign: int, out):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            self._collect_add_sub_terms(node.left, sign, out)
            self._collect_add_sub_terms(node.right, sign, out)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            self._collect_add_sub_terms(node.left, sign, out)
            self._collect_add_sub_terms(node.right, -sign, out)
            return
        out.append((self.visit(node), sign))

    def _collect_mul_terms(self, node: ast.AST, out):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            self._collect_mul_terms(node.left, out)
            self._collect_mul_terms(node.right, out)
            return
        out.append(self.visit(node))

    def _materialize_constexpr_terms(self, terms):
        anchor = None
        for term in terms:
            if isinstance(term, AlloValue):
                anchor = term.dtype
                break
        if anchor is None:
            return terms

        materialized = []
        for term in terms:
            if isinstance(term, ConstexprValue):
                materialized.append(self.builder.cast(term, anchor))
            else:
                materialized.append(term)
        return materialized

    def _lower_nary_add_sub(self, node: ast.BinOp):
        signed_terms = []
        self._collect_add_sub_terms(node, sign=1, out=signed_terms)
        values = [value for value, _ in signed_terms]
        signs = [sign for _, sign in signed_terms]

        if all(isinstance(value, ConstexprValue) for value in values):
            total = 0
            for value, sign in zip(values, signs):
                total += sign * value.value
            return ConstexprValue(total)

        values = self._materialize_constexpr_terms(values)
        if not all(isinstance(value, AlloValue) for value in values):
            return self.compile_error(
                "n-ary add/sub lowering expects runtime values to be AlloValues"
            )

        if all(isinstance(value.type, DType) for value in values):
            dtypes = [value.dtype for value in values]
            op_name = "sub" if any(sign < 0 for sign in signs) else "add"
            dst_ty = self.builder.get_promoted_dtype_nary(
                op_name, dtypes, term_signs=signs
            )
            casted = [self.builder.cast_to_dtype(value, dst_ty) for value in values]
            floating = dst_ty.is_float()
            if any(sign < 0 for sign in signs):
                return self.builder.create_sub_nary(casted, signs, floating=floating)
            return self.builder.create_add_nary(casted, floating=floating)

        if all(sign > 0 for sign in signs):
            return self.builder.reduce_balanced(
                values,
                lambda lhs, rhs: self.call_operator(arith_ops.add, [lhs, rhs]),
            )

        result = None
        for value, sign in zip(values, signs):
            if result is None:
                result = (
                    value
                    if sign > 0
                    else self.call_operator(arith_ops.sub, [ConstexprValue(0), value])
                )
            elif sign > 0:
                result = self.call_operator(arith_ops.add, [result, value])
            else:
                result = self.call_operator(arith_ops.sub, [result, value])
        assert result is not None
        return result

    def _lower_nary_mul(self, node: ast.BinOp):
        terms = []
        self._collect_mul_terms(node, terms)

        if all(isinstance(term, ConstexprValue) for term in terms):
            product = 1
            for term in terms:
                product *= term.value
            return ConstexprValue(product)

        terms = self._materialize_constexpr_terms(terms)
        if not all(isinstance(term, AlloValue) for term in terms):
            return self.compile_error(
                "n-ary mul lowering expects runtime values to be AlloValues"
            )

        if all(isinstance(term.type, DType) for term in terms):
            dtypes = [term.dtype for term in terms]
            dst_ty = self.builder.get_promoted_dtype_nary("mul", dtypes)
            casted = [self.builder.cast_to_dtype(term, dst_ty) for term in terms]
            return self.builder.create_mul_nary(casted, floating=dst_ty.is_float())

        return self.builder.reduce_balanced(
            terms,
            lambda lhs, rhs: self.call_operator(arith_ops.mul, [lhs, rhs]),
        )

    def visit_BinOp(self, node):
        if self.builder.typing_style == "hls":
            if (
                not self.options.fast_math
                and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult))
                and self._ast_expr_may_be_float(node)
            ):
                return self._lower_binop_tree(node)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return self._lower_nary_add_sub(node)
            if isinstance(node.op, ast.Mult):
                return self._lower_nary_mul(node)

        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        library_op = self._available_binary_methods.get(type(node.op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported binary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    _available_binary_methods: dict[Type[ast.operator], Operator] = {
        ast.Add: arith_ops.add,
        ast.Sub: arith_ops.sub,
        ast.Mult: arith_ops.mul,
        ast.Div: arith_ops.div,
        ast.FloorDiv: arith_ops.floordiv,
        ast.Mod: arith_ops.mod,
        ast.Pow: arith_ops.pow,
        ast.LShift: arith_ops.lshift,
        ast.RShift: arith_ops.rshift,
        ast.BitAnd: arith_ops.bitwise_and,
        ast.BitOr: arith_ops.bitwise_or,
        ast.BitXor: arith_ops.bitwise_xor,
    }

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        fn = self._available_unary_methods.get(type(node.op), None)
        if fn is None:
            return self.compile_error(
                f"Unsupported unary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operator(fn, [operand])

    _available_unary_methods: dict[Type[ast.unaryop], Operator] = {
        ast.UAdd: arith_ops.pos,
        ast.USub: arith_ops.neg,
        ast.Not: arith_ops.logical_not,
        ast.Invert: arith_ops.invert,
    }

    def visit_BoolOp(self, node):
        library_op = self._available_boolop_methods.get(type(node.op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported boolean operator '{type(node.op).__name__}' in allo kernel functions",
            )
        nontrivial_values = []

        for subnode in node.values:
            value = self.visit(subnode)
            if isinstance(value, ConstexprValue):
                # constant folding
                bv = builtins.bool(unwrap_if_constexpr(value))
                if (bv is False) and (library_op is arith_ops.logical_and):
                    return ConstexprValue(False)
                if (bv is True) and (library_op is arith_ops.logical_or):
                    return ConstexprValue(True)
                # otherwise constexpr has no effect, so can be skipped
            elif isinstance(value, AlloValue) and isinstance(value.type, ShapedType):
                return self.compile_error(
                    "non-scalar values are not supported in boolean operations"
                )
            else:
                nontrivial_values.append(value)

        if len(nontrivial_values) == 0:
            # all values are constant folded
            if library_op == arith_ops.logical_and:
                return ConstexprValue(True)
            else:
                return ConstexprValue(False)

        while len(nontrivial_values) >= 2:
            # reduce from left to right
            rhs = nontrivial_values.pop()
            lhs = nontrivial_values.pop()
            res = self.call_operator(library_op, [lhs, rhs])
            nontrivial_values.append(res)

        assert len(nontrivial_values) == 1
        return nontrivial_values[0]

    _available_boolop_methods: dict[Type[ast.boolop], Operator] = {
        ast.And: arith_ops.logical_and,
        ast.Or: arith_ops.logical_or,
    }

    def visit_Break(self, node):
        return self.compile_error(
            "'break' statement is not supported in allo kernel functions"
        )

    def visit_Continue(self, node):
        return self.compile_error(
            "'continue' statement is not supported in allo kernel functions"
        )

    def visit_Return(self, node: ast.Return):
        if node.value is None or (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            return_vals = []
        elif isinstance(node.value, ast.Tuple):
            return_vals = [self.visit(elt) for elt in node.value.elts]
        else:
            return_vals = [self.visit(node.value)]

        if len(return_vals) > 0 and not self.has_explicit_return_annotation:
            return self.compile_error(
                "Return values require an explicit return annotation."
            )
        if len(return_vals) != len(self.res_types):
            return self.compile_error(
                f"Return value count mismatch: expected {len(self.res_types)}, got {len(return_vals)}."
            )

        coerced = []
        for value, dst_type in zip(return_vals, self.res_types):
            if not isinstance(value, (AlloValue, ConstexprValue)):
                return self.compile_error(
                    f"Unsupported return value '{value}' of type '{type(value).__name__}'."
                )
            coerced.append(self.builder.cast(value, dst_type))

        ReturnOp(self.builder, [value.handle for value in coerced])
        self.block_terminated = True

    def _visit_if_with_return_impl(
        self, cond: AlloValue, node: ast.If, then_has_return, else_has_return
    ):
        continue_vals = None
        end_if = None
        if_terminated = False
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()
            parent_region = ip.get_block().get_parent_region()
            then_block = self.builder.create_block(parent_region)
            else_block = self.builder.create_block(parent_region)
            end_if = self.builder.create_block(parent_region)

            # branch out from current block to then/else
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            CondBranchOp(self.builder, cond.handle, then_block, else_block)

            liveins = self.lscope.copy()

            # then branch
            self.builder.set_insertion_point_to_start(then_block)
            self.block_terminated = False
            self.visit_compound_stmts(node.body)
            then_vals = self.lscope.copy()
            then_terminated = self.block_terminated

            # else branch
            self.lscope = liveins
            self.builder.set_insertion_point_to_start(else_block)
            self.block_terminated = False
            if node.orelse:
                self.visit_compound_stmts(node.orelse)
                else_vals = self.lscope.copy()
            else:
                else_vals = liveins.copy()
            else_terminated = self.block_terminated

            # if both branches return, there is no fallthrough path
            if then_terminated and else_terminated:
                continue_vals = liveins
                end_if.erase()
                if_terminated = True

            # if exactly one branch returns, continue with the non-returning branch.
            elif then_terminated and not else_terminated:
                self.builder.set_insertion_point_to_end(else_block)
                BranchOp(self.builder, end_if, [])
                continue_vals = else_vals

            elif not then_terminated and else_terminated:
                self.builder.set_insertion_point_to_end(then_block)
                BranchOp(self.builder, end_if, [])
                continue_vals = then_vals

            else:
                assert not (then_has_return or else_has_return)

        assert end_if is not None and continue_vals is not None
        self.block_terminated = if_terminated
        if if_terminated:
            self.lscope = continue_vals.copy()
            return
        self.builder.set_insertion_point_to_start(end_if)
        self.lscope = continue_vals.copy()

    def visit_if_impl(self, cond: AlloValue, node: ast.If):
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            parent_region = ip.get_block().get_parent_region()
            then_block = self.builder.create_block(parent_region)
            else_block = self.builder.create_block(parent_region)

            # compute phi arguments
            self.scf_stack.append(node)
            phi_names, phi_types, then_handles, else_handles = (
                self._visit_then_else_block(node, then_block, else_block)
            )
            self.scf_stack.pop()

            # create if op
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            # if we have phi arguments, we must create else region
            has_else = len(node.orelse) > 0 or len(phi_names) > 0
            phi_ir_types = [ty.materialize(self.context) for ty in phi_types]
            if_op = IfOp(self.builder, phi_ir_types, cond.handle, has_else)
            then_block.merge_before(if_op.get_then_block())
            then_block = if_op.get_then_block()
            then_block.remove_terminator()  # remove the default created
            self.builder.set_insertion_point_to_end(then_block)
            SCFYieldOp(self.builder, then_handles)
            if has_else:
                else_block.merge_before(if_op.get_else_block())
                else_block = if_op.get_else_block()
                else_block.remove_terminator()  # remove the default created
                self.builder.set_insertion_point_to_end(else_block)
                SCFYieldOp(self.builder, else_handles)
            else:
                else_block.erase()

        # update lscope with phi results
        res_handles = if_op.get_results()
        phi_proxies = [
            AlloValue(handle, ty) for handle, ty in zip(res_handles, phi_types)
        ]
        for name, proxy in zip(phi_names, phi_proxies):
            self._set_value_with_loc(name, proxy)

    def _visit_then_else_block(
        self, node: ast.If, then_block: Block, else_block: Block
    ):
        # get a copy of current live-ins
        liveins = self.lscope.copy()
        # visit then block
        self.builder.set_insertion_point_to_start(then_block)
        self.visit_compound_stmts(node.body)
        then_vals = self.lscope.copy()  # capture live-ins in then block
        # restore lscope for else visiting
        self.lscope = liveins
        # visit else block
        self.builder.set_insertion_point_to_start(else_block)
        if node.orelse:
            self.visit_compound_stmts(node.orelse)
            else_vals = self.lscope.copy()  # capture live-ins in else block
        else:
            else_vals = liveins.copy()

        # compute phi arguments
        phi_names = []
        phi_types: list[TypeBase] = []
        then_handles = []
        else_handles = []
        for name, value in liveins.items():
            then_proxy = then_vals.get(name, value)
            else_proxy = else_vals.get(name, value)
            if not isinstance(then_proxy, AlloValue) or not isinstance(
                else_proxy, AlloValue
            ):
                continue
            then_handle = then_proxy.handle
            else_handle = else_proxy.handle
            if then_handle == else_handle:
                continue  # value is not redefined in either block, no need for phi
            # type check
            if isinstance(value, ConstexprValue):
                return self.compile_error(
                    f"Variable '{name}' is defined as a constexpr in the outer scope, but is assigned to non-constexpr values in the then vs else branches."
                )
            outer_ty = value.handle.get_type()
            then_ty = then_handle.get_type()
            else_ty = else_handle.get_type()
            if then_ty != else_ty or then_ty != outer_ty:
                return self.compile_error(
                    f"Variable '{name}' has incompatible types in outer scope vs then vs else branches: {outer_ty} vs {then_ty} vs {else_ty}."
                )
            phi_types.append(then_proxy.type)
            phi_names.append(name)
            then_handles.append(then_handle)
            else_handles.append(else_handle)
        return phi_names, phi_types, then_handles, else_handles

    def visit_IfExp(self, node: ast.IfExp):
        cond = self.visit(node.test)
        if isinstance(cond, AlloValue):
            cond = self.builder.scalar_cast(cond, bool)
            # if exp cannot define new variables
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            then_val = self.visit(node.body)
            else_val = self.visit(node.orelse)

            # type check
            # Case 1: both branches are constexprs
            then_is_constexpr = isinstance(then_val, ConstexprValue)
            else_is_constexpr = isinstance(else_val, ConstexprValue)
            if then_is_constexpr and else_is_constexpr:
                # TODO: support this case
                return self.compile_error(
                    f"Cannot deduce type of ternary expression because both branches are constexprs. Please use if statement instead of if expression in this case, or make sure at least one branch is non-constexpr so that the type can be deduced."
                )
            # Case 2: both branches are AlloValues:
            if not then_is_constexpr and not else_is_constexpr:
                if then_val.type != else_val.type:
                    return self.compile_error(
                        f"Type mismatch between then vs else branches of ternary expression: {then_val.type} vs {else_val.type}."
                    )
                res_type = then_val.type
            # Case 3: exactly one branch is a constexpr, use the other branch's type as the result type
            res_type = then_val.type if not then_is_constexpr else else_val.type
            if then_is_constexpr:
                then_val = self.builder.cast(then_val, res_type)
            if else_is_constexpr:
                else_val = self.builder.cast(else_val, res_type)

            # create select op
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            sel_op = SelectOp(
                self.builder, cond.handle, then_val.handle, else_val.handle
            )
            return AlloValue(sel_op, res_type)
        else:
            # constexpr path
            assert isinstance(cond, ConstexprValue)
            cond = unwrap_if_constexpr(cond)
            if type(cond) not in self._condition_types:
                return self.compile_error(
                    "Ternary expression conditionals can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in self._condition_types),
                        type(cond).__name__,
                    ),
                )
            selected = node.body if cond else node.orelse
            return self.visit(selected)

    _condition_types = {
        bool,
        int,
        type(None),
    }

    def _branch_has_return(self, stmts):
        # TODO: maybe a better checking
        return any(isinstance(stmt, ast.Return) for stmt in stmts)

    def visit_If(self, node: ast.If):
        cond = self.visit(node.test)
        if isinstance(cond, AlloValue):
            cond = self.builder.scalar_cast(cond, bool)
            then_has_return = self._branch_has_return(node.body)
            else_has_return = self._branch_has_return(node.orelse)
            if then_has_return or else_has_return:
                self._visit_if_with_return_impl(
                    cond, node, then_has_return, else_has_return
                )
            else:
                self.visit_if_impl(cond, node)
        else:
            # constexpr path
            assert isinstance(cond, ConstexprValue)
            cond = unwrap_if_constexpr(cond)
            if type(cond) not in self._condition_types:
                return self.compile_error(
                    "`if` conditionals can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in self._condition_types),
                        type(cond).__name__,
                    ),
                )
            selected = node.body if cond else node.orelse
            self.visit_compound_stmts(selected)

    def visit_Attribute(self, node):
        lhs = unwrap_if_constexpr(self.visit(node.value))
        return getattr(lhs, node.attr)

    def visit_Subscript(self, node: ast.Subscript):
        return self.visit_Subscript_Load(node)

    def visit_Subscript_Store(self, node, value):
        assert isinstance(node.ctx, ast.Store)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        slices = (
            tuple([slices])
            if isinstance(slices, (AlloValue, ConstexprValue))
            else slices
        )
        return self.call_operator(mem_ops.store, [lhs, slices, value])

    def visit_Subscript_Load(self, node):
        assert isinstance(node.ctx, ast.Load)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if isinstance(lhs, tuple) and isinstance(slices, ConstexprValue):
            return lhs[slices.value]
        slices = (
            tuple([slices])
            if isinstance(slices, (AlloValue, ConstexprValue))
            else slices
        )
        return self.call_operator(mem_ops.load, [lhs, slices])

    def visit_ListComp(self, node):
        if len(node.generators) != 1:
            return self.compile_error(
                "only single generator is supported in list comprehensions"
            )
        comp = node.generators[0]
        iter = self.visit(comp.iter)
        if not isinstance(iter, tuple):
            return self.compile_error(
                "only tuple iteration is supported in list comprehensions"
            )

        results = []
        for item in iter:
            if not isinstance(comp.target, ast.Name):
                return self.compile_error(
                    "only simple variable targets are supported in list comprehensions",
                )
            self._set_value(comp.target.id, item)
            results.append(self.visit(node.elt))
        return tuple(results)

    def visit_Store(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Load(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Tuple(self, node):
        elts = [self.visit(e) for e in node.elts]
        return tuple(elts)

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id
        return self.lookup(node.id)

    def visit_List(self, node):
        ctx = self.visit(node.ctx)
        assert ctx is None
        return tuple([self.visit(e) for e in node.elts])

    def _flatten_list_initializer(self, node: ast.AST):
        if isinstance(node, ast.List):
            values = []
            shapes = []
            for elt in node.elts:
                shape, flat_values = self._flatten_list_initializer(elt)
                shapes.append(shape)
                values.extend(flat_values)
            if len(shapes) == 0:
                return (0,), values
            first_shape = shapes[0]
            if any(shape != first_shape for shape in shapes):
                return self.compile_error(
                    f"Ragged list initializer '{ast.unparse(node)}' is not supported."
                )
            return (len(node.elts), *first_shape), values

        value = unwrap_if_constexpr(self.visit(node))
        if type(value) not in (builtins.int, builtins.float):
            return self.compile_error(
                f"List initializer elements must be compile-time int or float constants, got '{ast.unparse(node)}'."
            )
        return (), [value]

    def _visit_shaped_list_initializer(
        self, node: ast.List, dst_type: ShapedType, name: str
    ):
        shape, values = self._flatten_list_initializer(node)
        if tuple(shape) != tuple(dst_type.shape):
            return self.compile_error(
                f"List initializer shape mismatch for '{name}': expected {tuple(dst_type.shape)}, got {shape}."
            )
        return self.builder.make_shaped_constant(values, dst_type, name)

    def visit_AugAssign(self, node: ast.AugAssign):
        lhs = copy.deepcopy(node.target)
        lhs.ctx = ast.Load()
        rhs = ast.BinOp(left=lhs, op=node.op, right=node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        for x in ["lineno", "col_offset", "end_lineno", "end_col_offset"]:
            if hasattr(node, x):
                y = getattr(node, x)
                setattr(rhs, x, y)
                setattr(assign, x, y)
        self.visit(assign)

    def _resolve_annotation_symbol(self, annotation: ast.AST):
        if isinstance(annotation, ast.Name):
            return self.lookup(annotation.id)
        if isinstance(annotation, ast.Attribute):
            base = unwrap_if_constexpr(
                self._resolve_annotation_symbol(annotation.value)
            )
            return getattr(base, annotation.attr)
        return self.compile_error(
            f"Unsupported annotation expression '{ast.unparse(annotation)}'."
        )

    def _type_annotation_scope(self):
        scope = self.builtin_namespace.copy()
        scope.update(self.gscope)
        scope.update(self.closure_scope)
        scope.update(self.fscope)
        scope.update(self.lscope)
        for key, value in list(scope.items()):
            if self._is_python_scalar_const(value):
                scope[key] = ConstexprValue(value)
        return scope

    def _parse_annotation(self, annotation: ast.AST, name: str) -> TypeBase:
        scope = self._type_annotation_scope()
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return self.compile_error(f"Missing type annotation for '{name}'.")
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                return self.kernel.parse_type_annotation(annotation.value, scope=scope)
            except Exception as e:
                return self.compile_error(
                    f"Unsupported type annotation '{annotation.value}' for '{name}': {e}"
                )
        if isinstance(annotation, (ast.Name, ast.Attribute)):
            resolved = self._resolve_annotation_symbol(annotation)
            try:
                return self.kernel.parse_type_annotation(resolved, scope=scope)
            except Exception:
                pass

        annotation_text = ast.unparse(annotation)
        if annotation_text in {"constexpr", "Constexpr"}:
            return constexpr
        try:
            return self.kernel.parse_type_annotation(annotation_text, scope=scope)
        except Exception as e:
            return self.compile_error(
                f"Unsupported type annotation '{annotation_text}' for '{name}': {e}"
            )

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Attribute):
            return self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if not isinstance(node.target, ast.Name):
            return self.compile_error(
                "annotated assignment only supports simple variable targets"
            )
        if node.target.id in self.lscope:
            return self.compile_error(
                f"Variable '{node.target.id}' is already defined in the current scope."
            )

        parsed_type = self._parse_annotation(node.annotation, node.target.id)
        if node.value is None:
            if isinstance(parsed_type, ShapedType):
                self._set_value_with_loc(
                    node.target.id, self.builder.make_buffer(parsed_type)
                )
                return
            return self.compile_error(
                f"Annotated variable '{node.target.id}' must have an initializer."
            )

        if isinstance(parsed_type, ShapedType) and isinstance(node.value, ast.List):
            with self._name_loc_prefix(node.target.id):
                value = self._visit_shaped_list_initializer(
                    node.value, parsed_type, node.target.id
                )
            self._set_value_with_loc(node.target.id, value)
            return

        with self._name_loc_prefix(node.target.id):
            value = self.visit(node.value)

        if isinstance(parsed_type, ConstexprType):
            if isinstance(value, AlloValue):
                return self.compile_error(
                    f"Unsupported assignment with type annotation 'constexpr' and value of type '{value.type}'."
                )
            self._set_value(node.target.id, ConstexprValue(value))
            return

        if not isinstance(value, (AlloValue, ConstexprValue)):
            return self.compile_error(
                f"Unsupported initializer for variable '{node.target.id}' with type annotation '{ast.unparse(node.annotation)}'."
            )
        self._set_value_with_loc(node.target.id, self.builder.cast(value, parsed_type))

    def visit_Assign(self, node: ast.Assign):
        targets = node.targets
        if len(targets) != 1:
            return self.compile_error("multiple assignment targets are not supported")
        target = targets[0]
        if isinstance(target, ast.Name):
            with self._name_loc_prefix(target.id):
                value = self.visit(node.value)
        else:
            value = self.visit(node.value)
        self._do_assignment(target, value)

    def _do_assignment(self, target, value: ConstexprValue | AlloValue):
        assert isinstance(target.ctx, ast.Store)
        if isinstance(target, ast.Subscript):
            return self.visit_Subscript_Store(target, value)
        if isinstance(target, ast.Tuple):
            assert isinstance(value, tuple)
            for i, elt in enumerate(target.elts):
                self._do_assignment(elt, value[i])
            return
        if isinstance(target, ast.Attribute):
            return self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if isinstance(target, ast.Name):
            target = self.visit(target)
            # the first time we see a variable is considered its definition site, and its type if inferred from the assigned value. subsequent assignments to the same variable must be type-compatible with the first definition.
            if target not in self.lscope:
                if isinstance(value, ConstexprValue):
                    return self.compile_error(
                        "Constexpr variables must be explcitly declared with type annotation. Please add a type annotation of 'constexpr' to this variable."
                    )
                self._set_value_with_loc(target, value)
                return
            proxy = self.lscope[target]
            if isinstance(proxy, ConstexprValue):
                return self.compile_error(
                    f"Cannot reassign to variable '{target}' defined as a constexpr"
                )
            if isinstance(value, ConstexprValue):
                ret = self.builder.materialize_literal_like(value.value, proxy)
            elif isinstance(value, AlloValue):
                ret = self.builder.cast(value, proxy.type)
            self._set_value_with_loc(target, ret)

    def _set_value_with_loc(self, target, value):
        self._set_value(target, value)
        self._maybe_set_loc_to_name(target, value)

    def _test_loop_iter_args(self, node, liveins: dict, ignore: set[str]):
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        # create dummy block
        block = self.builder.create_block(ip.get_block().get_parent_region())
        self.builder.set_insertion_point_to_start(block)
        # dry visit
        old_dry_run = self.dry_run_loop_analysis
        self.dry_run_loop_analysis = True
        self.scf_stack.append(node)
        try:
            self.visit_compound_stmts(node.body)
        finally:
            self.scf_stack.pop()
            self.dry_run_loop_analysis = old_dry_run
        # restore state
        block.erase()
        self.builder.set_insertion_point_and_loc(ip, last_loc)

        # compute live-outs
        init_types = []
        init_handles = []
        names = []

        for name, livein in liveins.items():
            if name in ignore:
                continue
            if isinstance(livein, ConstexprValue):
                continue
            assert isinstance(livein, AlloValue)
            loop_val = self.lscope[name]
            if loop_val.handle == livein.handle:
                continue  # variable is not assigned in the loop body
            # type check
            if type(loop_val) != type(livein) or loop_val.type != livein.type:
                return self.compile_error(
                    f"Loop variable '{name}' has incompatible types in outer scope vs loop body: {livein.type} vs {loop_val.type}."
                )
            names.append(name)
            init_handles.append(livein.handle)
            init_types.append(livein.type)

        # restore lscope
        self.lscope = liveins.copy()
        return names, init_handles, init_types

    def visit_While(self, node: ast.While):
        if node.orelse:
            return self.compile_error(
                "'while' statement with 'else' block is not supported"
            )
        with EnterSubRegion(self):
            liveins = self.lscope.copy()
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore=set()
            )
            # create while op
            init_ir_types = [ty.materialize(self.context) for ty in init_types]
            while_op = WhileOp(self.builder, init_ir_types, init_handles)

            # create before region
            before_block = self.builder.create_block(
                while_op.get_before(), init_ir_types
            )
            self.builder.set_insertion_point_to_start(before_block)
            block_args = before_block.get_args()
            for name, arg, ty in zip(names, block_args, init_types):
                proxy = AlloValue(arg, ty)
                self._set_value_with_loc(name, proxy)

            # visit condition
            cond = self.visit(node.test)
            self.builder.set_insertion_point_to_end(before_block)
            assert isinstance(cond, AlloValue)
            # create cond
            ConditionOp(self.builder, cond.handle, block_args)

            # create after region
            after_block = self.builder.create_block(while_op.get_after(), init_ir_types)
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = after_block.get_args()
            for name, arg, ty in zip(names, body_handles, init_types):
                proxy = AlloValue(arg, ty)
                self._set_value_with_loc(name, proxy)

            # visit loop body
            self.scf_stack.append(node)
            self.visit_compound_stmts(node.body)
            self.scf_stack.pop()

            # create yield
            yield_handles = [self.lscope[name].handle for name in names]
            self.builder.set_insertion_point_to_end(after_block)
            # remove the default terminator
            after_block.remove_terminator()
            SCFYieldOp(self.builder, yield_handles)

        # update lscope with iter args
        res_handles = while_op.get_results()
        res_proxies = [
            AlloValue(handle, ty) for handle, ty in zip(res_handles, init_types)
        ]
        for name, proxy in zip(names, res_proxies):
            self._set_value_with_loc(name, proxy)

    def visit_For(self, node: ast.For):
        if node.orelse:
            return self.compile_error(
                "'for' statement with 'else' block is not supported"
            )
        if not isinstance(node.iter, ast.Call):
            return self.compile_error(
                "Only 'for' loops over 'range()/grid()' are supported"
            )

        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = {kw.arg: self.visit(kw.value) for kw in node.iter.keywords}

        if IteratorClass is Range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)  # type: ignore
            lb = iterator.start
            ub = iterator.stop
            step = iterator.step
        elif IteratorClass is Grid:
            iterator = IteratorClass(*iter_args, **iter_kwargs)  # type: ignore
            return self.visit_Grid(node, iterator)
        else:
            return self.compile_error(
                "Only 'for' loops over 'range()' and 'grid()' are supported"
            )

        if not isinstance(node.target, ast.Name):
            return self.compile_error(
                "loop target must be a single variable in 'for' loops"
            )

        if isinstance(step, ConstexprValue) and step.value <= 0:
            return self.compile_error(
                "loop step must be a positive integer in 'for' loops"
            )

        lb, ub, step = self.builder.normalize_indices((lb, ub, step), expected_len=3)

        with EnterSubRegion(self):
            index_ty = index.materialize(self.context)
            iv_placeholder = PoisonOp(self.builder, index_ty)
            self._set_value(node.target.id, AlloValue(iv_placeholder, index))

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore={node.target.id}
            )
            # create for op
            for_op = ForOp(
                self.builder, lb.handle, ub.handle, step.handle, init_handles
            )
            self.scf_stack.append(node)
            for_op_body = for_op.get_body()
            self.builder.set_insertion_point_to_start(for_op_body)
            block_handles = [
                # skip the first argument which is the induction variable
                for_op_body.get_arg_at(i + 1)
                for i in range(len(init_handles))
            ]
            block_args = [
                AlloValue(handle, ty) for handle, ty in zip(block_handles, init_types)
            ]
            for iter_name, proxy in zip(names, block_args):
                self._set_value_with_loc(iter_name, proxy)
            # visit loop body
            self.visit_compound_stmts(node.body)
            self.scf_stack.pop()
            # create yield
            yield_handles = [self.lscope[iter_name].handle for iter_name in names]
            self.builder.set_insertion_point_to_end(for_op_body)
            # remove the default terminator
            for_op_body.remove_terminator()
            SCFYieldOp(self.builder, yield_handles)
            assert for_op.get_num_regions() == 1

            # update induction variable with the actual one
            iv = for_op.get_induction_var()
            iv_placeholder.get_result_at(0).replace_all_uses_with(iv)
            iv_placeholder.erase()
            self._set_value_with_loc(node.target.id, AlloValue(iv, index))

        # update lscope with iter args
        res_handles = for_op.get_results()
        for iter_name, handle, ty in zip(names, res_handles, init_types):
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(iter_name, proxy)

    def visit_Grid(self, node: ast.For, iterator: Grid):
        if len(iterator.starts) <= 1:
            return self.compile_error(
                "Use range() for single-dimensional loops; grid() requires at least two dimensions."
            )
        if not isinstance(node.target, ast.Tuple):
            return self.compile_error(
                "loop target must be a tuple of variables in 'for' loops over 'grid()'"
            )
        if len(node.target.elts) != len(iterator.starts):
            return self.compile_error(
                f"loop target must have the same number of variables as the dimensions of the grid iterator. Expected {len(iterator.starts)} variables, but got {len(node.target.elts)}."
            )

        lbs = iterator.starts
        ubs = iterator.stops
        steps = iterator.steps

        if any(isinstance(step, ConstexprValue) and step.value <= 0 for step in steps):
            return self.compile_error(
                "loop step must be a positive integer in 'for' loops"
            )

        lb_proxies = self.builder.normalize_indices(lbs)
        ub_proxies = self.builder.normalize_indices(ubs)
        step_proxies = self.builder.normalize_indices(steps)

        with EnterSubRegion(self):
            index_ty = index.materialize(self.context)
            iv_placeholders = [PoisonOp(self.builder, index_ty) for _ in lbs]
            targets = set()
            for i, target in enumerate(node.target.elts):
                if not isinstance(target, ast.Name):
                    return self.compile_error(
                        "loop target must be a single variable in 'for' loops over 'grid()'"
                    )
                self._set_value(target.id, AlloValue(iv_placeholders[i], index))
                targets.add(target.id)

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore=targets
            )
            if len(init_handles) > 0:
                raise NotImplementedError(
                    f"Non-trivial loop-carried dependencies are not supported in 'for' loops over 'grid()' at this moment."
                )
            # create parallel op
            par_op = ParallelOp(
                self.builder,
                [lb.handle for lb in lb_proxies],
                [ub.handle for ub in ub_proxies],
                [step.handle for step in step_proxies],
                init_handles,
            )
            self.scf_stack.append(node)
            par_op_body = par_op.get_body()
            self.builder.set_insertion_point_to_start(par_op_body)
            # no iter args now, so no block arguments other than induction variables
            # visit loop body
            self.visit_compound_stmts(node.body)
            self.scf_stack.pop()
            # parallel op use scf.reduce as terminator
            # see: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfparallel-scfparallelop

            ivs = par_op.get_induction_vars()
            for iv, placeholder in zip(ivs, iv_placeholders):
                placeholder.get_result_at(0).replace_all_uses_with(iv)
                placeholder.erase()
            for iv, target in zip(ivs, node.target.elts):
                proxy = AlloValue(iv, index)
                self._set_value_with_loc(target.id, proxy)  # type: ignore

        # update lscope with iter args
        res_handles = par_op.get_results()
        for name, handle, ty in zip(names, res_handles, init_types):
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(name, proxy)

    def visit_JoinedStr(self, node):
        values = list(node.values)
        for i, value in enumerate(values):
            if isinstance(value, ast.Constant):
                values[i] = str(value.value)  # type: ignore
            elif isinstance(value, ast.FormattedValue):
                conversion_code = value.conversion
                evaluated = self.visit(value.value)
                if not isinstance(evaluated, ConstexprValue):
                    return self.compile_error(
                        "Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type "
                        + str(type(evaluated)),
                    )
                values[i] = (  # type: ignore
                    "{}" if conversion_code < 0 else "{!" + chr(conversion_code) + "}"
                ).format(evaluated.value)
            else:
                assert False, f"unexpected value type in JoinedStr: {type(value)}"
        return "".join(values)  # type: ignore

    def visit_Call(self, node):
        fn = unwrap_if_constexpr(self.visit(node.func))
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
                    assert isinstance(arg, tuple)
                    args.extend(arg)
                else:
                    ret = self.visit(arg)
                    args.append(ret)
        finally:
            self.visiting_consteval_fn = False
        return self.call_function(fn, args, kws)

    def call_function(self, fn, args, kws):
        """Dispatch callable targets across kernel/op/type/consteval frontends."""

        if isinstance(fn, NestedKernelSymbol):
            return self.call_nested_kernel(fn, args, kws)
        if isinstance(fn, Kernel):
            return self.call_kernel(fn, args, kws)
        if isinstance(fn, (Operator, BoundOperator)):
            return self.call_operator(fn, args, kws)
        if isinstance(fn, ConstevalFunction):
            try:
                ret = fn(*args, **kws)
                # TODO: check if returned value is valid
                return ConstexprValue(ret)
            except CompilationError:
                raise
            except Exception as e:
                return self.compile_error(
                    f"error when calling consteval function '{fn.__name__}': {e}"
                )
        fn_mod = getattr(fn, "__module__", type(fn).__module__)
        fn_name = getattr(fn, "__name__", type(fn).__name__)
        return self.compile_error(
            f"only allo kernel functions, operations, and consteval functions can be called in allo kernel functions, but got {fn_mod}.{fn_name}"
        )

    def _next_called_kernel_name(self, fn: Kernel | NestedKernelSymbol | str) -> str:
        if isinstance(fn, Kernel):
            callee_name = fn.func_name
        elif isinstance(fn, NestedKernelSymbol):
            callee_name = fn.name
        else:
            callee_name = fn
        base_name = f"{self.func_name}.{callee_name}"
        call_id = self._kernel_call_counter
        self._kernel_call_counter += 1
        if base_name in self._kernel_base_names:
            return f"{base_name}.{call_id}"
        self._kernel_base_names.add(base_name)
        return base_name

    def _kernel_call_key(self, fn: Kernel) -> str:
        return f"kernel:{fn.__module__}.{fn.__qualname__}"

    def _nested_call_key(self, nested: NestedKernelSymbol) -> str:
        return f"nested:{nested.owner_func_name}.{nested.name}"

    def _check_recursive_call(self, key: str):
        if key in self._active_kernel_calls:
            chain = " -> ".join(self._active_kernel_calls + [key])
            return self.compile_error(
                f"Recursive kernel calls are not supported: {chain}"
            )

    def _build_kernel_call_operand(
        self, value: object, expected_ty: TypeBase, arg_name: str
    ):
        if isinstance(expected_ty, ConstexprType):
            assert False, "constexpr arguments do not have call operands"
        if not isinstance(value, (AlloValue, ConstexprValue)):
            value = ConstexprValue(value)
        try:
            return self.builder.cast(value, expected_ty).handle
        except CompilationError:
            value_ty = value.type if isinstance(value, AlloValue) else "constexpr"
            return self.compile_error(
                f"Kernel call argument '{arg_name}' type mismatch: expected '{expected_ty}', got '{value_ty}'."
            )

    def _prepare_kernel_call_args(
        self, callee_name: str, bound_items, arg_types: Sequence[TypeBase]
    ):
        bound_items = list(bound_items)
        if len(arg_types) != len(bound_items):
            return self.compile_error(
                f"Kernel specialization argument count mismatch for '{callee_name}': expected {len(bound_items)}, got {len(arg_types)}."
            )

        callee_context: dict[str, object] = {}
        call_operands: list[Value] = []
        for (arg_name, arg_val), expected_ty in zip(bound_items, arg_types):
            if isinstance(expected_ty, ConstexprType):
                if isinstance(arg_val, AlloValue):
                    return self.compile_error(
                        f"Kernel call argument '{arg_name}' must be constexpr, but got runtime value of type '{arg_val.type}'."
                    )
                if not isinstance(arg_val, ConstexprValue):
                    arg_val = ConstexprValue(arg_val)
                callee_context[arg_name] = arg_val
                continue
            call_operands.append(
                self._build_kernel_call_operand(arg_val, expected_ty, arg_name)
            )
        return callee_context, call_operands

    def _decode_kernel_call_results(
        self, call_op: CallOp, res_types: Sequence[TypeBase]
    ):
        if any(isinstance(ty, ConstexprType) for ty in res_types):
            return self.compile_error(
                "Kernel calls returning constexpr values are not supported."
            )
        if len(res_types) == 0:
            return None
        if call_op.get_num_results() != len(res_types):
            return self.compile_error(
                f"Kernel call result count mismatch: expected {len(res_types)}, got {call_op.get_num_results()}."
            )
        results = [
            AlloValue(handle, ty)
            for handle, ty in zip(call_op.get_results(), res_types)
        ]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _make_dry_run_call_results(self, res_types: Sequence[TypeBase]):
        if any(isinstance(ty, ConstexprType) for ty in res_types):
            return self.compile_error(
                "Kernel calls returning constexpr values are not supported."
            )
        if len(res_types) == 0:
            return None
        results = [
            AlloValue(PoisonOp(self.builder, ty.materialize(self.context)), ty)
            for ty in res_types
        ]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _parse_return_types(self, node: ast.FunctionDef) -> list[TypeBase]:
        if node.returns is None or (
            isinstance(node.returns, ast.Constant) and node.returns.value is None
        ):
            return []
        if isinstance(node.returns, ast.Tuple):
            return [
                self._parse_annotation(ret, f"return[{i}]")
                for i, ret in enumerate(node.returns.elts)
            ]
        return [self._parse_annotation(node.returns, "return")]

    def _bind_nested_arguments(self, nested: NestedKernelSymbol, args, kws):
        fn_args = nested.node.args
        if (
            fn_args.posonlyargs
            or fn_args.kwonlyargs
            or fn_args.vararg is not None
            or fn_args.kwarg is not None
        ):
            return self.compile_error(
                f"Nested kernel '{nested.name}' only supports regular positional/keyword arguments."
            )

        params = fn_args.args
        param_names = [param.arg for param in params]
        if len(args) > len(params):
            return self.compile_error(
                f"Invalid arguments for nested kernel '{nested.name}': expected at most {len(params)} positional arguments, got {len(args)}."
            )

        bound = {name: value for name, value in zip(param_names, args)}
        for kw_name, kw_val in kws.items():
            if kw_name not in param_names:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': unexpected keyword argument '{kw_name}'."
                )
            if kw_name in bound:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': multiple values for argument '{kw_name}'."
                )
            bound[kw_name] = kw_val

        defaults = fn_args.defaults
        first_default_idx = len(params) - len(defaults)
        for idx, param in enumerate(params):
            if param.arg in bound:
                continue
            if idx < first_default_idx:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': missing required argument '{param.arg}'."
                )
            try:
                self.visiting_default_args = True
                bound[param.arg] = self.visit(defaults[idx - first_default_idx])
            finally:
                self.visiting_default_args = False

        return {param.arg: bound[param.arg] for param in params}

    def _infer_nested_arg_type(self, name: str, value: object) -> TypeBase:
        if isinstance(value, AlloValue):
            return value.type
        if isinstance(value, ConstexprValue):
            return constexpr
        return self.compile_error(
            f"Cannot infer type for nested kernel argument '{name}' from value of type '{type(value).__name__}'."
        )

    def _specialize_nested_kernel(self, nested: NestedKernelSymbol, bound):
        arg_types = []
        for param in nested.node.args.args:
            value = bound[param.arg]
            if param.annotation is None:
                arg_types.append(self._infer_nested_arg_type(param.arg, value))
            else:
                arg_types.append(self._parse_annotation(param.annotation, param.arg))
        return arg_types, self._parse_return_types(nested.node)

    def _build_nested_capture_scopes(self):
        closure_scope: dict[str, object] = {}
        forbidden_scope: dict[str, object] = {}
        for name, value in self.lscope.items():
            if self._is_allowed_static_value(name, value):
                closure_scope[name] = value
            else:
                forbidden_scope[name] = value
        return closure_scope, self.fscope.copy(), forbidden_scope

    def call_nested_kernel(self, nested: NestedKernelSymbol, args, kws):
        key = self._nested_call_key(nested)
        self._check_recursive_call(key)

        bound = self._bind_nested_arguments(nested, args, kws)
        sub_arg_types, sub_res_types = self._specialize_nested_kernel(nested, bound)
        callee_context, call_operands = self._prepare_kernel_call_args(
            nested.name, bound.items(), sub_arg_types
        )
        closure_scope, closure_fscope, forbidden_scope = (
            self._build_nested_capture_scopes()
        )

        if self.dry_run_loop_analysis:
            return self._make_dry_run_call_results(sub_res_types)

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        sub_generator = None
        self._active_kernel_calls.append(key)
        try:
            self.builder.set_insertion_point_to_end(self.module.get_body())
            self.builder.set_loc(
                Location(
                    self.file_name,
                    self.begin_line + nested.node.lineno - 1,
                    1,
                    self.context,
                )
            )
            self.builder.src = self.kernel.src
            sub_generator = MLIRCodeGenerator(
                self.context,
                self.module,
                self.builder,
                kernel=self.kernel,
                func_name=self._next_called_kernel_name(nested),
                file_name=self.file_name,
                begin_line=self.begin_line,
                gscope=self.gscope,
                arg_types=sub_arg_types,
                res_types=sub_res_types,
                options=self.options,
                callee_context=callee_context,
                fscope=closure_fscope,
                closure_scope=closure_scope,
                forbidden_closure_scope=forbidden_scope,
                active_kernel_calls=self._active_kernel_calls,
            )
            sub_generator.visit(nested.node)
            if sub_generator.generated_func is None:
                return self.compile_error(
                    f"Internal error: failed to materialize nested kernel '{nested.name}'."
                )
        except CompilationError as e:
            raise CompilationError(
                e.src if e.src is not None else self.kernel.src,
                f"error when compiling kernel '{nested.name}' called from '{self.func_name}': {e.error_msg}",
                e.node,
                file_name=e.file_name,
                begin_line=e.begin_line,
            ) from e
        finally:
            self._active_kernel_calls.pop()
            self.builder.src = self.kernel.src
            self.builder.file_name = self.file_name
            self.builder.begin_line = self.begin_line
            self.builder.set_insertion_point_and_loc(ip, last_loc)

        assert sub_generator is not None and sub_generator.generated_func is not None
        call_op = CallOp(self.builder, sub_generator.generated_func, call_operands)
        return self._decode_kernel_call_results(call_op, sub_res_types)

    def call_kernel(self, fn: Kernel, args, kws):
        """Lower/call a kernel specialization and decode structured return values."""

        key = self._kernel_call_key(fn)
        self._check_recursive_call(key)

        try:
            bound = fn.signature.bind(*args, **kws)
            bound.apply_defaults()
        except TypeError as e:
            return self.compile_error(
                f"Invalid arguments for kernel '{fn.func_name}': {e}."
            )

        try:
            sub_arg_types = list(fn.parse_argument_annotations())
            sub_res_types = list(fn.parse_return_annotation())
        except Exception as e:
            return self.compile_error(
                f"Failed to specialize kernel '{fn.func_name}': {e}"
            )

        callee_context, call_operands = self._prepare_kernel_call_args(
            fn.func_name, bound.arguments.items(), sub_arg_types
        )

        if self.dry_run_loop_analysis:
            return self._make_dry_run_call_results(sub_res_types)

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        sub_generator = None
        self._active_kernel_calls.append(key)
        try:
            self.builder.set_insertion_point_to_end(self.module.get_body())
            self.builder.set_loc(Location(fn.file_name, fn.begin_line, 1, self.context))
            self.builder.src = fn.src
            sub_generator = MLIRCodeGenerator(
                self.context,
                self.module,
                self.builder,
                kernel=fn,
                func_name=self._next_called_kernel_name(fn),
                file_name=fn.file_name,
                begin_line=fn.begin_line,
                gscope=fn.get_capture_scope(),
                arg_types=sub_arg_types,
                res_types=sub_res_types,
                options=fn.options,
                callee_context=callee_context,
                active_kernel_calls=self._active_kernel_calls,
            )
            sub_generator.visit(fn.parse())
            if sub_generator.generated_func is None:
                return self.compile_error(
                    f"Internal error: failed to materialize callee function for kernel '{fn.func_name}'."
                )
        except CompilationError as e:
            raise CompilationError(
                e.src if e.src is not None else fn.src,
                f"error when compiling kernel '{fn.func_name}' called from '{self.func_name}': {e.error_msg}",
                e.node,
                file_name=e.file_name,
                begin_line=e.begin_line,
            ) from e
        finally:
            self._active_kernel_calls.pop()
            self.builder.src = self.kernel.src
            self.builder.file_name = self.file_name
            self.builder.begin_line = self.begin_line
            self.builder.set_insertion_point_and_loc(ip, last_loc)

        assert sub_generator is not None and sub_generator.generated_func is not None
        call_op = CallOp(self.builder, sub_generator.generated_func, call_operands)
        return self._decode_kernel_call_results(call_op, sub_res_types)

    def call_operator(self, fn: Operator | BoundOperator, args, kwargs={}):
        if isinstance(fn, BoundOperator):
            args = fn.bind_args(args)
            fn = fn.op
        ip, last_loc = self.builder.get_insertion_point_and_loc()

        # try folding first
        if fn.fold_impl is not None:
            ret = fn.fold_impl(*args, **kwargs)
            if ret is not NO_FOLD:
                return ret

        # fold failed, build IR
        if fn.build_impl is None:
            return self.compile_error(
                f"Operator '{fn.__name__}' does not define a construction implementation"
            )
        try:
            return fn.build_impl(self.builder, *args, **kwargs)
        finally:
            # restore states
            self.builder.set_insertion_point_and_loc(ip, last_loc)

    @staticmethod
    def static_executor(python_fn):
        def ret(self, node: ast.Call):
            kws = {
                name: unwrap_if_constexpr(value)
                for name, value in (self.visit(keyword) for keyword in node.keywords)
            }
            args = [unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
            return ConstexprValue(python_fn(*args, **kws))

        return ret

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError(
                "`static_assert` requires one or two positional arguments only"
            )

        passed = unwrap_if_constexpr(self.visit(node.args[0]))
        if not isinstance(passed, builtins.bool):
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

            raise StaticAssertionError(
                self.kernel.src,
                unwrap_if_constexpr(message),
                self.builder.curr_node,  # type: ignore
                file_name=self.file_name,
                begin_line=self.begin_line,
            )
        return None

    statically_implemented_functions = {
        # dsl.static_assert: execute_static_assert,
        print: static_executor(print),
        len: static_executor(len),
    }


class EnterSubRegion:
    """Scoped helper that snapshots/restores frontend symbol state + insertion point."""

    def __init__(self, generator: MLIRCodeGenerator):
        self.generator = generator

    def __enter__(self):
        self.lscope = self.generator.lscope.copy()
        self.ip = self.generator.builder.save_insertion_point()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.lscope = self.lscope
        self.generator.builder.restore_insertion_point(self.ip)


def compile(
    fn: Kernel,
    arg_types: Sequence[TypeBase | str] = [],
    res_types: Sequence[TypeBase | str] = [],
    options: KernelOptions | None = None,
    show_traceback=False,
):
    """Compile a kernel function into an MLIR module."""
    import os

    if os.environ.get("ALLO_SHOW_COMPILER_TRACEBACK", "") == "1":
        show_traceback = True
    if not isinstance(fn, Kernel):
        raise TypeError(
            "Only allo.kernel functions can be compiled with allo.compile()"
        )
    if not arg_types:
        arg_types = fn.parse_argument_annotations()
    else:
        arg_types = [fn.parse_type_annotation(t) for t in arg_types]
    if len(arg_types) != len(fn.signature.parameters):
        raise ValueError(
            f"The number of provided argument types ({len(arg_types)}) does not match the number of arguments in the kernel signature ({len(fn.signature.parameters)})."
        )
    if not res_types:
        res_types = fn.parse_return_annotation()
    else:
        res_types = [fn.parse_type_annotation(t) for t in res_types]
    effective_options = fn.options if options is None else options

    try:
        context = Context()
        context.load_dialects()

        # initialize builder
        builder = AlloOpBuilder(context, typing_style=effective_options.typing_style)
        builder.src = fn.src
        builder.file_name = fn.file_name
        builder.begin_line = fn.begin_line
        builder.set_loc(Location(fn.file_name, fn.begin_line, 1, context))
        builder.curr_node = None
        module = ModuleOp(builder)
        builder.module = module
        builder.set_insertion_point_to_end(module.get_body())

        # start codegen
        generator = MLIRCodeGenerator(
            context,
            module,
            builder,
            kernel=fn,
            func_name=fn.func_name,
            file_name=fn.file_name,
            begin_line=fn.begin_line,
            gscope=fn.get_capture_scope(),
            arg_types=arg_types,
            res_types=res_types,
            options=effective_options,
            active_kernel_calls=[f"kernel:{fn.__module__}.{fn.__qualname__}"],
        )
        generator.visit(fn.parse())

        # verify
        if not module.verify():
            print(module)
            raise RuntimeError(
                f"In function: {fn.func_name}, module verification failed."
            )

        module.cse_and_canonicalize()
        fn.module = module
        # transfer the ownership of context to kernel
        fn.context = context
        return module
    except (StaticAssertionError, CompilationError) as exc:
        if show_traceback:
            raise
        else:
            raise exc.with_traceback(None) from None
    except Exception as exc:
        if show_traceback:
            raise
        raise RuntimeError(
            "Internal compiler error during Allo kernel compilation.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "Re-run with show_traceback=True to see the full traceback."
        ) from None
