# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import builtins

from dataclasses import dataclass

from types import ModuleType
from .._C.ir import (
    Context,
    ModuleOp,
    Location,
    FunctionType,
    Value,
    BlockArgument,
    Block,
)
from .._C import scf, func, ub as ub_d, arith, cf
from ..core.types import (
    BaseType,
    Constexpr,
    Proxy,
    ShapedType,
    BufferType,
    TensorType,
    DType,
    unwrap_if_constexpr,
    index,
)
from ..core import types
from .builder import AlloOpBuilder
from ..core.kernel import (
    Kernel,
    ConstevalFunction,
    CompileOptions,
    _infer_value_type,
    kernel as kernel_decorator,
)
from collections.abc import Sequence
from .. import dsl
from typing import Type, cast
from ..core.library import Operator, BoundOperator, NO_FOLD
from .errors import (
    CompilationError,
    CompileTimeAssertionFailure,
    raise_compilation_warning,
)


def serialize_function_signature(
    arg_types: Sequence[BaseType], ret_types: Sequence[BaseType], context: Context
) -> FunctionType:
    args = []
    for t in arg_types:
        if t == Constexpr:
            continue
        args.append(t.to_mlir(context))
    rets = []
    for t in ret_types:
        if t == Constexpr:
            continue
        rets.append(t.to_mlir(context))
    return FunctionType.get(args, rets, context)


class ReturnPlacementChecker(ast.NodeVisitor):
    def __init__(self, src: str):
        self.src = src
        self.loop_depth = 0
        self.if_depth = 0

    def visit_Return(self, ret_node: ast.Return):
        if self.loop_depth > 0:
            raise CompilationError(
                ret_node,
                "'return' is not supported inside loops (for/grid/while).",
                self.src,
            )
        if self.if_depth > 1:
            raise CompilationError(
                ret_node,
                "'return' is not supported inside nested 'if' statements.",
                self.src,
            )

    def visit_For(self, for_node: ast.For):
        self.loop_depth += 1
        self.generic_visit(for_node)
        self.loop_depth -= 1

    def visit_While(self, while_node: ast.While):
        self.loop_depth += 1
        self.generic_visit(while_node)
        self.loop_depth -= 1

    def visit_If(self, if_node: ast.If):
        self.if_depth += 1
        self.generic_visit(if_node)
        self.if_depth -= 1


@dataclass(frozen=True)
class NestedKernelSymbol:
    name: str
    node: ast.FunctionDef
    owner_func_name: str
    mapping_expr: ast.AST | None = None


class CodeGenerator(ast.NodeVisitor):
    def __init__(
        self,
        context: Context,
        module: ModuleOp,
        builder: AlloOpBuilder,
        kernel: Kernel,
        func_name: str,
        file_name: str,
        begin_line: int,
        gscope: dict,
        arg_types: Sequence[BaseType],
        res_types: Sequence[BaseType],
        options: CompileOptions = CompileOptions(),
        callee_context: dict[str, Proxy | Constexpr] | None = None,
        fscope: dict[str, NestedKernelSymbol | Kernel] | None = None,
        closure_scope: dict[str, object] | None = None,
        forbidden_closure_scope: dict[str, object] | None = None,
        active_nested_calls: list[str] | None = None,
    ):
        # setup basic fields and context
        self.context = context
        self.module = module
        self.builder = builder
        self.func_name = func_name
        self.file_name = file_name
        self.begin_line = begin_line
        self.kernel = kernel
        self.arg_types = arg_types
        self.res_types = list(res_types)
        self.actual_res_types: list[BaseType] = []
        self._actual_res_types_recorded = False
        self.seen_return_stmt = False
        self.options = options
        self._kernel_call_counter = 0
        self._kernel_base_names = set()
        self._entry_function_visited = False

        # trackers
        self.local_defs = {}  # track local variable definitions
        # track what can be seen in the current scope
        self.lscope: dict[str, Proxy | Constexpr] = (
            {} if callee_context is None else callee_context.copy()
        )
        # track callable symbols (nested kernels)
        self.fscope: dict[str, NestedKernelSymbol | Kernel] = (
            {} if fscope is None else fscope.copy()
        )
        # lexical closure values captured from outer kernels at call time
        self.closure_scope = {} if closure_scope is None else closure_scope.copy()
        self.forbidden_closure_scope = (
            {} if forbidden_closure_scope is None else forbidden_closure_scope.copy()
        )
        self._active_nested_calls = (
            [] if active_nested_calls is None else active_nested_calls
        )

        self.gscope = {}
        self.module_map = options.module_map
        for k, v in gscope.items():
            if isinstance(v, ModuleType):
                # module-level remap
                self.gscope[k] = self.module_map.get(v.__name__, v)
                continue
            module_name = getattr(v, "__module__", None)
            if module_name is not None and module_name in self.module_map:
                self.gscope[k] = getattr(self.module_map[module_name], v.__name__)
            else:
                self.gscope[k] = v
        self.scf_stack = []
        self.curr_fn = None
        self.builder.curr_node = None
        self.visiting_consteval = False
        self.visiting_default_args = False

        self.lookup = self._define_name_lookup()

        self.name_loc_prefix = None
        self.compile_error = self.builder.compile_error

    builtin_namespace = {
        range.__name__: types.range,
        min.__name__: dsl.min,
        max.__name__: dsl.max,
    }

    def _define_name_lookup(self):
        def local_lookup(name: str, absent):
            val = self.lscope.get(name, absent)
            if val is not absent:
                return val
            return self.fscope.get(name, absent)

        def closure_lookup(name: str, absent):
            val = self.closure_scope.get(name, absent)
            if val is not absent:
                return val
            if name in self.forbidden_closure_scope:
                captured = self.forbidden_closure_scope[name]
                if isinstance(captured, Proxy):
                    captured_ty = str(captured.type)
                else:
                    captured_ty = type(captured).__name__
                self.compile_error(
                    f"Invalid closure capture '{name}' in kernel '{self.func_name}'. Only BaseType, constexpr, and kernel symbols can be captured from outer scope, but got '{captured_ty}'."
                )
            return absent

        def global_lookup(name: str, absent):
            val = self.gscope.get(name, absent)
            if self._is_allowed_global_name(name, val, absent):
                return val
            self.compile_error(
                f"Cannot access global name '{name}' in current scope. Allo kernels can only access constexpr values, allo types from global scope, and imported modules."
            )

        absent_marker = object()

        def name_lookup(name: str):
            for lookup in (
                local_lookup,
                self.builtin_namespace.get,
                closure_lookup,
                global_lookup,
            ):
                val = lookup(name, absent_marker)
                if val is not absent_marker:
                    return val
            self.compile_error(f"Name '{name}' is not defined in current scope.")

        return name_lookup

    def _is_global_constexpr(self, name: str) -> bool:
        marker = object()
        val = self.gscope.get(name, marker)
        if val is marker:
            return False
        return isinstance(val, Constexpr)

    def _is_allowed_global_name(self, name: str, val: object, absent):
        if val is absent:
            return False
        if name in self.builtin_namespace:
            return True
        if self.visiting_consteval or self.visiting_default_args:
            return True

        allowed = (
            isinstance(val, ModuleType)
            or isinstance(val, ConstevalFunction)
            or isinstance(val, Kernel)
            or getattr(val, "__module__", "").startswith("allo.experimental.core")
            or isinstance(val, (Operator, BoundOperator))
            or isinstance(val, BaseType)
            or self._is_global_constexpr(name)
        )
        return allowed

    @contextlib.contextmanager
    def _name_loc_prefix(self, prefix):
        self.name_loc_prefix = prefix
        yield
        self.name_loc_prefix = None

    def _set_value(self, name: str, value: Proxy | Constexpr):
        self.lscope[name] = value
        self.local_defs[name] = value

    def _maybe_set_loc_to_name(self, val, name):
        if isinstance(val, (BlockArgument, Value)):
            named_loc = Location(val.get_loc(), name, self.context)
            val.set_loc(named_loc)
        elif isinstance(val, Proxy):
            named_loc = Location(val.handle.get_loc(), name, self.context)
            val.handle.set_loc(named_loc)

    ##################
    # Visitor methods
    ##################

    def visit(self, node: ast.AST):
        if node is None:
            return
        last_node = self.builder.curr_node
        last_loc = self.builder.get_loc()
        # update
        self.builder.curr_node = node
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            new_loc = Location(
                self.file_name,
                self.begin_line + node.lineno - 1,
                node.col_offset + 1,
                self.context,
            )
            if self.name_loc_prefix is not None:
                new_loc = Location(new_loc, self.name_loc_prefix, self.context)
            self.builder.set_loc(new_loc)
        # visit
        try:
            return super().visit(node)
        finally:
            # restore
            self.builder.curr_node = last_node
            self.builder.set_loc(last_loc)

    def generic_visit(self, node: ast.AST):
        self.compile_error(f"Unsupported syntax: {ast.unparse(node)}")

    def visit_compound_stmts(self, stmts, allow_nested_kernel_def: bool = False):
        if not isinstance(stmts, builtins.list):
            stmts = [stmts]
        for stmt in stmts:
            if isinstance(stmt, ast.FunctionDef):
                if not allow_nested_kernel_def:
                    self.compile_error(
                        "Nested kernel definitions are only supported at the top level of a kernel/accelerator body."
                    )
                self.visit(stmt)
                continue
            # ignore everything after return statement
            self.visit(stmt)
            if isinstance(stmt, ast.Return):
                break

    def visit_Module(self, node: ast.Module):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self._entry_function_visited:
            self._entry_function_visited = True
            self._visit_entry_function_def(node)
            return
        self._register_nested_kernel_def(node)

    def _visit_entry_function_def(self, node: ast.FunctionDef):
        self._precheck_return_placement(node)
        self.seen_return_stmt = False

        arg_names, _ = self.visit(node.args)
        # init defaults
        for i, default in enumerate(node.args.defaults[::-1]):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            if name in self.lscope:
                # Value provided by callsite binding already wins over default.
                continue
            st_target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                self.compile_error(
                    f"Default arguments must have type annotations to allow type inference. Please provide an explicit type annotation for the default argument '{name}'."
                )
            else:
                init_node = ast.AnnAssign(
                    target=st_target, annotation=annotation, value=default, simple=1
                )
            try:
                self.visiting_default_args = True
                self.visit(init_node)
            finally:
                self.visiting_default_args = False

        fn_ty = serialize_function_signature(
            self.arg_types, self.res_types, self.context
        )
        func_op = func.FuncOp(self.builder, self.func_name, fn_ty)
        if self.curr_fn is None:
            self.curr_fn = func_op
        entry_block = func_op.add_entry_block()

        arg_handles = entry_block.get_args()
        arg_idx = 0
        for name, ty in zip(arg_names, self.arg_types):
            if ty == Constexpr:
                callee_val = self.lscope.get(name, None)
                if not isinstance(callee_val, Constexpr):
                    self.compile_error(
                        f"Missing constexpr argument binding for parameter '{name}' in function '{self.func_name}'."
                    )
                continue
            if arg_idx >= len(arg_handles):
                self.compile_error(
                    f"Internal error: argument count mismatch while lowering function '{self.func_name}'."
                )
            proxy = Proxy(arg_handles[arg_idx], ty)
            arg_idx += 1
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)
        if arg_idx != len(arg_handles):
            self.compile_error(
                f"Internal error: unbound function arguments remain while lowering function '{self.func_name}'."
            )

        # visit function body
        self.builder.set_insertion_point_to_start(entry_block)
        self.visit_compound_stmts(node.body, allow_nested_kernel_def=True)

        if not self.seen_return_stmt:
            if len(self.res_types) == 0:
                self.builder.set_insertion_point_to_end(entry_block)
                func.ReturnOp(self.builder, [])
            else:
                self.compile_error(
                    "Missing return statement for non-void function. Please add a top-level return statement matching the declared return type."
                )
        # restore
        self.builder.set_insertion_point_after(func_op.get_operation())

    def _register_nested_kernel_def(self, node: ast.FunctionDef):
        if len(node.decorator_list) == 0:
            self.compile_error(
                f"Nested function '{node.name}' is not allowed. Nested functions must use bare '@kernel' decorator."
            )
        if len(node.decorator_list) != 1:
            self.compile_error(
                f"Nested function '{node.name}' must use exactly one '@kernel' decorator."
            )
        decorator = node.decorator_list[0]
        mapping_expr = None
        if isinstance(decorator, ast.Call):
            if decorator.args:
                raise_compilation_warning(
                    f"Nested kernel '{node.name}' got unexpected positional arguments in decorator and will be ignored.",
                )
            for kw in decorator.keywords:
                if kw.arg == "mapping":
                    mapping_expr = kw.value
                else:
                    raise_compilation_warning(
                        f"Nested kernel '{node.name}' got unexpected keyword argument '{kw.arg}' in decorator and will be ignored. Compile options are inherited from the parent kernel can cannot be overridden at nested kernel level.",
                    )
            decorator = self.visit(decorator.func)
        if isinstance(decorator, ast.Name) or isinstance(decorator, ast.Attribute):
            decorator = self.visit(decorator)
            mapping_expr = None

        if decorator is not kernel_decorator:
            self.compile_error(
                f"Nested function '{node.name}' is not allowed. Only allo kernels are supported for nested definitions."
            )

        if node.name in self.lscope or node.name in self.fscope:
            self.compile_error(
                f"Nested kernel name '{node.name}' conflicts with an existing local symbol."
            )
        self.fscope[node.name] = NestedKernelSymbol(
            name=node.name,
            node=node,
            owner_func_name=self.func_name,
            mapping_expr=mapping_expr,
        )

    def _precheck_return_placement(self, node: ast.FunctionDef):
        checker = ReturnPlacementChecker(self.kernel.src)
        for stmt in node.body:
            checker.visit(stmt)

    def visit_arguments(self, node: ast.arguments):
        arg_names = [self.visit(arg) for arg in node.args]
        kwargs_names = self.visit(node.kwarg)
        return arg_names, kwargs_names

    def visit_arg(self, node: ast.arg):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_keyword(self, node: ast.keyword):
        return node.arg, self.visit(node.value)

    def visit_Constant(self, node: ast.Constant):
        return Constexpr(node.value)

    def visit_Expr(self, node: ast.Expr):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return builtins.slice(lower, upper, step)

    def visit_Pass(self, node: ast.Pass):
        pass

    def visit_Compare(self, node: ast.Compare):
        if not (len(node.ops) == 1 and len(node.comparators) == 1):
            self.compile_error("simultaneous multi-way comparisons are not supported")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        library_op = self._available_comparison_methods.get(type(node.ops[0]), None)
        if library_op is None:
            self.compile_error(
                f"Unsupported comparison operator '{type(node.ops[0]).__name__}' in allo kernel functions",
            )
        return self._apply_binary_method(library_op, lhs, rhs)

    _available_comparison_methods: dict[Type[ast.cmpop], Operator] = {
        ast.Eq: dsl.eq,
        ast.NotEq: dsl.ne,
        ast.Lt: dsl.lt,
        ast.LtE: dsl.le,
        ast.Gt: dsl.gt,
        ast.GtE: dsl.ge,
    }

    def _apply_binary_method(self, library_op, lhs, rhs):
        return self.call_operator(library_op, [lhs, rhs], {})

    def _ast_expr_may_be_float(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, float)

        if isinstance(node, ast.Name):
            val = unwrap_if_constexpr(self.lookup(node.id))
            if isinstance(val, Proxy):
                return isinstance(val.dtype, DType) and val.dtype.is_float()
            if isinstance(val, Constexpr):
                return isinstance(val.value, float)
            return False

        if isinstance(node, ast.Subscript):
            return self._ast_expr_may_be_float(node.value)

        if isinstance(node, ast.UnaryOp):
            return self._ast_expr_may_be_float(node.operand)

        if isinstance(node, ast.BinOp):
            return self._ast_expr_may_be_float(
                node.left
            ) or self._ast_expr_may_be_float(node.right)

        if isinstance(node, ast.Call):
            # Be conservative for call sites where return type cannot be inferred
            # without lowering.
            return True

        return False

    def _lower_binop_tree(self, node: ast.BinOp):
        def lower_expr(expr):
            if isinstance(expr, ast.BinOp):
                lhs = lower_expr(expr.left)
                rhs = lower_expr(expr.right)
                library_op = self._available_binary_methods.get(type(expr.op), None)
                if library_op is None:
                    self.compile_error(
                        f"Unsupported binary operator '{type(expr.op).__name__}' in allo kernel functions",
                    )
                return self._apply_binary_method(library_op, lhs, rhs)
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
            if isinstance(term, Proxy):
                anchor = term.dtype
                break
        if anchor is None:
            return terms

        materialized = []
        for term in terms:
            if isinstance(term, Constexpr):
                materialized.append(self.builder.cast(term, anchor))
            else:
                materialized.append(term)
        return materialized

    def _lower_nary_add_sub(self, node: ast.BinOp):
        signed_terms = []
        self._collect_add_sub_terms(node, sign=1, out=signed_terms)
        values = [value for value, _ in signed_terms]
        signs = [sign for _, sign in signed_terms]

        if all(isinstance(value, Constexpr) for value in values):
            total = 0
            for value, sign in zip(values, signs):
                total += sign * value.value
            return Constexpr(total)

        values = self._materialize_constexpr_terms(values)
        if not all(isinstance(value, Proxy) for value in values):
            self.compile_error(
                "n-ary add/sub lowering expects runtime values to be Proxies"
            )

        if all(isinstance(value.type, DType) for value in values):
            dtypes = [value.dtype for value in values]
            op_name = "sub" if any(sign < 0 for sign in signs) else "add"
            dst_ty = self.builder.get_promoted_dtype_nary(
                op_name, dtypes, term_signs=signs
            )
            casted = [self.builder.scalar_cast(value, dst_ty) for value in values]
            floating = dst_ty.is_float()
            if any(sign < 0 for sign in signs):
                return self.builder.create_sub_nary(casted, signs, floating=floating)
            return self.builder.create_add_nary(casted, floating=floating)

        normalized = []
        for value, sign in zip(values, signs):
            if sign < 0:
                normalized.append(self.call_operator(dsl.neg, [value], {}))
            else:
                normalized.append(value)
        return self.builder.reduce_balanced(
            normalized, lambda lhs, rhs: self._apply_binary_method(dsl.add, lhs, rhs)
        )

    def _lower_nary_mul(self, node: ast.BinOp):
        terms = []
        self._collect_mul_terms(node, terms)

        if all(isinstance(term, Constexpr) for term in terms):
            product = 1
            for term in terms:
                product *= term.value
            return Constexpr(product)

        terms = self._materialize_constexpr_terms(terms)
        if not all(isinstance(term, Proxy) for term in terms):
            self.compile_error(
                "n-ary mul lowering expects runtime values to be Proxies"
            )

        if all(isinstance(term.type, DType) for term in terms):
            dtypes = [term.dtype for term in terms]
            dst_ty = self.builder.get_promoted_dtype_nary("mul", dtypes)
            casted = [self.builder.scalar_cast(term, dst_ty) for term in terms]
            return self.builder.create_mul_nary(casted, floating=dst_ty.is_float())

        return self.builder.reduce_balanced(
            terms, lambda lhs, rhs: self._apply_binary_method(dsl.mul, lhs, rhs)
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
            self.compile_error(
                f"Unsupported binary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self._apply_binary_method(library_op, lhs, rhs)

    _available_binary_methods: dict[Type[ast.operator], Operator] = {
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
            self.compile_error(
                f"Unsupported unary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operator(fn, [operand], {})

    _available_unary_methods: dict[Type[ast.unaryop], Operator] = {
        ast.UAdd: dsl.pos,
        ast.USub: dsl.neg,
        ast.Not: dsl.logical_not,
        ast.Invert: dsl.invert,
    }

    def visit_BoolOp(self, node):
        library_op = self._available_boolop_methods.get(type(node.op), None)
        if library_op is None:
            self.compile_error(
                f"Unsupported boolean operator '{type(node.op).__name__}' in allo kernel functions",
            )
        nontrivial_values = []

        for subnode in node.values:
            value = self.visit(subnode)
            if isinstance(value, Constexpr):
                # constant folding
                bv = bool(unwrap_if_constexpr(value))
                if (bv is False) and (library_op is dsl.logical_and):
                    return Constexpr(False)
                if (bv is True) and (library_op is dsl.logical_or):
                    return Constexpr(True)
                # otherwise constexpr has no effect, so can be skipped
            elif isinstance(value, Proxy) and isinstance(value.type, ShapedType):
                self.compile_error(
                    "non-scalar values are not supported in boolean operations"
                )
            else:
                nontrivial_values.append(value)

        if len(nontrivial_values) == 0:
            # all values are constant folded
            if library_op == dsl.logical_and:
                return Constexpr(True)
            else:
                return Constexpr(False)

        while len(nontrivial_values) >= 2:
            # reduce from left to right
            rhs = nontrivial_values.pop()
            lhs = nontrivial_values.pop()
            res = self._apply_binary_method(library_op, lhs, rhs)
            nontrivial_values.append(res)

        assert len(nontrivial_values) == 1
        return nontrivial_values[0]

    _available_boolop_methods: dict[Type[ast.boolop], Operator] = {
        ast.And: dsl.logical_and,
        ast.Or: dsl.logical_or,
    }

    ##########################
    # Control Flow Statements
    ##########################

    def visit_Break(self, node):
        self.compile_error(
            "'break' statement is not supported in allo kernel functions"
        )

    def visit_Continue(self, node):
        self.compile_error(
            "'continue' statement is not supported in allo kernel functions"
        )

    def visit_Return(self, node: ast.Return):
        self.seen_return_stmt = True

        if node.value is None or (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            return_vals = []
        elif isinstance(node.value, ast.Tuple):
            return_vals = [self.visit(elt) for elt in node.value.elts]
        else:
            return_vals = [self.visit(node.value)]

        curr_actual_res_types = []
        for i, value in enumerate(return_vals):
            if isinstance(value, Proxy):
                curr_actual_res_types.append(value.type)
            elif isinstance(value, Constexpr):
                if i < len(self.res_types):
                    curr_actual_res_types.append(self.res_types[i])
            else:
                self.compile_error(
                    f"Unsupported return value '{value}' of type '{type(value).__name__}'."
                )

        if not self._actual_res_types_recorded:
            self.actual_res_types = curr_actual_res_types
            self._actual_res_types_recorded = True

        if len(return_vals) != len(self.res_types):
            self.compile_error(
                f"Return value count mismatch: expected {len(self.res_types)}, got {len(return_vals)}."
            )

        coerced = []
        for value, dst_type in zip(return_vals, self.res_types):
            coerced.append(self.builder.cast(value, dst_type))

        func.ReturnOp(self.builder, [v.handle for v in coerced])

    def visit_If(self, node: ast.If):
        cond = self.visit(node.test)
        if isinstance(cond, Proxy):
            cond = self.builder.as_condition_scalar(cond, kind="if")
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
            assert isinstance(cond, Constexpr)
            cond = unwrap_if_constexpr(cond)
            if type(cond) not in self._condition_types:
                self.compile_error(
                    "`if` conditionals can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in self._condition_types),
                        type(cond).__name__,
                    ),
                )
            selected = node.body if cond else node.orelse
            self.visit_compound_stmts(selected)

    _condition_types = {
        bool,
        int,
        type(None),
    }

    @staticmethod
    def _branch_has_return(stmts) -> bool:
        return any(isinstance(stmt, ast.Return) for stmt in stmts)

    def _visit_if_with_return_impl(
        self, cond: Proxy, node: ast.If, then_has_return, else_has_return
    ):
        continue_vals = None
        end_if = None
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()
            parent_region = ip.get_block().get_parent_region()
            then_block = self.builder.create_block(parent_region)
            else_block = self.builder.create_block(parent_region)
            end_if = self.builder.create_block(parent_region)

            # branch out from current block to then/else
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            cf.CondBranchOp(self.builder, cond.handle, then_block, else_block)

            liveins = self.lscope.copy()

            # then branch
            self.builder.set_insertion_point_to_start(then_block)
            self.visit_compound_stmts(node.body)
            then_vals = self.lscope.copy()

            # else branch
            self.lscope = liveins
            self.builder.set_insertion_point_to_start(else_block)
            if node.orelse:
                self.visit_compound_stmts(node.orelse)
                else_vals = self.lscope.copy()
            else:
                else_vals = liveins.copy()

            # if both branches return, there is no fallthrough path
            if then_has_return and else_has_return:
                continue_vals = liveins
                end_if.erase()

            # if exactly one branch returns, continue with the non-returning branch.
            elif then_has_return and not else_has_return:
                self.builder.set_insertion_point_to_end(else_block)
                cf.BranchOp(self.builder, end_if, [])
                continue_vals = else_vals

            elif not then_has_return and else_has_return:
                self.builder.set_insertion_point_to_end(then_block)
                cf.BranchOp(self.builder, end_if, [])
                continue_vals = then_vals

            else:
                self.compile_error(
                    "Internal error: expected at least one direct return in if/else branches."
                )

        assert end_if is not None and continue_vals is not None
        self.builder.set_insertion_point_to_start(end_if)
        self.lscope = continue_vals.copy()

    def visit_if_impl(self, cond: Proxy, node: ast.If):
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
            phi_ir_types = [ty.to_mlir(self.context) for ty in phi_types]
            if_op = scf.IfOp(self.builder, phi_ir_types, cond.handle, has_else)
            then_block.merge_before(if_op.get_then_block())
            then_block = if_op.get_then_block()
            then_block.remove_terminator()  # remove the default created
            self.builder.set_insertion_point_to_end(then_block)
            scf.YieldOp(self.builder, then_handles)
            if has_else:
                else_block.merge_before(if_op.get_else_block())
                else_block = if_op.get_else_block()
                else_block.remove_terminator()  # remove the default created
                self.builder.set_insertion_point_to_end(else_block)
                scf.YieldOp(self.builder, else_handles)
            else:
                else_block.erase()

        # update lscope with phi results
        res_handles = if_op.get_results()
        phi_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, phi_types)]
        for name, proxy in zip(phi_names, phi_proxies):
            self._set_value(name, proxy)
            self._maybe_set_loc_to_name(proxy, name)

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
        phi_types: list[BaseType] = []
        then_handles = []
        else_handles = []
        for name, value in liveins.items():
            then_proxy = then_vals[name]
            else_proxy = else_vals[name]
            if not isinstance(then_proxy, Proxy) or not isinstance(else_proxy, Proxy):
                continue
            then_handle = then_proxy.handle
            else_handle = else_proxy.handle
            if then_handle == else_handle:
                continue  # value is not redefined in either block, no need for phi
            # type check
            if isinstance(value, Constexpr):
                self.compile_error(
                    f"Variable '{name}' is defined as a constexpr in the outer scope, but is assigned to non-constexpr values in the then vs else branches."
                )
            outer_ty = value.handle.get_type()
            then_ty = then_handle.get_type()
            else_ty = else_handle.get_type()
            if then_ty != else_ty or then_ty != outer_ty:
                self.compile_error(
                    f"Variable '{name}' has incompatible types in outer scope vs then vs else branches: {outer_ty} vs {then_ty} vs {else_ty}."
                )
            phi_types.append(then_proxy.type)
            phi_names.append(name)
            then_handles.append(then_handle)
            else_handles.append(else_handle)
        return phi_names, phi_types, then_handles, else_handles

    def visit_IfExp(self, node: ast.IfExp):
        cond = self.visit(node.test)
        if isinstance(cond, Proxy):
            cond = self.builder.as_condition_scalar(cond, kind="ifexp")
            # if exp cannot define new variables
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            then_val = self.visit(node.body)
            else_val = self.visit(node.orelse)

            # type check
            # Case 1: both branches are constexprs
            then_is_constexpr = isinstance(then_val, Constexpr)
            else_is_constexpr = isinstance(else_val, Constexpr)
            if then_is_constexpr and else_is_constexpr:
                self.compile_error(
                    f"Cannot deduce type of ternary expression because both branches are constexprs. Please use if statement instead of if expression in this case, or make sure at least one branch is non-constexpr so that the type can be deduced."
                )
            # Case 2: both branches are Proxies
            if not then_is_constexpr and not else_is_constexpr:
                if then_val.type != else_val.type:
                    self.compile_error(
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
            sel_op = arith.SelectOp(
                self.builder, cond.handle, then_val.handle, else_val.handle
            )
            return Proxy(sel_op, res_type)
        else:
            # constexpr path
            assert isinstance(cond, Constexpr)
            cond = unwrap_if_constexpr(cond)
            if type(cond) not in self._condition_types:
                self.compile_error(
                    "Ternary expression conditionals can only accept values of type {{{{}}}, not objects of type {}".format(
                        ", ".join(_.__name__ for _ in self._condition_types),
                        type(cond).__name__,
                    ),
                )
            selected = node.body if cond else node.orelse
            return self.visit(selected)

    def visit_For(self, node: ast.For):
        if node.orelse:
            self.compile_error("'for' statement with 'else' block is not supported")
        if not isinstance(node.iter, ast.Call):
            self.compile_error("Only 'for' loops over 'range()' are supported")

        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = {kw.arg: self.visit(kw.value) for kw in node.iter.keywords}

        if IteratorClass is types.range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            lb = iterator.start
            ub = iterator.stop
            step = iterator.step
        elif IteratorClass is types.grid:
            iterator = IteratorClass(*iter_args, **iter_kwargs)
            return self.visit_Grid(node, iterator)
        else:
            self.compile_error(
                "Only 'for' loops over 'range()' and 'grid()' are supported"
            )

        if not isinstance(node.target, ast.Name):
            self.compile_error("loop target must be a single variable in 'for' loops")

        if not (isinstance(step, Constexpr) and step.value > 0):
            self.compile_error("loop step must be a positive integer in 'for' loops")

        lb, ub, step = self.builder.normalize_indices((lb, ub, step), expected_len=3)

        with EnterSubRegion(self):
            index_ty = index.to_mlir(self.context)
            iv_placeholder = ub_d.PoisonOp(self.builder, index_ty)
            self._set_value(node.target.id, Proxy(iv_placeholder, index))

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            name, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore={node.target.id}
            )
            # create for op
            for_op = scf.ForOp(
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
                Proxy(handle, ty) for handle, ty in zip(block_handles, init_types)
            ]
            for name, proxy in zip(name, block_args):
                self._maybe_set_loc_to_name(proxy, name)
                self._set_value(name, proxy)
            # visit loop body
            self.visit_compound_stmts(node.body)
            self.scf_stack.pop()
            # create yield
            yield_handles = [self.lscope[name].handle for name in name]
            self.builder.set_insertion_point_to_end(for_op_body)
            # remove the default terminator
            for_op_body.remove_terminator()
            scf.YieldOp(self.builder, yield_handles)
            assert for_op.get_num_regions() == 1

            # update induction variable with the actual one
            iv = for_op.get_induction_var()
            iv_placeholder.get_result_at(0).replace_all_uses_with(iv)
            iv_placeholder.erase()
            self._set_value(node.target.id, Proxy(iv, index))
            self._maybe_set_loc_to_name(iv, node.target.id)

        # update lscope with iter args
        res_handles = for_op.get_results()
        res_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, init_types)]
        for name, proxy in zip(name, res_proxies):
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)

    def _test_loop_iter_args(self, node, liveins: dict, ignore: set[str]):
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        # create dummy block
        block = self.builder.create_block(ip.get_block().get_parent_region())
        self.builder.set_insertion_point_to_start(block)
        # dry visit
        self.scf_stack.append(node)
        self.visit_compound_stmts(node.body)
        self.scf_stack.pop()
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
            if isinstance(livein, Constexpr):
                continue
            assert isinstance(livein, Proxy)
            loop_val = self.lscope[name]
            if loop_val.handle == livein.handle:
                continue  # variable is not assigned in the loop body
            # type check
            if type(loop_val) != type(livein) or loop_val.type != livein.type:
                self.compile_error(
                    f"Loop variable '{name}' has incompatible types in outer scope vs loop body: {livein.type} vs {loop_val.type}."
                )
            names.append(name)
            init_handles.append(livein.handle)
            init_types.append(livein.type)

        # restore lscope
        self.lscope = liveins.copy()
        return names, init_handles, init_types

    def visit_Grid(self, node: ast.For, iterator: types.grid):
        if not isinstance(node.target, ast.Tuple):
            self.compile_error(
                "loop target must be a tuple of variables in 'for' loops over 'grid()'"
            )
        if len(node.target.elts) != len(iterator.starts):
            self.compile_error(
                f"loop target must have the same number of variables as the dimensions of the grid iterator. Expected {len(iterator.starts)} variables, but got {len(node.target.elts)}."
            )

        lbs = iterator.starts
        ubs = iterator.stops
        steps = iterator.steps

        if not all(isinstance(s, Constexpr) and s.value > 0 for s in steps):
            self.compile_error("loop step must be a positive integer in 'for' loops")

        lb_proxies = self.builder.normalize_indices(lbs)
        ub_proxies = self.builder.normalize_indices(ubs)
        step_proxies = self.builder.normalize_indices(steps)

        with EnterSubRegion(self):
            index_ty = index.to_mlir(self.context)
            iv_placeholders = [ub_d.PoisonOp(self.builder, index_ty) for _ in lbs]
            for i, target in enumerate(node.target.elts):
                if not isinstance(target, ast.Name):
                    self.compile_error(
                        "loop target must be a single variable in 'for' loops over 'grid()'"
                    )
                self._set_value(target.id, Proxy(iv_placeholders[i], index))

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore={target.id for target in node.target.elts}
            )
            if len(init_handles) > 0:
                raise NotImplementedError(
                    f"Non-trivial loop-carried dependencies are not supported in 'for' loops over 'grid()' at this moment."
                )
            # create parallel op
            par_op = scf.ParallelOp(
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
                self._set_value(target.id, Proxy(iv, index))
                self._maybe_set_loc_to_name(iv, target.id)

        # update lscope with iter args
        res_handles = par_op.get_results()
        res_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, init_types)]
        for name, proxy in zip(names, res_proxies):
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)

    def visit_While(self, node: ast.While):
        if node.orelse:
            self.compile_error("'while' statement with 'else' block is not supported")
        with EnterSubRegion(self):
            liveins = self.lscope.copy()
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore=set()
            )
            # create while op
            init_ir_types = [ty.to_mlir(self.context) for ty in init_types]
            while_op = scf.WhileOp(self.builder, init_ir_types, init_handles)

            # create before region
            before_block = self.builder.create_block(
                while_op.get_before(), init_ir_types
            )
            self.builder.set_insertion_point_to_start(before_block)
            block_args = before_block.get_args()
            for name, arg, ty in zip(names, block_args, init_types):
                proxy = Proxy(arg, ty)
                self._maybe_set_loc_to_name(proxy, name)
                self._set_value(name, proxy)

            # visit condition
            cond = self.visit(node.test)
            self.builder.set_insertion_point_to_end(before_block)
            assert isinstance(cond, Proxy)
            # create cond
            scf.ConditionOp(self.builder, cond.handle, block_args)

            # create after region
            after_block = self.builder.create_block(while_op.get_after(), init_ir_types)
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = after_block.get_args()
            for name, arg, ty in zip(names, body_handles, init_types):
                proxy = Proxy(arg, ty)
                self._maybe_set_loc_to_name(proxy, name)
                self._set_value(name, proxy)

            # visit loop body
            self.scf_stack.append(node)
            self.visit_compound_stmts(node.body)
            self.scf_stack.pop()

            # create yield
            yield_handles = [self.lscope[name].handle for name in names]
            self.builder.set_insertion_point_to_end(after_block)
            # remove the default terminator
            after_block.remove_terminator()
            scf.YieldOp(self.builder, yield_handles)

        # update lscope with iter args
        res_handles = while_op.get_results()
        res_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, init_types)]
        for name, proxy in zip(names, res_proxies):
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)

    #####################
    # Assignments
    #####################

    def _set_value_with_loc(self, target, proxy):
        self._set_value(target, proxy)
        self._maybe_set_loc_to_name(proxy, target)

    def visit_AnnAssign(self, node):
        annotation = self.visit(node.annotation)
        parsed_type = self.kernel.parse_type_annotation(annotation)

        target = node.target
        if isinstance(target, ast.Attribute):
            self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if isinstance(target, ast.Name) and node.value is not None:
            with self._name_loc_prefix(target.id):
                value = self.visit(node.value)
        else:
            value = self.visit(node.value) if node.value else None

        target = self.visit(target)
        if target in self.lscope:
            self.compile_error(
                f"Variable '{target}' is already defined in the current scope."
            )

        if value is None:
            if parsed_type == Constexpr:
                self.compile_error(
                    f"Constexpr variables must be initialized with a constant value. Please provide an initializer for this variable."
                )
            proxy = parsed_type.make_default(self.builder)
            self._set_value_with_loc(target, proxy)
        else:
            if parsed_type == Constexpr:
                if isinstance(value, Constexpr):
                    self._set_value(target, value)
                    return
                if isinstance(value, Proxy):
                    self.compile_error(
                        f"Unsupported assignment with type annotation '{annotation}' and value of type '{value.type}'."
                    )
                self.compile_error(
                    f"Unsupported initializer for variable assignment with type annotation '{annotation}'."
                )

            if not isinstance(value, (Proxy, Constexpr)):
                self.compile_error(
                    f"Unsupported initializer for variable assignment with type annotation '{annotation}'."
                )

            if (
                isinstance(parsed_type, BufferType)
                and isinstance(value, Proxy)
                and isinstance(value.type, BufferType)
            ):
                self.compile_error(
                    "Direct assignment between buffer types is not supported. If you want to copy data from one buffer to another, please use 'copy' operator to fill the target buffer with the source buffer."
                )

            try:
                proxy = self.builder.cast(value, parsed_type)
            except CompilationError:
                value_ty = value.type if isinstance(value, Proxy) else "constexpr"
                self.compile_error(
                    f"Unsupported assignment with type annotation '{annotation}' and value of type '{value_ty}'."
                )
            self._set_value_with_loc(target, proxy)

    def visit_Assign(self, node: ast.Assign):
        targets = node.targets
        if len(targets) != 1:
            self.compile_error("multiple assignment targets are not supported")
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
            assert isinstance(value, tuple)
            for i, elt in enumerate(target.elts):
                self._do_assignment(elt, value[i])
            return
        if isinstance(target, ast.Attribute):
            self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if isinstance(target, ast.Name):
            target = self.visit(target)
            # the first time we see a variable is considered its definition site, and its type if inferred from the assigned value. subsequent assignments to the same variable must be type-compatible with the first definition.
            if target not in self.lscope:
                if not self.options.allow_implicit_type_infer:
                    self.compile_error(
                        f"Cannot infer type for a new variable {target} without an initializer. Please provide an explicit type annotation for this variable."
                    )
                if isinstance(value, Constexpr):
                    self.compile_error(
                        "Constexpr variables must be explcitly declared with type annotation. Please add a type annotation of 'constexpr' to this variable."
                    )
                self._set_value_with_loc(target, value)
                return
            proxy = self.lscope[target]
            if isinstance(proxy, Constexpr):
                self.compile_error(
                    f"Cannot reassign to variable '{target}' defined as a constexpr"
                )
            if isinstance(value, Constexpr):
                ret = self.builder.make_scalar_or_shaped(value.value, proxy)
            if isinstance(value, Proxy):
                try:
                    ret = self.builder.cast(value, proxy.type)
                except CompilationError:
                    self.compile_error(
                        f"Cannot assign value of type '{value.type}' to variable '{target}' of type '{proxy.type}'."
                    )
            if ret is not None:
                self._maybe_set_loc_to_name(ret, target)
                self._set_value(target, ret)

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

    def visit_ListComp(self, node):
        if len(node.generators) != 1:
            raise CompilationError(node, "nested list comprehensions are not supported")
        comp = node.generators[0]
        iter = self.visit(comp.iter)
        if not isinstance(iter, tuple):
            self.compile_error(
                "only tuple iteration is supported in list comprehensions"
            )

        results = []
        for item in iter:
            if not isinstance(comp.target, ast.Name):
                self.compile_error(
                    "only simple variable targets are supported in list comprehensions",
                )
            self._set_value(comp.target.id, item)
            results.append(self.visit(node.elt))
        return tuple(results)

    def visit_JoinedStr(self, node):
        values = list(node.values)
        for i, value in enumerate(values):
            if isinstance(value, ast.Constant):
                values[i] = str(value.value)
            elif isinstance(value, ast.FormattedValue):
                conversion_code = value.conversion
                evaluated = self.visit(value.value)
                if not isinstance(evaluated, Constexpr):
                    self.compile_error(
                        "Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type "
                        + str(type(evaluated)),
                    )
                values[i] = (
                    "{}" if conversion_code < 0 else "{!" + chr(conversion_code) + "}"
                ).format(evaluated.value)
            else:
                assert False, f"unexpected value type in JoinedStr: {type(value)}"
        return "".join(values)

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if isinstance(lhs, ModuleType):
            # follow module_map until reaching fixed-point:
            while (name := lhs.__name__) in self.module_map:
                lhs = self.module_map[name]
                if lhs.__name__ == name:
                    break
        return getattr(lhs, node.attr)

    def visit_Subscript(self, node: ast.Subscript):
        return self.visit_Subscript_Load(node)

    def visit_Subscript_Store(self, node, value):
        assert isinstance(node.ctx, ast.Store)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        slices = tuple([slices]) if isinstance(slices, (Proxy, Constexpr)) else slices
        return self.call_operator(dsl.store, [lhs, slices, value], {})

    def visit_Subscript_Load(self, node):
        assert isinstance(node.ctx, ast.Load)
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        if isinstance(lhs, tuple) and isinstance(slices, Constexpr):
            return lhs[slices.value]
        slices = tuple([slices]) if isinstance(slices, (Proxy, Constexpr)) else slices
        return self.call_operator(dsl.load, [lhs, slices], {})

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
                return Constexpr(ret)
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

    def _next_called_kernel_name(self, fn: Kernel | NestedKernelSymbol | str) -> str:
        if isinstance(fn, Kernel):
            callee_name = fn.func_name
        elif isinstance(fn, NestedKernelSymbol):
            callee_name = fn.name
        else:
            callee_name = fn
        call_id = self._kernel_call_counter
        self._kernel_call_counter += 1
        base_name = f"{self.func_name}.{callee_name}"
        if base_name in self._kernel_base_names:
            return base_name + f".{call_id}"
        self._kernel_base_names.add(base_name)
        return base_name

    def _build_kernel_call_operand(self, value, expected_ty: BaseType, arg_name: str):
        if isinstance(value, Proxy):
            if value.type != expected_ty:
                self.compile_error(
                    f"Kernel call argument type mismatch for '{arg_name}': expected '{expected_ty}', got '{value.type}'."
                )
            return value.handle

        if not isinstance(value, Constexpr):
            value = Constexpr(value)

        try:
            return self.builder.cast(value, expected_ty).handle
        except CompilationError:
            self.compile_error(
                f"Kernel call argument '{arg_name}' has unsupported destination type '{expected_ty}'."
            )

    def _decode_kernel_call_results(self, call_op, res_types: list[BaseType]):
        if len(res_types) == 0:
            return None
        if call_op.get_num_results() != len(res_types):
            self.compile_error(
                f"Kernel call result count mismatch: expected {len(res_types)}, got {call_op.get_num_results()}."
            )
        results = [
            Proxy(handle, ty) for handle, ty in zip(call_op.get_results(), res_types)
        ]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _resolve_annotation_symbol(self, annotation: ast.AST):
        if isinstance(annotation, ast.Name):
            return self.lookup(annotation.id)
        if isinstance(annotation, ast.Attribute):
            base = self._resolve_annotation_symbol(annotation.value)
            return getattr(base, annotation.attr)
        self.compile_error(
            f"Unsupported annotation expression '{ast.unparse(annotation)}' in nested kernel."
        )

    def _parse_nested_annotation(self, annotation: ast.AST, arg_name: str) -> BaseType:
        if isinstance(annotation, ast.Name) and annotation.id in {
            "constexpr",
            "Constexpr",
        }:
            return Constexpr
        if isinstance(annotation, (ast.Name, ast.Attribute)):
            resolved = self._resolve_annotation_symbol(annotation)
            try:
                return self.kernel.parse_type_annotation(resolved)
            except Exception:
                pass
        annotation_text = ast.unparse(annotation)
        if annotation_text in {"constexpr", "Constexpr"}:
            return Constexpr
        try:
            return self.kernel.parse_type_annotation(annotation_text)
        except Exception as e:
            self.compile_error(
                f"Unsupported type annotation '{annotation_text}' for nested kernel parameter '{arg_name}': {e}"
            )

    def _parse_nested_return_types(self, nested: NestedKernelSymbol) -> list[BaseType]:
        returns = nested.node.returns
        if returns is None or (
            isinstance(returns, ast.Constant) and returns.value is None
        ):
            return []
        if isinstance(returns, ast.Tuple):
            return [
                self._parse_nested_annotation(ret, f"return[{i}]")
                for i, ret in enumerate(returns.elts)
            ]
        return [self._parse_nested_annotation(returns, "return")]

    def _bind_nested_arguments(self, nested: NestedKernelSymbol, args, kws):
        fn_args = nested.node.args
        if (
            len(fn_args.posonlyargs) > 0
            or len(fn_args.kwonlyargs) > 0
            or fn_args.vararg is not None
            or fn_args.kwarg is not None
        ):
            self.compile_error(
                f"Nested kernel '{nested.name}' only supports regular positional/keyword arguments (no posonly/kwonly/*args/**kwargs)."
            )
        params = fn_args.args
        param_names = [param.arg for param in params]
        if len(args) > len(params):
            self.compile_error(
                f"Invalid arguments for nested kernel '{nested.name}': expected at most {len(params)} positional arguments, got {len(args)}."
            )

        bound = {name: value for name, value in zip(param_names, args)}
        for kw_name, kw_val in kws.items():
            if kw_name not in param_names:
                self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': unexpected keyword argument '{kw_name}'."
                )
            if kw_name in bound:
                self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': multiple values for argument '{kw_name}'."
                )
            bound[kw_name] = kw_val

        defaults = fn_args.defaults
        first_default_idx = len(params) - len(defaults)
        for idx, param in enumerate(params):
            if param.arg in bound:
                continue
            if idx < first_default_idx:
                self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': missing required argument '{param.arg}'."
                )
            default_expr = defaults[idx - first_default_idx]
            try:
                self.visiting_default_args = True
                bound[param.arg] = self.visit(default_expr)
            finally:
                self.visiting_default_args = False
        # Normalize to declared parameter order to keep type checking and
        # operand lowering deterministic for keyword-heavy call sites.
        return {param.arg: bound[param.arg] for param in params}

    def _infer_nested_arg_type(self, arg_name: str, arg_val) -> BaseType:
        if isinstance(arg_val, Proxy):
            return arg_val.type
        if isinstance(arg_val, Constexpr):
            try:
                return _infer_value_type(
                    arg_val.value, self.kernel.options.enable_tensor
                )
            except TypeError as e:
                self.compile_error(
                    f"Failed to infer type for nested kernel argument '{arg_name}' with value '{arg_val.value}': {e}"
                )
        self.compile_error(
            f"Cannot infer type for nested kernel argument '{arg_name}' from value of type '{type(arg_val).__name__}'."
        )

    def _specialize_nested_kernel(self, nested: NestedKernelSymbol, bound):
        arg_types = []
        for param in nested.node.args.args:
            arg_val = bound[param.arg]
            if param.annotation is None:
                arg_types.append(self._infer_nested_arg_type(param.arg, arg_val))
            else:
                arg_types.append(
                    self._parse_nested_annotation(param.annotation, param.arg)
                )
        ret_types = self._parse_nested_return_types(nested)
        return arg_types, ret_types

    def _bind_nested_mapping(self, nested: NestedKernelSymbol):
        mapping_expr = nested.mapping_expr
        if mapping_expr is None:
            return None
        mapping_value = self.visit(mapping_expr)
        from_constexpr_sequence = False
        if isinstance(mapping_value, Constexpr):
            mapping_value = mapping_value.value
            from_constexpr_sequence = True
        elif isinstance(mapping_value, Proxy):
            self.compile_error(
                f"Invalid mapping for nested kernel '{nested.name}': expected a constexpr integer sequence, but got runtime value of type '{mapping_value.type}'."
            )

        if not isinstance(mapping_value, Sequence) or isinstance(
            mapping_value, (str, bytes)
        ):
            self.compile_error(
                f"Invalid mapping for nested kernel '{nested.name}': expected a sequence of constexpr integers, but got '{type(mapping_value).__name__}'."
            )

        mapping: list[int] = []
        for idx, item in enumerate(mapping_value):
            if isinstance(item, Proxy):
                self.compile_error(
                    f"Invalid mapping for nested kernel '{nested.name}' at index {idx}: expected constexpr integer, but got runtime value of type '{item.type}'."
                )
            if isinstance(item, Constexpr):
                item_constexpr = item
            elif from_constexpr_sequence:
                item_constexpr = Constexpr(item)
            else:
                self.compile_error(
                    f"Invalid mapping for nested kernel '{nested.name}' at index {idx}: expected constexpr integer, but got '{type(item).__name__}'."
                )
            if type(item_constexpr.value) is not int:
                self.compile_error(
                    f"Invalid mapping for nested kernel '{nested.name}' at index {idx}: expected constexpr integer, but got value '{item_constexpr.value}' of type '{type(item_constexpr.value).__name__}'."
                )
            mapping.append(item_constexpr.value)
        return tuple(mapping)

    def _build_nested_closure_scopes(self):
        closure_scope: dict[str, object] = {}
        closure_fscope: dict[str, NestedKernelSymbol | Kernel] = {}
        forbidden_scope: dict[str, object] = {}

        for name, value in self.lscope.items():
            if isinstance(value, (BaseType, Constexpr, Kernel)):
                closure_scope[name] = value
            else:
                forbidden_scope[name] = value

        for name, value in self.fscope.items():
            if isinstance(value, (NestedKernelSymbol, Kernel)):
                closure_fscope[name] = value
            else:
                forbidden_scope[name] = value

        return closure_scope, closure_fscope, forbidden_scope

    def call_nested_kernel(self, nested: NestedKernelSymbol, args, kws):
        nested_key = f"{nested.owner_func_name}.{nested.name}"
        if nested_key in self._active_nested_calls:
            chain = " -> ".join(self._active_nested_calls + [nested_key])
            self.compile_error(
                f"Recursive nested kernel calls are not supported: {chain}"
            )

        bound = self._bind_nested_arguments(nested, args, kws)
        _ = self._bind_nested_mapping(nested)
        sub_arg_types, sub_res_types = self._specialize_nested_kernel(nested, bound)

        if len(sub_arg_types) != len(bound):
            self.compile_error(
                f"Nested kernel specialization argument count mismatch for '{nested.name}': expected {len(bound)}, got {len(sub_arg_types)}."
            )

        callee_context: dict[str, Proxy | Constexpr] = {}
        call_operands: list[Value] = []
        for (arg_name, arg_val), expected_ty in zip(bound.items(), sub_arg_types):
            if expected_ty == Constexpr:
                if not isinstance(arg_val, Constexpr):
                    if isinstance(arg_val, Proxy):
                        self.compile_error(
                            f"Kernel call argument '{arg_name}' must be constexpr, but got runtime value of type '{arg_val.type}'."
                        )
                    arg_val = Constexpr(arg_val)
                callee_context[arg_name] = arg_val
                continue
            call_operands.append(
                self._build_kernel_call_operand(arg_val, expected_ty, arg_name)
            )

        closure_scope, closure_fscope, forbidden_scope = (
            self._build_nested_closure_scopes()
        )

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        sub_generator = None
        self._active_nested_calls.append(nested_key)
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
            sub_generator = CodeGenerator(
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
                options=self.kernel.options,
                callee_context=callee_context,
                fscope=closure_fscope,
                closure_scope=closure_scope,
                forbidden_closure_scope=forbidden_scope,
                active_nested_calls=self._active_nested_calls,
            )
            sub_generator.visit(nested.node)
            if sub_generator.curr_fn is None:
                self.compile_error(
                    f"Internal error: failed to materialize nested kernel '{nested.name}'."
                )
        except CompilationError as e:
            raise CompilationError(
                e.node,
                f"error when compiling kernel '{nested.name}' called from '{self.func_name}': {e.message}",
                e.src if e.src is not None else self.kernel.src,
            ) from e
        finally:
            self._active_nested_calls.pop()
            self.builder.src = self.kernel.src
            self.builder.set_insertion_point_and_loc(ip, last_loc)

        assert sub_generator is not None and sub_generator.curr_fn is not None
        call_op = func.CallOp(self.builder, sub_generator.curr_fn, call_operands)
        return self._decode_kernel_call_results(call_op, sub_res_types)

    def call_kernel(self, fn: Kernel, args, kws):
        """Lower/call a kernel specialization and decode structured return values."""

        if fn.is_top:
            self.compile_error(
                f"Cannot call accelerator '{fn.func_name}' inside kernel '{self.func_name}'. Accelerators must be top-level entry kernels."
            )

        try:
            bound = fn.signature.bind(*args, **kws)
            bound.apply_defaults()
        except TypeError as e:
            self.compile_error(f"Invalid arguments for kernel '{fn.func_name}': {e}.")

        try:
            sub_arg_types = list(fn.specialize_arg_types(*args, **kws))
            sub_res_types = list(
                fn.parse_return_annotation(fn.signature.return_annotation)
            )
        except Exception as e:
            self.compile_error(f"Failed to specialize kernel '{fn.func_name}': {e}")

        if len(sub_arg_types) != len(bound.arguments):
            self.compile_error(
                f"Kernel specialization argument count mismatch for '{fn.func_name}': expected {len(bound.arguments)}, got {len(sub_arg_types)}."
            )

        callee_context: dict[str, Proxy | Constexpr] = {}
        call_operands: list[Value] = []
        for (arg_name, arg_val), expected_ty in zip(
            bound.arguments.items(), sub_arg_types
        ):
            if expected_ty == Constexpr:
                if not isinstance(arg_val, Constexpr):
                    if isinstance(arg_val, Proxy):
                        self.compile_error(
                            f"Kernel call argument '{arg_name}' must be constexpr, but got runtime value of type '{arg_val.type}'."
                        )
                    arg_val = Constexpr(arg_val)
                callee_context[arg_name] = arg_val
                continue
            call_operands.append(
                self._build_kernel_call_operand(arg_val, expected_ty, arg_name)
            )

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        sub_generator = None
        try:
            self.builder.set_insertion_point_to_end(self.module.get_body())
            self.builder.set_loc(Location(fn.file_name, fn.begin_line, 1, self.context))
            self.builder.src = fn.src
            sub_generator = CodeGenerator(
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
            )
            sub_generator.visit(fn.parse())
            if sub_generator.curr_fn is None:
                self.compile_error(
                    f"Internal error: failed to materialize callee function for kernel '{fn.func_name}'."
                )
        except CompilationError as e:
            raise CompilationError(
                e.node,
                f"error when compiling kernel '{fn.func_name}' called from '{self.func_name}': {e.message}",
                e.src if e.src is not None else fn.src,
            ) from e
        finally:
            self.builder.src = self.kernel.src
            self.builder.set_insertion_point_and_loc(ip, last_loc)
        assert sub_generator is not None and sub_generator.curr_fn is not None

        call_op = func.CallOp(self.builder, sub_generator.curr_fn, call_operands)
        return self._decode_kernel_call_results(call_op, sub_res_types)

    def call_operator(self, fn: Operator | BoundOperator, args, kws):  # noqa: ARG002
        if isinstance(fn, BoundOperator):
            args = fn.bind_args(args)
            fn = fn.op
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        try:
            err_msg = fn.run_validate(*args, **kws)
            if err_msg:
                self.compile_error(
                    "Invalid arguments for operator '{}': {}".format(
                        fn.__name__, err_msg
                    )
                )
        except TypeError:
            self.compile_error(f"Invalid argument number for operator '{fn.__name__}'.")
        folded = fn.run_const_fold(*args, **kws)
        if folded is not NO_FOLD:
            return folded
        ret = fn.run_lower(self.builder, *args, **kws)
        # restore states
        self.builder.set_insertion_point_and_loc(ip, last_loc)
        return ret

    ##################
    # Builtins
    ##################

    @staticmethod
    def static_executor(python_fn):
        def ret(self, node: ast.Call):
            kws = {
                name: unwrap_if_constexpr(value)
                for name, value in (self.visit(keyword) for keyword in node.keywords)
            }
            args = [unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
            return Constexpr(python_fn(*args, **kws))

        return ret

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError(
                "`static_assert` requires one or two positional arguments only"
            )

        passed = unwrap_if_constexpr(self.visit(node.args[0]))
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
                node, cast(str, unwrap_if_constexpr(message)), self.kernel.src
            )
        return None

    statically_implemented_functions = {
        dsl.static_assert: execute_static_assert,
        print: static_executor(print),
        len: static_executor(len),
    }


class EnterSubRegion:
    """Scoped helper that snapshots/restores frontend symbol state + insertion point."""

    def __init__(self, generator: CodeGenerator):
        self.generator = generator

    def __enter__(self):
        self.lscope = self.generator.lscope.copy()
        self.local_defs = self.generator.local_defs.copy()
        self.generator.local_defs = {}
        self.ip = self.generator.builder.save_insertion_point()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.lscope = self.lscope
        self.generator.local_defs = self.local_defs
        self.generator.builder.restore_insertion_point(self.ip)


def compile(
    fn: Kernel,
    arg_types: Sequence[BaseType | str] = [],
    res_types: Sequence[BaseType | str] = [],
    show_traceback: bool = False,
    options: CompileOptions = CompileOptions(),
):
    """Compile a kernel function into an MLIR module."""
    import os

    if os.environ.get("ALLO_SHOW_COMPILER_TRACEBACK", "") == "1":
        show_traceback = True
    if not isinstance(fn, Kernel):
        raise TypeError(
            "Only allo.kernel functions can be compiled with allo.compile()"
        )
    arg_types = [fn.parse_type_annotation(t) for t in arg_types]
    if len(arg_types) != len(fn.signature.parameters):
        raise ValueError(
            f"The number of provided argument types ({len(arg_types)}) does not match the number of arguments in the kernel signature ({len(fn.signature.parameters)})."
        )
    res_types = [fn.parse_type_annotation(t) for t in res_types]
    effective_options = options if options != CompileOptions() else fn.options

    try:
        context = Context()
        context.load_dialects()

        # initialize builder
        builder = AlloOpBuilder(context, typing_style=effective_options.typing_style)
        builder.src = fn.src
        builder.set_loc(Location(fn.file_name, fn.begin_line, 1, context))
        module = ModuleOp(builder)
        builder.module = module
        builder.set_insertion_point_to_end(module.get_body())

        # start codegen
        generator = CodeGenerator(
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
        )
        generator.visit(fn.parse())

        # verify
        if not module.verify():
            print(module)
            raise RuntimeError(
                f"In function: {fn.func_name}, module verification failed."
            )

        module.cse_and_canonicalize()
        return module, generator.context
    except Exception as exc:
        if show_traceback:
            raise
        if isinstance(exc, (CompilationError, CompileTimeAssertionFailure)):
            raise exc.with_traceback(None) from None
        raise RuntimeError(
            "Internal compiler error during Allo kernel compilation.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "Re-run with show_traceback=True to see the full traceback."
        ) from None
