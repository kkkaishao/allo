# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import builtins

from types import ModuleType
from .._C.ir import (
    Context,
    ModuleOp,
    Location,
    FunctionType,
    Value,
    BlockArgument,
)
from .._C import scf, func, ub as ub_d, arith
from ..core.types import (
    BaseType,
    Constexpr,
    Proxy,
    ShapedType,
    BufferType,
    TensorType,
    DType,
    unwrap_if_constexpr,
    int1,
    index,
)
from ..core import types
from .builder import AlloOpBuilder
from ..core.kernel import Kernel, ConstevalFunction
from collections.abc import Sequence
from .. import dsl
from typing import Type, cast
from ..core.library import Operator, BoundOperator, NO_FOLD
from .errors import CompilationError, CompileTimeAssertionFailure


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
        module_map: dict = {},
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
        self.res_types = res_types
        self.allow_implicit_type_infer = False

        # trackers
        self.local_defs = {}  # track local variable definitions
        self.lscope: dict[str, Proxy | Constexpr] = (
            {}
        )  # track what can be seen in the current scope
        self.gscope = {}
        for k, v in gscope.items():
            if isinstance(v, ModuleType):
                # module-level remap
                self.gscope[k] = module_map.get(v.__name__, v)
                continue
            module_name = getattr(v, "__module__", None)
            if module_name is not None and module_name in module_map:
                self.gscope[k] = getattr(module_map[module_name], v.__name__)
            else:
                self.gscope[k] = v
        self.module_map = module_map
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
            return self.lscope.get(name, absent)

        def global_lookup(name: str, absent):
            val = self.gscope.get(name, absent)
            if self._is_allowed_global_name(name, val, absent):
                return val
            self.compile_error(
                f"Cannot access global name '{name}' in current scope. Allo kernels can only access constexpr values, allo types from global scope, and imported modules."
            )

        absent_marker = object()

        def name_lookup(name: str):
            for lookup in (local_lookup, self.builtin_namespace.get, global_lookup):
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
            or getattr(val, "__module__", "").startswith("allo.experimental.core")
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

    def visit_compound_stmts(self, stmts):
        if not isinstance(stmts, builtins.list):
            stmts = [stmts]
        for stmt in stmts:
            self.visit(stmt)

    def visit_Module(self, node: ast.Module):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        arg_names, _ = self.visit(node.args)
        # init defaults
        for i, default in enumerate(node.args.defaults[::-1]):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
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
        entry_block = func_op.add_entry_block()

        arg_handles = [func_op.get_arg_at(i) for i in range(func_op.get_num_args())]
        arg_proxies = [
            Proxy(handle, ty) for handle, ty in zip(arg_handles, self.arg_types)
        ]
        for name, proxy in zip(arg_names, arg_proxies):
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)

        # visit function body
        self.builder.set_insertion_point_to_start(entry_block)
        self.visit_compound_stmts(node.body)

        # create a return op
        self.builder.set_insertion_point_to_end(entry_block)
        func.ReturnOp(self.builder, [])
        # restore
        self.builder.set_insertion_point_after(func_op.get_operation())

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

    def visit_BinOp(self, node):
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

    def visit_If(self, node: ast.If):
        cond = self.visit(node.test)
        if isinstance(cond, Proxy):
            if isinstance(cond.type, ShapedType):
                self.compile_error(
                    "Condition of 'if' statement cannot be a shaped type."
                )
            cond = self.builder.scalar_cast(cond, int1)
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

    def visit_if_impl(self, cond: Proxy, node: ast.If):
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            parent_region = ip.get_block().get_parent_region()
            then_block = self.builder.create_free_block(parent_region)
            else_block = self.builder.create_free_block(parent_region)
            # get a copy of current live-ins
            liveins = self.lscope.copy()
            self.scf_stack.append(node)

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
                if not isinstance(then_proxy, Proxy) or not isinstance(
                    else_proxy, Proxy
                ):
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
        res_handles = [if_op.get_result_at(i) for i in range(len(phi_names))]
        phi_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, phi_types)]
        for name, proxy in zip(phi_names, phi_proxies):
            self._set_value(name, proxy)
            self._maybe_set_loc_to_name(proxy, name)

    def visit_IfExp(self, node: ast.IfExp):
        cond = self.visit(node.test)
        if isinstance(cond, Proxy):
            if isinstance(cond.type, ShapedType):
                self.compile_error(
                    "Condition of ternary expression cannot be a shaped type."
                )
            cond = self.builder.scalar_cast(cond, int1)
            # if exp cannot define new variables
            ip, last_loc = self.builder.get_insertion_point_and_loc()
            parent_region = ip.get_block().get_parent_region()

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
            # if the non-constexpr branch is a shaped type, we need to broadcast the constexpr branch to the same shape
            if isinstance(res_type, ShapedType):
                if then_is_constexpr:
                    const = self.builder.make_scalar(then_val.value, res_type.dtype)
                    buffer = self.builder.make_buffer(res_type)
                    ret = self.builder.fill_buffer(buffer, const)
                    then_val = ret if ret is not None else buffer
                if else_is_constexpr:
                    const = self.builder.make_scalar(else_val.value, res_type.dtype)
                    buffer = self.builder.make_buffer(res_type)
                    ret = self.builder.fill_buffer(buffer, const)
                    else_val = ret if ret is not None else buffer
            else:
                if then_is_constexpr:
                    then_val = self.builder.make_scalar(then_val.value, res_type)
                if else_is_constexpr:
                    else_val = self.builder.make_scalar(else_val.value, res_type)

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

        lb = self.builder.make_or_cast_scalar(lb, index)
        ub = self.builder.make_or_cast_scalar(ub, index)
        step = self.builder.make_or_cast_scalar(step, index)

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
        res_handles = [for_op.get_result_at(i) for i in range(len(name))]
        res_proxies = [Proxy(handle, ty) for handle, ty in zip(res_handles, init_types)]
        for name, proxy in zip(name, res_proxies):
            self._maybe_set_loc_to_name(proxy, name)
            self._set_value(name, proxy)

    def _test_loop_iter_args(self, node, liveins: dict, ignore: set[str]):
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        # create dummy block
        block = self.builder.create_free_block(ip.get_block().get_parent_region())
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

        lb_proxies = [self.builder.make_or_cast_scalar(lb, index) for lb in lbs]
        ub_proxies = [self.builder.make_or_cast_scalar(ub, index) for ub in ubs]
        step_proxies = [self.builder.make_or_cast_scalar(step, index) for step in steps]

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
        res_handles = [par_op.get_result_at(i) for i in range(len(names))]
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
            before_block = self.builder.create_free_block(
                while_op.get_before(), init_ir_types, self.builder.get_loc()
            )
            self.builder.set_insertion_point_to_start(before_block)
            block_args = [before_block.get_arg_at(i) for i in range(len(names))]
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
            after_block = self.builder.create_free_block(
                while_op.get_after(), init_ir_types, self.builder.get_loc()
            )
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = [after_block.get_arg_at(i) for i in range(len(names))]
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
        res_handles = [while_op.get_result_at(i) for i in range(len(names))]
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

        if value is None:
            # we only allow buffer to be declared without initializer
            if not isinstance(parsed_type, ShapedType):
                self.compile_error(
                    f"Type annotation is required for variable declaration without initializer, and it must be a shaped type. Got '{annotation}'."
                )
            proxy = self.builder.make_buffer(parsed_type)
            self._set_value_with_loc(target, proxy)
        else:
            if isinstance(value, Constexpr):
                # Case 1: target is also constexpr
                if parsed_type == Constexpr:
                    self._set_value(target, value)
                # Case 2: target is a scalar
                elif isinstance(parsed_type, DType):
                    proxy = self.builder.make_scalar(value.value, parsed_type)
                    self._set_value_with_loc(target, proxy)
                # Case 3: target is a shaped type
                elif isinstance(parsed_type, ShapedType):
                    buffer = self.builder.make_buffer(parsed_type)
                    const = self.builder.make_scalar(value.value, parsed_type.dtype)
                    proxy = self.builder.fill_buffer(buffer, const)
                    if proxy is None:
                        self._set_value_with_loc(target, buffer)
                    else:
                        self._set_value_with_loc(target, proxy)
            elif isinstance(value, Proxy):
                # Case 4: target is dtype
                if isinstance(parsed_type, DType) and isinstance(value.type, DType):
                    proxy = self.builder.cast(value, parsed_type)
                    self._set_value_with_loc(target, proxy)
                # Case 5: target is tensor type
                elif isinstance(parsed_type, TensorType):
                    if isinstance(value.type, DType):
                        buffer = self.builder.make_buffer(parsed_type)
                        value = self.builder.cast(value, parsed_type.dtype)
                        proxy = self.builder.fill_buffer(buffer, value)
                        assert proxy is not None
                        self._set_value_with_loc(target, proxy)
                    elif isinstance(value.type, TensorType):
                        if value.type != parsed_type:
                            self.compile_error(
                                f"Cannot assign a tensor of type '{value.type}' to a variable of type '{parsed_type}'."
                            )
                        proxy = self.builder.tensor_cast(value, parsed_type.dtype)
                        self._set_value_with_loc(target, proxy)
                # Case 6: target is buffer type
                elif isinstance(parsed_type, BufferType):
                    if isinstance(value.type, DType):
                        buffer = self.builder.make_buffer(parsed_type)
                        value = self.builder.cast(value, parsed_type.dtype)
                        self.builder.fill_buffer(buffer, value)
                        self._set_value_with_loc(target, buffer)
                    elif isinstance(value.type, BufferType):
                        self.compile_error(
                            "Direct assignment between buffer types is not supported. If you want to copy data from one buffer to another, please use 'copy' operator to fill the target buffer with the source buffer."
                        )
                else:
                    self.compile_error(
                        f"Unsupported assignment with type annotation '{annotation}' and value of type '{value.type}'."
                    )
            else:
                self.compile_error(
                    f"Unsupported initializer for variable assignment with type annotation '{annotation}'."
                )

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
                if not self.allow_implicit_type_infer:
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

        if isinstance(fn, Kernel):
            return self.call_kernel(fn, args, kws)
        if isinstance(fn, Operator):
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

    def call_kernel(self, fn: Kernel, args, kws):
        """Lower/call a kernel specialization and decode structured return values."""

        sub_arg_types = fn.specialize_arg_types(args, kws)
        sub_res_types = []
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        try:
            self.builder.set_insertion_point_to_end(self.module.get_body())
            self.builder.set_loc(Location(fn.file_name, fn.begin_line, 1, self.context))
            self.builder.src = fn.src
            sub_generator = CodeGenerator(
                self.context,
                self.module,
                self.builder,
                kernel=fn,
                func_name=fn.func_name,
                file_name=fn.file_name,
                begin_line=fn.begin_line,
                gscope=fn.get_capture_scope(),
                module_map=self.module_map,
                arg_types=sub_arg_types,
                res_types=sub_res_types,
            )
            sub_generator.visit(fn.parse())
        except CompilationError as e:
            raise CompilationError(
                e.node,
                f"error when compiling kernel '{fn.func_name}' called from '{self.kernel.func_name}': {e.message}",
                self.kernel.src,
            ) from e
        finally:
            self.builder.src = self.kernel.src
            self.builder.set_insertion_point_and_loc(ip, last_loc)

    def call_operator(self, fn: Operator | BoundOperator, args, kws):  # noqa: ARG002
        if isinstance(fn, BoundOperator):
            args = fn.bind_args(args)
            fn = fn.op
        err_msg = fn.run_validate(*args, **kws)
        if err_msg:
            self.compile_error(
                "Invalid arguments for operator '{}': {}".format(fn.__name__, err_msg)
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
    module_map: dict = {},
):
    """Compile a kernel function into an MLIR module."""
    if not isinstance(fn, Kernel):
        raise TypeError(
            "Only allo.kernel functions can be compiled with allo.compile()"
        )
    arg_types = [fn.parse_type_annotation(t) for t in arg_types]
    res_types = [fn.parse_type_annotation(t) for t in res_types]

    context = Context()
    context.load_dialects()

    # initialize builder
    builder = AlloOpBuilder(context)
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
        module_map=module_map,
        arg_types=arg_types,
        res_types=res_types,
    )
    generator.visit(fn.parse())

    # verify
    if not module.verify():
        print(module)
        raise RuntimeError(f"In function: {fn.func_name}, module verification failed.")

    return module
