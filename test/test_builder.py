# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import pytest

import allo.exp.lang.core as allo_core
from allo.exp.compiler.errors import CompilationError
from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.lang.core import (
    bool as allo_bool,
    constexpr,
    f32,
    grid as allo_grid,
    i32,
    range as allo_range,
    u1,
    u32,
)
from allo.exp.lang.kernel import KernelOptions, consteval, kernel
from allo.exp.operators.arith import max as allo_max

_GLOBAL_SHAPE_M = 2
_GLOBAL_SHAPE_N = 3
_GLOBAL_INT_CONST = 3
_GLOBAL_FLOAT_CONST = 1.5


def _compile_ir(fn, *, options=None) -> str:
    return str(compile_kernel(fn, options=options))


def _assert_contains(ir: str, *patterns: str):
    for pattern in patterns:
        assert pattern in ir


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


def test_compilation_error_plain_diagnostic_points_to_source():
    src = "def broken(x):\n    return x + y\n"
    module = ast.parse(src)
    fn = module.body[0]
    assert isinstance(fn, ast.FunctionDef)
    ret = fn.body[0]
    assert isinstance(ret, ast.Return)
    expr = ret.value
    assert isinstance(expr, ast.BinOp)

    err = CompilationError(
        src,
        "Name 'y' is not defined",
        expr.right,
        file_name="broken.py",
        begin_line=10,
    )
    message = err.render(color=False)

    assert "broken.py:11:16: error: Name 'y' is not defined" in message
    assert "11 |     return x + y" in message
    assert "^" in message
    assert "\x1b[" not in message
    assert str(err).startswith("\n")


def test_scalar_int_add():
    @kernel
    def scalar_int_add(x: i32, y: i32, out: "i32[1]"):
        out[0] = x + y

    ir = _compile_ir(scalar_int_add)
    _assert_contains(
        ir,
        "func.func @scalar_int_add",
        "arith.extsi",
        "to i33",
        "arith.addi",
        "i33 to i32",
        "memref.store",
    )


def test_hls_nary_add_sub():
    @kernel
    def hls_nary_add_sub(x: i32, y: i32, z: i32, out: "i32[1]"):
        out[0] = x + y - z

    ir = _compile_ir(hls_nary_add_sub)
    _assert_contains(
        ir,
        "func.func @hls_nary_add_sub",
        "arith.constant 0 : i34",
        "to i34",
        "arith.subi",
        "arith.addi",
        "i34 to i32",
        "memref.store",
    )


def test_hls_nary_mul():
    @kernel
    def hls_nary_mul(x: i32, y: i32, z: i32, out: "i32[1]"):
        out[0] = x * y * z

    ir = _compile_ir(hls_nary_mul)
    _assert_contains(
        ir,
        "func.func @hls_nary_mul",
        "to i96",
        "arith.muli",
        "i96 to i32",
        "memref.store",
    )


def test_mixed_int_float_add():
    @kernel
    def mixed_int_float_add(x: i32, y: f32, out: "f32[1]"):
        out[0] = x + y

    ir = _compile_ir(mixed_int_float_add)
    _assert_contains(
        ir,
        "func.func @mixed_int_float_add",
        "arith.sitofp",
        "i32 to f32",
        "arith.addf",
        "memref.store",
    )


def test_float_add():
    @kernel
    def float_add(x: f32, y: f32, out: "f32[1]"):
        out[0] = x + y

    ir = _compile_ir(float_add)
    _assert_contains(ir, "func.func @float_add", "arith.addf", "memref.store")


def test_unary_neg():
    @kernel
    def unary_neg(x: i32, out: "i32[1]"):
        out[0] = -x

    ir = _compile_ir(unary_neg)
    _assert_contains(
        ir,
        "func.func @unary_neg",
        "arith.constant 0 : i33",
        "arith.extsi",
        "arith.subi",
        "i33 to i32",
        "memref.store",
    )


def test_bitwise_xor():
    @kernel
    def bitwise_xor(x: u32, y: u32, out: "u32[1]"):
        out[0] = x ^ y

    ir = _compile_ir(bitwise_xor)
    _assert_contains(ir, "func.func @bitwise_xor", "arith.xori", "memref.store")


def test_comparison_lt():
    @kernel
    def comparison_lt(x: i32, y: i32, out: "u1[1]"):
        out[0] = x < y

    ir = _compile_ir(comparison_lt)
    _assert_contains(
        ir,
        "func.func @comparison_lt",
        "arith.cmpi slt",
        "memref<1xi1>",
        "memref.store",
    )


def test_bool_and_not():
    @kernel
    def bool_and_not(x: allo_bool, y: allo_bool, out: "u1[1]"):
        out[0] = x and not y

    ir = _compile_ir(bool_and_not)
    _assert_contains(
        ir,
        "func.func @bool_and_not",
        "arith.constant true",
        "arith.xori",
        "arith.andi",
        "memref<1xi1>",
        "memref.store",
    )


def test_if_statement_phi():
    @kernel
    def if_statement_phi(cond: allo_bool, x: i32, y: i32, out: "i32[1]"):
        v = x
        if cond:
            v = y
        else:
            v = x + y
        out[0] = v

    ir = _compile_ir(if_statement_phi)
    _assert_contains(
        ir,
        "func.func @if_statement_phi",
        "scf.if",
        "-> (i32)",
        "scf.yield",
        "memref.store",
    )


def test_ternary_expression():
    @kernel
    def ternary_expression(cond: allo_bool, x: i32, y: i32, out: "i32[1]"):
        out[0] = x if cond else y

    ir = _compile_ir(ternary_expression)
    _assert_contains(
        ir, "func.func @ternary_expression", "arith.select", "memref.store"
    )


def test_memref_load_store():
    @kernel
    def memref_load_store(inp: "i32[4]", out: "i32[1]"):
        out[0] = inp[0]

    ir = _compile_ir(memref_load_store)
    _assert_contains(
        ir,
        "func.func @memref_load_store",
        "memref.load",
        "memref<4xi32>",
        "memref.store",
        "memref<1xi32>",
    )


def test_range_loop_store():
    @kernel
    def range_loop_store(out: "i32[4]"):
        for i in allo_range(4):
            out[i] = i

    ir = _compile_ir(range_loop_store)
    _assert_contains(
        ir,
        "func.func @range_loop_store",
        "scf.for",
        "to %c4 step %c1",
        "arith.index_cast",
        "index to i32",
        "memref.store",
    )


def test_python_builtin_range_loop_store():
    @kernel
    def python_builtin_range_loop_store(out: "i32[4]"):
        for i in range(4):
            out[i] = i

    ir = _compile_ir(python_builtin_range_loop_store)
    _assert_contains(
        ir,
        "func.func @python_builtin_range_loop_store",
        "scf.for",
        "to %c4 step %c1",
        "memref.store",
    )


def test_grid_loop_store():
    @kernel
    def grid_loop_store(out: "i32[2, 2]"):
        for i, j in allo_grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(grid_loop_store)
    _assert_contains(
        ir,
        "func.func @grid_loop_store",
        "scf.parallel",
        "step (%c1, %c1)",
        "arith.addi",
        "arith.index_cast",
        "memref<2x2xi32>",
        "memref.store",
    )


def test_direct_operator_call():
    @kernel
    def direct_operator_call(x: i32, y: i32, out: "i32[1]"):
        out[0] = allo_max(x, y)

    ir = _compile_ir(direct_operator_call)
    _assert_contains(
        ir, "func.func @direct_operator_call", "arith.maxsi", "memref.store"
    )


def test_python_builtin_max_min_calls():
    @kernel
    def python_builtin_max_min_calls(x: i32, y: i32, out: "i32[2]"):
        out[0] = max(x, y)
        out[1] = min(x, y)

    ir = _compile_ir(python_builtin_max_min_calls)
    _assert_contains(
        ir,
        "func.func @python_builtin_max_min_calls",
        "arith.maxsi",
        "arith.minsi",
        "memref.store",
    )


def test_global_scalar_constants_are_constexpr():
    @kernel
    def global_scalar_constants_are_constexpr(x: i32, y: f32, out: "f32[2]"):
        out[0] = x + _GLOBAL_INT_CONST
        out[1] = y + _GLOBAL_FLOAT_CONST

    ir = _compile_ir(global_scalar_constants_are_constexpr)
    _assert_contains(
        ir,
        "func.func @global_scalar_constants_are_constexpr",
        "arith.constant 3",
        "arith.constant 1.500000e+00",
        "arith.addi",
        "arith.addf",
        "memref.store",
    )


def test_global_constexpr_shape_expression_annotation():
    @kernel
    def global_constexpr_shape_expression_annotation(
        inp: "i32[_GLOBAL_SHAPE_M * _GLOBAL_SHAPE_N]",
        out: "i32[_GLOBAL_SHAPE_M, _GLOBAL_SHAPE_N]",
    ):
        for i in range(_GLOBAL_SHAPE_M):
            for j in range(_GLOBAL_SHAPE_N):
                out[i, j] = inp[i * _GLOBAL_SHAPE_N + j]

    ir = _compile_ir(global_constexpr_shape_expression_annotation)
    _assert_contains(
        ir,
        "func.func @global_constexpr_shape_expression_annotation",
        "memref<6xi32>",
        "memref<2x3xi32>",
        "scf.for",
        "memref.store",
    )


def test_definition_scope_shape_annotation():
    rows = 2
    cols = 2

    @kernel
    def definition_scope_shape_annotation(out: "i32[rows, cols]"):
        for i, j in allo_grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(definition_scope_shape_annotation)
    _assert_contains(
        ir,
        "func.func @definition_scope_shape_annotation",
        "memref<2x2xi32>",
        "scf.parallel",
        "memref.store",
    )


def test_local_memref_declaration_without_initializer():
    @kernel
    def local_memref_declaration_without_initializer(out: "i32[4]"):
        N: constexpr = 4
        buf: "i32[N]"
        for i in range(N):
            buf[i] = i
            out[i] = buf[i]

    ir = _compile_ir(local_memref_declaration_without_initializer)
    _assert_contains(
        ir,
        "func.func @local_memref_declaration_without_initializer",
        "memref.alloc",
        "memref<4xi32>",
        "memref.load",
        "memref.store",
    )


def test_local_tensor_declaration_without_initializer():
    @kernel(options=KernelOptions(enable_tensor=True))
    def local_tensor_declaration_without_initializer() -> "f32[4]":
        N: constexpr = 4
        buf: "f32[N]"
        return buf

    ir = _compile_ir(local_tensor_declaration_without_initializer)
    _assert_contains(
        ir,
        "func.func @local_tensor_declaration_without_initializer",
        "tensor.empty",
        "tensor<4xf32>",
        "return",
    )


def test_memref_list_initializer_uses_global():
    @kernel
    def memref_list_initializer_uses_global(out: "i32[2,2]"):
        scale: constexpr = _GLOBAL_INT_CONST
        buf: "i32[2,2]" = [[1, scale], [scale + 1, scale + 2]]
        for i, j in allo_grid(2, 2):
            out[i, j] = buf[i, j]

    ir = _compile_ir(memref_list_initializer_uses_global)
    _assert_contains(
        ir,
        'memref.global "private" @buf_initializer_0',
        "memref.get_global @buf_initializer_0",
        "dense<[[1, 3], [4, 5]]>",
        "memref.load",
    )
    assert "memref.copy" not in ir


def test_tensor_list_initializer_uses_arith_constant():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_list_initializer_uses_arith_constant() -> "i32[2,2]":
        buf: "i32[2,2]" = [[1, 2], [3, 4]]
        return buf

    ir = _compile_ir(tensor_list_initializer_uses_arith_constant)
    _assert_contains(
        ir,
        "arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>",
        "return",
        "tensor<2x2xi32>",
    )


def test_while_loop_carried_values():
    @kernel
    def while_loop_carried_values(out: "i32[1]"):
        i: i32 = 0
        acc: i32 = 0
        while i < 4:
            acc += i
            i += 1
        out[0] = acc

    ir = _compile_ir(while_loop_carried_values)
    _assert_contains(
        ir,
        "func.func @while_loop_carried_values",
        "scf.while",
        "scf.condition",
        "scf.yield",
        "memref.store",
    )


def test_consteval_value_in_expression():
    @consteval
    def factor():
        return 3

    @kernel
    def consteval_value_in_expression(x: i32, out: "i32[1]"):
        out[0] = x + factor()

    ir = _compile_ir(consteval_value_in_expression)
    _assert_contains(
        ir,
        "func.func @consteval_value_in_expression",
        "arith.constant 3 : i33",
        "arith.extsi",
        "arith.addi",
        "i33 to i32",
        "memref.store",
    )


def test_nested_kernel_call_store():
    @kernel
    def nested_kernel_call_store(x: i32, out: "i32[1]"):
        @kernel
        def add_one(v: i32) -> i32:
            return v + 1

        out[0] = add_one(x)

    ir = _compile_ir(nested_kernel_call_store)
    _assert_contains(
        ir,
        "func.func @nested_kernel_call_store.add_one",
        "call @nested_kernel_call_store.add_one",
        "memref.store",
    )


def test_nested_kernel_multiple_returns():
    @kernel
    def nested_kernel_multiple_returns(x: i32, y: i32, out: "i32[1]"):
        @kernel
        def pair(a: i32, b: i32) -> (i32, i32):
            return a, b

        lhs, rhs = pair(x, y)
        out[0] = lhs + rhs

    ir = _compile_ir(nested_kernel_multiple_returns)
    _assert_contains(
        ir,
        "func.func @nested_kernel_multiple_returns.pair",
        "-> (i32, i32)",
        "call @nested_kernel_multiple_returns.pair",
        "arith.addi",
        "memref.store",
    )


def test_nested_kernel_captures_constexpr_value():
    @kernel
    def nested_kernel_captures_constexpr_value(x: i32, out: "i32[1]"):
        offset: constexpr = 3

        @kernel
        def add_offset(v: i32) -> i32:
            return v + offset

        out[0] = add_offset(x)

    ir = _compile_ir(nested_kernel_captures_constexpr_value)
    _assert_contains(
        ir,
        "func.func @nested_kernel_captures_constexpr_value.add_offset",
        "arith.constant 3",
        "memref.store",
    )


def test_nested_kernel_captures_type_alias():
    @kernel
    def nested_kernel_captures_type_alias(out: "i32[1]"):
        T: constexpr = i32

        @kernel
        def emit() -> T:
            return 7

        out[0] = emit()

    ir = _compile_ir(nested_kernel_captures_type_alias)
    _assert_contains(
        ir,
        "func.func @nested_kernel_captures_type_alias.emit",
        "-> i32",
        "arith.constant 7 : i32",
        "memref.store",
    )


def test_nested_kernel_captures_consteval_function():
    @consteval
    def amount():
        return 5

    @kernel
    def nested_kernel_captures_consteval_function(x: i32, out: "i32[1]"):
        @kernel
        def add_amount(v: i32) -> i32:
            return v + amount()

        out[0] = add_amount(x)

    ir = _compile_ir(nested_kernel_captures_consteval_function)
    _assert_contains(
        ir,
        "func.func @nested_kernel_captures_consteval_function.add_amount",
        "arith.constant 5",
        "memref.store",
    )


def test_nested_kernel_captures_kernel_alias():
    @kernel
    def plus_two(v: i32) -> i32:
        return v + 2

    @kernel
    def nested_kernel_captures_kernel_alias(x: i32, out: "i32[1]"):
        callee: constexpr = plus_two

        @kernel
        def apply(v: i32) -> i32:
            return callee(v)

        out[0] = apply(x)

    ir = _compile_ir(nested_kernel_captures_kernel_alias)
    _assert_contains(
        ir,
        "func.func @nested_kernel_captures_kernel_alias.apply.plus_two",
        "call @nested_kernel_captures_kernel_alias.apply.plus_two",
        "call @nested_kernel_captures_kernel_alias.apply",
        "memref.store",
    )


def test_nested_kernel_captures_module_alias():
    @kernel
    def nested_kernel_captures_module_alias(out: "i32[2]"):
        M: constexpr = allo_core

        @kernel
        def fill(buf: "i32[2]"):
            for i in M.range(2):
                buf[i] = i

        fill(out)

    ir = _compile_ir(nested_kernel_captures_module_alias)
    _assert_contains(
        ir,
        "func.func @nested_kernel_captures_module_alias.fill",
        "scf.for",
        "memref.store",
    )


def test_cpp_typing_style_compile_path():
    @kernel(options=KernelOptions(typing_style="cpp"))
    def cpp_typing_style_compile_path(x: u32, y: i32, out: "u32[1]"):
        out[0] = x + y

    ir = _compile_ir(cpp_typing_style_compile_path)
    _assert_contains(
        ir,
        "func.func @cpp_typing_style_compile_path",
        "arith.addi",
        ": i32",
        "memref.store",
    )


def test_return_scalar_value():
    @kernel
    def return_scalar_value(x: i32, y: i32) -> i32:
        return x + y

    ir = _compile_ir(return_scalar_value)
    _assert_contains(
        ir,
        "func.func @return_scalar_value",
        "-> i32",
        "arith.addi",
        "i33 to i32",
        "return",
    )


def test_return_constexpr_literal():
    @kernel
    def return_constexpr_literal() -> i32:
        return 3

    ir = _compile_ir(return_constexpr_literal)
    _assert_contains(
        ir,
        "func.func @return_constexpr_literal",
        "-> i32",
        "arith.constant 3 : i32",
        "return",
    )


def test_return_multiple_values():
    @kernel
    def return_multiple_values(x: i32, y: f32) -> (i32, f32):
        return x, y

    ir = _compile_ir(return_multiple_values)
    _assert_contains(
        ir,
        "func.func @return_multiple_values",
        "-> (i32, f32)",
        "return",
        ": i32, f32",
    )


def test_return_from_if_else():
    @kernel
    def return_from_if_else(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        else:
            return y

    ir = _compile_ir(return_from_if_else)
    _assert_contains(
        ir,
        "func.func @return_from_if_else",
        "cf.cond_br",
        "return",
        ": i32",
    )


def test_return_from_if_with_fallthrough():
    @kernel
    def return_from_if_with_fallthrough(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        return y

    ir = _compile_ir(return_from_if_with_fallthrough)
    _assert_contains(
        ir,
        "func.func @return_from_if_with_fallthrough",
        "cf.cond_br",
        "return",
        ": i32",
    )


def test_return_value_requires_annotation():
    @kernel
    def return_value_requires_annotation(x: i32):
        return x

    _assert_compile_error(
        return_value_requires_annotation,
        "Return values require an explicit return annotation.",
    )


def test_return_missing_for_non_void_function():
    @kernel
    def return_missing_for_non_void_function(x: i32) -> i32:
        y = x + x

    _assert_compile_error(
        return_missing_for_non_void_function,
        "Missing return statement for non-void function",
    )


def test_return_value_count_mismatch():
    @kernel
    def return_value_count_mismatch(x: i32, y: i32) -> (i32, i32):
        return x

    _assert_compile_error(
        return_value_count_mismatch,
        "Return value count mismatch: expected 2, got 1.",
    )


def test_return_type_mismatch():
    @kernel
    def return_type_mismatch(x: "i32[2]") -> "i32[1]":
        return x

    _assert_compile_error(
        return_type_mismatch,
        "Cannot cast from memref<2xint32> to memref<1xint32>",
    )


def test_return_inside_loop_is_rejected():
    @kernel
    def return_inside_loop_is_rejected(x: i32) -> i32:
        for i in allo_range(4):
            return x
        return x

    _assert_compile_error(
        return_inside_loop_is_rejected,
        "'return' is not supported inside loops",
    )


def test_return_inside_nested_if_is_rejected():
    @kernel
    def return_inside_nested_if_is_rejected(
        cond: allo_bool, inner: allo_bool, x: i32
    ) -> i32:
        if cond:
            if inner:
                return x
        return x

    _assert_compile_error(
        return_inside_nested_if_is_rejected,
        "'return' is not supported inside nested 'if' statements.",
    )
