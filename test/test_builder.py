# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import pytest

import allo.exp.lang.core as allo_core
from allo.exp.compiler.errors import CompilationError
from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.lang.core import (
    Template,
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


def _assert_type_error(fn, *patterns: str):
    with pytest.raises(TypeError) as exc_info:
        _compile_ir(fn)
    message = str(exc_info.value)
    for pattern in patterns:
        assert pattern in message


def test_error_diagnostic_source():
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
    def top(x: i32, y: i32, out: "i32[1]"):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.extsi",
        "to i33",
        "arith.addi",
        "i33 to i32",
    )


def test_hls_nary_add_sub():
    @kernel
    def top(x: i32, y: i32, z: i32, out: "i32[1]"):
        out[0] = x + y - z

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 0 : i34",
        "to i34",
        "arith.subi",
        "arith.addi",
        "i34 to i32",
    )


def test_hls_nary_mul():
    @kernel
    def top(x: i32, y: i32, z: i32, out: "i32[1]"):
        out[0] = x * y * z

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "to i96",
        "arith.muli",
        "i96 to i32",
    )


def test_mixed_int_float_add():
    @kernel
    def top(x: i32, y: f32, out: "f32[1]"):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.sitofp",
        "i32 to f32",
        "arith.addf",
    )


def test_float_add():
    @kernel
    def top(x: f32, y: f32, out: "f32[1]"):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.addf")


def test_unary_neg():
    @kernel
    def top(x: i32, out: "i32[1]"):
        out[0] = -x

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 0 : i33",
        "arith.extsi",
        "arith.subi",
        "i33 to i32",
    )


def test_bitwise_xor():
    @kernel
    def top(x: u32, y: u32, out: "u32[1]"):
        out[0] = x ^ y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.xori")


def test_shift_by_range_index():
    @kernel
    def top(x: i32, out: "i32[4]"):
        for i in range(4):
            out[i] = x >> (i * 2)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.index_cast",
        "arith.shrui",
    )


def test_comparison_lt():
    @kernel
    def top(x: i32, y: i32, out: "u1[1]"):
        out[0] = x < y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.cmpi slt",
        "memref<1xi1>",
    )


def test_bool_and_not():
    @kernel
    def top(x: allo_bool, y: allo_bool, out: "u1[1]"):
        out[0] = x and not y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant true",
        "arith.xori",
        "arith.andi",
        "memref<1xi1>",
    )


def test_if_statement_phi():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32, out: "i32[1]"):
        v = x
        if cond:
            v = y
        else:
            v = x + y
        out[0] = v

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.if",
        "-> (i32)",
        "scf.yield",
    )


def test_if_branch_local_buffers():
    @kernel
    def top(out: "i32[8]"):
        for r in range(2):
            r_i32: i32 = r
            if r_i32 == 0:
                then_buf: "i32[4]"
                for j in range(4):
                    then_buf[j] = j
                    out[j] = then_buf[j]
            else:
                else_buf: "i32[4]"
                for j in range(4):
                    else_buf[j] = j + 1
                    out[j + 4] = else_buf[j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.if",
        "scf.for",
        "memref.alloc",
    )


def test_ternary_expression():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32, out: "i32[1]"):
        out[0] = x if cond else y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.select")


def test_memref_load_store():
    @kernel
    def top(inp: "i32[4]", out: "i32[1]"):
        out[0] = inp[0]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref.load",
        "memref<4xi32>",
        "memref.store",
        "memref<1xi32>",
    )


def test_range_loop_store():
    @kernel
    def top(out: "i32[4]"):
        for i in allo_range(4):
            out[i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.for",
        "to %c4 step %c1",
        "arith.index_cast",
        "index to i32",
    )


def test_index_runtime_arithmetic():
    @kernel
    def top(stride: i32, out: "i32[8]"):
        offset: i32 = 2
        for i in range(4):
            out[offset + stride * i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.index_cast",
        "arith.addi",
        "arith.muli",
    )


def test_builtin_range_loop_store():
    @kernel
    def top(out: "i32[4]"):
        for i in range(4):
            out[i] = i

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.for",
        "to %c4 step %c1",
    )


def test_grid_loop_store():
    @kernel
    def top(out: "i32[2, 2]"):
        for i, j in allo_grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "scf.parallel",
        "step (%c1, %c1)",
        "arith.addi",
        "arith.index_cast",
        "memref<2x2xi32>",
    )


def test_direct_operator_invoke():
    @kernel
    def top(x: i32, y: i32, out: "i32[1]"):
        out[0] = allo_max(x, y)

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.maxsi")


def test_builtin_max_min():
    @kernel
    def top(x: i32, y: i32, out: "i32[2]"):
        out[0] = max(x, y)
        out[1] = min(x, y)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.maxsi",
        "arith.minsi",
    )


def test_global_scalar_constexpr():
    @kernel
    def top(x: i32, y: f32, out: "f32[2]"):
        out[0] = x + _GLOBAL_INT_CONST
        out[1] = y + _GLOBAL_FLOAT_CONST

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 3",
        "arith.constant 1.500000e+00",
        "arith.addi",
        "arith.addf",
    )


def test_global_shape_annotation():
    @kernel
    def top(
        inp: "i32[_GLOBAL_SHAPE_M * _GLOBAL_SHAPE_N]",
        out: "i32[_GLOBAL_SHAPE_M, _GLOBAL_SHAPE_N]",
    ):
        for i in range(_GLOBAL_SHAPE_M):
            for j in range(_GLOBAL_SHAPE_N):
                out[i, j] = inp[i * _GLOBAL_SHAPE_N + j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref<6xi32>",
        "memref<2x3xi32>",
        "scf.for",
    )


def test_scope_shape_annotation():
    rows = 2
    cols = 2

    @kernel
    def top(out: "i32[rows, cols]"):
        for i, j in allo_grid(2, 2):
            out[i, j] = i + j

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref<2x2xi32>",
        "scf.parallel",
    )


def test_template_signature_shape():
    T = Template("T")
    N = Template("N")

    @kernel(T, N)
    def top(x: T, out: "T[N]"):
        tmp: T = x
        for i in range(N):
            out[i] = tmp

    ir = _compile_ir(top[f32, 2])
    _assert_contains(
        ir,
        "f32",
        "memref<2xf32>",
        "scf.for",
    )


def test_template_helper_specialization():
    T = Template("T")

    @kernel(T)
    def worker(x: T) -> T:
        return x

    @kernel(T)
    def top(x: T, out: "T[1]"):
        out[0] = worker[T](x)

    ir = _compile_ir(top[i32])
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "invoke @top.worker",
        "i32",
    )


def test_template_specialization_object():
    T = Template("T")

    @kernel(T)
    def top(x: T, out: "T[1]"):
        out[0] = x

    specialized = top[f32]
    ir = _compile_ir(specialized)
    _assert_contains(ir, "f32", "memref<1xf32>")


def test_local_memref_declaration():
    @kernel
    def top(out: "i32[4]"):
        N: constexpr = 4
        buf: "i32[N]"
        for i in range(N):
            buf[i] = i
            out[i] = buf[i]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "memref.alloc",
        "memref<4xi32>",
        "memref.load",
    )


def test_local_tensor_declaration():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top() -> "f32[4]":
        N: constexpr = 4
        buf: "f32[N]"
        return buf

    ir = _compile_ir(top)
    _assert_contains(ir, "tensor.empty", "tensor<4xf32>")


def test_memref_list_initializer():
    @kernel
    def top(out: "i32[2,2]"):
        scale: constexpr = _GLOBAL_INT_CONST
        buf: "i32[2,2]" = [[1, scale], [scale + 1, scale + 2]]
        for i, j in allo_grid(2, 2):
            out[i, j] = buf[i, j]

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        'memref.global "private" @buf_initializer_0',
        "memref.get_global @buf_initializer_0",
        "dense<[[1, 3], [4, 5]]>",
        "memref.load",
    )


def test_tensor_list_initializer():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top() -> "i32[2,2]":
        buf: "i32[2,2]" = [[1, 2], [3, 4]]
        return buf

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>")


def test_stream_scalar_ir():
    @kernel
    def top(x: i32, out: "i32[1]"):
        fifo: "Stream[i32][2,2]"
        fifo[0, 1].put(x)
        out[0] = fifo[0, 1].get()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.stream.create : !allo.stream<i32,2,[2,2]>",
        "allo.stream.put",
        "allo.stream.get",
    )


def test_stream_nested_parameter_ir():
    @kernel
    def top(x: i32, out: "i32[1]"):
        fifo: "Stream[i32][2,2]"

        @kernel
        def worker(s: "Stream[i32][2,2]", v: i32):
            s[0, 1].put(v)

        worker(fifo, x)
        out[0] = fifo[0, 1].get()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.stream.create : !allo.stream<i32,2,[2,2]>",
        "allo.kernel private @top.worker",
        "(%s: !allo.stream<i32,2,[2,2]>",
        "invoke @top.worker",
        "allo.stream.put",
        "allo.stream.get",
    )


def test_bound_method_compile_errors():
    @kernel
    def top(x: i32):
        x.put(1)

    @kernel
    def worker():
        x: constexpr = 1
        x.put(1)

    _assert_compile_error(
        top,
        "Stream get/put expects a stream value, got 'int32'.",
    )
    _assert_compile_error(
        worker,
        "constexpr value '1' has no attribute 'put'.",
    )


def test_for_loop_carried_values():
    @kernel
    def top(out: "i32[1]"):
        acc: i32 = 0
        for i in range(4):
            i_i32: i32 = i
            acc += i_i32
        out[0] = acc

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.for", "iter_args", "scf.yield")


def test_while_loop_carried_values():
    @kernel
    def top(out: "i32[1]"):
        i: i32 = 0
        acc: i32 = 0
        while i < 4:
            acc += i
            i += 1
        out[0] = acc

    ir = _compile_ir(top)
    _assert_contains(ir, "scf.while", "scf.condition", "scf.yield")


def test_consteval_expression():
    @consteval
    def factor():
        return 3

    @kernel
    def top(x: i32, out: "i32[1]"):
        out[0] = x + factor()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "arith.constant 3 : i33",
        "arith.extsi",
        "arith.addi",
        "i33 to i32",
    )


def test_nested_invoke_store():
    @kernel
    def top(x: i32, out: "i32[1]"):
        @kernel
        def worker(v: i32) -> i32:
            return v + 1

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "invoke @top.worker")


def test_nested_multiple_returns():
    @kernel
    def top(x: i32, y: i32, out: "i32[1]"):
        @kernel
        def worker(a: i32, b: i32) -> (i32, i32):
            return a, b

        lhs, rhs = worker(x, y)
        out[0] = lhs + rhs

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "-> (i32, i32)",
        "invoke @top.worker",
        "arith.addi",
    )


def test_nested_capture_constexpr():
    @kernel
    def top(x: i32, out: "i32[1]"):
        offset: constexpr = 3

        @kernel
        def worker(v: i32) -> i32:
            return v + offset

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "arith.constant 3")


def test_nested_capture_type_alias():
    @kernel
    def top(out: "i32[1]"):
        T: constexpr = i32

        @kernel
        def worker() -> T:
            return 7

        out[0] = worker()

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker",
        "-> i32",
        "arith.constant 7 : i32",
    )


def test_nested_capture_consteval():
    @consteval
    def amount():
        return 5

    @kernel
    def top(x: i32, out: "i32[1]"):
        @kernel
        def worker(v: i32) -> i32:
            return v + amount()

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "arith.constant 5")


def test_nested_capture_kernel_alias():
    @kernel
    def callee(v: i32) -> i32:
        return v + 2

    @kernel
    def top(x: i32, out: "i32[1]"):
        invokeee: constexpr = callee

        @kernel
        def worker(v: i32) -> i32:
            return invokeee(v)

        out[0] = worker(x)

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "allo.kernel private @top.worker.callee",
        "invoke @top.worker.callee",
        "invoke @top.worker",
    )


def test_nested_capture_module_alias():
    @kernel
    def top(out: "i32[2]"):
        M: constexpr = allo_core

        @kernel
        def worker(buf: "i32[2]"):
            for i in M.range(2):
                buf[i] = i

        worker(out)

    ir = _compile_ir(top)
    _assert_contains(ir, "allo.kernel private @top.worker", "scf.for")


def test_cpp_typing_compile():
    @kernel(options=KernelOptions(typing_style="cpp"))
    def top(x: u32, y: i32, out: "u32[1]"):
        out[0] = x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "arith.addi", ": i32")


def test_return_scalar_value():
    @kernel
    def top(x: i32, y: i32) -> i32:
        return x + y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "-> i32",
        "arith.addi",
        "i33 to i32",
    )


def test_return_constexpr_literal():
    @kernel
    def top() -> i32:
        return 3

    ir = _compile_ir(top)
    _assert_contains(ir, "-> i32", "arith.constant 3 : i32")


def test_return_multiple_values():
    @kernel
    def top(x: i32, y: f32) -> (i32, f32):
        return x, y

    ir = _compile_ir(top)
    _assert_contains(
        ir,
        "-> (i32, f32)",
        "return",
        ": i32, f32",
    )


def test_return_if_else():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        else:
            return y

    ir = _compile_ir(top)
    _assert_contains(ir, "cf.cond_br", "return", ": i32")


def test_return_if_fallthrough():
    @kernel
    def top(cond: allo_bool, x: i32, y: i32) -> i32:
        if cond:
            return x
        return y

    ir = _compile_ir(top)
    _assert_contains(ir, "cf.cond_br", "return", ": i32")


def test_return_requires_annotation():
    @kernel
    def top(x: i32):
        return x

    _assert_compile_error(
        top,
        "Return values require an explicit return annotation.",
    )


def test_return_missing_non_void():
    @kernel
    def top(x: i32) -> i32:
        y = x + x

    _assert_compile_error(
        top,
        "Missing return statement for non-void function",
    )


def test_return_count_mismatch():
    @kernel
    def top(x: i32, y: i32) -> (i32, i32):
        return x

    _assert_compile_error(
        top,
        "Return value count mismatch: expected 2, got 1.",
    )


def test_return_type_mismatch():
    @kernel
    def top(x: "i32[2]") -> "i32[1]":
        return x

    _assert_compile_error(
        top,
        "Cannot cast from memref<2xint32> to memref<1xint32>",
    )


def test_return_inside_loop_error():
    @kernel
    def top(x: i32) -> i32:
        for i in allo_range(4):
            return x
        return x

    _assert_compile_error(
        top,
        "'return' is not supported inside loops",
    )


def test_return_nested_if_error():
    @kernel
    def top(cond: allo_bool, inner: allo_bool, x: i32) -> i32:
        if cond:
            if inner:
                return x
        return x

    _assert_compile_error(
        top,
        "'return' is not supported inside nested 'if' statements.",
    )
