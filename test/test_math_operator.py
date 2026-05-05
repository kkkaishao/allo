import pytest

from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.compiler.errors import CompilationError
from allo.exp.lang.core import f32, i32
from allo.exp.lang.kernel import KernelOptions, kernel
from allo.exp.operators import math as allo_math


def _compile_ir(fn) -> str:
    return str(compile_kernel(fn))


def _assert_contains(ir: str, *patterns: str):
    for pattern in patterns:
        assert pattern in ir


def _assert_not_contains(ir: str, *patterns: str):
    for pattern in patterns:
        assert pattern not in ir


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


@pytest.mark.parametrize(
    ("op", "pattern"),
    [
        pytest.param(allo_math.exp, "math.exp ", id="exp"),
        pytest.param(allo_math.exp2, "math.exp2", id="exp2"),
        pytest.param(allo_math.log, "math.log ", id="log"),
        pytest.param(allo_math.log2, "math.log2", id="log2"),
        pytest.param(allo_math.sqrt, "math.sqrt", id="sqrt"),
        pytest.param(allo_math.rsqrt, "math.rsqrt", id="rsqrt"),
        pytest.param(allo_math.sin, "math.sin", id="sin"),
        pytest.param(allo_math.cos, "math.cos", id="cos"),
        pytest.param(allo_math.tan, "math.tan", id="tan"),
        pytest.param(allo_math.floor, "math.floor", id="floor"),
        pytest.param(allo_math.ceil, "math.ceil", id="ceil"),
        pytest.param(allo_math.erf, "math.erf", id="erf"),
    ],
)
def test_unary_math_operator_mlir(op, pattern):
    @kernel
    def unary_math_operator(x: f32, out: "f32[1]"):
        out[0] = op(x)

    ir = _compile_ir(unary_math_operator)
    _assert_contains(ir, "func.func @unary_math_operator", pattern, "memref.store")


def test_abs_float_operator_mlir():
    @kernel
    def abs_float_operator(x: f32, out: "f32[1]"):
        out[0] = allo_math.abs(x)

    ir = _compile_ir(abs_float_operator)
    _assert_contains(ir, "func.func @abs_float_operator", "math.absf")


def test_abs_int_operator_mlir():
    @kernel
    def abs_int_operator(x: i32, out: "i32[1]"):
        out[0] = allo_math.abs(x)

    ir = _compile_ir(abs_int_operator)
    _assert_contains(ir, "func.func @abs_int_operator", "math.absi")


def test_pow_float_float_operator_mlir():
    @kernel
    def pow_float_float_operator(x: f32, y: f32, out: "f32[1]"):
        out[0] = allo_math.pow(x, y)

    ir = _compile_ir(pow_float_float_operator)
    _assert_contains(ir, "func.func @pow_float_float_operator", "math.powf")


def test_pow_float_int_operator_mlir():
    @kernel
    def pow_float_int_operator(x: f32, y: i32, out: "f32[1]"):
        out[0] = allo_math.pow(x, y)

    ir = _compile_ir(pow_float_int_operator)
    _assert_contains(ir, "func.func @pow_float_int_operator", "math.fpowi")


def test_pow_int_int_operator_mlir():
    @kernel
    def pow_int_int_operator(x: i32, y: i32, out: "i32[1]"):
        out[0] = allo_math.pow(x, y)

    ir = _compile_ir(pow_int_int_operator)
    _assert_contains(ir, "func.func @pow_int_int_operator", "math.ipowi")


def test_exp_zero_fold():
    @kernel
    def exp_zero_fold(out: "f32[1]"):
        out[0] = allo_math.exp(0)

    ir = _compile_ir(exp_zero_fold)
    _assert_contains(ir, "func.func @exp_zero_fold", "arith.constant 1")
    _assert_not_contains(ir, "math.exp")


def test_pow_zero_fold():
    @kernel
    def pow_zero_fold(x: f32, out: "f32[1]"):
        out[0] = allo_math.pow(x, 0)

    ir = _compile_ir(pow_zero_fold)
    _assert_contains(ir, "func.func @pow_zero_fold", "arith.constant 1")
    _assert_not_contains(ir, "math.powf", "math.fpowi", "math.ipowi")


def test_tensor_exp_uses_linalg_named_op():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_exp_operator(x: "f32[4]") -> "f32[4]":
        return allo_math.exp(x)

    ir = _compile_ir(tensor_exp_operator)
    _assert_contains(ir, "func.func @tensor_exp_operator", "linalg.exp")


def test_tensor_exp2_uses_linalg_generic_fallback():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_exp2_operator(x: "f32[4]") -> "f32[4]":
        return allo_math.exp2(x)

    ir = _compile_ir(tensor_exp2_operator)
    _assert_contains(
        ir, "func.func @tensor_exp2_operator", "linalg.generic", "math.exp2"
    )


def test_tensor_exp_reuses_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_exp_acc_operator(x: "f32[4]", acc: "f32[4]") -> "f32[4]":
        return allo_math.exp(x, acc=acc)

    ir = _compile_ir(tensor_exp_acc_operator)
    _assert_contains(ir, "func.func @tensor_exp_acc_operator", "linalg.exp")
    _assert_not_contains(ir, "tensor.empty")


def test_memref_exp_requires_acc():
    @kernel
    def memref_exp_missing_acc(x: "f32[4]", out: "f32[1]"):
        y: "f32[4]" = allo_math.exp(x)
        out[0] = y[0]

    _assert_compile_error(memref_exp_missing_acc, "requires acc for memref output")


def test_memref_exp_uses_acc():
    @kernel
    def memref_exp_acc_operator(x: "f32[4]", out: "f32[4]"):
        allo_math.exp(x, acc=out)

    ir = _compile_ir(memref_exp_acc_operator)
    _assert_contains(ir, "func.func @memref_exp_acc_operator", "linalg.exp")
    _assert_not_contains(ir, "memref.alloc", "-> tensor<4xf32>")


def test_scalar_exp_acc_error():
    @kernel
    def scalar_exp_acc_error(out: "f32[4]"):
        allo_math.exp(0, acc=out)

    _assert_compile_error(
        scalar_exp_acc_error, "acc requires at least one shaped operand"
    )


def test_scalar_pow_acc_error():
    @kernel
    def scalar_pow_acc_error(x: f32, y: f32, out: "f32[4]"):
        allo_math.pow(x, y, acc=out)

    _assert_compile_error(
        scalar_pow_acc_error, "acc requires at least one shaped operand"
    )


def test_tensor_exp_acc_shape_mismatch_error():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_exp_acc_shape_mismatch(x: "f32[4]", acc: "f32[2]") -> "f32[2]":
        return allo_math.exp(x, acc=acc)

    _assert_compile_error(tensor_exp_acc_shape_mismatch, "not broadcastable")


def test_memref_exp2_uses_linalg_generic_fallback():
    @kernel
    def memref_exp2_operator(x: "f32[4]", out: "f32[4]"):
        allo_math.exp2(x, acc=out)

    ir = _compile_ir(memref_exp2_operator)
    _assert_contains(
        ir, "func.func @memref_exp2_operator", "linalg.generic", "math.exp2"
    )
    _assert_not_contains(ir, "-> tensor<4xf32>")


def test_tensor_pow_broadcasts_scalar_to_acc_shape():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_pow_scalar_acc_operator(x: "f32[4]", acc: "f32[4]") -> "f32[4]":
        return allo_math.pow(x, 2, acc=acc)

    ir = _compile_ir(tensor_pow_scalar_acc_operator)
    _assert_contains(
        ir, "func.func @tensor_pow_scalar_acc_operator", "linalg.generic", "math.fpowi"
    )
