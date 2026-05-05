import pytest

from allo.exp.compiler.errors import CompilationError
from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.lang.core import f32, u1, u32
from allo.exp.lang.kernel import KernelOptions, kernel
from allo.exp.operators import arith as allo_arith


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


def test_tensor_add_uses_linalg_named_op():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_add_operator(x: "f32[4]", y: "f32[4]") -> "f32[4]":
        return x + y

    ir = _compile_ir(tensor_add_operator)
    _assert_contains(ir, "func.func @tensor_add_operator", "linalg.add")


def test_tensor_rank0_add_uses_linalg_named_op():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_rank0_add_operator(x: "f32[]", y: "f32[]") -> "f32[]":
        return x + y

    ir = _compile_ir(tensor_rank0_add_operator)
    _assert_contains(ir, "func.func @tensor_rank0_add_operator", "linalg.add")


def test_tensor_add_broadcasts_scalar():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_add_scalar_operator(x: "f32[4]") -> "f32[4]":
        return x + 1.0

    ir = _compile_ir(tensor_add_scalar_operator)
    _assert_contains(ir, "func.func @tensor_add_scalar_operator", "linalg.add")


def test_memref_add_python_operator_requires_acc():
    @kernel
    def memref_add_missing_acc(x: "f32[4]", y: "f32[4]", out: "f32[1]"):
        z: "f32[4]" = x + y
        out[0] = z[0]

    _assert_compile_error(memref_add_missing_acc, "requires acc for memref output")


def test_memref_add_direct_call_uses_acc():
    @kernel
    def memref_add_acc_operator(x: "f32[4]", y: "f32[4]", out: "f32[4]"):
        allo_arith.add(x, y, acc=out)

    ir = _compile_ir(memref_add_acc_operator)
    _assert_contains(ir, "func.func @memref_add_acc_operator", "linalg.add")
    _assert_not_contains(ir, "memref.alloc")


def test_memref_div_accepts_positional_acc_before_signed():
    @kernel
    def memref_div_acc_position(x: "u32[4]", y: "u32[4]", out: "u32[4]"):
        allo_arith.div(x, y, out, signed=False)

    ir = _compile_ir(memref_div_acc_position)
    _assert_contains(ir, "func.func @memref_div_acc_position", "linalg.div_unsigned")
    _assert_not_contains(ir, "memref.alloc")


def test_tensor_lt_uses_linalg_generic_fallback():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_lt_operator(x: "f32[4]", y: "f32[4]") -> "u1[4]":
        return x < y

    ir = _compile_ir(tensor_lt_operator)
    _assert_contains(
        ir, "func.func @tensor_lt_operator", "linalg.generic", "arith.cmpf"
    )


def test_tensor_lt_accepts_positional_acc_before_ordered():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_lt_acc_position(x: "f32[4]", y: "f32[4]", out: "u1[4]") -> "u1[4]":
        return allo_arith.lt(x, y, out, ordered=True)

    ir = _compile_ir(tensor_lt_acc_position)
    _assert_contains(
        ir, "func.func @tensor_lt_acc_position", "linalg.generic", "arith.cmpf"
    )


def test_tensor_max_accepts_positional_acc_before_propagate_nan():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_max_acc_position(x: "f32[4]", y: "f32[4]", out: "f32[4]") -> "f32[4]":
        return allo_arith.max(x, y, out, propagate_nan=True)

    ir = _compile_ir(tensor_max_acc_position)
    _assert_contains(ir, "func.func @tensor_max_acc_position", "linalg.generic")


def test_scalar_add_acc_error():
    @kernel
    def scalar_add_acc_error(x: f32, y: f32, out: "f32[4]"):
        allo_arith.add(x, y, acc=out)

    _assert_compile_error(
        scalar_add_acc_error, "acc requires at least one shaped operand"
    )
