import pytest

from allo.exp.compiler.errors import CompilationError
from allo.exp.compiler.mlir_codegen import compile as compile_kernel
from allo.exp.lang.core import f32
from allo.exp.lang.kernel import KernelOptions, kernel
from allo.exp.operators import linalg as allo_linalg


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


def test_tensor_matmul_operator_mlir():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_matmul_operator(a: "f32[2, 3]", b: "f32[3, 4]") -> "f32[2, 4]":
        return allo_linalg.matmul(a, b)

    ir = _compile_ir(tensor_matmul_operator)
    _assert_contains(ir, "func.func @tensor_matmul_operator", "linalg.matmul")


def test_memref_matmul_requires_acc():
    @kernel
    def memref_matmul_missing_acc(a: "f32[2, 3]", b: "f32[3, 4]", out: "f32[1]"):
        y: "f32[2, 4]" = allo_linalg.matmul(a, b)
        out[0] = y[0, 0]

    _assert_compile_error(memref_matmul_missing_acc, "requires acc for memref output")


def test_memref_matmul_uses_acc():
    @kernel
    def memref_matmul_acc_operator(a: "f32[2, 3]", b: "f32[3, 4]", out: "f32[2, 4]"):
        allo_linalg.matmul(a, b, acc=out)

    ir = _compile_ir(memref_matmul_acc_operator)
    _assert_contains(ir, "func.func @memref_matmul_acc_operator", "linalg.matmul")
    _assert_not_contains(ir, "memref.alloc")


def test_matmul_shape_mismatch_error():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_matmul_shape_mismatch(a: "f32[2, 3]", b: "f32[2, 4]") -> "f32[2, 4]":
        return allo_linalg.matmul(a, b)

    _assert_compile_error(
        tensor_matmul_shape_mismatch, "incompatible contraction dimensions"
    )


def test_tensor_dot_operator_mlir():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_dot_operator(a: "f32[4]", b: "f32[4]") -> "f32[]":
        return allo_linalg.dot(a, b)

    ir = _compile_ir(tensor_dot_operator)
    _assert_contains(ir, "func.func @tensor_dot_operator", "linalg.dot")
    _assert_not_contains(ir, "tensor.extract")


def test_tensor_dot_rank0_scalar_extract_mlir():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_dot_scalar_operator(a: "f32[4]", b: "f32[4]") -> f32:
        return allo_linalg.dot(a, b)[()]

    ir = _compile_ir(tensor_dot_scalar_operator)
    _assert_contains(
        ir, "func.func @tensor_dot_scalar_operator", "linalg.dot", "tensor.extract"
    )


def test_tensor_dot_uses_rank0_acc_annotation():
    @kernel(options=KernelOptions(enable_tensor=True))
    def tensor_dot_acc_operator(a: "f32[4]", b: "f32[4]", acc: "f32[]") -> "f32[]":
        return allo_linalg.dot(a, b, acc=acc)

    ir = _compile_ir(tensor_dot_acc_operator)
    _assert_contains(ir, "func.func @tensor_dot_acc_operator", "linalg.dot")
    _assert_not_contains(ir, "tensor.empty", "tensor.extract")


def test_memref_dot_uses_acc():
    @kernel
    def memref_dot_acc_operator(a: "f32[4]", b: "f32[4]", acc: "f32[]"):
        allo_linalg.dot(a, b, acc=acc)

    ir = _compile_ir(memref_dot_acc_operator)
    _assert_contains(ir, "func.func @memref_dot_acc_operator", "linalg.dot")
    _assert_not_contains(ir, "memref.load")
