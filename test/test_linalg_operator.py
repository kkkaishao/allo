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


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


def test_tensor_matmul_mlir():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(a: "f32[2, 3]", b: "f32[3, 4]") -> "f32[2, 4]":
        return allo_linalg.matmul(a, b)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.matmul")


def test_memref_matmul_requires_acc():
    @kernel
    def top(a: "f32[2, 3]", b: "f32[3, 4]", out: "f32[1]"):
        y: "f32[2, 4]" = allo_linalg.matmul(a, b)
        out[0] = y[0, 0]

    _assert_compile_error(top, "requires acc for memref output")


def test_memref_matmul_acc():
    @kernel
    def top(a: "f32[2, 3]", b: "f32[3, 4]", out: "f32[2, 4]"):
        allo_linalg.matmul(a, b, acc=out)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.matmul")


def test_matmul_shape_error():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(a: "f32[2, 3]", b: "f32[2, 4]") -> "f32[2, 4]":
        return allo_linalg.matmul(a, b)

    _assert_compile_error(top, "incompatible contraction dimensions")


def test_tensor_dot_mlir():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(a: "f32[4]", b: "f32[4]") -> "f32[]":
        return allo_linalg.dot(a, b)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.dot")


def test_tensor_dot_scalar_extract():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(a: "f32[4]", b: "f32[4]") -> f32:
        return allo_linalg.dot(a, b)[()]

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.dot", "tensor.extract")


def test_tensor_dot_rank0_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(a: "f32[4]", b: "f32[4]", acc: "f32[]") -> "f32[]":
        return allo_linalg.dot(a, b, acc=acc)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.dot")


def test_memref_dot_acc():
    @kernel
    def top(a: "f32[4]", b: "f32[4]", acc: "f32[]"):
        allo_linalg.dot(a, b, acc=acc)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.dot")
