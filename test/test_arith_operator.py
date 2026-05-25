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


def _assert_compile_error(fn, *patterns: str):
    with pytest.raises(CompilationError) as exc_info:
        _compile_ir(fn)
    message = exc_info.value.error_msg
    for pattern in patterns:
        assert pattern in message


def test_tensor_add_linalg():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[4]", y: "f32[4]") -> "f32[4]":
        return x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add")


def test_tensor_rank0_add_linalg():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[]", y: "f32[]") -> "f32[]":
        return x + y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add", "tensor<f32>")


def test_tensor_add_scalar_broadcast():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[4]") -> "f32[4]":
        return x + 1.0

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add", "arith.constant 1.000000e+00")


def test_memref_add_requires_acc():
    @kernel
    def top(x: "f32[4]", y: "f32[4]", out: "f32[1]"):
        z: "f32[4]" = x + y
        out[0] = z[0]

    _assert_compile_error(top, "requires acc for memref output")


def test_memref_add_acc():
    @kernel
    def top(x: "f32[4]", y: "f32[4]", out: "f32[4]"):
        allo_arith.add(x, y, acc=out)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.add")


def test_memref_div_positional_acc():
    @kernel
    def top(x: "u32[4]", y: "u32[4]", out: "u32[4]"):
        allo_arith.div(x, y, out, signed=False)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.div_unsigned")


def test_tensor_lt_generic():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[4]", y: "f32[4]") -> "u1[4]":
        return x < y

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic", "arith.cmpf")


def test_tensor_lt_positional_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[4]", y: "f32[4]", out: "u1[4]") -> "u1[4]":
        return allo_arith.lt(x, y, out, ordered=True)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic", "arith.cmpf")


def test_tensor_max_positional_acc():
    @kernel(options=KernelOptions(enable_tensor=True))
    def top(x: "f32[4]", y: "f32[4]", out: "f32[4]") -> "f32[4]":
        return allo_arith.max(x, y, out, propagate_nan=True)

    ir = _compile_ir(top)
    _assert_contains(ir, "linalg.generic")


def test_scalar_add_acc_error():
    @kernel
    def top(x: f32, y: f32, out: "f32[4]"):
        allo_arith.add(x, y, acc=out)

    _assert_compile_error(top, "acc requires at least one shaped operand")
