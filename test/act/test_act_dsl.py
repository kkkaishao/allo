import subprocess
from pathlib import Path

import pytest

from allo.exp.compiler.errors import ActError
from allo.exp.lang.act import ActPrimitive, ISA, parse_tensor_annotation
from allo.exp.lang.core import bf16, f32, i32
from allo.exp.operators import act as prim


def build_qkv():
    qkv = ISA("QKV")
    hbm = qkv.hbm("d0", shape=(131072,), dtype=bf16)
    d1 = qkv.vector("d1", slots=128, shape=(64,), dtype=bf16)
    d2 = qkv.vector("d2", slots=64, shape=(64,), dtype=bf16)

    @qkv.instruction("load_rm", src=hbm, dst=d1)
    def _(I):
        @I.access
        def _(I):
            addr_in, addr_out, size = I.addr_params("addr_in", "addr_out", "size")
            src = I.expand(
                I.strided(hbm, basis=addr_in, counts=size * 64, strides=1),
                [[0, 1]],
                shape=(size, 64),
            )
            dst = I.strided(d1, basis=addr_out, counts=size, strides=1)
            return src, dst

        @I.compute
        def _(x: "bf16[?,64]", out: "bf16[?,64]"):
            return prim.identity(x)

    @qkv.instruction("load_cm", src=hbm, dst=d1)
    def _(I):
        @I.access
        def _(I):
            addr_in, addr_out, size = I.addr_params("addr_in", "addr_out", "size")
            src = I.expand(
                I.strided(hbm, basis=addr_in, counts=size * 64, strides=1),
                [[0, 1]],
                shape=(size, 64),
            )
            dst = I.strided(d1, basis=addr_out, counts=size, strides=1)
            return I.transpose(src, [1, 0]), I.transpose(dst, [1, 0])

        @I.compute
        def _(x: "bf16[64,?]", out: "bf16[64,?]"):
            return prim.identity(x)

    @qkv.instruction("store_rm", src=d1, dst=hbm)
    def _(I):
        @I.access
        def _(I):
            addr_in, addr_out, size = I.addr_params("addr_in", "addr_out", "size")
            src = I.strided(d1, basis=addr_in, counts=size, strides=1)
            dst = I.expand(
                I.strided(hbm, basis=addr_out, counts=size * 64, strides=1),
                [[0, 1]],
                shape=(size, 64),
            )
            return src, dst

        @I.compute
        def _(x: "bf16[?,64]", out: "bf16[?,64]"):
            return prim.identity(x)

    @qkv.instruction("mov", src=d2, dst=d1)
    def _(I):
        @I.access
        def _(I):
            addr_in, addr_out, size = I.addr_params("addr_in", "addr_out", "size")
            return (
                I.strided(d2, basis=addr_in, counts=size, strides=1),
                I.strided(d1, basis=addr_out, counts=size, strides=1),
            )

        @I.compute
        def _(x: "bf16[?,64]", out: "bf16[?,64]"):
            return prim.identity(x)

    @qkv.instruction("gemm_f32acc", src=[d1, d1], dst=d2)
    def _(I):
        @I.access
        def _(I):
            addr_a, addr_b, addr_c = I.addr_params("addr_a", "addr_b", "addr_c")
            return (
                I.strided(d1, basis=addr_a, counts=64, strides=1),
                I.strided(d1, basis=addr_b, counts=64, strides=1),
                I.strided(d2, basis=addr_c, counts=64, strides=1),
            )

        @I.compute
        def _(a: "bf16[64,64]", b: "bf16[64,64]", c: "bf16[64,64]"):
            return prim.cast(prim.matmul(a, b, dtype=f32), dtype=bf16)

    @qkv.instruction("gemm", src=[d1, d1], dst=d2)
    def _(I):
        @I.access
        def _(I):
            addr_a, addr_b, addr_c = I.addr_params("addr_a", "addr_b", "addr_c")
            return (
                I.strided(d1, basis=addr_a, counts=64, strides=1),
                I.strided(d1, basis=addr_b, counts=64, strides=1),
                I.strided(d2, basis=addr_c, counts=64, strides=1),
            )

        @I.compute
        def _(a: "bf16[64,64]", b: "bf16[64,64]", c: "bf16[64,64]"):
            return prim.matmul(a, b)

    @qkv.instruction("softmax", src=d2, dst=d2)
    def _(I):
        @I.access
        def _(I):
            addr, n = I.addr_params("addr", "n")
            pat = I.strided(d2, basis=addr, counts=n, strides=1)
            return pat, pat

        @I.compute
        def _(x: "bf16[?,64]", out: "bf16[?,64]"):
            return prim.softmax(x, dim=1)

    return qkv


def test_parse_tensor_annotation():
    ty = parse_tensor_annotation("bf16[?,64]")
    assert ty.dtype == bf16
    assert ty.shape == (None, 64)
    assert ty.mlir() == "tensor<?x64xbf16>"
    assert parse_tensor_annotation("'bf16[?,64]'") == ty


def test_primitives_have_schema_and_nodes_reference_it():
    assert isinstance(prim.matmul, ActPrimitive)
    assert prim.matmul.infer_impl is not None
    assert prim.matmul.build_impl is not None
    assert prim.matmul.lower_impl is not None

    qkv = build_qkv()
    gemm = next(inst for inst in qkv.instructions if inst.name == "gemm")
    gemm_f32acc = next(inst for inst in qkv.instructions if inst.name == "gemm_f32acc")
    assert gemm.compute_spec is not None
    assert gemm_f32acc.compute_spec is not None
    assert gemm.compute_spec.nodes[0].primitive is prim.matmul
    assert gemm.compute_spec.nodes[0].kind == "matmul"
    assert gemm_f32acc.compute_spec.nodes[1].primitive is prim.cast


def test_invalid_compute_rejects_bare_tensor_return():
    qkv = ISA("Bad")
    d1 = qkv.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="produced by primitives"):

        @qkv.instruction("bad", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return x


def test_names_must_be_unique_and_valid():
    isa = ISA("Names")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="Duplicate symbol"):
        isa.vector("d1", slots=1, shape=(64,), dtype=bf16)
    with pytest.raises(ActError, match="Duplicate symbol"):
        isa.instruction("d1", src=d1, dst=d1)
    with pytest.raises(ActError, match="Invalid buffer name"):
        isa.vector("bad name", slots=1, shape=(64,), dtype=bf16)


def test_duplicate_instruction_name_is_rejected():
    isa = ISA("DupInst")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    @isa.instruction("copy", src=d1, dst=d1)
    def _(I):
        @I.access
        def _(I):
            addr, n = I.addr_params("addr", "n")
            pat = I.strided(d1, basis=addr, counts=n, strides=1)
            return pat, pat

        @I.compute
        def _(x: "bf16[?,64]", out: "bf16[?,64]"):
            return prim.identity(x)

    with pytest.raises(ActError):

        @isa.instruction("copy", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return prim.identity(x)


def test_address_params_are_unique_and_valid():
    isa = ISA("AddrParams")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError):

        @isa.instruction("dup_param", src=d1, dst=d1)
        def _(I):
            I.addr_params("addr", "addr")

    with pytest.raises(ActError):

        @isa.instruction("bad_param", src=d1, dst=d1)
        def _(I):
            I.addr_params("bad name")


def test_access_pattern_arity_and_base_buffer_are_checked():
    isa = ISA("Address")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)
    d2 = isa.vector("d2", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="1 src \\+ 1 dst"):

        @isa.instruction("bad_arity", src=d1, dst=d2)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                return I.strided(d1, basis=addr, counts=n, strides=1)

    with pytest.raises(ActError, match="dst\\[0\\] buffer 'd2'"):

        @isa.instruction("bad_base", src=d1, dst=d2)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat


def test_strided_access_parameters_are_checked():
    isa = ISA("Strided")
    hbm = isa.hbm("d0", shape=(16, 16), dtype=bf16)
    d1 = isa.vector("d1", slots=4, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="expects rank 2"):

        @isa.instruction("bad_hbm_rank", src=hbm, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                return (
                    I.strided(hbm, basis=addr, counts=n, strides=1),
                    I.strided(d1, basis=0, counts=1, strides=1),
                )

    with pytest.raises(ActError, match="counts must be positive"):

        @isa.instruction("bad_count", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                return (
                    I.strided(d1, basis=0, counts=0, strides=1),
                    I.strided(d1, basis=0, counts=1, strides=1),
                )

    with pytest.raises(ActError, match="out of bounds"):

        @isa.instruction("bad_bounds", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                return (
                    I.strided(d1, basis=3, counts=2, strides=1),
                    I.strided(d1, basis=0, counts=1, strides=1),
                )


def test_relayout_patterns_are_checked():
    isa = ISA("Relayout")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError):

        @isa.instruction("bad_expand", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr = I.addr_params("addr")
                pat = I.strided(d1, basis=addr, counts=1, strides=1)
                return I.expand(pat, [[0], [1]], shape=(1, 64))

    with pytest.raises(ActError, match="cover dimensions"):

        @isa.instruction("bad_collapse", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr = I.addr_params("addr")
                pat = I.expand(
                    I.strided(d1, basis=addr, counts=1, strides=1),
                    [[0, 1]],
                    shape=(1, 64),
                )
                return I.collapse(pat, [[1, 0]])

    with pytest.raises(ActError, match="does not match rank"):

        @isa.instruction("bad_transpose", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr = I.addr_params("addr")
                pat = I.strided(d1, basis=addr, counts=1, strides=1)
                return I.transpose(pat, [1, 0])


def test_access_and_compute_are_defined_once():
    isa = ISA("DefineOnce")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="already defines access patterns"):

        @isa.instruction("dup_access", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                pat = I.strided(d1, basis=0, counts=1, strides=1)
                return pat, pat

            @I.access
            def _(I):
                pat = I.strided(d1, basis=0, counts=1, strides=1)
                return pat, pat

    with pytest.raises(ActError, match="already defines compute semantics"):

        @isa.instruction("dup_compute", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                pat = I.strided(d1, basis=0, counts=1, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[64]", out: "bf16[64]"):
                return prim.identity(x)

            @I.compute
            def _(x: "bf16[64]", out: "bf16[64]"):
                return prim.identity(x)


def test_access_return_and_static_shape_consistency_are_checked():
    isa = ISA("AccessReturn")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="must return access pattern"):

        @isa.instruction("bad_return", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                return 1

    with pytest.raises(ActError, match="static shape mismatch"):

        @isa.instruction("bad_expand_product", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                pat = I.strided(d1, basis=0, counts=1, strides=1)
                return I.expand(pat, [[0, 1]], shape=(2, 33))


def test_collapse_static_product_and_dst_value_semantics():
    isa = ISA("StaticCollapse")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(ActError, match="dimension 0 expects 64"):

        @isa.instruction("bad_collapse_annotation", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                expanded = I.expand(
                    I.strided(d1, basis=0, counts=1, strides=1),
                    [[0, 1]],
                    shape=(8, 8),
                )
                collapsed = I.collapse(expanded, [[0, 1]])
                return collapsed, collapsed

            @I.compute
            def _(x: "bf16[32]", out: "bf16[64]"):
                return prim.identity(x)

    @isa.instruction("read_dst", src=d1, dst=d1)
    def _(I):
        @I.access
        def _(I):
            I.addr_params("addr")
            pat = I.strided(d1, basis=0, counts=1, strides=1)
            return pat, pat

        @I.compute
        def _(x: "bf16[64]", out: "bf16[64]"):
            return prim.identity(out)

    assert "act.define @read_dst" in isa.emit_mlir()


def test_compute_arguments_are_checked_against_buffers_and_patterns():
    isa = ISA("ComputeArgs")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)

    with pytest.raises(
        ActError, match="compute argument 'x' for src\\[0\\] buffer 'd1'"
    ):

        @isa.instruction("bad_dtype", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "f32[?,64]", out: "bf16[?,64]"):
                return prim.identity(x)

    with pytest.raises(ActError, match="expects 2 arguments"):

        @isa.instruction("bad_arity", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]"):
                return prim.identity(x)

    with pytest.raises(
        ActError,
        match="compute argument 'x' for src\\[0\\] buffer 'd1'.*dimension 1 expects 64",
    ):

        @isa.instruction("bad_dim", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,32]", out: "bf16[?,64]"):
                return prim.identity(x)

    with pytest.raises(ActError, match="default value"):

        @isa.instruction("bad_default", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]" = None):
                return prim.identity(x)

    with pytest.raises(ActError, match="require tensor type annotations"):

        @isa.instruction("missing_annotation", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x, out: "bf16[?,64]"):
                return prim.identity(x)


def test_primitive_and_return_types_are_checked():
    isa = ISA("Primitives")
    d1 = isa.vector("d1", slots=1, shape=(64,), dtype=bf16)
    a_buf = isa.vector("a", slots=2, shape=(3,), dtype=bf16)
    b_buf = isa.vector("b", slots=4, shape=(2,), dtype=bf16)
    c_buf = isa.vector("c", slots=2, shape=(2,), dtype=bf16)

    with pytest.raises(ActError, match="incompatible"):

        @isa.instruction("bad_matmul", src=[a_buf, b_buf], dst=c_buf)
        def _(I):
            @I.access
            def _(I):
                I.addr_params("addr")
                return (
                    I.strided(a_buf, basis=0, counts=2, strides=1),
                    I.strided(b_buf, basis=0, counts=4, strides=1),
                    I.strided(c_buf, basis=0, counts=2, strides=1),
                )

            @I.compute
            def _(a: "bf16[2,3]", b: "bf16[4,2]", c: "bf16[2,2]"):
                return prim.matmul(a, b)

    with pytest.raises(ActError, match="out of range"):

        @isa.instruction("bad_softmax_dim", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return prim.softmax(x, dim=2)

    with pytest.raises(ActError):

        @isa.instruction("bad_cast", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return prim.cast(x, dtype=i32)

    with pytest.raises(ActError, match="return value for dst\\[0\\] buffer 'd1'"):

        @isa.instruction("bad_return", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return prim.cast(prim.identity(x), dtype=f32)

    with pytest.raises(ActError, match="must return 1 value"):

        @isa.instruction("bad_return_arity", src=d1, dst=d1)
        def _(I):
            @I.access
            def _(I):
                addr, n = I.addr_params("addr", "n")
                pat = I.strided(d1, basis=addr, counts=n, strides=1)
                return pat, pat

            @I.compute
            def _(x: "bf16[?,64]", out: "bf16[?,64]"):
                return prim.identity(x), prim.identity(x)


def test_qkv_emit_smoke():
    text = build_qkv().emit_mlir()
    assert "act.buffer @d0 size(1) : !act.hbm<131072xbf16>" in text
    assert "act.define @gemm_f32acc" in text
    assert "linalg.matmul" in text
    assert "linalg.softmax dimension(1)" in text


def test_qkv_act_opt_integration(tmp_path):
    act_opt = Path("build/bin/act-opt")
    if not act_opt.exists():
        pytest.skip("build/bin/act-opt is not available")
    isa_path = tmp_path / "qkv_generated.mlir"
    build_qkv().emit_mlir(path=isa_path)

    subprocess.run([act_opt, isa_path, "--verify-diagnostics"], check=True)
    subprocess.run(
        [
            act_opt,
            "drafts/models/mm.mlir",
            f"--convert-canonical-form-to-act=isa-path={isa_path}",
        ],
        check=True,
    )
    subprocess.run(
        [
            act_opt,
            "drafts/models/flat_attention.mlir",
            f"--convert-canonical-form-to-act=isa-path={isa_path}",
        ],
        check=True,
    )
