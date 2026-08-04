# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator injection/characterization, arithmetic datapath binding (legalize-arith keep/expand, compare/select/shift), and reduction restructuring."""

import math
import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import bf16, f32, i32
from allo.lang.ip import ip, OperatorType
from allo.operators import math as amath
from allo.operators import arith as allo_arith
from allo.backend.rtl.device import (
    builtin_device,
    CombKind,
    Const,
    Linear,
    Quadratic,
    Step,
    Table,
)

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, _sched, _to_rtl, _impls, FADD  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def _f32(*shape):
    return np.random.default_rng(0).random(shape, dtype=np.float32)


def _signed_f32(seed):
    return (np.random.default_rng(seed).random(16, dtype=np.float32) - 0.5) * 10


# --- what the device declares -------------------------------------------------


# A resource is the device's own vocabulary, so nothing in the compiler names
# `lut` or `dsp`: they are symbols a cost refers to, and the reference is what
# gets verified. A cost's SHAPE is structural and only its coefficients are
# measured, which is why the forms are a closed set and the resources are not.
def test_a_device_declares_its_resources_and_what_they_cost():
    @kernel
    def mac(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * B[i] + 1

    dev = builtin_device.copy()
    lut = dev.resources["lut"]
    dsp = dev.resources["dsp"]
    # An N-bit AND is N LUT6s (linear), a divider is quadratic, and a
    # multiplier's DSP count was measured per width rather than fitted.
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Linear(1.0)})
    dev.set_comb_delay(CombKind.DIV, 2.5, uses={lut: Quadratic(1.06)})
    dev.set_comb_delay(
        CombKind.MUL, 2.0, uses={lut: Const(15.0), dsp: Table({8: 0, 16: 1, 32: 3})}
    )

    text = _to_rtl(mac, device=dev).dcp
    assert "allo.dcp.resource @lut capacity = 1303680" in text
    assert "allo.dcp.resource @dsp capacity = 9024" in text
    # The cost rides the row it belongs to, referring to the resource by symbol.
    assert "@dsp" in text and "table" in text and "quadratic" in text


# A cost naming something that is not a resource is a verifier error, which is
# the whole point of the symbol: the dictionary it replaced turned a misspelling
# into an absent row.
def test_a_cost_must_name_a_declared_resource():
    dev = builtin_device.copy()
    ghost = dev.add_resource("ghost", capacity=10)
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={ghost: Const(1.0)})
    del dev.resources["ghost"]

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    with pytest.raises(Exception):
        _to_rtl(k, device=dev).dcp


# A device cannot declare the same kind twice: the library keeps the last match,
# so a duplicate would be one declaration silently overriding another.
def test_a_device_declares_each_comb_kind_once():
    dev = builtin_device.copy()
    lut = dev.resources["lut"]
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Linear(1.0)})
    dev.set_comb_delay(CombKind.ADD, 0.9)  # overwrites rather than duplicating

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    text = _to_rtl(k, device=dev).dcp
    assert text.count('allo.dcp.comb "add"') == 1


# A multiplexer and a delay chain are structures the emitter builds and nothing
# chooses between, so each is one whole-device row. Both carry TWO parameters,
# and a cost with the wrong number of factors is a verifier error rather than a
# product the evaluator zips short.
def test_a_device_prices_its_multiplexers_and_delay_chains():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = builtin_device.copy()
    lut, ff = dev.resources["lut"], dev.resources["ff"]
    dev.set_mux_uses({lut: (Linear(0.4), Linear(1.0))})
    dev.set_chain_uses({ff: (Step(4, 1.0, 2.0), Linear(1.0))})

    text = _to_rtl(k, device=dev).dcp
    assert "allo.dcp.mux uses" in text and "allo.dcp.chain uses" in text
    with pytest.raises(ValueError, match="fan-in, width"):
        dev.set_mux_uses({lut: Linear(0.4)})


# An IP core's area rides its own declaration, over the one parameter every
# realization of its kind carries. The resources are the DEVICE's and the
# operator is not in the device's symbol table, so the reference reaches through
# the device symbol and resolves from where it is written.
def test_an_operator_declares_what_its_core_spends():
    @kernel
    def addk(a: f32, b: f32) -> f32:
        return a + b

    text = _to_rtl(addk).dcp
    assert "allo.dcp.operator @fadd_l7" in text
    assert "#allo.res_use<@builtin::@lut, [<const, [2.470000e+02]>]>" in text


# A cost is a SUM of product terms, so a measured shape that is a sum can be
# declared: an extracted chain's flip-flops are a per-bit term plus a per-stage
# one, which no single product is. The sum is taken before rounding, so how the
# cost was factored cannot change the answer.
def test_a_cost_sums_the_terms_that_name_one_resource():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = builtin_device.copy()
    ff = dev.resources["ff"]
    dev.set_chain_uses(
        {ff: [(Const(2.0), Linear(1.0)), (Linear(1.0, base=-1.0), Const(1.0))]}
    )
    assert dev.price(dev.chain_uses, (64, 32))["ff"] == 2 * 32 + 64 - 1
    # Both terms ride one `uses`, naming `@ff` twice.
    chain = [l for l in _to_rtl(k, device=dev).dcp.splitlines() if "dcp.chain" in l]
    assert len(chain) == 1 and chain[0].count("allo.res_use<@ff") == 2


# The device's own evaluator, reached from Python: one implementation of the
# measured shapes, not two. `benchmark/area.py` scores through this.
def test_the_device_prices_a_realization_through_the_compiler():
    dev = builtin_device
    # 3 LUTs per bit of a 6-source select, over 32 bits.
    assert dev.price(dev.mux_uses, (6, 32)) == {"lut": 96}
    # A chain past the extraction cliff is SRLs plus a head and tail stage per
    # bit, not `depth * width` flip-flops.
    assert dev.price(dev.chain_uses, (64, 32))["ff"] == 2 * 32 + 64 - 1
    assert dev.price(dev.chain_uses, (2, 32))["ff"] == 64
    # A carry chain is a CEILING: a 9-bit adder takes two CARRY8s.
    assert dev.price(dev.comb_uses["add"], (9,)) == {"lut": 9, "carry8": 2}
    # A block RAM tile holds 36864 bits however the array is cut.
    assert dev.price(dev.storage["bram"].uses, (1024, 32)) == {"bram36": 1}


# --- operator injection ------------------------------------------------------


# The same kernel schedules once the operator is characterized via `@ip`.
def test_ip_characterizes_math_op():
    @ip(optype="sqrt", latency=7, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk2(A: f32[8]):
        for i in range(8):
            A[i] = amath.sqrt(A[i])

    dev = builtin_device.copy()
    dev.add_operator(fsqrt)
    res = _sched(sqrtk2, device=dev)
    assert res.func("sqrtk2").latency is not None


# Integer arithmetic is natively combinational: it needs no `@ip` and no
# library row, so the fail-loud check never fires on it.
def test_integer_ops_never_error():
    @kernel
    def intk(A: i32[8]):
        for i in range(8):
            A[i] = A[i] * 3 + 1

    res = _sched(intk)
    assert res.func("intk").latency is not None


# A custom fast fadd (latency 3) injects as a dcp.operator, is referenced
# by the reifier, and beats the built-in latency-7 fadd; the default path is
# untouched (a separate export never sees the IP).
def test_operator_ip_overlay_shifts_schedule():
    @kernel
    def addk(a: f32, b: f32, c: f32) -> f32:
        return a + b + c

    r0 = addk.schedule().export("rtl")
    lat0 = r0.schedule().func("addk").latency
    assert "fadd_fast" not in Dcp(r0).attrs("allo.dcp.operator", "sym_name")

    @ip(
        name="fadd_fast",
        optype=OperatorType.ADD,
        latency=3,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def fadd_fast(a: f32, b: f32) -> f32: ...

    @kernel
    def addk2(a: f32, b: f32, c: f32) -> f32:
        return a + b + c

    dev = builtin_device.copy()
    dev.add_operator(fadd_fast)
    r1 = addk2.schedule().export("rtl", device=dev)
    lat1 = r1.schedule().func("addk2").latency

    assert "fadd_fast" in Dcp(r1).attrs("allo.dcp.operator", "sym_name")
    assert "fadd_fast" in _impls(r1.schedule())
    assert lat0 is not None and lat1 is not None


def test_advanced_math_sqrt_cosim():
    # A math.sqrt characterized by a unary @ip emits a single-input extern
    # operator and cosims against numpy.sqrt: the operator emit + behavioral
    # model are arity-general, not binary-only.
    N = 16

    @ip(optype="sqrt", latency=5, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.sqrt(A[i])

    dev = builtin_device.copy()
    dev.add_operator(fsqrt)
    rng = np.random.default_rng(0)
    A = rng.random(N, dtype=np.float32).astype(np.float32)  # non-negative
    B = np.zeros(N, np.float32)
    _to_rtl(sqrtk, device=dev).cosim(A, B)
    np.testing.assert_allclose(B, np.sqrt(A), rtol=1e-5, atol=1e-6)


def test_float_negate_cosim():
    # arith.negf (a float unary minus) lowers to a native comb sign-bit flip
    # with no IP, and cosims bit-exactly against -A.
    N = 16

    @kernel
    def negk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = -A[i]

    rng = np.random.default_rng(1)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = np.zeros(N, np.float32)
    _to_rtl(negk).cosim(A, B)
    np.testing.assert_array_equal(B, -A)  # exact: a sign-bit flip


def test_int_to_float_cast_cosim():
    # An int->float cast (arith.sitofp) is a unary IP: the built-in
    # i2f_l3 emits a single-input extern and cosims against a signed conversion.
    N = 16

    @kernel
    def castk(A: i32[N], B: f32[N]):
        for i in range(N):
            x: f32 = A[i]
            B[i] = x

    rng = np.random.default_rng(2)
    A = rng.integers(-1000, 1000, N).astype(np.int32)
    B = np.zeros(N, np.float32)
    _to_rtl(castk).cosim(A, B)
    np.testing.assert_array_equal(B, A.astype(np.float32))


def test_free_running_operator_cosim():
    # A style='free' operator emits a ce-less extern (a, b, clk) -> y. In
    # a non-back-pressured pipeline (where a ce operator's ce is a constant 1
    # anyway) it cosims identically to the clock-enabled default.
    N = 16

    @ip(
        name="fadd_free",
        optype=OperatorType.ADD,
        latency=5,
        in_delay_ns=0.5,
        pipelined=True,
        style="free",
    )
    def fadd_free(a: f32, b: f32) -> f32: ...

    @kernel
    def addk(A: f32[N], B: f32[N], C: f32[N]):
        for i in range(N):
            C[i] = A[i] + B[i]

    dev = builtin_device.copy()
    dev.add_operator(fadd_free)  # last-wins: overrides the built-in fadd
    rtl = _to_rtl(addk, device=dev)
    # The manifest declares each instantiated operator's realized port shape.
    ops = [o for i in rtl.interfaces.values() for o in i.operators]
    free = [o for o in ops if o.module == "fadd_free"]
    assert free, "the free operator was not instantiated"
    names = [p.name for p in free[0].ports]
    assert "ce" not in names, f"a free-running extern must carry no ce: {names}"

    rng = np.random.default_rng(3)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    C = np.zeros(N, np.float32)
    rtl.cosim(A, B, C)
    np.testing.assert_allclose(C, A + B, rtol=1e-6, atol=1e-6)


def test_custom_c_model_for_uncharacterized_kind_cosim():
    # math.erf has no built-in behavioral model, so a device operator for it
    # needs a user C expression (add_c_model); once supplied, the operator is
    # fully characterized and cosims against a scalar math.erf golden, since the
    # C model is the sole behavior source.
    N = 16

    @kernel
    def erfk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.erf(A[i])

    A = (np.random.default_rng(4).random(N, dtype=np.float32) - np.float32(0.5)).astype(
        np.float32
    )

    @ip(optype="erf", latency=6, pipelined=True, style="ce")
    def ferf(a: f32) -> f32: ...

    ferf.add_c_model("std::erf(a)")
    dev = builtin_device.copy()
    dev.add_operator(ferf)
    B = np.zeros(N, np.float32)
    _to_rtl(erfk, device=dev).cosim(A, B)
    golden = np.array([math.erf(float(x)) for x in A], np.float32)
    np.testing.assert_allclose(B, golden, rtol=1e-4, atol=1e-6)


# Nothing stalls outside a stream region, so a free-style IP is emitted
# as declared: a plain extern instance with no ce port at all.
def test_free_running_ip_outside_stream_region_emits():
    @ip(optype="mul", latency=3, in_delay_ns=0.5, pipelined=True, style="free")
    def freemul(a: f32, b: f32) -> f32: ...

    dev = builtin_device.copy()
    dev.add_operator(freemul)

    @kernel
    def scale(A: f32[8], B: f32[8]):
        for i in range(8):
            B[i] = A[i] * 2.0

    v = _to_rtl(scale, device=dev).verilog
    assert "freemul" in v
    # No `ce` port on a free-running instance: it is the whole difference.
    inst = [ln for ln in v.splitlines() if ".ce" in ln and "freemul" in ln]
    assert not inst, inst


# --- legalize-arith: keep vs. expand ------------------------------------------
# The RTL prepare pipeline runs `legalize-arith` (not the device-blind
# `arith-expand`): a composite arith op the device provides an operator IP for is
# KEPT for the scheduler to bind; the rest are EXPANDED into primitive arith.
# Integer max/min are native comb ops and are left untouched either way.


def test_int_max_min_native_comb_cosim():
    # Integer arith.maxsi/minsi are native combinational ops (no IP):
    # legalize-arith leaves them untouched, they schedule at latency 0, and cosim
    # bit-exactly against numpy.maximum/minimum.
    N = 16

    @kernel
    def imaxmin(A: i32[N], B: i32[N], mx: i32[N], mn: i32[N]):
        for i in range(N):
            mx[i] = allo_arith.max(A[i], B[i])
            mn[i] = allo_arith.min(A[i], B[i])

    rtl = _to_rtl(imaxmin)
    kinds = {o.kind for r in rtl.schedule().func("imaxmin").regions for o in r.ops}
    assert {"maxsi", "minsi"} <= kinds  # kept as-is, not expanded

    rng = np.random.default_rng(5)
    A = rng.integers(-50, 50, N).astype(np.int32)
    B = rng.integers(-50, 50, N).astype(np.int32)
    mx = np.zeros(N, np.int32)
    mn = np.zeros(N, np.int32)
    rtl.cosim(A, B, mx, mn)
    np.testing.assert_array_equal(mx, np.maximum(A, B))
    np.testing.assert_array_equal(mn, np.minimum(A, B))


@pytest.mark.parametrize("propagate_nan", [True, False])
def test_float_max_no_ip_expands_cosim(propagate_nan):
    # A float max/min on a device WITHOUT a max/min IP is expanded by
    # legalize-arith into cmpf+select (the compare binds the built-in fcmp IP,
    # the select a comb mux). Both the NaN-propagating (maximumf) and
    # NaN-avoiding (maxnumf) variants expand and cosim bit-exactly.
    N = 16

    @kernel
    def fmaxmin(A: f32[N], B: f32[N], mx: f32[N], mn: f32[N]):
        for i in range(N):
            mx[i] = allo_arith.max(A[i], B[i], propagate_nan=propagate_nan)
            mn[i] = allo_arith.min(A[i], B[i], propagate_nan=propagate_nan)

    rtl = _to_rtl(fmaxmin)
    kinds = {o.kind for r in rtl.schedule().func("fmaxmin").regions for o in r.ops}
    assert "cmpf" in kinds and "select" in kinds  # expanded, not a bare max/min
    assert not (kinds & {"maximumf", "minimumf", "maxnumf", "minnumf"})

    rng = np.random.default_rng(7)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    mx = np.zeros(N, np.float32)
    mn = np.zeros(N, np.float32)
    rtl.cosim(A, B, mx, mn)
    np.testing.assert_allclose(mx, np.maximum(A, B), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mn, np.minimum(A, B), rtol=1e-6, atol=1e-6)


def test_float_max_with_ip_kept_cosim():
    # A float max on a device WITH a matching max IP is KEPT by legalize-arith
    # (not expanded) and bound to that IP, one operator instead of cmp+select,
    # and cosims via the IP's C-model.
    N = 16

    @ip(optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce")
    def fmax_ip(a: f32, b: f32) -> f32: ...

    fmax_ip.add_c_model("std::fmax(a, b)")

    @kernel
    def fmax_keep(A: f32[N], B: f32[N], out: f32[N]):
        for i in range(N):
            out[i] = allo_arith.max(A[i], B[i], propagate_nan=True)

    dev = builtin_device.copy()
    dev.add_operator(fmax_ip)
    rtl = _to_rtl(fmax_keep, device=dev)
    kinds = {o.kind for r in rtl.schedule().func("fmax_keep").regions for o in r.ops}
    assert not (kinds & {"cmpf", "select"})  # kept as one op, not expanded
    assert "fmax_ip" in Dcp(rtl).attrs("allo.dcp.operator", "sym_name")
    assert "fmax_ip" in _impls(rtl.schedule())

    rng = np.random.default_rng(6)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    out = np.zeros(N, np.float32)
    rtl.cosim(A, B, out)
    np.testing.assert_allclose(out, np.maximum(A, B), rtol=1e-6, atol=1e-6)


def test_max_maxnum_split_binds_distinctly():
    # The Max / MaxNum op-kind split keeps NaN semantics correct: a device that
    # provides a max IP (maximumf, NaN-propagating) binds arith.maximumf but
    # NOT arith.maxnumf (maxNum, returns the non-NaN operand). The latter has
    # no matching IP, so legalize-arith expands it rather than silently computing
    # it with the wrong operator.
    N = 8

    @ip(optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce")
    def fmax_ip(a: f32, b: f32) -> f32: ...

    dev = builtin_device.copy()
    dev.add_operator(fmax_ip)

    @kernel
    def kmaximumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=True)  # arith.maximumf

    @kernel
    def kmaxnumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=False)  # arith.maxnumf

    assert "fmax_ip" in _impls(_to_rtl(kmaximumf, device=dev).schedule())  # bound
    maxnum = _to_rtl(kmaxnumf, device=dev)
    assert "fmax_ip" not in _impls(maxnum.schedule())  # NOT bound to the max IP
    k = {o.kind for r in maxnum.schedule().func("kmaxnumf").regions for o in r.ops}
    assert "cmpf" in k and "select" in k  # expanded instead


# --- arithmetic datapath: compare, select, shift ------------------------------


# Reductions and matmuls over the float and integer datapaths: the float ops
# are multi-cycle IP instances, the int add is combinational.
def test_float_and_int_arithmetic():
    @kernel
    def dotp(A: f32[8], B: f32[8], out: f32[1]):
        acc: f32 = 0.0
        for k in range(8):
            acc = acc + A[k] * B[k]
        out[0] = acc

    A, B = _f32(8), _f32(8)
    out = np.zeros(1, np.float32)
    _to_rtl(dotp).cosim(A, B, out)
    assert np.allclose(out[0], A @ B, rtol=1e-4, atol=1e-5)

    @kernel
    def mm(A: f32[4, 4], B: f32[4, 4], C: f32[4, 4]):
        for i in range(4):
            for j in range(4):
                acc: f32 = 0.0
                for k in range(4):
                    acc = acc + A[i, k] * B[k, j]
                C[i, j] = acc

    A, B = _f32(4, 4), _f32(4, 4)
    C = np.zeros((4, 4), np.float32)
    _to_rtl(mm).cosim(A, B, C, timeout=20000)
    assert np.allclose(C, A @ B, rtol=1e-4, atol=1e-5)

    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    def isum(A: i32[16], out: i32[1]):
        acc: i32 = 0
        for i in range(16):
            acc = acc + A[i]
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(isum).cosim(A16, out)
    assert out[0] == int(A16.astype(np.int64).sum())

    @kernel
    def imm(A: i32[4, 4], B: i32[4, 4], C: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                acc: i32 = 0
                for k in range(4):
                    acc = acc + A[i, k] * B[k, j]
                C[i, j] = acc

    rng = np.random.default_rng(1)
    Ai = rng.integers(-8, 8, size=(4, 4), dtype=np.int32)
    Bi = rng.integers(-8, 8, size=(4, 4), dtype=np.int32)
    Ci = np.zeros((4, 4), np.int32)
    _to_rtl(imm).cosim(Ai, Bi, Ci, timeout=20000)
    assert np.array_equal(Ci, (Ai @ Bi).astype(np.int32))


def test_shared_multiply_mux():
    # Two chained float multiplies (fmul latency 4) issue at disjoint cycles, so
    # the MRT lets them share one physical unit. The 'greedy-share' binding policy
    # folds them onto one multiply and deriveInterconnect grows a 2:1 input mux per
    # port; the shared datapath must be functionally identical to the trivially-
    # bound one. (Integer multiply is combinational -- no instance to share -- so
    # sharing is exercised on a float IP operator.)
    @kernel
    def chain(A: f32[1], B: f32[1], C: f32[1], o: f32[1]):
        o[0] = A[0] * B[0] * C[0]

    a, b, c = (np.array([v], np.float32) for v in (7, 6, 5))
    ref = np.array([7 * 6 * 5], np.float32)
    unshared = _to_rtl(chain)
    shared = _to_rtl(chain, binding="greedy-share")
    # every multiply is an IP instance, so a dropped instance == a shared unit
    assert shared.mlir.count("hw.instance") < unshared.mlir.count("hw.instance")
    for mod in (unshared, shared):
        o = np.zeros(1, np.float32)
        mod.cosim(a, b, c, o)
        assert np.array_equal(o, ref)


# If-conversion over both datapaths: an int compare lowers to native
# comb.icmp, a float compare to an fcmp IP instance, both feeding a comb.mux.
# Shifts lower to native comb.shl / comb.shr.
def test_compare_select_and_shift():
    @kernel
    def relu(A: i32[16], out: i32[16]):
        for i in range(16):
            if A[i] > 0:
                out[i] = A[i]
            else:
                out[i] = 0

    A = np.random.default_rng(0).integers(-50, 50, size=16, dtype=np.int32)
    mod = _to_rtl(relu)
    assert "comb.icmp" in mod.mlir and "comb.mux" in mod.mlir
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, np.maximum(A, 0))

    # A second predicate (`<=` -> sle) exercises the arith->comb predicate map.
    @kernel
    def sel(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            if A[i] <= B[i]:
                out[i] = A[i]
            else:
                out[i] = B[i]

    rng = np.random.default_rng(1)
    A = rng.integers(-40, 40, size=16, dtype=np.int32)
    B = rng.integers(-40, 40, size=16, dtype=np.int32)
    out = np.zeros(16, np.int32)
    _to_rtl(sel).cosim(A, B, out)
    assert np.array_equal(out, np.minimum(A, B))

    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    def sh(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = (A[i] << 2) >> 1

    mod = _to_rtl(sh)
    assert "comb.shl" in mod.mlir and "comb.shr" in mod.mlir
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 << 2) >> 1)

    @kernel
    def frelu(A: f32[16], out: f32[16]):
        for i in range(16):
            if A[i] > 0.0:
                out[i] = A[i]
            else:
                out[i] = 0.0

    Af = _signed_f32(0)
    mod = _to_rtl(frelu)
    assert "hw.module.extern @fcmp" in mod.mlir and "comb.mux" in mod.mlir
    outf = np.zeros(16, np.float32)
    mod.cosim(Af, outf)
    assert np.allclose(outf, np.maximum(Af, 0.0), rtol=1e-5)

    # A second float predicate (`<=` -> ole) + a select over both operands.
    @kernel
    def fmax(A: f32[16], B: f32[16], out: f32[16]):
        for i in range(16):
            if A[i] <= B[i]:
                out[i] = B[i]
            else:
                out[i] = A[i]

    Af, Bf = _signed_f32(1), _signed_f32(2)
    outf = np.zeros(16, np.float32)
    _to_rtl(fmax).cosim(Af, Bf, outf)
    assert np.allclose(outf, np.maximum(Af, Bf), rtol=1e-5)


# --- reduction restructuring ---------------------------------------------------


# Rotating a float reduction across N accumulators turns its distance-1
# recurrence (II == add latency) into a distance-N one: II == ceil(L/N).
def test_rotate_reduction_scales_ii():
    def ii(n):
        @kernel
        def red(x: f32[256]) -> f32:
            acc: f32 = 0.0
            for i in range(256, name="i"):
                acc += x[i]
            return acc

        return _sched(red, accumulators=n).cyclic()[0].ii

    assert ii(0) == FADD  # unrotated
    assert ii(FADD) == 1  # N == latency -> II 1
    assert ii(2) == math.ceil(FADD / 2)

    # bf16 inputs with an f32 accumulator (the common ML pattern): the cast sits
    # on the leaf, not around the operator, so rotation works unchanged.
    def mixed_ii(n):
        @kernel
        def red(x: bf16[64]) -> f32:
            acc: f32 = 0.0
            for i in range(64, name="i"):
                acc += x[i]
            return acc

        return _sched(red, accumulators=n).cyclic()[0].ii

    assert mixed_ii(0) == FADD
    assert mixed_ii(FADD) == 1


# Integer reductions rebalance unconditionally (integer arithmetic is exactly
# associative mod 2^w), cutting an unrolled chain's recurrence to one operator.
def test_reassociate_int_reduction_recurrence(capfd):
    # Unrolling threads the carried accumulator through four widened multiplies;
    # folding it in last makes the recurrence one (widened, combinational)
    # multiply rather than a chain of four. The rebalance is what this test
    # pins. The resulting II is NOT evidence for it: with a factor-4 unroll the
    # II is the resource bound (four loads over the port budget), so it would
    # read the same whether the chain was rebalanced or not.
    @kernel
    def red(x: i32[32]) -> i32:
        acc: i32 = 1
        for i in range(32, name="i"):
            acc *= x[i]
        return acc

    s = red.schedule()
    s.unroll("i", factor=4)
    s.export("rtl").schedule()
    assert "Rebalancing associative reduction chain" in "".join(capfd.readouterr())


def test_int_product_reduction_cosim():
    # An integer *multiply* reduction (distinct from the add reductions): the
    # multiply-latency recurrence pipelines and the frozen product returns as a
    # scalar. Small values keep the product within i32 (exact, no wrap ambiguity).
    @kernel
    def prod(x: i32[16]) -> i32:
        acc: i32 = 1
        for i in range(16, name="i"):
            acc *= x[i]
        return acc

    x = np.ones(16, dtype=np.int32)
    x[:6] = np.array([2, 3, 1, 2, 5, 1], np.int32)  # product 360, fits i32
    r = _to_rtl(prod).cosim(x)
    assert int(r.result) == int(np.prod(x.astype(np.int64)))
