# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator injection/characterization, arithmetic datapath binding (legalize-arith keep/expand, compare/select/shift), and reduction restructuring."""

import collections
import math
import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import bf16, f16, f32, i32, u32, KernelOptions
from allo.lang.core import APInt
from allo.lang.ip import operator_ip, OperatorType
from allo.operators import math as amath
from allo.operators import arith as allo_arith
from allo.backend.rtl import has_exact_scheduler
from allo.backend.rtl.devices import default_device
from allo.backend.rtl.device import (
    CombKind,
    Const,
    Linear,
    Quadratic,
    Step,
    Table,
)

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    Dcp,
    _sched,
    _to_rtl,
    _impls,
    _iis,
    _latency,
    FADD,
    comb_ns,
    comb_step_ns,
    REG_NS,
    PERIOD_NS,
)  # noqa: E402

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

    dev = default_device.copy()
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


# A cost naming something that is not a resource is a verifier error, so a
# misspelled name fails loudly instead of becoming an absent row.
def test_a_cost_must_name_a_declared_resource():
    dev = default_device.copy()
    ghost = dev.add_resource("ghost", capacity=10)
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={ghost: Const(1.0)})
    del dev.resources["ghost"]

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    with pytest.raises(RuntimeError):
        _to_rtl(k, device=dev).dcp


# A device cannot declare the same kind twice: the library keeps the last match,
# so a duplicate would be one declaration silently overriding another.
def test_a_device_declares_each_comb_kind_once():
    dev = default_device.copy()
    lut = dev.resources["lut"]
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Linear(1.0)})
    dev.set_comb_delay(CombKind.ADD, 0.9)  # overwrites rather than duplicating

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    text = _to_rtl(k, device=dev).dcp
    assert text.count("allo.dcp.comb add delay") == 1


# A multiplexer and a delay chain are structures the emitter builds and nothing
# chooses between, so each is one whole-device row. Both carry TWO parameters,
# and a cost with the wrong number of factors is a verifier error rather than a
# product the evaluator zips short.
def test_a_device_prices_its_multiplexers_and_delay_chains():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    lut, ff = dev.resources["lut"], dev.resources["ff"]
    dev.set_mux_uses({lut: (Linear(0.4), Linear(1.0))})
    dev.set_chain_uses({ff: (Step(4, 1.0, 2.0), Linear(1.0))})

    text = _to_rtl(k, device=dev).dcp
    assert "allo.dcp.mux uses" in text and "allo.dcp.chain uses" in text
    with pytest.raises(ValueError):
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
    core = "add_f32_f32_f32_l7"
    assert f"allo.dcp.operator @{core}" in text
    scope = default_device.name  # the device the reference reaches through
    # The count is read back off the device rather than restated: what this pins
    # is that the reference resolves through the device symbol.
    luts = dict(default_device.operator_uses[core])["lut"][0].coeffs[0]
    assert f"#allo.res_use<@{scope}::@lut, [<const, [{luts:.6e}]>]>" in text


# A cost is a sum of product terms, so a measured shape that is a sum can be
# declared: an extracted chain's flip-flops are a per-bit term plus a per-stage
# one. The sum is taken before rounding, so the factoring cannot change the
# answer.
def test_a_cost_sums_the_terms_that_name_one_resource():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    ff = dev.resources["ff"]
    dev.set_chain_uses(
        {ff: [(Const(2.0), Linear(1.0)), (Linear(1.0, base=-1.0), Const(1.0))]}
    )
    assert dev.price(dev.chain_uses, (64, 32))["ff"] == 2 * 32 + 64 - 1
    # Both terms ride one `uses`, naming `@ff` twice.
    chain = [l for l in _to_rtl(k, device=dev).dcp.splitlines() if "dcp.chain" in l]
    assert len(chain) == 1 and chain[0].count("allo.res_use<@ff") == 2


# The device's own evaluator, reached from Python: one implementation of the
# measured shapes, not two. `allo/backend/rtl/qor.py` estimates through this.
def test_the_device_prices_a_realization_through_the_compiler():
    dev = default_device
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


# The same kernel schedules once the operator is characterized via `@operator_ip`.
def test_ip_characterizes_math_op():
    @operator_ip(optype="sqrt", latency=7, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk2(A: f32[8]):
        for i in range(8):
            A[i] = amath.sqrt(A[i])

    dev = default_device.copy()
    dev.add_operator(fsqrt)
    res = _sched(sqrtk2, device=dev)
    assert res.func("sqrtk2").latency is not None


# Integer arithmetic is natively combinational: it needs no `@operator_ip` and no
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

    @operator_ip(
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

    dev = default_device.copy()
    dev.add_operator(fadd_fast)
    r1 = addk2.schedule().export("rtl", device=dev)
    lat1 = r1.schedule().func("addk2").latency

    # A faster core than the built-in add, so it takes a symbol of its own.
    assert fadd_fast.symbol not in Dcp(r0).attrs("allo.dcp.operator", "sym_name")
    assert fadd_fast.symbol in Dcp(r1).attrs("allo.dcp.operator", "sym_name")
    assert fadd_fast.symbol in _impls(r1.schedule())
    assert lat0 is not None and lat1 is not None


def test_advanced_math_sqrt_cosim():
    # A math.sqrt characterized by a unary @ip emits a single-input extern
    # operator and cosims against numpy.sqrt: the operator emit + behavioral
    # model are arity-general, not binary-only.
    N = 16

    @operator_ip(optype="sqrt", latency=5, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.sqrt(A[i])

    dev = default_device.copy()
    dev.add_operator(fsqrt)
    rng = np.random.default_rng(0)
    A = rng.random(N, dtype=np.float32).astype(np.float32)  # non-negative
    B = np.zeros(N, np.float32)
    _to_rtl(sqrtk, device=dev).cosim(A, B)
    np.testing.assert_allclose(B, np.sqrt(A), rtol=1e-5, atol=1e-6)


def test_non_pipelined_ip_bounds_the_initiation_interval():
    # A non-pipelined unit takes one input per latency window, so a loop that
    # re-issues it every II cycles needs II >= latency. Nothing else here bounds
    # the interval (two arrays, two ports each, no carried recurrence), so the
    # pipelined twin of the same IP runs at II=1 and the `pipelined` flag alone
    # is the difference.
    #
    # The behavioral model an `@operator_ip` emits accepts an input every cycle
    # whatever the flag says, so the cosim below passes either way. Only the II
    # catches a datapath that would feed a real unit faster than it accepts.
    N, LAT = 16, 3

    def _dev(pipelined):
        @operator_ip(
            optype="sqrt",
            latency=LAT,
            in_delay_ns=0.5,
            pipelined=pipelined,
            # A non-pipelined IP declares no stall style; it takes the ce default.
            style="ce" if pipelined else None,
        )
        def fsqrt(a: f32) -> f32: ...

        dev = default_device.copy()
        dev.add_operator(fsqrt)
        return dev

    @kernel
    def sqrtk4(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.sqrt(A[i])

    assert _iis(_sched(sqrtk4, device=_dev(True)).func("sqrtk4").regions) == [1]
    assert _iis(_sched(sqrtk4, device=_dev(False)).func("sqrtk4").regions) == [LAT]

    rng = np.random.default_rng(11)
    A = rng.random(N, dtype=np.float32).astype(np.float32)  # non-negative
    B = np.zeros(N, np.float32)
    _to_rtl(sqrtk4, device=_dev(False)).cosim(A, B)
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
    # An int->float cast (arith.sitofp) is a unary IP: the built-in core emits a
    # single-input extern and cosims against a signed conversion.
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

    @operator_ip(
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

    dev = default_device.copy()
    # Last-wins over the built-in f32 add: a different core takes a symbol of
    # its own, and the (kind, signature) it shares is what makes it override.
    dev.add_operator(fadd_free)
    rtl = _to_rtl(addk, device=dev)
    # The manifest declares each instantiated operator's realized port shape.
    ops = [o for i in rtl.interfaces.values() for o in i.operators]
    free = [o for o in ops if o.module == fadd_free.symbol]
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

    @operator_ip(optype="erf", latency=6, pipelined=True, style="ce")
    def ferf(a: f32) -> f32: ...

    ferf.add_c_model("std::erf(a)")
    dev = default_device.copy()
    dev.add_operator(ferf)
    B = np.zeros(N, np.float32)
    _to_rtl(erfk, device=dev).cosim(A, B)
    golden = np.array([math.erf(float(x)) for x in A], np.float32)
    np.testing.assert_allclose(B, golden, rtol=1e-4, atol=1e-6)


# Nothing stalls outside a stream region, so a free-style IP is emitted
# as declared: a plain extern instance with no ce port at all.
def test_free_running_ip_outside_stream_region_emits():
    @operator_ip(optype="mul", latency=3, in_delay_ns=0.5, pipelined=True, style="free")
    def freemul(a: f32, b: f32) -> f32: ...

    dev = default_device.copy()
    dev.add_operator(freemul)

    @kernel
    def scale(A: f32[8], B: f32[8]):
        for i in range(8):
            B[i] = A[i] * 2.0

    v = _to_rtl(scale, device=dev).verilog
    assert freemul.symbol in v
    # No `ce` port on a free-running instance: it is the whole difference.
    inst = [ln for ln in v.splitlines() if ".ce" in ln and freemul.symbol in ln]
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
    # (not expanded) and bound to that IP, one operator instead of cmp+select.
    # `max` is an OperatorType, so the built-in model table supplies its
    # behavior and no add_c_model is needed.
    N = 16

    @operator_ip(
        optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def fmax_ip(a: f32, b: f32) -> f32: ...

    @kernel
    def fmax_keep(A: f32[N], B: f32[N], out: f32[N]):
        for i in range(N):
            out[i] = allo_arith.max(A[i], B[i], propagate_nan=True)

    dev = default_device.copy()
    dev.add_operator(fmax_ip)
    rtl = _to_rtl(fmax_keep, device=dev)
    kinds = {o.kind for r in rtl.schedule().func("fmax_keep").regions for o in r.ops}
    assert not (kinds & {"cmpf", "select"})  # kept as one op, not expanded
    assert fmax_ip.symbol in Dcp(rtl).attrs("allo.dcp.operator", "sym_name")
    assert fmax_ip.symbol in _impls(rtl.schedule())

    rng = np.random.default_rng(6)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    out = np.zeros(N, np.float32)
    rtl.cosim(A, B, out)
    np.testing.assert_allclose(out, np.maximum(A, B), rtol=1e-6, atol=1e-6)


def _wide_add_ip(width):
    wide = APInt(width, signed=True)

    @operator_ip(
        optype=OperatorType.ADD, latency=3, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def wadd(a: wide, b: wide) -> wide: ...

    return wide, wadd


def test_wide_int_operator_ip_cosim():
    # An operator core at a width the C types have no name for. Its model is
    # native RTL, which carries the port's own 48 bits, so the accumulator wraps
    # in simulation exactly where the declared type says it does.
    from allo.backend.rtl.device import operator_descs
    from allo.backend.rtl.sim import ip_models

    i48, wadd = _wide_add_ip(48)

    @kernel
    def dot(x: i32[8], y: i32[8]) -> i48:
        acc: i48 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    dev = default_device.copy()
    dev.add_operator(wadd)
    rtl = _to_rtl(dot, device=dev)
    ops = [o for i in rtl.interfaces.values() for o in i.operators]
    assert [p.width for p in ops[0].ports if p.role == "data"] == [48, 48]
    # No DPI at all: an integer core needs no C, so nothing caps it at 64 bits.
    assert ip_models.dpi_c(rtl.interfaces, operator_descs(dev.operators)) == ""

    rng = np.random.default_rng(0)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert abs(exact) > 2**47, "inputs must overflow i48 for the wrap to matter"
    assert int(rtl.cosim(x, y).result) == ((exact + 2**47) % 2**48) - 2**47


def test_behavior_language_follows_the_domain():
    # A core's behavior language follows from the core: an integer one is native
    # RTL (exact at any width), a float one is C over the DPI, and a user
    # `add_c_model` is C whatever the domain.
    from allo.backend.rtl.device import operator_descs
    from allo.backend.rtl.sim import ip_models

    i128, wadd = _wide_add_ip(128)

    # A latency the built-in i32 multiply core does not already occupy: a
    # symbol names one piece of hardware, so two cores of the same kind and
    # signature are told apart by their depth.
    @operator_ip(optype=OperatorType.MUL, latency=3, pipelined=True, style="ce")
    def imul(a: i32, b: i32) -> i32: ...

    imul.add_c_model("a * b")

    @kernel
    def wide(x: i32[4], y: i32[4], out: i32[4], z: f32[4]) -> i128:
        acc: i128 = 0
        for k in range(4, name="k"):
            out[k] = x[k] * y[k]
            z[k] = z[k] + z[k]
            acc = acc + out[k]
        return acc

    dev = default_device.copy()
    dev.add_operators(wadd, imul)
    rtl = _to_rtl(wide, device=dev)
    descs = operator_descs(dev.operators)
    sv = ip_models.sv_models(rtl.interfaces, descs)
    c = ip_models.dpi_c(rtl.interfaces, descs)
    # The 128-bit add is a wire in RTL; the user-modelled multiply and the float
    # add are the only two that reach C.
    assert f"module {wadd.symbol}(" in sv and "wire [127:0] f = a + b;" in sv
    assert wadd.symbol not in c
    assert f"allo_op_{imul.symbol}(" in c
    assert "allo_ld_f32(p0)" in c


def test_float_format_picks_its_own_codec():
    # Each float format decodes through the codec for its own layout. binary16
    # and bfloat16 are both 16 bits and neither is a narrowed binary32, so a
    # model falling back to the nearest-looking format computes garbage.
    from allo.backend.rtl.device import operator_descs
    from allo.backend.rtl.sim import ip_models

    @operator_ip(
        optype=OperatorType.ADD, latency=3, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def hadd(a: f16, b: f16) -> f16: ...

    @kernel
    def addk(A: f16[16], B: f16[16], C: f16[16]):
        for i in range(16):
            C[i] = A[i] + B[i]

    dev = default_device.copy()
    dev.add_operator(hadd)
    rtl = _to_rtl(addk, device=dev)
    c = ip_models.dpi_c(rtl.interfaces, operator_descs(dev.operators))
    body = c.split(f"allo_op_{hadd.symbol}(")[1]
    assert "allo_ld_f16(p0)" in body and "allo_st_f16(r, _r)" in body, body[:400]


def test_max_maxnum_split_binds_distinctly():
    # The Max / MaxNum op-kind split keeps NaN semantics correct: a device that
    # provides a max IP (maximumf, NaN-propagating) binds arith.maximumf but
    # NOT arith.maxnumf (maxNum, returns the non-NaN operand). The latter has
    # no matching IP, so legalize-arith expands it rather than silently computing
    # it with the wrong operator.
    N = 8

    @operator_ip(
        optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def fmax_ip(a: f32, b: f32) -> f32: ...

    dev = default_device.copy()
    dev.add_operator(fmax_ip)

    @kernel
    def kmaximumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=True)  # arith.maximumf

    @kernel
    def kmaxnumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=False)  # arith.maxnumf

    assert fmax_ip.symbol in _impls(_to_rtl(kmaximumf, device=dev).schedule())  # bound
    maxnum = _to_rtl(kmaxnumf, device=dev)
    assert fmax_ip.symbol not in _impls(maxnum.schedule())  # NOT bound to the max IP
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


def test_bit_slice_lowers_to_arithmetic():
    # No phase below the frontend models a bit field, so `legalize-arith`
    # expands it into the integer arithmetic the operator library prices, before
    # the schedule is cut so the chaining solve sees the field access at its real
    # combinational depth. Covers every shape the frontend admits: constant and
    # dynamic offset, read and write, and the width-one slice a bare `x[k]` is.
    @kernel
    def fields(A: u32[16], B: u32[16], C: u32[16], D: u32[16]):
        for i in range(16):
            B[i] = A[i][8:16]  # constant offset, read
            w: u32 = 0
            w[0:8] = A[i][0:8]  # constant offset, write
            w[8:16] = A[i][24:32]
            C[i] = w
            v: u32 = 0
            v[i : i + 8] = A[i][i : i + 8]  # dynamic offset, both ways
            v[3] = A[i][0]  # width-one slice
            D[i] = v

    A = np.random.default_rng(11).integers(0, 2**32, 16, dtype=np.uint64)
    A = A.astype(np.uint32)
    idx = np.arange(16, dtype=np.uint32)
    want_c = (A & 0xFF) | (((A >> 24) & 0xFF) << 8)
    want_d = ((((A >> idx) & 0xFF) << idx) & ~np.uint32(1 << 3)) | ((A & 1) << 3)

    mod = _to_rtl(fields)
    # Nothing of the allo dialect survives into the datapath.
    assert "allo.bit" not in mod.dcp_module.operation.get_asm()
    # A constant offset is a bit selection, not a shifter: CIRCT folds a shift
    # by a literal back into extract / concat. A dynamic offset cannot fold,
    # `comb.extract` taking its low bit as an attribute, so it keeps a shifter.
    assert ">>" not in mod.verilog.split("B_wr0_data")[1].split(";")[0]

    B, C, D = (np.zeros(16, np.uint32) for _ in range(3))
    mod.cosim(A.copy(), B, C, D)
    assert np.array_equal(B, (A >> 8) & 0xFF)
    assert np.array_equal(C, want_c)
    assert np.array_equal(D, want_d.astype(np.uint32))


def _op_kinds(fn):
    """How many of each operation the schedule placed in the leaf region."""
    ops = _sched(fn).func(fn.__name__).regions[0].ops
    return collections.Counter(o.kind for o in ops)


def test_bit_field_write_drops_redundant_masks():
    # Splicing a field masks the hole it fills, and the splices chain, so four
    # field writes put four AND-OR pairs on one combinational path. Where the
    # bits a mask clears are ones the value provably never sets (every field of
    # a word that started at zero) the mask computes nothing and the forward bit
    # walk in `narrow-demanded-bits` removes it, leaving the concatenation the
    # write really is.
    @kernel
    def pack(A: u32[16], B: u32[16]):
        for i in range(16):
            w: u32 = 0
            w[0:8] = A[i][0:8]
            w[8:16] = A[i][8:16]
            w[16:24] = A[i][16:24]
            w[24:32] = A[i][24:32]
            B[i] = w

    @kernel
    def copy(A: u32[16], B: u32[16]):
        for i in range(16):
            B[i] = A[i]

    # A mask over a field that already holds data is load-bearing and stays, as
    # is one over a signed shift, whose high bits are the replicated sign.
    @kernel
    def overwrite(A: u32[16], V: u32[16], B: u32[16]):
        for i in range(16):
            w: u32 = A[i]
            w[8:16] = V[i][0:8]
            B[i] = w

    @kernel
    def signed_mask(A: i32[16], B: i32[16]):
        for i in range(16):
            s: i32 = A[i] >> 4
            B[i] = s & 65535

    assert _op_kinds(pack)["andi"] == 0
    assert _op_kinds(overwrite)["andi"] == 1
    assert _op_kinds(signed_mask)["andi"] == 1
    # The payoff: a word rebuilt field by field costs no more than copying it.
    assert _latency(pack) == _latency(copy)

    rng = np.random.default_rng(9)
    A = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    V = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    Ai = rng.integers(-(2**31), 2**31, 16).astype(np.int32)

    B = np.zeros(16, np.uint32)
    _to_rtl(pack).cosim(A.copy(), B)
    assert np.array_equal(B, A)
    B = np.zeros(16, np.uint32)
    _to_rtl(overwrite).cosim(A.copy(), V.copy(), B)
    assert np.array_equal(B, (A & np.uint32(0xFFFF00FF)) | ((V & 0xFF) << 8))
    B = np.zeros(16, np.int32)
    _to_rtl(signed_mask).cosim(Ai.copy(), B)
    assert np.array_equal(B, (Ai >> 4) & 0xFFFF)


def test_disjoint_or_is_a_concatenation():
    # Two values sharing no set bit concatenate rather than combine: every result
    # bit takes one side while the other contributes a constant zero. Three such
    # ORs chained cost nothing and settle at one sub-cycle position, where three
    # overlapping ones stand a gate delay apart. The forward bit walk in
    # `narrow-demanded-bits` is what tells them apart.
    @kernel
    def disjoint(A: u32[64], B: u32[64], C: u32[64]):
        for i in range(64):
            lo: u32 = A[i][0:8]
            hi: u32 = B[i][0:8]
            a: u32 = lo | (hi << 8)
            b: u32 = a | (lo << 16)
            C[i] = b | (hi << 24)

    @kernel
    def overlapping(A: u32[64], B: u32[64], C: u32[64]):
        for i in range(64):
            a: u32 = A[i] | B[i]
            b: u32 = a | A[i]
            C[i] = b | B[i]

    def _or_offsets(fn):
        ops = _sched(fn).func(fn.__name__).regions[0].ops
        return sorted({round(o.z, 3) for o in ops if o.kind == "ori"})

    assert len(_or_offsets(disjoint)) == 1
    spaced = _or_offsets(overlapping)
    assert len(spaced) == 3
    assert all(
        b - a == pytest.approx(comb_step_ns("or"), abs=1e-3)
        for a, b in zip(spaced, spaced[1:])
    )

    rng = np.random.default_rng(2)
    A = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    B = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    lo, hi = A & 0xFF, B & 0xFF
    C = np.zeros(64, np.uint32)
    _to_rtl(disjoint).cosim(A.copy(), B.copy(), C)
    assert np.array_equal(C, lo | (hi << 8) | (lo << 16) | (hi << 24))
    C = np.zeros(64, np.uint32)
    _to_rtl(overlapping).cosim(A.copy(), B.copy(), C)
    assert np.array_equal(C, A | B)


def test_literal_shift_is_wiring():
    # A shift by a literal renames bits: `comb` folds it into an extract /
    # concat, so it costs no logic. The device's shift row prices a barrel
    # shifter, which is what a runtime amount pays for. The two kernels have the
    # same operation count and memory traffic and differ only in the shift
    # amount, so any gap is the shifter's delay alone.
    @kernel
    def literal(A: u32[16], C: u32[16], B: u32[16]):
        for i in range(16):
            a: u32 = (A[i] << 3) ^ C[i]
            b: u32 = (a << 3) ^ C[i]
            c: u32 = (b << 3) ^ C[i]
            B[i] = (c << 3) ^ C[i]

    @kernel
    def runtime(A: u32[16], C: u32[16], B: u32[16]):
        for i in range(16):
            a: u32 = (A[i] << C[i]) ^ C[i]
            b: u32 = (a << C[i]) ^ C[i]
            c: u32 = (b << C[i]) ^ C[i]
            B[i] = (c << C[i]) ^ C[i]

    assert len(_sched(literal).func("literal").regions[0].ops) == len(
        _sched(runtime).func("runtime").regions[0].ops
    )
    # The premise, read off the device rather than assumed: a runtime shift
    # costs a barrel shifter's step and forces cuts a literal one does not.
    assert comb_step_ns("shl") > 0
    assert _latency(literal) < _latency(runtime)

    rng = np.random.default_rng(4)
    A = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    C = (rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32),)[0]
    want = A
    for _ in range(4):
        want = ((want << np.uint32(3)) ^ C).astype(np.uint32)
    B = np.zeros(16, np.uint32)
    _to_rtl(literal).cosim(A.copy(), C.copy(), B)
    assert np.array_equal(B, want)


def _shared_units(mod):
    """How many operations each shared unit carries, across every region."""
    return sorted(
        u.bound_ops for r in mod.microarch.top.regions for u in r.shared_units
    )


def _mux_fanins(mod):
    """Every multiplexer's fan-in, one entry per mux."""
    return sorted(
        f
        for r in mod.microarch.top.regions
        for m in r.muxes
        for f in [m.fanin] * m.count
    )


def test_shared_reduction_reinjects_its_identity():
    # A loop-carried accumulator may share its adder with ordinary ops: the
    # identity it re-injects on the first iteration rides an arm of that unit's
    # input mux (`Mux::Phase`), since a time-shared port has no cycle of its own
    # to time a 2:1 mux against. The identity is non-zero, so an arm that never
    # fires, or fires on the wrong iteration, shows up in the sum.
    @kernel
    def fred(A: f32[64], B: f32[64], out: f32[1]):
        s: f32 = 1.5
        for i in range(0, 64, 4):
            s = s + A[i]
            B[i] = A[i] + A[i + 1]
            B[i + 1] = A[i + 1] + A[i + 2]
            B[i + 2] = A[i + 2] + A[i + 3]
            B[i + 3] = A[i + 3] + A[i]
        out[0] = s

    A = np.random.default_rng(5).standard_normal(64).astype(np.float32)
    want = np.float32(1.5)
    for i in range(0, 64, 4):
        want = np.float32(want + A[i])

    shared = _to_rtl(fred, binding="greedy-share")
    assert _shared_units(shared), "the reduction's adder was not shared at all"
    # The recurrence port carries one arm per bound op plus the identity's.
    assert max(_mux_fanins(shared)) > max(u for u in _shared_units(shared))

    for mod in (_to_rtl(fred), shared):
        B, out = np.zeros(64, np.float32), np.zeros(1, np.float32)
        mod.cosim(A.copy(), B, out)
        assert abs(out[0] - want) < 1e-3
        assert np.allclose(B[0::4], A[0::4] + A[1::4], rtol=1e-5)


def test_shared_mux_delay_accumulates_along_a_chain():
    # A mux's delay does not stop at the unit it feeds: two shared units on one
    # combinational chain both shift what they drive, so the binder charges the
    # whole cone rather than one fold at a time. Charging one fold at a time
    # admits plans the period check then refuses. What holds at every clock is
    # that greedy sharing produces a datapath, the same one trivial binding
    # computes.
    @kernel
    def chain(A: i32[64], B: i32[64]):
        for i in range(0, 64, 4):
            a0: i32 = A[i] + A[i + 1]
            a1: i32 = A[i + 2] + A[i + 3]
            b0: i32 = a0 + a1
            b1: i32 = a0 - a1
            B[i] = b0 + b1
            B[i + 1] = b0 - b1
            B[i + 2] = a0 + b0
            B[i + 3] = a1 + b1

    A = np.random.default_rng(3).integers(-50, 50, size=64).astype(np.int32)
    ref = np.zeros(64, np.int32)
    for i in range(0, 64, 4):
        a0, a1 = A[i] + A[i + 1], A[i + 2] + A[i + 3]
        b0, b1 = a0 + a1, a0 - a1
        ref[i], ref[i + 1] = b0 + b1, b0 - b1
        ref[i + 2], ref[i + 3] = a0 + b0, a1 + b1

    # The schedule moves with the clock, so the fold count is no invariant. What
    # is invariant is that sharing never refuses a clock the trivial binding
    # (one unit per operation) accepts.
    for freq in (200, 300, 400, 450, 500):
        try:
            _to_rtl(chain, binding="trivial", freq_mhz=freq).mlir
        except RuntimeError:
            continue  # a clock this kernel cannot hold however it is bound
        _to_rtl(chain, binding="greedy-share", freq_mhz=freq).mlir
    assert _shared_units(_to_rtl(chain, binding="greedy-share"))
    for mod in (_to_rtl(chain), _to_rtl(chain, binding="greedy-share")):
        B = np.zeros(64, np.int32)
        mod.cosim(A.copy(), B)
        assert np.array_equal(B, ref)


@pytest.mark.skipif(not has_exact_scheduler(), reason="build has no OR-Tools")
def test_planned_allocation_is_never_looser_than_greedy():
    # The exact scheduler decides how many copies of each operator a region
    # builds and 'planned' builds exactly that. Its search starts from, and falls
    # back on, the tightest count its own schedule admits, which is what the
    # area-agnostic greedy binder would fold that schedule to. So the decided
    # allocation ties with greedy sharing and never loses to it.
    @kernel
    def chain(A: f32[1], B: f32[1], C: f32[1], D: f32[1], o: f32[1]):
        o[0] = A[0] * B[0] * C[0] * D[0]

    args = [np.array([v], np.float32) for v in (7, 6, 5, 2)]
    ref = np.array([7 * 6 * 5 * 2], np.float32)
    greedy = _to_rtl(chain, binding="greedy-share")
    planned = _to_rtl(chain, binding="planned").set_scheduler_opt(scheduler="exact")
    assert planned.mlir.count("hw.instance") <= greedy.mlir.count("hw.instance")
    o = np.zeros(1, np.float32)
    planned.cosim(*args, o)
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
    fcmp = next(o for o in default_device.operators if o.optype is OperatorType.CMP)
    # The predicate rides the module name, so the extern is the symbol plus it.
    assert f"hw.module.extern @{fcmp.symbol}_ogt" in mod.mlir
    assert "comb.mux" in mod.mlir
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

        res = _to_rtl(red).set_scheduler_opt(accumulators=n).schedule()
        return res.cyclic()[0].interval

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

        res = _to_rtl(red).set_scheduler_opt(accumulators=n).schedule()
        return res.cyclic()[0].interval

    assert mixed_ii(0) == FADD
    assert mixed_ii(FADD) == 1


# Integer reductions rebalance unconditionally (integer arithmetic is exactly
# associative mod 2^w), cutting an unrolled chain's recurrence to one operator.
def test_reassociate_int_reduction_recurrence():
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
    region = s.export("rtl").schedule().cyclic()[0]
    # The five terms are the four unrolled multiplies plus the accumulator,
    # folded in last, so the carried path is one multiply rather than the four a
    # chain would leave on it. The recurrence bounds the II, so a tree fits in a
    # span a chain could not. Measured against the device rather than pinned.
    assert region.interval * PERIOD_NS < REG_NS + 4 * comb_step_ns("mul")


# Bit growth types an expression at its natural width and applies the declared
# type as a trailing truncation, so every operator in between is built at a
# width nothing reads. `narrow-demanded-bits` sinks that truncation onto the
# leaves, where it collapses into the extends bit growth put there.
def test_narrow_demanded_bits_widths():
    from allo.backend.base import run_pipeline
    from allo.backend.rtl.schedule import RTL_PREPARE_PIPELINE
    from allo.compiler.mlir_codegen import compile as compile_kernel

    i48 = APInt(48, signed=True)

    @kernel
    def mac(b: i32, c: i32, d: i32) -> i48:
        a: i48 = b * c + d
        return a

    module = compile_kernel(mac)
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    # The natural widths: a 64-bit product feeding a 65-bit add, then truncated
    # to the 48 bits the declaration asked for.
    before = str(module)
    assert "i64" in before and "i65" in before and "arith.trunci" in before

    run_pipeline(module, "builtin.module(func.func(narrow-demanded-bits))")
    after = str(module)
    assert "arith.muli" in after and "arith.addi" in after
    assert "i64" not in after and "i65" not in after
    # Nothing is discarded any more, so the truncation is gone rather than moved.
    assert "arith.trunci" not in after


# The narrowing is bit-exact: an i48 accumulator wraps identically whether its
# adder is 48 or 65 bits wide. The inputs are sized so the exact sum overflows.
def test_narrow_demanded_bits_wraps_exactly():
    i48 = APInt(48, signed=True)

    @kernel
    def dot(x: i32[8], y: i32[8]) -> i48:
        acc: i48 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    rng = np.random.default_rng(0)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert abs(exact) > 2**47, "inputs must overflow i48 for the wrap to matter"
    wrapped = ((exact + 2**47) % 2**48) - 2**47

    r = _to_rtl(dot).cosim(x, y)
    assert int(r.result) == wrapped


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
