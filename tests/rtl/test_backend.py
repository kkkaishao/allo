# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Allo's RTL backend."""

import math
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, bf16, u8, index, Stream
from allo.schedule import Schedule
from _common import (
    _sched,
    _to_rtl,
    _latency,
    _iis,
    FADD,
    FDIV,
    IMUL,
    MEM,
    MEM_REDUCE_II,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
B16 = (np.arange(16, dtype=np.int32) * 5 + 3) & 0xFF


def test_elementwise_and_addressing():
    """Elementwise kernels over the basic address shapes: direct, neighbour
    offset, constant stride, a scalar argument, and a func-scope literal."""

    @kernel
    def vand(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & B[i]

    out = np.zeros(16, np.int32)
    r = _to_rtl(vand).cosim(A16, B16, out)
    assert np.array_equal(out, A16 & B16)
    assert r.cycles > 0

    @kernel
    def shift(A: i32[16], out: i32[16]):
        for i in range(15):
            out[i] = A[i] & A[i + 1]

    out = np.zeros(16, np.int32)
    _to_rtl(shift).cosim(A16, out)
    assert np.array_equal(out[:15], A16[:15] & A16[1:16])

    # A[2*i]: the address linearizes to iv*2 -- a multiply by the constant stride.
    @kernel
    def stride2(A: i32[16], out: i32[8]):
        for i in range(8):
            out[i] = A[2 * i] & A[2 * i]

    out = np.zeros(8, np.int32)
    _to_rtl(stride2).cosim(A16, out)
    assert np.array_equal(out, A16[0:16:2])

    @kernel
    def scaled(A: i32[16], out: i32[16], s: i32):
        for i in range(16):
            out[i] = A[i] & s

    out = np.zeros(16, np.int32)
    _to_rtl(scaled).cosim(A16, out, np.int32(0x0F))
    assert np.array_equal(out, A16 & 0x0F)

    # A func-scope literal tied into the compute.
    @kernel
    def constd(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & 5

    out = np.zeros(16, np.int32)
    _to_rtl(constd).cosim(A16, out)
    assert np.array_equal(out, A16 & 5)


def test_csim_golden_matches_reference():
    """csim delegates to the CPU/LLVM-JIT path; it is the golden cosim compares to."""

    @kernel
    def vand(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] & B[i]

    golden = np.zeros(16, np.int32)
    _to_rtl(vand).csim(A16, B16, golden)
    assert np.array_equal(golden, A16 & B16)


def test_array_return_rejected_at_emission():
    """An array return has no meaning at a hardware port, so emission rejects it;
    the kernel still schedules."""

    @kernel
    def arr(A: i32[8]) -> i32[8]:
        B: i32[8] = 0
        for i in range(8):
            B[i] = A[i] + 1
        return B

    mod = _to_rtl(arr)
    assert mod.schedule().cyclic()[0].ii == 1
    with pytest.raises(TypeError, match="does not support returning arrays"):
        mod.compile()


def test_scalar_recurrences():
    """Recurrences through memory and through a register: an RMW cell, a reduction
    handed to an epilogue store, and a running iter_arg."""

    # acc[0] &= A[i] recurs through memory: one array, both read and written.
    @kernel
    def racc(A: i32[16], acc: i32[1]):
        for i in range(16):
            acc[0] = acc[0] & A[i]

    acc = np.array([-1], np.int32)  # all bits set
    _to_rtl(racc).cosim(A16, acc)
    ref = -1
    for x in A16:
        ref &= int(x)
    assert int(acc[0]) == ref

    @kernel
    def reduce_then_store(A: i32[16], out: i32[1]):
        acc: i32 = 0
        for i in range(16):
            acc = acc | A[i]
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(reduce_then_store).cosim(A16, out)
    assert out[0] == np.bitwise_or.reduce(A16)

    # A loop-carried scalar iter_arg stored each iteration: a datapath accumulator
    # register, no memory recurrence.
    @kernel
    def orred(A: i32[16], out: i32[16]):
        acc: i32 = 0
        for i in range(16):
            acc = acc | A[i]
            out[i] = acc

    out = np.zeros(16, np.int32)
    _to_rtl(orred).cosim(A16, out)
    assert np.array_equal(out, np.bitwise_or.accumulate(A16))


A8 = (np.arange(8, dtype=np.int32) * 7 + 13) & 0xFF


def test_constant_rom_cosim():
    """A constant-initialized local array lowers to a read-only ROM -- a
    module-level `memref.global` referenced by `memref.get_global` -- realized as
    an indexed constant table rather than a writable on-chip buffer. Covers a byte
    table read in a loop under a data-dependent index, and a wider (i32) table of
    non-power-of-two length read by a scalar index."""

    TBL = [10, 20, 30, 40, 50, 60, 70, 80]

    @kernel
    def table_lookup(A: u8[16], out: u8[16]):
        tbl: u8[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(16):
            idx: index = A[i] % 8
            out[i] = tbl[idx]

    m = _to_rtl(table_lookup)
    assert "hw.aggregate_constant" in m.mlir  # a ROM, not a writable hlmem
    A = np.arange(16, dtype=np.uint8) * 5 + 1
    out = np.zeros(16, np.uint8)
    m.cosim(A, out)
    assert np.array_equal(out, np.array(TBL, np.uint8)[A % 8])

    SQ = [i * i for i in range(12)]

    @kernel
    def square_table(x: i32, out: i32[1]):
        sq: i32[12] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]
        idx: index = x
        out[0] = sq[idx]

    m = _to_rtl(square_table)
    out = np.zeros(1, np.int32)
    for x in (0, 3, 7, 11):
        m.cosim(np.int32(x), out)
        assert out[0] == SQ[x]


def test_banked_internal_buffer():
    """A partitioned internal buffer splits into per-bank on-chip memories. A
    statically-resolvable index routes to its bank directly; an index whose bank
    varies at runtime gets a crossbar (read every bank + mux, write-enable demux)."""

    @kernel
    def ibuf(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = A[i] & 5
        for i in range(8):
            out[i] = buf[i] & A[i]

    out = np.zeros(8, np.int32)
    _to_rtl(ibuf).cosim(A8, out)
    assert np.array_equal(out, (A8 & 5) & A8)

    # Cyclic-2 accessed at even/odd indices -> two statically-banked halves. Each
    # bank runs a distinct op (+1 vs +100), so a swapped route or a fall-back to
    # one memory corrupts the golden.
    @kernel
    def bank(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(8):
            buf[2 * i] = A[2 * i] + 1
            buf[2 * i + 1] = A[2 * i + 1] + 100
        for i in range(8):
            out[2 * i] = buf[2 * i] & 255
            out[2 * i + 1] = buf[2 * i + 1] & 255

    s = bank.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2  # two per-bank memories, not one

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    ref = A16.copy()
    ref[0::2] = (A16[0::2] + 1) & 255
    ref[1::2] = (A16[1::2] + 100) & 255
    assert np.array_equal(out, ref)

    # buf[i] under cyclic-2 is NOT statically banked (the bank alternates with the
    # loop counter), so the emitter builds the crossbar. Correctness rides on the
    # bank (i & 1) / offset (i >> 1) split and on aligning the bank select with
    # the 1-cycle read latency.
    @kernel
    def dbank(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1
        for i in range(16):
            out[i] = buf[i] & 255

    s = dbank.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2  # two banks, crossbar-addressed

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 + 1) & 255)


def test_banked_boundary_argument():
    """A partitioned argument array becomes one boundary interface per bank, and
    the cosim splits the numpy argument into cyclic bank slices (joining on
    writeback). A runtime-varying bank crossbars over those interfaces."""

    @kernel
    def ext(A: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = A[2 * i] + 1
            out[2 * i + 1] = A[2 * i + 1] + 100

    s = ext.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    s.partition("out", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # The boundary carries per-port bank info (both banks reached).
    iface = mod.interfaces[mod.top]
    assert {r["bank"] for acc in iface["reads"] for r in acc} == {0, 1}

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    ref = A16.copy()
    ref[0::2] = A16[0::2] + 1
    ref[1::2] = A16[1::2] + 100
    assert np.array_equal(out, ref)

    @kernel
    def dext(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] + 1

    s = dext.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    s.partition("out", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # Each argument presents two bank interfaces (_b0/_b1), not one flat port.
    iface = mod.interfaces[mod.top]
    rbases = {r["base"] for acc in iface["reads"] for r in acc}
    wbases = {w["base"] for acc in iface["writes"] for w in acc}
    assert {"A_rd_b0", "A_rd_b1"} <= rbases
    assert {"out_wr_b0", "out_wr_b1"} <= wbases

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 + 1)


def test_nested_banked_static_split():
    # A 2D nest accessing a cyclic-partitioned buffer on its inner (partitioned)
    # dim. flatten-perfect-loops must NOT coalesce the inner loop: coalescing
    # delinearizes j and would defeat static bank resolution, falling back to the
    # runtime crossbar. With the skip, buf banks *statically* (two per-bank
    # memories, no _b<k> crossbar); the result is correct either way.
    @kernel
    def nb(A: i32[4, 8], out: i32[4, 8]):
        buf: i32[4, 8]
        for i in range(4):
            for j in range(4):
                buf[i, 2 * j] = A[i, 2 * j] + 1
                buf[i, 2 * j + 1] = A[i, 2 * j + 1] + 100
        for i in range(4):
            for j in range(4):
                out[i, 2 * j] = buf[i, 2 * j] & 255
                out[i, 2 * j + 1] = buf[i, 2 * j + 1] & 255

    s = nb.schedule()
    s.partition("buf", dim=2, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # buf split into two per-bank memories statically (not the _b<k> crossbar).
    assert mod.mlir.count("= seq.hlmem") == 2
    assert "@buf_b" not in mod.mlir

    A = ((np.arange(32, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 8)
    out = np.zeros((4, 8), np.int32)
    mod.cosim(A, out)
    ref = A.copy()
    ref[:, 0::2] = (A[:, 0::2] + 1) & 255
    ref[:, 1::2] = (A[:, 1::2] + 100) & 255
    assert np.array_equal(out, ref)


def _f32(*shape):
    return np.random.default_rng(0).random(shape, dtype=np.float32)


def test_loop_structures():
    """The loop shapes around a datapath: a 2-D nest, sibling loops chained
    through an array, a reduction nest, and a loop-free straight line."""

    @kernel
    def nest(A: i32[4, 4], out: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                out[i, j] = A[i, j] & 5

    A2 = ((np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 4)
    out = np.zeros((4, 4), np.int32)
    _to_rtl(nest).cosim(A2, out)
    assert np.array_equal(out, A2 & 5)

    @kernel
    def two(A: i32[8], B: i32[8], C: i32[8]):
        for i in range(8):
            B[i] = A[i] & 5
        for i in range(8):
            C[i] = B[i] & A[i]

    A8 = (np.arange(8, dtype=np.int32) * 7 + 13) & 0xFF
    B = np.zeros(8, np.int32)
    C = np.zeros(8, np.int32)
    _to_rtl(two).cosim(A8, B, C)
    assert np.array_equal(B, A8 & 5)
    assert np.array_equal(C, (A8 & 5) & A8)

    # A container `for i` with two children per outer iteration -- an inner
    # store-less reduction and a store of its result -- exercising multi-child
    # container sequencing, the cross-child survivor, and the retriggered
    # accumulator (its init re-injected each row).
    @kernel
    def rowor(A: i32[4, 4], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for j in range(4):
                acc = acc | A[i, j]
            out[i] = acc

    out = np.zeros(4, np.int32)
    _to_rtl(rowor).cosim(A2, out)
    assert np.array_equal(out, np.bitwise_or.reduce(A2, axis=1))

    # No loop -> one acyclic (dcp.sequential) region; each array touched once.
    @kernel
    def strl(A: i32[1], B: i32[1], C: i32[1], D: i32[1], o1: i32[1], o2: i32[1]):
        o1[0] = A[0] & B[0]
        o2[0] = C[0] & D[0]

    a, b, c, d = (np.array([v], np.int32) for v in (0xF0, 0xFF, 0x3C, 0x0F))
    o1, o2 = np.zeros(1, np.int32), np.zeros(1, np.int32)
    _to_rtl(strl).cosim(a, b, c, d, o1, o2)
    assert o1[0] == (0xF0 & 0xFF) and o2[0] == (0x3C & 0x0F)


def test_float_and_int_arithmetic():
    """Reductions and matmuls over the float and integer datapaths: the float ops
    are multi-cycle IP instances, the int add is combinational."""

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


def test_multi_store_deepest_drains():
    """Two stores in one region at different pipeline stages: `done` must wait for
    the DEEPEST store, or the deeper store's tail iterations are dropped."""

    # A shallow B[i]=A[i]+1 (one fadd) and a deeper C[i]=A[i]*A[i]+2 (fmul then
    # fadd). Counting store write-enables over-counts when the two retire in the
    # same cycle from different in-flight iterations.
    @kernel
    def twostore(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            B[i] = A[i] + 1.0
            C[i] = A[i] * A[i] + 2.0

    A = _f32(8)
    B = np.zeros(8, np.float32)
    C = np.zeros(8, np.float32)
    _to_rtl(twostore).cosim(A, B, C)
    assert np.allclose(B, A + 1.0, rtol=1e-4, atol=1e-5)
    assert np.allclose(C, A * A + 2.0, rtol=1e-4, atol=1e-5)


def test_acyclic_scalar_survivors():
    """A straight-line (dcp.sequential) region can yield a value to a sibling, not
    only retire stores: each result is captured into its own survivor register and
    the region's done drains on the latest one, so the consumer reads them valid."""

    # A top-level prologue loads a scalar and hands it to a sibling loop.
    @kernel
    def prol(A: i32[4], out: i32[4]):
        x: i32 = A[0]
        for i in range(4):
            out[i] = x + A[i]

    A = np.arange(4, dtype=np.int32) * 7 + 3
    out = np.zeros(4, np.int32)
    _to_rtl(prol).cosim(A, out)
    assert np.array_equal(out, A[0] + A)

    # An imperfect nest whose prologue becomes an acyclic child of the outer
    # container, re-run each outer iteration and read against the freshly
    # advanced outer counter.
    @kernel
    def imperfect(A: i32[4], B: i32[4, 4], out: i32[4, 4]):
        for i in range(4):
            x: i32 = A[i]
            for j in range(4):
                out[i, j] = B[i, j] + x

    B = (np.arange(16, dtype=np.int32) * 3).reshape(4, 4)
    out = np.zeros((4, 4), np.int32)
    _to_rtl(imperfect).cosim(A, B, out)
    assert np.array_equal(out, B + A[:, None])

    # A prologue that both inits an accumulator and loads an invariant fuses into
    # ONE multi-result acyclic region yielding (0, A[0]); each result gets its own
    # survivor, and the constant identity is still re-injected as the reduction
    # init even though it now arrives as a region result.
    @kernel
    def prol_reduce(A: i32[4], out: i32[1]):
        x: i32 = A[0]
        acc: i32 = 0
        for i in range(4):
            acc = acc + A[i] * x
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(prol_reduce).cosim(A, out)
    assert out[0] == np.sum(A * A[0])


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


# A kernel that RETURNS an array (by value) is intentionally unsupported: a
# returned buffer has no meaning at a hardware boundary. Write outputs through an
# explicit out-parameter argument instead (see e.g. `test_int_relu_if_conversion`
# below, whose `out` is a written argument). Scalar returns (`-> i32`) stay a
# first-class output port.


def _signed_f32(seed):
    return (np.random.default_rng(seed).random(16, dtype=np.float32) - 0.5) * 10


def test_compare_select_and_shift():
    """If-conversion over both datapaths: an int compare lowers to native
    comb.icmp, a float compare to an fcmp IP instance, both feeding a comb.mux.
    Shifts lower to native comb.shl / comb.shr."""

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


# --- scheduling-result validation ------------------------------------------
# Scheduler-feature cases (streams, dynamic-trip scf.for, while loops,
# allo.assume hints, pipeline directives, reduction restructuring, storage-impl,
# timing/chaining) that the polybench/machsuite workloads do not reach.


# --- Storage implementation: per-impl access latency -----------------------


def _matvec_recurrence_ii(bind=None, complete=False):
    """Schedule a memory-carried matvec accumulate (`y[i] += A[i,k]*x[k]`) with
    the accumulator `y` optionally bound to a storage impl or complete-partitioned
    (-> registers); return the inner loop II (the read->add->write recurrence)."""

    @kernel
    def mv(A: f32[8, 8], x: f32[8], y: f32[8]):
        for i in range(8):
            for k in range(8):
                y[i] += A[i, k] * x[k]

    s = mv.schedule()
    if complete:
        s.partition("y", kind=s.Complete)
    elif bind is not None:
        s.bind_storage("y", impl=bind, mem_type=s.RAM_T2P)
    res = s.export("rtl").schedule()
    return min(r.ii for r in res.cyclic())


def test_storage_impl_shifts_recurrence_ii():
    """The scheduler times a memory access by the array's storage impl, so
    bind_storage -- and a complete partition (-> registers) -- move the II of a
    memory-carried recurrence."""
    # The recurrence II is read + FADD + write. Default LUTRAM (1/1) gives
    # FADD + 2; binding the accumulator to URAM (read 2, write 1) adds one cycle.
    lutram_ii = _matvec_recurrence_ii()
    assert lutram_ii == FADD + 2
    assert _matvec_recurrence_ii(bind=Schedule.URAM) == FADD + 3
    # A complete partition scatters `y` into FFs: the read is combinational (0),
    # but the FF write still costs a cycle, so the recurrence is FADD + 1 -- one
    # below LUTRAM, not a full collapse to the bare add latency.
    reg_ii = _matvec_recurrence_ii(complete=True)
    assert reg_ii == FADD + 1
    assert reg_ii < lutram_ii


def test_residual_loops_closed_into_pipelines():
    """The scheduled IR is closed over the dcp dialect: every counted loop -- an
    imperfect-nest wrapper or a non-flattenable dynamic band -- materializes into
    a dcp.pipeline, so no raw affine.for / scf.for survives."""
    N = 8

    @kernel
    def imperfect(A: f32[N, N], x: f32[N], y: f32[N]):
        for i in range(N):
            y[i] = 0.0
            for j in range(N):
                y[i] += A[i, j] * x[j]

    mod = _to_rtl(imperfect)
    # An outer sequential wrapper (ii = body length) around the inner pipeline.
    assert any(r.is_wrapper for r in mod.schedule().funcs[0].regions)

    @kernel
    def band(A: f32[N, N], y: f32[N], n: index):
        for i in range(n):  # dynamic trip -> band cannot be flattened
            for j in range(N):
                y[i] += A[i, j]

    mod = _to_rtl(band)
    # Dynamic outer trip: the wrapper's II is still concrete (inner-derived), but
    # its trip is unknown.
    wrapper = next(r for r in mod.schedule().funcs[0].regions if r.is_wrapper)
    assert wrapper.trip is None and wrapper.ii > 0

    @kernel
    def dyn_inner(A: f32[N, N], y: f32[N], n: index):
        for i in range(N):  # static outer
            for j in range(n):  # DYNAMIC inner trip -> body length data-dependent
                y[i] += A[i, j]

    mod = _to_rtl(dyn_inner)
    # The outer wrapper's body length is data-dependent, so it carries no static
    # II (done-based sequential controller), but the loop still closes into a
    # dcp.pipeline (its static trip is known).
    wrapper = next(r for r in mod.schedule().funcs[0].regions if r.is_wrapper)
    assert wrapper.ii is None and wrapper.trip == N


def test_imperfect_reduction_nest_cosim():
    # The `imperfect` matvec above -- an imperfect nest (init prologue `y[i]=0.0` +
    # scalar-carried inner reduction) closed into a sequential-wrapper pipeline --
    # run end-to-end: pins that the residual-loop closure computes y = A @ x.
    N = 8

    @kernel
    def imperfect(A: f32[N, N], x: f32[N], y: f32[N]):
        for i in range(N):
            y[i] = 0.0
            for j in range(N):
                y[i] += A[i, j] * x[j]

    A = (np.arange(N * N, dtype=np.float32) * 0.1).reshape(N, N)
    x = np.arange(N, dtype=np.float32) * 0.1 + 1.0
    y = np.zeros(N, np.float32)
    _to_rtl(imperfect).cosim(A, x, y)
    assert np.allclose(y, A @ x, rtol=1e-3, atol=1e-3)


def test_guards_over_loops_close_into_dcp_select():
    """A guard that can neither be predicated (a loop is not speculatable) nor
    folded into a loop bound survives to the reifier, which closes it into a
    dcp.select wrapping the materialized children -- no raw if remains."""
    N, M = 64, 32

    # A data-dependent guard: the predicate is not affine in the IV.
    @kernel
    def cond_reduce(A: f32[N, M], flag: i32[M], out: f32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: f32 = 0.0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    mod = _to_rtl(cond_reduce)
    res = mod.schedule()
    assert _iis(res.func("cond_reduce").cyclic()) == [FADD]  # guarded reduction
    assert "dcp.select" in mod.dcp
    guard = next(r for r in res.funcs[0].regions if r.kind == "guard")
    assert guard.conditional and guard.container

    # An affine guard that cannot fold into a bound: a two-constraint set
    # (`i > j and i < j+4`) cannot tighten a single bound, so the reifier
    # materializes its IntegerSet predicate into an i1.
    A8 = 8

    @kernel
    def agf(x: f32[A8], out: f32[A8]):
        for i in range(A8):
            for j in range(A8):
                if i > j and i < j + 4:
                    for k in range(A8):
                        out[i] += x[j]

    mod = _to_rtl(agf)
    res = mod.schedule()
    assert "dcp.select" in mod.dcp
    # Phase A lifts the IntegerSet predicate into start-0 dcp.compute units (the
    # conjunction `andi` of two `sge` compares, predicate 5), so the guard
    # condition is a first-class Source -- no raw arith.cmpi/andi survives for the
    # emitter to re-interpret.
    assert "comb andi" in mod.dcp and mod.dcp.count("predicate = 5 : i64") >= 2
    guard = next(r for r in res.funcs[0].regions if r.kind == "guard")
    assert guard.conditional and guard.container
    assert _iis(res.cyclic()) == [MEM_REDUCE_II]  # memory-carried `out[i] +=`

    # An affine guard that does not span its enclosing loop body (a trailing store
    # follows it) keeps `for j` an imperfect wrapper rather than a flattenable
    # band; the dcp.select's counter then references the wrapper's own IV.
    @kernel
    def imp(A: f32[A8, A8], B: f32[A8, A8], out: f32[A8, A8], C: f32[A8, A8]):
        for i in range(A8):
            for j in range(A8):
                if i > j:
                    acc: f32 = 0.0
                    for k in range(A8):
                        acc += A[i, k] * B[k, j]
                    out[i, j] = acc
                C[i, j] = 1.0  # trailing store -> guard does not span the body

    mod = _to_rtl(imp)
    res = mod.schedule()
    assert "dcp.select" in mod.dcp
    # Scalar-carried reduction inside the guard -> register recurrence (II=FADD).
    assert _iis(res.cyclic()) == [FADD]
    assert any(r.kind == "guard" for r in res.funcs[0].regions)


def test_scf_if_guard_store_gated_cosim():
    # C2/B8: the data-dependent guard `if flag[j]>0: out[j]=Σ` closes into a
    # dcp.select. Pins that the predicate reaches the guarded store: out[j] is
    # written ONLY where flag[j]>0; the flag<=0 columns stay 0 (the store never
    # fires). Before the guard-region gate landed, the predicate was computed into
    # a dead survivor and out[j] was written for every j (a silent miscompile).
    N, M = 8, 4

    @kernel
    def cond_reduce(A: f32[N, M], flag: i32[M], out: f32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: f32 = 0.0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    A = (np.arange(N * M, dtype=np.float32) * 0.1).reshape(N, M)
    flag = np.array([1, 0, 1, 0], dtype=np.int32)
    out = np.zeros(M, np.float32)
    _to_rtl(cond_reduce).cosim(A, flag, out)
    golden = np.where(flag > 0, A.sum(axis=0), 0.0).astype(np.float32)
    assert np.allclose(out, golden, rtol=1e-3, atol=1e-3)
    # The guarded-false columns are untouched (0), not the reduction of an
    # ungated store (which would leak the previous column's acc into them).
    assert out[1] == 0.0 and out[3] == 0.0


def test_affine_if_guard_store_gated_cosim():
    # An affine guard `if i>j and i<j+4` over a memory-carried reduction whose
    # store lives INSIDE the guard (`out[i] += x[j]`, committed every k). The
    # perfect `i,j` band coalesces, so the two-constraint predicate becomes a raw
    # comb-arith tree over the flattened counter (a signed-divide delinearization
    # emitted natively) evaluated by evalRawArith. Pins the store fires only where
    # the predicate holds.
    N, M = 8, 8

    @kernel
    def agf(x: f32[N], out: f32[N]):
        for i in range(N):
            for j in range(N):
                if i > j and i < j + 4:
                    for k in range(M):
                        out[i] += x[j]

    x = np.arange(N, dtype=np.float32) * 0.1 + 1.0
    out = np.zeros(N, np.float32)
    golden = np.zeros(N, np.float32)
    for i in range(N):
        for j in range(N):
            if i > j and i < j + 4:
                golden[i] += M * x[j]
    _to_rtl(agf).cosim(x, out)
    assert np.allclose(out, golden, rtol=1e-3, atol=1e-3)


def test_imperfect_wrapper_guard_cosim():
    # An imperfect wrapper carrying two stores: a guarded reduction store
    # (`out[i,j]=Σ`, INSIDE the guard `i>j`) and an UNGUARDED trailing store
    # (`C[i,j]=1.0`, a sibling outside the guard). Pins both -- the guard gates
    # exactly its own store: C is written for every (i,j), out only for i>j.
    N, M = 8, 8

    @kernel
    def imp(A: f32[N, M], B: f32[M, N], out: f32[N, N], C: f32[N, N]):
        for i in range(N):
            for j in range(N):
                if i > j:
                    acc: f32 = 0.0
                    for k in range(M):
                        acc += A[i, k] * B[k, j]
                    out[i, j] = acc
                C[i, j] = 1.0

    A = (np.arange(N * M, dtype=np.float32) * 0.05).reshape(N, M)
    B = (np.arange(M * N, dtype=np.float32) * 0.03).reshape(M, N)
    out = np.zeros((N, N), np.float32)
    C = np.zeros((N, N), np.float32)
    _to_rtl(imp).cosim(A, B, out, C)
    assert np.allclose(out, np.tril(A @ B, -1), rtol=1e-2, atol=1e-2)
    assert np.allclose(C, np.ones((N, N), np.float32))


def test_result_mux_select():
    # A dcp.select with a non-empty else / yielded results: both arms run
    # mutually-exclusively under the predicate, and a yielded value is muxed
    # `cond ? then : else`. Covers a dual guard (a store loop in each arm) and a
    # result-mux (a guarded reduction yielding an accumulator).
    N = 16

    # Dual guard, affine predicate: the taken arm's store loop fires, the other's
    # never issues.
    @kernel
    def dual_affine(a: i32[4, N], b: i32[4, N], out: i32[4, N]):
        for g in range(4):
            if g < 2:
                for i in range(N):
                    out[g, i] = a[g, i] + 1
            else:
                for i in range(N):
                    out[g, i] = b[g, i] * 2

    mod = _to_rtl(dual_affine)
    assert "dcp.select" in mod.dcp
    a = np.arange(4 * N, dtype=np.int32).reshape(4, N)
    b = a + 1000
    out = np.zeros((4, N), np.int32)
    mod.cosim(a.copy(), b.copy(), out)
    assert np.array_equal(out, np.where(np.arange(4).reshape(4, 1) < 2, a + 1, b * 2))

    # Dual guard, data-dependent predicate (a ping-pong `if sel[g]==0: ... else`):
    # the predicate reads memory, lifting to a settled-survivor dcp.compute.
    @kernel
    def dual_ddep(sel: i32[4], a: i32[4, N], b: i32[4, N], out: i32[4, N]):
        for g in range(4):
            if sel[g] == 0:
                for i in range(N):
                    out[g, i] = a[g, i] + 1
            else:
                for i in range(N):
                    out[g, i] = b[g, i] * 2

    sel = np.array([0, 1, 0, 1], dtype=np.int32)
    out = np.zeros((4, N), np.int32)
    _to_rtl(dual_ddep).cosim(sel, a.copy(), b.copy(), out)
    assert np.array_equal(out, np.where(sel.reshape(4, 1) == 0, a + 1, b * 2))

    # Result-mux: the guard wraps a reduction loop and yields the accumulator; the
    # empty else passes the initial value through -> `cond ? sum : 0`.
    @kernel
    def rmux(a: i32[4, N], out: i32[4]):
        for g in range(4):
            acc: i32 = 0
            if g < 2:
                for i in range(N):
                    acc += a[g, i]
            out[g] = acc

    mod = _to_rtl(rmux)
    assert "dcp.select" in mod.dcp
    out = np.zeros(4, np.int32)
    mod.cosim(a.copy(), out)
    assert np.array_equal(out, np.where(np.arange(4) < 2, a.sum(1), 0).astype(np.int32))


# --- Streams (dataflow through FIFOs) --------------------------------------


def test_stream_li_shell():
    """A single latency-insensitive stream process: its schedule shape, then
    cosim determinism at full rate and under stall for a combinational and for a
    multi-cycle IP datapath."""

    @kernel
    def prod(srm: Stream[i32]):
        for i in range(10):
            srm.put(i)

    @kernel
    def cons(srm: Stream[i32], out: i32[1]):
        acc: i32 = 0
        for i in range(10):
            acc += srm.get()
        out[0] = acc

    @kernel
    def top(out: i32[1]):
        srm: Stream[i32]
        prod(srm)
        cons(srm, out)

    res = _sched(top)
    loop = res.func("cons").cyclic()[0]
    assert loop.ii == 1
    assert loop.op("stream.get").t <= loop.op("addi").t
    # The epilogue store lands in its own acyclic region.
    assert any(
        o.kind == "store"
        for r in res.func("cons").regions
        if r.kind == "acyclic"
        for o in r.ops
    )

    # One input stream, one output stream, counted loop, combinational datapath;
    # cocotb drives the FIFO {data,valid,ready} ports directly. KPN determinism:
    # the shell bubbles on an empty input and freezes on a full output, so it
    # never loses or duplicates a token and the result is stall-independent.
    @kernel
    def stage(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(16):
            y_out.put(x_in.get() + 7)

    rtl = _to_rtl(stage)
    x = np.arange(16, dtype=np.int32) * 5 - 3
    exp = x + 7
    for gap in (0.0, 0.5, 0.8):
        y = np.zeros(16, dtype=np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, exp), f"gap={gap}: {list(y)} != {list(exp)}"

    # The same shell with a multi-cycle IP datapath (fmul_l4 -> fadd_l7, an
    # 11-deep pipeline between get and put). The clock-enable stall contract
    # (`ce`) freezes the IP pipeline in lockstep with the shell's shift chains; a
    # free-running IP would keep clocking under back-pressure and desync.
    @kernel
    def fstage(x_in: Stream[f32], y_out: Stream[f32]):
        for i in range(16):
            y_out.put(x_in.get() * 2.0 + 1.0)

    frtl = _to_rtl(fstage)
    fx = (np.arange(16, dtype=np.float32) * 0.5 - 3.0).astype(np.float32)
    fexp = fx * 2.0 + 1.0
    for gap in (0.0, 0.5, 0.8):
        fy = np.zeros(16, dtype=np.float32)
        frtl.cosim(fx, fy, stall_prob=gap)
        assert np.allclose(fy, fexp), f"gap={gap}: {list(fy)} != {list(fexp)}"


def test_stream_ii_gt1_with_memory_read_producer():
    """A slow (II>1) f32-accumulate stream consumer draining a memory-read-fed
    producer"""

    def build(K):
        @kernel
        def top(A: f32[K], out: f32[1]):
            fifo: Stream[f32]

            @kernel(mapping=[2])
            def pe(A: f32[K], out: f32[1], fifo: Stream[f32]):
                p = allo.get_wid(0)
                if p == 0:
                    for k in range(K):  # memory-read-fed put (II=1 producer)
                        fifo.put(A[k])
                else:
                    c: f32 = 0.0
                    for k in range(K):  # recurrence -> II == FADD (slow drain)
                        c += fifo.get()
                    out[0] = c

            pe(A, out, fifo)

        return top

    # The consumer's inner loop is recurrence-bound: the shell runs the modulo
    # (II>1) regime, not the II==1 fast path.
    iis = [r.ii for f in _sched(build(8)).funcs for r in f.regions if r.ii is not None]
    assert max(iis) == FADD and FADD > 1

    for K in (8, 16):
        A = (2.0 ** np.arange(K)).astype(np.float32)  # 1, 2, 4, ... 2**(K-1)
        exp = float(A.sum())  # == 2**K - 1
        out = np.zeros(1, dtype=np.float32)
        _to_rtl(build(K)).cosim(A, out)
        assert abs(out[0] - exp) < 0.5, f"K={K}: {out[0]} != {exp} (dropped a token)"


def test_sequential_two_kernel_shared_array():
    """Two plain sub-kernels chained through a shared boundary array: the parent
    schedules child2 as a fixed-latency node after child1, so the composed
    latency is the sum of the child latencies -- reported and actual."""

    @kernel
    def sc1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def sc2(B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = B[i] * 2

    @kernel
    def seq_top(A: i32[16], B: i32[16], out: i32[16]):
        sc1(A, B)
        sc2(B, out)

    l1, l2 = _latency(sc1), _latency(sc2)
    assert l1 is not None and l2 is not None
    # The container reports its last child's completion (max over calls of
    # start + latency), not the straight-line region depth -- which counts only
    # to a call's start and so undercounts a call node by its own latency.
    assert _latency(seq_top) == l1 + l2

    B = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = _to_rtl(seq_top).cosim(A16, B, out)
    assert np.array_equal(out, (A16 + 1) * 2)  # out = child2(child1(A))
    assert r.cycles == l1 + l2  # serial: the children do not overlap


def test_sequential_internal_buffer_shared():
    """Two plain sub-kernels chained through a container-LOCAL buffer: the
    buffer lowers to an on-chip `seq.hlmem` rather than a top port, serialized by
    the RAW dependence that orders the two calls."""

    @kernel
    def sib_prod(A: i32[16], tmp: i32[16]):
        for i in range(16):
            tmp[i] = A[i] * 3

    @kernel
    def sib_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] - 7

    @kernel
    def sib_top(A: i32[16], out: i32[16]):
        tmp: i32[16]  # container-local buffer -> on-chip hlmem, not a top port
        sib_prod(A, tmp)
        sib_cons(tmp, out)

    mod = _to_rtl(sib_top)
    assert "seq.hlmem" in mod.mlir  # the internal buffer, on-chip in the top
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)  # tmp is not a top port, so cosim drives only A / out
    assert np.array_equal(out, A * 3 - 7)


def test_loop_over_calls():
    """A container whose body is a LOOP over a sub-kernel call: one child
    instance is instantiated once and fired N times, a counter driving its index
    and each invocation advancing on the child's `done`."""

    @kernel
    def lc_step(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def lc_top(A: i32[16], B: i32[16]):
        for i in range(16):
            lc_step(A, B, i)  # invoke the sub-kernel 16 times

    mod = _to_rtl(lc_top)
    # R2: the loop-over-calls container lowers to the leaf (its call reifies to a
    # `dcp.instance`), one child instance fired N times by the counter.
    assert "dcp.instance" in mod.dcp
    assert "loop_iv" in mod.mlir  # the loop counter driving the child's index
    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.zeros(16, np.int32)
    # One instance is live at a time, so its memory ports mirror to the
    # boundaries directly -- serial, no muxing.
    mod.cosim(A, B)
    assert np.array_equal(B, A * 2 + 1)


def test_sequential_independent_kernels():
    """Two plain sub-kernels with no data dependence overlap: disjoint memory
    footprints mean no ordering edge, both fire at cycle 0, and the composed
    latency is the max rather than the sum."""

    @kernel
    def ic1(A: i32[16], oa: i32[16]):
        for i in range(16):
            oa[i] = A[i] + 1

    @kernel
    def ic2(B: i32[16], ob: i32[16]):
        for i in range(16):
            ob[i] = B[i] * 2

    @kernel
    def indep_top(A: i32[16], B: i32[16], oa: i32[16], ob: i32[16]):
        ic1(A, oa)
        ic2(B, ob)

    l1 = _latency(ic1)
    l2 = _latency(ic2)
    oa = np.zeros(16, np.int32)
    ob = np.zeros(16, np.int32)
    r = _to_rtl(indep_top).cosim(A16, B16, oa, ob)
    assert np.array_equal(oa, A16 + 1)
    assert np.array_equal(ob, B16 * 2)
    assert r.cycles == max(l1, l2)  # data-independent: the children overlap


def test_concurrent_shared_array_access():
    """Sub-kernels sharing one array are ordered only when they may touch a
    common element: disjoint write slices and read-only sharing overlap, a real
    WAW serializes. Covers boundary arrays and a container-local buffer."""

    # Two sub-kernels WRITING one shared array. The per-argument callee footprint
    # proves they cannot collide -- cw1 writes B[0:8], cw2 writes B[8:16], so the
    # polyhedral regions of the two stores intersect to nothing and no edge is
    # added. (The conservative `summarizeOp` cannot see through a call, so it
    # marks every memref operand read+write and would serialize this.)
    @kernel
    def cw1(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i] = A[i] + 1

    @kernel
    def cw2(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i + 8] = A[i + 8] * 2

    @kernel
    def cw_top(A: i32[16], B: i32[16]):
        cw1(A, B)
        cw2(A, B)

    l1 = _latency(cw1)
    l2 = _latency(cw2)
    mod = _to_rtl(cw_top)
    # Two children writing one array at the same time genuinely need two ports,
    # so each access owns a port group and there is no top-side mux to
    # time-share one; the harness services every group of an argument against
    # its one backing array.
    wr = [w[0] for w in mod.interfaces["cw_top"]["writes"]]
    assert [w["base"] for w in wr] == ["B_wr", "B_wr_1"]
    assert {w["arg"] for w in wr} == {1}
    B = np.zeros(16, np.int32)
    r = mod.cosim(A16, B)
    assert np.array_equal(B, np.concatenate([A16[:8] + 1, A16[8:] * 2]))
    assert r.cycles == max(l1, l2)  # disjoint slices: the writers overlap

    # Two sub-kernels READING one shared input array: neither writes it, so there
    # is no ordering constraint at all. This needs the per-argument DIRECTION of
    # the callee footprint -- the conservative op summary gives two pure readers
    # a false WAW and serializes them.
    @kernel
    def sr1(A: i32[16], o1: i32[16]):
        for i in range(16):
            o1[i] = A[i] + 1

    @kernel
    def sr2(A: i32[16], o2: i32[16]):
        for i in range(16):
            o2[i] = A[i] * 2

    @kernel
    def sr_top(A: i32[16], o1: i32[16], o2: i32[16]):
        sr1(A, o1)
        sr2(A, o2)

    sl1 = _latency(sr1)
    sl2 = _latency(sr2)
    o1 = np.zeros(16, np.int32)
    o2 = np.zeros(16, np.int32)
    r = _to_rtl(sr_top).cosim(A16, o1, o2)
    assert np.array_equal(o1, A16 + 1)
    assert np.array_equal(o2, A16 * 2)
    assert r.cycles == max(sl1, sl2)  # read-only sharing: the readers overlap

    # The dual of the disjoint case, and the guard on its soundness: two writers
    # of the SAME elements are a real WAW, so the region intersection is
    # non-empty and the scheduler orders them. The emitter realizes that order
    # (each child at its scheduled offset), so the shared array still needs no
    # arbitration even though both children write it.
    @kernel
    def ow1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def ow2(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] * 2

    @kernel
    def ow_top(A: i32[16], B: i32[16]):
        ow1(A, B)
        ow2(A, B)  # overwrites every element ow1 wrote

    ol1 = _latency(ow1)
    ol2 = _latency(ow2)
    ob = np.zeros(16, np.int32)
    r = _to_rtl(ow_top).cosim(A16, ob)
    assert np.array_equal(ob, A16 * 2)  # the later writer wins, so they ran in order
    assert r.cycles == ol1 + ol2  # a real WAW: the writers do NOT overlap

    # A container-local buffer (one on-chip `seq.hlmem`) filled by TWO children
    # writing disjoint halves concurrently, then read by a third. Each access is
    # its own hlmem port -- CIRCT lowers the two writers to one `always_ff` with
    # a per-port enabled write -- so multiple writers need no arbitration for the
    # same reason the shared boundary needs no mux. The reader DOES conflict with
    # both (it reads what they wrote), so it is ordered after both.
    @kernel
    def ibw1(A: i32[16], t: i32[16]):
        for i in range(8):
            t[i] = A[i] + 1

    @kernel
    def ibw2(A: i32[16], t: i32[16]):
        for i in range(8):
            t[i + 8] = A[i + 8] * 2

    @kernel
    def ibrd(t: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = t[i] - 3

    @kernel
    def ibw_top(A: i32[16], out: i32[16]):
        t: i32[16]  # container-local -> on-chip hlmem, two writers + one reader
        ibw1(A, t)
        ibw2(A, t)
        ibrd(t, out)

    lw1 = _latency(ibw1)
    lw2 = _latency(ibw2)
    lrd = _latency(ibrd)
    mod = _to_rtl(ibw_top)
    assert "seq.hlmem" in mod.mlir  # the shared buffer stays on-chip
    out = np.zeros(16, np.int32)
    r = mod.cosim(A16, out)
    assert np.array_equal(out, np.concatenate([A16[:8] + 1, A16[8:] * 2]) - 3)
    # the writers overlap; the reader waits for both to drain
    assert r.cycles == max(lw1, lw2) + lrd


def test_composed_banking():
    """Banking a COMPOSED array: a partition stated once where the array lives
    reaches every callee parameter, so each child emits a port group per bank --
    for a container-local buffer and for a container argument."""

    # A partition is a property of the array, stated once on the container's
    # local `tmp`, but a sub-kernel sees it only through its own parameter -- a
    # different memref in a different function. `propagate-partition` pushes
    # `allo.part` onto every callee parameter before scheduling, so each child is
    # scheduled against its per-bank ResII and emits a port group per bank; the
    # container then materializes the banks its children already agree on -- one
    # `seq.hlmem` per bank, half-depth, bank k wired straight to hlmem k. No
    # crossbar: the child already addresses its bank's own index space.
    @kernel
    def cbi_prod(A: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = A[2 * i] + 1
            tmp[2 * i + 1] = A[2 * i + 1] + 100

    @kernel
    def cbi_cons(tmp: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = tmp[2 * i] & 255
            out[2 * i + 1] = tmp[2 * i + 1] & 255

    @kernel
    def cbi_top(A: i32[16], out: i32[16]):
        tmp: i32[16]  # container-local, partitioned -> two on-chip banks
        cbi_prod(A, tmp)
        cbi_cons(tmp, out)

    s = cbi_top.schedule()
    s.partition("tmp", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert "dcp.instance @cbi_top.cbi_prod(" in mod.dcp
    assert re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir) == [
        ("tmp_b0", "8"),
        ("tmp_b1", "8"),
    ]
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    # even lanes +1, odd lanes +100 -- a swapped bank route corrupts the golden
    assert np.array_equal(out, np.where(np.arange(16) % 2 == 0, A + 1, A + 100) & 255)

    # The boundary dual: a partitioned container ARGUMENT. The partition reaches
    # the child the same way, so the child exposes one port group per bank and
    # the container mirrors them onto the top -- each carrying its own
    # `bank`/`factor`, which is how the cosim harness knows to back them with the
    # argument's cyclic slices rather than two whole copies.
    @kernel
    def cbb(A: i32[16], o: i32[16]):
        for i in range(8):
            o[2 * i] = A[2 * i] + 1
            o[2 * i + 1] = A[2 * i + 1] + 100

    @kernel
    def cbb_top(A: i32[16], o: i32[16]):
        cbb(A, o)

    s = cbb_top.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert "dcp.instance @cbb_top.cbb(" in mod.dcp
    rd = [g[0] for g in mod.interfaces["cbb_top"]["reads"]]
    assert {(r["arg"], r["bank"], r["factor"]) for r in rd} == {(0, 0, 2), (0, 1, 2)}
    o = np.zeros(16, np.int32)
    mod.cosim(A16, o)
    assert np.array_equal(o, np.where(np.arange(16) % 2 == 0, A16 + 1, A16 + 100))


def test_nested_sequential_composition():
    """Seq-in-seq: a container whose first child is itself a container. The inner
    container is emitted first and its composed latency reaches the parent, so
    the parent places the following child after the WHOLE inner container."""

    @kernel
    def nt_leaf1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def nt_leaf2(B: i32[16], C: i32[16]):
        for i in range(16):
            C[i] = B[i] * 2

    @kernel
    def nt_mid(A: i32[16], B: i32[16], C: i32[16]):
        nt_leaf1(A, B)  # B = A + 1
        nt_leaf2(B, C)  # C = (A + 1) * 2

    @kernel
    def nt_leaf3(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + 3

    @kernel
    def nt_top(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        nt_mid(A, B, C)  # a nested CONTAINER child: C = (A + 1) * 2
        nt_leaf3(C, out)  # out = C + 3  (reads the inner container's output)

    # Composed from the inner container's own calls (max over calls of
    # start + callee-latency); the inner region depth alone would fire nt_leaf3
    # early and read C before it is written.
    lmid = _latency(nt_mid)
    l3 = _latency(nt_leaf3)
    assert lmid is not None and l3 is not None

    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    rtl = _to_rtl(nt_top)
    assert "dcp.instance @nt_top.nt_mid(" in rtl.dcp
    r = rtl.cosim(A16, B, C, out)
    assert np.array_equal(out, (A16 + 1) * 2 + 3)
    assert r.cycles == lmid + l3  # nt_leaf3 waits for the whole inner container


# --- Mixed containers (loose datapath beside sub-kernel calls) --------------


def test_mixed_container_internal_buffer_call():
    """A container that mixes its own datapath regions with a sub-kernel call
    mastering only container-local buffers: the call reifies to a scheduled node
    instantiated in the container's own module, reading and writing the shared
    on-chip buffers, serially correct against the surrounding regions."""

    @kernel
    def ib_child(B: i32[16], C: i32[16]):  # internal -> internal, no boundary
        for i in range(16):
            C[i] = B[i] + 10

    @kernel
    def ib_top(A: i32[16], out: i32[16]):
        B: i32[16]  # region 0 writes B (boundary A -> internal B)
        C: i32[16]  # the child reads B, writes C; the last region reads C
        for i in range(16):
            B[i] = A[i] + 1
        ib_child(B, C)
        for i in range(16):
            out[i] = C[i] * 2

    rtl = _to_rtl(ib_top)
    assert "dcp.instance @ib_top.ib_child" in rtl.dcp  # a scheduled call node
    assert "hw.instance" in rtl.mlir  # instantiated in the container's module
    assert "seq.hlmem" in rtl.mlir  # the shared buffers, on-chip
    A = np.arange(1, 17, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    r = rtl.cosim(A, out)
    assert r.cycles > 0
    assert np.array_equal(out, ((A + 1) + 10) * 2)


def test_mixed_container_loose_region_between_calls():
    """A loose datapath region interleaved between two calls: the first child
    masters a boundary read, the second a boundary write, and the parent's own
    region bridges their internal buffers -- each region scheduled in program
    order against the calls it depends on."""

    @kernel
    def mr1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1  # boundary read A, internal write B

    @kernel
    def mr2(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] * 2  # internal read C, boundary write out

    @kernel
    def mr_top(A: i32[16], out: i32[16]):
        B: i32[16]
        C: i32[16]
        mr1(A, B)
        for i in range(16):  # loose region between the two calls
            C[i] = B[i] + 5
        mr2(C, out)

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    r = _to_rtl(mr_top).cosim(A, out)
    assert r.cycles > 0
    assert np.array_equal(out, ((A + 1) + 5) * 2)


def test_mixed_container_scalar_result_handoff():
    """A child returns a scalar (a reduction over an internal buffer) that a
    sibling child consumes as a scalar operand: the result crosses between the
    two instances as a survivor gated by the producer's completion."""

    @kernel
    def accum(B: i32[16]) -> i32:
        s: i32 = 0
        for i in range(16):
            s += B[i]
        return s

    @kernel
    def scale(s: i32, out: i32[16]):
        for i in range(16):
            out[i] = s * 2

    @kernel
    def sh_top(A: i32[16], out: i32[16]):
        B: i32[16]
        for i in range(16):  # loose region feeds the reduction
            B[i] = A[i] + 1
        s: i32 = accum(B)  # scalar result over the internal buffer B
        scale(s, out)  # consumes the result -> boundary out

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    rtl = _to_rtl(sh_top)
    assert "dcp.instance" in rtl.dcp
    r = rtl.cosim(A, out)
    assert r.cycles > 0
    s = int((A + 1).sum())
    assert np.array_equal(out, np.full(16, s * 2, dtype=np.int32))


def test_mixed_container_scalar_survivor_across_region():
    """A scalar result that escapes past an intervening loose region: the
    producing and consuming calls land in separate regions, so the result is
    latched at the producer's completion and read back as a cross-region
    survivor rather than a same-region live value."""

    @kernel
    def xr_accum(B: i32[16]) -> i32:
        s: i32 = 0
        for i in range(16):
            s += B[i]
        return s

    @kernel
    def xr_bias(s: i32, C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + s  # scalar operand from an earlier region's result

    @kernel
    def xr_top(A: i32[16], out: i32[16]):
        B: i32[16]
        C: i32[16]
        for i in range(16):
            B[i] = A[i] + 1
        s: i32 = xr_accum(B)  # result consumed only after the C region
        for i in range(16):  # intervening region -> the calls are separate regions
            C[i] = A[i] * 2
        xr_bias(s, C, out)

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    r = _to_rtl(xr_top).cosim(A, out)
    assert r.cycles > 0
    s = int((A + 1).sum())
    assert np.array_equal(out, A * 2 + s)


def test_mixed_container_shared_boundary_serial_masters():
    """A boundary array write-mastered by two serial children: they time-share
    one write port through a priority mux carrying addr, data, and we -- each
    child self-gates its we outside its own phase, so the idle master never
    writes. The children write disjoint halves, so any leaked write corrupts
    the result."""

    @kernel
    def wm(s: i32[8], out: i32[8]):
        for i in range(4):
            out[i] = s[i] + 1  # writes out[0:4]

    @kernel
    def wn(s: i32[8], out: i32[8]):
        for i in range(4):
            out[i + 4] = s[i + 4] * 2  # writes out[4:8], sharing out's write port

    @kernel
    def sm_top(A: i32[8], out: i32[8]):
        s: i32[8]
        for i in range(8):  # loose region -> mixed container; s read by both
            s[i] = A[i] + 5
        wm(s, out)
        wn(s, out)

    A = np.arange(8, dtype=np.int32) + 1
    out = np.zeros(8, dtype=np.int32)
    r = _to_rtl(sm_top).cosim(A, out)
    assert r.cycles > 0
    exp = np.empty(8, dtype=np.int32)
    exp[:4] = (A[:4] + 5) + 1
    exp[4:] = (A[4:] + 5) * 2
    assert np.array_equal(out, exp)


def test_mixed_dataflow_sequential():
    """A container may mix an `await` dataflow sub-network with plain sequential
    kernels: an independent one runs concurrently, one that consumes the
    network's output is gated on the producer's real `done`."""

    # Data-INDEPENDENT of the dataflow processes (disjoint memory), so all run
    # concurrently: the container broadcasts `start` and joins every `done`, and
    # the df pair streams through its FIFO while the plain kernel runs beside it.
    @kernel
    async def mx_prod(s: Stream[i32]):
        for i in range(16):
            s.put(i * 2)

    @kernel
    async def mx_cons(s: Stream[i32], o1: i32[16]):
        for i in range(16):
            o1[i] = s.get() + 1

    @kernel
    def mx_post(D: i32[16], o2: i32[16]):  # plain (non-async) kernel, disjoint
        for i in range(16):
            o2[i] = D[i] + 100

    @kernel
    async def mx_top(D: i32[16], o1: i32[16], o2: i32[16]):
        fifo: Stream[i32]
        await mx_prod(fifo)
        await mx_cons(fifo, o1)
        mx_post(D, o2)

    mod = _to_rtl(mx_top)
    # A structural top holding both the df processes (+ FIFO) and the seq kernel.
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir

    D = (np.arange(16, dtype=np.int32) + 5) & 0xFF
    o1 = np.zeros(16, np.int32)
    o2 = np.zeros(16, np.int32)
    mod.cosim(D, o1, o2)
    assert np.array_equal(o1, np.array([2 * i + 1 for i in range(16)], np.int32))
    assert np.array_equal(o2, D + 100)

    # A plain kernel that CONSUMES the dataflow network's array output cannot
    # broadcast-start -- an async producer has no static latency, so there is no
    # offset to place it at. Its `start` is gated on the consumer's real `done`,
    # and it shares the `tmp` boundary serially: the writer fully drains first.
    @kernel
    async def rd_prod(s: Stream[i32]):
        for i in range(16):
            s.put(i * 2)

    @kernel
    async def rd_cons(s: Stream[i32], tmp: i32[16]):
        for i in range(16):
            tmp[i] = s.get() + 1

    @kernel
    def rd_post(tmp: i32[16], out: i32[16]):  # plain: consumes the df output
        for i in range(16):
            out[i] = tmp[i] * 3

    @kernel
    async def rd_top(tmp: i32[16], out: i32[16]):
        fifo: Stream[i32]
        await rd_prod(fifo)
        await rd_cons(fifo, tmp)  # the df network writes tmp
        rd_post(tmp, out)  # reads tmp, so it is gated on rd_cons's done

    mod = _to_rtl(rd_top)
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir
    assert "done_edge" in mod.mlir  # the handshake edge detector, not a broadcast
    assert "func.call @rd_top.rd_post(" in mod.dcp
    assert "dcp.instance" not in mod.dcp  # no child reified in a concurrent container

    tmp = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    mod.cosim(tmp, out)
    exp = np.array([(2 * i + 1) * 3 for i in range(16)], np.int32)
    assert np.array_equal(out, exp), list(out)

    # The same handshake reached from a PURE-sequential container: the producer
    # is a data-dependent `while` leaf with no static latency, so the consumer
    # that reads what it wrote must stall on its real `done`. Keyed on the
    # callee's determinacy (no `dcp.latency`), not a container-wide mode -- the
    # same per-child start map gives a determinate producer a static offset.
    @kernel
    def sp_prod(n0: i32, B: i32[16]):
        c: i32 = 0
        x: i32 = n0
        while x > 1:  # data-dependent trip -> whole-kernel latency unknown
            x = x - 1
            c = c + 1  # c = n0 - 1, escapes to the store loop (not DCE-able)
        for i in range(16):
            B[i] = i + c

    @kernel
    def sp_cons(B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = B[i] + 100

    @kernel
    def sp_top(n0: i32, B: i32[16], out: i32[16]):
        sp_prod(n0, B)  # indeterminate producer writes B
        sp_cons(B, out)  # reads B -> must wait for prod's real done

    assert _latency(sp_prod) is None

    mod = _to_rtl(sp_top)
    assert "dcp.instance @sp_top.sp_prod(" in mod.dcp
    assert _latency(sp_top) is None  # container inherits the while's indeterminacy

    spB = np.zeros(16, np.int32)
    spout = np.zeros(16, np.int32)
    mod.cosim(np.int32(5), spB, spout)  # n0 = 5 -> c = 4
    exp = np.array([i + 4 + 100 for i in range(16)], np.int32)
    assert np.array_equal(spout, exp), list(spout)


def test_dataflow_linear_chains():
    """Linear SPSC chains of `async def` processes wired through internal FIFOs:
    2/3/4 stages, scrambled spawn order, a float payload, and a declared FIFO
    depth. KPN determinism -- cosim == csim golden at any stall rate."""

    N = 16

    # Two processes wired producer -> FIFO -> consumer. `await` spawns each
    # concurrently; the enclosing async region forks `start` to both and joins on
    # both `done`. cosim drives only the composed top's boundary (`out`).
    @kernel
    async def spsc_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def spsc_cons(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def spsc_top(out: i32[N]):
        fifo: Stream[i32]
        await spsc_prod(fifo)
        await spsc_cons(fifo, out)

    mod = _to_rtl(spsc_top)
    # A structural top: it instantiates the leaf processes and a FIFO, not a
    # datapath of its own.
    assert "hw.instance" in mod.mlir and "seq.fifo" in mod.mlir

    golden = np.zeros(N, np.int32)
    mod.csim(golden)  # CPU dataflow-runtime golden
    exp = np.array([2 * i + 1 for i in range(N)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-stage chain, two internal channels: fork `start` to three, join three
    # `done`s, wire two seq.fifos in a row.
    @kernel
    async def c3_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def c3_mid(s: Stream[i32], t: Stream[i32]):
        for i in range(N):
            t.put(s.get() * 2)

    @kernel
    async def c3_cons(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get() + 1

    @kernel
    async def c3_top(out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await c3_prod(s)
        await c3_mid(s, t)
        await c3_cons(t, out)

    mod = _to_rtl(c3_top)
    assert mod.mlir.count("hw.instance") >= 3 and mod.mlir.count("seq.fifo") >= 2

    golden = np.zeros(N, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(N)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A deeper 4-stage chain, three internal channels -- the structural top
    # scales past three processes.
    N4 = 12

    @kernel
    async def c4_prod(s: Stream[i32]):
        for i in range(N4):
            s.put(i)

    @kernel
    async def c4_m1(s: Stream[i32], t: Stream[i32]):
        for i in range(N4):
            t.put(s.get() + 3)

    @kernel
    async def c4_m2(t: Stream[i32], u: Stream[i32]):
        for i in range(N4):
            u.put(t.get() * 2)

    @kernel
    async def c4_cons(u: Stream[i32], out: i32[N4]):
        for i in range(N4):
            out[i] = u.get() - 1

    @kernel
    async def c4_top(out: i32[N4]):
        s: Stream[i32]
        t: Stream[i32]
        u: Stream[i32]
        await c4_prod(s)
        await c4_m1(s, t)
        await c4_m2(t, u)
        await c4_cons(u, out)

    mod = _to_rtl(c4_top)
    assert mod.mlir.count("hw.instance") >= 4 and mod.mlir.count("seq.fifo") >= 3

    golden = np.zeros(N4, np.int32)
    mod.csim(golden)
    exp = np.array([(i + 3) * 2 - 1 for i in range(N4)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N4, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # The topology is defined by stream SSA wiring, not by spawn order: spawn a
    # 3-stage chain scrambled (cons, mid, prod) and it must still wire
    # prod -> mid -> cons.
    @kernel
    async def oo_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def oo_mid(s: Stream[i32], t: Stream[i32]):
        for i in range(N):
            t.put(s.get() * 2)

    @kernel
    async def oo_cons(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get() + 1

    @kernel
    async def oo_top(out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await oo_cons(t, out)  # spawned before its producer
        await oo_mid(s, t)
        await oo_prod(s)

    mod = _to_rtl(oo_top)
    golden = np.zeros(N, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(N)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-stage f32 chain with an input boundary array: the LI shell / FIFO carry
    # a float payload (as its bit pattern) across a multi-stage pipeline.
    @kernel
    async def fp_prod(a: f32[N], s: Stream[f32]):
        for i in range(N):
            s.put(a[i])

    @kernel
    async def fp_mid(s: Stream[f32], t: Stream[f32]):
        for i in range(N):
            t.put(s.get() * 2.0)

    @kernel
    async def fp_cons(t: Stream[f32], out: f32[N]):
        for i in range(N):
            out[i] = t.get() + 1.0

    @kernel
    async def fp_top(a: f32[N], out: f32[N]):
        s: Stream[f32]
        t: Stream[f32]
        await fp_prod(a, s)
        await fp_mid(s, t)
        await fp_cons(t, out)

    mod = _to_rtl(fp_top)
    fa = np.arange(N, dtype=np.float32)
    fexp = fa * 2.0 + 1.0
    golden = np.zeros(N, np.float32)
    mod.csim(fa, golden)
    assert np.array_equal(golden, fexp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(N, np.float32)
        mod.cosim(fa, out, stall_prob=gap)
        assert np.array_equal(out, fexp), f"gap={gap}: {list(out)}"

    # A user-declared internal FIFO depth (`Stream[i32, D]`) sizes the emitted
    # seq.fifo. The depth is part of the stream type, so it must be spelled
    # consistently on the channel and on every process parameter that touches it
    # (a mismatched depth is a type error). Back-pressure keeps any depth correct.
    @kernel
    async def fd_prod(s: Stream[i32, 4]):
        for i in range(N):
            s.put(i)

    @kernel
    async def fd_mid(s: Stream[i32, 4], t: Stream[i32, 1]):
        for i in range(N):
            t.put(s.get() * 2)

    @kernel
    async def fd_cons(t: Stream[i32, 1], out: i32[N]):
        for i in range(N):
            out[i] = t.get() + 1

    @kernel
    async def fd_top(out: i32[N]):
        s: Stream[i32, 4]  # deep internal FIFO
        t: Stream[i32, 1]  # tight internal FIFO
        await fd_prod(s)
        await fd_mid(s, t)
        await fd_cons(t, out)

    mod = _to_rtl(fd_top)
    # The deep channel keeps its depth; the depth-1 channel is raised to 2 (the
    # seq.fifo minimum, so it never appears as "depth 1") rather than crashing on
    # zero-width pointers, and the design still builds and runs.
    assert "depth 4" in mod.mlir and "depth 1" not in mod.mlir, mod.mlir
    golden = np.zeros(N, np.int32)
    mod.csim(golden)
    exp = np.array([2 * i + 1 for i in range(N)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


def test_dataflow_fanout_fanin():
    """Non-linear topologies: a producer branching to two independent consumer
    chains, and elastic joins reading two / three input streams per firing."""

    N = 16

    # Two output channels feeding two independent consumer chains (a branch, no
    # reconvergence). The producer's two puts share one region, so the out-hazard
    # ORs their back-pressure (all-or-nothing per firing). These are distinct
    # SPSC channels, not a broadcast. Two boundary output arrays exercise
    # multi-output wiring.
    @kernel
    async def split(a: Stream[i32], b: Stream[i32]):
        for i in range(N):
            a.put(i)
            b.put(i * 10)

    @kernel
    async def br_cons_a(a: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = a.get() + 1

    @kernel
    async def br_cons_b(b: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = b.get() - 1

    @kernel
    async def br_top(outa: i32[N], outb: i32[N]):
        a: Stream[i32]
        b: Stream[i32]
        await split(a, b)
        await br_cons_a(a, outa)
        await br_cons_b(b, outb)

    mod = _to_rtl(br_top)
    assert mod.mlir.count("hw.instance") >= 3 and mod.mlir.count("seq.fifo") >= 2

    ga = np.zeros(N, np.int32)
    gb = np.zeros(N, np.int32)
    mod.csim(ga, gb)
    expa = np.array([i + 1 for i in range(N)], np.int32)
    expb = np.array([i * 10 - 1 for i in range(N)], np.int32)
    assert np.array_equal(ga, expa), list(ga)
    assert np.array_equal(gb, expb), list(gb)
    for gap in (0.0, 0.5, 0.8):
        oa = np.zeros(N, np.int32)
        ob = np.zeros(N, np.int32)
        mod.cosim(oa, ob, stall_prob=gap)
        assert np.array_equal(oa, expa), f"gap={gap}: a={list(oa)}"
        assert np.array_equal(ob, expb), f"gap={gap}: b={list(ob)}"

    # A stage reading TWO input streams unconditionally in one region
    # (c = a.get() + b.get()) -- an elastic join. It consumes one token from EACH
    # per firing and pops them together, so under independent random stalls on
    # the two inputs no token is lost (the leading input waits for the lagging).
    @kernel
    async def j2_prodA(a: Stream[i32]):
        for i in range(N):
            a.put(i)

    @kernel
    async def j2_prodB(b: Stream[i32]):
        for i in range(N):
            b.put(i * 10)

    @kernel
    async def j2_join(a: Stream[i32], b: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = a.get() + b.get()

    @kernel
    async def j2_top(out: i32[N]):
        a: Stream[i32]
        b: Stream[i32]
        await j2_prodA(a)
        await j2_prodB(b)
        await j2_join(a, b, out)

    mod = _to_rtl(j2_top)
    golden = np.zeros(N, np.int32)
    mod.csim(golden)
    exp = np.array([i + i * 10 for i in range(N)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A 3-input elastic join -- the all-inputs-pop-together gating scales past
    # two inputs.
    N3 = 12

    @kernel
    async def j3_prodA(a: Stream[i32]):
        for i in range(N3):
            a.put(i)

    @kernel
    async def j3_prodB(b: Stream[i32]):
        for i in range(N3):
            b.put(i * 10)

    @kernel
    async def j3_prodC(c: Stream[i32]):
        for i in range(N3):
            c.put(i * 100)

    @kernel
    async def j3_join(a: Stream[i32], b: Stream[i32], c: Stream[i32], out: i32[N3]):
        for i in range(N3):
            out[i] = a.get() + b.get() + c.get()

    @kernel
    async def j3_top(out: i32[N3]):
        a: Stream[i32]
        b: Stream[i32]
        c: Stream[i32]
        await j3_prodA(a)
        await j3_prodB(b)
        await j3_prodC(c)
        await j3_join(a, b, c, out)

    mod = _to_rtl(j3_top)
    golden = np.zeros(N3, np.int32)
    mod.csim(golden)
    exp = np.array([i + i * 10 + i * 100 for i in range(N3)], np.int32)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.8):
        out = np.zeros(N3, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


def test_dataflow_deterministic_merge():
    """Deterministic MPSC merge: a stage consuming ONE of two inputs per firing,
    chosen by a data-determined selector -- from a control stream and from a
    memory array. Data-driven, not arrival-driven, so it stays in KPN."""

    N = 16

    # The selector is a control-stream token read at stage 0, so the chosen
    # `a`/`b` get lands at stage 1 (fifo read latency) -- a MULTI-STAGE join: the
    # selected mid-pipeline get FREEZES the pipeline when its input is empty,
    # while a non-selected empty input never stalls. Producers emit matching
    # counts (rate law): each of a/b is chosen N/2 times.
    pattern = np.array([i % 2 for i in range(N)], np.int32)  # 0,1,0,1,...

    @kernel
    async def cs_prodA(a: Stream[i32]):
        for i in range(N // 2):
            a.put(i)

    @kernel
    async def cs_prodB(b: Stream[i32]):
        for i in range(N // 2):
            b.put(100 + i)

    @kernel
    async def cs_prodSel(p: i32[N], sel: Stream[i32]):
        for i in range(N):
            sel.put(p[i])

    @kernel
    async def cs_merge(sel: Stream[i32], a: Stream[i32], b: Stream[i32], out: i32[N]):
        for i in range(N):
            s: i32 = sel.get()
            x: i32 = 0
            if s == 0:
                x = a.get()
            else:
                x = b.get()
            out[i] = x

    @kernel
    async def cs_top(p: i32[N], out: i32[N]):
        a: Stream[i32]
        b: Stream[i32]
        sel: Stream[i32]
        await cs_prodA(a)
        await cs_prodB(b)
        await cs_prodSel(p, sel)
        await cs_merge(sel, a, b, out)

    exp = np.zeros(N, np.int32)
    ca = cb = 0
    for i in range(N):
        if pattern[i] == 0:
            exp[i] = ca
            ca += 1
        else:
            exp[i] = 100 + cb
            cb += 1

    mod = _to_rtl(cs_top)
    golden = np.zeros(N, np.int32)
    mod.csim(pattern, golden)
    assert np.array_equal(golden, exp), (list(golden), list(exp))
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(pattern, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # The canonical case: the selector is a MEMORY-array read `sel[i]`. The load's
    # read latency puts the predicate at stage 1 and the selected get at stage 2 --
    # deeper than the control-stream form, so the multi-stage freeze must handle
    # arbitrary get depth.
    sel = np.array([i % 2 for i in range(N)], np.int32)

    @kernel
    async def ds_prodA(a: Stream[i32]):
        for i in range(N // 2):
            a.put(i)

    @kernel
    async def ds_prodB(b: Stream[i32]):
        for i in range(N // 2):
            b.put(100 + i)

    @kernel
    async def ds_merge(sel: i32[N], a: Stream[i32], b: Stream[i32], out: i32[N]):
        for i in range(N):
            # One scalar per branch and a single store: two stores to out[i]
            # would need two write ports -> II=2, which the II==1 shell rejects.
            x: i32 = 0
            if sel[i] == 0:
                x = a.get()
            else:
                x = b.get()
            out[i] = x

    @kernel
    async def ds_top(sel: i32[N], out: i32[N]):
        a: Stream[i32]
        b: Stream[i32]
        await ds_prodA(a)
        await ds_prodB(b)
        await ds_merge(sel, a, b, out)

    exp = np.zeros(N, np.int32)
    ca = cb = 0
    for i in range(N):
        if sel[i] == 0:
            exp[i] = ca
            ca += 1
        else:
            exp[i] = 100 + cb
            cb += 1

    mod = _to_rtl(ds_top)
    golden = np.zeros(N, np.int32)
    mod.csim(sel, golden)
    assert np.array_equal(golden, exp), (list(golden), list(exp))
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(sel, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


def test_transient_din_stability_under_backpressure():
    """FIFO-din stability under back-pressure (regression for the transient-din
    register). A stream producer whose token is a STAGE>=1 transient value -- a
    combinational function of a memory load, `f(load) = B[k]*3` -- has a delayed
    (shift-chain) valid, so output back-pressure holds it into the loop's drain
    where the counter resets. Unless the din is captured into a chain-enable-
    frozen register (bump the put one stage = Vitis's `v3_reg`), the held valid
    re-addresses the live read and commits a corrupted FINAL token. A STAGE-0
    counter-fed put (`put(k)`) instead freezes atomically with the issue pulse and
    must stay correct WITHOUT being over-registered (the `dcpStart(put)>=1` guard).

    A depth<K systolic column drives both deterministically: the interior PE
    can't keep up, so it back-pressures the border producer's last token onto the
    counter reset. M=1 keeps the flow a chain (back-pressure, no depth<K deadlock),
    so the value bug -- not a hang -- is what the assert catches."""
    M, N, K, DEPTH = 1, 2, 3, 2  # depth < K => the last put is held into the drain
    P0, P1 = M + 2, N + 2

    @kernel
    def sa_fload(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        fifo_A: Stream[i32, DEPTH][P0, P1]
        fifo_B: Stream[i32, DEPTH][P0, P1]

        @kernel(mapping=[P0, P1])
        def pe(
            A: i32[M, K],
            B: i32[K, N],
            C: i32[M, N],
            fifo_A: Stream[i32, DEPTH][P0, P1],
            fifo_B: Stream[i32, DEPTH][P0, P1],
        ):
            i = allo.get_wid(0)
            j = allo.get_wid(1)
            if (i == 0 or i == M + 1) and (j == 0 or j == N + 1):
                pass
            elif j == 0:
                for k in range(K):
                    fifo_A[i, j + 1].put(A[i - 1, k])
            elif i == 0:
                for k in range(K):
                    fifo_B[i + 1, j].put(B[k, j - 1] * 3)  # f(load): stage>=1
            elif i == M + 1:
                for k in range(K):
                    b: i32 = fifo_B[i, j].get()
            elif j == N + 1:
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
            else:
                c: i32 = 0
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
                    b: i32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                    fifo_B[i + 1, j].put(b)
                C[i - 1, j - 1] = c

        pe(A, B, C, fifo_A, fifo_B)

    @kernel
    def sa_counter(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        fifo_A: Stream[i32, DEPTH][P0, P1]
        fifo_B: Stream[i32, DEPTH][P0, P1]

        @kernel(mapping=[P0, P1])
        def pe(
            A: i32[M, K],
            B: i32[K, N],
            C: i32[M, N],
            fifo_A: Stream[i32, DEPTH][P0, P1],
            fifo_B: Stream[i32, DEPTH][P0, P1],
        ):
            i = allo.get_wid(0)
            j = allo.get_wid(1)
            if (i == 0 or i == M + 1) and (j == 0 or j == N + 1):
                pass
            elif j == 0:
                for k in range(K):
                    fifo_A[i, j + 1].put(A[i - 1, k])
            elif i == 0:
                for k in range(K):
                    fifo_B[i + 1, j].put(k)  # counter: stage-0, atomically frozen
            elif i == M + 1:
                for k in range(K):
                    b: i32 = fifo_B[i, j].get()
            elif j == N + 1:
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
            else:
                c: i32 = 0
                for k in range(K):
                    a: i32 = fifo_A[i, j].get()
                    b: i32 = fifo_B[i, j].get()
                    c += a * b
                    fifo_A[i, j + 1].put(a)
                    fifo_B[i + 1, j].put(b)
                C[i - 1, j - 1] = c

        pe(A, B, C, fifo_A, fifo_B)

    A = np.array([[4, 3, 2]], dtype=np.int32)
    B = np.array([[1, 1], [2, 0], [1, 3]], dtype=np.int32)

    # f(load): the forwarded b-token is 3*B[k], so C = A @ (3*B). The final put is
    # held onto the counter reset -- without the din register it commits 3*B[0].
    mod = _to_rtl(sa_fload)
    out = np.zeros((M, N), np.int32)
    mod.cosim(A, B, out)
    exp = A @ (3 * B)
    assert np.array_equal(out, exp), (list(out.ravel()), list(exp.ravel()))

    # counter: the forwarded b-token is k (every column), so C[i,j] = sum_k A[i,k]*k.
    # The stage-0 put must be correct WITHOUT the extra register.
    mod = _to_rtl(sa_counter)
    out = np.zeros((M, N), np.int32)
    mod.cosim(A, B, out)
    exp = A @ np.repeat(np.arange(K, dtype=np.int32)[:, None], N, axis=1)
    assert np.array_equal(out, exp), (list(out.ravel()), list(exp.ravel()))


def test_dataflow_nested_containers():
    """A process that is itself a container: the CPU golden across two and three
    nesting levels, then RTL emit of a container-as-callee and of a container
    whose stream args cross its boundary."""

    N = 16

    # The runtime flattens the nest onto one marl scheduler (a nested
    # `allo_df_open` reuses the enclosing scheduler instead of binding a second
    # one to the fiber's thread, which aborts); each level keeps its own
    # WaitGroup so joins are scoped.
    a_nc = (np.arange(N, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def nc_produce(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i])

    @kernel
    async def nc_inner_a(x: Stream[i32], y: Stream[i32]):
        for i in range(N):
            y.put(x.get() + 1)

    @kernel
    async def nc_inner_b(y: Stream[i32], z: Stream[i32]):
        for i in range(N):
            z.put(y.get() * 2)

    @kernel
    async def nc_mid(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await nc_inner_a(x, y)
        await nc_inner_b(y, z)

    @kernel
    async def nc_consume(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get()

    @kernel
    async def nc_top(a: i32[N], out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await nc_produce(a, s)
        await nc_mid(s, t)
        await nc_consume(t, out)

    mod = _to_rtl(nc_top)
    exp = (a_nc + 1) * 2
    # csim is deterministic (KPN); repeat to surface any scheduler/WaitGroup race.
    for _ in range(8):
        out = np.zeros(N, np.int32)
        mod.csim(a_nc, out)
        assert np.array_equal(out, exp), list(out)

    # Three container levels: top -> mid -> deep -> {da, db}. The scheduler-reuse
    # flattening holds at arbitrary nesting depth.
    a_dn = (np.arange(N, dtype=np.int32) * 3 + 5) & 0xFF

    @kernel
    async def dn_produce(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i])

    @kernel
    async def dn_da(x: Stream[i32], y: Stream[i32]):
        for i in range(N):
            y.put(x.get() + 3)

    @kernel
    async def dn_db(y: Stream[i32], z: Stream[i32]):
        for i in range(N):
            z.put(y.get() * 2)

    @kernel
    async def dn_deep(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await dn_da(x, y)
        await dn_db(y, z)

    @kernel
    async def dn_mid(x: Stream[i32], z: Stream[i32]):
        await dn_deep(x, z)

    @kernel
    async def dn_consume(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get()

    @kernel
    async def dn_top(a: i32[N], out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await dn_produce(a, s)
        await dn_mid(s, t)
        await dn_consume(t, out)

    mod = _to_rtl(dn_top)
    out = np.zeros(N, np.int32)
    mod.csim(a_dn, out)
    assert np.array_equal(out, (a_dn + 3) * 2), list(out)

    # A spawned process that is itself a container, with only MEMREF boundaries
    # -- no stream crosses cc_mid's boundary, isolating container-as-callee from
    # stream boundary ports. Emit must build cc_mid as its own hw.module before
    # the top, keep the outermost (uncalled) container as the DUT, and forward
    # cc_mid's memref boundaries through it.
    a_cc = (np.arange(N, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def cc_inner_p(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i] * 2)

    @kernel
    async def cc_inner_c(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def cc_mid(a: i32[N], out: i32[N]):
        s: Stream[i32]
        await cc_inner_p(a, s)
        await cc_inner_c(s, out)

    @kernel
    async def cc_top(a: i32[N], out: i32[N]):
        await cc_mid(a, out)

    mod = _to_rtl(cc_top)
    mods = re.findall(r"hw\.module @([\w.]+)", mod.mlir)
    assert mods[0] == "cc_top", mods
    assert "cc_top.cc_mid" in mods, mods
    assert mods.index("cc_top.cc_mid") > mods.index("cc_top"), mods

    golden = np.zeros(N, np.int32)
    mod.csim(a_cc, golden)
    assert np.array_equal(golden, a_cc * 2 + 1), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(a_cc, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A channel crossing a container boundary: sb_mid's two stream args are block
    # args forwarded to its inner processes, so it must expose them as stream
    # ports (data/valid/ready) that look exactly like a leaf's, and the parent
    # wires a FIFO on each side. The full hierarchical composition.
    a_sb = (np.arange(N, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def sb_produce(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i])

    @kernel
    async def sb_inner_a(x: Stream[i32], y: Stream[i32]):
        for i in range(N):
            y.put(x.get() + 1)

    @kernel
    async def sb_inner_b(y: Stream[i32], z: Stream[i32]):
        for i in range(N):
            z.put(y.get() * 2)

    @kernel
    async def sb_mid(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await sb_inner_a(x, y)
        await sb_inner_b(y, z)

    @kernel
    async def sb_consume(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get()

    @kernel
    async def sb_top(a: i32[N], out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await sb_produce(a, s)
        await sb_mid(s, t)
        await sb_consume(t, out)

    mod = _to_rtl(sb_top)
    ir = mod.mlir
    assert re.findall(r"hw\.module @([\w.]+)", ir)[0] == "sb_top"
    assert "_strm_data" in ir and "_strm_valid" in ir and "_strm_ready" in ir
    assert ir.count("seq.fifo") >= 3, ir.count("seq.fifo")  # s, t, and mid's y

    golden = np.zeros(N, np.int32)
    mod.csim(a_sb, golden)
    assert np.array_equal(golden, (a_sb + 1) * 2), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(a_sb, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


def test_dataflow_channel_init_and_rejections():
    """Channels seeded with initial tokens -- a feedback cycle and a NumPy-array
    capture -- plus the two topologies the compiler must reject rather than emit:
    an unseeded cycle (deadlocks) and a channel read by two processes."""

    N = 8

    # A dataflow CYCLE seeded with one token: `fb_emit` reads the feedback channel
    # `t`, records it, and produces x+1 into `s`; `fb_fwd` forwards s -> t. The
    # preloaded token turns the cycle, so out = [0, 1, ..., N-1]. Golden (csim)
    # seeds the FIFO; RTL (cosim) prepends the token on the consumer read port.
    @kernel
    async def fb_emit(t: Stream[i32], s: Stream[i32], out: i32[N]):
        for i in range(N):
            x = t.get()
            out[i] = x
            s.put(x + 1)

    @kernel
    async def fb_fwd(s: Stream[i32], t: Stream[i32]):
        for i in range(N):
            t.put(s.get())

    @kernel
    async def fb_top(out: i32[N]):
        s: Stream[i32]
        t: Stream[i32] = [0]  # feedback channel, one initial token
        await fb_emit(t, s, out)
        await fb_fwd(s, t)

    mod = _to_rtl(fb_top)
    ir = mod.mlir
    # The seeded channel keeps the plain `seq.fifo` and adds the init-prepend
    # shim (its down-counter) on the consumer side.
    assert "hw.instance" in ir and "seq.fifo" in ir and "fifo_init_rem" in ir

    golden = np.zeros(N, np.int32)
    mod.csim(golden)  # CPU dataflow-runtime golden: seeded, no deadlock
    assert np.array_equal(golden, np.arange(N, dtype=np.int32)), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(N, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # Initial tokens from an externally-defined NumPy array captured into the
    # kernel -- more elements than a hand-written list, exercising the shim's
    # init-ROM mux chain + multi-bit down-counter. Acyclic SPSC: the channel
    # history is [init] ++ [produced], so cons reads the K seeded tokens first,
    # then prod's M values (back-pressure carries the producer tokens through the
    # depth-2 FIFO while the init drains).
    K, M = 8, 8
    INIT = np.random.default_rng(0).integers(0, 1000, size=K, dtype=np.int32)

    @kernel
    async def cap_prod(c: Stream[i32]):
        for i in range(M):
            c.put(100 + i)

    @kernel
    async def cap_cons(c: Stream[i32], out: i32[K + M]):
        for i in range(K + M):
            out[i] = c.get()

    @kernel
    async def cap_top(out: i32[K + M]):
        c: Stream[i32] = INIT  # seeded from the captured NumPy array
        await cap_prod(c)
        await cap_cons(c, out)

    mod = _to_rtl(cap_top)
    exp = np.concatenate([INIT, 100 + np.arange(M, dtype=np.int32)])
    golden = np.zeros(K + M, np.int32)
    mod.csim(golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.7):
        out = np.zeros(K + M, np.int32)
        mod.cosim(out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # A feedback cycle with NO initial token deadlocks, so the emit-stage liveness
    # check must reject it rather than emit a design that hangs.
    @kernel
    async def un_emit(t: Stream[i32], s: Stream[i32], out: i32[8]):
        for i in range(8):
            x = t.get()
            out[i] = x
            s.put(x + 1)

    @kernel
    async def un_fwd(s: Stream[i32], t: Stream[i32]):
        for i in range(8):
            t.put(s.get())

    @kernel
    async def un_top(out: i32[8]):
        s: Stream[i32]
        t: Stream[i32]  # feedback channel, UNSEEDED -> would deadlock
        await un_emit(t, s, out)
        await un_fwd(s, t)

    with pytest.raises(Exception, match="no initial tokens"):
        _to_rtl(un_top).mlir  # accessing the IR forces the (lazy) emit + the check

    # One channel read by two processes (SPMC broadcast) is not inserted
    # automatically; it must be rejected with a user-facing error pointing at the
    # one-channel-per-consumer idiom, not crash on an internal assert.
    @kernel
    async def spmc_prod(s: Stream[i32]):
        for i in range(8):
            s.put(i)

    @kernel
    async def spmc_cons_a(s: Stream[i32], out: i32[8]):
        for i in range(8):
            out[i] = s.get()

    @kernel
    async def spmc_cons_b(s: Stream[i32], out: i32[8]):
        for i in range(8):
            out[i] = s.get()

    @kernel
    async def spmc_top(out0: i32[8], out1: i32[8]):
        fifo: Stream[i32]
        await spmc_prod(fifo)
        await spmc_cons_a(fifo, out0)
        await spmc_cons_b(fifo, out1)  # second consumer of the same channel

    with pytest.raises(Exception, match="more than one"):
        _to_rtl(spmc_top).mlir


# --- Predicated stream access (conditional get / put) -----------------------


def test_dataflow_predicated_stream_access():
    """Data-dependent conditional `put` / `get`: the branch condition becomes the
    access's i1 predicate, so it stays in the pipelined region. End-to-end, a
    filter's output rate and a gated read's input rate are data-dependent."""

    # Masked in place rather than serialized into a guard region.
    @kernel
    async def pp_prod(a: i32[16], y: Stream[i32]):
        for i in range(16):
            x = a[i]
            if x > 0:
                y.put(x)

    mod = _to_rtl(pp_prod)
    res = mod.schedule()
    # The put carries a predicate (`... if %c`), the loop pipelines at II=1, and
    # no guard (dcp.select) / raw scf.if is left.
    assert "allo.stream.put" in mod.dcp and "if %" in mod.dcp
    assert "scf.if" not in mod.dcp and "dcp.select" not in mod.dcp
    assert _iis(res.func("pp_prod").cyclic()) == [1]
    assert not any(r.kind == "guard" for r in res.funcs[0].regions)

    # A filter process puts only the tokens that pass a data-dependent test, so
    # its output rate is data-dependent (non-SDF -- what Vitis dataflow rejects).
    # The consumer reads the M tokens that pass.
    N, M = 16, 8

    @kernel
    async def pf_prod(a: i32[N], y: Stream[i32]):
        for i in range(N):
            x = a[i]
            if x > 0:
                y.put(x)

    @kernel
    async def pf_cons(y: Stream[i32], out: i32[M]):
        for i in range(M):
            out[i] = y.get()

    @kernel
    async def pf_top(a: i32[N], out: i32[M]):
        y: Stream[i32]
        await pf_prod(a, y)
        await pf_cons(y, out)

    rtl = _to_rtl(pf_top)
    # a[i] positive at even i -> exactly M tokens pass the filter.
    a = np.array([(i + 1) if i % 2 == 0 else -(i + 1) for i in range(N)], np.int32)
    exp = np.array([i + 1 for i in range(N) if i % 2 == 0], np.int32)

    golden = np.zeros(M, np.int32)
    rtl.csim(a, golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(M, np.int32)
        rtl.cosim(a, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"

    # The consumer side: a gated read consumes a token only where a data-dependent
    # select holds, else emits a default WITHOUT popping the channel. The
    # predicated get pops only when consuming and never stalls the pipeline on the
    # (empty) channel in the skipped iterations.
    NG, MG = 8, 4

    @kernel
    async def pg_prod(y: Stream[i32]):
        for i in range(MG):  # exactly as many tokens as are read
            y.put(i)

    @kernel
    async def pg_cons(sel: i32[NG], y: Stream[i32], out: i32[NG]):
        for i in range(NG):
            v: i32 = -1
            if sel[i] > 0:
                v = y.get()  # pop only where sel>0; single store below -> II=1
            out[i] = v

    @kernel
    async def pg_top(sel: i32[NG], out: i32[NG]):
        y: Stream[i32]
        await pg_prod(y)
        await pg_cons(sel, y, out)

    rtl = _to_rtl(pg_top)
    sel = np.array([1, -1, 1, -1, 1, -1, 1, -1], np.int32)  # 4 reads
    exp = np.array([0, -1, 1, -1, 2, -1, 3, -1], np.int32)
    golden = np.zeros(NG, np.int32)
    rtl.csim(sel, golden)
    assert np.array_equal(golden, exp), list(golden)
    for gap in (0.0, 0.5, 0.8):
        out = np.zeros(NG, np.int32)
        rtl.cosim(sel, out, stall_prob=gap)
        assert np.array_equal(out, golden), f"gap={gap}: {list(out)}"


# --- Dynamic-trip loops (scf.for) ------------------------------------------


def test_dynamic_trip_scheduling():
    """A memory-loaded bound is not affine, so the loop stays a runtime-trip band:
    it still pipelines, but the latency is deferred rather than faked. A carried
    memory recurrence under such a bound is closed conservatively."""

    @kernel
    def dyn(A: i32[128], out: i32[1]):
        n: index = A[0]
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    loop = _sched(dyn).cyclic()[0]
    assert loop.ii == 1  # scalar int accumulate, add is combinational
    assert loop.latency is None  # unknown trip -> latency deferred, not faked

    @kernel
    def recur(A: i32[128], nb: i32[1]):
        n: index = nb[0]
        for i in range(1, n):
            A[i] = A[i - 1] + A[i]

    # A[i] reads A[i-1]: a conservative distance-1 back edge forces II > 1;
    # without it the II would be an unsound, optimistic 1.
    assert _sched(recur).cyclic()[0].ii > 1


def test_dynamic_trip_cosim():
    """A dynamic-trip loop stays a free-running / modulo pipeline terminating on
    `counter == bound` against a runtime value: the count is data, the
    per-iteration timing stays static. No stall, no flush."""

    # Store-less reduction: the runtime bound free-runs the pipeline and its
    # result flows to the epilogue store (capture-based done).
    @kernel
    def dyn(A: i32[128], out: i32[1]):
        n: index = A[0]
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    for N in (5, 1, 12):
        A = np.zeros(128, np.int32)
        A[0] = N
        A[1:N] = np.arange(1, N, dtype=np.int32) * 3 + 2
        out = np.zeros(1, np.int32)
        _to_rtl(dyn).cosim(A, out)
        assert out[0] == int(A[:N].sum())

    # Store-ful: the store-counting done retires `bound` stores, gated by a
    # has-run latch so a runtime bound reading 0 at reset cannot fire done before
    # the loop issues.
    @kernel
    def dynstore(A: i32[64], nb: i32[1], out: i32[64]):
        n: index = nb[0]
        for i in range(n):
            out[i] = A[i] * 2

    for N in (7, 3):
        A = np.arange(64, dtype=np.int32) * 2 + 1
        out = np.zeros(64, np.int32)
        _to_rtl(dynstore).cosim(A, np.array([N], np.int32), out)
        assert np.array_equal(out[:N], A[:N] * 2)

    # Runtime bound on a modulo (II>1) pipeline: the float accumulate recurrence
    # forces II=FADD, and termination is `counter+1 == bound` on the issue.
    @kernel
    def dynfsum(A: f32[64], nb: i32[1], out: f32[1]):
        n: index = nb[0]
        s: f32 = 0.0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    for N in (7, 3):
        Af = np.arange(64, dtype=np.float32) * 0.5 + 1.0
        outf = np.zeros(1, np.float32)
        _to_rtl(dynfsum).cosim(Af, np.array([N], np.int32), outf)
        assert abs(float(outf[0]) - float(Af[:N].sum())) < 1e-3


# --- A+ induction: the counter holds the real IV (lb/step), fixing lb != 0 -----


def test_nonzero_lb_stencil_cosim():
    # An affine static stencil with lb=1 (no loop-carried dep). The induction
    # register runs the real IV (1..N-2), so it writes B[1..N-2] and reads
    # A[i-1..i+1] with no off-by-lb address shift.
    @kernel
    def jac(A: f32[8], B: f32[8]):
        for i in range(1, 7):
            B[i] = (A[i - 1] + A[i] + A[i + 1]) * 0.5

    A = _f32(8)
    B = np.zeros(8, np.float32)  # a pure-output buffer is zero-inited by cosim
    _to_rtl(jac).cosim(A, B)
    exp = np.zeros(8, np.float32)
    exp[1:7] = (A[:-2] + A[1:-1] + A[2:]) * 0.5  # B[0], B[7] stay 0 (untouched)
    assert np.allclose(B, exp, rtol=1e-4, atol=1e-5)


def test_runtime_lb_fixed_window_cosim():
    # The fixed-window idiom `for j in range(i, i+K)`: a RUNTIME lower bound (the
    # enclosing counter i) with a COMPILE-TIME trip K. The reifier keeps trip=K
    # and wires i as a runtime lbBound (no dynamicBound), so the loop's upper
    # bound must be computed as `lb + K*step` from the resolved runtime lb/step,
    # NOT the lb/step attributes (which default to 0/1 here). The i>=K iteration
    # is the telltale: an attribute-derived ub=K makes it spuriously empty.
    A = np.arange(16, dtype=np.int32)

    # (1) leaf window, step 1. i runs 0..3 with K=3, so i=3 is exactly the old
    #     spurious-empty edge (ub would have been konst(3), and lb=3 >= 3).
    @kernel
    def win(A: i32[16], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 3):
                s = s + A[j]
            out[i] = s

    out = np.zeros(4, np.int32)
    _to_rtl(win).cosim(A, out)
    assert np.array_equal(out, [A[i] + A[i + 1] + A[i + 2] for i in range(4)])

    # (2) non-unit step: the span is trip*step (=3*2), still anchored at runtime i.
    @kernel
    def stride(A: i32[16], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 6, 2):  # j = i, i+2, i+4
                s = s + A[j]
            out[i] = s

    out = np.zeros(4, np.int32)
    _to_rtl(stride).cosim(A, out)
    assert np.array_equal(out, [A[i] + A[i + 2] + A[i + 4] for i in range(4)])

    # (3) the window loop is a CONTAINER (its body nests a k loop), exercising the
    #     emitContainer terminatorOf path, not just the leaf pipeline.
    @kernel
    def wcont(A: i32[8, 4], out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(i, i + 3):
                for k in range(4):
                    s = s + A[j, k]
            out[i] = s

    A2 = np.arange(32, dtype=np.int32).reshape(8, 4)
    out = np.zeros(4, np.int32)
    _to_rtl(wcont).cosim(A2, out)
    assert np.array_equal(
        out,
        [
            sum(int(A2[j, k]) for j in range(i, i + 3) for k in range(4))
            for i in range(4)
        ],
    )


def test_negative_lb_signed_counter_cosim():
    # A compile-time NEGATIVE lower bound: the induction counter runs through
    # negative values (-4..3), so the bound tests (isLast/isEmpty) must be SIGNED.
    # An unsigned compare reads lb=-4 as ~4.3e9 >= ub, so `isEmpty` fires and the
    # whole loop is dropped; the all-8-outputs result proves it is not. `i` is
    # used both as a shifted address (A[i+4]) and a signed compute operand (+ i).
    @kernel
    def neglb(A: i32[8], out: i32[8]):
        for i in range(-4, 4):
            out[i + 4] = A[i + 4] + i

    A = np.arange(8, dtype=np.int32) * 10
    out = np.zeros(8, np.int32)
    _to_rtl(neglb).cosim(A, out)
    assert np.array_equal(out, [A[i + 4] + i for i in range(-4, 4)])

    # Negative lb on a reduction (II is the memory-carried recurrence): the
    # counter still seeds -4 and the signed bound test bounds the trip at 8.
    @kernel
    def neg_reduce(A: i32[8], out: i32[1]):
        acc: i32 = 0
        for i in range(-4, 4):
            acc = acc + A[i + 4] * i
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(neg_reduce).cosim(A, out)
    assert out[0] == sum(int(A[i + 4]) * i for i in range(-4, 4))


def test_heat_3d_stencil_cosim():
    # A 3-D 7-point stencil (polybench heat_3d, shrunk): a perfect nest whose i/j
    # loops are non-zero-lb *containers* and whose innermost k loop reads/writes at
    # the real IV. Cross-buffer (B from A, then A from B), so the two sweeps
    # sequence with no in-place recurrence; correctness turns on every nested
    # container counting from lb=1, not 0.
    N = 5

    @kernel
    def heat(A: f32[N, N, N], B: f32[N, N, N]):
        c0: f32 = 0.125
        c1: f32 = 2.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        c0 * (A[i + 1, j, k] - c1 * A[i, j, k] + A[i - 1, j, k])
                        + c0 * (A[i, j + 1, k] - c1 * A[i, j, k] + A[i, j - 1, k])
                        + c0 * (A[i, j, k + 1] - c1 * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    A[i, j, k] = (
                        c0 * (B[i + 1, j, k] - c1 * B[i, j, k] + B[i - 1, j, k])
                        + c0 * (B[i, j + 1, k] - c1 * B[i, j, k] + B[i, j - 1, k])
                        + c0 * (B[i, j, k + 1] - c1 * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )

    A = _f32(N, N, N)
    B = (_f32(N, N, N) + np.float32(0.5)).astype(np.float32)  # decorrelate B from A
    Ag, Bg = A.copy(), B.copy()
    c0, c1 = np.float32(0.125), np.float32(2.0)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                Bg[i, j, k] = (
                    c0 * (Ag[i + 1, j, k] - c1 * Ag[i, j, k] + Ag[i - 1, j, k])
                    + c0 * (Ag[i, j + 1, k] - c1 * Ag[i, j, k] + Ag[i, j - 1, k])
                    + c0 * (Ag[i, j, k + 1] - c1 * Ag[i, j, k] + Ag[i, j, k - 1])
                    + Ag[i, j, k]
                )
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                Ag[i, j, k] = (
                    c0 * (Bg[i + 1, j, k] - c1 * Bg[i, j, k] + Bg[i - 1, j, k])
                    + c0 * (Bg[i, j + 1, k] - c1 * Bg[i, j, k] + Bg[i, j - 1, k])
                    + c0 * (Bg[i, j, k + 1] - c1 * Bg[i, j, k] + Bg[i, j, k - 1])
                    + Bg[i, j, k]
                )
    _to_rtl(heat).cosim(A, B)
    assert np.allclose(A, Ag, rtol=2e-3, atol=2e-3)
    assert np.allclose(B, Bg, rtol=2e-3, atol=2e-3)


def test_seidel_2d_inplace_recurrence_cosim():
    # A 2-D in-place stencil (polybench seidel_2d, shrunk): A[i,j] reads its
    # already-updated neighbours A[i-1,*] and A[i,j-1], a genuine loop-carried
    # memory recurrence over a non-zero-lb nest. The recurrence forces II>1, which
    # serializes each read after the prior write, so the in-place sweep reproduces
    # the sequential result exactly.
    N = 6

    @kernel
    def seidel(A: f32[N, N]):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A[i, j] = (
                    A[i - 1, j - 1]
                    + A[i - 1, j]
                    + A[i - 1, j + 1]
                    + A[i, j - 1]
                    + A[i, j]
                    + A[i, j + 1]
                    + A[i + 1, j - 1]
                    + A[i + 1, j]
                    + A[i + 1, j + 1]
                ) / 9.0

    A = _f32(N, N)
    Ag = A.copy()
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            Ag[i, j] = (
                Ag[i - 1, j - 1]
                + Ag[i - 1, j]
                + Ag[i - 1, j + 1]
                + Ag[i, j - 1]
                + Ag[i, j]
                + Ag[i, j + 1]
                + Ag[i + 1, j - 1]
                + Ag[i + 1, j]
                + Ag[i + 1, j + 1]
            ) / np.float32(9.0)
    _to_rtl(seidel).cosim(A)
    assert np.allclose(A, Ag, rtol=2e-3, atol=2e-3)


def test_nested_reduction_container_cosim():
    # A reduction whose accumulator crosses TWO loop levels: the outer `for m`
    # loop is a counted *container* that carries `acc` into the inner `for n`
    # reduction. The container latches its iter-arg into a survivor register
    # (init at start, advanced by the inner loop's result each outer iteration),
    # which both the inner accumulator (its init) and the outer store read. A
    # single-level reduction (`for j: acc += …`) uses a fused accumulator instead.
    @kernel
    def red2(A: i32[4, 4, 4], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for m in range(4):
                for n in range(4):
                    acc = acc + A[i, m, n]
            out[i] = acc

    A = (np.arange(64, dtype=np.int32) % 7 + 1).reshape(4, 4, 4)
    out = np.zeros(4, np.int32)
    _to_rtl(red2).cosim(A.copy(), out)
    assert np.array_equal(out, A.reshape(4, -1).sum(axis=1))


def test_stencil2d_grid_reduction_cosim():
    # MachSuite stencil2d (shrunk): a 2-D grid() whose body accumulates a 3x3
    # window into `temp`, then stores it. Exercises two gaps together: the window
    # `for m: for n: temp += …` is the nested-reduction container above, and the
    # grid (ROW-2 = COL-2 = 3, a non-power-of-two) coalesces to one loop whose
    # index delinearizes by `div/mod 3` -- a general unsigned divide/remainder,
    # not a shift/mask. `sol` is written through an explicit out-parameter (the
    # interior; the caller-zeroed border stays 0).
    ROW, COL, F = 5, 5, 9

    @kernel
    def stencil2d(orig: i32[ROW, COL], filt: i32[F], sol: i32[ROW, COL]):
        for i, j in allo.grid(ROW - 2, COL - 2):
            temp: i32 = 0
            for m in range(3):
                for n in range(3):
                    mul: i32 = filt[m * 3 + n] * orig[i + m, j + n]
                    temp += mul
            sol[i, j] = temp

    orig = (np.arange(ROW * COL, dtype=np.int32) % 5 + 1).reshape(ROW, COL)
    filt = np.arange(F, dtype=np.int32) % 3 + 1
    sol = np.zeros((ROW, COL), np.int32)
    _to_rtl(stencil2d).cosim(orig.copy(), filt.copy(), sol)
    exp = np.zeros((ROW, COL), np.int32)
    for i in range(ROW - 2):
        for j in range(COL - 2):
            exp[i, j] = sum(
                int(filt[m * 3 + n]) * int(orig[i + m, j + n])
                for m in range(3)
                for n in range(3)
            )
    assert np.array_equal(sol, exp)


def test_stencil3d_grid_boundary_cosim():
    # MachSuite stencil3d (shrunk): four grid() sweeps -- three boundary copies
    # plus an interior 6-neighbour sum -- over a shared out-param `sol`. No nested
    # reduction (the body is straight-line), but the boundary grids coalesce to
    # non-power-of-two extents (C, R = 6) whose index delinearizes by `div/mod 6`,
    # while the interior grid (4x4x4) delinearizes by a power-of-two shift -- so
    # both address-lowering paths run.
    R, C, H = 6, 6, 6

    @kernel
    def stencil3d(coeff: i32[2], orig: i32[R, C, H], sol: i32[R, C, H]):
        for j, k in allo.grid(C, R):
            sol[k, j, 0] = orig[k, j, 0]
            sol[k, j, H - 1] = orig[k, j, H - 1]
        for i, k in allo.grid(H - 1, R):
            sol[k, 0, i + 1] = orig[k, 0, i + 1]
            sol[k, C - 1, i + 1] = orig[k, C - 1, i + 1]
        for j, i in allo.grid(C - 2, H - 2):
            sol[0, j + 1, i + 1] = orig[0, j + 1, i + 1]
            sol[R - 1, j + 1, i + 1] = orig[R - 1, j + 1, i + 1]
        for i, j, k in allo.grid(H - 2, C - 2, R - 2):
            sum0: i32 = orig[k + 1, j + 1, i + 1]
            sum1: i32 = (
                orig[k + 1, j + 1, i + 2]
                + orig[k + 1, j + 1, i]
                + orig[k + 1, j + 2, i + 1]
                + orig[k + 1, j, i + 1]
                + orig[k + 2, j + 1, i + 1]
                + orig[k, j + 1, i + 1]
            )
            sol[k + 1, j + 1, i + 1] = sum0 * coeff[0] + sum1 * coeff[1]

    coeff = np.array([2, 3], np.int32)
    orig = (np.arange(R * C * H, dtype=np.int32) % 5 + 1).reshape(R, C, H)
    sol = np.zeros((R, C, H), np.int32)
    _to_rtl(stencil3d).cosim(coeff.copy(), orig.copy(), sol)
    exp = np.zeros((R, C, H), np.int32)
    for j in range(C):
        for k in range(R):
            exp[k, j, 0] = orig[k, j, 0]
            exp[k, j, H - 1] = orig[k, j, H - 1]
    for i in range(H - 1):
        for k in range(R):
            exp[k, 0, i + 1] = orig[k, 0, i + 1]
            exp[k, C - 1, i + 1] = orig[k, C - 1, i + 1]
    for j in range(C - 2):
        for i in range(H - 2):
            exp[0, j + 1, i + 1] = orig[0, j + 1, i + 1]
            exp[R - 1, j + 1, i + 1] = orig[R - 1, j + 1, i + 1]
    for i in range(H - 2):
        for j in range(C - 2):
            for k in range(R - 2):
                s1 = int(
                    orig[k + 1, j + 1, i + 2]
                    + orig[k + 1, j + 1, i]
                    + orig[k + 1, j + 2, i + 1]
                    + orig[k + 1, j, i + 1]
                    + orig[k + 2, j + 1, i + 1]
                    + orig[k, j + 1, i + 1]
                )
                exp[k + 1, j + 1, i + 1] = int(orig[k + 1, j + 1, i + 1]) * 2 + s1 * 3
    assert np.array_equal(sol, exp)


def test_static_lb_and_step_cosim():
    """The induction register holds the real IV, so a non-zero lower bound or a
    non-unit step addresses without an off-by-lb shift and touches only the
    indices the loop actually visits."""

    # lb=2: the IV runs 2..15, so out[0..1] is left alone.
    @kernel
    def shifted(A: i32[16], out: i32[16]):
        for i in range(2, 16):
            out[i] = A[i] * 3

    out = np.zeros(16, np.int32)
    _to_rtl(shifted).cosim(A16, out)
    assert np.array_equal(out[2:], A16[2:] * 3)
    assert np.all(out[:2] == 0)

    # step=2 (lb=0): the IV runs 0,2,4,...,14.
    @kernel
    def stride2(A: i32[16], out: i32[16]):
        for i in range(0, 16, 2):
            out[i] = A[i] + 5

    out = np.zeros(16, np.int32)
    _to_rtl(stride2).cosim(A16, out)
    exp = np.zeros(16, np.int32)
    exp[0:16:2] = A16[0:16:2] + 5  # odd indices stay 0: only the even IV writes
    assert np.array_equal(out, exp)

    # A static empty loop (trip=0): lb >= ub, so it issues nothing and completes
    # on `start` rather than deadlocking (a store-drain done would never fire).
    # The sibling store must still run.
    @kernel
    def zt(A: i32[8], out: i32[8]):
        for i in range(1, 1):
            out[i] = A[i] + 99
        for i in range(8):
            out[i] = A[i] * 2

    out = np.zeros(8, np.int32)
    _to_rtl(zt).cosim(A16[:8].copy(), out)
    assert np.array_equal(out, A16[:8] * 2)


def test_runtime_lb_and_step_cosim():
    """A runtime lower bound / upper bound / stride is wired as a bound operand,
    so the induction register still runs the real IV. Includes the runtime
    zero-trip cases (lb >= ub), which must complete rather than deadlock."""

    # Constant lb=1 with a RUNTIME ub: the in-place recurrence must index
    # correctly (no spurious A[0] = A[-1] + A[0]). n==1 is the dynamic zero-trip.
    @kernel
    def recur(A: i32[16], nb: i32[1]):
        n: index = nb[0]
        for i in range(1, n):
            A[i] = A[i - 1] + A[i]

    for N in (16, 5, 1):
        A = (np.arange(16, dtype=np.int32) % 7 + 1).copy()
        exp = A.copy()
        for i in range(1, N):
            exp[i] = exp[i - 1] + exp[i]
        _to_rtl(recur).cosim(A, np.array([N], np.int32))
        assert np.array_equal(A, exp), N

    # BOTH bounds SSA. Swept including m==0 (the operand carries a runtime 0) and
    # m >= n (empty on runtime operands).
    @kernel
    def rng(A: i32[16], mb: i32[1], nb: i32[1], out: i32[16]):
        m: index = mb[0]
        n: index = nb[0]
        for i in range(m, n):
            out[i] = A[i] * 2

    for m, n in [(0, 16), (3, 12), (5, 6), (7, 7), (10, 3)]:
        out = np.zeros(16, np.int32)
        exp = np.zeros(16, np.int32)
        for i in range(m, n):
            exp[i] = A16[i] * 2
        _to_rtl(rng).cosim(
            A16.copy(), np.array([m], np.int32), np.array([n], np.int32), out
        )
        assert np.array_equal(out, exp), (m, n)

    # An SSA stride: the induction register advances 0, s, 2s, ...
    @kernel
    def rstep(A: i32[16], sb: i32[1], out: i32[16]):
        s: index = sb[0]
        for i in range(0, 16, s):
            out[i] = A[i] * 2

    for st in (1, 2, 3, 4):
        out = np.zeros(16, np.int32)
        exp = np.zeros(16, np.int32)
        for i in range(0, 16, st):
            exp[i] = A16[i] * 2
        _to_rtl(rstep).cosim(A16.copy(), np.array([st], np.int32), out)
        assert np.array_equal(out, exp), st

    # A zero-trip run RE-RUN by an enclosing loop (the CSR empty-row shape).
    # `done` is a latched level the container completes on the rising edge of, so
    # an empty run must let that level fall to 0 before rising again: completing
    # on `start` itself would hold it high from the previous (non-empty)
    # iteration and the container would wait forever for an edge that never
    # comes. Only the FIRST run starts with done already 0, so sweep where the
    # empty row sits -- last, first and interior take different paths.
    @kernel
    def rows(ptr: i32[4], out: i32[3]):
        for r in range(3):
            b: index = ptr[r]
            e: index = ptr[r + 1]
            for j in range(b, e):
                out[r] += 1

    for ptr in ([0, 2, 4, 6], [0, 2, 2, 4], [0, 0, 2, 4], [0, 2, 4, 4], [0, 0, 0, 0]):
        out = np.zeros(3, np.int32)
        _to_rtl(rows).cosim(np.array(ptr, np.int32), out)
        assert np.array_equal(out, np.diff(np.array(ptr, np.int32))), ptr


# --- Intra-iteration memory dependence (loop-independent, dist 0) ----------


def test_intra_iteration_dependence():
    """A loop-independent (distance-0) conflict between different subscripts is
    ordered within the iteration; provably disjoint subscripts are not, so the
    path does not degenerate into blanket same-array serialization."""

    @kernel
    def alias(A: f32[64], C: f32[32]):
        for i in range(32):
            A[2 * i] = 1.0  # write A[2i]
            C[i] = A[i]  # read A[i] -- aliases the write only at i == 0

    # The accesses land on the same element only at i == 0. The pair also carries
    # a loop-carried edge (i >= 1); keeping the tightest (dist 0) distance is what
    # preserves the same-iteration ordering.
    loop = _sched(alias).cyclic()[0]
    assert loop.ii == 1  # a dist-0 edge orders within the iteration, no recurrence
    assert loop.op("store").t < loop.op("load").t

    @kernel
    def disjoint(A: f32[64], C: f32[32]):
        for i in range(32):
            A[2 * i] = 1.0  # write even indices
            C[i] = A[2 * i + 1]  # read odd indices -- never the same element

    loop = _sched(disjoint).cyclic()[0]
    assert loop.op("store").t == 0
    assert loop.op("load").t == 0


# --- while loops -----------------------------------------------------------


def test_while_scheduling():
    """A counted while is raised to a for and schedules identically to one; a
    data-dependent while stays conditional, scheduled as a flushing pipeline with
    its trip -- and therefore latency -- left unknown."""

    @kernel
    def wc(A: i32[128], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while i < 128:
            s = s + A[i]
            i = i + 1
        out[0] = s

    @kernel
    def fc(A: i32[128], out: i32[1]):
        s: i32 = 0
        for i in range(128):
            s = s + A[i]
        out[0] = s

    w = _sched(wc).cyclic()[0]
    f = _sched(fc).cyclic()[0]
    # Raised to a constant-bound for, so the schedule matches `range(128)`
    # exactly -- same II, length, and (non-bound) latency -- and is not
    # conditional (no flushing controller).
    assert (w.ii, w.length, w.latency) == (f.ii, f.length, f.latency)
    assert not w.conditional and not w.latency_is_bound

    @kernel
    def wr(n0: i32, out: i32[1]):
        x: i32 = n0
        c: i32 = 0
        while x > 1:
            x = x - 1
            c = c + 1
        out[0] = c

    mod = _to_rtl(wr)
    loop = mod.schedule().cyclic()[0]
    assert loop.conditional is True
    assert loop.latency is None
    assert "dcp.condition" in mod.dcp  # reified while terminator


def test_while_flushing_pipeline_cosim():
    # The flushing pipeline emitted end-to-end: `running` gated by the exit
    # condition, each loop-carried iter-arg frozen into a survivor register at
    # exit, and the sibling store reading the frozen count. `x > 1` runs x-1
    # steps, so c = max(0, n0-1) -- including the zero-iteration case (n0<=1).
    @kernel
    def wr(n0: i32, out: i32[1]):
        x: i32 = n0
        c: i32 = 0
        while x > 1:
            x = x - 1
            c = c + 1
        out[0] = c

    mod = _to_rtl(wr)
    for n0 in (1, 2, 3, 7, 20):
        out = np.zeros(1, np.int32)
        r = mod.cosim(np.int32(n0), out)
        assert out[0] == max(0, n0 - 1)
        assert r.cycles > 0


def test_while_two_carried_accumulate_cosim():
    # A while carrying TWO recurrences whose result depends on both: acc folds x
    # while x counts down, so the frozen `acc` survivor is the triangular sum.
    @kernel
    def wacc(n0: i32, out: i32[1]):
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + x
            x = x - 1
        out[0] = acc

    mod = _to_rtl(wacc)
    for n0 in (0, 1, 5, 9):
        out = np.zeros(1, np.int32)
        mod.cosim(np.int32(n0), out)
        assert out[0] == n0 * (n0 + 1) // 2


def test_while_multistage_flush_cosim():
    # A store-less while whose *body* spans two stages (the `A[x-1]` load pushes
    # `next_acc` to stage 1) but whose condition `x > 0` is combinational. The
    # flushing pipeline drains the deeper survivor: `acc` advances one cycle after
    # each issue, and the exit is delayed to match, so the frozen `acc` is the
    # correct sum. Distinct from a memory-*dependent* condition (still deferred).
    N = 64

    @kernel
    def wsum(n0: i32, A: i32[N], out: i32[1]):
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + A[x - 1]
            x = x - 1
        out[0] = acc

    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    mod = _to_rtl(wsum)
    for n0 in (0, 1, 4, 10, 25):
        out = np.zeros(1, np.int32)
        mod.cosim(np.int32(n0), A, out)
        assert out[0] == int(A[:n0].sum())


def test_while_in_loop_store_cosim():
    # A leaf flushing-while that writes memory in its body. The doomed exit
    # iteration is issued but must commit nothing: emitAccesses gates each store's
    # write-enable by the continue-condition (`issue & cond`), the same rule the
    # loop-carried survivors follow. Covers a single-stage store, a multi-stage
    # store fed by an in-loop carried scalar (deeper drain), and the zero-trip
    # case (no write). Unwritten output elements read back as the memory init (0).
    N = 32

    @kernel
    def wstore(A: i32[N], B: i32[N], n0: i32):  # write-once per iteration
        x: i32 = n0
        while x > 0:
            B[x - 1] = A[x - 1] * 2
            x = x - 1

    @kernel
    def wscan(A: i32[N], B: i32[N], n0: i32):  # store the running prefix sum
        x: i32 = n0
        acc: i32 = 0
        while x > 0:
            acc = acc + A[x - 1]
            B[x - 1] = acc
            x = x - 1

    ma, mb = _to_rtl(wstore), _to_rtl(wscan)
    assert ma.schedule().cyclic()[0].conditional and "dcp.condition" in ma.dcp
    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    for n0 in (0, 1, 7, N):
        B = np.zeros(N, np.int32)
        ma.cosim(A, B, np.int32(n0))
        gold = np.zeros(N, np.int32)
        gold[:n0] = A[:n0] * 2
        assert np.array_equal(B, gold)

        B = np.zeros(N, np.int32)
        mb.cosim(A, B, np.int32(n0))
        gold = np.zeros(N, np.int32)
        gold[:n0] = np.cumsum(A[:n0][::-1])[::-1]  # acc counts x down from n0
        assert np.array_equal(B, gold)


def test_while_mem_condition_cosim():
    """A `while` loop whose continue-condition reads memory (`A[i] != key`): the
    loop index advances until the searched element is found, and the loop-carried
    value is read after the loop. Covers a single-value carry, a two-value carry
    (the index and a step counter), and a zero-iteration exit (the condition false
    on entry)."""
    A = np.arange(16, dtype=np.int32)  # A[i] == i, so the found index equals key

    @kernel
    def linsearch(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        while A[i] != key:
            i = i + 1
        out[0] = i

    out = np.zeros(1, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(11), out)
    assert out[0] == 11

    @kernel
    def search_steps(A: i32[16], key: i32, out: i32[1]):
        i: i32 = 0
        c: i32 = 0
        while A[i] != key:
            i = i + 1
            c = c + 1
        out[0] = c

    out = np.zeros(1, np.int32)
    _to_rtl(search_steps).cosim(A, np.int32(9), out)
    assert out[0] == 9

    # A[0] == key: the condition is false on entry, so the body never runs and the
    # carried index holds its initial value.
    out = np.full(1, 999, np.int32)
    _to_rtl(linsearch).cosim(A, np.int32(0), out)
    assert out[0] == 0


def test_while_mem_condition_shared_array_cosim():
    # A while loop that reads the same array in BOTH its continue-condition
    # (`A[i] > 0`) and its body (`s += A[i]`). Each access is a distinct memory
    # read, so the condition and the body do not contend for a port.
    @kernel
    def wmem(A: i32[16], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while A[i] > 0:
            s = s + A[i]
            i = i + 1
        out[0] = s

    A = np.array([5, 3, 8, 2, 0] + [9] * 11, dtype=np.int32)  # sentinel 0 at idx 4
    out = np.zeros(1, np.int32)
    _to_rtl(wmem).cosim(A, out)
    assert out[0] == 5 + 3 + 8 + 2  # sum until A[4] == 0 stops the loop


def test_while_ip_condition_cosim():
    """A `while` whose continue-condition is a multi-cycle floating-point
    operation rather than a memory read. The loop iterates until the float
    condition settles false; the body advances a float-carried value. Covers a
    single float comparison (`r > tol`) and a float subtraction feeding a
    comparison (`x - b > 0`), the latter a multi-stage condition cone. The
    condition is not settled in the issue cycle, so the loop runs sequentially
    (a conditional region) rather than as a flushing pipeline."""

    @kernel
    def fconverge(x: f32, tol: f32, out: f32[1]):
        r: f32 = x
        while r > tol:
            r = r * 0.5
        out[0] = r

    mod = _to_rtl(fconverge)
    assert mod.schedule().cyclic()[0].conditional
    assert "hw.module.extern @fcmp" in mod.mlir

    def gold_halve(x, tol):
        r = np.float32(x)
        while r > np.float32(tol):
            r = np.float32(r * np.float32(0.5))
        return r

    for x, tol in [(100.0, 1.0), (7.0, 1.0), (0.5, 1.0)]:  # last exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(x), np.float32(tol), out)
        assert out[0] == gold_halve(x, tol)

    @kernel
    def fcountdown(a: f32, b: f32, out: f32[1]):
        x: f32 = a
        while x - b > 0.0:
            x = x - 1.0
        out[0] = x

    mod = _to_rtl(fcountdown)
    assert mod.schedule().cyclic()[0].conditional

    def gold_count(a, b):
        x = np.float32(a)
        while np.float32(x - np.float32(b)) > np.float32(0.0):
            x = np.float32(x - np.float32(1.0))
        return x

    for a, b in [(10.0, 2.5), (5.0, 5.0), (3.0, 0.0)]:  # middle exits on entry
        out = np.zeros(1, np.float32)
        mod.cosim(np.float32(a), np.float32(b), out)
        assert out[0] == gold_count(a, b)


def test_nested_while_cosim():
    # A sequential-wrapper while (outer `s`) around a flushing-pipeline while
    # (inner `t`), carrying a cross-region accumulator `total`. The outer while is
    # a conditional container: its iter-args are survivor registers advanced by
    # the children's results, the raw `s > 0` condition is evaluated over those
    # registers, and the children re-run each outer iteration. total ends as
    # sum_{s=1..N} sum_{t=1..s} A[t-1].
    N = 8

    @kernel
    def nested(A: i32[N], out: i32[1]):
        total: i32 = 0
        s: i32 = N
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        out[0] = total

    A = (np.arange(N, dtype=np.int32) * 3 + 1) & 0xFF
    expected = sum(int(A[:s].sum()) for s in range(1, N + 1))
    out = np.zeros(1, np.int32)
    _to_rtl(nested).cosim(A, out)
    assert out[0] == expected


# --- scalar function returns -----------------------------------------------


def test_scalar_return_cosim():
    """A scalar result is an output port driven by the returning region's
    survivor, sampled at `done` and surfaced on CosimResult.result -- for an int,
    a float (decoded from its bit pattern), and a conditional container's frozen
    iter-arg."""
    N = 16

    @kernel
    def ssum(A: i32[N]) -> i32:
        s: i32 = 0
        for i in range(N):
            s = s + A[i]
        return s

    A = (np.arange(N, dtype=np.int32) * 7 + 3) & 0xFF
    r = _to_rtl(ssum).cosim(A)
    assert r.result == int(A.sum())

    @kernel
    def fsum(A: f32[8]) -> f32:
        s: f32 = 0.0
        for i in range(8):
            s = s + A[i]
        return s

    Af = np.arange(8, dtype=np.float32) + 1.0
    r = _to_rtl(fsum).cosim(Af)
    assert np.isclose(float(r.result), float(Af.sum()))

    # The returned survivor is the conditional container's frozen iter-arg.
    @kernel
    def nested(A: i32[8]) -> i32:
        total: i32 = 0
        s: i32 = 8
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        return total

    A8i = (np.arange(8, dtype=np.int32) * 3 + 1) & 0xFF
    expected = sum(int(A8i[:s].sum()) for s in range(1, 9))
    r = _to_rtl(nested).cosim(A8i)
    assert r.result == expected


def test_while_with_nested_while():
    # Two decreasing (hence un-raised) whiles nested. The inner while's straight-
    # line body schedules as a flushing pipeline; the outer while's body is
    # decomposed around it and the outer runs sequentially. Exercises the
    # nested-loop-in-while decomposition recursing through a while child -- the
    # shape of MachSuite fft/strided.
    N = 64

    @kernel
    def nested_while(A: i32[N]) -> i32:
        total: i32 = 0
        s: i32 = N
        while s > 0:
            t: i32 = s
            while t > 0:
                total += A[t - 1]
                t -= 1
            s -= 1
        return total

    mod = _to_rtl(nested_while)
    res = mod.schedule()
    assert len(res.cyclic()) >= 1  # the inner while pipelines
    assert res.func("nested_while").latency is None  # data-dependent trips
    # Both whiles close: the inner -> flushing pipeline, the outer -> sequential
    # while dcp.pipeline wrapping it. No raw scf.while; two dcp.condition ends.
    assert "scf.while" not in mod.dcp
    assert mod.dcp.count("dcp.condition") == 2


# --- allo.assume hints -----------------------------------------------------


def test_assume_hints():
    """`allo.assume` feeds the scheduler facts the polyhedral test cannot prove:
    a bound on a dynamic trip, and the absence of an inter-iteration dependence.
    `allo.grid` carries the same independence guarantee implicitly."""

    @kernel
    def k(A: i32[128], out: i32[1], n: index):
        allo.assume(n < 100)
        s: i32 = 0
        for i in range(n):
            s = s + A[i]
        out[0] = s

    # A bounded dynamic trip reports a worst-case latency flagged as a bound
    # rather than deferring it.
    loop = _sched(k).cyclic()[0]
    assert loop.latency > 0
    assert loop.latency_is_bound is True

    def hist(hint):
        @kernel
        def h(idx: i32[128], acc: i32[64]):
            for i in range(128):
                if hint:
                    allo.assume(acc, i, type="inter")
                acc[idx[i]] = acc[idx[i]] + 1

        return h

    # Without the hint the aliasing histogram update keeps a conservative
    # loop-carried edge; asserting no inter-iteration dependence prunes it.
    assert _sched(hist(False)).cyclic()[0].ii == 2
    assert _sched(hist(True)).cyclic()[0].ii == 1

    # A grid()'s independence guarantee lowers to `assume.nodep` on the written
    # array, dropping the conservative back edge on a non-affine aliasing write --
    # whereas the identical body in a sequential range() nest keeps it.
    N = 64

    @kernel
    def par_scatter(val: f32[N, N], out: f32[N]):
        for i, j in allo.grid(N, N):
            out[i * j] = out[i * j] + val[i, j]

    @kernel
    def seq_scatter(val: f32[N, N], out: f32[N]):
        for i in range(N):
            for j in range(N):
                out[i * j] = out[i * j] + val[i, j]

    assert _iis(_sched(par_scatter).cyclic()) == [1]
    assert _iis(_sched(seq_scatter).cyclic()) == [MEM_REDUCE_II]


# --- Pipeline directives ---------------------------------------------------


def test_pipeline_target_ii_raises_ii():
    def vadd():
        @kernel
        def v(A: i32[32], B: i32[32], C: i32[32]):
            for i in range(32, name="i"):
                C[i] = A[i] + B[i]

        return v

    assert _sched(vadd()).cyclic()[0].ii == 1  # natural minimum

    s = vadd().schedule()
    s.pipeline("i", ii=3)
    mod = s.export("rtl")
    assert mod.schedule().cyclic()[0].ii == 3  # target honored as a floor


def test_pipeline_disabled_runs_sequentially():
    def mac():
        @kernel
        def m(A: i32[8], B: i32[8], out: i32[8]):
            for i in range(8, name="i"):
                out[i] = A[i] * B[i]

        return m

    pl = _sched(mac()).cyclic()[0]

    s = mac().schedule()
    s.pipeline("i", ii=-1)
    mod = s.export("rtl")
    npl = mod.schedule().cyclic()[0]

    assert npl.ii == npl.length  # no overlap: II = body length
    assert npl.latency == 8 * npl.length  # trip * depth
    assert pl.ii < npl.ii  # pipelining packs iterations tighter
    assert pl.latency < npl.latency


def test_multiregion_latency_matches_cosim():
    # The whole-kernel `dcp.latency` must equal the cycle the emitter's `done`
    # rises (what cosim counts from `start`), so a parent that composes this
    # kernel and treats it as a fixed-latency node captures its result at the
    # right cycle. A cross-region *survivor* is handed to the next region through
    # a capture register -- one cycle `perInvocationLatency` (a datapath depth)
    # does not count -- while a store-terminated region hands off through memory
    # and adds none. This pins both cases and guards the reify latency model
    # against drift from the emitter's `done` timing.
    def survivors():  # two survivor hand-offs (s, then t) + a store region
        @kernel
        def survivors(A: i32[16], out: i32[16]):
            s: i32 = 0
            for i in range(16):
                s = s + A[i]
            t: i32 = 0
            for i in range(16):
                t = t + A[i] * s
            for i in range(16):
                out[i] = A[i] + t

        return survivors

    def stores():  # three store-terminated regions, no survivor register
        @kernel
        def stores(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
            for i in range(16):
                B[i] = A[i] + 1
            for i in range(16):
                C[i] = B[i] * 2
            for i in range(16):
                out[i] = C[i] + 3

        return stores

    lat = _latency(survivors())
    assert lat is not None  # every trip is static
    out = np.zeros(16, np.int32)
    r = _to_rtl(survivors()).cosim(A16, out)
    assert r.cycles == lat  # survivor registers counted -> no drift

    lat = _latency(stores())
    assert lat is not None
    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = _to_rtl(stores()).cosim(A16, B, C, out)
    assert r.cycles == lat  # store hand-off adds nothing -> still exact
    assert np.array_equal(out, (A16 + 1) * 2 + 3)


def test_independent_siblings_run_concurrently_cosim():
    # two sibling sweeps on DISJOINT arrays (no shared memref, no
    # survivor) have no dependence, so the composer starts them together instead
    # of serializing
    @kernel
    def indep(A: i32[64], B: i32[64], C: i32[64], D: i32[64]):
        for i in range(64):
            C[i] = A[i] + 1
        for i in range(64):
            D[i] = B[i] * 2

    serial = _latency(indep)  # the reifier still sums the two regions serially
    assert serial is not None
    A = np.arange(64, dtype=np.int32)
    B = np.arange(64, dtype=np.int32) + 100
    C = np.zeros(64, np.int32)
    D = np.zeros(64, np.int32)
    r = _to_rtl(indep).cosim(A, B, C, D)
    assert np.array_equal(C, A + 1)
    assert np.array_equal(D, B * 2)
    # The two ~64-cycle sweeps overlap: the kernel completes in about one sweep,
    # comfortably under half again the serial sum (observed ~69 vs 135).
    assert r.cycles < serial


def test_pipeline_directive_preserves_result_cosim():
    # A pipeline directive changes the schedule (II), not the result: a forced
    # II=3 pipeline and a pipeline-off (sequential) loop both still compute the
    # elementwise op. Pins that the directive knob is correctness-neutral.
    @kernel
    def v(A: i32[32], B: i32[32], C: i32[32]):
        for i in range(32, name="i"):
            C[i] = A[i] + B[i]

    A = np.arange(32, dtype=np.int32)
    B = np.arange(32, dtype=np.int32) * 3
    s = v.schedule()
    s.pipeline("i", ii=3)  # forced II above the natural minimum
    C = np.zeros(32, np.int32)
    s.export("rtl").cosim(A, B, C)
    assert np.array_equal(C, A + B)

    @kernel
    def m(A: i32[8], B: i32[8], out: i32[8]):
        for i in range(8, name="i"):
            out[i] = A[i] * B[i]

    A8 = np.arange(8, dtype=np.int32) + 1
    B8 = np.arange(8, dtype=np.int32) + 2
    s = m.schedule()
    s.pipeline("i", ii=-1)  # pipelining disabled -> sequential
    out = np.zeros(8, np.int32)
    s.export("rtl").cosim(A8, B8, out)
    assert np.array_equal(out, A8 * B8)


# --- Phase B: pipelined imperfect nests (fused-overlap scheduling) ----------


def test_phase_b_recurrence_under_scf_ancestor():
    # An imperfect pipelined nest schedules via Phase B (the outer loop pipelined
    # over its inner loop as a fixed-latency node) when `unroll_under_pipeline`
    # is off. Here the pipelined level `j` sits under a *runtime-bounded* outer
    # loop `i` -- a memory-loaded trip makes `i` an `scf.for`, so the nest MIXES
    # scf.for and affine.for. The level carries a memory recurrence (`acc[0]`
    # divided every iteration). The affine dependence components index only the
    # affine loop `j` (an scf iv is no affine dim), so attributing the recurrence
    # to `j` by positional nesting depth would over-count past the scf ancestor
    # and silently drop the edge -- an unsound II = the 4-cycle inner-loop
    # occupancy. Matching the level to its component by identity keeps the edge,
    # so the II is the recurrence bound read -> div -> write.
    RECUR_II = MEM + FDIV + MEM  # 14

    def mixed():
        @kernel
        def mix(A: f32[64, 64], acc: f32[1], B: f32[64, 2], nbuf: index[1]):
            n: index = nbuf[0]  # memory-loaded bound => scf.for ancestor
            allo.assume(n < 64)
            for i in range(n):
                for j in range(64, name="j"):  # pipelined affine level
                    acc[0] = acc[0] / A[i, j]  # level-carried recurrence
                    for p in range(2):  # inner loop kept rolled (Phase B node)
                        B[j, p] = A[i, j]

        return mix

    s = mixed().schedule()
    s.pipeline("j")
    mod = s.export("rtl", unroll_under_pipeline=False)
    # Two cyclic regions: the pipelined level `j` and its inner loop `p` (II=1).
    # The level's II is the recurrence bound; without the identity-matched
    # projection it would collapse to the 4-cycle inner-loop occupancy.
    iis = _iis(mod.schedule().cyclic())
    assert max(iis) == RECUR_II


def test_phase_b_outer_level_cuts_combinational_chain():
    # The Phase B level problem is timing-aware: a combinational chain among the
    # outer level's loose ops is cut at the cycle boundary, exactly as in the leaf
    # body (an inner-loop node is a registered boundary, so it never joins the
    # chain). Here the level carries a 3-deep int-add chain (3 x 1.2ns = 3.6ns);
    # extra stores of the intermediates keep it from being reassociated into a
    # tree. At a tight clock the chain spans two cycles; at a slack clock, one.
    def level_add_cycles(freq_mhz):
        @kernel
        def chain(
            A: i32[8, 8],
            b: i32[8],
            c: i32[8],
            d: i32[8],
            o0: i32[8],
            o1: i32[8],
            o2: i32[8],
        ):
            for i in range(8, name="i"):  # pipelined imperfect nest -> Phase B
                s0: i32 = b[i] + c[i]
                s1: i32 = s0 + d[i]
                s2: i32 = s1 + b[i]  # 3-deep combinational chain at the level
                o0[i] = s0  # extra uses -> not a reassociable single-use chain
                o1[i] = s1
                for p in range(8, name="p"):  # inner loop node (registered)
                    A[i, p] = s2
                o2[i] = s2

        s = chain.schedule()
        s.pipeline("i")
        mod = s.export("rtl", unroll_under_pipeline=False, freq_mhz=freq_mhz)
        res = mod.schedule()
        # The level region is the pipelined outer loop holding the 3-add chain.
        level = next(
            r for r in res.cyclic() if sum(o.kind == "addi" for o in r.ops) == 3
        )
        return len({o.t for o in level.ops if o.kind == "addi"})

    assert level_add_cycles(1000 / 3.0) == 2  # 3.6ns chain cut by a 3.0ns clock
    assert level_add_cycles(1000 / 6.0) == 1  # fits in one 6.0ns cycle -> not cut


# --- Reduction restructuring (rotate / reassociate) ------------------------


def test_rotate_reduction_scales_ii():
    """Rotating a float reduction across N accumulators turns its distance-1
    recurrence (II == add latency) into a distance-N one: II == ceil(L/N)."""

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


def test_reassociate_int_reduction_recurrence():
    """Integer reductions rebalance unconditionally (integer arithmetic is exactly
    associative mod 2^w), cutting an unrolled chain's recurrence to one operator."""

    # Unrolling threads the carried accumulator through four widened multiplies;
    # folding it in last makes the recurrence one (widened, combinational) multiply
    # rather than a chain of four. Integer multiply is combinational, so the
    # recurrence II is that one multiply's delay (2 cycles here), not 4x it.
    @kernel
    def red(x: i32[32]) -> i32:
        acc: i32 = 1
        for i in range(32, name="i"):
            acc *= x[i]
        return acc

    s = red.schedule()
    s.unroll("i", factor=4)
    mod = s.export("rtl")
    assert mod.schedule().cyclic()[0].ii == 2


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


# --- Timing model (chaining) -----------------------------------------------


def test_chaining_inserts_register():
    def chain():
        @kernel
        def c(A: i32[8], out: i32[8]):
            for i in range(8):
                x: i32 = A[i] + A[i]
                y: i32 = x + A[i]
                z: i32 = y + A[i]
                out[i] = z + A[i]

        return c

    # Four combinational int adds (1.2 ns each) cannot fit one 3.33 ns cycle, so
    # the chaining scheduler splits the chain across cycles -- more register
    # stages than under a huge cycle time, where the whole chain settles in one.
    tight = _sched(chain()).cyclic()[0]
    loose = _sched(chain(), freq_mhz=1.0).cyclic()[0]  # a 1000ns cycle
    assert tight.last_t() > loose.last_t()
