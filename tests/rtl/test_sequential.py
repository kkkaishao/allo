# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequential (non-dataflow) sub-kernel call composition: chaining through shared arrays, mixed containers, concurrency inference, and nested composition."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32, index

sys.path.insert(0, os.path.dirname(__file__))
from _common import _latency, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
B16 = (np.arange(16, dtype=np.int32) * 5 + 3) & 0xFF


# --- Chained calls through shared storage ------------------------------------


# Two plain sub-kernels chained through a shared boundary array: the composed
# latency is the sum of the child latencies, both reported and actual.
def test_sequential_two_kernel_shared_array():
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
    assert _latency(seq_top) == l1 + l2

    B = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = _to_rtl(seq_top).cosim(A16, B, out)
    assert np.array_equal(out, (A16 + 1) * 2)  # out = child2(child1(A))
    assert r.cycles == l1 + l2  # serial: the children do not overlap


# Two sub-kernels chained through a container-LOCAL buffer: it lowers to an
# on-chip seq.hlmem rather than a top port, serialized by the RAW dependence.
def test_sequential_internal_buffer_shared():
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


# A container whose body is a loop over a sub-kernel call: one child instance
# is instantiated once and fired N times, a counter driving its index.
def test_loop_over_calls():
    @kernel
    def lc_step(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def lc_top(A: i32[16], B: i32[16]):
        for i in range(16):
            lc_step(A, B, i)  # invoke the sub-kernel 16 times

    mod = _to_rtl(lc_top)
    # The loop-over-calls container lowers to the leaf (its call reifies to a
    # dcp.instance), one child instance fired N times by the counter.
    assert "dcp.instance" in mod.dcp
    # The loop counter driving the child's index, labelled with the source IV.
    assert "%i = seq.compreg" in mod.mlir
    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.zeros(16, np.int32)
    mod.cosim(A, B)
    assert np.array_equal(B, A * 2 + 1)


# A zero-trip loop over a sub-kernel call must complete without firing the
# child (regression: this used to hang on cosim's watchdog).
def test_zero_trip_loop_over_calls():
    @kernel
    def zlc_step(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2 + 1

    @kernel
    def zlc_top(A: i32[16], B: i32[16]):
        for i in range(0):
            zlc_step(A, B, i)

    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.full(16, 9, np.int32)
    _to_rtl(zlc_top).cosim(A, B)
    # A write-only argument is not preloaded, so the backing array starts at
    # zero; the child never fired, so it stays there -- not A * 2 + 1.
    assert np.array_equal(B, np.zeros(16, np.int32))
    assert not np.array_equal(B, A * 2 + 1)


# --- Concurrency inference between independent calls -------------------------


# Two sub-kernels with disjoint memory footprints get no ordering edge: both
# fire at cycle 0, and the composed latency is the max rather than the sum.
def test_sequential_independent_kernels():
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


# Interprocedural per-argument footprint analysis: disjoint writers and pure
# readers overlap, a genuine WAW serializes, and a container-local buffer with
# two writers + one reader gets one port group per accessor, never a mux.
def test_concurrent_shared_array_access():
    # Two sub-kernels WRITING one shared array in disjoint slices: the
    # per-argument callee footprint proves they cannot collide, so no edge is
    # added and each access gets its own port group (never a shared mux).
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
    wr = [w[0] for w in mod.interfaces["cw_top"]["writes"]]
    assert [w["base"] for w in wr] == ["B_wr0", "B_wr1"]
    assert {w["arg"] for w in wr} == {1}
    B = np.zeros(16, np.int32)
    r = mod.cosim(A16, B)
    assert np.array_equal(B, np.concatenate([A16[:8] + 1, A16[8:] * 2]))
    assert r.cycles == max(l1, l2)  # disjoint slices: the writers overlap

    # Two sub-kernels READING one shared input array: neither writes it, so
    # there is no ordering constraint at all.
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

    # The dual and the soundness guard: two writers of the SAME elements are a
    # real WAW, so the scheduler orders them and they do not overlap.
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
    assert np.array_equal(ob, A16 * 2)  # the later writer wins: they ran in order
    assert r.cycles == ol1 + ol2  # a real WAW: the writers do NOT overlap

    # A container-local buffer filled by TWO children writing disjoint halves
    # concurrently, then read by a third. The reader conflicts with both
    # writers and so is ordered after both.
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
    assert r.cycles == max(lw1, lw2) + lrd  # the writers overlap; the reader waits


# Two pure-seq calls on disjoint arrays have no shared-memref dependence, so
# the leaf starts them concurrently and finishes in one child's latency.
def test_independent_calls_on_disjoint_arrays_overlap():
    @kernel
    def ov1(A: i32[16], oa: i32[16]):
        for i in range(16):
            oa[i] = A[i] + 1

    @kernel
    def ov2(B: i32[16], ob: i32[16]):
        for i in range(16):
            ob[i] = B[i] * 2

    @kernel
    def ov_top(A: i32[16], B: i32[16], oa: i32[16], ob: i32[16]):
        ov1(A, oa)
        ov2(B, ob)  # disjoint from ov1 -> overlaps it on the leaf

    rtl = _to_rtl(ov_top)
    assert "allo.dcp.instance" in rtl.dcp  # leaf CallUnit path (structural lock)
    l1, l2 = _latency(ov1), _latency(ov2)
    A = np.arange(16, dtype=np.int32)
    B = np.arange(16, dtype=np.int32) + 100
    oa = np.zeros(16, np.int32)
    ob = np.zeros(16, np.int32)
    r = rtl.cosim(A, B, oa, ob)
    assert np.array_equal(oa, A + 1)
    assert np.array_equal(ob, B * 2)
    assert r.cycles == max(l1, l2)  # concurrent, not l1 + l2


# --- Nested composition -------------------------------------------------------


# Seq-in-seq: a container whose first child is itself a container. The parent
# places the following sibling after the WHOLE inner container's latency.
def test_nested_sequential_composition():
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
        nt_leaf3(C, out)  # reads the inner container's output

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


# Seq-in-seq on the leaf: the inner container instantiates as a plain CallUnit,
# wired exactly like any leaf, since its interface is memory-port based.
def test_nested_container_instantiates_as_a_plain_call():
    @kernel
    def r1b_l1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    @kernel
    def r1b_l2(B: i32[16], C: i32[16]):
        for i in range(16):
            C[i] = B[i] * 2

    @kernel
    def r1b_mid(A: i32[16], B: i32[16], C: i32[16]):
        r1b_l1(A, B)  # B = A + 1
        r1b_l2(B, C)  # C = (A + 1) * 2

    @kernel
    def r1b_l3(C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = C[i] + 3

    @kernel
    def r1b_top(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        r1b_mid(A, B, C)  # a nested CONTAINER child (CountedStatic)
        r1b_l3(C, out)  # reads the container's output -> serial on the leaf

    rtl = _to_rtl(r1b_top)
    # The container child instantiates in r1b_top's OWN body (the "(" after
    # r1b_mid distinguishes the outer invoke from the inner r1b_mid.r1b_l* ones).
    assert "allo.dcp.instance @r1b_top.r1b_mid(" in rtl.dcp
    assert "allo.dcp.instance @r1b_top.r1b_l3(" in rtl.dcp
    A = np.arange(16, dtype=np.int32)
    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    out = np.zeros(16, np.int32)
    r = rtl.cosim(A, B, C, out)
    assert r.cycles > 0
    assert np.array_equal(out, (A + 1) * 2 + 3)


# --- Mixed containers (loose datapath beside sub-kernel calls) ---------------


# A pure SERIAL call graph (no loose datapath) still lowers via the leaf
# CallUnit path: both children instantiate in the container's own module.
def test_pure_sequential_still_emits():
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

    rtl = _to_rtl(seq_top)  # must not raise
    assert "allo.dcp.instance" in rtl.dcp  # leaf CallUnit path (structural lock)
    assert rtl.mlir.count("hw.instance") >= 2  # both children instantiated


# A container mixing its own datapath with a call mastering only
# container-local buffers: the call instantiates in the container's own module.
def test_mixed_container_internal_buffer_call():
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


# A loose datapath region interleaved between two calls schedules in program
# order against the calls it depends on.
def test_mixed_container_loose_region_between_calls():
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


# A loose region that writes the boundary output after a call must not be
# silently dropped by the leaf CallUnit path.
def test_loose_region_after_a_call_writes_boundary_output():
    @kernel
    def mcb1(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1  # A boundary read, B internal write

    @kernel
    def mcb_top(A: i32[16], out: i32[16]):
        B: i32[16]
        mcb1(A, B)
        for i in range(16):  # loose region writing the top output (parent access)
            out[i] = B[i] + 5

    A = np.arange(16, dtype=np.int32)
    out = np.zeros(16, dtype=np.int32)
    r = _to_rtl(mcb_top).cosim(A, out)
    assert r.cycles > 0
    assert np.array_equal(out, (A + 1) + 5)


# Two ADJACENT calls with no intervening loose op reify into ONE region; they
# still serialize, and the boundary arg they both read is wired to two ports.
def test_adjacent_calls_with_no_loose_op_between_them():
    @kernel
    def cc1(x: i32[8], p: i32[8], q: i32[8]):
        for i in range(8):
            p[i] = x[i] + 1  # x read twice (two ports) -> internal p, q
            q[i] = x[i] + 2

    @kernel
    def cc2(p: i32[8], q: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = p[i] * 10 + q[i]  # reads internals cc1 wrote

    @kernel
    def cc_top(x: i32[8], out: i32[8]):
        p: i32[8]
        q: i32[8]
        d: i32[8]
        for i in range(8):
            d[i] = 0  # loose region -> mixed container
        cc1(x, p, q)  # adjacent calls, one region: cc1 must finish before cc2
        cc2(p, q, out)

    x = np.arange(8, dtype=np.int32) + 1
    out = np.zeros(8, dtype=np.int32)
    r = _to_rtl(cc_top).cosim(x, out)
    assert r.cycles > 0
    assert np.array_equal(out, (x + 1) * 10 + (x + 2))


# A boundary array write-mastered by two serial children time-shares one write
# port through a self-gated priority mux, so the idle master never writes.
def test_mixed_container_shared_boundary_serial_masters():
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


# --- Loop-body call sequencing ------------------------------------------------

A_LOOP16 = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F


# The second child reads what the first wrote in the same iteration: the
# consumer is sequenced on the producer's real done, not a static offset.
def test_two_calls_in_one_loop_body_chained_through_a_buffer():
    @kernel
    def ch_a(A: i32[16], T: i32[16], i: index):
        T[i] = A[i] * 2

    @kernel
    def ch_b(T: i32[16], C: i32[16], i: index):
        C[i] = T[i] + 1

    @kernel
    def ch_top(A: i32[16], C: i32[16]):
        T: i32[16]
        for i in range(16):
            ch_a(A, T, i)
            ch_b(T, C, i)

    C = np.zeros(16, np.int32)
    _to_rtl(ch_top).cosim(A_LOOP16, C)
    assert np.array_equal(C, A_LOOP16 * 2 + 1)


# A call and unrelated arithmetic in one loop body: the loose store is the
# part a lone-call leaf controller would drop.
def test_call_beside_loose_compute_in_a_loop_body():
    @kernel
    def lc_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def lc_top(A: i32[16], B: i32[16], C: i32[16]):
        for i in range(16):
            lc_child(A, B, i)
            C[i] = A[i] + 1

    B = np.zeros(16, np.int32)
    C = np.zeros(16, np.int32)
    _to_rtl(lc_top).cosim(A_LOOP16, B, C)
    assert np.array_equal(B, A_LOOP16 * 2)
    assert np.array_equal(C, A_LOOP16 + 1)


# Loose work computes a scalar the call consumes, ordered before the call and
# crossing into it as a cross-region survivor.
def test_loose_compute_feeding_the_call_it_shares_a_body_with():
    @kernel
    def sf_child(A: i32[16], B: i32[16], i: index, k: i32):
        B[i] = A[i] * k

    @kernel
    def sf_top(A: i32[16], B: i32[16]):
        for i in range(16):
            k: i32 = A[i] + 1
            sf_child(A, B, i, k)

    B = np.zeros(16, np.int32)
    _to_rtl(sf_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, A_LOOP16 * (A_LOOP16 + 1))


# B[i] = f(A, i): storing a scalar-returning call's result forces the same
# loop-body decomposition as any other loose work sharing the body.
def test_loop_over_a_scalar_returning_call():
    @kernel
    def sr_child(A: i32[16], i: index) -> i32:
        v: i32 = A[i] * 2 + 5
        return v

    @kernel
    def sr_top(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = sr_child(A, i)

    B = np.zeros(16, np.int32)
    _to_rtl(sr_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, A_LOOP16 * 2 + 5)


# Two was the pair; three checks that the sequencing composes rather than
# special-casing a producer/consumer pair.
def test_three_calls_in_one_loop_body():
    @kernel
    def th_a(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def th_b(A: i32[16], C: i32[16], i: index):
        C[i] = A[i] + 1

    @kernel
    def th_c(A: i32[16], D: i32[16], i: index):
        D[i] = A[i] - 1

    @kernel
    def th_top(A: i32[16], B: i32[16], C: i32[16], D: i32[16]):
        for i in range(16):
            th_a(A, B, i)
            th_b(A, C, i)
            th_c(A, D, i)

    B, C, D = (np.zeros(16, np.int32) for _ in range(3))
    _to_rtl(th_top).cosim(A_LOOP16, B, C, D)
    assert np.array_equal(B, A_LOOP16 * 2)
    assert np.array_equal(C, A_LOOP16 + 1)
    assert np.array_equal(D, A_LOOP16 - 1)


# An if guarding a call: the call becomes a predicated child region rather
# than loose work inside a predicated span.
def test_call_guarded_by_an_if_inside_a_loop():
    @kernel
    def gi_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def gi_top(A: i32[16], B: i32[16]):
        for i in range(16):
            if A[i] > 20:
                gi_child(A, B, i)

    B = np.zeros(16, np.int32)
    _to_rtl(gi_top).cosim(A_LOOP16, B)
    assert np.array_equal(B, np.where(A_LOOP16 > 20, A_LOOP16 * 2, 0))


# A body that is exactly one call keeps the cheap leaf loop-over-calls
# controller (one flat dcp.pipeline holding the invoke). Structural only.
def test_a_lone_call_body_stays_on_the_leaf_controller():
    @kernel
    def lone_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def lone_top(A: i32[16], B: i32[16]):
        for i in range(16):
            lone_child(A, B, i)

    dcp = _to_rtl(lone_top).dcp
    top = dcp[dcp.index("func.func public @lone_top") :]
    top = top[: top.index("func.func private")]
    assert top.count("allo.dcp.pipeline") == 1
    assert "allo.dcp.sequential" not in top  # no child region: still a leaf


# A body with a call plus loose work decomposes into a container with
# sub-regions, so the loose work sequences against the call's real done.
# Structural only.
def test_a_mixed_call_body_becomes_a_container():
    @kernel
    def mx_child(A: i32[16], B: i32[16], i: index):
        B[i] = A[i] * 2

    @kernel
    def mx_top(A: i32[16], B: i32[16], C: i32[16]):
        for i in range(16):
            mx_child(A, B, i)
            C[i] = A[i] + 1

    dcp = _to_rtl(mx_top).dcp
    top = dcp[dcp.index("func.func public @mx_top") :]
    top = top[: top.index("func.func private")]
    # One outer pipeline wrapping a dcp.sequential that holds the invoke, plus
    # a second child region for the loose store.
    assert top.count("allo.dcp.pipeline") == 1
    assert top.count("allo.dcp.sequential") == 2
    call_region = top[: top.index("allo.dcp.instance")].rsplit(
        "allo.dcp.sequential", 1
    )[1]
    assert "dcp.store" not in call_region  # the call region holds only the call
