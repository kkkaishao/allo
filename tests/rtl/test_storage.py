# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-chip storage realization: banking, ROM-vs-RAM classification, multi-cycle access latency, and cross-region/container buffer identity."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import f32, i32, u8, index, Stateful, Stream
from allo.schedule import Schedule
from allo.backend.rtl.device import builtin_device, MemoryKind

sys.path.insert(0, os.path.dirname(__file__))
from _common import _sched, _to_rtl, _iis, FADD  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

N = 8
A8 = np.arange(1, 9, dtype=np.int32)
A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF


# --- banking ------------------------------------------------------------


def test_banked_internal_buffer():
    # A partitioned internal buffer splits into per-bank on-chip memories: a
    # statically-resolvable index routes to its bank directly, a runtime-varying
    # one gets a crossbar (read every bank + mux, write-enable demux).
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
    # loop counter), so the emitter builds the crossbar: bank (i & 1) / offset
    # (i >> 1), aligned with the 1-cycle read latency.
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
    # A partitioned argument array becomes one boundary interface per bank; the
    # cosim harness splits the numpy argument into cyclic bank slices, joining on
    # writeback. A runtime-varying bank crossbars over those interfaces.
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
    assert {"A_rd0_b0", "A_rd0_b1"} <= rbases
    assert {"out_wr0_b0", "out_wr0_b1"} <= wbases

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 + 1)


def test_banking_beyond_1d_pow2_cyclic():
    # Banking is decomposed in element space, not restricted to the 1-D
    # power-of-two cyclic case flat address arithmetic can express: covers a
    # BLOCK partition kind, a non-power-of-two cyclic FACTOR, and multi-dim
    # RANK, on both internal buffers and boundary arguments.

    # Kind: a BLOCK partition, internal. Reading at `15 - i` breaks index
    # symmetry, so a self-consistent but wrong bank select still scrambles.
    @kernel
    def blk(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1
        for i in range(16):
            out[i] = buf[15 - i] & 255

    s = blk.schedule()
    s.partition("buf", dim=1, kind=s.Block, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16[::-1] + 1) & 255)

    # Kind, boundary side: the manifest publishes the block decomposition and
    # the host shards the argument by it.
    @kernel
    def eblk(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] + 1

    s = eblk.schedule()
    s.partition("A", dim=1, kind=s.Block, factor=2)
    mod = s.export("rtl")
    rd = [r for acc in mod.interfaces[mod.top]["reads"] for r in acc]
    assert rd[0]["axes"] == [{"dim": 0, "factor": 2, "block": True}]
    assert rd[0]["shape"] == [16]
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 + 1)

    # Factor: cyclic by 3 (a divider, not a shift) over a length the factor does
    # not divide, so banks 1..2 each carry a padding slot.
    A10 = (np.arange(10, dtype=np.int32) * 3 + 1) & 0xFF

    @kernel
    def np2(A: i32[10], out: i32[10]):
        for i in range(10):
            out[i] = A[i] + 1

    s = np2.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=3)
    mod = s.export("rtl")
    out = np.zeros(10, np.int32)
    mod.cosim(A10, out)
    assert np.array_equal(out, A10 + 1)

    # Rank: a 2-D ARGUMENT, cyclic on the last dim, with an ODD row length, so
    # the element-space bank (`j % 2`) and a flat one (`(i*5 + j) % 2`) differ on
    # every odd row.
    A45 = ((np.arange(20, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 5)

    @kernel
    def ext2d(A: i32[4, 5], out: i32[4, 5]):
        for i in range(4):
            for j in range(5):
                out[i, j] = A[i, j] + 1

    s = ext2d.schedule()
    s.partition("A", dim=2, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    out = np.zeros((4, 5), np.int32)
    mod.cosim(A45, out)
    assert np.array_equal(out, A45 + 1)

    # Rank, internal + data-dependent: cyclic on dim 1 (ROWS) of a 2-D buffer.
    # The bank is the row parity `i % 2`, which a flat address cannot express.
    A48 = ((np.arange(32, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 8)

    @kernel
    def int2d(A: i32[4, 8], out: i32[4, 8]):
        buf: i32[4, 8]
        for i in range(4):
            for j in range(8):
                buf[i, j] = A[i, j] + 1
        for i in range(4):
            for j in range(8):
                out[i, j] = buf[3 - i, j] & 255

    s = int2d.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    out = np.zeros((4, 8), np.int32)
    mod.cosim(A48, out)
    assert np.array_equal(out, (A48[::-1, :] + 1) & 255)


def test_nested_banked_static_split():
    # A 2D nest accessing a cyclic-partitioned buffer on its inner (partitioned)
    # dim. `flatten-perfect-loops` must not coalesce the inner loop -- coalescing
    # would delinearize j and defeat static bank resolution, falling back to a
    # runtime crossbar. With the skip, buf banks statically (two per-bank
    # memories, no `_b<k>` crossbar).
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
    assert mod.mlir.count("= seq.hlmem") == 2
    assert "@buf_b" not in mod.mlir

    A = ((np.arange(32, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 8)
    out = np.zeros((4, 8), np.int32)
    mod.cosim(A, out)
    ref = A.copy()
    ref[:, 0::2] = (A[:, 0::2] + 1) & 255
    ref[:, 1::2] = (A[:, 1::2] + 100) & 255
    assert np.array_equal(out, ref)


def test_composed_banking():
    # Banking a COMPOSED array: a partition stated once where the array lives
    # (`propagate-partition`) reaches every callee parameter, so each child
    # emits a port group per bank and the container materializes one memory per
    # bank with no crossbar of its own. Covers a container-local buffer and a
    # container argument.
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
    assert "dcp.instance @cbi_top.cbi_prod(" in mod.dcp  # the leaf CallUnit path
    assert re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir) == [
        ("tmp_b0", "8"),
        ("tmp_b1", "8"),
    ]
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    # even lanes +1, odd lanes +100 -- a swapped bank route corrupts the golden
    assert np.array_equal(out, np.where(np.arange(16) % 2 == 0, A + 1, A + 100) & 255)

    # The boundary dual: a partitioned container ARGUMENT. The child exposes one
    # port group per bank and the container mirrors them onto the top, each
    # carrying its own `bank`/`factor` for the cosim harness to shard by.
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


def test_a_partitioned_container_local_buffer():
    # `propagate-partition` gives every child the same `allo.part`, so the
    # container just materializes the banks they already agree on: `bk_prod`
    # writes STATIC banks (one single-bank port group each), while `bk_cons`
    # reads a DATA-DEPENDENT one (crossbarred inside the child). Same shape the
    # leaf path takes, on the structural top.
    @kernel
    async def bk_src(s: Stream[i32]):
        s.put(42)

    @kernel
    async def bk_side(s: Stream[i32], o0: i32[1]):
        o0[0] = s.get()

    @kernel
    async def bk_prod(x: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = x[2 * i] + 1
            tmp[2 * i + 1] = x[2 * i + 1] + 100

    @kernel
    def bk_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] & 255

    @kernel
    async def bk_top(x: i32[16], out: i32[16], o0: i32[1]):
        f: Stream[i32]
        tmp: i32[16]
        await bk_src(f)
        await bk_side(f, o0)
        await bk_prod(x, tmp)
        bk_cons(tmp, out)

    s = bk_top.schedule()
    s.partition("tmp", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    m = mod.mlir
    assert "seq.hlmem @tmp_b0" in m and "seq.hlmem @tmp_b1" in m
    x = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    o0 = np.zeros(1, np.int32)
    mod.cosim(x, out, o0)
    # even lanes +1, odd lanes +100; a swapped bank route corrupts the golden
    exp = np.where(np.arange(16) % 2 == 0, x + 1, x + 100) & 255
    assert np.array_equal(out, exp), list(out)


# --- address linearization -------------------------------------------------


@pytest.mark.parametrize("cols", [24, 16])
def test_a_coalesced_nest_addresses_with_the_bare_counter(cols):
    # `flatten-perfect-loops` coalesces the nest and delinearizes the subscripts
    # against the single counter (`A[iv floordiv N, iv mod N]`); the memref's
    # row-major linearization composes straight back to `iv`. That cancellation
    # must happen on the affine EXPRESSION: rebuilding it out of comb ops costs
    # a divider, a modulo and a multiplier per port (a shift pair when N is a
    # power of two) to recompute an index the counter already holds, and nothing
    # downstream can fold them away.
    @kernel
    def flat(A: i32[6, cols], out: i32[6, cols]):
        for i in range(6):
            for j in range(cols):
                out[i, j] = A[i, j] + 1

    mod = _to_rtl(flat)
    for op in ("comb.divu", "comb.modu", "comb.mul", "comb.shru"):
        assert op not in mod.mlir, f"{op} in the address path of a flat nest"

    A = (np.arange(6 * cols, dtype=np.int32) % 251).reshape(6, cols)
    out = np.zeros((6, cols), np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, A + 1)


def test_a_partial_coalesced_subscript_keeps_its_address_arithmetic():
    # The counterpart: an inner loop the coalescing does not absorb leaves the
    # map with two live dims, so the row-major fold is a real shift/add rather
    # than the identity. It must still be emitted, and correctly.
    @kernel
    def part(A: i32[4, 8], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for k in range(8):
                acc += A[i, k]
            out[i] = acc

    mod = _to_rtl(part)
    A = (np.arange(32, dtype=np.int32) % 13).reshape(4, 8)
    out = np.zeros(4, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, A.sum(axis=1))


# --- ROM vs RAM classification -------------------------------------------


def test_constant_rom_cosim():
    # A constant-initialized local array lowers to a read-only ROM (an indexed
    # constant table) rather than a writable on-chip buffer: a byte table read
    # under a data-dependent index, and a wider (i32) table of non-power-of-two
    # length read by a scalar index.
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


def test_single_element_constant_table():
    # The depth-1 ROM edge case: a `hw.aggregate_constant` needs the spare field
    # too, and the padding must land PAST element 0 (a hw.array indexes element
    # 0 as its last field, so the initializer is reversed).
    @kernel
    def onerom(A: i32[8], B: i32[8]):
        tbl: i32[1] = [77]
        for i in range(8):
            B[i] = tbl[0] + A[i]

    mod = _to_rtl(onerom)
    assert "aggregate_constant" in mod.mlir  # a table, not an hlmem
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, 77 + A8)


def test_constant_table_reads_are_unlimited_port():
    # Three table reads per iteration pipeline at II=1. A 2-port budget would
    # force II=2 (ceil(3/2)) for hardware that has no ports at all.
    @kernel
    def rom3(A: i32[8], B: i32[8]):
        tbl: i32[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            B[i] = tbl[A[i] % 8] + tbl[(A[i] + 1) % 8] + tbl[(A[i] + 2) % 8]

    mod = _to_rtl(rom3)
    # One combinational constant array, one array_get per access, no hlmem.
    assert mod.mlir.count("aggregate_constant") == 1
    assert len(re.findall(r"hw\.array_get", mod.mlir)) == 3
    assert "seq.hlmem" not in mod.mlir
    iis = _iis(mod.schedule().func("rom3").regions)
    assert iis == [1], "a constant table must not limit reads"

    A = np.array([0, 3, 5, 7, 1, 2, 6, 4], dtype=np.int32)
    B = np.zeros(8, np.int32)
    mod.cosim(A, B)
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
    assert np.array_equal(B, t[A % 8] + t[(A + 1) % 8] + t[(A + 2) % 8])


def test_written_array_keeps_its_port_limit():
    # The contrast that keeps the ROM grant narrow: the SAME three reads off an
    # array the kernel writes are still bound by its two ports (II=2). Read-only
    # is a property of the use, so writing it once makes it a real memory.
    @kernel
    def ram3(A: i32[8], B: i32[8]):
        t: i32[8]
        for i in range(8):
            t[i] = A[i]
        for i in range(8):
            B[i] = t[A[i] % 8] + t[(A[i] + 1) % 8] + t[(A[i] + 2) % 8]

    iis = _iis(_sched(ram3).func("ram3").regions)
    assert iis == [1, 2], f"a written array must keep its port limit, got {iis}"


def test_read_only_initialized_array_is_a_rom():
    # The classification is on the USE, so a never-written initialized array
    # keeps its combinational constant-table realization.
    @kernel
    def lookup(A: i32[8], B: i32[8]):
        tbl: i32[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            B[i] = tbl[i] + A[i]

    m = _to_rtl(lookup)
    assert "hw.aggregate_constant" in m.mlir
    assert "seq.hlmem" not in m.mlir
    B = np.zeros(8, dtype=np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) * 10 + A8)


@pytest.mark.parametrize("decl", ["const", "stateful"])
def test_initialized_and_written_array(decl):
    # The same array written even once is not a constant table: it needs a real
    # write port AND the declared contents as power-on state (an `initial` block
    # over the backing storage). Both declaration forms that carry a
    # compile-time initializer -- a list-initialized local and `Stateful` --
    # realize identically.
    if decl == "const":

        @kernel
        def rmw(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(8):
                tbl[i] = tbl[i] + A[i]
            for i in range(8):
                B[i] = tbl[i]

    else:

        @kernel
        def rmw(A: i32[8], B: i32[8]):
            tbl: Stateful[i32[8]] = [1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(8):
                tbl[i] = tbl[i] + A[i]
            for i in range(8):
                B[i] = tbl[i]

    m = _to_rtl(rmw)
    # A writable memory, not the ROM a read-only table would give.
    assert "seq.hlmem @tbl" in m.mlir and "hw.aggregate_constant" not in m.mlir
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) + A8), list(B)


def test_initialized_and_written_scalar():
    # A `Stateful` scalar is the same case at depth 1: a rank-0 memref, whose
    # single element addresses at 0 with no subscript at all. It must not slip
    # through as a ROM (which would drop every store) or as an uninitialized
    # register (which would start at X).
    @kernel
    def counter(A: i32[8], B: i32[8]):
        acc: Stateful[i32] = 100
        for i in range(8):
            acc = acc + A[i]
        for i in range(8):
            B[i] = acc

    m = _to_rtl(counter)
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.all(B == 100 + A8.sum()), list(B)


@pytest.mark.parametrize("written", [False, True])
def test_initialized_float_array(written):
    # A float table's contents are its element bit patterns, the same
    # convention the datapath carries every float by. The constant-table and
    # written-memory forms share one conversion, so they cannot disagree on
    # what the declared values are.
    if written:

        @kernel
        def scale(A: f32[8], B: f32[8]):
            tbl: f32[8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            for i in range(8):
                tbl[i] = tbl[i] * A[i]
            for i in range(8):
                B[i] = tbl[i]

    else:

        @kernel
        def scale(A: f32[8], B: f32[8]):
            tbl: f32[8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            for i in range(8):
                B[i] = tbl[i] * A[i]

    m = _to_rtl(scale)
    assert ("seq.hlmem @tbl" in m.mlir) == written
    Af = np.arange(1, 9, dtype=np.float32)
    B = np.zeros(8, dtype=np.float32)
    m.cosim(Af, B)
    assert np.allclose(B, np.arange(1, 9, dtype=np.float32) * Af), list(B)


@pytest.mark.parametrize("child", ["reads", "writes"])
def test_initialized_array_handed_to_a_sub_kernel(child):
    # Read-only is a property of the USE, calls included. A sub-kernel that
    # WRITES the table needs a real memory that merely starts with its
    # contents; a table every child only READS stays a constant array, the
    # parent serving the child's address off the aggregate.
    if child == "reads":

        @kernel
        def use(t: i32[8], A: i32[8], B: i32[8]):
            for i in range(8):
                B[i] = t[i] + A[i]

        @kernel
        def top(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            use(tbl, A, B)

    else:

        @kernel
        def bump(t: i32[8], A: i32[8]):
            for i in range(8):
                t[i] = t[i] + A[i]

        @kernel
        def top(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            bump(tbl, A)
            for i in range(8):
                B[i] = tbl[i]

    m = _to_rtl(top)
    written = child == "writes"
    assert ("seq.hlmem @tbl" in m.mlir) == written, m.mlir
    assert ("hw.aggregate_constant" in m.mlir) != written, m.mlir
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) + A8), list(B)


# --- multi-cycle access latency -------------------------------------------


def _matvec_recurrence_ii(bind=None, complete=False):
    """Schedule a memory-carried matvec accumulate (`y[i] += A[i,k]*x[k]`) with
    the accumulator `y` optionally bound to a storage impl or complete-partitioned
    (-> registers)"""

    @kernel
    def mv(A: f32[8, 8], x: f32[8], out: f32[8]):
        y: f32[8] = 0
        for i in range(8):
            for k in range(8):
                y[i] += A[i, k] * x[k]
        for i in range(8):
            out[i] = y[i]

    s = mv.schedule()
    if complete:
        s.partition("y", kind=s.Complete)
    elif bind is not None:
        s.bind_storage("y", impl=bind, mem_type=s.RAM_T2P)
    res = s.export("rtl").schedule()
    return max(r.ii for r in res.cyclic())


def test_storage_impl_shifts_recurrence_ii():
    # The scheduler times a memory access by the array's real storage impl:
    # `bind_storage` (URAM) and complete partitioning (-> registers) both shift
    # the II of a memory-carried recurrence.
    # The recurrence II is read + FADD + write. Default LUTRAM (1/1) gives
    # FADD + 2; binding the accumulator to URAM (read 2, write 1) adds a cycle.
    lutram_ii = _matvec_recurrence_ii()
    assert lutram_ii == FADD + 2
    assert _matvec_recurrence_ii(bind=Schedule.URAM) == FADD + 3
    # A complete partition scatters `y` into FFs: the read is combinational (0),
    # but the FF write still costs a cycle, so the recurrence is FADD + 1 -- one
    # below LUTRAM, not a full collapse to the bare add latency.
    reg_ii = _matvec_recurrence_ii(complete=True)
    assert reg_ii == FADD + 1
    assert reg_ii < lutram_ii


def _uram_buffer_rtl(impl):
    """A producer/consumer pair through an internal buffer, optionally bound to
    a storage impl. The consumer reads `buf` inside an II=1 pipeline, so a read
    port built at the wrong latency shifts the result by one iteration."""

    @kernel
    def urambuf(A: i32[16], out: i32[16]):
        buf: i32[16] = 0
        for i in range(16):
            buf[i] = A[i] * 3
        for i in range(16):
            out[i] = buf[i] + 1

    s = urambuf.schedule()
    if impl is not None:
        s.bind_storage("buf", impl=impl, mem_type=s.RAM_T2P)
    return s.export("rtl")


def test_multicycle_storage_read_cosim():
    # The emitted read port must be built at the memory's DEVICE read latency,
    # not a hardcoded 1: URAM reads in 2 cycles, and the scheduler places the
    # consumer accordingly. The extra read cycle shows up in the whole-kernel
    # latency.
    exp = A16 * 3 + 1
    out_default = np.zeros(16, np.int32)
    r = _uram_buffer_rtl(None)
    lat_default = r.schedule().func("urambuf").latency
    r.cosim(A16, out_default)
    np.testing.assert_array_equal(out_default, exp)

    out_uram = np.zeros(16, np.int32)
    r = _uram_buffer_rtl(Schedule.URAM)
    lat_uram = r.schedule().func("urambuf").latency
    r.cosim(A16, out_uram)
    np.testing.assert_array_equal(out_uram, exp)
    # The 2-cycle URAM read costs exactly one cycle more than the 1-cycle default.
    assert lat_uram == lat_default + 1


def test_multicycle_storage_on_argument_cosim():
    # A boundary array's port latency is a contract with the driver: the
    # emitted RTL expects the read datum `latency` cycles after the address,
    # with no delay elements of its own. That number rides the interface
    # manifest and the cosim harness honors it, so a multi-cycle ARGUMENT is
    # emittable and the extra cycle shows up as whole-kernel latency.
    def argmem_rtl(impl):
        @kernel
        def argmem(A: i32[16], out: i32[16]):
            for i in range(16):
                out[i] = A[i] + 1

        s = argmem.schedule()
        if impl is not None:
            s.bind_storage("A", impl=impl, mem_type=s.RAM_T2P)
        return s.export("rtl")

    exp = A16 + 1
    out_default = np.zeros(16, np.int32)
    r = argmem_rtl(None)
    lat_default = r.schedule().func("argmem").latency
    r.cosim(A16, out_default)
    np.testing.assert_array_equal(out_default, exp)

    out_uram = np.zeros(16, np.int32)
    r = argmem_rtl(Schedule.URAM)
    lat_uram = r.schedule().func("argmem").latency
    r.cosim(A16, out_uram)
    np.testing.assert_array_equal(out_uram, exp)
    assert lat_uram == lat_default + 1

    # The contract must be stated in the manifest, not just honored by luck:
    # the URAM argument's read ports declare 2 cycles, the 1-cycle default 1.
    def read_latencies(rtl):
        iface = rtl.interfaces["argmem"]
        return {p["latency"] for acc in iface["reads"] for p in acc}

    assert read_latencies(argmem_rtl(Schedule.URAM)) == {2}
    assert read_latencies(argmem_rtl(None)) == {1}


def _dev(write_latency: int):
    """The built-in device with the default on-chip storage rebound to a
    ``write_latency``-cycle write."""
    d = builtin_device.copy()
    d.set_memory(MemoryKind.LUTRAM, 1, write_latency, 0.5, 0.5)
    d.set_default_memory(MemoryKind.LUTRAM)
    return d


def test_internal_array_multi_cycle_write():
    # An on-chip buffer bound to a 2- and 3-cycle write. Both a plain
    # producer/consumer hand-off (the write must land before the next region
    # reads it) and a same-address accumulate (the recurrence's II is read +
    # add + write, so a mistimed write reads back a stale partial) are covered.
    @kernel
    def xfer(A: i32[8], B: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = A[i] * 2
        for i in range(8):
            B[i] = buf[i] + 1

    @kernel
    def accumulate(A: i32[8], B: i32[8]):
        s: i32[8] = 0
        for i in range(8):
            s[0] = s[0] + A[i]
        for i in range(8):
            B[i] = s[0]

    for wr in (1, 2, 3):
        B = np.zeros(8, dtype=np.int32)
        _to_rtl(xfer, device=_dev(wr)).cosim(A8, B)
        assert np.array_equal(B, A8 * 2 + 1), f"wr_lat={wr}: {list(B)}"

        C = np.zeros(8, dtype=np.int32)
        _to_rtl(accumulate, device=_dev(wr)).cosim(A8, C)
        assert np.all(C == A8.sum()), f"wr_lat={wr}: {list(C)}"


def test_multi_cycle_write_through_sub_kernel_call():
    # A buffer whose write port is mastered by a child kernel takes the same
    # pipelining: the parent drives the port, so it owes the child's write the
    # same delay as one of its own.
    @kernel
    def fill(b: i32[8], A: i32[8]):
        for i in range(8):
            b[i] = A[i] * 5

    @kernel
    def top(A: i32[8], B: i32[8]):
        buf: i32[8] = 0
        fill(buf, A)
        for i in range(8):
            B[i] = buf[i] + 2

    for wr in (1, 2, 3):
        B = np.zeros(8, dtype=np.int32)
        _to_rtl(top, device=_dev(wr)).cosim(A8, B)
        assert np.array_equal(B, A8 * 5 + 2), f"wr_lat={wr}: {list(B)}"


# --- container-local storage ----------------------------------------------


def test_a_container_local_buffer_is_on_chip_storage():
    # A buffer declared in a dataflow container is storage the top OWNS, not a
    # port it forwards: one `seq.hlmem` and one port per accessing process,
    # driven straight from that child's addr/data/we, invisible at the
    # boundary interface.
    @kernel
    async def ia_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def ia_cons(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = s.get() + 1

    @kernel
    def ia_post(tmp: i32[N], out: i32[N]):
        for i in range(N):
            out[i] = tmp[i] * 3

    @kernel
    async def ia_top(out: i32[N]):
        f: Stream[i32]
        tmp: i32[N]  # declared HERE, not an argument
        await ia_prod(f)
        await ia_cons(f, tmp)
        ia_post(tmp, out)

    mod = _to_rtl(ia_top)
    assert "seq.hlmem @tmp" in mod.mlir
    # The buffer is internal: it must not show up as a boundary interface.
    top = mod.interfaces[mod.top]
    assert not any(
        m["base"].startswith("tmp") for acc in top["reads"] + top["writes"] for m in acc
    ), top
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, (np.arange(N) * 2 + 1) * 3), list(out)


def test_two_sync_processes_share_a_container_local_buffer():
    # Both accessors are determinate, so neither takes the `done` handshake:
    # the reader fires at the static offset the scheduler placed it at, past
    # the writer's latency. That ordering is the schedule's to make, so the
    # emitter's whole-array gate is inert here.
    @kernel
    async def q_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def q_cons(s: Stream[i32], o: i32[N]):
        for i in range(N):
            o[i] = s.get()

    @kernel
    def q_w(b: i32[N]):
        for i in range(N):
            b[i] = i * 7

    @kernel
    def q_r(b: i32[N], o2: i32[N]):
        for i in range(N):
            o2[i] = b[i] + 1

    @kernel
    async def q_top(o: i32[N], o2: i32[N]):
        f: Stream[i32]
        tmp: i32[N]
        await q_prod(f)
        await q_cons(f, o)
        q_w(tmp)
        q_r(tmp, o2)

    mod = _to_rtl(q_top)
    o = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(o, o2)
    assert np.array_equal(o, np.arange(N) * 2), list(o)
    assert np.array_equal(o2, np.arange(N) * 7 + 1), list(o2)


def test_two_processes_read_one_container_local_buffer_concurrently():
    # Two readers do not hazard, so nothing orders them and they run together
    # on ports of their own: each accessor gets its own port instead of sharing
    # an arbitrated one, since a mux would time-share exactly the pair that is
    # safe to run in parallel.
    @kernel
    async def tr_fill(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = i * 2
        s.put(1)

    @kernel
    async def tr_sink(s: Stream[i32], o0: i32[1]):
        o0[0] = s.get()

    @kernel
    def tr_a(tmp: i32[N], o1: i32[N]):
        for i in range(N):
            o1[i] = tmp[i] + 1

    @kernel
    def tr_b(tmp: i32[N], o2: i32[N]):
        for i in range(N):
            o2[i] = tmp[i] + 100

    @kernel
    async def tr_top(o0: i32[1], o1: i32[N], o2: i32[N]):
        f: Stream[i32]
        tmp: i32[N]
        await tr_fill(f, tmp)
        await tr_sink(f, o0)
        tr_a(tmp, o1)
        tr_b(tmp, o2)

    mod = _to_rtl(tr_top)
    o0 = np.zeros(1, np.int32)
    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(o0, o1, o2)
    assert np.array_equal(o1, np.arange(N) * 2 + 1), list(o1)
    assert np.array_equal(o2, np.arange(N) * 2 + 100), list(o2)


def test_a_multidimensional_container_local_buffer():
    # Shape is the child's business: it flattens its own addressing and drives
    # a linear address, so the container declares one cell of `prod(shape)`
    # words whatever the rank.
    @kernel
    async def s5_prod(s: Stream[i32], tmp: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                tmp[i, j] = i * 4 + j
        s.put(1)

    @kernel
    async def s5_cons(s: Stream[i32], o1: i32[1]):
        o1[0] = s.get()

    @kernel
    def s5_post(tmp: i32[4, 4], out: i32[16]):
        for i in range(4):
            for j in range(4):
                out[i * 4 + j] = tmp[i, j] * 3

    @kernel
    async def s5_top(out: i32[16], o1: i32[1]):
        f: Stream[i32]
        tmp: i32[4, 4]
        await s5_prod(f, tmp)
        await s5_cons(f, o1)
        s5_post(tmp, out)

    mod = _to_rtl(s5_top)
    assert "seq.hlmem @tmp %0, %rst : <16xi32>" in mod.mlir
    out = np.zeros(16, np.int32)
    o1 = np.zeros(1, np.int32)
    mod.cosim(out, o1)
    assert np.array_equal(out, np.arange(16) * 3), list(out)


def test_a_container_local_constant_table():
    # A table nothing writes is a ROM even when it is container-local: one
    # `hw.aggregate_constant` read combinationally and registered to the
    # latency the children were timed against. Classification comes from the
    # accessors, not the declaration.
    @kernel
    async def ct_src(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def ct_use(tbl: i32[N], s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = tbl[i] + s.get()

    @kernel
    async def ct_top(out: i32[N]):
        f: Stream[i32]
        tbl: i32[N] = [10, 20, 30, 40, 50, 60, 70, 80]
        await ct_src(f)
        await ct_use(tbl, f, out)

    mod = _to_rtl(ct_top)
    m = mod.mlir
    assert "hw.aggregate_constant" in m and "seq.hlmem" not in m
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, np.arange(1, N + 1) * 10 + np.arange(N)), list(out)


def test_a_written_container_table_keeps_its_contents():
    # The same container-local declaration, once a process WRITES it, is not a
    # ROM but a real memory that starts with the declared contents: the
    # classification comes from the accessors, and the container owns the
    # initialized storage either way.
    @kernel
    async def wt_src(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def wt_use(tbl: i32[N], s: Stream[i32], out: i32[N]):
        for i in range(N):
            tbl[i] = tbl[i] + s.get()
            out[i] = tbl[i]

    @kernel
    async def wt_top(out: i32[N]):
        f: Stream[i32]
        tbl: i32[N] = [10, 20, 30, 40, 50, 60, 70, 80]
        await wt_src(f)
        await wt_use(tbl, f, out)

    mod = _to_rtl(wt_top)
    m = mod.mlir
    assert "seq.hlmem @tbl" in m and "hw.aggregate_constant" not in m
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, np.arange(1, N + 1) * 10 + np.arange(N)), list(out)


# --- cross-region buffer identity ------------------------------------------


def test_single_element_internal_buffer():
    # A depth-1 internal buffer written and read every iteration; rank does not
    # matter (`i32[1, 1]` behaves the same), only the element COUNT does.
    @kernel
    def one(A: i32[8], B: i32[8]):
        t: i32[1]
        for i in range(8):
            t[0] = A[i] * 2
            B[i] = t[0] + 1

    B = np.zeros(8, np.int32)
    _to_rtl(one).cosim(A8, B)
    assert np.array_equal(B, A8 * 2 + 1)

    @kernel
    def one2d(A: i32[8], B: i32[8]):
        t: i32[1, 1]
        for i in range(8):
            t[0, 0] = A[i] * 3
            B[i] = t[0, 0] - 1

    B2 = np.zeros(8, np.int32)
    _to_rtl(one2d).cosim(A8, B2)
    assert np.array_equal(B2, A8 * 3 - 1)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_buffer_threaded_across_regions_is_one_memory(depth):
    # A straight-line store, then a loop reading it: the store is its own
    # (acyclic) region, so the buffer crosses a region boundary as a region
    # result. Every depth is covered because the split has nothing to do with
    # depth: it must remain ONE memory, not one per accessing region.
    if depth == 1:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[1]
            t[0] = A[0]
            for i in range(8):
                B[i] = t[0] + A[i]

    elif depth == 2:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[2]
            t[0] = A[0]
            for i in range(8):
                B[i] = t[0] + A[i]

    else:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[4]
            t[0] = A[0]
            for i in range(8):
                B[i] = t[0] + A[i]

    mod = _to_rtl(cross)
    assert len(re.findall(r"= seq\.hlmem", mod.mlir)) == 1, mod.mlir
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, A8[0] + A8)


def test_two_stores_threaded_across_regions():
    # Two straight-line stores feeding one downstream loop: both must reach the
    # reader's memory (a split loses both, so the reads come back zero).
    @kernel
    def cross2(A: i32[8], B: i32[8]):
        t: i32[4]
        t[0] = A[0]
        t[1] = A[1]
        for i in range(8):
            B[i] = t[0] + t[1] + A[i]

    mod = _to_rtl(cross2)
    assert len(re.findall(r"= seq\.hlmem", mod.mlir)) == 1, mod.mlir
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, A8[0] + A8[1] + A8)
