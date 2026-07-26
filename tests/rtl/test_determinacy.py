# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dcp.determinacy classification: counted_static / conditional / indeterminate / concurrent, at both the region level and the whole-kernel level. These schedule and reify only, so they need no simulator."""

from allo import kernel
from allo.lang import i32, f32, index, Stream

from _common import _to_rtl  # noqa: E402


# The whole-kernel value is behavior-load-bearing: DataflowTop's
# `calleeDeterminate` reads it, so a callee is a static-offset producer iff it
# is `counted_static`. The region value is the declared controller-regime
# discriminant.
def _reg(det: str) -> str:
    """A dcp region op's determinacy, printed as a bare keyword right after the
    region body (just before the op's attr-dict)."""
    return f"}} {det}"


def _func(det: str) -> str:
    """A whole-kernel `dcp.determinacy` attribute (on the func)."""
    return f"dcp.determinacy = #allo<determinacy {det}>"


# A counted loop's region and its enclosing kernel are both counted_static.
def test_counted_loop_and_kernel_are_counted_static():
    @kernel
    def leaf(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    d = _to_rtl(leaf).dcp
    assert "allo.dcp.pipeline" in d and _reg("counted_static") in d
    assert _func("counted_static") in d


# A static sequential composition: the container and both leaves are exact,
# so a caller can release a consumer at a static offset.
def test_sequential_container_is_counted_static():
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

    d = _to_rtl(seq_top).dcp
    assert _func("counted_static") in d
    assert "concurrent" not in d  # a plain call graph is not a dataflow spawn


# A genuine (data-dependent-exit) while flushing-pipelines -> its region is
# conditional; the kernel's total latency is unknown -> indeterminate.
def test_data_dependent_while_is_conditional():
    @kernel
    def wr(n0: i32, out: i32[1]):
        x: i32 = n0
        c: i32 = 0
        while x > 1:
            x = x - 1
            c = c + 1
        out[0] = c

    d = _to_rtl(wr).dcp
    assert _reg("conditional") in d
    assert _func("indeterminate") in d


# A data-dependent guard closes into a dcp.select -> conditional.
def test_guard_select_is_conditional():
    N, M = 8, 4

    @kernel
    def cond_reduce(A: f32[N, M], flag: i32[M], out: f32[M]):
        for j in range(M):
            if flag[j] > 0:
                acc: f32 = 0.0
                for k in range(N):
                    acc += A[k, j]
                out[j] = acc

    d = _to_rtl(cond_reduce).dcp
    assert "allo.dcp.select" in d and _reg("conditional") in d
    assert _func("indeterminate") in d


# A dynamic outer trip has no exact latency -> the wrapper region and the
# kernel are both indeterminate (a bounded or unknown span, so no consumer
# can be placed at a static offset).
def test_dynamic_trip_wrapper_is_indeterminate():
    N = 4

    @kernel
    def band(A: f32[N, N], y: f32[N], n: index):
        for i in range(n):
            for j in range(N):
                y[i] += A[i, j]

    d = _to_rtl(band).dcp
    assert _reg("indeterminate") in d
    assert _func("indeterminate") in d


# An await-spawned dataflow container is concurrent (self-timed) -> a caller
# waits on its real done, never a static offset; the spawned leaves stay
# counted_static.
def test_async_dataflow_container_is_concurrent():
    N = 16

    @kernel
    async def dp(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def dc(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() + 1

    @kernel
    async def dtop(out: i32[N]):
        fifo: Stream[i32]
        await dp(fifo)
        await dc(fifo, out)

    d = _to_rtl(dtop).dcp
    assert _func("concurrent") in d
    assert _func("counted_static") in d  # the leaves
