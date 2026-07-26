# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A sub-kernel with no static latency (a data-dependent `while`), consumed by the caller that invoked it: the region partitioner must isolate such a call into its own sequencer-driven region, pinning the two failure modes (a scalar result read too early, a buffer read before the child's real `done`) plus the precision control and concurrency preservation."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

# A run whose length depends on the data: the while stops at the first
# non-positive element, so `A` decides how many cycles the child takes.
A_RUN = np.array([3, 4, 5, 0, 9, 9, 9, 9], np.int32)
RUN_SUM = 12  # 3 + 4 + 5


@kernel
def ic_sum(A: i32[8]) -> i32:
    i: i32 = 0
    s: i32 = 0
    while A[i] > 0:
        s += A[i]
        i += 1
    return s


@kernel
def ic_sum_out(A: i32[8], B: i32[1]):
    i: i32 = 0
    s: i32 = 0
    while A[i] > 0:
        s += A[i]
        i += 1
    B[0] = s


def _regions(m):
    """The caller's own top-level dcp regions, as their opening lines. The
    callee is printed after it, so stop at the next `func.func`."""
    body = m.dcp.split("func.func public @")[1].split("func.func private")[0]
    return [ln.strip() for ln in body.splitlines() if "allo.dcp.sequential at" in ln]


# --- the scalar result --------------------------------------------------------


# A scalar result read by consumers written in the caller's own span. The
# partitioner splits them off into a region started by the child's done.
def test_a_scalar_result_consumed_in_the_callers_own_span():
    @kernel
    def ic_scalar(A: i32[8], B: i32[2]):
        r: i32 = ic_sum(A)
        B[0] = r + 1
        B[1] = r * 2

    m = _to_rtl(ic_scalar)
    assert len(_regions(m)) == 2, _regions(m)
    B = np.zeros(2, np.int32)
    m.cosim(A_RUN, B)
    assert np.array_equal(B, [RUN_SUM + 1, RUN_SUM * 2]), list(B)


# A call-to-call hand-off already starts the consumer on the producer's
# `done`. With both isolated it goes through the sibling sequencer instead
# and must still see the settled value.
def test_a_scalar_result_handed_to_a_second_call():
    @kernel
    def ic_twice(v: i32) -> i32:
        return v * 3

    @kernel
    def ic_chain(A: i32[8], B: i32[1]):
        r: i32 = ic_sum(A)
        B[0] = ic_twice(r)

    B = np.zeros(1, np.int32)
    _to_rtl(ic_chain).cosim(A_RUN, B)
    assert B[0] == RUN_SUM * 3, list(B)


# --- the memory half ----------------------------------------------------------


# Sharing the span schedules the load at the call's own start cycle, so it
# reads the buffer before the child has written it: the right hardware
# shape, the wrong answer, and no diagnostic.
def test_a_buffer_the_child_writes_and_the_caller_reads():
    @kernel
    def ic_buf(A: i32[8], out: i32[1]):
        t: i32[1]
        ic_sum_out(A, t)
        out[0] = t[0] + 1

    m = _to_rtl(ic_buf)
    assert len(_regions(m)) == 2, _regions(m)
    out = np.zeros(1, np.int32)
    m.cosim(A_RUN, out)
    assert out[0] == RUN_SUM + 1, list(out)


# The same hazard through a kernel ARGUMENT the child masters, where a port
# group rather than an internal hlmem carries the writes.
def test_a_boundary_buffer_the_child_writes():
    @kernel
    def ic_bnd(A: i32[8], B: i32[1], out: i32[1]):
        ic_sum_out(A, B)
        out[0] = B[0] * 2

    B, out = np.zeros(1, np.int32), np.zeros(1, np.int32)
    _to_rtl(ic_bnd).cosim(A_RUN, B, out)
    assert (B[0], out[0]) == (RUN_SUM, RUN_SUM * 2), (list(B), list(out))


# --- the isolation is specific ----------------------------------------------


# The negative control for the whole file: isolation must not swallow the
# entry-block rule. A call with statically-known latency is a time-triggered
# node the sequencer may legitimately overlap with its neighbours, so it is
# NOT isolated and stays inside its straight-line span.
def test_a_determinate_call_still_shares_its_span():
    @kernel
    def ic_fixed(v: i32) -> i32:
        return v + 7

    @kernel
    def ic_det(A: i32[8], B: i32[2]):
        r: i32 = ic_fixed(A[0])
        B[0] = r + 1

    assert len(_regions(_to_rtl(ic_det))) == 1


# Isolation adds a region, not an ordering: a sibling that shares nothing
# with the call has no dependence on it and still starts with the kernel.
def test_an_independent_sibling_still_runs_concurrently():
    @kernel
    def ic_indep(A: i32[8], B: i32[1], C: i32[4]):
        B[0] = ic_sum(A)
        for i in range(4):
            C[i] = i * 2

    B, C = np.zeros(1, np.int32), np.zeros(4, np.int32)
    _to_rtl(ic_indep).cosim(A_RUN, B, C)
    assert B[0] == RUN_SUM and np.array_equal(C, [0, 2, 4, 6]), (list(B), list(C))


# The caller's answer must track a child whose length is decided by the
# data, not by the schedule.
@pytest.mark.parametrize("stop", [1, 4, 7])
def test_a_run_length_that_actually_varies(stop):
    @kernel
    def ic_var(A: i32[8], B: i32[1]):
        r: i32 = ic_sum(A)
        B[0] = r + 1

    m = _to_rtl(ic_var)
    A = np.where(np.arange(8) < stop, np.arange(8) + 1, 0).astype(np.int32)
    B = np.zeros(1, np.int32)
    m.cosim(A, B)
    assert B[0] == A[:stop].sum() + 1, (stop, list(B))
