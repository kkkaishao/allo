# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RTL scheduling pre-passes, driven directly rather than through
the backend's public API.

Each pre-pass emits an `INFO: [PREP] ...` line (to stderr) describing the
transform it performed and the operation it touched, or a `WARN: [PREP] ...`
line explaining why it declined to transform a candidate. The whole set is
exercised in one subprocess (so ALLO_LOG_LEVEL and the source-file locations
are deterministic) and the tests assert on the captured output.
"""

import os
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest

from allo._mlir.ir import Context, Location, Module
from allo._mlir.dialects import allo as allo_d
from allo.backend.base import run_pipeline

# Runs every pre-pass on a tailored kernel; all diagnostics go to stderr.
SCRIPT = textwrap.dedent(
    """
    from allo import kernel
    from allo.lang import i32, f32, index
    from allo.backend.vitis.core import HLS_PREPARE_PIPELINE
    from allo.backend.base import run_pipeline

    def prep(m):
        run_pipeline(m, HLS_PREPARE_PIPELINE)
        return m

    # flatten-perfect-loops: a constant-trip perfect nest.
    @kernel
    def flat(A: i32[4, 4], B: i32[4, 4], C: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                C[i, j] = A[i, j] + B[i, j]
    run_pipeline(prep(flat.module), "builtin.module(func.func(flatten-perfect-loops))")

    # fold-if-statements: a data-dependent conditional in a loop body predicates
    # (an affine guard would instead fold into the loop bound).
    @kernel
    def ifc(A: i32[8], out: i32[8]):
        for i in range(8):
            if A[i] > 0:
                out[i] = A[i]
    run_pipeline(prep(ifc.module), "builtin.module(func.func(fold-if-statements))")

    # perfectize-loop-nest: an imperfect nest (accumulator init + inner loop + store).
    @kernel
    def perf(A: i32[4, 4], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for j in range(4):
                acc += A[i, j]
            out[i] = acc
    run_pipeline(prep(perf.module), "builtin.module(func.func(perfectize-loop-nest))")

    # raise-counted-while: a counted while loop.
    @kernel
    def whl(A: i32[128], out: i32[1]):
        i: index = 0
        s: i32 = 0
        while i < 128:
            s = s + A[i]
            i = i + 1
        out[0] = s
    run_pipeline(prep(whl.module), "builtin.module(func.func(raise-counted-while))")

    # unroll-under-pipeline + reassociate-reductions: pipelining the outer loop
    # unrolls the inner reduction into a chain that reassociate then rebalances.
    @kernel
    def red(A: f32[4, 8], out: f32[4]):
        for i in range(4):
            s: f32 = 0.0
            for j in range(8):
                s = s + A[i, j]
            out[i] = s
    sched = red.schedule()
    sched.pipeline("i", ii=1)
    run_pipeline(
        prep(sched.payload),
        "builtin.module(func.func(unroll-under-pipeline,"
        "reassociate-reductions{float-reassoc=true}))",
    )

    # rotate-reductions: a float reduction rotated across N accumulators.
    @kernel
    def rot(A: f32[16], out: f32[1]):
        acc: f32 = 0.0
        for i in range(16):
            acc += A[i]
        out[0] = acc
    run_pipeline(
        prep(rot.module),
        "builtin.module(func.func(rotate-reductions{accumulators=4}))",
    )

    # perfectize-loop-nest bail: sibling inner loops (with live surrounding ops).
    @kernel
    def sib(A: i32[4, 4], out: i32[4], out2: i32[4]):
        for i in range(4):
            t: i32 = 0
            for j in range(4):
                t += A[i, j]
            out[i] = t
            for k in range(4):
                out2[i] += A[i, k]
    run_pipeline(prep(sib.module), "builtin.module(func.func(perfectize-loop-nest))")

    # fold-if-statements bail: a data-dependent if wrapping a nested loop can be
    # neither predicated (a loop is not speculatable) nor folded into a bound
    # (its guard is not affine).
    @kernel
    def opq(A: i32[8], out: i32[8]):
        for i in range(8):
            if A[i] > 0:
                for j in range(4):
                    out[i] += A[j]
    run_pipeline(prep(opq.module), "builtin.module(func.func(fold-if-statements))")

    # rotate-reductions bail: trip count below the requested accumulator count.
    @kernel
    def rotshort(A: f32[4], out: f32[1]):
        acc: f32 = 0.0
        for i in range(4):
            acc += A[i]
        out[0] = acc
    run_pipeline(
        prep(rotshort.module),
        "builtin.module(func.func(rotate-reductions{accumulators=8}))",
    )

    # unroll-under-pipeline bail: a pipelined outer loop with a dynamic inner loop.
    @kernel
    def dyn(A: i32[16], n: index, out: i32[4]):
        for i in range(4):
            s: i32 = 0
            for j in range(n):
                s += A[j]
            out[i] = s
    dsched = dyn.schedule()
    dsched.pipeline("i", ii=1)
    run_pipeline(prep(dsched.payload), "builtin.module(func.func(unroll-under-pipeline))")
    """
)


@pytest.fixture(scope="module")
def prepass_log():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(SCRIPT)
        path = f.name
    env = dict(os.environ)
    env["ALLO_LOG_LEVEL"] = "info"
    try:
        proc = subprocess.run(
            [sys.executable, path], env=env, capture_output=True, text=True
        )
    finally:
        os.unlink(path)
    assert proc.returncode == 0, proc.stderr
    return proc.stderr, os.path.basename(path)


def _at(base):
    """Regex for a source anchor `(at [loop ]'name' base:line:col)`."""
    return r"\(at (?:loop )?(?:'\w+' )?" + re.escape(base) + r":\d+:\d+\)"


def test_loop_structure_prepasses(prepass_log):
    """flatten-perfect-loops, if-conversion, perfectize, raise-counted-while."""
    err, base = prepass_log
    assert re.search(
        r"INFO: \[PREP\] Flattening perfect nest of \d+ loops " + _at(base), err
    ), err
    assert re.search(
        r"INFO: \[PREP\] Performing if-conversion on hyperblock " + _at(base), err
    ), err
    assert re.search(
        r"INFO: \[PREP\] Perfectizing imperfect loop nest by sinking \d+ "
        r"surrounding ops into the inner loop " + _at(base),
        err,
    ), err
    assert re.search(
        r"INFO: \[PREP\] Raising counted while loop into a counted for loop "
        + _at(base),
        err,
    ), err


def test_pipeline_and_reduction_prepasses(prepass_log):
    """unroll-under-pipeline, reassociate-reductions, rotate-reductions."""
    err, base = prepass_log
    assert re.search(
        r"INFO: \[PREP\] Automatically fully unrolled the loop implied by "
        r"pipelining on '\w+' " + re.escape(base) + r":\d+:\d+ " + _at(base),
        err,
    ), err
    assert re.search(
        r"INFO: \[PREP\] Rebalancing associative reduction chain of \d+ terms "
        r"into a balanced tree",
        err,
    ), err
    assert re.search(
        r"INFO: \[PREP\] Rotating reduction across \d+ accumulators " + _at(base),
        err,
    ), err


def test_prepass_bail_diagnostics(prepass_log):
    """Each pass names the reason and consequence when it declines a candidate."""
    err, base = prepass_log
    assert re.search(
        r"WARN: \[PREP\] imperfect loop nest not perfectized because it has "
        r"sibling inner loops; the scheduler schedules its body as sequential "
        r"sub-regions instead of one fused pipeline " + _at(base),
        err,
    ), err
    assert re.search(
        r"WARN: \[PREP\] conditional left as an opaque scheduling unit because "
        r"'affine\.for' cannot be predicated; the enclosing loop cannot pipeline "
        r"across it " + _at(base),
        err,
    ), err
    assert re.search(
        r"WARN: \[PREP\] reduction not rotated because its trip count \d+ is "
        r"below the requested \d+ accumulators " + _at(base),
        err,
    ), err
    assert re.search(
        r"WARN: \[PREP\] pipelined loop has a dynamic or uncounted inner loop; "
        r"not unrolled, so it falls back to pipelining the innermost loop only "
        + _at(base),
        err,
    ), err


def test_partition_conflict_reported():
    """Two callsites passing differently-partitioned arrays to ONE callee is a
    genuine conflict -- the callee has one body and one schedule, so it cannot be
    banked two ways -- and is reported rather than silently resolved."""

    # Written in MLIR because the frontend clones a kernel per callsite
    # (`top.sub` / `top.sub.1`), so it cannot express a shared callee -- this
    # guards the IR contract, not a reachable frontend program.
    src = """
    module {
      func.func @sub(%X: memref<16xi32>, %o: memref<16xi32>) { return }
      func.func @top(%A: memref<16xi32> {allo.part = #allo.partition<[(1,Cyclic,2)]>},
                     %B: memref<16xi32> {allo.part = #allo.partition<[(1,Cyclic,4)]>},
                     %o1: memref<16xi32>, %o2: memref<16xi32>) {
        func.call @sub(%A, %o1) : (memref<16xi32>, memref<16xi32>) -> ()
        func.call @sub(%B, %o2) : (memref<16xi32>, memref<16xi32>) -> ()
        return
      }
    }
    """
    with Context() as ctx, Location.unknown():
        allo_d.register_dialect(ctx)
        module = Module.parse(src)
        with pytest.raises(Exception, match="partitioning conflict"):
            run_pipeline(module, "builtin.module(propagate-partition{top=top})")
