# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from allo.lang.core import range, i32, f32, Template, Stateful
from allo.lang.kernel import kernel
from allo.schedule import Schedule
from allo.schedule.errors import ScheduleLookupError, ScheduleTransformError

AFFINE_LOOP_IR = r"""
module {
  func.func @kernel(%arg0: memref<16xf32>) {
    affine.for %i = 0 to 16 {
      %0 = affine.load %arg0[%i] : memref<16xf32>
    }
    return
  }
}
"""


# ===========================================================================
# Kept frontend/diagnosability tests
# ===========================================================================


def test_schedule_from_string():
    s = Schedule.from_string(AFFINE_LOOP_IR)
    loop = s.loop()

    s.pipeline(loop, ii=2).apply()

    assert "pipeline.ii = 2 : i64" in str(s.payload)
    assert s.payload.operation.verify()


def test_compose_missing_callee_raises():
    @kernel
    def worker(a: i32[16], b: i32[16]):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def top(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] + 1

    ws = worker.schedule()
    ts = top.schedule()  # top never calls worker
    with pytest.raises(ScheduleLookupError):
        ts.compose(ws)


def test_schedule_requires_bound_templates():
    N = Template("N")

    @kernel(N)
    def top(A: i32[N], B: i32[N]):
        for i in range(N, name="i"):
            B[i] = A[i] + 1

    # Unspecialized templated kernel cannot be scheduled.
    with pytest.raises(TypeError):
        top.schedule()

    # Specializing the template binds it; scheduling then works.
    s = top[16].schedule()
    s.pipeline(s.loop("i"), ii=2).apply()
    assert "pipeline.ii = 2 : i64" in str(s.payload)
    assert s.payload.operation.verify()


def test_compose_nested():
    @kernel
    def inner(a: i32[16], b: i32[16]):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def mid(a: i32[16], b: i32[16]):
        inner(a, b)

    @kernel
    def top(A: i32[16], B: i32[16]):
        mid(A, B)

    # inner's schedule -> composed into mid -> composed into top, transitively.
    inner_s = inner.schedule()
    inner_s.pipeline(inner_s.loop("i"), ii=2)
    mid_s = mid.schedule()
    mid_s.compose(inner_s)

    ts = top.schedule()
    ts.compose(mid_s)
    ts.apply()

    text = str(ts.payload)
    assert "@top.mid" in text
    assert "@top.mid.inner" in text
    assert "pipeline.ii = 2 : i64" in text
    assert ts.payload.operation.verify()


# ===========================================================================
# Migrated from tests/test_schedule_compute.py
# ===========================================================================


def test_split():
    M, N = 10, 20

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    outer, inner = s.split(s.loop("j"), factor=4)
    mod = s.export("cpu")

    assert outer.key in s.snapshot.ops_by_key
    assert inner.key in s.snapshot.ops_by_key

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_loop_query_by_induction_var():
    # Loops are queryable by their induction-variable name even without an
    # explicit `range(..., name=...)`.
    M, N = 10, 20

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M):
            for j in range(N):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    s.pipeline(s.loop("i"), ii=2)
    outer, inner = s.split(s.loop("j"), factor=4)
    mod = s.export("cpu")

    assert outer.key in s.snapshot.ops_by_key
    assert inner.key in s.snapshot.ops_by_key
    assert "pipeline.ii = 2 : i64" in str(s.payload)

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_split_indivisible_factor():
    M, N = 10, 20

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    # 20 is not divisible by 3: the split must still produce correct results.
    s = add.schedule()
    s.split(s.loop("j"), factor=3)
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_pipeline():
    M, N = 10, 20

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    s.pipeline(s.loop("i"), ii=4)
    mod = s.export("cpu")
    assert "pipeline.ii = 4 : i64" in str(s.payload)

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_unroll():
    M, N = 10, 20

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    s.unroll(s.loop("j"), factor=4, tag_only=True).apply()
    assert "unroll.f = 4 : i64" in str(s.payload)
    assert s.payload.operation.verify()


def test_reorder():
    M, N, K, L = 4, 4, 4, 4

    @kernel
    def add(A: i32[M, N, K, L], B: i32[M, N, K, L], C: i32[M, N, K, L]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    for l in range(L, name="l"):
                        C[i, j, k, l] = A[i, j, k, l] + B[i, j, k, l]

    # Reorder non-consecutive axes (l before i) inside the affine band.
    s = add.schedule()
    s.reorder(("l", "i"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N, K, L)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N, K, L)).astype(np.int32)
    C = np.zeros((M, N, K, L), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_split_reorder():
    M, N = 8, 8

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    i, j = s.loops("i", "j")
    io, ii = s.split(i, factor=2)
    jo, ji = s.split(j, factor=4)
    s.reorder((jo, io, ji, ii))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_tile():
    M, N = 8, 8

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    i, j = s.loops("i", "j")
    s.tile((i, j), factors=[2, 4])
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_flatten():
    M, N = 8, 8

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    # .fuse in the old frontend is .flatten in the new one.
    s = add.schedule()
    i, j = s.loops("i", "j")
    s.flatten((i, j))
    mod = s.export("cpu")

    assert str(s.payload).count("affine.for") == 1
    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_tile_scf():
    M, N = 8, 8

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    # Tile the scf.for nest directly (no affine() raise) to exercise the scf
    # tiling path, which the other tests never hit.
    s = add.schedule()
    s.tile((s.loop("i"), s.loop("j")), factors=[2, 4])
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_tile_3d():
    M, N, K = 8, 8, 8

    @kernel
    def add3(A: i32[M, N, K], B: i32[M, N, K], C: i32[M, N, K]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j, k] = A[i, j, k] + B[i, j, k]

    # Tile a 3-deep perfect band (only 2-deep tiling is covered elsewhere).
    s = add3.schedule()
    i, j, k = s.loops("i", "j", "k")
    s.tile((i, j, k), factors=[2, 4, 2])
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N, K)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N, K)).astype(np.int32)
    C = np.zeros((M, N, K), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_tile_indivisible_non_square():
    # Non-square extents with factors that do not divide them exercise the
    # point-loop min() upper-bound canonicalization (divisible tiles never hit
    # it).
    M, N = 6, 10

    @kernel
    def add(A: i32[M, N], B: i32[M, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = add.schedule()
    i, j = s.loops("i", "j")
    s.tile((i, j), factors=[4, 3])
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_flatten_three_non_square():
    # Distinct non-square extents at three levels stress the floordiv/mod
    # index reconstruction (a square nest can mask remap bugs).
    M, N, K = 4, 5, 6

    @kernel
    def add3(A: i32[M, N, K], B: i32[M, N, K], C: i32[M, N, K]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j, k] = A[i, j, k] + B[i, j, k]

    s = add3.schedule()
    i, j, k = s.loops("i", "j", "k")
    s.flatten((i, j, k))
    mod = s.export("cpu")

    assert str(s.payload).count("affine.for") == 1
    A = np.random.randint(0, 10, (M, N, K)).astype(np.int32)
    B = np.random.randint(0, 10, (M, N, K)).astype(np.int32)
    C = np.zeros((M, N, K), dtype=np.int32)
    mod(A, B, C)
    np.testing.assert_array_equal(C, A + B)


def test_gemm_split_reorder():
    M, N, K = 8, 8, 8

    @kernel
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm.schedule()
    i, j = s.loops("i", "j")
    io, ii = s.split(i, factor=2)
    jo, ji = s.split(j, factor=2)
    s.reorder((io, jo, ii, ji))
    mod = s.export("cpu")

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, rtol=1e-4)


def test_compute_at():
    H, W = 8, 8

    @kernel
    def two_band(A: i32[H, W], C: i32[H, W]):
        B: i32[H, W] = 0
        for bi in range(H, name="bi"):
            for bj in range(W, name="bj"):
                B[bi, bj] = A[bi, bj] + 1
        for ci in range(H, name="ci"):
            for cj in range(W, name="cj"):
                C[ci, cj] = B[ci, cj] * 2

    s = two_band.schedule()
    s.compute_at(s.loop("bi"), s.loop("ci"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    C = np.zeros((H, W), dtype=np.int32)
    mod(A, C)
    np.testing.assert_array_equal(C, (A + 1) * 2)


def test_compute_at_complex():
    P = 4

    @kernel
    def three_band(A: i32[P, P, P], D: i32[P, P, P]):
        B: i32[P, P, P] = 0
        for bi in range(P, name="bi"):
            for bj in range(P, name="bj"):
                for bm in range(P, name="bm"):
                    B[bi, bj, bm] = A[bi, bj, bm] * 2
        C: i32[P, P, P] = 0
        for ci in range(P, name="ci"):
            for cj in range(P, name="cj"):
                for cm in range(P, name="cm"):
                    C[ci, cj, cm] = B[ci, cj, cm] + 1
        for di in range(P, name="di"):
            for dj in range(P, name="dj"):
                for dm in range(P, name="dm"):
                    D[di, dj, dm] = C[di, dj, dm] % 3

    s = three_band.schedule()
    s.compute_at(s.loop("bj"), s.loop("cj"))
    s.compute_at(s.loop("cm"), s.loop("dm"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    D = np.zeros((P, P, P), dtype=np.int32)
    mod(A, D)
    np.testing.assert_array_equal(D, ((A * 2) + 1) % 3)


# ---------------------------------------------------------------------------
# compute_at no-dependence path: the two cases above hit the RAW-fusion path;
# these cover the independent-producer move/inline + IV-remap path.
# ---------------------------------------------------------------------------


def test_compute_at_no_dep():
    # Producer (writes C from A) and consumer (writes D from B) share no buffer,
    # so there is no dependence and compute_at moves the producer body into the
    # consumer loop.
    N = 8

    @kernel
    def two_independent(A: i32[N], B: i32[N], C: i32[N], D: i32[N]):
        for pi in range(N, name="pi"):
            C[pi] = A[pi] + 1
        for ci in range(N, name="ci"):
            D[ci] = B[ci] * 2

    s = two_independent.schedule()
    s.compute_at(s.loop("pi"), s.loop("ci"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (N,)).astype(np.int32)
    B = np.random.randint(0, 10, (N,)).astype(np.int32)
    C = np.zeros((N,), dtype=np.int32)
    D = np.zeros((N,), dtype=np.int32)
    mod(A, B, C, D)
    np.testing.assert_array_equal(C, A + 1)
    np.testing.assert_array_equal(D, B * 2)


def test_compute_at_no_dep_inner_axis():
    # No-dependence move at an inner axis of a 2-deep nest (prefix IV remap).
    H, W = 6, 8

    @kernel
    def two_independent(A: i32[H, W], B: i32[H, W], C: i32[H, W], D: i32[H, W]):
        for pi in range(H, name="pi"):
            for pj in range(W, name="pj"):
                C[pi, pj] = A[pi, pj] + 1
        for ci in range(H, name="ci"):
            for cj in range(W, name="cj"):
                D[ci, cj] = B[ci, cj] * 2

    s = two_independent.schedule()
    s.compute_at(s.loop("pj"), s.loop("cj"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.random.randint(0, 10, (H, W)).astype(np.int32)
    C = np.zeros((H, W), dtype=np.int32)
    D = np.zeros((H, W), dtype=np.int32)
    mod(A, B, C, D)
    np.testing.assert_array_equal(C, A + 1)
    np.testing.assert_array_equal(D, B * 2)


def test_compute_at_no_dep_subset_bounds():
    # Producer bounds (0..8) are a strict subset of consumer bounds (0..10):
    # the move must wrap the producer body in an affine.if so it only runs on
    # the producer domain.
    @kernel
    def f(A: i32[8], C: i32[8], D: i32[10]):
        for pi in range(8, name="pi"):
            C[pi] = A[pi] + 1
        for ci in range(10, name="ci"):
            D[ci] = ci * 2

    s = f.schedule()
    s.compute_at(s.loop("pi"), s.loop("ci"))
    mod = s.export("cpu")
    assert "affine.if" in str(s.payload)

    A = np.random.randint(0, 10, (8,)).astype(np.int32)
    C = np.zeros((8,), dtype=np.int32)
    D = np.zeros((10,), dtype=np.int32)
    mod(A, C, D)
    np.testing.assert_array_equal(C, A + 1)
    np.testing.assert_array_equal(D, np.arange(10, dtype=np.int32) * 2)


def test_compute_at_no_dep_deeper_producer():
    # producerDepth (2) > consumerDepth (1): exercises the subtree move branch
    # instead of body inlining.
    H, W = 6, 8

    @kernel
    def f(A: i32[H, W], B: i32[H], C: i32[H, W], D: i32[H]):
        for pi in range(H, name="pi"):
            for pj in range(W, name="pj"):
                C[pi, pj] = A[pi, pj] + 1
        for ci in range(H, name="ci"):
            D[ci] = B[ci] * 2

    s = f.schedule()
    s.compute_at(s.loop("pj"), s.loop("ci"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.random.randint(0, 10, (H,)).astype(np.int32)
    C = np.zeros((H, W), dtype=np.int32)
    D = np.zeros((H,), dtype=np.int32)
    mod(A, B, C, D)
    np.testing.assert_array_equal(C, A + 1)
    np.testing.assert_array_equal(D, B * 2)


def test_compute_at_war_only_unsupported():
    # Producer reads X, consumer writes X: a WAR-only dependence, which
    # compute_at deliberately refuses.
    @kernel
    def f(X: i32[8], C: i32[8], D: i32[8]):
        for pi in range(8, name="pi"):
            C[pi] = X[pi] + 1
        for ci in range(8, name="ci"):
            X[ci] = D[ci]

    s = f.schedule()
    s.compute_at(s.loop("pi"), s.loop("ci"))
    with pytest.raises(ScheduleTransformError):
        s.apply()


# ===========================================================================
# Migrated from tests/test_schedule_memory.py
# ===========================================================================


def test_buffer_at():
    M, N = 8, 8

    @kernel
    def addone(A: f32[M, N], B: f32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[i, j] = A[i, j] + 1.0

    s = addone.schedule()
    s.buffer_at(s.buffer("B"), s.loop("i"))
    mod = s.export("cpu")

    A = np.random.rand(M, N).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1.0, rtol=1e-5)


def test_interleaving_acc():
    M, N, K = 8, 8, 8

    @kernel
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm.schedule()
    s.reorder((s.loop("k"), s.loop("j")))
    s.buffer_at(s.buffer("C"), s.loop("i"))
    s.pipeline(s.loop("j"))
    mod = s.export("cpu")

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, rtol=1e-4)


def test_buffer_at_read_only():
    # Buffering a read-only input produces a copy-in only (the two tests above
    # buffer written outputs, exercising copy-out).
    M, N = 8, 8

    @kernel
    def addone(A: i32[M, N], B: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[i, j] = A[i, j] * 2

    s = addone.schedule()
    s.buffer_at(s.buffer("A"), s.loop("i"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (M, N)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A * 2)


def test_buffer_at_middle_axis():
    # Buffer at the middle axis of a 3-deep nest (multiple inner loops).
    P = 4

    @kernel
    def f(A: i32[P, P, P], B: i32[P, P, P]):
        for i in range(P, name="i"):
            for j in range(P, name="j"):
                for k in range(P, name="k"):
                    B[i, j, k] = A[i, j, k] + 1

    s = f.schedule()
    s.buffer_at(s.buffer("B"), s.loop("j"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    B = np.zeros((P, P, P), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A + 1)


def test_buffer_at_strided_1d():
    # 1D buffer whose footprint lower bound is an outer-symbol expression
    # (io*4); also exercises the separating-stride == extent boundary.
    @kernel
    def f(A: i32[16], B: i32[16]):
        for io in range(4, name="io"):
            for ii in range(4, name="ii"):
                B[io * 4 + ii] = A[io * 4 + ii] + 1

    s = f.schedule()
    s.buffer_at(s.buffer("B"), s.loop("io"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (16,)).astype(np.int32)
    B = np.zeros((16,), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A + 1)


def test_buffer_at_innermost_axis_rejected():
    M, N = 8, 8

    @kernel
    def f(A: i32[M, N], B: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[i, j] = A[i, j] + 1

    s = f.schedule()
    s.buffer_at(s.buffer("B"), s.loop("j"))  # innermost axis is illegal
    with pytest.raises(ScheduleTransformError):
        s.apply()


def test_buffer_at_non_separable_rejected():
    # The target buffer does not depend on the selected axis, so it cannot be
    # made private to each iteration.
    M, N = 8, 8

    @kernel
    def f(A: i32[N], B: i32[N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[j] = A[j] + i

    s = f.schedule()
    s.buffer_at(s.buffer("B"), s.loop("i"))
    with pytest.raises(ScheduleTransformError):
        s.apply()


def test_partition_basic():
    M, N = 10, 10

    @kernel
    def copy(A: i32[M, N], B: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[i, j] = A[i, j]

    s = copy.schedule()
    s.partition(s.buffer("A"))
    s.apply()
    assert s.payload.operation.verify()
    # Partition is recorded as a kernel-argument attribute at schedule level.
    assert "partition<[(0,Complete,0)]>" in str(s.payload)


def test_partition_dim_factor():
    M, N = 10, 10

    @kernel
    def copy(A: i32[M, N], B: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                B[i, j] = A[i, j]

    s = copy.schedule()
    s.partition(s.buffer("A"), dim=1, factor=2, kind=Schedule.Block)
    s.apply()
    assert s.payload.operation.verify()
    assert "partition<[(1,Block,2)]>" in str(s.payload)


def _memref_global_line(ir: str) -> str:
    return next(line for line in ir.splitlines() if "memref.global" in line)


def test_partition_list_initialized_global():
    # A buffer backed by a list initializer lowers to memref.get_global; the
    # partition attribute must land on the backing memref.global, not a kernel
    # argument.
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    s = lut.schedule()
    s.partition(s.buffer("table"), kind=Schedule.Complete)
    s.apply()
    assert s.payload.operation.verify()
    assert "partition<[(0,Complete,0)]>" in _memref_global_line(str(s.payload))


def test_partition_stateful_array_by_name():
    # Stateful arrays are addressable by their source variable name, and the
    # attribute lands on the backing memref.global.
    @kernel
    def accbuf(idx: i32, x: i32) -> i32:
        st: Stateful[i32[8]] = 0
        st[idx] = st[idx] + x
        return st[idx]

    s = accbuf.schedule()
    s.partition(s.buffer("st"), dim=1, factor=4, kind=Schedule.Cyclic)
    s.apply()
    assert s.payload.operation.verify()
    assert "partition<[(1,Cyclic,4)]>" in _memref_global_line(str(s.payload))


# ===========================================================================
# Migrated from tests/test_schedule_compose.py
# ===========================================================================


def test_compose_two_kernels():
    M, K, N = 8, 8, 8

    @kernel
    def gemm(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    @kernel
    def addone(C: i32[M, N], D: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                D[i, j] = C[i, j] + 1

    @kernel
    def top(A: i32[M, K], B: i32[K, N], C: i32[M, N], D: i32[M, N]):
        gemm(A, B, C)
        addone(C, D)

    gs = gemm.schedule()
    gs.pipeline(gs.loop("j"), ii=1)
    as_ = addone.schedule()
    as_.pipeline(as_.loop("j"), ii=1)

    ts = top.schedule()
    ts.compose(gs)
    ts.compose(as_)
    mod = ts.export("cpu")

    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    D = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C, D)
    np.testing.assert_array_equal(D, A @ B + 1)


def test_compose_variadic_siblings():
    M, N, K = 4, 4, 4

    @kernel
    def gemm(A: i32[M, K], B: i32[K, N], C: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    @kernel
    def addone(C: i32[M, N], D: i32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                D[i, j] = C[i, j] + 1

    @kernel
    def top(A: i32[M, K], B: i32[K, N], C: i32[M, N], D: i32[M, N]):
        gemm(A, B, C)
        addone(C, D)

    gs = gemm.schedule()
    gs.pipeline(gs.loop("j"), ii=1)
    as_ = addone.schedule()
    as_.pipeline(as_.loop("j"), ii=1)

    # One compose call over both direct callees == composing each in turn.
    ts = top.schedule()
    ts.compose(gs, as_)
    mod = ts.export("cpu")

    A = np.random.randint(0, 10, (M, K)).astype(np.int32)
    B = np.random.randint(0, 10, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    D = np.zeros((M, N), dtype=np.int32)
    mod(A, B, C, D)
    np.testing.assert_array_equal(D, A @ B + 1)


def test_compose_gemm_scheduled():
    M, N, K = 8, 8, 8

    @kernel
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    @kernel
    def top(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        gemm(A, B, C)

    gs = gemm.schedule()
    gs.reorder((gs.loop("k"), gs.loop("j")))
    gs.buffer_at(gs.buffer("C"), gs.loop("i"))
    gs.pipeline(gs.loop("j"))

    ts = top.schedule()
    ts.compose(gs)
    mod = ts.export("cpu")

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, rtol=1e-4)


def test_compose_dependent_primitives():
    @kernel
    def worker(A: i32[32]):
        for i in range(32, name="i"):
            A[i] = i

    @kernel
    def top(A: i32[32]):
        worker(A)

    ws = worker.schedule()
    outer, inner = ws.split(ws.loop("i"), factor=2)
    ws.pipeline(inner, ii=1)

    ts = top.schedule()
    ts.compose(ws)
    mod = ts.export("cpu")

    A = np.zeros((32,), dtype=np.int32)
    mod(A)
    np.testing.assert_array_equal(A, np.arange(32, dtype=np.int32))


# ===========================================================================
# reuse_at
# ===========================================================================


def test_reuse_blur_x():
    H, W = 10, 10

    @kernel
    def blur(A: i32[H, W], B: i32[H, 8]):
        for y in range(H, name="y"):
            for x in range(8, name="x"):
                B[y, x] = A[y, x] + A[y, x + 1] + A[y, x + 2]

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("x"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((H, 8), dtype=np.int32)
    mod(A, B)
    ref = A[:, 0:8] + A[:, 1:9] + A[:, 2:10]
    np.testing.assert_array_equal(B, ref)


def test_reuse_blur_y():
    H, W = 10, 10

    @kernel
    def blur(A: i32[H, W], B: i32[8, W]):
        for y in range(8, name="y"):
            for x in range(W, name="x"):
                B[y, x] = A[y, x] + A[y + 1, x] + A[y + 2, x]

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("y"))  # reuse over the outer axis
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((8, W), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A[0:8] + A[1:9] + A[2:10])


def test_reuse_blur_x_3d():
    P = 10

    @kernel
    def blur(A: i32[P, P, P], B: i32[P, P, 8]):
        for i in range(P, name="i"):
            for j in range(P, name="j"):
                for k in range(8, name="k"):
                    B[i, j, k] = A[i, j, k] + A[i, j, k + 1] + A[i, j, k + 2]

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("k"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    B = np.zeros((P, P, 8), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A[:, :, 0:8] + A[:, :, 1:9] + A[:, :, 2:10])


def test_reuse_blur_y_3d():
    P = 10

    @kernel
    def blur(A: i32[P, P, P], B: i32[P, 8, P]):
        for i in range(P, name="i"):
            for j in range(8, name="j"):
                for k in range(P, name="k"):
                    B[i, j, k] = A[i, j, k] + A[i, j + 1, k] + A[i, j + 2, k]

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("j"))  # reuse over a middle axis
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    B = np.zeros((P, 8, P), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A[:, 0:8] + A[:, 1:9] + A[:, 2:10])


def test_reuse_blur_x_y():
    H, W = 10, 10

    @kernel
    def blur(A: i32[H, W], B: i32[8, 8]):
        for y in range(8, name="y"):
            for x in range(8, name="x"):
                B[y, x] = A[y, x] + A[y + 1, x + 1] + A[y + 2, x + 2]

    s = blur.schedule()
    rb_y = s.reuse_at(s.buffer("A"), s.loop("y"))
    s.reuse_at(rb_y, s.loop("x"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((8, 8), dtype=np.int32)
    mod(A, B)
    # Note: `range` is shadowed by allo's kernel range in this module, so the
    # reference is computed with numpy slicing rather than a Python loop.
    ref = A[0:8, 0:8] + A[1:9, 1:9] + A[2:10, 2:10]
    np.testing.assert_array_equal(B, ref)


def test_reuse_blur_box_x_y():
    # Chained reuse over a dense 2x2 box stencil: the x-window is fully covered
    # by accesses (unlike the diagonal), exercising the dense-footprint path.
    H, W = 10, 10

    @kernel
    def blur(A: i32[H, W], B: i32[8, 8]):
        for y in range(8, name="y"):
            for x in range(8, name="x"):
                B[y, x] = A[y, x] + A[y, x + 1] + A[y + 1, x] + A[y + 1, x + 1]

    s = blur.schedule()
    rb_y = s.reuse_at(s.buffer("A"), s.loop("y"))
    s.reuse_at(rb_y, s.loop("x"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((8, 8), dtype=np.int32)
    mod(A, B)
    ref = A[0:8, 0:8] + A[0:8, 1:9] + A[1:9, 0:8] + A[1:9, 1:9]
    np.testing.assert_array_equal(B, ref)


def test_reuse_blur_x_wide():
    # A wider (5-tap) stencil exercises a larger sliding window.
    H, W = 10, 12

    @kernel
    def blur(A: i32[H, W], B: i32[H, 8]):
        for y in range(H, name="y"):
            for x in range(8, name="x"):
                B[y, x] = (
                    A[y, x] + A[y, x + 1] + A[y, x + 2] + A[y, x + 3] + A[y, x + 4]
                )

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("x"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((H, 8), dtype=np.int32)
    mod(A, B)
    ref = A[:, 0:8] + A[:, 1:9] + A[:, 2:10] + A[:, 3:11] + A[:, 4:12]
    np.testing.assert_array_equal(B, ref)


def test_reuse_blur_ring():
    # Ring-buffer strategy (rotating head index) instead of physical shifting.
    H, W = 10, 10

    @kernel
    def blur(A: i32[H, W], B: i32[H, 8]):
        for y in range(H, name="y"):
            for x in range(8, name="x"):
                B[y, x] = A[y, x] + A[y, x + 1] + A[y, x + 2]

    s = blur.schedule()
    s.reuse_at(s.buffer("A"), s.loop("x"), ring=True)
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (H, W)).astype(np.int32)
    B = np.zeros((H, 8), dtype=np.int32)
    mod(A, B)
    np.testing.assert_array_equal(B, A[:, 0:8] + A[:, 1:9] + A[:, 2:10])


def test_reuse_blur_x_y_z_3d():
    # Three-level chained reuse over a 3D diagonal stencil: each stage targets
    # the previous stage's window and needs the full-window warmup fill.
    P = 8

    @kernel
    def blur(A: i32[P, P, P], B: i32[6, 6, 6]):
        for i in range(6, name="i"):
            for j in range(6, name="j"):
                for k in range(6, name="k"):
                    B[i, j, k] = (
                        A[i, j, k] + A[i + 1, j + 1, k + 1] + A[i + 2, j + 2, k + 2]
                    )

    s = blur.schedule()
    ri = s.reuse_at(s.buffer("A"), s.loop("i"))
    rj = s.reuse_at(ri, s.loop("j"))
    s.reuse_at(rj, s.loop("k"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    B = np.zeros((6, 6, 6), dtype=np.int32)
    mod(A, B)
    ref = A[0:6, 0:6, 0:6] + A[1:7, 1:7, 1:7] + A[2:8, 2:8, 2:8]
    np.testing.assert_array_equal(B, ref)


def test_reuse_conv2d_reduction_axis():
    # The reused load feeds the output store *through* an inner reduction loop
    # (acc += ...), so the store value is a loop-carried `affine.for` result
    # rather than a flat expression. The spatial-axis classifier must trace the
    # dependence through the reduction's yield to keep `oh` a spatial axis;
    # otherwise it is misread as reduction-only and reuse_at is rejected.
    IH, IW, OC, K = 12, 12, 4, 3
    OH, OW = IH - K + 1, IW - K + 1

    @kernel
    def conv2d(
        inp: f32[IH, IW],
        Wc: f32[OC, K, K],
        Wb: f32[OC, OH, OW],
        out: f32[OC, OH, OW],
    ):
        for oc in range(OC):
            for oh in range(OH, name="oh"):
                for ow in range(OW, name="ow"):
                    acc: f32 = Wb[oc, oh, ow]
                    for kh in range(K, name="kh"):
                        for kw in range(K, name="kw"):
                            acc += inp[oh + kh, ow + kw] * Wc[oc, kh, kw]
                    out[oc, oh, ow] = acc

    s = conv2d.schedule()
    s.reuse_at(s.buffer("inp"), s.loop("oh"))  # row line-buffer reuse
    mod = s.export("cpu")

    inp = np.random.randn(IH, IW).astype(np.float32)
    Wc = np.random.randn(OC, K, K).astype(np.float32)
    Wb = np.random.randn(OC, OH, OW).astype(np.float32)
    out = np.zeros((OC, OH, OW), dtype=np.float32)
    mod(inp, Wc, Wb, out)

    # `range` is shadowed by allo's kernel range here, so the reference avoids
    # Python loops: slide a KxK window over `inp` and contract against `Wc`.
    windows = np.lib.stride_tricks.sliding_window_view(inp, (K, K))
    ref = Wb + np.einsum("hwij,oij->ohw", windows, Wc)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)


# ===========================================================================
# streamline: memory-boundary -> on-chip stream fusion
# ===========================================================================


def _kernel_chunk(ir, sym):
    """The IR text of the allo.kernel @{sym} *definition* (up to the next kernel
    def). Matches the signature line only, so an `invoke @{sym}` in another
    kernel's body is not mistaken for the definition."""
    for chunk in ir.split("allo.kernel"):
        if f"@{sym}(" in chunk.split("\n", 1)[0]:
            return chunk
    return ""


def _signature(chunk):
    """The argument-list portion of a kernel chunk (before the first ')')."""
    return chunk.split(")")[0]


def test_streamline_passthrough():
    # Producer writes row-major, consumer reads row-major: both sides choose
    # PASSTHROUGH (put/get in place), so neither stage kernel allocates a buffer.
    N = 16

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def dbl(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] * 2.0

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        dbl(T, O)

    ts = top.schedule()
    ts.streamline("src", "dbl")
    ts.dataflow()

    ir = str(ts.payload)
    assert "allo.stream" in ir  # the boundary was actually converted
    assert "memref.alloc" not in _kernel_chunk(ir, "top.src")
    assert "memref.alloc" not in _kernel_chunk(ir, "top.dbl")

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) * 2.0, rtol=1e-4)


def test_streamline_stage_transpose():
    # Consumer reads T transposed -> not row-major -> it STAGES a buffer to
    # reorder. The producer is still passthrough; the result must be correct.
    N = 16

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def trans(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[j, i] * 2.0

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        trans(T, O)

    ts = top.schedule()
    ts.streamline("src", "trans")
    ts.dataflow()

    ir = str(ts.payload)
    assert "allo.stream" in ir
    assert "memref.alloc" in _kernel_chunk(ir, "top.trans")  # staging buffer
    assert "memref.alloc" not in _kernel_chunk(ir, "top.src")

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0).T * 2.0, rtol=1e-4)


def test_streamline_lanes():
    # lanes=L widens the boundary to L parallel FIFOs (!allo.stream<...,[L]>).
    N, L = 16, 4

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def dbl(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] * 2.0

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        dbl(T, O)

    ts = top.schedule()
    ts.streamline("src", "dbl", lanes=L)
    ts.dataflow()

    ir = str(ts.payload)
    assert f",[{L}]>" in ir  # the stream type carries L lanes

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) * 2.0, rtol=1e-4)


def test_streamline_chain():
    # 3-stage chain: streamline both boundaries; the middle kernel becomes
    # stream-in AND stream-out (both DRAM intermediates removed).
    N = 16

    @kernel
    def s1(X: f32[N, N], T1: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T1[i, j] = X[i, j] + 1.0

    @kernel
    def s2(T1: f32[N, N], T2: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T2[i, j] = T1[i, j] * 2.0

    @kernel
    def s3(T2: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T2[i, j] + 3.0

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T1: f32[N, N]
        T2: f32[N, N]
        s1(X, T1)
        s2(T1, T2)
        s3(T2, O)

    ts = top.schedule()
    ts.streamline("s1", "s2")
    ts.streamline("s2", "s3")
    ts.dataflow()

    ir = str(ts.payload)
    assert _signature(_kernel_chunk(ir, "top.s2")).count("allo.stream") == 2

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) * 2.0 + 3.0, rtol=1e-4)


def test_streamline_fanout():
    # One producer output T feeds TWO consumers (a residual/skip pattern). A
    # stream can't be read twice, so a generated tee broadcasts the boundary.
    N = 16

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def c1(T: f32[N, N], O1: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O1[i, j] = T[i, j] * 2.0

    @kernel
    def c2(T: f32[N, N], O2: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O2[i, j] = T[i, j] + 3.0

    @kernel
    def top(X: f32[N, N], O1: f32[N, N], O2: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        c1(T, O1)
        c2(T, O2)

    ts = top.schedule()
    ts.streamline("src", ["c1", "c2"])
    ts.dataflow()

    ir = str(ts.payload)
    assert "streamline_tee" in ir
    tee = _kernel_chunk(ir, "streamline_tee")
    assert tee.count("stream.get") == 1
    assert tee.count("stream.put") == 2  # broadcast to both consumers

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O1 = np.zeros((N, N), dtype=np.float32)
    O2 = np.zeros((N, N), dtype=np.float32)
    mod(X, O1, O2)
    np.testing.assert_allclose(O1, (X + 1.0) * 2.0, rtol=1e-4)
    np.testing.assert_allclose(O2, (X + 1.0) + 3.0, rtol=1e-4)


def test_streamline_fanout_lanes():
    # Fan-out composed with lanes: the tee broadcasts L lanes to each consumer.
    N, L = 16, 4

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def c1(T: f32[N, N], O1: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O1[i, j] = T[i, j] * 2.0

    @kernel
    def c2(T: f32[N, N], O2: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O2[i, j] = T[i, j] + 3.0

    @kernel
    def top(X: f32[N, N], O1: f32[N, N], O2: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        c1(T, O1)
        c2(T, O2)

    ts = top.schedule()
    ts.streamline("src", ["c1", "c2"], lanes=L)
    ts.dataflow()

    ir = str(ts.payload)
    assert "streamline_tee" in ir
    assert f",[{L}]>" in ir
    tee = _kernel_chunk(ir, "streamline_tee")
    assert tee.count("stream.get") == L
    assert tee.count("stream.put") == 2 * L

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O1 = np.zeros((N, N), dtype=np.float32)
    O2 = np.zeros((N, N), dtype=np.float32)
    mod(X, O1, O2)
    np.testing.assert_allclose(O1, (X + 1.0) * 2.0, rtol=1e-4)
    np.testing.assert_allclose(O2, (X + 1.0) + 3.0, rtol=1e-4)


def test_streamline_depth():
    # depth=D sets the FIFO depth of every stream the boundary creates.
    N = 8

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def dbl(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] * 2.0

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T: f32[N, N]
        src(X, T)
        dbl(T, O)

    ts = top.schedule()
    ts.streamline("src", "dbl", depth=8)
    ts.dataflow()

    ir = str(ts.payload)
    assert "f32, 8, []" in ir or "f32,8,[]" in ir  # depth-8 FIFO type

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) * 2.0, rtol=1e-4)


def _build_residual(depth):
    # A reconvergent diamond: T = src(X) fans out to `mid` and `jn`; `jn` also
    # reads U = mid(T). So `jn` joins a short branch (T direct) and a long branch
    # (T -> mid -> U) -- the residual/skip pattern.
    N = 8

    @kernel
    def src(X: f32[N, N], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def mid(T: f32[N, N], U: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                U[i, j] = T[i, j] * 2.0

    @kernel
    def jn(T: f32[N, N], U: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] + U[i, j]

    @kernel
    def top(X: f32[N, N], O: f32[N, N]):
        T: f32[N, N]
        U: f32[N, N]
        src(X, T)
        mid(T, U)
        jn(T, U, O)

    ts = top.schedule()
    ts.streamline("src", ["mid", "jn"], depth=depth)  # T fans out (tee)
    ts.streamline("mid", "jn", depth=depth)  # U: mid -> jn, closes the diamond
    ts.dataflow()
    return ts, N


def test_streamline_reconvergent_warns(capfd):
    # A shallow FIFO on the reconvergent short branch may deadlock; streamline
    # warns (naming the join) so the depth can be raised. Result still correct.
    ts, N = _build_residual(depth=2)
    ir = str(ts.payload)  # apply() runs the transform + the reconvergence check
    text = "".join(capfd.readouterr())
    assert "reconvergent" in text
    assert "top.jn" in text

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) + (X + 1.0) * 2.0, rtol=1e-4)


def test_streamline_reconvergent_deep_no_warn(capfd):
    # A depth >= the worst-case skew (the whole tensor) is deadlock-safe: no warn.
    ts, N = _build_residual(depth=8 * 8)
    ir = str(ts.payload)
    text = "".join(capfd.readouterr())
    assert "reconvergent" not in text

    mod = ts.export("cpu")
    X = np.random.rand(N, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, O)
    np.testing.assert_allclose(O, (X + 1.0) + (X + 1.0) * 2.0, rtol=1e-4)


def test_streamline_fanin():
    # Two producers fill disjoint contiguous row-major blocks of T (top / bottom
    # halves); one consumer reads the whole tensor. A generated `merge` kernel
    # concatenates the blocks in order -- the fan-in pattern.
    N = 8
    H = N // 2

    @kernel
    def p0(X: f32[H, N], T: f32[N, N]):
        for i in range(H, name="i"):
            for j in range(N, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def p1(Y: f32[H, N], T: f32[N, N]):
        for i in range(H, name="i"):
            for j in range(N, name="j"):
                T[H + i, j] = Y[i, j] + 2.0

    @kernel
    def c(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] * 3.0

    @kernel
    def top(X: f32[H, N], Y: f32[H, N], O: f32[N, N]):
        T: f32[N, N]
        p0(X, T)
        p1(Y, T)
        c(T, O)

    ts = top.schedule()
    ts.streamline(["p0", "p1"], "c")
    ts.dataflow()

    ir = str(ts.payload)
    assert "streamline_merge" in ir
    merge = _kernel_chunk(ir, "streamline_merge")
    assert merge.count("stream.get") == 2  # one drain loop per block
    assert merge.count("stream.put") == 2

    mod = ts.export("cpu")
    X = np.random.rand(H, N).astype(np.float32)
    Y = np.random.rand(H, N).astype(np.float32)
    O = np.zeros((N, N), dtype=np.float32)
    mod(X, Y, O)
    ref = np.empty((N, N), dtype=np.float32)
    ref[:H] = (X + 1.0) * 3.0
    ref[H:] = (Y + 2.0) * 3.0
    np.testing.assert_allclose(O, ref, rtol=1e-4)


def test_streamline_fanin_noncontiguous_errors():
    # Column-split producers each write T[:, block] -- not contiguous in
    # row-major order -- so fan-in cannot reconstruct the tensor by concatenating
    # streams;
    N = 8
    H = N // 2

    @kernel
    def p0(X: f32[N, H], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(H, name="j"):
                T[i, j] = X[i, j] + 1.0

    @kernel
    def p1(Y: f32[N, H], T: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(H, name="j"):
                T[i, H + j] = Y[i, j] + 2.0

    @kernel
    def c(T: f32[N, N], O: f32[N, N]):
        for i in range(N, name="i"):
            for j in range(N, name="j"):
                O[i, j] = T[i, j] * 3.0

    @kernel
    def top(X: f32[N, H], Y: f32[N, H], O: f32[N, N]):
        T: f32[N, N]
        p0(X, T)
        p1(Y, T)
        c(T, O)

    ts = top.schedule()
    ts.streamline(["p0", "p1"], "c")
    ts.dataflow()

    with pytest.raises(ScheduleTransformError):
        ts.apply()


def test_streamline_windowed_stencil():
    # A 3-tap vertical stencil reads a sliding window of input rows. streamline
    # stages only a K=3 row circular line buffer (not the full PxW tensor) and
    # fills it just-in-time -- minimal/windowed staging for conv/stencil.
    P, W = 16, 8
    K = 3

    @kernel
    def src(X: f32[P, W], A: f32[P, W]):
        for i in range(P, name="i"):
            for j in range(W, name="j"):
                A[i, j] = X[i, j] + 1.0

    @kernel
    def blur(A: f32[P, W], B: f32[P - 2, W]):
        for i in range(P - 2, name="i"):
            for j in range(W, name="j"):
                B[i, j] = A[i, j] + A[i + 1, j] + A[i + 2, j]

    @kernel
    def top(X: f32[P, W], B: f32[P - 2, W]):
        A: f32[P, W]
        src(X, A)
        blur(A, B)

    ts = top.schedule()
    ts.streamline("src", "blur")
    ts.dataflow()

    ir = str(ts.payload)
    blur_chunk = _kernel_chunk(ir, "top.blur")
    # the staged buffer is a K-row line buffer, not the full P rows
    assert f"memref<{K}x{W}xf32>" in blur_chunk
    assert f"memref<{P}x{W}xf32>" not in blur_chunk
    assert f"mod {K}" in blur_chunk  # circular row indexing
    # a purely vertical window fuses the fill into the compute loop: warmup
    # (outer + inner) + one fused main (outer + inner) == 4 loops, not 5.
    assert blur_chunk.count("affine.for") == 4

    mod = ts.export("cpu")
    X = np.random.rand(P, W).astype(np.float32)
    B = np.zeros((P - 2, W), dtype=np.float32)
    mod(X, B)
    A = X + 1.0
    np.testing.assert_allclose(B, A[: P - 2] + A[1 : P - 1] + A[2:P], rtol=1e-4)
