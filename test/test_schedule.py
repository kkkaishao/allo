import numpy as np
import pytest

from allo.exp.lang.core import range, i32, f32, Template
from allo.exp.lang.kernel import kernel
from allo.exp.schedule import Schedule
from allo.exp.schedule.errors import ScheduleLookupError, ScheduleTransformError

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
    s.unroll(s.loop("j"), factor=4).apply()
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
    i, j, k, l = s.affine(s.loops("i", "j", "k", "l"))
    s.reorder((l, i))
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
    i, j = s.affine(s.loops("i", "j"))
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
    i, j = s.affine(s.loops("i", "j"))
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
    i, j = s.affine(s.loops("i", "j"))
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
    i, j, k = s.affine(s.loops("i", "j", "k"))
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
    i, j = s.affine(s.loops("i", "j"))
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
    i, j, k = s.affine(s.loops("i", "j", "k"))
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
    i, j, k = s.affine(s.loops("i", "j", "k"))
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
    s.affine(s.loops("bi", "bj", "ci", "cj"))
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
    s.affine(s.loops())
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
    s.affine(s.loops("pi", "ci"))
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
    s.affine(s.loops("pi", "pj", "ci", "cj"))
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
    s.affine(s.loops("pi", "ci"))
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
    s.affine(s.loops("pi", "pj", "ci"))
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
    s.affine(s.loops("pi", "ci"))
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
    s.affine(s.loops("i", "j"))
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
    s.affine(s.loops("i", "j", "k"))
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
    s.affine(s.loops("i", "j"))
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
    s.affine(s.loops("i", "j", "k"))
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
    s.affine(s.loops("io", "ii"))
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
    s.affine(s.loops("i", "j"))
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
    s.affine(s.loops("i", "j"))
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
    gs.affine(gs.loops("i", "j", "k"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("i", "j", "k"))
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
    s.affine(s.loops("i", "j", "k"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("y", "x"))
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
    s.affine(s.loops("i", "j", "k"))
    ri = s.reuse_at(s.buffer("A"), s.loop("i"))
    rj = s.reuse_at(ri, s.loop("j"))
    s.reuse_at(rj, s.loop("k"))
    mod = s.export("cpu")

    A = np.random.randint(0, 10, (P, P, P)).astype(np.int32)
    B = np.zeros((6, 6, 6), dtype=np.int32)
    mod(A, B)
    ref = A[0:6, 0:6, 0:6] + A[1:7, 1:7, 1:7] + A[2:8, 2:8, 2:8]
    np.testing.assert_array_equal(B, ref)
