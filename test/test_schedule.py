import numpy as np
import pytest

from allo.exp.lang.core import range, i32, f32, Template
from allo.exp.lang.kernel import kernel
from allo.exp.schedule import Schedule
from allo.exp.schedule.errors import ScheduleLookupError

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
    def worker(a: "i32[16]", b: "i32[16]"):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    ws = worker.schedule()
    ts = top.schedule()  # top never calls worker
    with pytest.raises(ScheduleLookupError):
        ts.compose(ws)


def test_schedule_requires_bound_templates():
    N = Template("N")

    @kernel(N)
    def top(A: "i32[N]", B: "i32[N]"):
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
    def inner(a: "i32[16]", b: "i32[16]"):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def mid(a: "i32[16]", b: "i32[16]"):
        inner(a, b)

    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N,K,L]", B: "i32[M,N,K,L]", C: "i32[M,N,K,L]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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
    def add(A: "i32[M,N]", B: "i32[M,N]", C: "i32[M,N]"):
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


def test_gemm_split_reorder():
    M, N, K = 8, 8, 8

    @kernel
    def gemm(A: "f32[M,K]", B: "f32[K,N]", C: "f32[M,N]"):
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
    def two_band(A: "i32[H,W]", C: "i32[H,W]"):
        B: "i32[H,W]" = 0
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
    def three_band(A: "i32[P,P,P]", D: "i32[P,P,P]"):
        B: "i32[P,P,P]" = 0
        for bi in range(P, name="bi"):
            for bj in range(P, name="bj"):
                for bm in range(P, name="bm"):
                    B[bi, bj, bm] = A[bi, bj, bm] * 2
        C: "i32[P,P,P]" = 0
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


# ===========================================================================
# Migrated from tests/test_schedule_memory.py
# ===========================================================================


def test_buffer_at():
    M, N = 8, 8

    @kernel
    def addone(A: "f32[M,N]", B: "f32[M,N]"):
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
    def gemm(A: "f32[M,K]", B: "f32[K,N]", C: "f32[M,N]"):
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


def test_partition_basic():
    M, N = 10, 10

    @kernel
    def copy(A: "i32[M,N]", B: "i32[M,N]"):
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
    def copy(A: "i32[M,N]", B: "i32[M,N]"):
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
    def gemm(A: "i32[M,K]", B: "i32[K,N]", C: "i32[M,N]"):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    @kernel
    def addone(C: "i32[M,N]", D: "i32[M,N]"):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                D[i, j] = C[i, j] + 1

    @kernel
    def top(A: "i32[M,K]", B: "i32[K,N]", C: "i32[M,N]", D: "i32[M,N]"):
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
    def gemm(A: "f32[M,K]", B: "f32[K,N]", C: "f32[M,N]"):
        for i in range(M, name="i"):
            for j in range(N, name="j"):
                for k in range(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    @kernel
    def top(A: "f32[M,K]", B: "f32[K,N]", C: "f32[M,N]"):
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
    def worker(A: "i32[32]"):
        for i in range(32, name="i"):
            A[i] = i

    @kernel
    def top(A: "i32[32]"):
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
# reuse_at: not yet wired into the new schedule -> expected failure.
# ===========================================================================


@pytest.mark.xfail(reason="reuse_at is not yet implemented in the new schedule")
def test_reuse_blur_x():
    H, W = 10, 10

    @kernel
    def blur(A: "i32[H,W]", B: "i32[H,8]"):
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


@pytest.mark.xfail(reason="reuse_at is not yet implemented in the new schedule")
def test_reuse_blur_x_y():
    H, W = 10, 10

    @kernel
    def blur(A: "i32[H,W]", B: "i32[8,8]"):
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
    ref = np.zeros((8, 8), dtype=np.int32)
    for y in range(8):
        for x in range(8):
            ref[y, x] = A[y, x] + A[y + 1, x + 1] + A[y + 2, x + 2]
    np.testing.assert_array_equal(B, ref)
