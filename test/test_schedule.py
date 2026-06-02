import pytest

from allo.exp.lang.core import grid, i32, range, Template
from allo.exp.lang.kernel import kernel
from allo.exp.schedule import Schedule
from allo.exp.schedule.errors import (
    ConsumedHandleError,
    ScheduleLookupError,
    ScheduleStateError,
)

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


def test_schedule_from_string():
    s = Schedule.from_string(AFFINE_LOOP_IR)
    loop = s.loop()

    s.pipeline(loop, ii=2).apply()

    assert "pipeline.ii = 2 : i64" in str(s.payload)
    assert s.payload.operation.verify()


def test_pipeline_kernel_loop():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    loop = s.loop()

    s.pipeline(loop, ii=2).apply()

    text = str(s.payload)
    assert "allo.kernel public @top" in text
    assert "scf.for" in text
    assert "pipeline.ii = 2 : i64" in text
    assert s.payload.operation.verify()


def test_named_range_loop():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16, name="i"):
            B[i] = A[i] + 1

    s = top.schedule()
    loop = s.loop("i")

    s.pipeline(loop, ii=2).apply()

    assert loop.name == "i"
    assert "pipeline.ii = 2 : i64" in str(s.payload)
    assert s.payload.operation.verify()


def test_split_returns_live_loops():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    loop = s.loop()

    outer, inner = s.split(loop, factor=4)
    s.pipeline(inner, ii=1).apply()

    text = str(s.payload)
    assert outer.key in s.snapshot.ops_by_key
    assert inner.key in s.snapshot.ops_by_key
    assert text.count("scf.for") == 2
    assert "pipeline.ii = 1 : i64" in text
    assert s.payload.operation.verify()


def test_named_nested_range_loops():
    @kernel
    def top(A: "i32[4,4]", B: "i32[4,4]"):
        for i in range(4, name="i"):
            for j in range(4, name="j"):
                B[i, j] = A[i, j] + 1

    s = top.schedule()
    i, j = s.loops("i", "j")

    i, j = s.affine((i, j))
    flat = s.flatten((i, j))
    s.apply()

    assert [loop.name for loop in (i, j)] == ["i", "j"]
    assert flat.key in s.snapshot.ops_by_key
    assert str(s.payload).count("affine.for") == 1
    assert s.payload.operation.verify()


def test_flatten_nested_loops():
    @kernel
    def top(A: "i32[4,4]", B: "i32[4,4]"):
        for i in range(4):
            for j in range(4):
                B[i, j] = A[i, j] + 1

    s = top.schedule()
    loops = s.affine(s.loops())
    flat = s.flatten(loops)
    s.apply()

    text = str(s.payload)
    assert flat.key in s.snapshot.ops_by_key
    assert text.count("affine.for") == 1
    assert "scf.for" not in text
    assert s.payload.operation.verify()


def test_named_grid_loop_like_op():
    @kernel
    def top(A: "i32[4,4]", B: "i32[4,4]"):
        for i, j in grid(4, 4, name="ij"):
            B[i, j] = A[i, j] + 1

    s = top.schedule()
    loop = s.loop("ij")

    assert loop.name == "ij"
    assert loop.kind == "scf.parallel"
    assert loop.key in s.snapshot.ops_by_key


def test_compute_at_kernel():
    @kernel
    def top(A: "i32[8]", C: "i32[8]"):
        B: "i32[8]" = 0
        for i in range(8, name="i"):
            B[i] = A[i] * 2
        for j in range(8, name="j"):
            C[j] = B[j] + 1

    s = top.schedule()
    producer_loop, consumer_loop = s.affine(s.loops("i", "j"))

    loop = s.compute_at(producer_loop, consumer_loop)
    s.apply()

    text = str(s.payload)
    assert loop.key in s.snapshot.ops_by_key
    assert text.count("affine.for") == 1
    assert text.count("affine.store") == 2
    assert s.payload.operation.verify()


def test_buffer_at_single_level():
    @kernel
    def top(out: "i32[4,4]"):
        B: "i32[4,4]" = 0
        for i in range(4, name="i"):
            for j in range(4, name="j"):
                B[i, j] = i + j
                out[i, j] = B[i, j]

    s = top.schedule()
    s.affine(s.loops())
    buffer = s.buffer("B")
    axis = s.loop("i")

    local = s.buffer_at(buffer, axis)
    s.apply()

    assert local.key in s.snapshot.values_by_key
    assert "memref<1x4xi32>" in s.snapshot.values_by_key[local.key].type
    assert s.payload.operation.verify()


def test_pending_transforms_gate_real_ir():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    loop = s.loop()
    outer, inner = s.split(loop, factor=4)

    # Real IR is unavailable while transforms are pending.
    with pytest.raises(ScheduleStateError):
        _ = s.payload
    with pytest.raises(ScheduleStateError):
        _ = s.snapshot

    # Handles still resolve lazily against the predicted snapshot.
    assert inner.kind == "scf.for"

    s.apply()
    assert outer.key in s.snapshot.ops_by_key
    assert s.payload.operation.verify()


def test_apply_is_idempotent():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    outer, inner = s.split(s.loop(), factor=4)
    s.pipeline(inner, ii=1).apply()
    first = str(s.payload)

    # Re-applying with no pending work is a no-op; handles stay live.
    s.apply()
    assert outer.key in s.snapshot.ops_by_key
    assert inner.key in s.snapshot.ops_by_key
    assert str(s.payload) == first


def test_apply_incremental():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    outer, inner = s.split(s.loop(), factor=4)
    s.apply()
    assert str(s.payload).count("scf.for") == 2

    # A second apply runs only the new primitive (does not re-split).
    s.pipeline(inner, ii=2).apply()
    text = str(s.payload)
    assert text.count("scf.for") == 2
    assert "pipeline.ii = 2 : i64" in text
    assert s.payload.operation.verify()
    assert s.script._applied == len(s.script.includes)


def test_apply_incremental_matches_batched():
    def build():
        @kernel
        def top(A: "i32[16]", B: "i32[16]"):
            for i in range(16):
                B[i] = A[i] + 1

        return top.schedule()

    # All-at-once.
    s1 = build()
    o1, i1 = s1.split(s1.loop(), factor=4)
    s1.pipeline(i1, ii=2).apply()

    # Same transforms split across two apply batches.
    s2 = build()
    o2, i2 = s2.split(s2.loop(), factor=4)
    s2.apply()
    s2.pipeline(i2, ii=2).apply()

    assert str(s1.payload) == str(s2.payload)


def test_consumed_handle_raises():
    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        for i in range(16):
            B[i] = A[i] + 1

    s = top.schedule()
    loop = s.loop()
    s.split(loop, factor=4)

    # `loop` was consumed by split; reusing it is a frontend error.
    with pytest.raises(ConsumedHandleError):
        s.pipeline(loop, ii=1)


def test_compose_applies_callee_schedule():
    @kernel
    def worker(a: "i32[16]", b: "i32[16]"):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        worker(A, B)

    # Schedule the callee standalone (lazy), then compose it into the parent's copy.
    ws = worker.schedule()
    ws.pipeline(ws.loop("i"), ii=2)

    ts = top.schedule()
    ts.compose(ws)
    ts.apply()

    text = str(ts.payload)
    assert "@top.worker" in text
    assert "pipeline.ii = 2 : i64" in text
    assert ts.payload.operation.verify()


def test_compose_targets_specific_copy_with_id():
    @kernel
    def worker(a: "i32[16]", b: "i32[16]"):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        worker(A, B)
        worker(B, A)

    ws = worker.schedule()
    ws.pipeline(ws.loop("i"), ii=3)

    ts = top.schedule()
    # The second call is specialized as `top.worker.1`.
    ts.compose(ws, id=1)
    ts.apply()

    assert "pipeline.ii = 3 : i64" in str(ts.payload)
    assert ts.payload.operation.verify()


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
    # inner's pipeline reaches the transitive copy under top.
    assert "pipeline.ii = 2 : i64" in text
    assert ts.payload.operation.verify()


def test_compose_nested_with_own_primitive():
    @kernel
    def inner(a: "i32[16]", b: "i32[16]"):
        for i in range(16, name="i"):
            b[i] = a[i] + 1

    @kernel
    def mid(a: "i32[16]", b: "i32[16]"):
        inner(a, b)
        for k in range(16, name="k"):
            b[k] = b[k] + 1

    @kernel
    def top(A: "i32[16]", B: "i32[16]"):
        mid(A, B)

    inner_s = inner.schedule()
    inner_s.pipeline(inner_s.loop("i"), ii=2)
    mid_s = mid.schedule()
    mid_s.pipeline(mid_s.loop("k"), ii=4)  # mid's own loop
    mid_s.compose(inner_s)  # plus inner's schedule

    ts = top.schedule()
    ts.compose(mid_s)
    ts.apply()

    text = str(ts.payload)
    # Both bodies land on their respective copies (mid's own loop + inner's copy).
    assert "pipeline.ii = 4 : i64" in text
    assert "pipeline.ii = 2 : i64" in text
    assert ts.payload.operation.verify()
