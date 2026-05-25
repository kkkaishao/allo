from allo.exp.lang.core import grid, i32, range
from allo.exp.lang.kernel import kernel
from allo.exp.schedule import Schedule

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
    assert s.payload.verify()


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
    assert s.payload.verify()


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
    assert s.payload.verify()


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
    assert outer.id in s.snapshot.ops_by_id
    assert inner.id in s.snapshot.ops_by_id
    assert text.count("scf.for") == 2
    assert "pipeline.ii = 1 : i64" in text
    assert s.payload.verify()


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

    assert [loop.name for loop in (i, j)] == ["i", "j"]
    assert flat.id in s.snapshot.ops_by_id
    assert str(s.payload).count("affine.for") == 1
    assert s.payload.verify()


def test_flatten_nested_loops():
    @kernel
    def top(A: "i32[4,4]", B: "i32[4,4]"):
        for i in range(4):
            for j in range(4):
                B[i, j] = A[i, j] + 1

    s = top.schedule()
    loops = s.affine(s.loops())
    flat = s.flatten(loops)

    text = str(s.payload)
    assert flat.id in s.snapshot.ops_by_id
    assert text.count("affine.for") == 1
    assert "scf.for" not in text
    assert s.payload.verify()


def test_named_grid_loop_like_op():
    @kernel
    def top(A: "i32[4,4]", B: "i32[4,4]"):
        for i, j in grid(4, 4, name="ij"):
            B[i, j] = A[i, j] + 1

    s = top.schedule()
    loop = s.loop("ij")

    assert loop.name == "ij"
    assert loop.kind == "scf.parallel"
    assert loop.id in s.snapshot.ops_by_id


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

    text = str(s.payload)
    assert loop.id in s.snapshot.ops_by_id
    assert text.count("affine.for") == 1
    assert text.count("affine.store") == 2
    assert s.payload.verify()


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

    assert local.id in s.snapshot.values_by_id
    assert "memref<1x4xi32>" in s.snapshot.values_by_id[local.id].type
    assert s.payload.verify()
