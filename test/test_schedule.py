import pytest

from allo.exp.schedule import Schedule
from allo.exp.schedule.errors import (
    InvalidScheduleArgumentError,
    ScheduleTypeError,
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


NESTED_AFFINE_LOOP_IR = r"""
module {
  func.func @kernel(%arg0: memref<16x16xf32>) {
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 16 {
        %0 = affine.load %arg0[%i, %j] : memref<16x16xf32>
      }
    }
    return
  }
}
"""


SCF_LOOP_IR = r"""
module {
  func.func @kernel() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
    }
    return
  }
}
"""


BUFFER_IR = r"""
module {
  func.func @kernel(%arg0: memref<16xf32>) {
    return
  }
}
"""


COMPUTE_AT_IR = r"""
module {
  func.func @kernel(%tmp: memref<8xi32>, %dst: memref<8xi32>) {
    affine.for %i = 0 to 8 {
      %c7 = arith.constant 7 : i32
      affine.store %c7, %tmp[%i] {sym_name = "producer_store"} : memref<8xi32>
    } {sym_name = "producer_loop"}
    affine.for %j = 0 to 8 {
      %v = affine.load %tmp[%j] : memref<8xi32>
      affine.store %v, %dst[%j] : memref<8xi32>
    } {sym_name = "consumer_loop"}
    return
  }
}
"""


BUFFER_AT_IR = r"""
module {
  func.func @kernel() {
    %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
    %c1 = arith.constant 1 : i32
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.store %c1, %tmp[%i, %j] : memref<8x8xi32>
      } {sym_name = "j"}
    } {sym_name = "i"}
    return
  }
}
"""


def test_pipeline_and_tag_unroll_apply_to_loop():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.pipeline(loop, ii=2).unroll(loop, factor=4).apply()

    text = str(schedule.payload)
    assert "pipeline.ii = 2 : i64" in text
    assert "unroll.f = 4 : i64" in text


def test_partition_apply_to_buffer_arg():
    schedule = Schedule.from_string(BUFFER_IR)
    buffer = schedule.query.buffer().one()

    schedule.partition(buffer).apply()

    text = str(schedule.payload)
    assert "allo.part" in text
    assert "#allo.partition<[(0,Complete,0)]>" in text


def test_non_topology_apply_keeps_existing_refs_live():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.pipeline(loop, ii=2).apply()
    schedule.unroll(loop, factor=4).apply()

    text = str(schedule.payload)
    assert schedule.epoch == loop.epoch
    assert "pipeline.ii = 2 : i64" in text
    assert "unroll.f = 4 : i64" in text


def test_generic_passes_apply_on_root():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)

    schedule.cse().dce().canonicalize().apply()

    assert schedule.epoch == 1
    assert schedule.payload.verify()


def test_polyhedral_raises_scf_loop_to_affine():
    schedule = Schedule.from_string(SCF_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.polyhedral(loop).apply()

    text = str(schedule.payload)
    assert "affine.for" in text
    assert "scf.for" not in text


def test_polyhedral_handle_can_feed_later_pipeline_before_apply():
    schedule = Schedule.from_string(SCF_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.polyhedral(loop).pipeline(loop, ii=3).apply()

    text = str(schedule.payload)
    assert "affine.for" in text
    assert "pipeline.ii = 3 : i64" in text


def test_split_returns_live_refs_and_inner_can_be_scheduled():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    outer, inner = schedule.split(loop, factor=4)

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert outer.epoch == schedule.epoch
    assert inner.epoch == schedule.epoch
    assert outer.id in schedule.snapshot.ops_by_id
    assert inner.id in schedule.snapshot.ops_by_id
    assert text.count("affine.for") == 2

    schedule.pipeline(inner, ii=1).apply()
    assert "pipeline.ii = 1 : i64" in str(schedule.payload)


def test_live_rebinds_stale_loop_ref():
    schedule = Schedule.from_string(NESTED_AFFINE_LOOP_IR)
    i, j = schedule.query.loop().all()

    schedule.split(i, factor=4)
    j = schedule.live(j)

    assert j.epoch == schedule.epoch
    assert j.id in schedule.snapshot.ops_by_id

    schedule.pipeline(j, ii=1).apply()
    assert "pipeline.ii = 1 : i64" in str(schedule.payload)


def test_reorder_returns_live_refs():
    schedule = Schedule.from_string(NESTED_AFFINE_LOOP_IR)
    i, j = schedule.query.loop().all()

    j, i = schedule.reorder([j, i])

    assert schedule.epoch == 1
    assert i.epoch == schedule.epoch
    assert j.epoch == schedule.epoch
    assert schedule.snapshot.ops_by_id[i.id].parent_id == j.id
    assert schedule.payload.verify()


def test_tile_returns_live_tile_and_point_refs():
    schedule = Schedule.from_string(NESTED_AFFINE_LOOP_IR)
    loops = schedule.query.loop().all()

    tiles, points = schedule.tile(loops, factors=[4, 4])

    assert schedule.epoch == 1
    assert len(tiles) == 2
    assert len(points) == 2
    assert all(loop.id in schedule.snapshot.ops_by_id for loop in tiles)
    assert all(loop.id in schedule.snapshot.ops_by_id for loop in points)
    assert str(schedule.payload).count("affine.for") == 4

    schedule.pipeline(points[-1], ii=1).apply()
    assert "pipeline.ii = 1 : i64" in str(schedule.payload)


def test_tile_broadcasts_integer_factor():
    schedule = Schedule.from_string(NESTED_AFFINE_LOOP_IR)
    loops = schedule.query.loop().all()

    tiles, points = schedule.tile(loops, factors=4)

    assert len(tiles) == 2
    assert len(points) == 2
    assert schedule.payload.verify()


def test_flatten_returns_live_ref():
    schedule = Schedule.from_string(NESTED_AFFINE_LOOP_IR)
    loops = schedule.query.loop().all()

    flat = schedule.flatten(loops)

    assert schedule.epoch == 1
    assert flat.id in schedule.snapshot.ops_by_id
    assert str(schedule.payload).count("affine.for") == 1

    schedule.pipeline(flat, ii=1).apply()
    assert "pipeline.ii = 1 : i64" in str(schedule.payload)


def test_outline_returns_live_func_and_call_refs():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    func, call = schedule.outline(loop, func_name="stage0")

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert func.id in schedule.snapshot.ops_by_id
    assert call.id in schedule.snapshot.ops_by_id
    assert "func.func @stage0" in text
    assert "@stage0" in text
    assert "call @stage0" in text
    assert schedule.payload.verify()


def test_outline_with_integer_mapping_returns_live_kernel_and_invoke_refs():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    kernel, call = schedule.outline(loop, func_name="stage0", mapping=2)

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert kernel.id in schedule.snapshot.ops_by_id
    assert call.id in schedule.snapshot.ops_by_id
    assert "allo.kernel @stage0" in text
    assert "allo.invoke @stage0" in text
    assert "mapping=[2]" in text
    assert schedule.payload.verify()


def test_outline_with_sequence_mapping():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    kernel, call = schedule.outline(loop, func_name="stage0", mapping=[2, 1])

    text = str(schedule.payload)
    assert kernel.id in schedule.snapshot.ops_by_id
    assert call.id in schedule.snapshot.ops_by_id
    assert "allo.kernel @stage0" in text
    assert "allo.invoke @stage0" in text
    assert "mapping=[2, 1]" in text
    assert schedule.payload.verify()


def test_physical_unroll_applies_immediately():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.unroll(loop, factor=4, tag_only=False)

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert text.count("affine.load") == 4
    assert schedule.payload.verify()


def test_physical_full_unroll_applies_immediately():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    schedule.unroll(loop, factor=0, tag_only=False)

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert "affine.for" not in text
    assert text.count("affine.load") == 16
    assert schedule.payload.verify()


def test_compute_at_returns_live_axis_ref():
    schedule = Schedule.from_string(COMPUTE_AT_IR)
    producer = schedule.query.op("producer_store").one()
    axis = schedule.query.loop("consumer_loop").one()

    axis = schedule.compute_at(producer, axis)

    text = str(schedule.payload)
    assert schedule.epoch == 1
    assert axis.id in schedule.snapshot.ops_by_id
    assert text.count("affine.for") == 1
    assert "producer_store" in text

    outer, inner = schedule.split(axis, factor=4)
    assert outer.id in schedule.snapshot.ops_by_id
    assert inner.id in schedule.snapshot.ops_by_id


def test_buffer_at_returns_live_buffer_ref():
    schedule = Schedule.from_string(BUFFER_AT_IR)
    buffer = schedule.query.buffer("tmp").one()
    axis = schedule.query.loop("i").one()

    local = schedule.buffer_at(buffer, axis)

    assert schedule.epoch == 1
    assert local.id in schedule.snapshot.values_by_id
    assert local.source == "res"
    assert "memref<1x8xi32>" in schedule.snapshot.values_by_id[local.id].type

    schedule.partition(local).apply()
    assert "allo.part" in str(schedule.payload)


def test_schedule_primitive_diagnostics():
    schedule = Schedule.from_string(AFFINE_LOOP_IR)
    loop = schedule.query.loop().one()

    with pytest.raises(InvalidScheduleArgumentError):
        schedule.pipeline(loop, ii=0)

    with pytest.raises(ScheduleTypeError):
        schedule.pipeline(schedule.query.op("kernel").one())
