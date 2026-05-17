from __future__ import annotations

from typing import Protocol

from .errors import capture_schedule_location
from .model import BufferRef, OpRef, ScheduleSnapshot
from .._C import ir, schedule as schedule_d, transform as tran_d


class TransformHost(Protocol):
    context: ir.Context
    epoch: int
    snapshot: ScheduleSnapshot


class TransformScript:
    def __init__(self, schedule: TransformHost):
        self.schedule = schedule
        self.context = schedule.context
        self.builder = ir.AlloOpBuilder(self.context)
        self.builder.set_unknown_loc()

        self.module = ir.ModuleOp(self.builder)
        self.module.set_attr(
            "transform.with_named_sequence", ir.UnitAttr.get(self.context)
        )

        self.builder.set_insertion_point_to_start(self.module.get_body())
        root_type = tran_d.OperationType.get(self.context, "builtin.module")
        self.sequence = tran_d.NamedSequenceOp(
            self.builder, "__transform_main", root_type, []
        )

        entry = self.sequence.get_entry_block()
        self.builder.set_insertion_point_to_end(entry)
        tran_d.YieldOp(self.builder, [])
        self.builder.set_insertion_point_to_start(entry)

        self.root = self.sequence.get_arg_at(0)
        self.any_op_type = tran_d.AnyOpType.get(self.context)
        self._op_handles = {schedule.snapshot.root_id: self.root}
        self._value_handles = {}

    def set_callsite_loc(self) -> None:
        loc = capture_schedule_location()
        if loc is None:
            self.builder.set_unknown_loc()
            return
        self.builder.set_loc(
            ir.Location(
                loc.file_name,
                loc.line,
                loc.col + 1,
                self.context,
            )
        )

    def op_handle(self, ref: OpRef) -> ir.Value:
        handle = self._op_handles.get(ref.id)
        if handle is not None:
            return handle

        node = self.schedule.snapshot.ops_by_id[ref.id]
        attrs = ir.DictionaryAttr.get(
            self.context,
            {schedule_d.SCHEDULE_ID_ATTR_NAME: self.builder.get_string_attr(ref.id)},
        )
        handle = tran_d.MatchOp(
            self.builder, self.root, self.any_op_type, [node.kind], attrs
        ).get_result_at(0)
        self._op_handles[ref.id] = handle
        return handle

    def value_handle(self, ref: BufferRef) -> ir.Value:
        handle = self._value_handles.get(ref.id)
        if handle is not None:
            return handle

        owner = self._owner_ref(ref)
        owner_handle = self.op_handle(owner)
        source_kind = {"arg": 1, "res": 2}[ref.source]
        handle = tran_d.MatchValueOp(
            self.builder, owner_handle, ref.number, source_kind
        ).get_result_at(0)
        self._value_handles[ref.id] = handle
        return handle

    def set_op_handle(self, ref: OpRef, handle: ir.Value) -> None:
        self._op_handles[ref.id] = handle

    def defining_op_handle(self, handle: ir.Value) -> ir.Value:
        return tran_d.GetDefiningOp(self.builder, handle).get_result_at(0)

    def annotate_schedule_id(self, handle: ir.Value, schedule_id: str) -> None:
        tran_d.AnnotateOp(
            self.builder,
            handle,
            schedule_d.SCHEDULE_ID_ATTR_NAME,
            self.builder.get_string_attr(schedule_id),
        )

    def _owner_ref(self, ref: BufferRef) -> OpRef:
        node = self.schedule.snapshot.ops_by_id[ref.owner_id]
        return OpRef(
            id=node.id,
            epoch=self.schedule.epoch,
            kind=node.kind,
            name=node.name,
            path=node.path,
        )
