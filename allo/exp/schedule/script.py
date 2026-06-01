from __future__ import annotations

from typing import Protocol

from ..._mlir import ir
from ..._mlir.dialects import transform as t
from ..._mlir.dialects.transform import structured as ts
from ..._mlir.dialects.transform import allo as ta
from ..._mlir.schedule import SCHEDULE_ID_ATTR_NAME, SCHEDULE_NAME_ATTR_NAME

from .errors import capture_schedule_location
from .model import BufferRef, OpRef, ScheduleSnapshot
from ..compiler.builder import AlloOpBuilder


class TransformHost(Protocol):
    context: ir.Context
    epoch: int
    snapshot: ScheduleSnapshot


class TransformScript:
    """Builds a ``__transform_main`` named sequence on upstream `allo._mlir`.

    Ops are inserted *before* the sequence's terminating ``transform.yield`` so
    they accumulate in forward (data-dependency) order across primitive calls.
    """

    def __init__(self, schedule: TransformHost):
        self.schedule = schedule
        self.context = schedule.context
        self.builder = AlloOpBuilder(self.context)
        self.builder.set_unknown_loc()

        with self.context, ir.Location.unknown(self.context):
            self.module = ir.Module.create()
            self.module.operation.attributes["transform.with_named_sequence"] = (
                ir.UnitAttr.get(self.context)
            )
            self.any_op_type = t.AnyOpType.get()
            self.any_value_type = t.AnyValueType.get()
            self.any_param_type = t.AnyParamType.get()
            root_type = t.OperationType.get("builtin.module")

            self.builder.set_insertion_point_to_end(self.module.body)
            self.sequence = t.NamedSequenceOp(
                "__transform_main",
                [root_type],
                [],
                ip=self.builder._ip,
                loc=self.builder._loc,
            )

        entry = self.sequence.body
        self.root = self.sequence.bodyTarget
        yield_op = t.YieldOp([], ip=ir.InsertionPoint(entry), loc=self.builder._loc)
        # Subsequent primitives insert immediately before the terminator.
        self.builder.restore_insertion_point(ir.InsertionPoint(yield_op.operation))

        self._op_handles = {schedule.snapshot.root_id: self.root}
        self._value_handles = {}

    @property
    def kw(self) -> dict:
        """``ip``/``loc`` kwargs for upstream ODS op construction."""
        return {"ip": self.builder._ip, "loc": self.builder._loc}

    def set_callsite_loc(self) -> None:
        loc = capture_schedule_location()
        if loc is None:
            self.builder.set_unknown_loc()
            return
        self.builder.set_loc(
            ir.Location.file(loc.file_name, loc.line, loc.col + 1, self.context)
        )

    def op_handle(self, ref: OpRef) -> ir.Value:
        handle = self._op_handles.get(ref.id)
        if handle is not None:
            return handle

        node = self.schedule.snapshot.ops_by_id[ref.id]
        handle = ts.MatchOp(
            self.any_op_type,
            self.root,
            ops=[node.kind],
            op_attrs={SCHEDULE_ID_ATTR_NAME: ir.StringAttr.get(ref.id, self.context)},
            ip=self.builder._ip,
            loc=self.builder._loc,
        ).results[0]
        self._op_handles[ref.id] = handle
        return handle

    def value_handle(self, ref: BufferRef) -> ir.Value:
        handle = self._value_handles.get(ref.id)
        if handle is not None:
            return handle

        owner = self._owner_ref(ref)
        owner_handle = self.op_handle(owner)
        source_kind = {"arg": 1, "res": 2}[ref.source]
        handle = ta.MatchValueOp(
            self.any_value_type,
            owner_handle,
            ref.number,
            source_kind=source_kind,
            ip=self.builder._ip,
            loc=self.builder._loc,
        ).result
        self._value_handles[ref.id] = handle
        return handle

    def set_op_handle(self, ref: OpRef, handle: ir.Value) -> None:
        self._op_handles[ref.id] = handle

    def defining_op_handle(self, handle: ir.Value) -> ir.Value:
        return t.GetDefiningOp(
            self.any_op_type, handle, ip=self.builder._ip, loc=self.builder._loc
        ).result

    def _annotate(self, handle: ir.Value, name: str, value: str) -> None:
        # Upstream `transform.annotate` only attaches a param's value, so wrap the
        # static string in a `transform.param.constant` first.
        param = t.ParamConstantOp(
            self.any_param_type,
            ir.StringAttr.get(value, self.context),
            ip=self.builder._ip,
            loc=self.builder._loc,
        ).param
        t.AnnotateOp(
            handle, name, param=param, ip=self.builder._ip, loc=self.builder._loc
        )

    def annotate_schedule_id(self, handle: ir.Value, schedule_id: str) -> None:
        self._annotate(handle, SCHEDULE_ID_ATTR_NAME, schedule_id)

    def annotate_schedule_name(self, handle: ir.Value, schedule_name: str) -> None:
        self._annotate(handle, SCHEDULE_NAME_ATTR_NAME, schedule_name)

    def _owner_ref(self, ref: BufferRef) -> OpRef:
        node = self.schedule.snapshot.ops_by_id[ref.owner_id]
        return OpRef(
            id=node.id,
            epoch=self.schedule.epoch,
            kind=node.kind,
            name=node.name,
            path=node.path,
        )
