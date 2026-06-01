from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

from .errors import (
    AmbiguousLookupError,
    InvalidScheduleArgumentError,
    ScheduleLookupError,
    ScheduleTypeError,
)
from .model import (
    BufferRef,
    LoopRef,
    OpNode,
    OpRef,
    Ref,
    ScheduleSnapshot,
    ValueNode,
)
from ..._mlir.schedule import ScheduleOpTrait

RefT = TypeVar("RefT", OpRef, LoopRef, BufferRef)
Node = OpNode | ValueNode


class QueryHost(Protocol):
    epoch: int
    last_effect: str | None
    snapshot: ScheduleSnapshot


def _candidate_note(node: Node) -> str:
    loc = "" if node.loc is None else f" loc={node.loc.format()}"
    kind = getattr(node, "kind", getattr(node, "type", "value"))
    return f"{node.path} kind={kind}{loc}"


class RefSelection(Generic[RefT]):
    def __init__(
        self,
        schedule: QueryHost,
        nodes: Iterable[Node],
        ref_type: type[RefT],
        desc: str,
    ):
        self.schedule = schedule
        self.nodes = tuple(nodes)
        self.ref_type = ref_type
        self.desc = desc

    def all(self) -> list[RefT]:
        return [self._make_ref(node) for node in self.nodes]

    def first(self) -> RefT:
        if not self.nodes:
            raise ScheduleLookupError(f"no {self.desc} matched the query")
        return self._make_ref(self.nodes[0])

    def one(self) -> RefT:
        if not self.nodes:
            raise ScheduleLookupError(f"no {self.desc} matched the query")
        if len(self.nodes) > 1:
            raise AmbiguousLookupError(
                f"{self.desc} query is ambiguous",
                notes=[_candidate_note(node) for node in self.nodes],
            )
        return self._make_ref(self.nodes[0])

    def names(self, *names: str) -> tuple[RefT, ...]:
        out: list[RefT] = []
        for name in names:
            matches = [node for node in self.nodes if node.name == name]
            if not matches:
                raise ScheduleLookupError(f"no {self.desc} named '{name}'")
            if len(matches) > 1:
                raise AmbiguousLookupError(
                    f"{self.desc} name '{name}' is ambiguous",
                    notes=[_candidate_note(node) for node in matches],
                )
            out.append(self._make_ref(matches[0]))
        return tuple(out)

    def _make_ref(self, node: Node) -> RefT:
        epoch = self.schedule.epoch
        if isinstance(node, ValueNode):
            assert self.ref_type is BufferRef
            return BufferRef(
                id=node.id,
                epoch=epoch,
                kind="buffer",
                name=node.name,
                path=node.path,
                owner_id=node.owner_id,
                number=node.number,
                source=node.source,
            )
        assert isinstance(node, OpNode)
        if self.ref_type is LoopRef:
            return LoopRef(
                id=node.id,
                epoch=epoch,
                kind=node.kind,
                name=node.name,
                path=node.path,
            )
        assert self.ref_type is OpRef
        return OpRef(
            id=node.id,
            epoch=epoch,
            kind=node.kind,
            name=node.name,
            path=node.path,
        )


class Query:
    def __init__(self, schedule: QueryHost):
        self.schedule = schedule

    @property
    def snapshot(self) -> ScheduleSnapshot:
        return self.schedule.snapshot

    def op(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> RefSelection[OpRef]:
        return RefSelection(
            self.schedule,
            self._op_nodes(name=name, under=under, kind=kind, path=path),
            OpRef,
            "operation",
        )

    def loop(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> RefSelection[LoopRef]:
        nodes = [
            node
            for node in self._op_nodes(name=name, under=under, path=path)
            if node.has_trait(ScheduleOpTrait.LOOP_LIKE)
        ]
        return RefSelection(self.schedule, nodes, LoopRef, "loop")

    def loops(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> RefSelection[LoopRef]:
        return self.loop(name, under=under, path=path)

    def buffer(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> RefSelection[BufferRef]:
        anchor = self._resolve_under(under)
        if path is not None:
            node = self.snapshot.values_by_path.get(path)
            nodes = [] if node is None else [node]
        elif name is None:
            nodes = list(self.snapshot.values)
        else:
            nodes = list(self.snapshot.values_by_name.get(name, []))
        if anchor is not None:
            nodes = [
                node
                for node in nodes
                if self.snapshot.is_under(node.owner_id, anchor.id)
            ]
        return RefSelection(self.schedule, nodes, BufferRef, "buffer")

    def resolve_op(self, target: OpRef | str, *, desc: str = "operation") -> OpRef:
        if isinstance(target, OpRef):
            self.snapshot.require_live(target, last_effect=self.schedule.last_effect)
            return target
        if isinstance(target, str):
            return self.op(target).one()
        raise InvalidScheduleArgumentError(
            f"{desc} must be an operation ref or name, got {type(target).__name__}"
        )

    def resolve_loop(self, target: LoopRef | OpRef | str) -> LoopRef:
        if isinstance(target, LoopRef):
            self.snapshot.require_live(target, last_effect=self.schedule.last_effect)
            return target
        if isinstance(target, OpRef):
            self.snapshot.require_live(target, last_effect=self.schedule.last_effect)
            node = self.snapshot.ops_by_id[target.id]
            if not node.has_trait(ScheduleOpTrait.LOOP_LIKE):
                raise ScheduleTypeError(
                    f"{target.describe()} is not loop-like and cannot be used as a loop"
                )
            return LoopRef(
                target.id, target.epoch, target.kind, target.name, target.path
            )
        if isinstance(target, str):
            return self.loop(target).one()
        raise InvalidScheduleArgumentError(
            f"loop target must be a loop ref or name, got {type(target).__name__}"
        )

    def resolve_buffer(self, target: BufferRef | str) -> BufferRef:
        if isinstance(target, BufferRef):
            self.snapshot.require_live(target, last_effect=self.schedule.last_effect)
            return target
        if isinstance(target, str):
            return self.buffer(target).one()
        raise InvalidScheduleArgumentError(
            f"buffer target must be a buffer ref or name, got {type(target).__name__}"
        )

    def _op_nodes(
        self,
        *,
        name: str | None,
        under: OpRef | str | None,
        kind: str | None = None,
        path: str | None = None,
    ) -> list[OpNode]:
        anchor = self._resolve_under(under)
        if path is not None:
            node = self.snapshot.ops_by_path.get(path)
            nodes = [] if node is None else [node]
        elif name is None:
            nodes = list(self.snapshot.ops)
        else:
            nodes = list(self.snapshot.ops_by_name.get(name, []))
        if anchor is not None:
            nodes = [
                node for node in nodes if self.snapshot.is_under(node.id, anchor.id)
            ]
        if kind is not None:
            nodes = [node for node in nodes if node.kind == kind]
        return nodes

    def _resolve_under(self, under: OpRef | str | None) -> OpRef | None:
        if under is None:
            return None
        if isinstance(under, OpRef):
            self.snapshot.require_live(under, last_effect=self.schedule.last_effect)
            return under
        if isinstance(under, str):
            return self.op(under).one()
        if isinstance(under, Ref):
            raise ScheduleTypeError(
                f"query scope must be an operation ref, got {under.describe()}"
            )
        raise InvalidScheduleArgumentError(
            f"query scope must be an operation ref or name, got {type(under).__name__}"
        )
