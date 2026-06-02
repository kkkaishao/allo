from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Protocol

from .errors import (
    AmbiguousLookupError,
    InvalidScheduleArgumentError,
    ScheduleLookupError,
    ScheduleTypeError,
)
from .model import (
    BufferRef,
    LoopRef,
    OpRef,
    PredictedOp,
    PredictedSnapshot,
    PredictedValue,
    Ref,
)
from ..._mlir.schedule import ScheduleOpTrait

RefKind = Literal["op", "loop", "buffer"]
Node = PredictedOp | PredictedValue


class QueryHost(Protocol):
    predicted: PredictedSnapshot
    _primary_path: str


def _candidate_note(node: Node) -> str:
    if isinstance(node, PredictedValue):
        return f"{node.key} (scope {node.scope}) value"
    return f"{node.key} (scope {node.scope}) kind={node.kind}"


class RefSelection:
    def __init__(
        self,
        predicted: PredictedSnapshot,
        nodes: Iterable[Node],
        ref_kind: RefKind,
        desc: str,
    ):
        self.predicted = predicted
        self.nodes: tuple[Node, ...] = tuple(nodes)
        self.ref_kind = ref_kind
        self.desc = desc

    def all(self) -> list[Ref]:
        return [self._make_ref(node) for node in self.nodes]

    def first(self) -> Ref:
        if not self.nodes:
            raise ScheduleLookupError(f"no {self.desc} matched the query")
        return self._make_ref(self.nodes[0])

    def one(self) -> Ref:
        if not self.nodes:
            raise ScheduleLookupError(f"no {self.desc} matched the query")
        if len(self.nodes) > 1:
            raise AmbiguousLookupError(
                f"{self.desc} query is ambiguous",
                notes=[_candidate_note(node) for node in self.nodes],
            )
        return self._make_ref(self.nodes[0])

    def names(self, *names: str) -> tuple[Ref, ...]:
        out: list[Ref] = []
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

    def _make_ref(self, node: Node) -> Ref:
        if isinstance(node, PredictedValue):
            return self.predicted.make_buffer_ref(node)
        if self.ref_kind == "loop":
            return self.predicted.make_loop_ref(node)
        return self.predicted.make_op_ref(node)


class Query:
    def __init__(self, host: QueryHost):
        self.host = host

    @property
    def predicted(self) -> PredictedSnapshot:
        return self.host.predicted

    # --- selections -------------------------------------------------------

    def op(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> RefSelection:
        return RefSelection(
            self.predicted,
            self._op_nodes(name=name, under=under, kind=kind, key=path),
            "op",
            "operation",
        )

    def loop(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> RefSelection:
        nodes = [
            node
            for node in self._op_nodes(name=name, under=under, key=path)
            if node.has_trait(ScheduleOpTrait.LOOP_LIKE)
        ]
        return RefSelection(self.predicted, nodes, "loop", "loop")

    def buffer(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> RefSelection:
        anchor = self._resolve_under(under)
        if path is not None:
            nodes = [v for v in self.predicted.values if v.key == path]
        elif name is None:
            nodes = list(self.predicted.values)
        else:
            nodes = self.predicted.values_by_name(name)
        if anchor is not None:
            nodes = [v for v in nodes if self._value_under(v, anchor.skey)]
        else:
            nodes = [v for v in nodes if v.scope == self.host._primary_path]
        return RefSelection(self.predicted, nodes, "buffer", "buffer")

    # --- target resolution ------------------------------------------------

    def resolve_op(self, target: OpRef | str, *, desc: str = "operation") -> OpRef:
        if isinstance(target, OpRef):
            self.predicted.require_live(target)
            return target
        if isinstance(target, str):
            ref = self.op(target).one()
            assert isinstance(ref, OpRef)
            return ref
        raise InvalidScheduleArgumentError(
            f"{desc} must be an operation ref or name, got {type(target).__name__}"
        )

    def resolve_loop(self, target: LoopRef | OpRef | str) -> LoopRef:
        if isinstance(target, LoopRef):
            self.predicted.require_live(target)
            return target
        if isinstance(target, OpRef):
            self.predicted.require_live(target)
            node = self.predicted.op(target.scope, target.key)
            if node is None or not node.has_trait(ScheduleOpTrait.LOOP_LIKE):
                raise ScheduleTypeError(
                    f"{target.describe()} is not loop-like and cannot be used as a loop"
                )
            return self.predicted.make_loop_ref(node)
        if isinstance(target, str):
            ref = self.loop(target).one()
            assert isinstance(ref, LoopRef)
            return ref
        raise InvalidScheduleArgumentError(
            f"loop target must be a loop ref or name, got {type(target).__name__}"
        )

    def resolve_buffer(self, target: BufferRef | str) -> BufferRef:
        if isinstance(target, BufferRef):
            self.predicted.require_live(target)
            return target
        if isinstance(target, str):
            ref = self.buffer(target).one()
            assert isinstance(ref, BufferRef)
            return ref
        raise InvalidScheduleArgumentError(
            f"buffer target must be a buffer ref or name, got {type(target).__name__}"
        )

    # --- internals --------------------------------------------------------

    def _op_nodes(
        self,
        *,
        name: str | None,
        under: OpRef | str | None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[PredictedOp]:
        anchor = self._resolve_under(under)
        if key is not None:
            scope = anchor.scope if anchor is not None else self.host._primary_path
            node = self.predicted.op(scope, key)
            nodes = [] if node is None else [node]
        elif name is None:
            nodes = list(self.predicted.ops)
        else:
            nodes = self.predicted.ops_by_name(name)
        if anchor is not None:
            nodes = [n for n in nodes if self.predicted.is_under(n, anchor.skey)]
        elif key is None:
            # Default scope: the primary function (not nested-callee loops).
            nodes = [n for n in nodes if n.scope == self.host._primary_path]
        if kind is not None:
            nodes = [node for node in nodes if node.kind == kind]
        return nodes

    def _resolve_under(self, under: OpRef | str | None) -> OpRef | None:
        if under is None:
            return None
        if isinstance(under, OpRef):
            self.predicted.require_live(under)
            return under
        if isinstance(under, str):
            ref = self.op(under).one()
            assert isinstance(ref, OpRef)
            return ref
        if isinstance(under, Ref):
            raise ScheduleTypeError(
                f"query scope must be an operation ref, got {under.describe()}"
            )
        raise InvalidScheduleArgumentError(
            f"query scope must be an operation ref or name, got {type(under).__name__}"
        )

    def _value_under(self, value: PredictedValue, ancestor: tuple[str, str]) -> bool:
        owner = self.predicted.op(*value.owner)
        if owner is None:
            return False
        return self.predicted.is_under(owner, ancestor)
