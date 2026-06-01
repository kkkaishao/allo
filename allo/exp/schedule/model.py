from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .errors import StaleRefError
from ..._mlir.schedule import ScheduleOpTrait


@dataclass(frozen=True)
class Effect:
    id: str
    name: str
    epoch: int
    topology: bool
    data: dict[str, Any]


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int

    @classmethod
    def from_raw(cls, raw: Any) -> SourceLoc | None:
        if raw is None:
            return None
        return cls(str(raw["file"]), int(raw["line"]), int(raw["col"]))

    def format(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass(frozen=True)
class Ref:
    id: str
    epoch: int
    kind: str
    name: str | None
    path: str

    def display_name(self) -> str:
        return self.name or self.path

    def describe(self) -> str:
        name = self.display_name()
        return f"{self.__class__.__name__}('{name}', path='{self.path}')"


@dataclass(frozen=True)
class OpRef(Ref):
    pass


@dataclass(frozen=True)
class LoopRef(OpRef):
    pass


@dataclass(frozen=True)
class BufferRef(Ref):
    owner_id: str
    number: int
    source: Literal["arg", "res"]


@dataclass(frozen=True)
class OpNode:
    id: str
    kind: str
    name: str | None
    path: str
    parent_id: str | None
    children: tuple[str, ...]
    loc: SourceLoc | None
    traits: ScheduleOpTrait

    def display_name(self) -> str:
        return self.name or self.path

    def has_trait(self, trait: ScheduleOpTrait) -> bool:
        return bool(self.traits & trait)


@dataclass(frozen=True)
class ValueNode:
    id: str
    owner_id: str
    name: str | None
    type: str
    number: int
    source: Literal["arg", "res"]
    path: str
    loc: SourceLoc | None

    def display_name(self) -> str:
        return self.name or self.path


SingleTarget = Ref | str
Targets = SingleTarget | Iterable[SingleTarget] | None


class ScheduleSnapshot:
    def __init__(
        self,
        *,
        epoch: int,
        root_id: str,
        ops: list[OpNode],
        values: list[ValueNode],
    ):
        self.epoch = epoch
        self.root_id = root_id
        self.ops = tuple(ops)
        self.values = tuple(values)

        self.ops_by_id = {node.id: node for node in self.ops}
        self.values_by_id = {node.id: node for node in self.values}
        assert len(self.ops_by_id) == len(self.ops), "duplicate operation ids"
        assert len(self.values_by_id) == len(self.values), "duplicate value ids"
        assert self.root_id in self.ops_by_id, "root operation id missing"

        self.ops_by_name: dict[str, list[OpNode]] = defaultdict(list)
        self.values_by_name: dict[str, list[ValueNode]] = defaultdict(list)
        self.ops_by_path: dict[str, OpNode] = {}
        self.values_by_path: dict[str, ValueNode] = {}

        for node in self.ops:
            if node.name is not None:
                self.ops_by_name[node.name].append(node)
            self.ops_by_path[node.path] = node
        for node in self.values:
            if node.name is not None:
                self.values_by_name[node.name].append(node)
            self.values_by_path[node.path] = node

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, epoch: int) -> ScheduleSnapshot:
        ops = [
            OpNode(
                id=str(item["id"]),
                kind=str(item["kind"]),
                name=None if item["name"] is None else str(item["name"]),
                path=str(item["path"]),
                parent_id=(
                    None if item["parent_id"] is None else str(item["parent_id"])
                ),
                children=tuple(str(child) for child in item["children"]),
                loc=SourceLoc.from_raw(item["loc"]),
                traits=ScheduleOpTrait(int(item["traits"])),
            )
            for item in raw["ops"]
        ]
        values = [
            ValueNode(
                id=str(item["id"]),
                owner_id=str(item["owner_id"]),
                name=None if item["name"] is None else str(item["name"]),
                type=str(item["type"]),
                number=int(item["number"]),
                source=item["source"],
                path=str(item["path"]),
                loc=SourceLoc.from_raw(item["loc"]),
            )
            for item in raw["values"]
        ]
        return cls(epoch=epoch, root_id=str(raw["root_id"]), ops=ops, values=values)

    def require_live(self, ref: Ref, *, last_effect: str | None = None) -> None:
        if ref.epoch != self.epoch:
            notes = []
            if last_effect is not None:
                notes.append(f"last topology-changing transform: {last_effect}")
            raise StaleRefError(
                f"{ref.describe()} is stale: created at epoch {ref.epoch}, "
                f"current epoch is {self.epoch}",
                notes=notes,
            )
        if isinstance(ref, BufferRef):
            assert ref.id in self.values_by_id, "live buffer ref id missing"
        elif isinstance(ref, OpRef):
            assert ref.id in self.ops_by_id, "live operation ref id missing"
        else:
            assert False, f"unsupported ref type: {type(ref)}"

    def op_ref(self, op_id: str) -> OpRef:
        node = self.ops_by_id[op_id]
        return OpRef(
            id=node.id,
            epoch=self.epoch,
            kind=node.kind,
            name=node.name,
            path=node.path,
        )

    def loop_ref(self, op_id: str) -> LoopRef:
        node = self.ops_by_id[op_id]
        assert node.has_trait(ScheduleOpTrait.LOOP_LIKE), "operation is not loop-like"
        return LoopRef(
            id=node.id,
            epoch=self.epoch,
            kind=node.kind,
            name=node.name,
            path=node.path,
        )

    def buffer_ref(self, value_id: str) -> BufferRef:
        node = self.values_by_id[value_id]
        return BufferRef(
            id=node.id,
            epoch=self.epoch,
            kind="buffer",
            name=node.name,
            path=node.path,
            owner_id=node.owner_id,
            number=node.number,
            source=node.source,
        )

    def is_under(self, op_id: str, ancestor_id: str) -> bool:
        current = self.ops_by_id.get(op_id)
        while current is not None:
            if current.id == ancestor_id:
                return True
            if current.parent_id is None:
                return False
            current = self.ops_by_id.get(current.parent_id)
        return False

    def depth(self, op_id: str) -> int:
        depth = 0
        current = self.ops_by_id[op_id]
        while current.parent_id is not None:
            depth += 1
            current = self.ops_by_id[current.parent_id]
        return depth

    def format_tree(self, *, include_values: bool = True) -> str:
        lines: list[str] = []

        def append_node(node: OpNode, prefix: str, is_last: bool) -> None:
            marker = "" if node.id == self.root_id else ("`- " if is_last else "|- ")
            child_prefix = prefix
            if node.id != self.root_id:
                child_prefix += "   " if is_last else "|  "
            name = node.display_name()
            loc = "" if node.loc is None else f" loc={node.loc.format()}"
            lines.append(
                f"{prefix}{marker}{name} kind={node.kind} path={node.path} "
                f"id={node.id}{loc}"
            )
            if include_values:
                node_values = [
                    value for value in self.values if value.owner_id == node.id
                ]
                for idx, value in enumerate(node_values):
                    value_last = idx == len(node_values) - 1 and not node.children
                    value_marker = "`- " if value_last else "|- "
                    lines.append(
                        f"{child_prefix}{value_marker}{value.display_name()} "
                        f"type={value.type} path={value.path} id={value.id}"
                    )
            for idx, child_id in enumerate(node.children):
                child = self.ops_by_id[child_id]
                append_node(child, child_prefix, idx == len(node.children) - 1)

        append_node(self.ops_by_id[self.root_id], "", True)
        return "\n".join(lines)
