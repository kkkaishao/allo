# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..._mlir.ir import Module, StringAttr, WalkResult
from ..._mlir.schedule import (
    ScheduleOpTrait,
    SCHEDULE_ID_ATTR_NAME,
)

if TYPE_CHECKING:
    from .model import ScheduleSnapshot

# The frontend-owned attribute used as the ``transform.structured.match`` filter.
# Unlike ``allo.schedule.id`` (globally unique, regenerated every apply), a key is a
# *bare* identifier relative to its enclosing function, derived deterministically from
# the op's user name or scope-relative structure and never embedding a global counter.
# Matching is rooted at the function handle, so a bare key is unambiguous within a
# function even when the module holds several functions (a parent and its callee copies).
# That is what lets ``.compose()`` run a callee's program verbatim on a renamed copy.
SCHEDULE_KEY_ATTR_NAME = "allo.schedule.key"


def derived_key(base_key: str, role: str) -> str:
    """Deterministic key for an op a primitive creates (e.g. ``i`` -> ``i.inner``).

    No global counter: the same sequence re-run on a copy yields the same key.
    """
    return f"{base_key}.{role}"


@dataclass(frozen=True)
class KeyTables:
    """Derived key maps for one real snapshot."""

    scope_by_id: dict[str, str]
    relkey_by_id: dict[str, str]
    # (scope, relkey) -> id.
    id_by_scope_key: dict[tuple[str, str], str]


def _enclosing_function_path(ops_by_id: dict, node) -> str | None:
    parent_id = node.parent_id
    while parent_id is not None:
        parent = ops_by_id.get(parent_id)
        if parent is None:
            return None
        if parent.has_trait(ScheduleOpTrait.FUNCTION_LIKE):
            return parent.path
        parent_id = parent.parent_id
    return None


def scope_of(ops_by_id: dict, root_id: str, node) -> str:
    """The scope a node's key is relative to: its nearest enclosing function, else
    the module root (for the function and root nodes themselves)."""
    func_path = _enclosing_function_path(ops_by_id, node)
    if func_path is not None:
        return func_path
    return ops_by_id[root_id].path


def assign_keys(
    snapshot: "ScheduleSnapshot", stamped: dict[str, str] | None = None
) -> KeyTables:
    """Compute the scope and bare relative key for every op in a real snapshot.

    When an op already carries a stamped ``allo.schedule.key`` (passed in ``stamped``
    as id -> key), that value wins — this is how derived keys (``i.inner``) and
    user keys persist across applies. Otherwise the key is derived structurally: the
    op's user name when unique in its scope, else its scope-relative path (which the
    C++ collector makes positionally unique via ``L<idx>``/``O<idx>``/symbol segments).
    """
    stamped = stamped or {}
    ops_by_id = snapshot.ops_by_id
    root_id = snapshot.root_id

    scope_by_id: dict[str, str] = {}
    name_counts: dict[tuple[str, str], int] = {}
    for node in snapshot.ops:
        scope = scope_of(ops_by_id, root_id, node)
        scope_by_id[node.id] = scope
        if node.name is not None:
            name_counts[(scope, node.name)] = name_counts.get((scope, node.name), 0) + 1

    relkey_by_id: dict[str, str] = {}
    id_by_scope_key: dict[tuple[str, str], str] = {}
    for node in snapshot.ops:
        scope = scope_by_id[node.id]
        if node.id in stamped:
            relkey = stamped[node.id]
        elif node.name is not None and name_counts[(scope, node.name)] == 1:
            relkey = node.name
        elif node.path == scope:
            relkey = node.path.rsplit("/", 1)[-1]
        else:
            prefix = scope + "/"
            assert node.path.startswith(
                prefix
            ), f"node path '{node.path}' not under scope '{scope}'"
            relkey = node.path[len(prefix) :]
        relkey_by_id[node.id] = relkey
        id_by_scope_key[(scope, relkey)] = node.id

    return KeyTables(
        scope_by_id=scope_by_id,
        relkey_by_id=relkey_by_id,
        id_by_scope_key=id_by_scope_key,
    )


def read_schedule_keys(module: Module) -> dict[str, str]:
    """Walk the payload and return id -> bare key for ops carrying both
    ``allo.schedule.id`` and ``allo.schedule.key`` (the keys persisted on the IR)."""
    out: dict[str, str] = {}

    def visit(op):
        attrs = op.attributes
        if SCHEDULE_ID_ATTR_NAME in attrs and SCHEDULE_KEY_ATTR_NAME in attrs:
            sid = StringAttr(attrs[SCHEDULE_ID_ATTR_NAME]).value
            out[sid] = StringAttr(attrs[SCHEDULE_KEY_ATTR_NAME]).value
        return WalkResult.ADVANCE

    module.operation.walk(visit)
    return out


def annotate_schedule_keys(module: Module, relkey_by_id: dict[str, str]) -> None:
    """Stamp the bare ``allo.schedule.key`` onto every payload op that lacks one.

    Idempotent: derived keys written by the transform sequence (via
    ``transform.annotate``) are preserved, so created ops keep stable keys across
    applies. Pre-existing ops are keyed from ``relkey_by_id`` (indexed by the C++
    ``allo.schedule.id`` already present on the op).
    """
    ctx = module.context

    def visit(op):
        attrs = op.attributes
        if SCHEDULE_KEY_ATTR_NAME in attrs:
            return WalkResult.ADVANCE
        if SCHEDULE_ID_ATTR_NAME not in attrs:
            return WalkResult.ADVANCE
        sid = StringAttr(attrs[SCHEDULE_ID_ATTR_NAME]).value
        relkey = relkey_by_id.get(sid)
        if relkey is not None:
            attrs[SCHEDULE_KEY_ATTR_NAME] = StringAttr.get(relkey, ctx)
        return WalkResult.ADVANCE

    module.operation.walk(visit)
