from dataclasses import dataclass
from enum import Enum

from . import ir

IDENTIFIER_ATTR_NAME: str = ...

def build_proxy(module: ir.ModuleOp) -> OperationProxy: ...
def create_operation_proxy(
    identifier: str, kind: str, hier_name: str
) -> OperationProxy: ...
def parse_from_string(context: ir.Context, s: str) -> ir.ModuleOp: ...
def parse_from_file(context: ir.Context, filename: str) -> ir.ModuleOp: ...
def complete_hierarchy_name(proxy: OperationProxy) -> str: ...

class ProxyState(Enum):
    VALID = ...
    INVALID = ...
    STALE = ...

VALID = ProxyState.VALID
INVALID = ProxyState.INVALID
STALE = ProxyState.STALE

@dataclass
class FrontendProxy:
    kind_str: str
    hierarchy_name: str
    identifier: str
    state: ProxyState

@dataclass
class OperationProxy(FrontendProxy):
    parent: OperationProxy | None
    children: list[OperationProxy]
    values: list[ValueProxy]

    def set_parent(self, parent: OperationProxy | None) -> None: ...
    def add_child(
        self, idx: int, id: str, kind: str, hier_name: str
    ) -> OperationProxy: ...
    def add_value(
        self, id: str, kind: str, hier_name: str, number: int = 0
    ) -> ValueProxy: ...
    def splice(self, src: OperationProxy, start: int = 0) -> None: ...

@dataclass
class ValueProxy(FrontendProxy):
    owner: OperationProxy
    number: int
