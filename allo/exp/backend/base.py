"""Common backend interfaces for the frontend."""

from __future__ import annotations

import hashlib
import json
import os

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, ParamSpec, TypeVar

from ..._mlir import ir
from ..._mlir._mlir_libs._allo import ir_ext
from ..._mlir.ir import SymbolTable, UnitAttr
from ..._mlir.passmanager import PassManager
from ..._mlir.dialects.allo import register_passes as _register_allo_passes
from ..lang.kernel import Kernel

# Allo passes live in the process-global MLIR pass registry; register them once
# (std::call_once-guarded in C++) so backend pipelines (`allo-lower-to-llvm`,
# `grid-mapping`, `convert-allo-to-func`, ...) resolve via upstream PassManager.
_register_allo_passes()


def lookup_kernel(module: ir.Module, name: str):
    """Return the top-level kernel op named ``name`` (an OpView) or ``None``."""
    try:
        return SymbolTable(module.operation)[name]
    except KeyError:
        return None


def set_top_llvm_c_wrapper(module: ir.Module, name: str):
    op = lookup_kernel(module, name)
    if op is None:
        return False
    op.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get(module.context)
    return True


def run_pipeline(module: ir.Module, pipeline: str) -> None:
    """Run a textual pass pipeline on ``module`` in its own context."""
    PassManager.parse(pipeline, module.context).run(module.operation)


_PROCESS_CACHE: dict[tuple[str, str], Any] = {}
_DEFAULT_CACHE_DIR = Path.home() / ".allo" / "cache"


def clear_process_cache() -> None:
    _PROCESS_CACHE.clear()


def _normalize_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_cache_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_cache_value(item) for item in value]
    return str(value)


def stable_cache_json(value: Any) -> str:
    return json.dumps(
        _normalize_cache_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_cache_hash(value: Any) -> str:
    return hashlib.sha256(stable_cache_json(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text_if_changed(path: str | os.PathLike[str], text: str) -> bool:
    output = Path(path)
    if output.exists() and output.read_text(encoding="utf-8") == text:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return True


def write_json_if_changed(path: str | os.PathLike[str], value: Any) -> bool:
    return write_text_if_changed(path, stable_cache_json(value) + "\n")


P = ParamSpec("P")
R = TypeVar("R")


class Backend(ABC, Generic[P, R]):
    """Base class for experimental Allo backends.

    A backend owns backend-specific lowering, project scaffolding, tool
    invocation, and report parsing. Frontend MLIR construction should stay
    outside this layer.
    """

    name: ClassVar[str] = "backend"

    def __init__(self, kernel: Kernel[P, R]):
        self.module: ir.Module = ir_ext.clone_module(kernel.compile())
        self.kernel = kernel
        self._kernel_cache = self._compute_kernel_cache()

    def _compute_kernel_cache(self) -> dict[str, Any]:
        return {
            "top": self.kernel.func_name,
            "arg_types": [str(arg) for arg in self.kernel.parse_argument_annotations()],
            "res_types": [str(res) for res in self.kernel.parse_return_annotation()],
            "options": vars(self.kernel.options),
            "template_bindings": {
                name: str(value)
                for name, value in sorted(self.kernel.template_bindings.items())
            },
            "module_sha256": text_hash(str(self.module)),
        }

    def _cache_key(self, *parts: Any) -> str:
        return stable_cache_hash(
            {
                "kernel": self._compute_kernel_cache(),
                "parts": parts,
            }
        )

    def _cache_dir(self, *parts: str) -> Path:
        return _DEFAULT_CACHE_DIR.joinpath(*parts)

    def _pcache_get(self, namespace: str, key: str) -> Any | None:
        return _PROCESS_CACHE.get((namespace, key))

    def _pcache_set(self, namespace: str, key: str, value: Any) -> None:
        _PROCESS_CACHE[(namespace, key)] = value

    def _pcache_pop(self, namespace: str, key: str) -> Any | None:
        return _PROCESS_CACHE.pop((namespace, key), None)

    def _process_cached(
        self, namespace: str, key: str, factory: Callable[[], Any]
    ) -> Any:
        """Return the cached value for ``(namespace, key)`` or build it once."""
        value = _PROCESS_CACHE.get((namespace, key))
        if value is None:
            value = factory()
            _PROCESS_CACHE[(namespace, key)] = value
        return value

    @abstractmethod
    def compile(self) -> Any:
        """Run backend-specific lowering and return the lowered artifacts."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the backend and return the results."""

    @abstractmethod
    def scaffold_project(
        self,
        project: str | None = None,
        *,
        exist_ok: bool = True,
    ) -> Path:
        """Create backend project files and return the project directory."""
