"""Common backend interfaces for the frontend."""

from __future__ import annotations

import atexit
import hashlib
import json
import os

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, ClassVar

from .._C import ir
from ..lang.kernel import Kernel

_PROCESS_CACHE: dict[tuple[str, str], Any] = {}
_CURRENT_BACKEND: ContextVar[Backend | None] = ContextVar(
    "allo_curr_backend",
    default=None,
)


def clear_process_cache() -> None:
    _PROCESS_CACHE.clear()


# avoid ModuleOp livetime issues
atexit.register(clear_process_cache)


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


def cache_root() -> Path:
    return Path.home() / ".allo" / "cache"


def current_backend() -> "Backend | None":
    return _CURRENT_BACKEND.get()


class Backend(ABC):
    """Base class for experimental Allo backends.

    A backend owns backend-specific lowering, project scaffolding, tool
    invocation, and report parsing. Frontend MLIR construction should stay
    outside this layer.
    """

    name: ClassVar[str] = "backend"

    def __init__(self, kernel: Kernel | None = None):
        self._kernel = kernel
        self.module: ir.ModuleOp | None = None
        self._module_owner: ir.OwningModuleOp | None = None
        self._context_tokens: list[Any] = []

    @property
    def kernel(self) -> Kernel:
        if self._kernel is None:
            raise RuntimeError(
                f"{self.__class__.__name__} backend is not bound to a kernel. "
                "Pass a kernel to the backend constructor or use the backend as "
                "a context manager around kernel calls."
            )
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: Kernel | None) -> None:
        self._kernel = kernel

    def __enter__(self):
        self._context_tokens.append(_CURRENT_BACKEND.set(self))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._context_tokens:
            _CURRENT_BACKEND.reset(self._context_tokens.pop())
        return False

    def _get_working_module(self) -> ir.ModuleOp:
        """Return a backend-owned module clone for backend-specific mutation."""
        if self.module is None:
            self._module_owner = self.kernel.compile().clone()
            self.module = self._module_owner.get()
        return self.module

    def _kernel_cache_payload(self) -> dict[str, Any]:
        module_text = str(self.kernel.compile())
        return {
            "top": self.kernel.func_name,
            "arg_types": [str(arg) for arg in self.kernel.parse_argument_annotations()],
            "res_types": [str(res) for res in self.kernel.parse_return_annotation()],
            "options": vars(self.kernel.options),
            "template_bindings": {
                name: str(value)
                for name, value in sorted(self.kernel.template_bindings.items())
            },
            "module_sha256": text_hash(module_text),
        }

    def _cache_key(self, *parts: Any) -> str:
        return stable_cache_hash(
            {
                "kernel": self._kernel_cache_payload(),
                "parts": parts,
            }
        )

    def _cache_dir(self, *parts: str) -> Path:
        return cache_root().joinpath(*parts)

    def _process_cache_get(self, namespace: str, key: str) -> Any | None:
        return _PROCESS_CACHE.get((namespace, key))

    def _process_cache_set(self, namespace: str, key: str, value: Any) -> None:
        _PROCESS_CACHE[(namespace, key)] = value

    def _process_cache_pop(self, namespace: str, key: str) -> Any | None:
        return _PROCESS_CACHE.pop((namespace, key), None)

    @abstractmethod
    def call_kernel(self, kernel: Kernel, *args, **kwargs) -> Any:
        """Run a kernel through this backend context."""

    @abstractmethod
    def compile(self) -> Any:
        """Run backend-specific lowering and return the lowered artifacts."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the backend and return the results.

        For CPU backend, the behavior of this method is to execute the compiled kernel and return the output.

        For hardware backend, the behavior of this method is to run the complete implementation flow, including
        synthesis and implementation.
        """

    @abstractmethod
    def scaffold_project(
        self,
        project: str | None = None,
        *,
        exist_ok: bool = True,
    ) -> Path:
        """Create backend project files and return the project directory."""
