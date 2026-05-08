"""Common backend interfaces for the frontend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, NoReturn, Sequence, Literal

from .._C import ir
from ..lang.kernel import Kernel


class Backend(ABC):
    """Base class for experimental Allo backends.

    A backend owns backend-specific lowering, project scaffolding, tool
    invocation, and report parsing. Frontend MLIR construction should stay
    outside this layer.
    """

    name: ClassVar[str] = "backend"

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.module: ir.ModuleOp | None = None
        self._module_owner: ir.OwningModuleOp | None = None

    def _get_working_module(self) -> ir.ModuleOp:
        """Return a backend-owned module clone for backend-specific mutation."""
        if self.module is None:
            self._module_owner = self.kernel.compile().clone()
            self.module = self._module_owner.get()
        return self.module

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
        overwrite: bool = False,
    ) -> Path:
        """Create backend project files and return the project directory."""
