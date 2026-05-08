"""Common backend interfaces for the frontend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, NoReturn, Sequence, Literal, TYPE_CHECKING

from ..lang.kernel import Kernel

if TYPE_CHECKING:
    from .._C.ir import ModuleOp, OwningModuleOp


class BackendStage(str, Enum):
    """Execution stages shared by Allo backends."""

    SIMULATION = "simulation"
    SYNTHESIS = "synthesis"
    COSIMULATION = "cosimulation"
    IMPLEMENTATION = "implementation"


class UnsupportedStageError(RuntimeError):
    """Raised when a backend does not implement a requested stage."""


@dataclass(frozen=True)
class BackendConfig:
    """Backend-independent configuration.

    Concrete backends should subclass this dataclass for tool-specific options
    such as device, clock frequency, platform, or memory mapping.
    """

    project: Path | str | None = None
    verbose: bool = False
    env: Mapping[str, str] = field(default_factory=dict)

    def project_path(self) -> Path | None:
        if self.project is None:
            return None
        return Path(self.project)


@dataclass(frozen=True)
class BackendArtifact:
    """A file or directory produced by a backend stage."""

    name: str
    path: Path
    kind: Literal["file", "dir"] = "file"


@dataclass
class BackendReport:
    """Structured summary parsed from backend logs or report files."""

    stage: BackendStage
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, BackendArtifact] = field(default_factory=dict)


@dataclass
class BackendRunResult:
    """Result returned by a backend stage invocation."""

    stage: BackendStage
    returncode: int = 0
    command: Sequence[str] | None = None
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, BackendArtifact] = field(default_factory=dict)
    report: BackendReport | None = None


class Backend(ABC):
    """Base class for experimental Allo backends.

    A backend owns backend-specific lowering, project scaffolding, tool
    invocation, and report parsing. Frontend MLIR construction should stay
    outside this layer.
    """

    name: ClassVar[str] = "backend"
    supported_stages: ClassVar[frozenset[BackendStage]] = frozenset()

    def __init__(
        self,
        kernel: Kernel,
        config: BackendConfig = BackendConfig(),
    ):
        self.kernel = kernel
        self.config = config
        self.module: ModuleOp | None = None
        self._module_owner: OwningModuleOp | None = None

    def _get_working_module(self) -> ModuleOp:
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
        project: Path | str | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Create backend project files and return the project directory."""

    def supports(self, stage: BackendStage) -> bool:
        return stage in self.supported_stages

    def simulate(self, *args: Any, **kwargs: Any) -> BackendRunResult:
        return self._raise_unsupported(BackendStage.SIMULATION)

    def synth(self, **kwargs: Any) -> BackendRunResult:
        return self._raise_unsupported(BackendStage.SYNTHESIS)

    def cosim(self, *args: Any, **kwargs: Any) -> BackendRunResult:
        return self._raise_unsupported(BackendStage.COSIMULATION)

    def implement(self, **kwargs: Any) -> BackendRunResult:
        return self._raise_unsupported(BackendStage.IMPLEMENTATION)

    def report(self, stage: BackendStage | None = None) -> BackendReport:
        stage_suffix = "" if stage is None else f" for {stage.value}"
        raise UnsupportedStageError(
            f"{self.name} backend does not provide report parsing{stage_suffix}"
        )

    def _raise_unsupported(self, stage: BackendStage) -> NoReturn:
        raise UnsupportedStageError(
            f"{self.name} backend does not support {stage.value}"
        )
