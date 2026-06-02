import ast
import inspect
import io
import linecache
from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.text import Text

from ..errors import AlloError


class CompilationError(AlloError):
    def __init__(
        self,
        src: str,
        error_msg: str,
        node: ast.AST | None = None,
        *,
        file_name: str | None = None,
        begin_line: int = 1,
    ):
        self.src = src
        self.error_msg = error_msg
        self.node = node
        self.file_name = file_name
        self.begin_line = begin_line

    def __str__(self):
        return "\n" + self.render()

    def _source_line(self):
        if self.node is None or not hasattr(self.node, "lineno"):
            return None
        lines = self.src.splitlines()
        line_index = getattr(self.node, "lineno") - 1
        if line_index < 0 or line_index >= len(lines):
            return None
        return lines[line_index]

    def _location(self):
        if self.node is None or not hasattr(self.node, "lineno"):
            return None
        lineno = getattr(self.node, "lineno")
        col = getattr(self.node, "col_offset", 0)
        end_lineno = getattr(self.node, "end_lineno", lineno)
        end_col = getattr(self.node, "end_col_offset", col + 1)
        return self.begin_line + lineno - 1, col, end_lineno == lineno, end_col

    def _use_color(self):
        return Console(stderr=True).is_terminal

    def render(self, *, color: bool | None = None) -> str:
        if color is None:
            color = self._use_color()

        console = Console(
            file=io.StringIO(),
            record=True,
            force_terminal=color,
            color_system="auto" if color else None,
            width=120,
        )
        location = self._location()
        if location is None:
            console.print(
                Text.assemble(("error", "bold red"), ": ", str(self.error_msg))
            )
            return console.export_text(styles=color).rstrip()

        abs_lineno, col, same_line, end_col = location
        file_name = self.file_name or "<unknown>"
        header = f"{file_name}:{abs_lineno}:{col + 1}"
        console.print(
            Text.assemble(
                (header, "bold"),
                ": ",
                ("error", "bold red"),
                ": ",
                str(self.error_msg),
            )
        )

        source_line = self._source_line()
        if source_line is not None:
            line_no_width = len(str(abs_lineno))
            console.print(
                Text.assemble(
                    (f"{abs_lineno:>{line_no_width}}", "bold cyan"),
                    " | ",
                    source_line,
                )
            )
            span = end_col - col if same_line and end_col > col else 1
            console.print(
                Text.assemble(
                    " " * line_no_width,
                    " | ",
                    " " * col,
                    ("^" * max(1, span), "bold green"),
                )
            )
        return console.export_text(styles=color).rstrip()


class StaticAssertionError(CompilationError):
    pass


class InternalCompilerError(AlloError):
    """Raised when the compiler produces invalid IR or hits an inconsistent
    internal state. Signals a compiler bug, not a user error, so it is kept
    distinct from ``CompilationError`` and carries a detailed diagnostic
    message (typically including an IR dump)."""


@dataclass(frozen=True)
class DiagnosticLocation:
    file_name: str
    line: int
    col: int = 0
    source_line: str | None = None
    span: int = 1


_ACT_INTERNAL_SUFFIXES = (
    "/allo/exp/compiler/errors.py",
    "/allo/exp/compiler/act_codegen.py",
    "/allo/exp/lang/act.py",
    "/allo/exp/operators/act.py",
)


def _first_code_col(source_line: str | None) -> int:
    if source_line is None:
        return 0
    return len(source_line) - len(source_line.lstrip())


def _source_line(file_name: str, line: int) -> str | None:
    text = linecache.getline(file_name, line).rstrip("\n")
    return text if text else None


def _is_act_internal(file_name: str) -> bool:
    file_name = file_name.replace("\\", "/")
    return any(file_name.endswith(suffix) for suffix in _ACT_INTERNAL_SUFFIXES)


def _location_from_frame(frame) -> DiagnosticLocation:
    file_name = frame.f_code.co_filename
    line = frame.f_lineno
    source_line = _source_line(file_name, line)
    return DiagnosticLocation(
        file_name, line, _first_code_col(source_line), source_line
    )


def capture_act_location() -> DiagnosticLocation | None:
    frame = inspect.currentframe()
    try:
        if frame is not None:
            frame = frame.f_back
        while frame is not None and _is_act_internal(frame.f_code.co_filename):
            frame = frame.f_back
        if frame is None:
            return None
        return _location_from_frame(frame)
    finally:
        del frame


def callable_diagnostic_location(
    fn: Callable, *, marker: str | None = None
) -> DiagnosticLocation | None:
    try:
        lines, begin_line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        return None

    file_name = inspect.getsourcefile(fn) or fn.__code__.co_filename
    line_index = 0
    if marker is not None:
        for i, line in enumerate(lines):
            if marker in line:
                line_index = i
                break
    else:
        for i, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                line_index = i
                break

    source_line = lines[line_index].rstrip("\n")
    col = _first_code_col(source_line)
    if marker is not None and marker in source_line:
        col = source_line.index(marker)
    return DiagnosticLocation(file_name, begin_line + line_index, col, source_line)


class ActError(AlloError):
    def __init__(
        self,
        error_msg: str,
        *,
        location: DiagnosticLocation | None = None,
    ):
        super().__init__(error_msg)
        self.error_msg = error_msg
        self.location = location or capture_act_location()

    def __str__(self):
        return "\n" + self.render()

    def attach_location(
        self, location: DiagnosticLocation | None, *, override: bool = False
    ):
        if location is not None and (override or self.location is None):
            self.location = location
        return self

    def _use_color(self):
        return Console(stderr=True).is_terminal

    def render(self, *, color: bool | None = None) -> str:
        if color is None:
            color = self._use_color()

        console = Console(
            file=io.StringIO(),
            record=True,
            force_terminal=color,
            color_system="auto" if color else None,
            width=4096,
        )
        if self.location is None:
            console.print(
                Text.assemble(("error", "bold red"), ": ", str(self.error_msg))
            )
            return console.export_text(styles=color).rstrip()

        location = self.location
        header = f"{location.file_name}:{location.line}:{location.col + 1}"
        console.print(
            Text.assemble(
                (header, "bold"),
                ": ",
                ("error", "bold red"),
                ": ",
                str(self.error_msg),
            )
        )

        if location.source_line is not None:
            line_no_width = len(str(location.line))
            console.print(
                Text.assemble(
                    (f"{location.line:>{line_no_width}}", "bold cyan"),
                    " | ",
                    location.source_line,
                )
            )
            console.print(
                Text.assemble(
                    " " * line_no_width,
                    (" | ", "bold cyan"),
                    " " * location.col,
                    ("^" * max(1, location.span), "bold green"),
                )
            )
        return console.export_text(styles=color).rstrip()
