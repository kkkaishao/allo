# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import io
import linecache
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.text import Text

from ..errors import AlloError


@dataclass(frozen=True)
class DiagnosticLocation:
    file_name: str
    line: int
    col: int = 0
    source_line: str | None = None
    span: int = 1


def _source_line(file_name: str, line: int) -> str | None:
    text = linecache.getline(file_name, line).rstrip("\n")
    return text if text else None


def _first_code_col(source_line: str | None) -> int:
    if source_line is None:
        return 0
    return len(source_line) - len(source_line.lstrip())


def _is_schedule_internal(file_name: str) -> bool:
    normalized = file_name.replace("\\", "/")
    return "/allo/exp/schedule/" in normalized


def _location_from_frame(frame) -> DiagnosticLocation:
    file_name = frame.f_code.co_filename
    line = frame.f_lineno
    source_line = _source_line(file_name, line)
    return DiagnosticLocation(
        file_name=file_name,
        line=line,
        col=_first_code_col(source_line),
        source_line=source_line,
    )


def capture_schedule_location() -> DiagnosticLocation | None:
    frame = inspect.currentframe()
    try:
        if frame is not None:
            frame = frame.f_back
        while frame is not None and _is_schedule_internal(frame.f_code.co_filename):
            frame = frame.f_back
        if frame is None:
            return None
        return _location_from_frame(frame)
    finally:
        del frame


class ScheduleError(AlloError):
    def __init__(
        self,
        message: str,
        *,
        location: DiagnosticLocation | None = None,
        notes: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.location = location or capture_schedule_location()
        self.notes = [] if notes is None else notes

    def __str__(self) -> str:
        return "\n" + self.render()

    def _use_color(self) -> bool:
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
            console.print(Text.assemble(("error", "bold red"), ": ", self.message))
            self._render_notes(console)
            return console.export_text(styles=color).rstrip()

        loc = self.location
        file_name = str(Path(loc.file_name))
        header = f"{file_name}:{loc.line}:{loc.col + 1}"
        console.print(
            Text.assemble(
                (header, "bold"),
                ": ",
                ("error", "bold red"),
                ": ",
                self.message,
            )
        )
        if loc.source_line is not None:
            line_width = len(str(loc.line))
            console.print(
                Text.assemble(
                    (f"{loc.line:>{line_width}}", "bold cyan"),
                    " | ",
                    loc.source_line,
                )
            )
            console.print(
                Text.assemble(
                    " " * line_width,
                    " | ",
                    " " * loc.col,
                    ("^" * max(1, loc.span), "bold green"),
                )
            )
        self._render_notes(console)
        return console.export_text(styles=color).rstrip()

    def _render_notes(self, console: Console) -> None:
        for note in self.notes:
            console.print(Text.assemble(("note", "bold cyan"), ": ", note))


class ScheduleLookupError(ScheduleError):
    pass


class AmbiguousLookupError(ScheduleLookupError):
    pass


class StaleRefError(ScheduleError):
    pass


class ConsumedHandleError(StaleRefError):
    pass


class ScheduleStateError(ScheduleError):
    pass


class ScheduleTypeError(ScheduleError):
    pass


class InvalidScheduleArgumentError(ScheduleError):
    pass


class ScheduleTransformError(ScheduleError):
    pass
