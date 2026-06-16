# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import linecache
from pathlib import Path

from ..diagnostics import DiagnosticError, DiagnosticLocation, first_code_col


def _source_line(file_name: str, line: int) -> str | None:
    text = linecache.getline(file_name, line).rstrip("\n")
    return text if text else None


def _is_schedule_internal(file_name: str) -> bool:
    normalized = file_name.replace("\\", "/")
    return "/allo/schedule/" in normalized


def _location_from_frame(frame) -> DiagnosticLocation:
    file_name = frame.f_code.co_filename
    line = frame.f_lineno
    source_line = _source_line(file_name, line)
    return DiagnosticLocation(
        file_name=file_name,
        line=line,
        col=first_code_col(source_line),
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


class ScheduleError(DiagnosticError):
    render_width = 4096

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

    def _diagnostic(self):
        loc = self.location
        if loc is not None:
            loc = DiagnosticLocation(
                file_name=str(Path(loc.file_name)),
                line=loc.line,
                col=loc.col,
                source_line=loc.source_line,
                span=loc.span,
            )
        return self.message, loc, self.notes


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
