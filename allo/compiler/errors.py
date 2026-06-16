# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

from ..diagnostics import DiagnosticError, DiagnosticLocation
from ..errors import AlloError


class CompilationError(DiagnosticError):
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

    def _source_line(self):
        if self.node is None or not hasattr(self.node, "lineno"):
            return None
        lines = self.src.splitlines()
        line_index = getattr(self.node, "lineno") - 1
        if line_index < 0 or line_index >= len(lines):
            return None
        return lines[line_index]

    def _location(self) -> DiagnosticLocation | None:
        if self.node is None or not hasattr(self.node, "lineno"):
            return None
        lineno = getattr(self.node, "lineno")
        col = getattr(self.node, "col_offset", 0)
        end_lineno = getattr(self.node, "end_lineno", lineno)
        end_col = getattr(self.node, "end_col_offset", col + 1)
        same_line = end_lineno == lineno
        span = end_col - col if same_line and end_col > col else 1
        return DiagnosticLocation(
            file_name=self.file_name or "<unknown>",
            line=self.begin_line + lineno - 1,
            col=col,
            source_line=self._source_line(),
            span=span,
        )

    def _diagnostic(self):
        return self.error_msg, self._location(), ()


class StaticAssertionError(CompilationError):
    pass


class InternalCompilerError(AlloError):
    """Raised when the compiler produces invalid IR or hits an inconsistent
    internal state. Signals a compiler bug, not a user error, so it is kept
    distinct from ``CompilationError`` and carries a detailed diagnostic
    message (typically including an IR dump)."""
