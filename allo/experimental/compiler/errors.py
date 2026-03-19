# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import io
import sys
from typing import Optional, Union

from ..errors import AlloError

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    _HAS_RICH = True
except Exception:  # pragma: no cover - rich is in deps; keep fallback robust.
    _HAS_RICH = False


class VerificationError(AlloError):
    pass


def raise_compilation_warning(warning_msg: str):
    assert warning_msg, "Warning message cannot be empty."
    print(warning_msg + "\n", file=sys.stderr)


class CompilationError(AlloError):
    source_line_count_max_in_message = 12

    def __init__(
        self, node: Union[ast.AST, None], error_msg: str, src: Optional[str] = None
    ):
        self.node = node
        self.error_msg = error_msg
        self.src = src
        self.msg = self._format_message()

    @property
    def message(self) -> str:
        # Compatibility for historical callsites accessing `e.message`.
        return self.error_msg

    def _get_location(self) -> tuple[int | None, int | None]:
        if self.node is None or not hasattr(self.node, "lineno"):
            return None, None
        line = int(self.node.lineno)
        col = int(getattr(self.node, "col_offset", 0))
        return line, max(0, col)

    def _source_window(self, line: int | None) -> tuple[int, int]:
        if self.src is None:
            return 1, 1
        lines = self.src.splitlines()
        if not lines:
            return 1, 1
        if line is None:
            return 1, min(len(lines), self.source_line_count_max_in_message)
        half = max(1, self.source_line_count_max_in_message // 2)
        start = max(1, line - half)
        end = min(len(lines), line + half)
        return start, end

    def _source_line_with_caret(self, line: int | None, col: int | None) -> str:
        if self.src is None or line is None:
            return ""
        lines = self.src.splitlines()
        if line < 1 or line > len(lines):
            return ""
        src_line = lines[line - 1].expandtabs(4)
        caret_col = max(0, min(col if col is not None else 0, len(src_line)))
        return f"{line:4} | {src_line}\n" f"     | {' ' * caret_col}^"

    def _format_plain_message(self) -> str:
        line, col = self._get_location()
        location = (
            f"at {line}:{(col + 1) if col is not None else 1}"
            if line is not None
            else "at <unknown location>"
        )
        parts = [location]

        snippet = self._source_line_with_caret(line, col)
        if snippet:
            parts.append(snippet)
        elif self.src:
            parts.append(self.src)
        else:
            parts.append("<source unavailable>")

        if self.error_msg:
            parts.append(self.error_msg)
        return "\n".join(parts)

    def _format_rich_message(self) -> Optional[str]:
        if not _HAS_RICH:
            return None
        try:
            line, col = self._get_location()
            use_color = bool(getattr(sys.stderr, "isatty", lambda: False)())
            console = Console(
                record=True,
                file=io.StringIO(),
                force_terminal=use_color,
                color_system="auto" if use_color else None,
                width=80,
            )

            summary = Text()
            if line is not None:
                summary.append(
                    f"at line {line}, column {(col or 0) + 1}\n", style="yellow"
                )
            else:
                summary.append("at unknown source location\n", style="yellow")
            if self.error_msg:
                summary.append(self.error_msg, style="bold red")
            else:
                summary.append("Compilation failed.", style="bold red")

            title = (
                "Compile-Time Assertion Failed"
                if isinstance(self, CompileTimeAssertionFailure)
                else "Compilation Error"
            )
            console.print(Panel(summary, title=title, border_style="red"))

            if self.src is not None and self.src.strip():
                start, end = self._source_window(line)
                highlight = {line} if line is not None else set()
                syntax = Syntax(
                    self.src,
                    "python",
                    line_numbers=True,
                    line_range=(start, end),
                    highlight_lines=highlight,
                    word_wrap=False,
                    tab_size=2,
                    theme="github-dark",
                )
                console.print(
                    Panel(syntax, title="Source Context", border_style="blue")
                )

                pointer = self._source_line_with_caret(line, col)
                if pointer:
                    console.print(
                        Panel(pointer, title="Precise Location", border_style="magenta")
                    )

            rendered = console.export_text(styles=use_color).rstrip()
            # Keep the top border aligned when Python prints
            # "ExceptionType: <message>" on one line.
            return "\n" + rendered
        except Exception:
            return None

    def _format_message(self) -> str:
        rich_msg = self._format_rich_message()
        if rich_msg is not None:
            return rich_msg
        return self._format_plain_message()

    def __str__(self):
        return self.msg

    def __reduce__(self):
        return type(self), (self.node, self.error_msg, self.src)


class CompileTimeAssertionFailure(CompilationError):
    pass
