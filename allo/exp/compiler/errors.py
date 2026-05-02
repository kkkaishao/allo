import ast
import io
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
                    (" | ", "bold cyan"),
                    " " * col,
                    ("^" * max(1, span), "bold green"),
                )
            )
        return console.export_text(styles=color).rstrip()


class StaticAssertionError(CompilationError):
    pass
