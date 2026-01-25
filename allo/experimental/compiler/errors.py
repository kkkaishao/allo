import ast
from typing import Optional, Union
from ..errors import AlloError


class VerificationError(AlloError):
    pass


class CompilationError(AlloError):
    source_line_count_max_in_message = 12

    def __init__(
        self, node: Union[ast.AST, None], error_msg: str, src: Optional[str] = None
    ):
        self.node = node
        self.error_msg = error_msg
        self.src = src
        self.msg = self._format_message()

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = ""
        else:
            if hasattr(node, "lineno"):
                source_excerpt = self.src.split("\n")[: node.lineno][
                    -self.source_line_count_max_in_message :
                ]
                if source_excerpt:
                    source_excerpt.append(" " * node.col_offset + "^")
                    source_excerpt = "\n".join(source_excerpt)
                else:
                    source_excerpt = " <source empty>"
            else:
                source_excerpt = self.src

        message = (
            "at {}:{}:\n{}".format(node.lineno, node.col_offset, source_excerpt)
            if hasattr(node, "lineno")
            else source_excerpt
        )
        if self.error_msg:
            message += "\n" + self.error_msg
        return message

    def __str__(self):
        return self.msg

    def __reduce__(self):
        return type(self), (self.node, self.error_msg, self.src)


class CompileTimeAssertionFailure(CompilationError):
    pass
