# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import subprocess

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, NoReturn, TypeVar, cast

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

console = Console(stderr=True)
F = TypeVar("F", bound=Callable[..., Any])
ErrorCallback = Callable[[Exception], None]
ExitCallback = Callable[[], None]


def captured_output(stdout: str, stderr: str) -> str:
    return "\n".join(stream.rstrip() for stream in (stdout, stderr) if stream.strip())


def text_tail(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


@dataclass
class CommandError(RuntimeError):
    cmd: Sequence[str | os.PathLike[str]]
    returncode: int
    cwd: str | os.PathLike[str] | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return captured_output(self.stdout, self.stderr)

    def output_tail(self, max_lines: int) -> str:
        return text_tail(self.output, max_lines)

    def __str__(self) -> str:
        message = (
            f"Command failed with exit code {self.returncode}: "
            f"{shlex.join(os.fspath(arg) for arg in self.cmd)}"
        )
        if self.cwd is not None:
            message += f"\nWorking directory: {self.cwd}"
        return message


def terminate(error: Exception, *, exit_code: int = 1) -> NoReturn:
    reason = str(error)
    reason = reason if reason else error.__class__.__name__
    console.print(f"[red]Error[/] {escape(reason)}")
    raise SystemExit(exit_code) from None


def log_detail(message: str) -> None:
    text = message.rstrip()
    if text:
        console.print(escape(text), style="dim")


def log_info(message: str) -> None:
    text = message.rstrip()
    if text:
        log_detail(f"INFO {text}")


def log_tail(title: str, text: str, *, max_lines: int = 100) -> None:
    tail = text_tail(text, max_lines)
    if tail:
        log_detail(f"{title} (last {max_lines} lines):\n{tail}")


def log_debug(message: str) -> None:
    if os.getenv("ALLO_DEBUG") is not None:
        text = message.rstrip()
        if text:
            log_detail(f"DEBUG {text}")


def log_warning(message: str) -> None:
    text = message.rstrip()
    if text:
        console.print(f"[yellow]Warning[/] {escape(text)}")


def log_fatal(message: str) -> None:
    text = message.strip()
    console.print(f"[red]Fatal[/] {escape(text)}")
    raise SystemExit(1) from None


def log_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    table = Table(title=title, title_style="cyan", show_lines=False)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(Text(str(value)) for value in row))
    console.print(table)


def read_text_tail(path: str | os.PathLike[str], *, max_lines: int = 100) -> str:
    try:
        return text_tail(
            Path(path).read_text(encoding="utf-8", errors="replace"),
            max_lines,
        )
    except OSError:
        return ""


def completed_output(result: subprocess.CompletedProcess[str]) -> str:
    return captured_output(result.stdout or "", result.stderr or "")


def terminate_on_error(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            terminate(error)

    return cast(F, wrapper)


@contextmanager
def stage(
    name: str,
    *,
    on_error: ErrorCallback | None = None,
    on_exit: ExitCallback | None = None,
) -> Iterator[None]:
    try:
        with console.status(f"[cyan]{name}[/]", spinner="dots"):
            yield
    except Exception as error:
        console.print(f"[red]Fail[/] {name}")
        if on_error is not None:
            on_error(error)
        if on_exit is not None:
            on_exit()
        terminate(error)
    else:
        console.print(f"[green]Success[/] {name}")
        if on_exit is not None:
            on_exit()


def run_command(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stage_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    def invoke() -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [os.fspath(arg) for arg in cmd],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CommandError(
                cmd=cmd,
                cwd=cwd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    if stage_name is None:
        return invoke()

    with stage(stage_name):
        return invoke()
