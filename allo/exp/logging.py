from __future__ import annotations

import os
import shlex
import subprocess

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator, NoReturn, TypeVar, cast

from rich.console import Console
from rich.markup import escape

console = Console(stderr=True)
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CommandError(RuntimeError):
    cmd: Sequence[str | os.PathLike[str]]
    returncode: int
    cwd: str | os.PathLike[str] | None = None
    stdout: str = ""
    stderr: str = ""

    def __str__(self) -> str:
        message = (
            f"Command failed with exit code {self.returncode}: "
            f"{shlex.join(os.fspath(arg) for arg in self.cmd)}"
        )
        if self.cwd is not None:
            message += f"\nWorking directory: {self.cwd}"
        return message


def error_reason(error: Exception) -> str:
    reason = str(error)
    return reason if reason else error.__class__.__name__


def terminate(error: Exception, *, exit_code: int = 1) -> NoReturn:
    console.print(f"[red]Error[/] {escape(error_reason(error))}")
    raise SystemExit(exit_code) from None


def terminate_on_error(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            terminate(error)

    return cast(F, wrapper)


@contextmanager
def stage(name: str) -> Iterator[None]:
    try:
        with console.status(f"[cyan]{name}[/]", spinner="dots"):
            yield
    except Exception as error:
        console.print(f"[red]Fail[/] {name}")
        terminate(error)
    else:
        console.print(f"[green]Success[/] {name}")


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
