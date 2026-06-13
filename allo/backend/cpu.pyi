# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the CPU backend."""
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

class CPU(Generic[P, R]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
