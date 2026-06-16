# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar, ParamSpec, Generic

from .kernel import Kernel
from ..schedule import Schedule

P = ParamSpec("P")
R = TypeVar("R")


class Module(Generic[P, R]):
    """Base class for library modules."""

    def __init__(self, name: str, module: Kernel[P, R], schedule: Schedule):
        self.name = name
        self.module = module
        self.schedule = schedule

        self.__doc__ = module.__doc__
        self.__signature__ = module.signature
        self.__wrapped__ = module.fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.module(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Module<{self.name}>"
