# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Generic, TypeVar, ParamSpec, Callable, Any, overload

P = ParamSpec("P")
R = TypeVar("R")

NO_FOLD = object()


class Operator(Generic[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        *,
        cls: type | tuple[type, ...] | None = None,
    ):
        self.fn = fn
        self.n_args = len(inspect.signature(fn).parameters)
        self.fold_impl = None
        self.build_impl = None
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        if cls is not None:
            classes = cls if isinstance(cls, tuple) else (cls,)
            for owner in classes:
                existing = getattr(owner, fn.__name__, None)
                assert existing is None or existing is self
                setattr(owner, fn.__name__, self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise RuntimeError(
            "Cannot execute operation outside of allo compilation context"
        )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundOperator(self, instance)

    def fold(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        assert (
            len(inspect.signature(fn).parameters) == self.n_args
        ), "Fold function for operator must have the same number of parameters as the operator itself"
        self.fold_impl = fn
        return fn

    def build(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        assert (
            len(inspect.signature(fn).parameters) == self.n_args + 1
        ), "Build function for operator must have one more parameter than the operator itself (for the builder)"
        self.build_impl = fn
        return fn


class BoundOperator:
    def __init__(self, op: Operator, receiver: Any):
        self.op = op
        self.receiver = receiver
        self.__name__ = op.__name__
        self.__doc__ = op.__doc__

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot execute operation outside of allo compilation context"
        )

    def bind_args(self, args):
        return [self.receiver, *args]


@overload
def operator(fn: Callable[P, R]) -> Operator[P, R]: ...


@overload
def operator(
    *, cls: type | tuple[type, ...] | None = None
) -> Callable[[Callable[P, R]], Operator[P, R]]: ...


def operator(
    fn: Callable[P, R] | None = None,
    *,
    cls: type | tuple[type, ...] | None = None,
) -> Operator[P, R] | Callable[[Callable[P, R]], Operator[P, R]]:
    def decorator(fn: Callable[P, R]) -> Operator[P, R]:
        return Operator(fn, cls=cls)

    if fn is not None:
        return decorator(fn)
    return decorator
