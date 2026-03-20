# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Generic, ParamSpec, TypeVar, overload, Optional

P = ParamSpec("P")
R = TypeVar("R")

NO_FOLD = object()


class Operator(Generic[P, R]):
    """
    Allo frontend operation definition.

    Each operation can define:
    - `validate(*args, **kwargs) -> str`: return empty string if valid.
    - `const_fold(*args, **kwargs) -> Any`: return folded value or `NO_FOLD`.
    - `lower(builder, *args, **kwargs) -> Any`: emit IR and return proxy value.
    """

    def __init__(self, fn: Callable[P, R]):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self._validate_impl: Callable[..., str] = lambda *args, **kwargs: ""
        self._const_fold_impl: Callable[..., Any] = lambda *args, **kwargs: NO_FOLD
        self._lower_impl: Callable[..., Any] | None = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise RuntimeError(
            "Cannot execute operation outside of allo compilation context"
        )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundOperator(self, instance)

    def validate(self, fn: Callable[..., str]) -> Callable[..., str]:
        self._validate_impl = fn
        return fn

    def const_fold(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self._const_fold_impl = fn
        return fn

    def lower(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self._lower_impl = fn
        return fn

    def run_validate(self, *args, **kwargs) -> str:
        return self._validate_impl(*args, **kwargs)

    def run_const_fold(self, *args, **kwargs) -> Any:
        return self._const_fold_impl(*args, **kwargs)

    def run_lower(self, builder, *args, **kwargs) -> Any:
        if self._lower_impl is None:
            raise RuntimeError(f"Operation '{self.__name__}' does not define lowering")
        return self._lower_impl(builder, *args, **kwargs)


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
def operator() -> Callable[[Callable[P, R]], Operator[P, R]]: ...


def operator(
    fn: Optional[Callable[P, R]] = None,
) -> Operator[P, R] | Callable[[Callable[P, R]], Operator[P, R]]:
    def wrap(inner_fn) -> Operator[P, R]:
        return Operator(inner_fn)

    if fn is not None:
        return wrap(fn)
    return wrap


def resolve_operator(value) -> Operator | BoundOperator | None:
    if isinstance(value, (Operator, BoundOperator)):
        return value
    return None


def as_member_function(fn: Operator[P, R]) -> Operator[P, R]:
    from .types import Proxy

    setattr(Proxy, fn.__name__, fn)
    return fn
