from __future__ import annotations

import builtins
from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union, Sequence

if TYPE_CHECKING:
    from .core import types

    IterableType = Union[list[Any], tuple[Any, ...], types.tuple, types.tuple_type]
    ObjPath = tuple[int, ...]


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    from .core import types

    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    assert isinstance(prev, types.tuple)
    prev[path[-1]] = val


def is_iterable(x):
    from .core import types

    return isinstance(x, (list, builtins.tuple, types.tuple, types.tuple_type))


def apply_with_path(value: Any, fn: Callable[[ObjPath, Any], None], _path=None) -> None:
    if _path is None:
        _path = ()

    if is_iterable(value):
        for idx, item in enumerate(value):
            apply_with_path(item, fn, _path=(*_path, idx))
    else:
        fn(_path, value)


def find_paths_if(
    iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]
) -> list[ObjPath]:
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def validate_block_shape(shape: Sequence):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]"
            )
        numel *= d
    return numel


def is_namedtuple(val):
    return (
        isinstance(val, type)
        and issubclass(val, builtins.tuple)
        and hasattr(val, "_fields")
    )


def _tuple_create(arg, contents):
    # NamedTuples and tuples have different construction semantics. NamedTuple
    # has a constructor that takes individual arguments, while tuple takes an
    # iterable. Both have type "tuple" making it difficult to distinguish
    # between them, but only NamedTuple has "_fields" and apparently this is how
    # everyone does the check.
    return type(arg)(*contents) if hasattr(arg, "_fields") else type(arg)(contents)
