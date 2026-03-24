# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..core.library import operator
from .ops_arith import (
    add,
    sub,
    mul,
    div,
    floordiv,
    mod,
    pow,
    lshift,
    rshift,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    pos,
    neg,
    invert,
    logical_and,
    logical_not,
    logical_or,
    max,
    min,
    cast,
)

from .ops_memory import load, store
from .ops_spmd import get_wid, get_num_workers, get, put


# placeholder for static_assert
@operator
def static_assert(condition: bool, msg: str = ""):
    pass
