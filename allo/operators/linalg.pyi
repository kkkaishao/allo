# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for linear-algebra operators."""
from typing import Any

__all__ = ["matmul", "dot"]

def matmul(lhs: Any, rhs: Any, acc: Any = ...) -> Any: ...
def dot(lhs: Any, rhs: Any, acc: Any = ...) -> Any: ...
