# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for memory / bit-slice operators."""
from typing import Any

__all__ = ["load", "store"]

def load(lhs: Any, slices: Any) -> Any: ...
def store(dst: Any, slices: Any, value: Any) -> Any: ...
