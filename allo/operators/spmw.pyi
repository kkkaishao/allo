# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for single-program-multiple-worker (systolic) primitives."""
from typing import Any

__all__ = ["get_wid", "get_nw", "get", "put"]

# Worker id / worker count along a mapping axis; returned as ``int`` so that
# comparisons (``i == 0``) and stream indexing (``fifo[i, j]``) type-check
# inside PE bodies.
def get_wid(axis: int) -> int: ...
def get_nw(axis: int) -> int: ...

# ``get``/``put`` are normally used as methods on stream values; the module-level
# operator handles are exposed for completeness.
get: Any
put: Any
