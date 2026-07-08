# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RTL backend: SDC scheduling frontend.

Exposes the hardware-facing operator timing library and the ``schedule`` driver
for the ``allo-schedule`` pass.
"""

from .operator_library import OperatorLibrary, OP_KINDS, OP_DTYPES
from .schedule import schedule

__all__ = ["OperatorLibrary", "schedule", "OP_KINDS", "OP_DTYPES"]
