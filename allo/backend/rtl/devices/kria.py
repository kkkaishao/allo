# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kria system-on-modules.

Zynq UltraScale+ silicon, so the fabric is `ultrascalex` and the area tables are
shared with the Alveo cards legitimately: what differs is the die's capacity and
the SPEED GRADE, and a grade is exactly what `TIMING` is keyed on. The -2LV
tables were measured on this part; on the same operator at 32 bits it is about
half again slower than the -2L the Alveo cards are binned at.
"""

from __future__ import annotations

from . import ultrascalex
from .spec import Part

#: Capacities read off the part itself (`get_property LUT_ELEMENTS` and
#: friends on `xck26-sfvc784-2LV-c`), not off a data sheet.
kv260 = ultrascalex.build(
    Part(
        name="kv260",
        part="xck26-sfvc784-2LV-c",
        grade=ultrascalex.GRADE_2LV,
        capacity={
            "lut": 117_120,
            "ff": 234_240,
            "dsp": 1_248,
            "bram36": 144,
            "uram288": 64,
        },
    )
)

DEVICES = (kv260,)
