# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alveo data-centre cards."""

from __future__ import annotations

from . import ultrascalex
from .spec import Part

_2L = ultrascalex.GRADE_2L

#: LUT/FF/DSP are the Virtex UltraScale+ product table's for XCVU47P, the die
#: xcu55c is built on; BRAM/URAM tile counts are DS978's. `bram36` and `uram288`
#: are TILE counters, named for their size; the storage realizations an array
#: binds to are `bram` and `uram`, distinct names in the same symbol table.
u55c = ultrascalex.build(
    Part(
        name="u55c",
        part="xcu55c-fsvh2892-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_303_680,
            "ff": 2_607_360,
            "dsp": 9_024,
            "bram36": 2_016,
            "uram288": 960,
        },
    )
)

# PLACEHOLDER capacities: scaffolding for the layering, not read off a data
# sheet yet. The area and timing tables above them are xcu55c's and are shared
# legitimately, since the fabric is the same silicon; only these five numbers
# per card are still owed.
u280 = ultrascalex.build(
    Part(
        name="u280",
        part="xcu280-fsvh2892-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_303_680,
            "ff": 2_607_360,
            "dsp": 9_024,
            "bram36": 2_016,
            "uram288": 960,
        },
    )
)

u250 = ultrascalex.build(
    Part(
        name="u250",
        part="xcu250-figd2104-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_728_000,
            "ff": 3_456_000,
            "dsp": 12_288,
            "bram36": 2_688,
            "uram288": 1_280,
        },
    )
)

DEVICES = (u55c, u280, u250)
