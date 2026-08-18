# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store->load forwarding on a RAM read-modify-write recurrence.

A store commits one cycle after it issues, so a same-array load one iteration
behind it would wait out the full storage round trip and the II with it. The
scheduler relaxes that RAW edge through a shadow register pair; the emitter
compares the two addresses in the shared issue cycle and muxes the store's
datum into the load's data path. The inputs here contain adjacent duplicate
indices on purpose: an input that never collides would pass without the shadow
ever selecting.
"""

import os
import sys

import numpy as np

from allo import kernel
from allo.lang import i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _iis  # noqa: E402

N, BINS = 32, 8


def _bumpy(rng):
    # Runs of equal values, so consecutive iterations hit one bin: exactly the
    # collision the shadow serves.
    x = np.repeat(rng.integers(0, BINS, N // 2), 2).astype(np.int32)[:N]
    return x


def _hist():
    @kernel
    def hist(x: i32[N], h: i32[BINS]):
        for i in range(N):
            v: i32 = x[i]
            h[v] = h[v] + 1

    return hist


def test_a_ram_rmw_loop_forwards_the_uncommitted_store():
    # At a 4 ns period the load->add->store chain fits one cycle, so the store
    # issues in the very cycle the next iteration's load does and the relaxed
    # recurrence reaches II=1. Without forwarding the round trip pins II at 3.
    s = _hist().schedule()
    mod = s.export("rtl", freq_mhz=250)
    assert _iis(mod.schedule().func("hist").regions) == [1]

    rng = np.random.default_rng(0)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))


def test_forwarding_survives_the_chain_break_at_the_default_clock():
    # At the default clock the same chain is split, the store lands two cycles
    # after the load, and the collision is one whole interval away: the shadow
    # then pairs iteration k's store with iteration k+1's load. Still one cycle
    # better than the unforwarded round trip.
    s = _hist().schedule()
    mod = s.export("rtl")
    assert _iis(mod.schedule().func("hist").regions) == [2]

    rng = np.random.default_rng(1)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))


def test_an_unrolled_rmw_body_forwards_from_every_paired_store():
    # Unrolled by two, each load pairs with both stores (the same-iteration one
    # at distance 0, the carried ones at distance 1) and its data out muxes
    # over several shadow arms. At most one arm can match in a cycle, and the
    # duplicates make every arm fire somewhere in the run.
    s = _hist().schedule()
    s.unroll(s.loop("i"), factor=2)
    mod = s.export("rtl", freq_mhz=250)

    rng = np.random.default_rng(2)
    x = _bumpy(rng)
    h = np.zeros(BINS, np.int32)
    mod.cosim(x, h)
    assert np.array_equal(h, np.bincount(x, minlength=BINS).astype(np.int32))
