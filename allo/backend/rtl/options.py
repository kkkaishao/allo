# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compile was ASKED for, split by which half of it reads the knob."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrepassOptions:
    """The IR rewrites run BEFORE the scheduler, which decide what problem it is
    handed rather than how it solves one.

    Never published. What each of them did is already visible where it landed:
    a directive they cost the scheduler is an ``unhonored_directives`` entry, an
    array they kept in registers is absent from the memory report, and a
    reduction they rebalanced is in the region's own ops.

    Args:
        float_reassoc: rebalance float reduction chains into logarithmic trees.
            Not bit-exact.
        accumulators: rotate float reductions across this many accumulators,
            dropping their II to ``ceil(latency / accumulators)`` (0 = off).
        unroll_under_pipeline: fully unroll the loops nested inside a pipelined
            loop, so the nest pipelines at one II (Vitis ``#pragma HLS pipeline``
            semantics). ``False`` keeps them rolled and the directive is then not
            honored.
        perfectize: sink an imperfect nest's prologue/epilogue into the inner
            loop under a guard, fusing it into one pipeline. A QoR alternative;
            the scheduler handles imperfect nests without it.
        scalarize_threshold: keep arrays of at most this many elements in
            registers rather than a memory (0 = off).
    """

    float_reassoc: bool = True
    accumulators: int = 0
    unroll_under_pipeline: bool = True
    perfectize: bool = False
    scalarize_threshold: int = 16


@dataclass(frozen=True)
class SchedulerOptions:
    """What the SCHEDULER itself was asked for.

    Every field is the EFFECTIVE value the solve ran under, not "unset": this is
    what a second result has to match to be comparable with the first, and the
    IR the scheduler leaves behind carries its decisions, not the knobs it was
    turned with. It is also the knob list, which ``RTL.set_scheduler_opt`` turns
    by field name.

    Args:
        scheduler: the solver that settles the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement; ``"exact"``
            is CP-SAT over the same problem, keeping the chain-breaking pre-pass;
            ``"exact-chaining"`` additionally decides where to break a too-long
            combinational chain in the solver. Both exact modes need OR-Tools.
        cycle_ns: the period every combinational chain was cut to. Derived from
            the handle's ``freq_mhz``, which the cosim clock also reads.
        budget: what ONE exact solve may spend, in the solver's deterministic
            time units (roughly a second of one core each). Raising it buys a
            better placement on the few regions large enough to exhaust it and
            costs nothing on the rest.
        workers: how many search workers ONE exact solve runs. The default of
            one is what makes the deterministic budget bite reproducibly: a
            portfolio races, so which incumbent the budget stops on depends on
            thread timing and the RTL of two identical compiles can differ.
            Raising it reaches a good schedule sooner; take it for exploring,
            not for a result that has to reproduce.
        seed: the exact solver's random seed. Only shifts which optimum of equal
            cost a solve lands on, so varying it samples that tie set.
    """

    scheduler: str = "heuristic"
    cycle_ns: float = 5.0
    budget: float = 30.0
    workers: int = 1
    seed: int = 0
