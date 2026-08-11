# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compile was asked for, split by which stage reads the knob."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrepassOptions:
    """The IR rewrites run before the scheduler. They decide what problem it is
    handed rather than how it solves one, and are not published in any report.

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
            loop under a guard, fusing it into one pipeline. Optional; the
            scheduler handles imperfect nests without it.
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
    """What the scheduler itself was asked for.

    Every field is the effective value the solve ran under, and the set of them
    is the knob list ``RTL.set_scheduler_opt`` turns by field name.

    Args:
        scheduler: the solver that settles the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement; ``"exact"``
            is CP-SAT over the same problem, keeping the chain-breaking pre-pass;
            ``"exact-chaining"`` additionally decides where to break a too-long
            combinational chain in the solver. Both exact modes need OR-Tools.
        cycle_ns: the period every combinational chain was cut to. Derived from
            the handle's ``freq_mhz``, which the cosim clock also reads.
        budget: what one exact solve may spend, in the solver's deterministic
            time units (roughly a second of one core each).
        workers: how many search workers one exact solve runs. Above one the
            portfolio is interleaved, so the deterministic budget still bounds a
            deterministic search. Not only a speed knob: the same budget buys
            more search, so a budget-limited region can settle on a different
            schedule than it does at one worker.
        seed: the exact solver's random seed. Shifts which optimum of equal cost
            a solve lands on.
    """

    scheduler: str = "heuristic"
    cycle_ns: float = 5.0
    budget: float = 30.0
    workers: int = 8
    seed: int = 0
