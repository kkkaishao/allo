# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang.core import i32, f32, u1, Stateful, range as arange
from allo.lang.kernel import kernel


def test_stateless_scalar():
    """A plain local scalar re-initializes on every call."""

    @kernel
    def acc_stateless(x: i32) -> i32:
        acc: i32 = 0
        acc = acc + x
        return acc

    assert int(acc_stateless(5)) == 5
    assert int(acc_stateless(10)) == 10


def test_stateful_scalar():
    """A stateful scalar accumulates across calls."""

    @kernel
    def acc_stateful(x: i32) -> i32:
        acc: Stateful[i32] = 0
        acc = acc + x
        return acc

    assert int(acc_stateful(5)) == 5
    assert int(acc_stateful(10)) == 15
    assert int(acc_stateful(3)) == 18


def test_simple_counter():
    """A no-argument stateful counter increments on each call."""

    @kernel
    def counter() -> i32:
        count: Stateful[i32] = 0
        count = count + 1
        return count

    assert [int(counter()) for _ in range(5)] == [1, 2, 3, 4, 5]


def test_stateful_array():
    """A stateful array persists element writes across calls."""

    @kernel
    def array_stateful(x: f32) -> f32:
        buffer: Stateful[f32[10]] = 0.0
        buffer[0] = buffer[0] + x
        return buffer[0]

    assert np.isclose(float(array_stateful(5.0)), 5.0)
    assert np.isclose(float(array_stateful(10.0)), 15.0)
    assert np.isclose(float(array_stateful(3.0)), 18.0)


def test_stateful_in_loop():
    """A stateful scalar mutated inside a loop must not become an SSA iter-arg."""

    @kernel
    def acc_loop(x: i32) -> i32:
        acc: Stateful[i32] = 0
        for i in arange(4, name="i"):
            acc = acc + x
        return acc

    assert int(acc_loop(1)) == 4  # 0 + 1*4
    assert int(acc_loop(2)) == 12  # 4 + 2*4


def test_stateful_reset():
    """A stateful scalar written in both branches of an if/else."""

    @kernel
    def acc_reset(x: i32, rst: u1) -> i32:
        s: Stateful[i32] = 0
        if rst:
            s = 0
        else:
            s = s + x
        return s

    assert int(acc_reset(5, 0)) == 5
    assert int(acc_reset(10, 0)) == 15
    assert int(acc_reset(100, 1)) == 0  # reset
    assert int(acc_reset(7, 0)) == 7


def test_stateful_shared_across_instances():
    """A nested kernel with state, invoked from several call sites, shares one
    global (old-frontend semantics): both invocations accumulate into the same
    storage."""

    @kernel
    def top(x: i32) -> i32:
        @kernel
        def inc(d: i32) -> i32:
            c: Stateful[i32] = 0
            c = c + d
            return c

        a: i32 = inc(x)  # c: 0 -> x
        b: i32 = inc(x)  # c: x -> 2x  (shared global)
        return a * 1000 + b

    # call 1: a=5, b=10 -> 5010 ; call 2: a=15, b=20 -> 15020
    assert int(top(5)) == 5010
    assert int(top(5)) == 15020


def test_stateful_distinct_decls():
    """Two distinct stateful declarations get independent globals."""

    @kernel
    def two(x: i32) -> i32:
        p: Stateful[i32] = 0
        q: Stateful[i32] = 100
        p = p + x
        q = q + x
        return p * 1000 + q

    assert int(two(5)) == 5105  # p=5, q=105
    assert int(two(10)) == 15115  # p=15, q=115
