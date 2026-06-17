# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
import allo
from allo import kernel
from allo.lang import range
from allo.lang.core import DType
from allo.operators import math as m
from allo.lang import Module

INV_SQRT2 = 0.7071067811865476
GELU_C = 0.7978845608028654  # sqrt(2/pi)
_UNARY = ("silu", "gelu", "gelu_tanh", "relu")
_BINARY = ("add", "mul", "swiglu")


def _sched(top, ii):
    s = top.schedule()
    s.unroll("l")
    s.pipeline(s.flatten(("r", "ct")), ii=ii)
    return top, s


def _make(
    Tin,
    Tout,
    M,
    N,
    *,
    variant: Literal[
        "silu", "gelu", "gelu_tanh", "relu", "add", "mul", "swiglu"
    ] = "silu",
    L=16,
    ii=1,
):
    if variant in _UNARY:
        return _sched(_make_unary(Tin, Tout, M, N, variant, L), ii)
    if variant in _BINARY:
        return _sched(_make_binary(Tin, Tout, M, N, variant, L), ii)
    raise ValueError(f"unknown variant {variant!r}; unary={_UNARY} binary={_BINARY}")


def _make_unary(Tin, Tout, M, N, kind, L):
    CT = N // L
    if kind == "silu":

        @kernel
        def top(x: Tin[M, N], y: Tout[M, N]):
            """SiLU ``y = x * sigmoid(x)``, ``L`` lanes/cycle, one II=1 pass."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        v: Tin = x[r, ct * L + l]
                        y[r, ct * L + l] = v * (1.0 / (1.0 + m.exp(-v)))

    elif kind == "gelu":  # exact erf

        @kernel
        def top(x: Tin[M, N], y: Tout[M, N]):
            """Exact GELU ``y = 0.5 x (1 + erf(x/sqrt2))`` (erf is DSP-heavy; use
            the ``gelu_tanh`` variant for ~7x fewer DSP)."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        v: Tin = x[r, ct * L + l]
                        y[r, ct * L + l] = 0.5 * v * (1.0 + m.erf(v * INV_SQRT2))

    elif kind == "gelu_tanh":  # tanh approximation (cheaper than exact erf)

        @kernel
        def top(x: Tin[M, N], y: Tout[M, N]):
            """Tanh-approx GELU ``y = 0.5 x (1 + tanh(c(x+0.044715 x^3)))`` -- the
            cheap GELU (~7x fewer DSP than the exact ``gelu`` variant)."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        v: Tin = x[r, ct * L + l]
                        u: Tin = GELU_C * (v + 0.044715 * v * v * v)
                        y[r, ct * L + l] = 0.5 * v * (1.0 + m.tanh(u))

    else:  # relu

        @kernel
        def top(x: Tin[M, N], y: Tout[M, N]):
            """ReLU ``y = max(x, 0)`` (no DSP)."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        v: Tin = x[r, ct * L + l]
                        y[r, ct * L + l] = allo.max(v, 0.0)

    return top


def _make_binary(Tin, Tout, M, N, kind, L):
    CT = N // L
    if kind == "add":

        @kernel
        def top(a: Tin[M, N], b: Tin[M, N], y: Tout[M, N]):
            """Residual add ``y = a + b``, ``L`` lanes/cycle, one II=1 pass."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        y[r, ct * L + l] = a[r, ct * L + l] + b[r, ct * L + l]

    elif kind == "mul":

        @kernel
        def top(a: Tin[M, N], b: Tin[M, N], y: Tout[M, N]):
            """Elementwise multiply ``y = a * b``, ``L`` lanes/cycle, II=1."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        y[r, ct * L + l] = a[r, ct * L + l] * b[r, ct * L + l]

    else:  # swiglu: silu(a) * b

        @kernel
        def top(a: Tin[M, N], b: Tin[M, N], y: Tout[M, N]):
            """SwiGLU gate*up fuse ``y = silu(a) * b`` in one pass (the gate and
            up projections feed ``a`` and ``b``)."""
            for r in range(M, name="r"):
                for ct in range(CT, name="ct"):
                    for l in range(L, name="l"):  # unrolled
                        av: Tin = a[r, ct * L + l]
                        sa: Tin = av * (1.0 / (1.0 + m.exp(-av)))
                        y[r, ct * L + l] = sa * b[r, ct * L + l]

    return top


class Activation(Module):
    """**Elementwise activation / binary op for the FFN / residual path**

    Signature: ``activation(Tin[M, N], Tout[M, N])`` for unary ops (e.g. SiLU,
    GELU); ``activation(Tin[M, N], Tin[M, N], Tout[M, N])`` for binary ops (e.g.
    add, mul, swiglu).

    Parameters
    ----------
    variant : {"silu", "gelu", "gelu_tanh", "relu", "add", "mul", "swiglu"}, default "silu"
        Selects the op (and the kernel's input arity):

        * unary (one input ``x``): ``silu`` (LLaMA gate), ``gelu`` (exact erf,
          DSP-heavy), ``gelu_tanh`` (~7x fewer DSP than exact), ``relu``.
        * binary (two inputs ``a, b``): ``add`` (residual), ``mul``, ``swiglu``
          (``silu(a)*b``, the gate*up fuse).
    """

    def __init__(
        self,
        Tin,
        Tout,
        M,
        N,
        *,
        variant: Literal[
            "silu", "gelu", "gelu_tanh", "relu", "add", "mul", "swiglu"
        ] = "silu",
        L=16,
        ii=1,
    ):
        if not isinstance(Tin, DType) or not isinstance(Tout, DType):
            raise ValueError("Activation data types must be DType instances")
        if N % L != 0:
            raise ValueError("Lanes L must tile N")
        if variant not in _UNARY + _BINARY:
            raise ValueError(
                f"unknown variant {variant!r}; unary={_UNARY} binary={_BINARY}"
            )
        top, s = _make(Tin, Tout, M, N, variant=variant, L=L, ii=ii)
        name = f"{variant.capitalize()}_M{M}_N{N}_L{L}"
        super().__init__(name, top, s)


__all__ = ["Activation"]
