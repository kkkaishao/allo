# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator cores a fabric may offer, as ARCHETYPES."""

from __future__ import annotations

from ....lang import f32, f64, bf16, i32, bool as _bool
from ....lang.ip import operator_ip, OperatorType

# An `@operator_ip` body is `...`: the parameters exist to declare the
# signature. The declared latency is a placeholder each fabric's table replaces.
# pylint: disable=unused-argument

_ARCHETYPE = {"latency": 1, "in_delay_ns": 0.5, "pipelined": True, "style": "ce"}


@operator_ip(optype=OperatorType.ADD, **_ARCHETYPE)
def fadd(a: f32, b: f32) -> f32: ...


@operator_ip(optype=OperatorType.SUB, **_ARCHETYPE)
def fsub(a: f32, b: f32) -> f32: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def fmul(a: f32, b: f32) -> f32: ...


@operator_ip(optype=OperatorType.DIV, **_ARCHETYPE)
def fdiv(a: f32, b: f32) -> f32: ...


@operator_ip(optype=OperatorType.CMP, **_ARCHETYPE)
def fcmp(a: f32, b: f32) -> _bool: ...


@operator_ip(optype=OperatorType.ADD, **_ARCHETYPE)
def dadd(a: f64, b: f64) -> f64: ...


@operator_ip(optype=OperatorType.SUB, **_ARCHETYPE)
def dsub(a: f64, b: f64) -> f64: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def dmul(a: f64, b: f64) -> f64: ...


@operator_ip(optype=OperatorType.DIV, **_ARCHETYPE)
def ddiv(a: f64, b: f64) -> f64: ...


@operator_ip(optype=OperatorType.CMP, **_ARCHETYPE)
def dcmp(a: f64, b: f64) -> _bool: ...


@operator_ip(optype=OperatorType.ADD, **_ARCHETYPE)
def bfadd(a: bf16, b: bf16) -> bf16: ...


@operator_ip(optype=OperatorType.SUB, **_ARCHETYPE)
def bfsub(a: bf16, b: bf16) -> bf16: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def bfmul(a: bf16, b: bf16) -> bf16: ...


# int <-> float conversion and float resize: one archetype per exact width pair,
# since a core's signature fixes its widths.
@operator_ip(optype=OperatorType.INT_FLOAT_CAST, **_ARCHETYPE)
def i2f(a: i32) -> f32: ...


@operator_ip(optype=OperatorType.INT_FLOAT_CAST, **_ARCHETYPE)
def f2i(a: f32) -> i32: ...


@operator_ip(optype=OperatorType.FLOAT_CAST, **_ARCHETYPE)
def fcvt(a: f32) -> f64: ...


@operator_ip(optype=OperatorType.FLOAT_CAST, **_ARCHETYPE)
def bf2f(a: bf16) -> f32: ...


# pylint: enable=unused-argument

#: Every archetype, so a fabric's table can be checked against the catalog
#: rather than against a reader's memory.
CATALOG = (
    fadd,
    fsub,
    fmul,
    fdiv,
    fcmp,
    dadd,
    dsub,
    dmul,
    ddiv,
    dcmp,
    bfadd,
    bfsub,
    bfmul,
    i2f,
    f2i,
    fcvt,
    bf2f,
)
