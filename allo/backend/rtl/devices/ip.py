# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The archetypes of the operator cores a fabric may offer."""

from __future__ import annotations

from ....lang import (
    f32,
    f64,
    bf16,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    bool as _bool,
)
from ....lang.ip import operator_ip, OperatorType

# An `@operator_ip` body is `...`: the parameters only declare the signature.
# The declared latency is a placeholder each fabric's table replaces.
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


# Int/float conversion and float resize: one archetype per exact width pair,
# since a core's signature fixes its widths.
@operator_ip(optype=OperatorType.INT_FLOAT_CAST, **_ARCHETYPE)
def i2f(a: i32) -> f32: ...


@operator_ip(optype=OperatorType.INT_FLOAT_CAST, **_ARCHETYPE)
def f2i(a: f32) -> i32: ...


@operator_ip(optype=OperatorType.FLOAT_CAST, **_ARCHETYPE)
def fcvt(a: f32) -> f64: ...


@operator_ip(optype=OperatorType.FLOAT_CAST, **_ARCHETYPE)
def bf2f(a: bf16) -> f32: ...


# Integer multiply. `arith.muli` on `iN` keeps the low N bits, which are the
# same for a signed and an unsigned product, so one core serves both and the
# abstract kind selects it.
@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def imul8(a: i8, b: i8) -> i8: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def imul16(a: i16, b: i16) -> i16: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def imul32(a: i32, b: i32) -> i32: ...


@operator_ip(optype=OperatorType.MUL, **_ARCHETYPE)
def imul64(a: i64, b: i64) -> i64: ...


# A fused multiply-add: `(a * b + c) mod 2^N`, the product's low bits plus the
# addend, so the multiply-to-add hop never leaves the core. Signedness-
# transparent like the plain multiply. Matched by `allo.muladd`'s mnemonic.
@operator_ip(optype="muladd", **_ARCHETYPE)
def imuladd32(a: i32, b: i32, c: i32) -> i32: ...


# A 64-bit product whose operands PROVE narrow: a genuine 33x33 core, its
# 66-bit product sliced to the low 64, exact modulo 2^64 for operands of at
# most 33 significant bits, which is what `fed_width` restricts the row to.
# A widening 32x32 product and a reciprocal's magic multiply are the feeders;
# a delivered 64x64 core netlist prunes nothing, so the narrow core is real
# hardware rather than an optimization hope.
@operator_ip(optype=OperatorType.MUL, mnemonic="mulw", fed_width=33, **_ARCHETYPE)
def imulw33(a: i64, b: i64) -> i64: ...


# Integer divide and remainder. `divsi` and `divui` compute different numbers
# and Xilinx builds them from different cores, but MLIR's `i32` is signless and
# both classify as `div`, so each binds by its own MLIR mnemonic instead of by
# the abstract kind. Their areas differ by under 8% in LUTs and under 1% in
# FFs, so a fabric prices the pair from one measurement.
@operator_ip(optype="divsi", **_ARCHETYPE)
def idiv8(a: i8, b: i8) -> i8: ...


@operator_ip(optype="divsi", **_ARCHETYPE)
def idiv16(a: i16, b: i16) -> i16: ...


@operator_ip(optype="divsi", **_ARCHETYPE)
def idiv32(a: i32, b: i32) -> i32: ...


@operator_ip(optype="divsi", **_ARCHETYPE)
def idiv64(a: i64, b: i64) -> i64: ...


@operator_ip(optype="divui", **_ARCHETYPE)
def udiv8(a: u8, b: u8) -> u8: ...


@operator_ip(optype="divui", **_ARCHETYPE)
def udiv16(a: u16, b: u16) -> u16: ...


@operator_ip(optype="divui", **_ARCHETYPE)
def udiv32(a: u32, b: u32) -> u32: ...


@operator_ip(optype="divui", **_ARCHETYPE)
def udiv64(a: u64, b: u64) -> u64: ...


@operator_ip(optype="remsi", **_ARCHETYPE)
def irem8(a: i8, b: i8) -> i8: ...


@operator_ip(optype="remsi", **_ARCHETYPE)
def irem16(a: i16, b: i16) -> i16: ...


@operator_ip(optype="remsi", **_ARCHETYPE)
def irem32(a: i32, b: i32) -> i32: ...


@operator_ip(optype="remsi", **_ARCHETYPE)
def irem64(a: i64, b: i64) -> i64: ...


@operator_ip(optype="remui", **_ARCHETYPE)
def urem8(a: u8, b: u8) -> u8: ...


@operator_ip(optype="remui", **_ARCHETYPE)
def urem16(a: u16, b: u16) -> u16: ...


@operator_ip(optype="remui", **_ARCHETYPE)
def urem32(a: u32, b: u32) -> u32: ...


@operator_ip(optype="remui", **_ARCHETYPE)
def urem64(a: u64, b: u64) -> u64: ...


# pylint: enable=unused-argument

#: Every archetype, for checking a fabric's table against.
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
    imul8,
    imul16,
    imul32,
    imul64,
    imulw33,
    imuladd32,
    idiv8,
    idiv16,
    idiv32,
    idiv64,
    udiv8,
    udiv16,
    udiv32,
    udiv64,
    irem8,
    irem16,
    irem32,
    irem64,
    urem8,
    urem16,
    urem32,
    urem64,
)
