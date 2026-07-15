# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral (simulation-only) models for the extern IP operators the datapath
emitter instantiates -- a float/double core, the widened integer multiply/divide,
or a floating-point compare, none of which has a comb lowering.

The module name encodes op + latency and a leading kind letter ``f``/``d``/``i``
for single-float / double / integer: a binary op is ``<k><op>_l<lat>``
(``fadd_l7`` / ``dadd_l14`` / ``imul_l3``); a float compare is
``<k>cmp_l<lat>_<pred>`` (``fcmp_l1_ogt``), one module per predicate, taking two
operand-width inputs and yielding i1.

Each model is a ``latency``-deep shift of the result, gated by a ``ce``
clock-enable when the IP carries one: ``ce == 0`` freezes the pipe, so it stays
aligned with the shell's frozen shift chains under back-pressure. Float math goes
through DPI-C because Verilator's ``$shortrealtobits`` returns 0, leaving no float
bitcast; a C function reinterprets the raw bits at the port width (4 bytes for
``f``, 8 for ``d``). An integer IP computes natively at its (<=64-bit,
sign-extended) port width.
"""

from __future__ import annotations

import re

from collections import namedtuple

_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/", "rem": "%"}
# Float-compare predicate -> C comparison operator. Ordered (o*) and unordered
# (u*) map to the same operator: cosim inputs are NaN-free, so the only place they
# differ (a NaN operand) does not arise.
_CMP = {
    "oeq": "==",
    "one": "!=",
    "ogt": ">",
    "oge": ">=",
    "olt": "<",
    "ole": "<=",
    "ueq": "==",
    "une": "!=",
    "ugt": ">",
    "uge": ">=",
    "ult": "<",
    "ule": "<=",
}

# One extern IP operator. `dpi` is the DPI function suffix (arith op for a binary,
# predicate for a compare); `cexpr` its C operator; `is_cmp` picks the DPI body
# (a compare returns 0/1, a binary reinterprets the arithmetic result); `has_ce`
# is the clock-enable freeze bit (present when the IP's stall contract is
# clock-enabled, i.e. `(a, b, clk, ce) -> y`).
Extern = namedtuple("Extern", "name in_w out_w kind dpi cexpr latency is_cmp has_ce")


def externs(ir: str):
    """The extern IP operators the emitter declared, as :class:`Extern` records."""
    out = []
    for m in re.finditer(
        r"hw\.module\.extern @(\w+)\(in %a : i(\d+), in %b : i\d+, "
        r"in %clk : i1(, in %ce : i1)?, out \w+ : i(\d+)\)",
        ir,
    ):
        name, in_w = m.group(1), int(m.group(2))
        has_ce, out_w = m.group(3) is not None, int(m.group(4))
        binary = re.match(r"([fdi])(add|sub|mul|div|rem)_l(\d+)$", name)
        compare = re.match(r"([fd])cmp_l(\d+)_(\w+)$", name)
        if binary:
            k, op, lat = binary.group(1), binary.group(2), int(binary.group(3))
            out.append(Extern(name, in_w, out_w, k, op, _OPS[op], lat, False, has_ce))
        elif compare:
            k, lat, pred = compare.group(1), int(compare.group(2)), compare.group(3)
            assert pred in _CMP, f"unsupported float-compare predicate '{pred}'"
            out.append(
                Extern(name, in_w, out_w, k, pred, _CMP[pred], lat, True, has_ce)
            )
        else:
            assert False, f"no behavioral model for extern operator '{name}'"
    return out


def sv_models(ir: str) -> str:
    """SystemVerilog behavioral models + DPI import decls for the extern IP ops."""
    ext = externs(ir)
    if not ext:
        return ""
    used = sorted({(e.kind, e.dpi) for e in ext})
    ctype = lambda k: "int" if k == "f" else "longint"  # f=32-bit, d/i=64-bit
    imports = "".join(
        f'import "DPI-C" function {ctype(k)} {k}_{op}'
        f"(input {ctype(k)} a, input {ctype(k)} b);\n"
        for k, op in used
    )
    out = [imports]
    for e in ext:
        # A clock-enabled IP guards its shift on `ce`, so a low `ce` freezes the
        # whole pipe.
        ce_port = ", input ce" if e.has_ce else ""
        guard = "if (ce) " if e.has_ce else ""
        out.append(
            f"module {e.name}(input [{e.in_w - 1}:0] a, input [{e.in_w - 1}:0] b, "
            f"input clk{ce_port}, output [{e.out_w - 1}:0] y);\n"
            f"  reg [{e.out_w - 1}:0] p [0:{e.latency - 1}];\n  integer i;\n"
            f"  always @(posedge clk) {guard}begin\n"
            f"    p[0] <= {e.kind}_{e.dpi}(a, b);\n"
            f"    for (i = 1; i < {e.latency}; i = i + 1) p[i] <= p[i - 1];\n"
            f"  end\n"
            f"  assign y = p[{e.latency - 1}];\n"
            f"endmodule\n"
        )
    return "\n".join(out)


def dpi_c(ir: str) -> str:
    """C implementations of the DPI operators used: a float op reinterprets the
    raw bits, computes, and reinterprets back; a float compare returns 0/1; an
    integer op computes natively (64-bit, operands pre-extended)."""
    used = sorted({(e.kind, e.dpi, e.cexpr, e.is_cmp) for e in externs(ir)})
    if not used:
        return ""
    lines = []
    for k, op, cexpr, is_cmp in used:
        if k == "i":
            lines.append(
                f'extern "C" long long i_{op}(long long a, long long b) '
                f"{{ return a {cexpr} b; }}"
            )
            continue
        # f: 32-bit float; d: 64-bit double. Reinterpret the raw integer bits at
        # the port width.
        cty, fty, n = ("int", "float", 4) if k == "f" else ("long long", "double", 8)
        if is_cmp:
            lines.append(
                f'extern "C" {cty} {k}_{op}({cty} a, {cty} b) {{ {fty} x, y; '
                f"memcpy(&x,&a,{n}); memcpy(&y,&b,{n}); return x {cexpr} y; }}"
            )
        else:
            lines.append(
                f'extern "C" {cty} {k}_{op}({cty} a, {cty} b) {{ {fty} x, y, r; '
                f"memcpy(&x,&a,{n}); memcpy(&y,&b,{n}); r = x {cexpr} y; "
                f"{cty} o; memcpy(&o,&r,{n}); return o; }}"
            )
    return "#include <cstring>\n" + "\n".join(lines) + "\n"
