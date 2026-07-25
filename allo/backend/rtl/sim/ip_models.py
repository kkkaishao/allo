# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral (simulation-only) models for the extern IP operators the datapath
emitter instantiates -- a float/double arithmetic core, a floating-point compare,
a float<->float resize, an int<->float cast, or an advanced ``math.*`` op, none of
which has a comb lowering.

The behavior is not inferred from the module name. The device's operator table,
the same ``@ip`` records the scheduler was characterized from, is threaded in as
:class:`OpDesc` descriptors, and the port manifest supplies the *structural*
facts a descriptor cannot:

* which operators were actually instantiated;
* their realized port names, widths and roles;
* the per-instance predicate of a float compare.

Every one of those is authored by the emitter and read here verbatim, so this
module never parses IR or re-derives a name.

An operator's behavior is a single **C expression** over its operands, bound as
the positional typed C variables ``a``, ``b``, ``c``, ... . A built-in operator's
expression comes from its abstract ``kind`` (:data:`_KIND_EXPR`); a user IP may
override it with ``@ip.add_c_model("<expr>")`` (``OpDesc.c_expr``). Both feed one
renderer: the operands are reinterpreted from raw bits at their dtype (so one
``"a + b"`` covers f32/f64/bf16 -- the binding is per-operator, the expression is
type-generic), the expression is evaluated, and the result's bits are returned.
The SystemVerilog shell and the DPI-C function are rendered from the templates in
``templates/`` (str.format), mirroring the Vitis backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- descriptors (from the device operator table) --------------------------


@dataclass(frozen=True)
class Ty:
    """One operand/result dtype, as the behavioral model needs to see it."""

    name: str  # allo dtype name: float32 / float64 / bfloat16 / int32 / uint32 ...
    width: int  # bit width
    is_float: bool
    signed: bool  # meaningful for integers only


@dataclass(frozen=True)
class OpDesc:
    """One device operator IP, the behavioral source of truth. ``name`` is the
    operator's ``sym_name`` -- the extern module's base name (the emitter may
    append a ``_<predicate>`` suffix per compare instance). ``c_expr`` is a user
    ``add_c_model`` C expression over the operands ``a``, ``b``, ...; ``None``
    falls back to the built-in :data:`_KIND_EXPR` for ``kind``."""

    name: str
    kind: str  # abstract kind: add/sub/mul/div/rem/cmp/ifcast/fcast/<math mnemonic>
    latency: int
    arg_types: tuple[Ty, ...]
    ret_type: Ty
    c_expr: str | None = None


# --- the externs, as the port manifest declares them ------------------------


@dataclass(frozen=True)
class _Extern:
    """An instantiated extern operator module: the descriptor plus the realized
    port shape from the manifest, with names preserved so the behavioral module
    matches the extern's ports exactly. Each port carries its role, so the clock,
    the optional clock-enable and the result are found structurally; renaming one
    in the emitter cannot silently turn it into a data operand."""

    name: str  # the extern module's RTL name
    ports: tuple[tuple[str, int, str], ...]  # (name, width, role) in order
    pred: str  # compare predicate; "" if none
    desc: OpDesc

    def _of_role(self, role: str) -> list[tuple[str, int]]:
        return [(n, w) for n, w, r in self.ports if r == role]

    def data_inputs(self) -> list[tuple[str, int]]:
        return self._of_role("data")

    @property
    def has_ce(self) -> bool:
        return bool(self._of_role("ce"))

    @property
    def clk(self) -> str:
        clks = self._of_role("clk")
        assert clks, f"extern {self.name} has no clock port"
        return clks[0][0]

    @property
    def out(self) -> tuple[str, int]:
        outs = self._of_role("out")
        assert outs, f"extern {self.name} has no output port"
        return outs[0]


_TEMPLATE_DIR = Path(__file__).with_name("templates")


def _render(template: str, **kw: object) -> str:
    return (_TEMPLATE_DIR / template).read_text(encoding="utf-8").format(**kw)


# --- built-in behavior: abstract kind -> a C expression over `a`, `b` -------

# Float-compare predicate -> C comparison operator. The ordered (o*) and unordered
# (u*) RELATIONAL predicates map to the same operator: cosim inputs are NaN-free,
# so the only place they would differ (a NaN operand) does not arise.
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
# The non-relational predicates, which are not `a <op> b`: the NaN tests `uno`
# (either operand is NaN) / `ord` (neither is) -- modeled exactly via std::isnan,
# so the max/min expansion's `cmpf uno` NaN guard cosims -- and the two constant
# predicates.
_CMP_NONREL = {
    "uno": "std::isnan(a) || std::isnan(b)",
    "ord": "!std::isnan(a) && !std::isnan(b)",
    "true": "true",
    "false": "false",
}
_ARITH = {"add": "a + b", "sub": "a - b", "mul": "a * b", "div": "a / b"}
# Advanced math mnemonic -> C++ <cmath> function (overloaded on float/double).
_LIBM = {
    "sqrt": "std::sqrt",
    "exp": "std::exp",
    "log": "std::log",
    "sin": "std::sin",
    "cos": "std::cos",
    "tan": "std::tan",
    "tanh": "std::tanh",
    "abs": "std::fabs",
    "absf": "std::fabs",
    "floor": "std::floor",
    "ceil": "std::ceil",
}


def _kind_expr(desc: OpDesc, pred: str) -> str:
    """The built-in C expression for ``desc.kind`` over operands ``a``, ``b``."""
    k = desc.kind
    if k in _ARITH:
        return _ARITH[k]
    if k == "rem":
        return "std::fmod(a, b)"
    if k == "cmp":
        if pred in _CMP:
            return f"a {_CMP[pred]} b"
        expr = _CMP_NONREL.get(pred)
        assert expr is not None, f"unsupported float-compare predicate '{pred}'"
        return expr
    if k in _LIBM:  # advanced unary math
        return f"{_LIBM[k]}(a)"
    if k in ("ifcast", "fcast"):  # unary conversion: value cast src -> dst
        return f"({_cscalar(desc.ret_type)})a"
    raise NotImplementedError(
        f"no cosim behavioral model for operator kind '{k}' (operator "
        f"'{desc.name}'); attach one with @ip.add_c_model(\"<C expression>\")"
    )


def _compute_expr(e: _Extern) -> str:
    """The operator's C expression: a user ``add_c_model`` override, else the
    built-in expression for its kind."""
    return e.desc.c_expr if e.desc.c_expr is not None else _kind_expr(e.desc, e.pred)


# --- operand / result bit-pattern ABI --------------------------------------


def _cscalar(ty: Ty) -> str:
    """The C scalar type a value of ``ty`` is computed in."""
    if ty.is_float:
        return "double" if ty.name == "float64" else "float"  # bf16 computes in float
    return f"int{ty.width}_t" if ty.signed else f"uint{ty.width}_t"


def _load(ty: Ty, raw: str, name: str) -> str:
    """C statement(s) binding operand ``name`` to a typed value from raw bits ``raw``."""
    if ty.is_float:
        if ty.name == "bfloat16":
            return (
                f"float {name}; {{ unsigned int _u = "
                f"((unsigned int)({raw} & 0xFFFFu)) << 16; memcpy(&{name}, &_u, 4); }}"
            )
        cty, n = ("double", 8) if ty.name == "float64" else ("float", 4)
        return f"{cty} {name}; memcpy(&{name}, &{raw}, {n});"
    assert ty.width in (1, 8, 16, 32, 64), f"unsupported int width {ty.width}"
    ity = f"int{ty.width}_t" if ty.signed else f"uint{ty.width}_t"
    return f"{ity} {name} = ({ity}){raw};"


def _store(ty: Ty, val: str) -> str:
    """C expression producing the ``long long`` raw bits of typed value ``val``."""
    if ty.is_float:
        if ty.name == "bfloat16":
            return f"({{ unsigned int _u; memcpy(&_u, &{val}, 4); (long long)(_u >> 16); }})"
        n = 8 if ty.name == "float64" else 4
        return f"({{ long long _o = 0; memcpy(&_o, &{val}, {n}); _o; }})"
    return f"(long long)({val})"


# --- join externs to descriptors -------------------------------------------


def _plan(interfaces: dict, descs) -> list[_Extern]:
    """The extern operators instantiated across every emitted module, each joined
    to its device descriptor. The manifest names the descriptor (``impl``) and the
    per-instance predicate outright, so the join is a plain lookup. One entry per
    module, since several kernels may instantiate the same operator and share one
    behavioral model."""
    by_name = {d.name: d for d in descs}
    seen: dict[str, _Extern] = {}
    for iface in interfaces.values():
        for op in iface["operators"]:
            if op["module"] in seen:
                continue
            desc = by_name.get(op["impl"])
            assert desc, f"extern operator '{op['impl']}' has no device operator"
            ports = tuple((p["name"], p["width"], p["role"]) for p in op["ports"])
            seen[op["module"]] = _Extern(op["module"], ports, op["predicate"], desc)
    return list(seen.values())


# --- rendering -------------------------------------------------------------


def _dpi_name(e: _Extern) -> str:
    """A DPI function name unique per behavior (the operator + its predicate)."""
    return f"allo_op_{e.desc.name}" + (f"_{e.pred}" if e.pred else "")


def _dpi_slots(e: _Extern) -> tuple[str, str]:
    """The rendered ``binds`` and ``body`` for one operator's DPI function."""
    d = e.desc
    binds = "".join(
        f"  {_load(t, f'p{k}', chr(ord('a') + k))}\n" for k, t in enumerate(d.arg_types)
    )
    expr = _compute_expr(e)
    if d.ret_type.is_float:
        body = f"  {_cscalar(d.ret_type)} _r = ({expr});\n  return {_store(d.ret_type, '_r')};"
    else:  # int / bool result (a compare, a float->int cast): no reinterpret
        body = f"  return {_store(d.ret_type, f'({expr})')};"
    return binds, body


def dpi_c(interfaces: dict, descs) -> str:
    """C implementations of the DPI operators the instantiated externs need."""
    plan = _plan(interfaces, descs)
    if not plan:
        return ""
    fns: dict[str, str] = {}
    for e in plan:
        name = _dpi_name(e)
        if name in fns:
            continue
        params = ", ".join(f"long long p{k}" for k in range(len(e.desc.arg_types)))
        binds, body = _dpi_slots(e)
        fns[name] = _render(
            "dpi_op.c.in", name=name, params=params, binds=binds, body=body
        )
    return _render("dpi_c.in", functions="\n".join(fns.values()))


def sv_models(interfaces: dict, descs) -> str:
    """SystemVerilog behavioral models + DPI import decls for the instantiated
    extern IP operators."""
    plan = _plan(interfaces, descs)
    if not plan:
        return ""
    imports: dict[str, str] = {}
    modules = []
    for e in plan:
        dpi = _dpi_name(e)
        params = ", ".join(f"input longint p{k}" for k in range(len(e.desc.arg_types)))
        imports[dpi] = f'import "DPI-C" function longint {dpi}({params});'

        lat = e.desc.latency
        assert (
            lat >= 1
        ), f"operator '{e.desc.name}' needs latency >= 1 for a shift model"
        ins = e.data_inputs()
        assert len(ins) == len(e.desc.arg_types), (
            f"operator '{e.desc.name}': extern has {len(ins)} data ports but the "
            f"descriptor declares {len(e.desc.arg_types)} operands"
        )
        out_name, outw = e.out
        ports = ", ".join(f"input [{w - 1}:0] {n}" for n, w in ins)
        ports += f", input {e.clk}"
        if e.has_ce:
            ports += ", input ce"
        ports += f", output [{outw - 1}:0] {out_name}"
        call = f"{dpi}(" + ", ".join(n for n, _ in ins) + ")"
        modules.append(
            _render(
                "sv_op.sv.in",
                name=e.name,
                ports=ports,
                msb=outw - 1,
                last=lat - 1,
                clk=e.clk,
                guard="if (ce) " if e.has_ce else "",
                call=call,
                latency=lat,
                out_name=out_name,
            )
        )
    return _render(
        "sv_models.in",
        imports="\n".join(imports.values()),
        modules="\n".join(modules),
    )
