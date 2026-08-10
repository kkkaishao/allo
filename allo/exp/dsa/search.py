# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search backend: compile a source program onto an ISA.

The source program is a TOSA-dialect MLIR module supplied as *text* (e.g. from
torch_mlir's TOSA backend) — ``compile_program`` parses it; we own no source
generator. A ``Catalog`` indexes the ISA's compute instructions by the root *prim
tag* of their pattern DAG (and recognizes the tag of a source op); three stages
then run, top to bottom in compilation order:

- ``match_program`` (Stage 1) — cover the source compute DAG with instruction
  patterns via cost-aware tree-DP, folding a multi-node subgraph into one
  instruction; binds each instruction's source buffers to source SSA values.
- ``solve`` (Stage 2) — infer each instruction's shape params by unifying its
  symbolic visible shapes with the bound source shapes (exact-fit; no tiling).
- ``solve_layouts`` (Stage 2b) — infer the access params that describe *residence*
  (strides, a ``layout``'s dimension ordering) by unifying the index maps of every
  access of one value; program I/O is pinned to the host ABI.
- ``plan`` (Stage 3) — liveness-driven slot allocation + data movement (routing
  and spilling), producing a ``CompiledProgram`` (a placed program + I/O map).

The public entry is ``ISA.compile_program(source)`` (sugar over ``compile_program``
here); the returned ``CompiledProgram`` is callable — ``prog(*inputs)`` runs it on
the functional simulator (the same oracle backbone hand-written assembly uses) — and
``prog.dump()`` prints the emitted instruction sequence.

See ``todos/search.md`` for the full per-stage algorithm analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod

import ml_dtypes
import numpy as np
import sympy

from ..._mlir import ir
from ..._mlir.dialects import tosa
from . import primitive
from .core import (
    ISA,
    Instruction,
    ScalarProxy,
    _index_params,
    access_map,
    arity,
    buffer_weights,
    compute_params,
    dense_strides,
    layout_params,
    param_roles,
    pin_access,
    residence,
    show_map,
    trace_instruction,
)
from .errors import (
    AcceleratorDescriptionError,
    AllocationError,
    AssemblyError,
    CompileError,
    DTypeError,
    LayoutError,
    NoMatchError,
    ShapeError,
)
from .oracle import EmitRecord, OracleConfig, OracleProgram, _np_dtype, simulate

# ==========================================================================#
# Catalog: prim-tag index + source-op recognizer
# ==========================================================================#

# Source op-name -> prim tag. The source is value-semantics TOSA throughout, and a
# prim's source op name is always `tosa.<tag>`, so the map is derived from the prim
# registry; matmul/transpose are bespoke (not in the registry). relu is recognized
# separately (it is a tosa.clamp with min == 0, not a tosa.relu). tosa.matmul is
# batched 3-D (its 2-D<->3-D reshapes are handled by `_canon`).
_NAMED_TAG = {f"tosa.{tag}": tag for tag in primitive.REGISTRY}
# bespoke prims (not in the registry): matmul / transpose / reverse and the conv family.
for _bespoke in (
    "matmul",
    "transpose",
    "reverse",
    "conv2d",
    "depthwise_conv2d",
    "max_pool2d",
    "avg_pool2d",
):
    _NAMED_TAG[f"tosa.{_bespoke}"] = _bespoke

# Pure layout / constant ops: transparent to matching (reshape is an alias; consts
# carry no compute). `_canon` peels reshapes; use-counting skips all of these.
_LAYOUT_AND_CONST = {"tosa.reshape", "tosa.const", "tosa.const_shape"}


def _canon(value):
    """Peel ``tosa.reshape`` chains to the underlying value. Reshape is a layout
    alias (e.g. TOSA's batched-matmul 2-D<->3-D wrapping at I/O), transparent for
    both matching and allocation — two reshapes of one value share its slot."""
    while True:
        owner = value.owner
        if isinstance(owner, ir.Block) or owner.operation.name != "tosa.reshape":
            return value
        value = owner.operands[0]  # reshape input1 (the data operand)


def _reads_index_ge(node, n_src: int) -> bool:
    """True if the compute DAG reads a buffer arg at index >= ``n_src`` (a
    destination). Such an instruction reads its own output (an in-place accumulate)."""
    if node.kind == "arg":
        return node.buffer_index >= n_src
    return any(_reads_index_ge(a, n_src) for a in node.args)


def instruction_pattern(instruction: Instruction):
    """The compute pattern of an instruction: the root ``TensorProxy`` of its
    semantics DAG. Internal nodes are prim ops; ``arg`` leaves bind to the
    instruction's source buffers (by ``buffer_index``). A 1:1 instruction is just
    a depth-1 pattern. Returns ``None`` for data-movement (identity), multi-output
    instructions (not matched yet), or an instruction whose compute reads its own
    destination buffer (an in-place accumulate): the matcher cannot bind a dst read,
    so it is oracle-only — to compile an accumulate, model the accumulator as a
    source operand (``dst = add(c_in, matmul(a, b))``), which the allocator then
    coalesces back onto one slot."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return None
    root = results[0]
    if root.kind in ("identity", "arg"):
        return None
    if _reads_index_ge(root, len(instruction.spec.sources)):
        return None  # reads its own destination -> not selectable (would mis-bind)
    return root


# Recognized source ops carrying trailing *non-data* operands: the count of leading
# data operands. Everything after them — tosa.mul's shift, the matmul / conv /
# negate / avg_pool zero-points — is quantization, which this frontend does not
# model. They are therefore not dropped but *required to be neutral*, else a `>>3`
# fixed-point multiply would select a plain float `mul` and run.
_DATA_OPERANDS = {
    "tosa.mul": 2,  # + shift
    "tosa.matmul": 2,  # + a_zp, b_zp
    "tosa.negate": 1,  # + input_zp, output_zp
    "tosa.conv2d": 3,  # input, weight, bias (+ input_zp, weight_zp)
    "tosa.depthwise_conv2d": 3,
    "tosa.avg_pool2d": 1,  # + input_zp, output_zp
}

# The largest finite value of each float type. torch_mlir's TOSA backend spells
# relu's *open* upper bound as this value rather than +inf, so both count as "does
# not clip from above".
_FLOAT_MAX = {
    "f16": float(np.finfo(np.float16).max),
    "bf16": float(ml_dtypes.finfo(ml_dtypes.bfloat16).max),
    "f32": float(np.finfo(np.float32).max),
    "f64": float(np.finfo(np.float64).max),
}


def _source_ins(op) -> list:
    """The data-input operands of a recognized (value-semantics TOSA) source op."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return list(op.operands)[:n] if n is not None else list(op.operands)


def const_elements(value) -> list | None:
    """The elements of a source ``tosa.const``, or ``None`` if ``value`` is not a
    constant we can read.

    ``None`` covers both "not a constant" and "a constant whose data lives in a
    ``dialect_resource`` blob" — torch's TOSA backend stores model weights that way
    and the Python bindings expose no reader. Callers must treat unknown as
    *unusable*, never as a default value."""
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name != "tosa.const":
        return None
    attr = owner.operation.attributes["values"]
    if not isinstance(attr, (ir.DenseFPElementsAttr, ir.DenseIntElementsAttr)):
        return None
    return list(attr)


def _is_zero_const(value) -> bool:
    """True only if ``value`` is *provably* an all-zero constant."""
    elems = const_elements(value)
    return elems is not None and all(v == 0 for v in elems)


def _quantization_is_neutral(op) -> bool:
    """Whether ``op``'s trailing shift / zero-point operands are all zero."""
    n = _DATA_OPERANDS.get(op.operation.name)
    return n is None or all(_is_zero_const(v) for v in list(op.operands)[n:])


def source_tag(op) -> str | None:
    """Recognize a source op into a prim tag, or None if unsupported.

    Recognition is **fail-safe**: an op earns a tag only when every part of its
    definition is accounted for by a prim. Anything unmodeled — a non-zero shift or
    zero-point, a clamp that is not relu — yields ``None``, which costs a clean
    "no instruction matches" rather than a program that compiles and computes a
    different function."""
    name = op.operation.name
    tag = _NAMED_TAG.get(name)
    if tag is None:
        if name != "tosa.clamp" or not _is_relu_clamp(op):
            return None
        tag = "relu"
    return tag if _quantization_is_neutral(op) else None


def _is_relu_clamp(op) -> bool:
    """``tosa.clamp`` is relu only when it clamps to ``[0, +inf)`` over a float
    type — the form torch_mlir's TOSA backend emits for aten.relu.

    *Both* bounds are semantics. Checking only the lower one recognized relu6
    (``[0, 6]``) as relu, which compiled and ran and returned wrong numbers. An
    integer clamp carries ``IntegerAttr`` bounds — a different attribute type, and
    no unbounded value — so it is never this pattern."""
    elt = ir.ShapedType(op.operands[0].type).element_type
    if str(elt) not in _FLOAT_MAX:
        return False
    attrs = op.operation.attributes
    return (
        ir.FloatAttr(attrs["min_val"]).value == 0.0
        and ir.FloatAttr(attrs["max_val"]).value >= _FLOAT_MAX[str(elt)]
    )


def pattern_alpha(node) -> set:
    """The computational-attribute (α) indices a pattern's ``const`` leaves read."""
    if node.kind == "const":
        return (
            {node.value.param_index} if isinstance(node.value, ScalarProxy) else set()
        )
    out: set = set()
    for a in node.args:
        out |= pattern_alpha(a)
    return out


class Catalog:
    """Indexes an ISA's compute instructions by the *root* prim tag of their
    pattern, so single-op and multi-node instructions are looked up uniformly."""

    def __init__(self, isa: ISA):
        self.isa = isa
        self.patterns: dict[str, list[tuple[Instruction, object]]] = {}
        for spec in isa.instructions:
            instr = isa._ops[spec.name]
            root = instruction_pattern(instr)
            if root is None:
                continue  # oracle-only: its α (if any) is supplied by hand
            # Every α a *selectable* instruction declares has to be bound from the
            # source, and the only thing that binds one is a const leaf. One that
            # appears nowhere in the pattern has no value the compiler could give it.
            missing = set(range(len(compute_params(spec)))) - pattern_alpha(root)
            if missing:
                names = [compute_params(spec)[i] for i in sorted(missing)]
                raise AcceleratorDescriptionError(
                    f"{spec.name}: compute param(s) {names} never appear in the "
                    f"semantics (a compute param reaches the compute DAG through "
                    f"primitive.const), so nothing binds them at a match"
                )
            self.patterns.setdefault(root.kind, []).append((instr, root))

    def candidates(self, tag: str | None) -> list[tuple[Instruction, object]]:
        return self.patterns.get(tag, []) if tag is not None else []


# ==========================================================================#
# Stage 1 — semantic matching
# ==========================================================================#


@dataclass
class Match:
    instruction: object  # Instruction
    operand_values: list  # leaf bindings, in source-buffer order
    result_value: object  # the tile's output ir.Value
    # Solved access param -> int. ``None`` until Stage 1 solves it, and left ``None``
    # for a candidate whose shapes do NOT fit, which is what Stage 2 re-solves to get
    # the diagnostic. Stage 2b then adds the residence params to the same dict.
    shape_params: dict | None = None
    alpha: dict = field(default_factory=dict)  # compute param -> bound immediate

    @property
    def bound_values(self) -> list:
        """Source values bound to the instruction buffers, in [src..., dst] order."""
        return self.operand_values + [self.result_value]


@dataclass
class Selection:
    func: object
    matches: list


def _entry_block(module):
    for op in module.body.operations:
        if op.operation.name == "func.func":
            return op, op.regions[0].blocks[0]
    raise CompileError("source module has no func.func")


# ==========================================================================#
# Source normalization (run on the parsed TOSA before matching)
# ==========================================================================#


def _const_shape(shape):
    n = len(shape)
    vals = ir.Attribute.parse(
        f"dense<[{', '.join(map(str, shape))}]> : tensor<{n}xindex>"
    )
    return tosa.ConstShapeOp(ir.Type.parse(f"!tosa.shape<{n}>"), vals).result


def normalize_source(module):
    """Canonicalize torch_mlir's TOSA so instruction patterns can match it.

    Sinks ``reshape(transpose(X, p))`` -> ``transpose(reshape(X), p')`` when the
    reshape only prepends unit (batch) dims. torch lowers ``a @ b.T`` as a 2-D
    ``tosa.transpose`` (perms ``[1, 0]``) *then* a batch reshape to 3-D, whereas an
    instruction's semantics carry the weight transpose in batched 3-D form (perms
    ``[0, 2, 1]``). Without this rewrite the two never line up and the systolic
    matmul cannot absorb the transpose."""
    _, block = _entry_block(module)
    for op in list(block.operations):
        if op.operation.name != "tosa.reshape":
            continue
        t = op.operands[0].owner
        if isinstance(t, ir.Block) or t.operation.name != "tosa.transpose":
            continue
        in_ty = ir.RankedTensorType(t.operands[0].type)
        t_out = ir.RankedTensorType(t.results[0].type)
        r_out = ir.RankedTensorType(op.results[0].type)
        k = r_out.rank - t_out.rank  # number of prepended dims
        if (
            k <= 0
            or list(r_out.shape[:k]) != [1] * k
            or list(r_out.shape[k:]) != list(t_out.shape)
        ):
            continue  # reshape does more than prepend unit dims -> leave it alone
        perms = list(ir.DenseI32ArrayAttr(t.operation.attributes["perms"]))
        new_perms = list(range(k)) + [p + k for p in perms]
        new_shape = [1] * k + list(in_ty.shape)
        with ir.InsertionPoint(op), ir.Location.unknown():
            reshaped = tosa.ReshapeOp(
                t.operands[0],
                _const_shape(new_shape),
                results=[ir.RankedTensorType.get(new_shape, in_ty.element_type)],
            )
            new_t = tosa.TransposeOp(r_out, reshaped.result, new_perms)
        op.results[0].replace_all_uses_with(new_t.result)
        op.operation.erase()
        if not list(
            t.results[0].uses
        ):  # the old transpose precedes op, already visited
            t.operation.erase()


_COMMUTATIVE = {tag for tag, p in primitive.REGISTRY.items() if p.commutative}
_REDUCE_TAGS = {
    tag for tag, p in primitive.REGISTRY.items() if p.category == primitive.REDUCE
}


def _perms(op) -> list:
    """The permutation of a ``tosa.transpose`` source op (an ``array<i32>`` attr)."""
    return list(ir.DenseI32ArrayAttr(op.operation.attributes["perms"]))


def _axis_attr(op) -> int:
    """The ``axis`` i32 attr of a ``tosa.reduce_*`` / ``tosa.reverse`` source op."""
    return ir.IntegerAttr(op.operation.attributes["axis"]).value


def _i64_attr(op, name) -> list:
    return list(ir.DenseI64ArrayAttr(op.operation.attributes[name]))


_CONV_TAGS = {"conv2d", "depthwise_conv2d"}
_POOL_TAGS = {"max_pool2d", "avg_pool2d"}


def _attrs_match(pnode, op) -> bool:
    """Spatial attributes are part of the semantics (like transpose's perms)."""
    if pnode.kind in _CONV_TAGS:
        a = pnode.attrs
        return all(_i64_attr(op, k) == a[k] for k in ("pad", "stride", "dilation"))
    if pnode.kind in _POOL_TAGS:
        a = pnode.attrs
        return all(_i64_attr(op, k) == a[k] for k in ("kernel", "stride", "pad"))
    return True


def _match_const(pnode, value, alpha: dict) -> bool:
    """A ``const`` pattern leaf against a source constant — compared, or bound.

    A **fixed** literal is part of the semantics — ``pow(2, x)`` and ``pow(3, x)`` are
    different functions — so this is the constant's counterpart of a transpose's
    permutation check. Compared after rounding through the pattern's dtype, since
    the source holds an f32 while the ISA was written with a Python float.

    A **parametric** one (a ``ScalarProxy``: ACT's α) is instead *bound* into ``alpha``:
    the ISA states that this operand is an immediate the instruction word carries, and
    the match reads its value off the source. This is the only place a source value
    flows into an instruction's encoding rather than into memory."""
    elems = const_elements(_canon(value))
    if elems is None:
        return False
    if isinstance(pnode.value, ScalarProxy):
        return _bind_alpha(pnode.value, elems, alpha)
    cast = _np_dtype(pnode.dtype).type
    want = cast(pnode.value)
    return all(cast(v) == want for v in elems)


def _bind_alpha(param: ScalarProxy, elems: list, alpha: dict) -> bool:
    """Bind one α from a source constant's elements, or reject.

    An immediate field holds **one integer**, so the constant has to be a splat whose
    value is an exact integer: ``x + 3.0`` binds ``#k = 3``, ``x + 0.5`` does not match
    at all (rounding it would silently compile a different function), and a
    non-uniform constant is program data, not an immediate. A param appearing twice in
    one pattern must bind the same value both times."""
    first = elems[0]
    if any(v != first for v in elems) or int(first) != first:
        return False
    return alpha.setdefault(param.param_index, int(first)) == int(first)


def _operand_orders(pnode, ins):
    """Operand orderings to try; both orders for a commutative binary prim."""
    if pnode.kind in _COMMUTATIVE and len(ins) == 2:
        return (ins, [ins[1], ins[0]])
    return (ins,)


def _match_pattern(pnode, value, def_op, alpha, bindings, within, interior) -> bool:
    """Match pattern node ``pnode`` against source ``value``.

    ``arg`` leaves bind a source buffer to ``value`` and ``const`` leaves either check
    or bind a literal (``alpha``); internal nodes must align with a recognized source
    op of the same prim tag and recurse on its inputs.
    Records each folded source value (whose defining op is absorbed into the tile)
    in ``interior``, and the within-tile use count of each operand in ``within``, so
    the caller can reject a fold in which a folded non-root value *escapes* (is also
    used outside the tile and therefore must be materialized). This deferred
    cut-point test permits internal fan-out — e.g. softmax's ``exp`` feeding both the
    reduce and the divide — which a per-node single-use test would wrongly forbid.
    Mutates ``alpha``/``bindings``/``within``/``interior`` in place, rolling back on a
    failed branch.
    """
    if pnode.kind == "arg":
        prev = bindings.get(pnode.buffer_index)
        if prev is not None and prev != value:
            return False
        bindings[pnode.buffer_index] = value
        return True
    if pnode.kind == "const":
        return _match_const(pnode, value, alpha)
    # Recognize an internal (folded) op through any reshape wrappers (torch emits
    # 2-D<->3-D reshapes around matmul); arg leaves above stay raw so their shapes
    # still drive the shape solver.
    op = def_op.get(_canon(value))
    if op is None or source_tag(op) != pnode.kind:
        return False
    if pnode.kind == "transpose" and _perms(op) != list(pnode.permutation):
        return False  # the permutation is part of the semantics, not just the tag
    if pnode.kind in _REDUCE_TAGS and _axis_attr(op) != pnode.axis:
        return False  # the reduced axis is part of the semantics
    if pnode.kind == "reverse" and _axis_attr(op) != pnode.axis:
        return False  # the reversed axis is part of the semantics
    if not _attrs_match(pnode, op):
        return False  # conv/pool spatial attrs (pad/stride/dilation/kernel)
    ins = _source_ins(op)
    if len(ins) != len(pnode.args):
        return False
    for order in _operand_orders(pnode, ins):
        saved = dict(bindings), dict(alpha), dict(within), set(interior)
        for sv in order:  # each operand is one within-tile use of that value
            within[_canon(sv)] = within.get(_canon(sv), 0) + 1
        if all(
            _match_pattern(pa, sv, def_op, alpha, bindings, within, interior)
            for pa, sv in zip(pnode.args, order)
        ):
            interior.add(_canon(value))  # canonical key (matches use/within counting)
            return True
        for live, restore in zip((bindings, alpha, within, interior), saved):
            live.clear()
            live.update(restore)
    return False


@dataclass
class _Choice:
    cost: float
    instruction: object
    operands: list
    shape_params: dict | None  # solved sizes, or None if the shapes do not fit
    alpha: dict  # compute params bound from the source's constants


def _pattern_has(node, kind) -> bool:
    return node.kind == kind or any(_pattern_has(a, kind) for a in node.args)


def _describe_pattern(node) -> str:
    """A compact source-level rendering of an instruction's compute pattern, e.g.
    ``matmul(%0, transpose(%1))`` — the shape of source DAG it matches. A ``const``
    leaf shows its literal, or ``#name`` when it is a compute param (α)."""
    if node.kind == "arg":
        return f"%{node.buffer_index}"
    if node.kind == "const":
        return f"{node.value}"  # ScalarProxy renders as `#name`
    return f"{node.kind}({', '.join(_describe_pattern(a) for a in node.args)})"


def _no_match_error(op, catalog) -> str:
    """An actionable message for an unmatched source op: show its operand shapes,
    the candidate instructions' patterns, and — the common case — a hint when an
    instruction consumes an operand transposed but the source provides it plain."""
    tag = source_tag(op)
    shapes = [tuple(ir.RankedTensorType(o.type).shape) for o in _source_ins(op)]
    head = f"no instruction matches source op '{op.operation.name}' with operand shapes {shapes}"
    if not _quantization_is_neutral(op):
        return (
            f"{head}: it carries a non-zero shift / zero-point operand, and this "
            f"frontend models no quantization — so the op is not recognized at all "
            f"rather than matched as its unquantized namesake."
        )
    if tag is None:
        return f"{head}: no prim in the compute vocabulary models this op."
    candidates = catalog.candidates(tag)
    if not candidates:
        return f"{head}: the ISA defines no instruction computing '{tag}'."
    lines = [
        f"{head}.",
        f"  '{tag}' instruction(s) exist but none matches structurally:",
    ]
    lines += [
        f"    {instr.name}: {_describe_pattern(root)}" for instr, root in candidates
    ]
    if any(_pattern_has(root, "transpose") for _, root in candidates):
        lines.append(
            "  hint: an instruction consumes an operand transposed (the systolic "
            "computes X @ W^T). Write the source op in that form (e.g. `a @ b.T`) "
            "or pre-transpose the operand on the host."
        )
    return "\n".join(lines)


def match_program(catalog: Catalog, source_module) -> Selection:
    """Cover the source compute DAG with instruction patterns via cost-aware
    tree-DP. A value used more than once is a forced cut point (it cannot be
    folded into a consumer's tile), so the foldable subgraphs are trees and a
    per-value DP is globally optimal.

    ``materialize(v)`` returns the cheapest tile rooted at ``v``: instruction cost
    plus the materialization cost of its operands — but only *single-use* operands
    are charged, because a shared (multi-use) operand is materialized once as its
    own root and must not be billed to every consumer. The optimum is reconstructed
    from the returned values and scheduled in def-before-use order."""
    func, block = _entry_block(source_module)
    ops = list(block.operations)
    terminator = ops[-1]

    def_op: dict = {}  # ir.Value -> the recognized op defining it
    index: dict = {}  # that op's result value -> block position (for scheduling)
    for i, op in enumerate(ops):
        if source_tag(op) is not None:
            def_op[op.results[0]] = op
            index[op.results[0]] = i

    # Use-counts on canonical values, skipping pure layout/const ops (a reshape's
    # use of its input is plumbing, not a real consumer).
    use: dict = {}
    for op in ops:
        if op.operation.name in _LAYOUT_AND_CONST:
            continue
        for v in op.operands:
            cv = _canon(v)
            use[cv] = use.get(cv, 0) + 1

    memo: dict = {}  # canonical value -> _Choice (optimal tile to materialize it)

    def materialize(v) -> _Choice:
        if v in memo:
            return memo[v]
        op = def_op[v]
        fitting = None  # cheapest candidate that also *fits* the source shapes
        fallback = None  # first structural match that does not fit (error reporting)
        for instr, root in catalog.candidates(source_tag(op)):
            bindings, alpha, within, interior = {}, {}, {}, set()
            if not _match_pattern(root, v, def_op, alpha, bindings, within, interior):
                continue
            # Deferred cut-point test: a folded (non-root) value must be used only
            # within this tile; if its global use count exceeds its within-tile use
            # count it escapes and must be its own root, so this fold is invalid.
            if any(use.get(iv, 0) != within.get(iv, 0) for iv in interior if iv != v):
                continue
            n_src = len(instr.spec.sources)
            if not all(i in bindings for i in range(n_src)):
                continue
            operands = [bindings[i] for i in range(n_src)]
            # An ISA may offer both a fixed-size tile instruction and a layer-level
            # (parametric, @expand-ing) one for the same prim. They match the same
            # structure, so structure alone cannot choose: solve here and prefer a
            # candidate that actually *fits* the source shapes. The solved params are
            # carried on (Match.shape_params) so no later stage re-solves them; keeping
            # one unfitting candidate lets Stage 2 report *why* nothing fits when that
            # is the real error.
            fit = _fit(instr, operands, v)
            if fit is None:
                if fallback is None:
                    fallback = _Choice(0.0, instr, operands, None, alpha)
                continue
            cost = instr.spec.cost_of(fit) + sum(
                materialize(_canon(ov)).cost
                for ov in operands
                if _canon(ov) in def_op and use.get(_canon(ov), 0) == 1
            )
            # Strictly cheaper wins, so equal-cost candidates resolve to the
            # earlier-declared one — deterministic, and the right default when a
            # parametric op degenerates to exactly the fixed one it would expand into.
            if fitting is None or cost < fitting.cost:
                fitting = _Choice(cost, instr, operands, fit, alpha)
        chosen = fitting or fallback
        if chosen is None:
            raise NoMatchError(_no_match_error(op, catalog))
        memo[v] = chosen
        return chosen

    matches: list[Match] = []
    visited: set = set()

    def schedule(v):
        if v in visited:
            return
        visited.add(v)
        ch = materialize(v)
        matches.append(Match(ch.instruction, ch.operands, v, ch.shape_params, ch.alpha))
        for ov in ch.operands:
            if _canon(ov) in def_op:
                schedule(_canon(ov))

    for v in terminator.operands:
        cv = _canon(v)
        if cv in def_op:
            schedule(cv)
        elif not isinstance(cv.owner, ir.Block):
            # A returned value the recognizer refused outright: name that op and
            # why, rather than reporting the whole program as unmatched.
            raise NoMatchError(_no_match_error(cv.owner, catalog))

    if not matches:
        raise NoMatchError("no source compute ops matched any instruction")
    matches.sort(key=lambda m: index[m.result_value])
    return Selection(func, matches)


# ==========================================================================#
# Stage 2 — parameter solving / shape validation
# ==========================================================================#


def _shape(value) -> tuple:
    return tuple(ir.RankedTensorType(value.type).shape)


def _static_shape(value) -> list[int]:
    shape = list(_shape(value))
    if any(d < 0 for d in shape):
        raise ShapeError(f"source value has dynamic shape {shape}")
    return shape


def _strip_leading_units(shape) -> list:
    """Drop leading statically-1 dims. An element is an ``int`` (a source shape) or
    an ``IndexExpr`` (an instruction's visible shape); a symbolic dim is never
    dropped, since its value is unknown."""

    def unit(dim) -> bool:
        if isinstance(dim, int):
            return dim == 1
        return not _index_params(dim) and dim.static_int() == 1

    i = 0
    while i < len(shape) and unit(shape[i]):
        i += 1
    return list(shape[i:])


def _check_dtype(name: str, buf, value) -> None:
    """Unify one instruction buffer's element type with the bound source value's.

    Structural matching is type-blind, and running an ``i32`` program on an ``f32``
    datapath is a *different function* (``add`` wraps, ``intdiv`` truncates), so the
    element types have to be reconciled here.

    The one deliberate relaxation is **float-to-float**. Every float op is already
    approximate, and running it narrower changes the rounding error, not the
    operation — which is exactly what reduced-precision hardware is for (QKV's bf16
    datapath runs an f32-typed source graph, and its examples diff against a bf16
    tolerance). Integer width is not like that: ``i32`` on an ``i8`` datapath
    wraps around, so there the types must be equal."""
    elt = ir.RankedTensorType(value.type).element_type
    dtype = buf.kind.dtype
    if dtype.is_float() and str(elt) in _FLOAT_MAX:
        return
    if dtype.materialize(elt.context) != elt:
        raise DTypeError(
            f"{name}: buffer '{buf.name}' is {dtype} but the source value is {elt}"
        )


def _align_ranks(ishape, sshape) -> tuple[list, list]:
    """Align an instruction's visible shape with a bound source shape *modulo
    leading unit (batch) dims* — the shape-solver counterpart of ``_canon``.

    A leading ``1`` does not change the linear value sequence, so it is a rank alias
    carrying no shape information: ``[1, 4, 4]`` and ``[4, 4]`` describe the same 16
    values. torch_mlir makes this unavoidable — it brackets every 2-D ``a @ b`` in
    reshapes to batched 3-D and back, so within one chain the matmuls are 3-D while
    the elementwise ops around them are 2-D, and an instruction written at either
    rank meets the other (FeatherX's 3-D ``mac`` accumulates a 2-D partial sum;
    QKV's 2-D ``softmax`` consumes a 3-D matmul). Stripping only when the ranks
    actually differ leaves every same-rank comparison an exact-fit check."""
    if len(ishape) == len(sshape):
        return list(ishape), list(sshape)
    return _strip_leading_units(ishape), _strip_leading_units(sshape)


def _to_sympy(e, symtab: dict):
    """An ``IndexExpr`` (over shape params) -> a sympy expression, registering one
    nonnegative-integer ``Symbol`` per access-param index in ``symtab``."""
    if e.kind == "const":
        return sympy.Integer(e.value)
    if e.kind == "param":
        return symtab.setdefault(
            e.param_index,
            sympy.Symbol(f"p{e.param_index}", integer=True, nonnegative=True),
        )
    if e.kind == "add":
        return _to_sympy(e.lhs, symtab) + _to_sympy(e.rhs, symtab)
    if e.kind == "mul":
        return _to_sympy(e.lhs, symtab) * _to_sympy(e.rhs, symtab)
    raise NotImplementedError(f"index expr '{e.kind}'")


def _is_affine(expr, syms) -> bool:
    """True if ``expr`` is degree <= 1 in ``syms`` (a linear shape constraint). A
    higher degree is a product of params (e.g. a collapse of two symbolic dims):
    its factorization is ambiguous, so we reject rather than guess."""
    try:
        return sympy.Poly(expr, *syms).total_degree() <= 1
    except sympy.PolynomialError:
        return False


def _solve_match(m: Match) -> None:
    """Solve one match's shape params in place (see ``solve`` for the method); raises
    ``ShapeError`` if the instruction does not fit the bound source shapes."""
    spec = m.instruction.spec
    name = m.instruction.name
    _, arg_shapes, _ = trace_instruction(spec)
    bound = m.bound_values
    # By construction: `trace_instruction` yields one pattern per src+dst buffer and
    # the matcher binds exactly those. A genuine invariant, so it stays an assert.
    assert len(arg_shapes) == len(
        bound
    ), f"{name}: {len(arg_shapes)} access operands but {len(bound)} bound values"
    m.shape_params = {}
    symtab: dict = {}
    eqs = []
    for buf, ishape, value in zip(spec.buffers, arg_shapes, bound):
        _check_dtype(name, buf, value)
        ishape, sshape = _align_ranks(ishape, _static_shape(value))
        if len(ishape) != len(sshape):
            raise ShapeError(f"{name}: rank mismatch {ishape} vs {sshape}")
        for idim, sdim in zip(ishape, sshape):
            if _index_params(idim):  # depends on shape params -> an equation
                eqs.append(sympy.Eq(_to_sympy(idim, symtab), sdim))
            else:  # statically known -> exact-fit check
                fixed = idim if isinstance(idim, int) else idim.static_int()
                if fixed != sdim:
                    raise ShapeError(
                        f"{name}: shape mismatch — expects {fixed} but source is "
                        f"{sdim} (no tiling)"
                    )
    if not symtab:
        return

    syms = [symtab[i] for i in sorted(symtab)]
    for eq in eqs:
        if not _is_affine(eq.lhs - eq.rhs, syms):
            raise ShapeError(
                f"{name}: shape constraint is nonlinear in its params (a collapse of "
                f"multiple symbolic dims is ambiguous) — under-determined"
            )
    solutions = sympy.linsolve(eqs, syms)
    if not solutions:
        raise ShapeError(
            f"{name}: shapes are inconsistent — the source does not fit (no tiling)"
        )
    (values,) = solutions  # a consistent linear system has one solution tuple
    for i, val in zip(sorted(symtab), values):
        if val.free_symbols:
            raise ShapeError(
                f"{name}: shape param p{i} is under-constrained ({val}); no source "
                f"dimension pins it"
            )
        if not (val.is_integer and val >= 0):
            raise ShapeError(
                f"{name}: shape param p{i} = {val} is not a non-negative integer "
                f"(no tiling)"
            )
        m.shape_params[i] = int(val)


def solve(selection: Selection) -> Selection:
    """Infer each instruction's shape params by unifying its symbolic visible shape
    with the bound source shapes — shape inference as constraint solving.

    Every operand+result dimension yields one constraint ``visible_dim == source_dim``
    (the access patterns each contribute their own dims; ``trace_instruction`` has
    already composed them). A param-free dim is checked directly (exact fit — no
    tiling); a param-bearing dim becomes a linear equation, and the per-match system
    is solved with ``linsolve``:

    - empty solution    -> the shapes are inconsistent (the instruction does not fit);
    - a free symbol left -> a param is under-constrained (a future explicit constraint
      could pin it — for now reject and name it);
    - a unique solution  -> each param must resolve to a non-negative integer.

    Nonlinear constraints (a collapse of multiple symbolic dims) are rejected up front.
    Params that describe *residence* rather than shape — a stride, a dimension
    ordering — leave no trace in a visible shape at all, so they are not solvable here;
    ``solve_layouts`` (Stage 2b) pins them from the maps instead.

    Stage 1 already solved every *fitting* candidate (it had to, to choose among them
    and to cost a parametric instruction), so this stage only has work to do for a
    match that is known NOT to fit — precisely the case where ``_solve_match``'s
    message is the diagnostic the user needs."""
    for m in selection.matches:
        if m.shape_params is None:
            _solve_match(m)
    return selection


# ==========================================================================#
# Stage 2b — layout solving (the access params that describe residence)
# ==========================================================================#


def _dense_map(shape, buf) -> tuple:
    """The dense (row-major) residence of a value of ``shape`` in ``buf``: a flat pool
    packs it with suffix-product strides, a multi-dimensional array gives it the
    array's own pitch.

    In the I/O buffer this is the **host ABI**, the one map the compiler does not get
    to choose: ``CompiledProgram.__call__`` writes an input into the region the
    allocator gave it, densely, and reads an output back the same way. Elsewhere it is
    the default a free group falls back to."""
    if buf.address_rank == 1:
        return residence(list(zip(shape, dense_strides(shape))))
    weight = buffer_weights(buf)
    dims = _placement_dims(shape, buf)
    return residence([(d, weight[k]) for k, d in enumerate(dims)])


def _site_map(m: Match, pattern) -> tuple | None:
    """One access's residence, or ``None`` while a param of it is unsolved."""
    mapping = access_map(pattern, m.shape_params)
    if any(stride is None for _size, stride in mapping):
        return None
    return residence(mapping)


def solve_layouts(isa, selection: Selection) -> Selection:
    """Stage 2b — solve the access params that describe **residence**: strides, and
    the dimension ordering of a ``layout``.

    Neither shows up in a visible shape, so Stage 2 cannot see them. What pins them is
    the residence its neighbours describe: accesses are grouped per ``(value, buffer)``
    and a parametric one adopts the map a concrete one in its group states, which is a
    unification of index maps on the SSA edge rather than a vote among enum labels —
    the whole difference between solving an ordering and picking one. Program I/O and
    the constant pool seed their groups with the host ABI.

    This stage **solves; it does not check.** Two accesses of one value may still
    disagree afterwards, and whether that is compilable depends on the machine having a
    mover that repacks between them — which only ``plan`` knows, so ``plan`` decides
    (and inserts the relayout). Adopting the *first* concrete map is what keeps a
    parametric access from ever needing one: matches are in source order, so that map
    is the producer's.

    A group with no concrete map at all is free, and takes the dense row-major packing
    — the host's — because a cost model with no memory model prices every ordering the
    same, so anything else would be a coin flip dressed up as a choice."""
    io = _io_buffer(isa)
    block = selection.func.regions[0].blocks[0]

    sites: dict = {}  # (value, buffer name) -> [(match, pattern)]
    for m in selection.matches:
        spec = m.instruction.spec
        patterns, _, _ = trace_instruction(spec)
        for buf, pattern, value in zip(spec.buffers, patterns, m.bound_values):
            sites.setdefault((_canon(value), buf.name), []).append((m, pattern))

    # Host-supplied data: the arguments, the results, and the constant pool — all of
    # them written into (or read out of) the I/O buffer densely before/after the run,
    # so their residence there is the ABI's rather than the compiler's. ACT Def 3.8
    # puts inputs and constants in one ASM for exactly this reason.
    host = list(block.arguments) + list(list(block.operations)[-1].operands)
    host += [
        v
        for m in selection.matches
        for v in m.operand_values
        if _const_array(v) is not None
    ]
    pinned: dict = {}  # (value, buffer name) -> the residence map it must have
    for value in host:
        key = (_canon(value), io.name)
        if key in sites:
            pinned[key] = _dense_map(_static_shape(_canon(value)), io)

    def propagate() -> bool:
        moved = False
        for key, group in sites.items():
            target = pinned.get(key)
            if target is None:
                # The first concrete access sets the group's target. Matches are in
                # source order and a producer precedes its consumers, so that is the
                # producer's own map when it has one, and the host ABI when the value
                # is program data (pinned above). Accesses that then *disagree* are not
                # an error here: whether the machine can repack between them is the
                # move graph's business, so `plan` decides (and inserts a relayout).
                target = next(
                    (r for r in (_site_map(m, p) for m, p in group) if r is not None),
                    None,
                )
                if target is None:
                    continue
                pinned[key] = target
                moved = True
            _value, name = key
            for m, pattern in group:
                if _site_map(m, pattern) is not None:
                    continue
                who = f"{m.instruction.name} on '{name}'"
                # Pinning resolves a whole pattern at once, so an access sharing a
                # residence param with one already pinned is no longer unsolved and
                # is skipped above — nothing is pinned twice.
                for i, val in pin_access(pattern, m.shape_params, target, who).items():
                    assert i not in m.shape_params, f"{who}: p{i} pinned twice"
                    m.shape_params[i] = val
                    moved = True
        return moved

    while True:
        while propagate():
            pass
        free = next((k for k in sites if k not in pinned), None)
        if free is None:
            break
        pinned[free] = _dense_map(_static_shape(free[0]), isa.buffers[free[1]])

    for m in selection.matches:
        roles, _ = param_roles(m.instruction.spec)
        loose = [
            i
            for i, role in roles.items()
            if role in ("stride", "layout") and i not in m.shape_params
        ]
        if loose:
            raise LayoutError(
                f"{m.instruction.name}: residence param(s) {loose} are "
                f"under-constrained — they address a dimension no operand of this "
                f"instruction spans, so nothing pins them"
            )
    return selection


def _fit(instr, operands, result_value) -> dict | None:
    """The shape params ``instr`` solves to at this site, or ``None`` if it does not
    fit. Both a candidate filter and the source of the params a parametric
    instruction's cost is a function of, so Stage 1 solves once and carries the
    result (``Match.shape_params``) rather than any stage re-deriving it."""
    probe = Match(instr, operands, result_value)
    try:
        _solve_match(probe)
    except CompileError:
        return None
    return probe.shape_params


# ==========================================================================#
# Stage 3 — allocation, data movement, scheduling, emission
# ==========================================================================#


@dataclass
class CompiledProgram:
    isa: object
    io_buffer: object  # the global buffer holding program I/O
    emits: list  # list[EmitRecord] (the compute + data-movement stream)
    inputs: list  # per func arg: (offset, shape)
    outputs: list  # per func result: (offset, shape, label)
    constants: list = field(default_factory=list)  # (offset, ndarray), preloaded

    def _issue(self, rec) -> tuple[str, float, float]:
        """``(unit name, issue cycles, pipeline depth)`` for one emitted instruction.

        *Issue* is ``ii * trips`` — the slots the instruction occupies on its unit;
        *depth* is drain, paid once per unit rather than per instruction. Stage 2's
        solved shape params are recovered from the emitted address list (a shape
        param's slot holds its solved size).

        Requires the instruction to be bound (``ISA.bind``) to a unit with a declared
        ``ISA.latency``: an abstract search weight is not a cycle count, so an ISA with
        no modeled microarchitecture is refused rather than reported in made-up units.
        Always the latency-derived value, even where the ISA overrode ``cost`` with an
        abstract weight — these methods report cycles, not the search objective."""
        spec = self.isa._ops[rec.name].spec
        if spec.unit_latency is None or not spec.unit_latency.declared:
            raise AcceleratorDescriptionError(
                f"{self.isa.name}: '{rec.name}' has no cycle model — bind it to a "
                f"@unit (ISA.bind) and declare that unit's ISA.latency(ii=, depth=)"
            )
        roles, _ = param_roles(spec)
        shape_params = {i: rec.addr[i] for i, r in roles.items() if r == "shape"}
        lat = spec.unit_latency
        return (
            spec.unit.func_name,
            lat.ii * spec.trips_at(shape_params),
            float(lat.depth),
        )

    def unit_cycles(self) -> dict[str, float]:
        """How long each hardware unit is engaged: ``sum(ii * trips) + depth``.

        A unit issuing instructions back to back pays its pipeline drain *once*, not
        per instruction — which is exactly what ``cycles()`` cannot express, since it
        has no notion of which unit runs what. This is the quantity that says where a
        program's time actually sits, and the one to watch when a transformation moves
        work between units (tiling, or trading recompute for data movement)."""
        busy: dict[str, float] = {}
        depth: dict[str, float] = {}
        for rec in self.emits:
            unit, issue, d = self._issue(rec)
            busy[unit] = busy.get(unit, 0.0) + issue
            depth[unit] = d
        return {u: busy[u] + depth[u] for u in busy}

    def bottleneck_cycles(self) -> float:
        """The busiest unit's engaged time — a **lower** bound on the placed program
        (a roofline): every unit runs concurrently and the slowest one sets the pace.

        Pair it with ``cycles()``, which bounds the same program from **above** by
        assuming nothing overlaps. The frontend models issue cost per unit, not a
        schedule, so the true count is not derivable — but it is bracketed, and which
        end it sits near is precisely what a schedule decides. Use this one to compare
        the *shapes* of two compilations: a variant that moves work off the bottleneck
        unit is genuinely faster, whereas ``cycles()`` would charge it for the extra
        instructions as if they could never overlap with anything."""
        return max(self.unit_cycles().values(), default=0.0)

    def cycles(self) -> float:
        """Serial cycle estimate of the placed program: ``sum(depth + ii * trips)``
        over every emit.

        This assumes **no** overlap — one instruction at a time, its pipeline fully
        drained before the next issues — so it is an **upper** bound, and it charges a
        program for parallelism it may well have. ``bottleneck_cycles()`` bounds the
        same program from below; see there for which to use when."""
        total = 0.0
        for rec in self.emits:
            _unit, issue, depth = self._issue(rec)
            total += depth + issue
        return total

    def _format(self) -> str:
        io = self.io_buffer.name
        lines = [f"CompiledProgram[{self.isa.name}]  io={io}", "  inputs:"]
        for i, (off, shape) in enumerate(self.inputs):
            lines.append(f"    arg{i} = {io}{list(off)}  shape={tuple(shape)}")
        if self.constants:
            lines.append("  constants:")
            for i, (off, data) in enumerate(self.constants):
                lines.append(f"    c{i} = {io}{list(off)}  shape={tuple(data.shape)}")
        lines.append("  program:")
        for rec in self.emits:
            # `#v` marks a computational attribute (α) — an immediate in the
            # instruction word, not an address.
            args = [str(a) for a in rec.addr] + [f"#{v}" for v in rec.compute]
            lines.append(f"    {rec.name}({', '.join(args)})")
        lines.append("  outputs:")
        for off, shape, label in self.outputs:
            lines.append(f"    {label} = {io}{list(off)}  shape={tuple(shape)}")
        return "\n".join(lines)

    def dump(self) -> None:
        """Print the compiled instruction sequence (I/O map + emit stream)."""
        print(self._format())

    def __str__(self) -> str:
        return self._format()

    def _region(self, offset, shape) -> tuple:
        """The slice of the I/O buffer a value of ``shape`` placed at ``offset``
        occupies — one component per address axis, so it indexes a flat pool and a
        multi-dimensional array alike."""
        dims = _placement_dims(shape, self.io_buffer)
        return tuple(slice(o, o + d) for o, d in zip(offset, dims))

    def __call__(self, *inputs):
        """Run the compiled program on the functional simulator; returns the result
        array (or a list of arrays for a multi-output program)."""
        if len(inputs) != len(self.inputs):
            raise AssemblyError(
                f"expected {len(self.inputs)} inputs, got {len(inputs)}"
            )
        buf = self.io_buffer
        # The I/O pool's own dtype, not f32: staging an i32 program's inputs through
        # float silently rounds anything past 2**24.
        np_dt = _np_dtype(buf.kind.dtype)
        init = np.zeros(buf.memref_shape, np_dt)

        def load(offset, shape, arr):
            region = self._region(offset, shape)
            init[region] = np.asarray(arr, np_dt).reshape(init[region].shape)

        for offset, data in self.constants:
            load(offset, data.shape, data)
        for (offset, shape), arr in zip(self.inputs, inputs):
            load(offset, shape, arr)

        program = OracleProgram()
        program.steps.extend(("emit", e) for e in self.emits)
        for offset, shape, label in self.outputs:
            program.record_inspect(buf, self._region(offset, shape), label)

        results = simulate(self.isa, program, OracleConfig(init={buf: init}))
        outs = [results[label].reshape(shape) for _o, shape, label in self.outputs]
        return outs[0] if len(outs) == 1 else outs


def _solve_move_params(spec, value_size: int) -> dict:
    """Shape params for a planner-inserted movement instruction.

    A move is an identity copy, so each of its access patterns transfers the moved
    value's ``value_size`` elements: a shape param therefore satisfies
    ``prod(visible_shape) == value_size``. This is the move analogue of Stage-2
    ``solve``, which runs on matched compute instructions and so never sees a move
    the planner inserted in Stage 3. Solving the product (rather than equating the
    param with a word count) is what handles an access that *scales* its param, e.g.
    ``view(d0, a, (n, 64))``, where ``n`` is rows and the word count is ``64·n``."""
    _, arg_shapes, _ = trace_instruction(spec)
    roles, _ = param_roles(spec)
    shape_idxs = {i for i, r in roles.items() if r == "shape"}
    symtab, eqs = {}, []
    exprs = []
    for ishape in arg_shapes:
        prod_expr = sympy.Integer(1)
        for d in ishape:
            prod_expr *= (
                _to_sympy(d, symtab)
                if _index_params(d)
                else sympy.Integer(d if isinstance(d, int) else d.static_int())
            )
        exprs.append(prod_expr)
        if {i for d in ishape for i in _index_params(d)} & shape_idxs:
            eqs.append(sympy.Eq(prod_expr, value_size))
    out = {}
    if symtab:
        syms = [symtab[i] for i in sorted(symtab)]
        (vals,) = sympy.linsolve(eqs, syms)
        for i, val in zip(sorted(symtab), vals):
            if not (val.is_integer and val >= 0):
                raise ShapeError(
                    f"{spec.name}: move shape param p{i} = {val} is not a "
                    f"non-negative integer"
                )
            out[i] = int(val)
    # A move copies a whole value, so each of its patterns must transfer exactly that
    # many elements. Checked for *every* pattern, including the param-free ones a
    # fixed-size relayout is made of: routing picks moves by buffer pair, so without
    # this a value of the wrong size would be silently truncated to the tile the
    # instruction happens to describe.
    for ishape, expr in zip(arg_shapes, exprs):
        moved = expr.subs({symtab[i]: v for i, v in out.items()})
        if moved != value_size:
            raise AllocationError(
                f"{spec.name}: moves {moved} element(s) per access but the value has "
                f"{value_size} — no data-movement instruction fits this value"
            )
    return out


def _io_buffer(isa):
    globals_ = [b for b in isa.buffers.values() if b.is_global]
    if len(globals_) != 1:
        raise AcceleratorDescriptionError(
            f"expected exactly one global buffer, got {[b.name for b in globals_]}"
        )
    return globals_[0]


def _movement_catalog(isa) -> list[str]:
    """The identity (single src -> single dst) move mnemonics, in declaration order —
    the instructions the data-movement graph is built from (``_move_edges``).

    A **relayout** — an identity move whose two access patterns *differ*, e.g. a rank-2
    block of a row-major array gathered into a contiguous tile — is an ordinary move.
    Its multi-dimensional basis is filled from the operand's placement coordinate, so
    routing and spilling can use it like any other; ``_solve_move_params`` then rejects
    a value it does not fit. A machine may declare several movers between the same two
    buffers differing only in what they do to the layout, so which one applies to a
    value is decided by residence, never by the buffer pair."""
    moves = []
    for spec in isa.instructions:
        if len(spec.sources) != 1 or len(spec.destinations) != 1:
            continue
        _, _, results = trace_instruction(spec)
        if len(results) != 1 or results[0].kind != "identity":
            continue
        if compute_params(spec):
            # The planner inserts moves itself, so there is no source constant to
            # bind an immediate from.
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction cannot take compute "
                f"params (α) — nothing supplies them"
            )
        roles, offset_of = param_roles(spec)
        loose = [i for i, r in roles.items() if r in ("stride", "layout")]
        if loose:
            # Residence params are pinned by unifying the maps of a value's *matched*
            # accesses; a move is inserted by the planner, so it takes part in no such
            # unification and nothing would supply them.
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction cannot take solvable "
                f"stride / ordering params {loose} — write the relayout it performs "
                f"explicitly"
            )
        if _alias_groups(offset_of):
            # A move is inserted between two *independently placed* locations, so it
            # has no way to honour "read and write at one address".
            raise AcceleratorDescriptionError(
                f"{spec.name}: a data-movement instruction cannot share an address "
                f"param between its source and destination"
            )
        moves.append(spec.name)
    return moves


@dataclass(eq=False)  # identity-based: distinct residences must never compare equal
class _Loc:
    """A *location*: one value's residence in one buffer, the unit of allocation.

    A value may hold several locations over its life (e.g. a ``bram`` copy and a
    ``vreg`` copy, or — after a spill — two ``vreg`` copies split around the spill
    gap). Each occupies ``size`` contiguous units at ``base`` and is read at the steps
    in ``uses`` (``last_use`` = the last), at which point its space is released;
    spilling is just ending one location and opening another.

    **``map`` is the residence itself** (see ``core.residence``): which address
    inside the location holds which logical element. A value may hold two
    locations in the *same* buffer differing only in that — one row-major, one
    channel-last — which is what makes a relayout a routing step rather than a special
    case. It is also what an instruction's access demands of an operand: a location is
    usable only if its map is the one that access describes.

    **Placement is a coordinate, not a number.** A buffer declared with more than one
    extent needs one component per axis, which is what an access pattern's
    multi-dimensional ``basis`` consumes. Allocation still packs a *single*
    axis — the outermost — so ``base`` stays one integer and the free list stays 1-D;
    the remaining components are 0. A value therefore occupies a whole band
    ``[base : base+size, 0 :, ...]`` rather than a sub-rectangle: packing rectangles is
    2-D bin packing, and the price of not doing it is unused columns, not wrong code."""

    value: object
    buffer: object
    size: int  # units along the allocated (outermost) axis
    map: tuple = ()  # the residence: (size, stride) per spanning dim
    base: int = -1  # that axis's coordinate; -1 until allocated
    last_use: int = -1
    uses: list = field(default_factory=list)  # step indices that read this location
    freed: bool = False

    @property
    def offset(self) -> tuple:
        """The placement coordinate, one component per address axis of the buffer."""
        return (self.base,) + (0,) * (self.buffer.address_rank - 1)


@dataclass
class _Move:
    name: str
    read: _Loc
    write: _Loc


@dataclass
class _Compute:
    name: str
    reads: list  # list[_Loc], in source-buffer order
    write: _Loc
    offset_of: dict  # access param -> [(buffer position, coordinate axis)]
    shape_params: dict  # access param -> solved size
    reusable: set  # source-operand indices whose slot the result may reuse in place
    alpha: list  # computational attributes (α), bound from the source's constants


def _alias_groups(offset_of: dict) -> list:
    """The must-alias constraints an instruction's access states: one entry per
    address param that is the basis of more than one buffer.

    Only axis 0 is constrained in practice — allocation packs that axis and leaves the
    rest at 0, so a shared param on any other axis is satisfied by construction."""
    return [
        (param, [pos for pos, _axis in refs])
        for param, refs in offset_of.items()
        if len(refs) > 1 and refs[0][1] == 0
    ]


# Prim tags whose output position i depends only on input position i. Derived from
# the registry categories, plus the two bespoke elementwise prims. An **allowlist**,
# so anything else — a reduction, a contraction, a permutation, a windowed op, or a
# prim added later — is conservatively excluded: guessing wrong here silently
# overwrites an operand that the result does not line up with.
_POSITION_PRESERVING = {"identity", "relu"} | {
    tag
    for tag, p in primitive.REGISTRY.items()
    if p.category
    in (
        primitive.UNARY,
        primitive.UNARY_ZP,
        primitive.BINARY,
        primitive.BINARY_SHIFT,
        primitive.COMPARE,
        primitive.SELECT,
        primitive.CAST,
    )
}


def _reusable_operands(instruction) -> set:
    """Source-operand indices whose slot the result may safely reuse in place.

    Reuse is safe for an operand iff its value reaches the result through only
    position-preserving ops: output position i then depends only on that operand's
    position i, so writing the result over the operand is fine. A matmul mixes
    positions, so any operand feeding a matmul is not reusable — but an *accumulator*
    read only by an element-wise add (``c + a @ b``) is, which lets a K-reduction
    chain collapse onto a single accumulator slot (the hardware's block buffer). For
    a purely element-wise op every operand is reusable; for a plain matmul none is.

    The functional oracle cannot check this — it reads all operands before writing any
    destination, so an overwrite it should have observed simply does not happen there.
    Hence the conservative allowlist rather than a denylist of known mixers."""
    _, _, results = trace_instruction(instruction.spec)
    if len(results) != 1:
        return set()
    n_src = len(instruction.spec.sources)
    safe: set = set()

    def walk(node, position_preserving: bool):
        if node.kind == "arg":
            if position_preserving and node.buffer_index < n_src:
                safe.add(node.buffer_index)
            return
        child_ok = position_preserving and node.kind in _POSITION_PRESERVING
        for a in node.args:
            walk(a, child_ok)

    walk(results[0], True)
    return safe


def _colocatable(m: Match) -> set:
    """Source-operand indices laid out exactly like the destination.

    Position-preserving semantics say output tensor position ``i`` depends only on
    operand position ``i``; writing the result over that operand is safe only if the
    two positions are also the same *address*. An instruction that relayouts while it
    computes (elementwise semantics, differing maps) breaks that, so residence has to
    agree before a slot is handed over."""
    spec = m.instruction.spec
    patterns, _, _ = trace_instruction(spec)
    n_src = len(spec.sources)
    dst = residence(access_map(patterns[n_src], m.shape_params))
    return {
        i
        for i in range(n_src)
        if residence(access_map(patterns[i], m.shape_params)) == dst
    }


# ==========================================================================#
# Tile expansion — lower one layer-level match to a run of tile instructions
# ==========================================================================#


class _ExpandRecorder:
    """Collects the instruction calls an ``@expand`` body issues.

    Same protocol as ``OracleProgram``: ``Instruction.__call__`` records into
    whatever ``isa._active_oracle`` holds."""

    def __init__(self, name: str):
        self.name = name
        self.emits: list[EmitRecord] = []

    def record_emit(self, name, addr, compute):
        self.emits.append(EmitRecord(name, list(addr), list(compute)))


def expand_emits(isa, spec, addr: list) -> list:
    """Run an instruction's ``@expand`` body on its solved address params and return
    the run of emits it issues.

    An instruction that carries ``@expand`` is *layer-level*: it matches (and is
    allocated) as one operation, then lowers to many. Because expansion happens after
    allocation, the body receives concrete numbers — allocated buffer offsets for the
    offset params, Stage-2-solved sizes for the shape params — so it can compute each
    tile's address arithmetically. Its ``@compute`` region stays the layer's
    semantics: the catalog states what the expansion must equal, and the oracle
    executes the expansion, so the two can be diffed."""
    if compute_params(spec):
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand and compute params (α) cannot be combined — the "
            f"expansion body receives address params only, so it has no way to pass "
            f"the bound immediate on to the tile instructions it issues"
        )
    if layout_params(spec):
        # The body *is* handed the ordering, but a sub-block of a layer laid out in
        # some ordering does not sit at a fixed stride from the layer's base -- the
        # translation is the tiling this frontend leaves to the mid-end. Refused
        # rather than trusted to a body nothing here can check.
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand and an ordering param cannot be combined — the "
            f"tiles the expansion issues address sub-blocks whose residence is not "
            f"the layer's map"
        )
    recorder = _ExpandRecorder(spec.name)
    prev = isa._active_oracle
    isa._active_oracle = recorder
    try:
        spec.expand_fn(*addr)
    finally:
        isa._active_oracle = prev
    if not recorder.emits:
        raise AcceleratorDescriptionError(
            f"{spec.name}: @expand issued no instructions"
        )
    return recorder.emits


def _placement_dims(shape, buf) -> tuple:
    """The extent a value of ``shape`` occupies in ``buf``'s coordinates.

    A slot-addressed buffer takes one number (how many slots the value fills); a
    multi-dimensional buffer takes the value's own dims, aligned to the buffer's rank
    modulo leading unit dims (the same rank alias ``_align_ranks`` looks through, so a
    torch-batched ``1x4x4`` places as a ``4x4`` block)."""
    if buf.address_rank == 1:
        return (max(prod(shape) // buf.slot_size, 1),)
    dims = list(shape)
    if len(dims) != buf.address_rank:
        dims = _strip_leading_units(dims)
    if len(dims) != buf.address_rank:
        raise AllocationError(
            f"'{buf.name}' is addressed by {buf.address_rank} indices but the value's "
            f"shape is {tuple(shape)} — a multi-dimensional buffer holds values of its "
            f"own rank"
        )
    if any(d > e for d, e in zip(dims[1:], buf.extents[1:])):
        raise AllocationError(
            f"'{buf.name}': a value of shape {tuple(dims)} does not fit the array's "
            f"{buf.extents} extent"
        )
    return tuple(dims)


def _loc_size(value, buf) -> int:
    """Units of the allocated (outermost) axis one location of ``value`` occupies."""
    return _placement_dims(_shape(value), buf)[0]


def _const_array(value) -> np.ndarray | None:
    """A source constant's data, shaped like its tensor type, or ``None`` if it is
    not a constant this frontend can read (see ``const_elements``)."""
    elems = const_elements(_canon(value))
    if elems is None:
        return None
    shape = _shape(_canon(value))
    arr = np.array(elems)
    if arr.size == 1:  # a splat prints as one element whatever its extent
        arr = np.full(prod(shape), elems[0])
    return arr.reshape(shape)


@dataclass(frozen=True)
class _Edge:
    """One usable data-movement step for one value size.

    ``relayout`` is the ``(read, write)`` residence pair when the move's two access
    patterns describe *different* maps, and ``None`` when they describe the same one.
    That distinction is the whole of what a move does to a layout:

    - **equal maps** — the address correspondence is the identity, so the move copies
      the region verbatim and carries **whatever** residence the value had. This is why
      a plain dma can spill a channel-last value and reload it unharmed.
    - **different maps** — the correspondence is a genuine permutation of addresses, so
      the move is usable only on a value laid out exactly as it *reads*, and it then
      lays it out exactly as it *writes*. Applied to any other residence it would
      produce a scrambling no instruction asked for, so it simply is not an edge there.
    """

    src: str
    dst: str
    name: str
    relayout: tuple | None

    def follow(self, res: tuple) -> tuple | None:
        """The residence a value laid out as ``res`` has after this move, or ``None``
        if the move does not apply to it."""
        if self.relayout is None:
            return res
        read, write = self.relayout
        return write if read == res else None


def _move_edges(isa, moves: list, size: int) -> list:
    """The data-movement edges available to a value of ``size`` elements.

    A move that does not *fit* the value is not an edge: ``_solve_move_params`` sizes
    each mover against the value, so a fixed-size relayout only ever appears for the
    values it can actually carry."""
    edges = []
    for name in moves:
        spec = isa._ops[name].spec
        try:
            params = _solve_move_params(spec, size)
        except CompileError:
            continue
        patterns, _, _ = trace_instruction(spec)
        read = residence(access_map(patterns[0], params))
        write = residence(access_map(patterns[1], params))
        edges.append(
            _Edge(
                spec.sources[0].name,
                spec.destinations[0].name,
                name,
                None if read == write else (read, write),
            )
        )
    return edges


def _reachable(edges: list, starts) -> list:
    """Every ``(buffer, residence)`` state a value can be moved into, nearest first."""
    seen = list(starts)
    queue = deque(seen)
    while queue:
        buf, res = queue.popleft()
        for edge in edges:
            carried = edge.follow(res) if edge.src == buf else None
            if carried is None or (edge.dst, carried) in seen:
                continue
            seen.append((edge.dst, carried))
            queue.append((edge.dst, carried))
    return seen


def _route(edges: list, starts, goal: tuple) -> list | None:
    """Shortest path from any of ``starts`` to ``goal`` over ``(buffer, residence)``
    states, as ``[(state, move name | None), ...]`` starting at the reached start, or
    ``None`` if unreachable. BFS = fewest hops.

    Routing over *states* rather than buffers is what makes a relayout something the
    planner can find on its own: a value that is in the right buffer but the wrong
    layout is simply a state one repacking edge away — including a repack from a buffer
    to itself, which as a plain buffer path would have been a zero-hop no-op."""
    prev: dict = {s: None for s in starts}
    queue = deque(prev)
    while queue:
        state = queue.popleft()
        if state == goal:
            path = []
            while state is not None:
                path.append((state, prev[state][1] if prev[state] else None))
                state = prev[state][0] if prev[state] else None
            return list(reversed(path))
        buf, res = state
        for edge in edges:
            if edge.src != buf:
                continue
            carried = edge.follow(res)
            nxt = (edge.dst, carried)
            if carried is None or nxt in prev:
                continue
            prev[nxt] = (state, edge.name)
            queue.append(nxt)
    return None


def plan(isa, selection: Selection) -> CompiledProgram:
    """Liveness-driven, buffer-aware allocation over *locations* (see ``_Loc``).

    1. *Schedule* — lower each match to a linear stream of moves + computes,
       inserting data movement (``bring_to``) whenever a value is not resident in
       the buffer an instruction needs it in **and laid out the way that instruction
       reads it**. Routing runs over ``(buffer, residence)`` states, so a repacking
       move is found the same way a copy is; each hop is a short-lived intermediate
       location (**P-C**). Program I/O lives in the global buffer, in the host ABI's
       layout at both ends.
    2. *Liveness* — def step + last-use step (and the full use list) per location.
    3. *Allocation* — best-fit free-list per buffer, releasing a location at its
       last use so slots are reused; a result coalesces in place onto a dying
       element-wise operand. On overflow a Belady victim (resident, not used at the
       overflow step, farthest next use) is **spilled** to the backing store and
       reloaded before its next use (**P-B**); inserting the spill grows the
       schedule, so liveness + allocation are re-run to a fixpoint."""
    io = _io_buffer(isa)
    moves = _movement_catalog(isa)
    block = selection.func.regions[0].blocks[0]
    func_args = list(block.arguments)

    # --- pass 1: schedule (moves + computes) over locations ------------------
    loc: dict = {}  # value -> {(buffer name, residence): _Loc}
    steps: list = []
    constants: list = []  # (_Loc in io, ndarray) for each source constant used as data
    edges_for: dict = {}  # value element count -> the moves usable at that size

    def make_loc(value, buf, res) -> _Loc:
        l = _Loc(value, buf, _loc_size(value, buf), res)
        loc.setdefault(value, {})[(buf.name, res)] = l
        return l

    def route_move(cur: _Loc, path: list, sink: list) -> _Loc:
        """Append a move per hop along ``path`` (states from ``_route``, starting at
        ``cur``'s own state); return the final location."""
        for (name, res), move in path[1:]:
            dst = make_loc(cur.value, isa.buffers[name], res)
            sink.append(_Move(move, cur, dst))
            cur = dst
        return cur

    def edges(value) -> list:
        size = prod(_shape(value))
        if size not in edges_for:
            edges_for[size] = _move_edges(isa, moves, size)
        return edges_for[size]

    def abi(value) -> tuple:
        """The host ABI's residence for a value in the I/O buffer."""
        return _dense_map(_static_shape(value), io)

    def wants(pattern, params) -> tuple:
        """The residence one access of a match describes."""
        return residence(access_map(pattern, params))

    def bring_to(value, target, want, who) -> _Loc:
        """A location of ``value`` in ``target`` laid out as ``want``.

        A value that is in the right buffer but the wrong layout is not resident: it is
        one repacking edge away, and finding that edge is the same BFS as finding a
        route between buffers. Which is the point — a relayout is data movement, so the
        planner inserts it exactly the way it inserts any other move."""
        here = loc.get(value, {})
        if (target.name, want) in here:
            return here[(target.name, want)]
        if not here:
            # A constant used as a *data* operand is program data, not part of any
            # instruction: ACT Def 3.8 puts it in the ASM alongside the inputs
            # (`concat(bflat(X), bflat(const))`), which is exactly a location in the
            # I/O buffer that `CompiledProgram.__call__` fills before the run.
            data = _const_array(value)
            if data is None:
                raise CompileError(
                    f"a value of shape {_shape(value)} has no location to move from: "
                    f"it is neither a program input, nor the result of a matched "
                    f"instruction, nor a readable constant (a constant whose data is "
                    f"a `dialect_resource` blob cannot be read — pass it as a function "
                    f"argument instead)"
                )
            constants.append((make_loc(value, io, abi(value)), data))
            here = loc[value]
            if (target.name, want) in here:
                return here[(target.name, want)]
        path = _route(edges(value), list(here), (target.name, want))
        if path is None:
            raise _unroutable(value, here, target, want, who)
        return route_move(here[path[0][0]], path, steps)

    def _unroutable(value, here, target, want, who) -> CompileError:
        """Say *why* a value cannot get where it is needed: an unreachable buffer, or a
        reachable one in the wrong layout with nothing that repacks it."""
        avail = edges(value)
        anywhere = {
            res for buf, res in _reachable(avail, list(here)) if buf == target.name
        }
        where = ", ".join(f"'{b}' as {show_map(r)}" for b, r in here)
        if not anywhere:
            return AllocationError(
                f"{who}: no data-movement route from {sorted({b for b, _r in here})} "
                f"to '{target.name}'"
            )
        return LayoutError(
            f"{who}: needs a value of shape {_shape(value)} in '{target.name}' laid "
            f"out as {show_map(want)}, but it lives in {where} and no data movement "
            f"relayouts between them — declare the relayout as a move, or have the "
            f"two ends agree on a layout"
        )

    input_locs = [make_loc(a, io, abi(a)) for a in func_args]

    for m in selection.matches:
        spec = m.instruction.spec
        patterns, _, _ = trace_instruction(spec)
        reads = [
            bring_to(
                _canon(v),
                buf,
                wants(p, m.shape_params),
                f"{m.instruction.name} operand {i}",
            )
            for i, (v, buf, p) in enumerate(
                zip(m.operand_values, spec.sources, patterns)
            )
        ]
        write = make_loc(
            _canon(m.result_value),
            spec.destinations[0],
            wants(patterns[len(spec.sources)], m.shape_params),
        )
        _, offset_of = param_roles(spec)
        steps.append(
            _Compute(
                m.instruction.name,
                reads,
                write,
                offset_of,
                m.shape_params,
                _reusable_operands(m.instruction) & _colocatable(m),
                [m.alpha[i] for i in range(len(compute_params(spec)))],
            )
        )

    terminator = list(block.operations)[-1]
    output_vals = list(terminator.operands)
    # The host reads a result back densely, so an output must *arrive* in the ABI's
    # layout: a program that computed it repacked needs the relayout inserted here.
    output_locs = [
        bring_to(_canon(v), io, abi(_canon(v)), f"result #{i}")
        for i, v in enumerate(output_vals)
    ]

    # --- helpers shared by the liveness + allocation fixpoint ----------------
    def reads_of(st) -> list:
        return [st.read] if isinstance(st, _Move) else st.reads

    def all_locs() -> list:
        seeds = input_locs + [l for l, _data in constants]
        seen, out = set(map(id, seeds)), list(seeds)
        for st in steps:
            for l in reads_of(st) + [st.write]:
                if id(l) not in seen:
                    seen.add(id(l))
                    out.append(l)
        return out

    def liveness():
        final = len(steps)  # virtual step: the terminator reads the outputs
        for l in all_locs():
            l.last_use, l.uses, l.base, l.freed = -1, [], -1, False
        for i, st in enumerate(steps):
            for r in reads_of(st):
                r.last_use = i
                r.uses.append(i)
        for l in output_locs:
            l.last_use = final
            l.uses.append(final)

    def allocate():
        """Assign offsets in one walk; on overflow return ``(victim, step)`` to
        spill, else ``None`` (offsets are final). Belady victim selection."""
        free = {name: [(0, buf.capacity)] for name, buf in isa.buffers.items()}
        live = {name: [] for name in isa.buffers}  # placed, not-yet-freed locations

        def release(l):
            runs = sorted(free[l.buffer.name] + [(l.base, l.size)])
            merged = [runs[0]]
            for off, sz in runs[1:]:
                poff, psz = merged[-1]
                if poff + psz == off:
                    merged[-1] = (poff, psz + sz)
                else:
                    merged.append((off, sz))
            free[l.buffer.name] = merged
            live[l.buffer.name].remove(l)
            l.freed = True

        def best_fit(buf, size) -> int | None:
            runs = free[buf.name]
            pick = min(
                (i for i, (_o, sz) in enumerate(runs) if sz >= size),
                key=lambda i: runs[i][1],
                default=-1,
            )
            if pick < 0:
                return None
            off, sz = runs.pop(pick)
            if sz > size:
                runs.append((off + size, sz - size))
            return off

        def place(l) -> bool:
            l.base = best_fit(l.buffer, l.size)
            if l.base is None:
                return False
            live[l.buffer.name].append(l)
            return True

        def forced_alias(st, reads, write, t):
            """The operand location this step's write **must** be placed on top of,
            or ``None`` if its access forces nothing.

            An address param used as the basis of two buffers is not a hint: the ISA is
            saying those operands are at one address (QKV's ``softmax`` reads and writes
            one ``addr``). Allocation therefore has to guarantee it, rather than leave
            it to the opportunistic reuse below — which does nothing when the operand
            outlives the instruction."""
            if not isinstance(st, _Compute):
                return None
            n = len(reads)
            target = None
            for param, positions in _alias_groups(st.offset_of):
                operands = [p for p in positions if p < n]
                for p in operands[1:]:
                    if reads[p] is not reads[operands[0]]:
                        raise AllocationError(
                            f"{st.name}: address param p{param} puts operands "
                            f"{operands[0]} and {p} at one address, but they are bound "
                            f"to different values"
                        )
                if n in positions and operands:
                    target = reads[operands[0]]
            if target is None:
                return None
            if target.buffer is not write.buffer or target.size != write.size:
                raise AllocationError(
                    f"{st.name}: writes its result over the operand it reads, but the "
                    f"two do not occupy the same space "
                    f"({target.buffer.name}[{target.size}] vs "
                    f"{write.buffer.name}[{write.size}])"
                )
            if target.last_use != t or target.freed:
                raise AllocationError(
                    f"{st.name}: writes its result over the operand it reads, but that "
                    f"operand is read again at step {min(u for u in target.uses if u > t)}"
                    f" — the in-place write would destroy it. Copy it first, or use an "
                    f"out-of-place instruction."
                )
            return target

        # Constants are resident from the start, exactly like inputs: nothing in the
        # program writes them, so the allocator must reserve their space up front.
        for l in input_locs + [l for l, _data in constants]:
            if not place(l):
                raise AllocationError(f"backing store '{io.name}' overflow on inputs")

        for t, st in enumerate(steps):
            reads, write = reads_of(st), st.write
            reused = forced_alias(st, reads, write, t)
            if reused is None and isinstance(st, _Compute) and st.reusable:
                reused = next(
                    (
                        reads[i]
                        for i in st.reusable
                        if reads[i].buffer is write.buffer
                        and reads[i].last_use == t
                        and reads[i].size == write.size
                        and not reads[i].freed
                    ),
                    None,
                )
            if reused is not None:
                write.base = reused.base  # coalesce: hand the slot to the result
                live[write.buffer.name].remove(reused)
                live[write.buffer.name].append(write)
                reused.freed = True
            elif not place(write):
                return _pick_victim(write.buffer, t, live[write.buffer.name]), t
            for r in reads:
                if r.last_use == t and not r.freed:
                    release(r)
        return None

    def _pick_victim(buf, t, resident):
        # Spill the resident location whose *next* use is farthest away (Belady),
        # excluding anything this very step still reads.
        candidates = [l for l in resident if t not in l.uses]
        if not candidates:
            raise AllocationError(
                f"buffer '{buf.name}' overflow at step {t}: no spillable location "
                f"(an instruction needs more slots than '{buf.name}' has)"
            )
        return max(candidates, key=lambda l: min(s for s in l.uses if s > t))

    def spill(victim, t):
        """Evict ``victim`` from its buffer to the backing store over [t, next-use):
        store it down before step t, reload it back before its next use, and repoint
        the later uses onto the reloaded copy. Grows ``steps`` (re-run liveness)."""
        u = min(s for s in victim.uses if s > t)
        if victim.buffer is io:
            raise AllocationError(f"backing store '{io.name}' overflow")
        # A spill must come back **as it left** — the later uses were routed to this
        # residence — but only the *round trip* has to preserve it, not each leg: a
        # machine whose only path to the backing store is a repacking dma spills fine
        # as long as the reload repacks back. So the store residence is searched for,
        # nearest first, rather than assumed to be the victim's own.
        home = (victim.buffer.name, victim.map)
        avail = edges(victim.value)
        down = up = None
        for state in _reachable(avail, [home]):
            if state[0] != io.name:
                continue
            down, up = _route(avail, [home], state), _route(avail, [state], home)
            if down and up:
                break
            down = up = None
        if not (down and up):
            raise AllocationError(
                f"cannot spill from '{victim.buffer.name}': no round trip to "
                f"'{io.name}' returns the value to {show_map(victim.map)}"
            )
        store_steps: list = []
        spilled = route_move(victim, down, store_steps)
        reload_steps: list = []
        reloaded = route_move(spilled, up, reload_steps)
        for i, st in enumerate(steps):
            if i >= u:
                if isinstance(st, _Move):
                    if st.read is victim:
                        st.read = reloaded
                else:
                    st.reads = [reloaded if r is victim else r for r in st.reads]
        steps[u:u] = reload_steps  # reload before the next use ...
        steps[t:t] = store_steps  # ... and store before the overflow step (t < u)

    # An instruction's operands are all live at once and none can be spilled (they
    # are in use), so a buffer that cannot even hold one instruction's operands is
    # infeasible — reject upfront. This also guarantees the spill loop terminates:
    # every remaining overflow then has a non-operand victim to evict, so each spill
    # resolves the earliest overflow and pushes the frontier strictly later.
    for st in steps:
        if isinstance(st, _Compute):
            need: dict = {}
            for r in dict.fromkeys(st.reads):  # distinct locations (handles a*a)
                need[r.buffer.name] = need.get(r.buffer.name, 0) + r.size
            for name, n in need.items():
                if n > isa.buffers[name].capacity:
                    raise AllocationError(
                        f"{st.name}: operands need {n} unit(s) of '{name}' but it "
                        f"holds only {isa.buffers[name].capacity} "
                        f"(capacity too small to spill into)"
                    )

    # --- passes 2+3: liveness + allocation, iterated to a no-spill fixpoint ---
    while True:
        liveness()
        outcome = allocate()
        if outcome is None:
            break
        spill(*outcome)

    # --- emit with concrete offsets ------------------------------------------
    def _addr(offset_of, locs, shape_params, n_addr) -> list:
        """Fill an instruction's address params. An offset param names one coordinate
        *component* of one operand — ``(buffer position, axis)`` — so a multi-index
        access takes several, all read off that operand's placement. A param naming
        that component in several operands has been forced to one address by
        ``allocate``, so any of its references gives the same number."""

        def component(i):
            pos, axis = offset_of[i][0]
            return locs[pos].offset[axis]

        return [
            component(i) if i in offset_of else shape_params[i] for i in range(n_addr)
        ]

    emits: list[EmitRecord] = []
    for st in steps:
        if isinstance(st, _Move):
            # A move fills its access params like a compute: offset params take the
            # source/destination placements; shape params are solved from the moved
            # value's element count (prod(visible) == value size), since the move was
            # inserted in Stage 3 and never went through Stage-2 solve.
            spec = isa._ops[st.name].spec
            _, offset_buffer = param_roles(spec)
            shape_params = _solve_move_params(spec, prod(_shape(st.read.value)))
            addr = _addr(
                offset_buffer,
                [st.read, st.write],
                shape_params,
                arity(spec.access_fn),
            )
            emits.append(EmitRecord(st.name, addr, []))
        else:
            spec = isa._ops[st.name].spec
            addr = _addr(
                st.offset_of,
                st.reads + [st.write],
                st.shape_params,
                arity(spec.access_fn),
            )
            if spec.expand_fn is None:
                emits.append(EmitRecord(st.name, addr, st.alpha))
            else:
                emits.extend(expand_emits(isa, spec, addr))

    inputs = [(l.offset, _shape(l.value)) for l in input_locs]
    outputs = [
        (l.offset, _shape(v), f"out{i}")
        for i, (l, v) in enumerate(zip(output_locs, output_vals))
    ]
    return CompiledProgram(
        isa, io, emits, inputs, outputs, [(l.offset, d) for l, d in constants]
    )


# ==========================================================================#
# Driver
# ==========================================================================#


def compile_program(source: str, isa) -> CompiledProgram:
    """Compile a source program onto ``isa``.

    The source program is a TOSA-dialect MLIR module given as *text* — we generate
    none ourselves; the caller hands us a module string (e.g. from torch_mlir's
    TOSA backend) and we ``Module.parse`` it here. The returned ``CompiledProgram``
    holds only plain data (no IR handles), so the parse context can be dropped."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        normalize_source(module)
        catalog = Catalog(isa)
        selection = match_program(catalog, module)
        solve(selection)
        solve_layouts(isa, selection)
        return plan(isa, selection)
