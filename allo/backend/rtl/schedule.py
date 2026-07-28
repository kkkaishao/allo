# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDC scheduling flow: the ``run_schedule`` driver + the schedule-result model.

``run_schedule(top, module, ...)`` runs the loop/reduction normalizations,
``sdc-scheduling`` and ``convert-schedule-to-dcp`` against an operator library,
reifying the schedule into the module as ``allo.dcp.*`` ops. It returns a
:class:`ScheduleResult` read structurally off those ops: regions from
``dcp.pipeline`` / ``dcp.sequential``, per-op start times from their ``start``
field, operator kinds and latencies from the referenced ``dcp.operator``.
``verify_schedule`` checks the register-depth invariant.

This is the low-level driver; :class:`~allo.backend.rtl.RTL` is the user-facing
entry point (``export("rtl", ...).schedule()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..base import run_pipeline
from ..._mlir.ir import (
    IntegerAttr,
    FloatAttr,
    StringAttr,
    FlatSymbolRefAttr,
    BlockArgument,
    OpResult,
)
from ..._mlir.dialects.allo import run_sdc_scheduling

RTL_PREPARE_PIPELINE = """
builtin.module(
grid-mapping,
fold-constant-calls,
canonicalize,
cse,
materialize-topology,
canonicalize,
cse,
convert-allo-to-func,
func.func(convert-linalg-to-affine-loops),legalize-arith,canonicalize,cse,
outline-loose-processes)
"""

# --- schedule result data model --------------------------------------------


class RegionKind(str, Enum):
    """The scheduling regime of a region. A ``str`` mixin keeps
    ``region.kind == "cyclic"`` working alongside ``region.kind is
    RegionKind.CYCLIC``."""

    CYCLIC = "cyclic"  # a pipelined loop (dcp.pipeline)
    ACYCLIC = "acyclic"  # a straight-line span (dcp.sequential)
    GUARD = "guard"  # a control select (dcp.select); carries no compute itself


@dataclass(frozen=True)
class ScheduledOp:
    """One scheduled operation inside a region."""

    kind: str  # operator mnemonic (addi/mulf/load/store/...); an open set, so str
    t: int  # start cycle within the region
    impl: str | None = None  # realization (IP module name / native keyword)
    z: float | None = None  # SDC z-slack, when carried


@dataclass(frozen=True)
class RegionSchedule:
    """One scheduling region (a dcp.pipeline / dcp.sequential / dcp.select)."""

    kind: RegionKind
    order: int  # program order among the func's regions
    depth: int  # nesting depth among dcp regions (0 = outermost)
    container: bool  # nests another region (a loop / guard wrapper)
    ops: list[ScheduledOp] = field(default_factory=list)
    ii: int | None = None  # cyclic only; None for a dynamic-trip sequential wrapper
    trip: int | None = None  # constant trip count, when known
    length: int | None = None  # single-iteration cycle span
    latency: int | None = None  # region latency (cycles)
    latency_is_bound: bool = False  # latency is an upper bound, not exact
    conditional: bool = False  # while-pipeline (dcp.condition) or a guard

    @property
    def is_wrapper(self) -> bool:
        """A container region carrying no compute of its own (a residual outer
        loop around leaf regions) -- a derived nesting node, not a scheduling
        decision."""
        return self.container and not self.ops

    @property
    def is_leaf(self) -> bool:
        return not self.container

    def op(self, kind: str) -> ScheduledOp:
        """The first op of the given kind (raises ``StopIteration`` if none)."""
        return next(o for o in self.ops if o.kind == kind)

    def has(self, kind: str) -> bool:
        return any(o.kind == kind for o in self.ops)

    def last_t(self) -> int:
        """The latest start cycle among this region's ops."""
        return max(o.t for o in self.ops)


@dataclass(frozen=True)
class FuncSchedule:
    """The schedule of one ``func.func``."""

    name: str
    regions: list[RegionSchedule] = field(default_factory=list)
    latency: int | None = None  # whole-func latency (cycles), when static
    latency_is_bound: bool = False

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        """This func's cyclic regions; pure sequential wrappers excluded unless
        ``wrappers=True``."""
        return [
            r
            for r in self.regions
            if r.kind is RegionKind.CYCLIC and (wrappers or not r.is_wrapper)
        ]


@dataclass(frozen=True)
class ScheduleResult:
    """The whole-module schedule result: the schedule of every ``func.func``."""

    funcs: list[FuncSchedule] = field(default_factory=list)

    def func(self, suffix: str) -> FuncSchedule:
        """The sub-function whose name ends with ``suffix`` (kernels compose by
        calling sub-kernels, so results carry ``top.sub`` funcs)."""
        return next(f for f in self.funcs if f.name.endswith(suffix))

    def regions(
        self, kind: RegionKind | None = None, *, wrappers: bool = False
    ) -> list[RegionSchedule]:
        """Regions across all funcs, optionally filtered by kind. Pure
        sequential wrappers are excluded by default (they carry a derived II, not
        a scheduling decision); pass ``wrappers=True`` for the full nesting
        tree."""
        return [
            r
            for f in self.funcs
            for r in f.regions
            if (kind is None or r.kind is kind) and (wrappers or not r.is_wrapper)
        ]

    def cyclic(self, *, wrappers: bool = False) -> list[RegionSchedule]:
        return self.regions(RegionKind.CYCLIC, wrappers=wrappers)


# --- reading dcp ops -------------------------------------------------------

_OPERATOR = "allo.dcp.operator"
_PIPELINE = "allo.dcp.pipeline"
_SEQUENTIAL = "allo.dcp.sequential"
_SELECT = "allo.dcp.select"
_COMPUTE = "allo.dcp.compute"
_LOAD = "allo.dcp.load"
_STORE = "allo.dcp.store"
_CONDITION = "allo.dcp.condition"
_TIMED = (_COMPUTE, _LOAD, _STORE)
_PRODUCER = (_COMPUTE, _LOAD)


# --- small attribute / traversal helpers -----------------------------------


def _module_op(module):
    return module.operation if hasattr(module, "operation") else module


def _body(op):
    """The single entry block of ``op``'s single-region body."""
    return op.regions[0].blocks[0]


def _int(op, name):
    return IntegerAttr(op.attributes[name]).value


def _opt_int(op, name):
    return _int(op, name) if name in op.attributes else None


def _str(op, name):
    return StringAttr(op.attributes[name]).value


def _walk(op):
    """Yield ``op`` and every nested operation in pre-order (program order)."""
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


_REGION = (_PIPELINE, _SEQUENTIAL, _SELECT)


def _region_depth(op):
    """Nesting depth of a dcp region among dcp regions (0 = outermost)."""
    depth = 0
    parent = op.parent
    while parent is not None:
        if parent.name in _REGION:
            depth += 1
        parent = parent.parent
    return depth


def _has_nested_region(op):
    """True iff a dcp region is nested inside ``op`` -- i.e. ``op`` is a wrapper /
    container (a loop or a guard), not a leaf scheduling region."""
    for i, child in enumerate(_walk(op)):
        if i and child.name in _REGION:
            return True
    return False


def _kind_map(mod_op):
    """symbol -> kind for every module-level ``dcp.operator``."""
    return {
        _str(o.operation, "sym_name"): _str(o.operation, "kind")
        for o in _body(mod_op).operations
        if o.operation.name == _OPERATOR
    }


def _latency_map(mod_op):
    """symbol -> latency for every module-level ``dcp.operator``."""
    return {
        _str(o.operation, "sym_name"): _int(o.operation, "latency")
        for o in _body(mod_op).operations
        if o.operation.name == _OPERATOR
    }


def _impl_map(mod_op):
    """symbol -> RTL module name for every ``dcp.operator``; the ``sym_name`` IS
    the module name (an IP realization)."""
    return {
        _str(o.operation, "sym_name"): _str(o.operation, "sym_name")
        for o in _body(mod_op).operations
        if o.operation.name == _OPERATOR
    }


# IP operators carry their abstract kind (`add`/`div`/...); reconstruct the
# arith mnemonic (IP compute is always floating-point) so a schedule op reads as
# the source op it came from, uniform with combinational and memory ops.
_IP_MNEMONIC = {
    "add": "addf",
    "sub": "subf",
    "mul": "mulf",
    "div": "divf",
    "rem": "remf",
    "cmp": "cmpf",
    "neg": "negf",
}


def _comb_kw(op):
    """The CombOpKind mnemonic of a combinational ``dcp.compute`` (e.g. ``addi``),
    parsed from its ``comb_kind`` enum attribute."""
    s = str(op.attributes["comb_kind"])
    inner = s[s.index("<") + 1 : s.rindex(">")] if "<" in s else s
    return inner.split()[-1]


def _op_kind(op, kinds):
    """The operator kind of a scheduled op, as an arith/affine mnemonic: an IP
    compute's reconstructed mnemonic (from its ``dcp.operator`` kind), a
    combinational compute's ``comb_kind``, a load/store's mnemonic, else the op's
    dialect-stripped name."""
    if op.name == _COMPUTE:
        if "op_type" in op.attributes:
            k = kinds.get(FlatSymbolRefAttr(op.attributes["op_type"]).value, "")
            return _IP_MNEMONIC.get(k, k)
        return _comb_kw(op)
    if op.name in (_LOAD, _STORE):
        return op.name.rsplit(".", 1)[1]
    return op.name.split(".", 1)[1]


# --- export ----------------------------------------------------------------


def _region_ops(body, kinds, impls) -> list[ScheduledOp]:
    ops = []
    for opview in body.operations:
        op = opview.operation
        if "start" not in op.attributes:
            continue
        impl = None
        if op.name == _COMPUTE and "op_type" in op.attributes:
            sym = FlatSymbolRefAttr(op.attributes["op_type"]).value
            impl = impls.get(sym)
        z = FloatAttr(op.attributes["z"]).value if "z" in op.attributes else None
        ops.append(
            ScheduledOp(kind=_op_kind(op, kinds), t=_int(op, "start"), impl=impl, z=z)
        )
    return ops


def _region_schedule(region_op, order, kinds, impls) -> RegionSchedule | None:
    """Build a :class:`RegionSchedule` from a dcp region op, or ``None`` if the
    op is not a region."""
    if region_op.name == _PIPELINE:
        kind = RegionKind.CYCLIC
        # `ii` is absent for a data-dependent sequential wrapper (an enclosed
        # dynamic-trip loop); a pipelined region always has one.
        ii = _opt_int(region_op, "ii")
        # A while pipeline terminates with dcp.condition (its leading i1 is the
        # continue condition); a counted one with dcp.uncondition.
        term = list(_body(region_op).operations)[-1].operation
        conditional = term.name == _CONDITION
        trip = _opt_int(region_op, "trip")
    elif region_op.name == _SEQUENTIAL:
        kind, ii, conditional, trip = RegionKind.ACYCLIC, None, False, None
    elif region_op.name == _SELECT:
        # A control guard: selects the active data path. It carries no compute of
        # its own (its branch children are reported in turn).
        kind, ii, conditional, trip = RegionKind.GUARD, None, True, None
    else:
        return None
    latency = _opt_int(region_op, "latency")
    return RegionSchedule(
        kind=kind,
        order=order,
        depth=_region_depth(region_op),
        container=_has_nested_region(region_op),
        ops=_region_ops(_body(region_op), kinds, impls),
        ii=ii,
        trip=trip,
        length=_opt_int(region_op, "length"),
        latency=latency,
        latency_is_bound=latency is not None
        and "latency_bound" in region_op.attributes,
        conditional=conditional,
    )


def export_schedule_result(module) -> ScheduleResult:
    """Read a scheduled module's ``allo.dcp.*`` ops into a
    :class:`ScheduleResult`: per-func regions with per-op start times keyed by
    operator kind, plus per-region / whole-kernel latency."""
    mod_op = _module_op(module)
    kinds = _kind_map(mod_op)
    impls = _impl_map(mod_op)

    funcs = []
    for fn in _body(mod_op).operations:
        fn = fn.operation
        if fn.name != "func.func":
            continue
        regions = []
        for region_op in _walk(fn):
            r = _region_schedule(region_op, len(regions), kinds, impls)
            if r is not None:
                regions.append(r)
        latency = _opt_int(fn, "dcp.latency")
        funcs.append(
            FuncSchedule(
                name=_str(fn, "sym_name"),
                regions=regions,
                latency=latency,
                latency_is_bound=latency is not None
                and "dcp.latency_bound" in fn.attributes,
            )
        )

    return ScheduleResult(funcs=funcs)


# --- verify ----------------------------------------------------------------


def _in(value, values):
    return any(value == v for v in values)


def _trace_iter_arg(carried, args, iter_arg_index):
    """Trace a pipeline iter-arg (0-based) back to the op defining its next
    value, counting one loop-carried distance per iter_arg-to-iter_arg shift.
    ``carried`` is the terminator's loop-carried next-values (the condition of a
    while pipeline is already stripped)."""
    v = carried[iter_arg_index]
    distance = 0
    seen = set()
    while isinstance(v, BlockArgument):
        n = v.arg_number
        if not _in(v, args) or n == 0 or n in seen:
            return None, 0
        seen.add(n)
        distance += 1
        v = carried[n - 1]
    if isinstance(v, OpResult):
        return v.owner, distance + 1
    return None, 0


def _check_region(region_op, body, ii, latencies, out):
    args = list(body.arguments)
    ops = [o.operation for o in body.operations]
    # The loop-carried next-values: dcp.uncondition carries them directly, while
    # dcp.condition's first operand is the continue condition (stripped here).
    carried = list(ops[-1].operands)
    if ops[-1].name == _CONDITION:
        carried = carried[1:]
    for use in ops:
        if use.name not in _TIMED:
            continue
        t_use = _int(use, "start")
        for v in use.operands:
            if isinstance(v, BlockArgument):
                # arg 0 is the pipeline counter (available at cycle 0); an arg of
                # an enclosing block is a cross-region value (no register).
                if not _in(v, args) or v.arg_number == 0:
                    continue
                defop, distance = _trace_iter_arg(carried, args, v.arg_number - 1)
            else:
                defop, distance = v.owner, 0
            # Only a same-region compute/load producer forms a timed edge.
            if defop is None or not _in(defop, ops) or defop.name not in _PRODUCER:
                continue
            sym = FlatSymbolRefAttr(defop.attributes["op_type"]).value
            latency = latencies.get(sym, 0)
            depth = distance * ii + (t_use - _int(defop, "start")) - latency
            if depth < 0:
                out.append(
                    f"schedule violates the register-depth invariant "
                    f"(depth {depth} < 0: II={ii}, distance={distance}, "
                    f"t_use={t_use}, t_def={_int(defop, 'start')}, "
                    f"latency={latency})"
                )


def verify_schedule(module) -> list[str]:
    """Check the schedule result of a materialized module (its ``allo.dcp.*``
    ops) against the register-depth invariant

        depth = d*II + (t_use - t_def) - latency(def) >= 0

    where ``d`` is the loop-carried distance (0 for an intra-iteration edge, >=1
    for a recurrence recovered by tracing iter-args). Returns the list of
    violation messages; an empty list means the schedule is consistent."""
    mod_op = _module_op(module)
    latencies = _latency_map(mod_op)
    out: list[str] = []
    for op in _walk(mod_op):
        if op.name == _PIPELINE:
            # A sequential wrapper has no static ii and carries no timed ops of
            # its own -- nothing to check (its children are checked in turn).
            ii = _opt_int(op, "ii")
            if ii is not None:
                _check_region(op, _body(op), ii, latencies, out)
        elif op.name == _SEQUENTIAL:
            _check_region(op, _body(op), 1, latencies, out)
    return out


# --- driver ----------------------------------------------------------------


def run_schedule(
    top,
    module,
    *,
    cycle_time=None,
    float_reassoc=True,
    accumulators=0,
    perfectize=False,
    unroll_under_pipeline=True,
    scalarize_threshold=16,
    scheduler="heuristic",
) -> ScheduleResult:
    """Schedule ``top`` and return the :class:`ScheduleResult`; ``module`` is
    rewritten in place, left holding the ``allo.dcp.*`` ops the schedule reifies
    into. Operator/device timing is read from the ``dcp.device`` / ``dcp.operator``
    ops injected into ``module`` before this call.

    Args:
        top: the name of the function to schedule.
        module: the MLIR module holding it.
        cycle_time: target clock period (ns); ``None`` falls back to 5.0.
        prepare: run the HLS preparation pipeline first (``False`` if the module
            is already lowered to affine form).
        float_reassoc: rebalance float reduction chains into logarithmic trees.
        accumulators: rotate float reductions across this many accumulators (0 =
            off).
        perfectize: sink an imperfect nest's prologue/epilogue into the inner
            loop under a guard, fusing it into one pipeline.
        unroll_under_pipeline: fully unroll the loops nested inside a pipelined
            loop, so the nest pipelines at one II.
        scalarize_threshold: scalarize memory accesses to arrays with this many or
            fewer elements, so they are kept in registers rather than a memory.
            Set to 0 to disable.
        scheduler: the solver that settles the resource half of each problem.
            ``"heuristic"`` is the SDC simplex plus greedy placement;
            ``"exact"`` is CP-SAT, available only in a build with OR-Tools.
    """
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    reassoc = (
        "reassociate-reductions{float-reassoc="
        f"{'true' if float_reassoc else 'false'}}}"
    )
    rotate = f"rotate-reductions{{accumulators={int(accumulators)}}}"
    loops = (
        "loop-canonicalization{"
        f"unroll-under-pipeline={'true' if unroll_under_pipeline else 'false'} "
        f"perfectize={'true' if perfectize else 'false'}}}"
    )
    part = f"propagate-partition{{top={top}}}"
    scalarize = f"scalarize-memory{{max-elements={scalarize_threshold}}}"
    pipeline = (
        f"builtin.module(func.func(raise-counted-while,elide-dead-init,{loops},"
        f"canonicalize,fold-if-statements,cse,{scalarize},"
        f"{reassoc},{rotate}),"
        f"{part},func.func(assign-banks))"
    )
    run_pipeline(module, pipeline)
    ret = run_sdc_scheduling(module, top, cycle_time or 5.0, scheduler)
    if not ret:
        raise RuntimeError(
            f"Scheduling step failed for {top}. Please check the log for details."
        )
    return export_schedule_result(module)
