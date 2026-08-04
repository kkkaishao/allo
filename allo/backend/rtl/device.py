# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device model for the RTL backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ...lang import f32, f64, bf16, i32, bool as _bool
from ...lang.ip import ip, IP, OperatorType
from .sim.ip_models import OpDesc, Ty


class MemoryKind(Enum):
    """A storage primitive. The value is the name the scheduler's
    ``MemoryImplEnum`` uses; keep them in sync."""

    REG = "register"
    LUTRAM = "lutram"
    BRAM = "bram"
    URAM = "uram"
    FIFO = "fifo"


@dataclass
class MemoryTiming:
    kind: MemoryKind
    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float


@dataclass(frozen=True)
class Resource:
    """A device resource: a counter with a capacity, and nothing else.

    The vocabulary is the DEVICE's, so a part with different primitives declares
    different names and the compiler, which only adds and multiplies these, does
    not care. ``capacity`` is a price input rather than a constraint: regions are
    scheduled independently, so a whole-device budget is not a quantity any one
    solve can enforce.
    """

    name: str
    capacity: int


@dataclass(frozen=True)
class Cost:
    """What one realization spends of one resource, as a function of ONE of that
    realization's parameters (an operand width, a multiplexer's fan-in).

    Build these through :func:`Const` and friends rather than directly. The
    SHAPE comes from hardware structure and only the coefficients are measured;
    a shape that is not structural belongs in a :func:`Table`.
    """

    form: str
    coeffs: tuple[float, ...]

    def _mlir(self) -> str:
        body = ", ".join(repr(float(c)) for c in self.coeffs)
        return f"#allo.cost<{self.form}, [{body}]>"


def Const(value: float) -> Cost:
    """A fixed amount, whatever the parameter."""
    return Cost("const", (float(value),))


def Linear(coeff: float, base: float = 0.0) -> Cost:
    """``base + coeff * p``."""
    return Cost("linear", (float(base), float(coeff)))


def Quadratic(coeff: float) -> Cost:
    """``coeff * p * p``, the shape of a divider."""
    return Cost("quadratic", (float(coeff),))


def Step(threshold: float, below_coeff: float, above: float) -> Cost:
    """``p < threshold ? below_coeff * p : above``: a shift-register cliff, where
    a chain past the threshold stops being flip-flops and stops growing."""
    return Cost("step", (float(threshold), float(below_coeff), float(above)))


def Table(points: dict[int, float]) -> Cost:
    """Measured point by point. A parameter between two points takes the lower
    one's value, and one below the first takes the first's."""
    if not points:
        raise ValueError("a cost table needs at least one point")
    flat: list[float] = []
    for p in sorted(points):
        flat += [float(p), float(points[p])]
    return Cost("table", tuple(flat))


class CombKind(Enum):
    """A combinational operator kind whose chaining delay a device may characterize."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    REM = "rem"
    NEG = "neg"
    CMP = "cmp"
    AND = "and"
    OR = "or"
    XOR = "xor"
    SHL = "shl"
    SHR = "shr"
    SELECT = "select"
    INT_CAST = "icast"
    INT_FLOAT_CAST = "ifcast"
    FLOAT_CAST = "fcast"


class Device:
    """A hardware platform: storage primitives, native-operator chaining delays,
    operator IPs and a default synthesis frequency. Built fluently through
    ``set_memory`` / ``set_comb_delay`` / ``add_operator``."""

    def __init__(self, name: str):
        self.name = name
        self.memory: dict[MemoryKind, MemoryTiming] = {}
        self.default_memory: MemoryTiming | None = None
        self.comb: dict[str, float] = {}  # native chaining delays: kind -> ns
        # What the device HAS, and what each native operator SPENDS of it.
        # Separate from `comb` because a delay is a timing fact and an area is
        # a resource fact; they are declared together and read by different
        # consumers.
        self.resources: dict[str, Resource] = {}
        self.comb_uses: dict[str, dict[str, Cost]] = {}  # kind -> resource -> cost
        self.operators: list[IP] = []  # built-in and user `@ip` operators
        self.default_freq_mhz: float = 100.0

    def add_resource(self, name: str, capacity: int) -> Resource:
        """Declare a resource this device has ``capacity`` of, and return the
        handle a cost refers to."""
        if name in self.resources:
            raise ValueError(f"resource {name!r} already declared")
        if capacity <= 0:
            raise ValueError(f"resource {name!r} must have a positive capacity")
        r = Resource(name, int(capacity))
        self.resources[name] = r
        return r

    def set_memory(
        self,
        kind: MemoryKind,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
    ) -> "Device":
        if not isinstance(kind, MemoryKind):
            raise TypeError(f"kind must be a MemoryKind, got {kind!r}")
        if read_latency < 0 or write_latency < 0:
            raise ValueError(f"{kind.name}: latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError(f"{kind.name}: delay must be non-negative")
        self.memory[kind] = MemoryTiming(
            kind=kind,
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
        )
        return self

    def set_default_memory(self, kind: MemoryKind) -> Device:
        if kind not in self.memory:
            raise ValueError(f"memory {kind!r} not set on device {self.name!r}")
        if kind is MemoryKind.FIFO:
            raise ValueError("FIFO is stream timing, not an array-storage default")
        self.default_memory = self.memory[kind]
        return self

    def set_comb_delay(
        self,
        kind: CombKind,
        delay_ns: float,
        uses: dict[Resource, Cost] | None = None,
    ) -> Device:
        """Set the combinational chaining delay (ns) of a native operator kind,
        and optionally what one instance of it spends. A comb kind carries ONE
        parameter, its operand width, so each cost is a function of that."""
        if not isinstance(kind, CombKind):
            raise TypeError(f"kind must be a CombKind, got {kind!r}")
        if delay_ns < 0:
            raise ValueError(f"comb delay for {kind.value!r} must be non-negative")
        self.comb[kind.value] = float(delay_ns)
        for resource, cost in (uses or {}).items():
            if self.resources.get(resource.name) is not resource:
                raise ValueError(
                    f"{resource.name!r} is not a resource of device {self.name!r}"
                )
            self.comb_uses.setdefault(kind.value, {})[resource.name] = cost
        return self

    def set_default_frequency(self, freq_mhz: float) -> Device:
        if freq_mhz <= 0:
            raise ValueError("default frequency must be positive")
        self.default_freq_mhz = float(freq_mhz)
        return self

    def add_operator(self, operator: IP) -> Device:
        if not isinstance(operator, IP):
            raise TypeError(f"expected an operator IP, got {type(operator).__name__}")
        if operator.optype is None:
            raise ValueError(
                f"IP {operator.func_name!r} is not an operator IP (it has no optype)"
            )
        self.operators.append(operator)
        return self

    def add_operators(self, *ips: IP) -> Device:
        for operator in ips:
            self.add_operator(operator)
        return self

    def copy(self) -> Device:
        """An independent copy, so extending it does not mutate this device. The
        timing and IP objects are shared, never mutated."""
        d = Device(self.name)
        d.memory = dict(self.memory)
        d.default_memory = self.default_memory
        d.comb = dict(self.comb)
        d.resources = dict(self.resources)
        d.comb_uses = {k: dict(v) for k, v in self.comb_uses.items()}
        d.operators = list(self.operators)
        d.default_freq_mhz = self.default_freq_mhz
        return d


# --- operator behavioral descriptors (for cosim) ---------------------------


def _ty(dtype) -> Ty:
    """A behavioral-model :class:`Ty` from an allo scalar dtype."""
    return Ty(
        name=dtype.name,
        width=dtype.primitive_width,
        is_float=dtype.is_float(),
        signed=getattr(dtype, "signed", False),
    )


def operator_descs(operators: Sequence[IP]) -> list[OpDesc]:
    """The device operators as behavioral :class:`OpDesc` descriptors, the cosim
    source of truth for each extern IP's kind, latency and dtypes. Non-operator
    IPs are skipped."""
    out = []
    for op in operators:
        if op.optype is None:
            continue
        kind = (
            op.optype.value if isinstance(op.optype, OperatorType) else str(op.optype)
        )
        rets = op.parse_return_annotation()
        assert len(rets) == 1, f"operator IP {op.func_name!r} must return one scalar"
        out.append(
            OpDesc(
                name=op.func_name,
                kind=kind,
                latency=op.timing.latency,
                arg_types=tuple(_ty(a) for a in op.parse_argument_annotations()),
                ret_type=_ty(rets[0]),
                c_expr=op.c_model,
            )
        )
    return out


# --- injection into the scheduled module -----------------------------------


# See mlir/include/allo/IR/AlloAttrs.td
_STALL_STYLE_TO_ENUM = {"ce": 0, "free": 1, "elastic": 2}


def inject_operators(module, operators: Sequence[IP]):
    """Inject each device operator as a module-level ``dcp.operator`` symbol the
    scheduler and reifier match concrete ``arith.*``/``math.*`` ops onto. The
    ``sym_name`` is the stem of the RTL module name the emitter instantiates:
    one declaration can cover several distinct pieces of hardware, so the
    emitter appends whatever else distinguishes them (a float compare's
    predicate: ``fcmp_l1`` -> ``fcmp_l1_ogt``)."""
    if not operators:
        return
    from ..._mlir.ir import (
        InsertionPoint,
        Location,
        TypeAttr,
        FloatAttr,
        F32Type,
    )
    from ..._mlir.dialects.allo import DCPathOperatorOp, StallContractAttr
    from ...compiler.utils import generate_function_type

    with module.context as ctx, Location.unknown():
        f32ty = F32Type.get()
        insert = InsertionPoint.at_block_begin(module.body)
        for op in operators:
            kind = (
                op.optype.value
                if isinstance(op.optype, OperatorType)
                else str(op.optype)
            )
            sig = generate_function_type(
                ctx, op.parse_argument_annotations(), op.parse_return_annotation()
            )
            t = op.timing
            # A pipelined IP's style, else the clock-enable default.
            stall = StallContractAttr.get(_STALL_STYLE_TO_ENUM[t.style or "ce"], ctx)
            DCPathOperatorOp(
                sym_name=op.func_name,
                kind=kind,
                signature=TypeAttr.get(sig),
                latency=t.latency,
                in_delay=FloatAttr.get(f32ty, t.in_delay_ns),
                out_delay=FloatAttr.get(f32ty, t.out_delay_ns),
                pipelined=t.pipelined,
                stall=stall,
                ip=insert,
            )


def inject_device(module, device: Device):
    """Inject the device technology tables as a module-level ``dcp.device`` op:
    the per-kind combinational chaining delays and the storage model, which
    override the built-in library defaults. Target frequency is not injected: it
    is a per-run scheduling parameter, not technology data."""
    from ..._mlir.ir import (
        ArrayAttr,
        Attribute,
        InsertionPoint,
        Location,
        DictAttr,
        FloatAttr,
        IntegerAttr,
        F32Type,
        IntegerType,
    )
    from ..._mlir.dialects.allo import (
        DCPathCombOp,
        DCPathDeviceOp,
        DCPathResourceOp,
    )

    with module.context, Location.unknown():
        f32ty = F32Type.get()
        i64 = IntegerType.get_signless(64)

        def _timing(t) -> DictAttr:
            return DictAttr.get(
                {
                    "rd_lat": IntegerAttr.get(i64, t.read_latency),
                    "wr_lat": IntegerAttr.get(i64, t.write_latency),
                    "rd_delay": FloatAttr.get(f32ty, t.read_delay_ns),
                    "wr_delay": FloatAttr.get(f32ty, t.write_delay_ns),
                }
            )

        memory = DictAttr.get(
            {
                t.kind.value: _timing(t)
                for t in device.memory.values()
                if t.kind is not MemoryKind.FIFO
            }
        )
        kwargs = {"memory": memory}
        fifo = device.memory.get(MemoryKind.FIFO)
        if fifo is not None:
            kwargs["fifo"] = _timing(fifo)
        if device.default_memory is not None:
            kwargs["default_memory"] = device.default_memory.kind.value
        dev = DCPathDeviceOp(
            sym_name=device.name,
            ip=InsertionPoint.at_block_begin(module.body),
            **kwargs,
        )
        # The body declares what the device HAS and what it realizes natively,
        # each a symbol the others refer to. One op to inject, one to erase.
        body = dev.regions[0].blocks.append()
        with InsertionPoint(body):
            for r in device.resources.values():
                DCPathResourceOp(
                    sym_name=r.name, capacity=IntegerAttr.get(i64, r.capacity)
                )
            for kind, delay in device.comb.items():
                uses = device.comb_uses.get(kind)
                DCPathCombOp(
                    kind=kind,
                    delay=FloatAttr.get(f32ty, delay),
                    uses=(
                        ArrayAttr.get(
                            [
                                Attribute.parse(
                                    f"#allo.res_use<@{name}, [{cost._mlir()}]>"
                                )
                                for name, cost in uses.items()
                            ]
                        )
                        if uses
                        else None
                    ),
                )


# --- built-in operators ----------------------------------------------------
# An `@ip` body is `...`: the parameters exist to declare the IP's signature.
# pylint: disable=unused-argument


@ip(optype=OperatorType.ADD, latency=7, in_delay_ns=0.5, pipelined=True, style="ce")
def fadd_l7(a: f32, b: f32) -> f32: ...


@ip(optype=OperatorType.SUB, latency=7, in_delay_ns=0.5, pipelined=True, style="ce")
def fsub_l7(a: f32, b: f32) -> f32: ...


@ip(optype=OperatorType.MUL, latency=4, in_delay_ns=0.5, pipelined=True, style="ce")
def fmul_l4(a: f32, b: f32) -> f32: ...


@ip(optype=OperatorType.DIV, latency=12, in_delay_ns=0.5, pipelined=True, style="ce")
def fdiv_l12(a: f32, b: f32) -> f32: ...


@ip(optype=OperatorType.CMP, latency=1, in_delay_ns=0.5, pipelined=True, style="ce")
def fcmp_l1(a: f32, b: f32) -> _bool: ...


@ip(optype=OperatorType.ADD, latency=14, in_delay_ns=0.5, pipelined=True, style="ce")
def dadd_l14(a: f64, b: f64) -> f64: ...


@ip(optype=OperatorType.SUB, latency=14, in_delay_ns=0.5, pipelined=True, style="ce")
def dsub_l14(a: f64, b: f64) -> f64: ...


@ip(optype=OperatorType.MUL, latency=9, in_delay_ns=0.5, pipelined=True, style="ce")
def dmul_l9(a: f64, b: f64) -> f64: ...


@ip(optype=OperatorType.DIV, latency=24, in_delay_ns=0.5, pipelined=True, style="ce")
def ddiv_l24(a: f64, b: f64) -> f64: ...


@ip(optype=OperatorType.CMP, latency=1, in_delay_ns=0.5, pipelined=True, style="ce")
def dcmp_l1(a: f64, b: f64) -> _bool: ...


@ip(optype=OperatorType.ADD, latency=4, in_delay_ns=0.5, pipelined=True, style="ce")
def bfadd_l4(a: bf16, b: bf16) -> bf16: ...


@ip(optype=OperatorType.SUB, latency=4, in_delay_ns=0.5, pipelined=True, style="ce")
def bfsub_l4(a: bf16, b: bf16) -> bf16: ...


@ip(optype=OperatorType.MUL, latency=2, in_delay_ns=0.5, pipelined=True, style="ce")
def bfmul_l2(a: bf16, b: bf16) -> bf16: ...


# int <-> float conversion and float resize (one IP per exact width pair).
@ip(
    optype=OperatorType.INT_FLOAT_CAST,
    latency=3,
    in_delay_ns=0.5,
    pipelined=True,
    style="ce",
)
def i2f_l3(a: i32) -> f32: ...


@ip(
    optype=OperatorType.INT_FLOAT_CAST,
    latency=3,
    in_delay_ns=0.5,
    pipelined=True,
    style="ce",
)
def f2i_l3(a: f32) -> i32: ...


@ip(
    optype=OperatorType.FLOAT_CAST,
    latency=2,
    in_delay_ns=0.5,
    pipelined=True,
    style="ce",
)
def fcvt_l2(a: f32) -> f64: ...


@ip(
    optype=OperatorType.FLOAT_CAST,
    latency=2,
    in_delay_ns=0.5,
    pipelined=True,
    style="ce",
)
def bf2f_l2(a: bf16) -> f32: ...


# pylint: enable=unused-argument

# The built-in device: storage + native chaining tables + the operators above.
builtin_device = Device("builtin")
builtin_device.set_memory(MemoryKind.REG, 0, 1, 0.1, 0.1)
builtin_device.set_memory(MemoryKind.LUTRAM, 1, 1, 0.5, 0.5)
builtin_device.set_memory(MemoryKind.BRAM, 1, 1, 0.7, 0.7)
builtin_device.set_memory(MemoryKind.URAM, 2, 1, 0.9, 0.9)
builtin_device.set_memory(MemoryKind.FIFO, 1, 1, 0.5, 0.5)
builtin_device.set_default_memory(MemoryKind.LUTRAM)
builtin_device.set_default_frequency(300.0)
builtin_device.add_operators(
    fadd_l7,
    fsub_l7,
    fmul_l4,
    fdiv_l12,
    fcmp_l1,
    dadd_l14,
    dsub_l14,
    dmul_l9,
    ddiv_l24,
    dcmp_l1,
    bfadd_l4,
    bfsub_l4,
    bfmul_l2,
    i2f_l3,
    f2i_l3,
    fcvt_l2,
    bf2f_l2,
)
# Native chaining delays. Integer arithmetic, mul/div/rem included, is
# combinational; float and cast go through the operators above.
builtin_device.set_comb_delay(CombKind.ADD, 1.2)
builtin_device.set_comb_delay(CombKind.SUB, 1.2)
builtin_device.set_comb_delay(CombKind.MUL, 2.0)
builtin_device.set_comb_delay(CombKind.DIV, 2.5)
builtin_device.set_comb_delay(CombKind.REM, 2.5)
builtin_device.set_comb_delay(CombKind.NEG, 1.0)
builtin_device.set_comb_delay(CombKind.CMP, 1.0)
builtin_device.set_comb_delay(CombKind.AND, 0.4)
builtin_device.set_comb_delay(CombKind.OR, 0.4)
builtin_device.set_comb_delay(CombKind.XOR, 0.4)
builtin_device.set_comb_delay(CombKind.SHL, 0.5)
builtin_device.set_comb_delay(CombKind.SHR, 0.5)
builtin_device.set_comb_delay(CombKind.SELECT, 0.5)
builtin_device.set_comb_delay(CombKind.INT_CAST, 0.3)

__ALL__ = [
    "Device",
    "MemoryKind",
    "MemoryTiming",
    "CombKind",
    "builtin_device",
    "inject_device",
    "inject_operators",
]
