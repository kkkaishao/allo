# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device model for the RTL backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ...lang import f32, f64, bf16, i32, bool as _bool
from ...lang.ip import ip, IP, OperatorType
from .area_tables import declare_xcu55c_area
from .sim.ip_models import OpDesc, Ty


class Port(Enum):
    """Port topology of a storage primitive: how many concurrent read/write
    ports it has, and whether reads and writes contend. The values are
    ``MemoryPortEnum``'s."""

    SINGLE = 0  # one shared read/write port
    S2P = 1  # one dedicated read + one dedicated write, no contention
    T2P = 2  # two shared read/write ports


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


def Tiled(bits_per_tile: int) -> Cost:
    """``ceil(depth * width / bits_per_tile)``: the shape of a tiled memory.

    The one form that reads the WHOLE parameter tuple rather than one of it, so
    it stands alone instead of multiplying a factor per parameter: a block-RAM
    tile holds so many bits however the array is cut, which puts the product
    inside the ceiling and does not separate."""
    if bits_per_tile <= 0:
        raise ValueError("a tile holds a positive number of bits")
    return Cost("tiled", (float(bits_per_tile),))


@dataclass(frozen=True)
class Storage:
    """A storage realization: one buildable structure an array can live in.

    NOT a resource. A resource is a counter (``@lut``, ``@bram36``); this is
    something the device can BUILD out of them, with timing and ports of its
    own, and its ``uses`` names what it spends. That split is why the vocabulary
    is open: a part whose primitives are not Xilinx's declares different names
    and nothing in the compiler switches on the list.

    ``allo.bind.storage impl=`` names one of these, and an array with no
    explicit choice takes the device's :meth:`Device.set_default_storage`.
    """

    name: str
    ports: Port
    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float
    # One (resource name, factors) pair per resource spent. Storage carries two
    # parameters, `(depth, width)`, so factors is two costs or one `Tiled`.
    uses: tuple[tuple[str, tuple[Cost, ...]], ...] = ()


@dataclass(frozen=True)
class StreamTiming:
    """Get/put timing of a stream channel. A stream is a FIFO, not array
    storage: nothing chooses its implementation and nothing binds its ports, so
    it is one row on the device rather than a :class:`Storage`."""

    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float


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
    """A hardware platform: what it HAS (resources), what it can REALIZE
    (storage structures, native operator kinds, operator IPs) and a default
    synthesis frequency. Built fluently through ``add_resource`` /
    ``add_storage`` / ``set_comb_delay`` / ``add_operator``."""

    def __init__(self, name: str):
        self.name = name
        self.comb: dict[str, float] = {}  # native chaining delays: kind -> ns
        # What the device HAS, and what each native operator SPENDS of it.
        # Separate from `comb` because a delay is a timing fact and an area is
        # a resource fact; they are declared together and read by different
        # consumers.
        self.resources: dict[str, Resource] = {}
        self.comb_uses: dict[str, dict[str, Cost]] = {}  # kind -> resource -> cost
        self.storage: dict[str, Storage] = {}
        # The default is a NAME, not a handle: redeclaring a row (a copied
        # device retuned) must not leave it pointing at the replaced one.
        self.default_storage: str | None = None
        self.stream_timing: StreamTiming | None = None
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

    def add_storage(
        self,
        name: str,
        *,
        ports: Port,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
        uses: dict[Resource, Cost | Sequence[Cost]] | None = None,
    ) -> Storage:
        """Declare a storage realization and return the handle ``bind_storage``
        and :meth:`set_default_storage` refer to.

        Redeclaring a name REPLACES the row: retuning one primitive of a copied
        device is the normal way to build a variant, and the default, being a
        name, keeps pointing at whatever is declared under it.

        Storage carries two parameters, ``(depth, width)``, so a ``uses`` entry
        is a pair of costs, or the single :func:`Tiled` that reads them
        together.
        """
        if not isinstance(ports, Port):
            raise TypeError(f"ports must be a Port, got {ports!r}")
        if read_latency < 0 or write_latency < 0:
            raise ValueError(f"storage {name!r}: latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError(f"storage {name!r}: delay must be non-negative")
        spent: list[tuple[str, tuple[Cost, ...]]] = []
        for resource, cost in (uses or {}).items():
            if self.resources.get(resource.name) is not resource:
                raise ValueError(
                    f"{resource.name!r} is not a resource of device {self.name!r}"
                )
            factors = (cost,) if isinstance(cost, Cost) else tuple(cost)
            if len(factors) != (1 if factors[0].form == "tiled" else 2):
                raise ValueError(
                    f"storage {name!r} spends {resource.name!r} over (depth, width), "
                    "so its cost is two factors or one Tiled"
                )
            spent.append((resource.name, factors))
        s = Storage(
            name=name,
            ports=ports,
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
            uses=tuple(spent),
        )
        self.storage[name] = s
        return s

    def set_default_storage(self, storage: Storage) -> Device:
        """The storage an array with no ``bind_storage`` resolves to. Takes a
        realization, so defaulting to a :class:`Resource` is a type error rather
        than a name that fails to resolve much later."""
        if not isinstance(storage, Storage):
            raise TypeError(
                f"the default must be a storage realization, got "
                f"{type(storage).__name__}"
            )
        if self.storage.get(storage.name) is not storage:
            raise ValueError(
                f"{storage.name!r} is not a storage of device {self.name!r}"
            )
        self.default_storage = storage.name
        return self

    def set_stream_timing(
        self,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
    ) -> Device:
        """Get/put timing of a stream channel."""
        if read_latency < 0 or write_latency < 0:
            raise ValueError("stream latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError("stream delay must be non-negative")
        self.stream_timing = StreamTiming(
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
        )
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
        d.comb = dict(self.comb)
        d.resources = dict(self.resources)
        d.comb_uses = {k: dict(v) for k, v in self.comb_uses.items()}
        d.storage = dict(self.storage)
        d.default_storage = self.default_storage
        d.stream_timing = self.stream_timing
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
        FlatSymbolRefAttr,
        InsertionPoint,
        Location,
        FloatAttr,
        IntegerAttr,
        F32Type,
        IntegerType,
    )
    from ..._mlir.dialects.allo import (
        DCPathCombOp,
        DCPathDefaultStorageOp,
        DCPathDeviceOp,
        DCPathResourceOp,
        DCPathStorageOp,
        DCPathStreamTimingOp,
        MemoryPortAttr,
    )

    with module.context as ctx, Location.unknown():
        f32ty = F32Type.get()
        i64 = IntegerType.get_signless(64)

        def _uses(spent) -> ArrayAttr | None:
            """``uses`` as a `#allo.res_use` array, or None when nothing is
            declared: an undeclared cost spends nothing, it is not a zero."""
            if not spent:
                return None
            return ArrayAttr.get(
                [
                    Attribute.parse(
                        f"#allo.res_use<@{name}, "
                        f"[{', '.join(c._mlir() for c in factors)}]>"
                    )
                    for name, factors in spent
                ]
            )

        def _timing(t) -> dict:
            return {
                "rd_latency": IntegerAttr.get(i64, t.read_latency),
                "rd_delay": FloatAttr.get(f32ty, t.read_delay_ns),
                "wr_latency": IntegerAttr.get(i64, t.write_latency),
                "wr_delay": FloatAttr.get(f32ty, t.write_delay_ns),
            }

        dev = DCPathDeviceOp(
            sym_name=device.name,
            ip=InsertionPoint.at_block_begin(module.body),
        )
        # The body declares what the device HAS and what it can REALIZE, each a
        # symbol the others refer to. One op to inject, one to erase.
        body = dev.regions[0].blocks.append()
        with InsertionPoint(body):
            for r in device.resources.values():
                DCPathResourceOp(
                    sym_name=r.name, capacity=IntegerAttr.get(i64, r.capacity)
                )
            for kind, delay in device.comb.items():
                # A comb kind carries ONE parameter, its operand width, so each
                # cost is a single factor.
                comb_uses = (device.comb_uses.get(kind) or {}).items()
                DCPathCombOp(
                    kind=kind,
                    delay=FloatAttr.get(f32ty, delay),
                    uses=_uses([(name, (cost,)) for name, cost in comb_uses]),
                )
            for s in device.storage.values():
                DCPathStorageOp(
                    sym_name=s.name,
                    ports=MemoryPortAttr.get(s.ports.value, ctx),
                    uses=_uses(s.uses),
                    **_timing(s),
                )
            if device.default_storage is not None:
                DCPathDefaultStorageOp(
                    storage=FlatSymbolRefAttr.get(device.default_storage)
                )
            if device.stream_timing is not None:
                DCPathStreamTimingOp(**_timing(device.stream_timing))


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
# The storage realizations, under the names an `allo.bind.storage impl=`
# resolves against. `register` is the one name the COMPILER also spells: a
# complete partition scatters an array into flip-flops whatever else it would
# have resolved to, so every device must declare a row under it. `srl` is a
# shift register, a realization of its own that happens to spend LUTs.
builtin_device.add_storage(
    "register",
    ports=Port.T2P,
    read_latency=0,
    write_latency=1,
    read_delay_ns=0.1,
    write_delay_ns=0.1,
)
_lutram = builtin_device.add_storage(
    "lutram",
    ports=Port.T2P,
    read_latency=1,
    write_latency=1,
    read_delay_ns=0.5,
    write_delay_ns=0.5,
)
builtin_device.add_storage(
    "bram",
    ports=Port.T2P,
    read_latency=1,
    write_latency=1,
    read_delay_ns=0.7,
    write_delay_ns=0.7,
)
builtin_device.add_storage(
    "uram",
    ports=Port.T2P,
    read_latency=2,
    write_latency=1,
    read_delay_ns=0.9,
    write_delay_ns=0.9,
)
builtin_device.add_storage(
    "srl",
    ports=Port.SINGLE,
    read_latency=1,
    write_latency=1,
    read_delay_ns=0.5,
    write_delay_ns=0.5,
)
builtin_device.set_default_storage(_lutram)
builtin_device.set_stream_timing(1, 1, 0.5, 0.5)
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
# What the part has and what each row above spends of it, measured on xcu55c.
declare_xcu55c_area(builtin_device)

__ALL__ = [
    "Device",
    "Port",
    "Storage",
    "StreamTiming",
    "CombKind",
    "builtin_device",
    "inject_device",
    "inject_operators",
]
