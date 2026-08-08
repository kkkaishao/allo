# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device schema for the RTL backend: what a device declares, how a cost is
expressed, and how both reach the IR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache

from ...lang.ip import OperatorIP, OperatorType
from .sim.ip_models import OpDesc, Ty


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
    #: The two arms of a :func:`Piecewise`, and empty for every other form.
    arms: tuple[Cost, ...] = ()

    def _attr(self):
        """This cost as an ``#allo.cost``, in whatever context is current."""
        from ..._mlir.dialects.allo import CostAttr

        return CostAttr.get(
            self.form,
            [float(c) for c in self.coeffs],
            [a._attr() for a in self.arms],
        )


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
    one's value, and one outside the first and last is not measured at all.

    A staircase, which fits a quantity that really steps: a multiply takes a
    whole number of DSP slices, never 1.7 of one.

    Sampling a continuous quantity into one under-states it at every parameter
    between two points, and nothing downstream re-checks a timing model: a
    48-bit divide reading the 32-bit delay row is 45% short.
    """
    if not points:
        raise ValueError("a cost table needs at least one point")
    flat: list[float] = []
    for p in sorted(points):
        flat += [float(p), float(points[p])]
    return Cost("table", tuple(flat))


def Interp(points: dict[int, float]) -> Cost:
    """The same measured points as a :func:`Table`, read continuously: a
    parameter between two points takes their linear interpolation, and one
    outside the first and last is not measured at all.

    For a quantity that is continuous in its parameter, such as an operator's
    delay in its operand width, which a staircase under-states at every
    parameter between two of its points.
    """
    if not points:
        raise ValueError("a cost table needs at least one point")
    flat: list[float] = []
    for p in sorted(points):
        flat += [float(p), float(points[p])]
    return Cost("interp", tuple(flat))


def Piecewise(bp: float, below: Cost, above: Cost) -> Cost:
    """``p < breakpoint ? below(p) : above(p)``, with arms of any form.

    The general form of :func:`Step`, whose lower arm is forced proportional and
    whose upper arm is forced constant.
    """
    return Cost("piecewise", (float(bp),), (below, above))


#: What one realization spends: ``(resource name, one cost factor per parameter
#: of its kind)`` pairs, which is what ``#allo.res_use`` carries. One pair is
#: one product TERM, and a resource that names several is spent their sum.
Spend = tuple[tuple[str, tuple[Cost, ...]], ...]


def _terms(cost: Cost | Sequence) -> tuple[tuple[Cost, ...], ...]:
    """A ``uses`` value as product terms. A :class:`Cost`, or a sequence of one
    per parameter, is ONE term; a sequence of those is a sum of them."""
    if isinstance(cost, Cost):
        return ((cost,),)
    seq = tuple(cost)
    if seq and isinstance(seq[0], Cost):
        return (seq,)
    return tuple(tuple(term) for term in seq)


#: Every `uses` attribute built so far, keyed by the declaration it came from:
#: pricing one design asks for the same handful of rows tens of thousands of
#: times.
_COST_ATTRS: dict[Spend, object] = {}

#: The same, for the single costs a `dcp.comb` row's delay is, which are not a
#: `Spend` and so cannot share the table above.
_DELAY_ATTRS: dict[Cost, object] = {}


@lru_cache(maxsize=None)
def _scratch_context():
    """The context the evaluation-only attributes below are uniqued in, one per
    process. Not the RTL module's context: an attribute holds its context alive,
    and :meth:`Device.price` is called with no module in reach."""
    from ..._mlir.ir import Context
    from ..._mlir.dialects.allo import register_dialect

    ctx = Context()
    register_dialect(ctx)
    return ctx


def _res_use_array(spent: Spend, scope: str = ""):
    """``spent`` as an ``#allo.res_use`` array, in whatever context is current.
    ``scope`` names the device symbol a reference from outside the device's
    region has to reach through, giving ``@u55c::@lut`` rather than ``@lut``."""
    from ..._mlir.ir import ArrayAttr, SymbolRefAttr
    from ..._mlir.dialects.allo import ResourceUseAttr

    return ArrayAttr.get(
        [
            ResourceUseAttr.get(
                SymbolRefAttr.get([scope, name] if scope else [name]),
                [c._attr() for c in factors],
            )
            for name, factors in spent
        ]
    )


def _res_use_attr(spent: Spend):
    """The ``#allo.res_use`` array for ``spent``, for evaluation only."""
    attr = _COST_ATTRS.get(spent)
    if attr is None:
        with _scratch_context():
            attr = _COST_ATTRS[spent] = _res_use_array(spent)
    return attr


def _cost_attr(cost: Cost):
    """The ``#allo.cost`` for one cost, for evaluation only."""
    attr = _DELAY_ATTRS.get(cost)
    if attr is None:
        with _scratch_context():
            attr = _DELAY_ATTRS[cost] = cost._attr()
    return attr


def _measured_over(cost: Cost) -> str:
    """A cost's measured points, for a diagnostic."""
    if cost.form in {"table", "interp"}:
        return f"{cost.form} measured over {cost.coeffs[0]:g}..{cost.coeffs[-2]:g}"
    return repr(cost)


def _unmeasured(uses: Spend, params: Sequence[int]) -> str:
    """Which cost of ``uses`` the compiler's evaluator declined to read at
    ``params``, named for a diagnostic. Found by asking the evaluator factor by
    factor, so the rule stays the compiler's."""
    for name, factors in uses:
        if len(factors) != len(params):
            continue  # a lone Tiled, which reads the whole tuple
        for factor, param in zip(factors, params):
            if _cost_attr(factor).evaluate(int(param)) is None:
                return f"the cost of {name!r} is a {_measured_over(factor)}"
    return "a cost is not measured"


def Tiled(bits_per_tile: int) -> Cost:
    """``ceil(depth * width / bits_per_tile)``: the shape of a tiled memory.

    Standing alone it reads the whole parameter tuple: a block-RAM tile holds so
    many bits however the array is cut, which puts the product inside the
    ceiling and does not separate. As one of a full set of factors it tiles its
    own parameter instead, ``ceil(p / bits_per_tile)``."""
    if bits_per_tile <= 0:
        raise ValueError("a tile holds a positive number of bits")
    return Cost("tiled", (float(bits_per_tile),))


@dataclass(frozen=True)
# One field per fact the device states about the structure, so the count tracks
# its vocabulary rather than any coupling between them.
class Storage:  # pylint: disable=too-many-instance-attributes
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
    read_latency: int
    write_latency: int
    read_delay_ns: float
    write_delay_ns: float
    # The row that is NOT a memory: one cell per element, no address, no port
    # limit. A completely partitioned array resolves here whatever it would
    # otherwise have taken, and a device declares at most one.
    is_scatter: bool = False
    # Ports of each direction the structure has, ``None`` for no limit. An array
    # needing more is built as a register file rather than as this row.
    max_reads: int | None = None
    max_writes: int | None = None
    # Ports the structure has altogether, each serving a read or a write in a
    # cycle. A block RAM's two ports are one pool, so two writers and a
    # concurrent reader need three of them and it has two. ``None`` where the
    # directions are independent structures, as a LUT RAM's single write port
    # against its replicated reads is. Never below either direction's limit.
    max_ports: int | None = None
    # The vendor attribute that pins an array to this structure, stamped on the
    # emitted array declaration (Xilinx spells it `ram_style = "block"`).
    # ``None`` where the part has no such attribute or nothing can be pinned to
    # the row, and the synthesizer then picks a structure of its own.
    ram_style: str | None = None
    # Whether the structure can come up holding contents. False for one that
    # powers up undefined, as an UltraRAM does: an array declared with
    # compile-time contents cannot be held there.
    can_init: bool = True
    # What it spends. Storage carries two parameters, `(depth, width)`, so each
    # entry is two cost factors or one `Tiled`.
    uses: Spend = ()


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
    NEG = "neg"  # `arith.negf` only: a float sign flip, not an integer negate
    # `arith.minsi`/`minui`/`maxsi`/`maxui`, realized as a compare feeding a
    # multiplexer. A fabric that declares no row for these prices them at the
    # default of 0.1 ns and free.
    MIN = "min"
    MAX = "max"
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


# One attribute per kind of thing a part declares, so the count tracks the
# vocabulary rather than any coupling between them.
# pylint: disable=too-many-instance-attributes
class Device:
    """A hardware platform: what it HAS (resources), what it can REALIZE
    (storage structures, native operator kinds, operator IPs, multiplexers,
    delay chains) and a default synthesis frequency. Built fluently through
    ``add_resource`` / ``add_storage`` / ``set_comb_delay`` / ``add_operator``
    and the ``set_*_uses`` declarations."""

    def __init__(
        self,
        name: str,
        *,
        part: str = "",
        fabric: str = "",
        grade: str = "",
    ):
        self.name = name
        # Identity, for a reader and for a sibling backend that targets the same
        # silicon by part number. Nothing in the compiler switches on these.
        self.part = part
        self.fabric = fabric
        self.grade = grade
        # Native chaining delays: kind -> ns as a function of the operand width.
        self.comb: dict[str, Cost] = {}
        # What a register-to-register path with no operator in it costs (ns).
        self.reg_delay_ns: float = 0.0
        # Separate from `comb`: a delay is a timing fact and an area is a
        # resource fact, read by different consumers.
        self.resources: dict[str, Resource] = {}
        self.comb_uses: dict[str, Spend] = {}  # comb kind -> what it spends
        self.operator_uses: dict[str, Spend] = {}  # IP symbol -> what it spends
        # The three structures the emitter builds that nothing chooses between,
        # so they are one row each rather than a named realization.
        self.mux_uses: Spend = ()
        self.chain_uses: Spend = ()
        self.rom_uses: Spend = ()
        self.storage: dict[str, Storage] = {}
        # The default is a NAME, not a handle: redeclaring a row (a copied
        # device retuned) must not leave it pointing at the replaced one.
        self.default_storage: str | None = None
        self.stream_timing: StreamTiming | None = None
        # Built-in and user `@operator_ip` cores. `operator_uses` above is keyed
        # on their `symbol`.
        self.operators: list[OperatorIP] = []
        self.default_freq_mhz: float = 100.0

    def _spend(
        self,
        what: str,
        params: str,
        uses: dict[Resource, Cost | Sequence] | None,
    ) -> Spend:
        """``uses`` as ``(resource name, factors)`` pairs, one per product term,
        checked against the parameter tuple ``params`` of the realization's
        kind: one factor per parameter, or the single :func:`Tiled` that reads
        them together. A resource whose value is a sequence of terms is spent
        their sum."""
        arity = len(params.split(","))
        spent: list[tuple[str, tuple[Cost, ...]]] = []
        for resource, cost in (uses or {}).items():
            if self.resources.get(resource.name) is not resource:
                raise ValueError(
                    f"{resource.name!r} is not a resource of device {self.name!r}"
                )
            for factors in _terms(cost):
                whole_tuple = len(factors) == 1 and factors[0].form == "tiled"
                if len(factors) != arity and not whole_tuple:
                    raise ValueError(
                        f"{what} is characterized by ({params}), so each term of "
                        f"its cost of {resource.name!r} is {arity} factor(s) or "
                        "one Tiled"
                    )
                spent.append((resource.name, factors))
        return tuple(spent)

    def price(self, uses: Spend, params: Sequence[int]) -> dict[str, int]:
        """What one instance of a realization spends at ``params``.

        Goes through the compiler's own ``CostAttr::evaluate``, so a consumer
        outside the compiler (``benchmark/area.py``) reads the same measured
        model the scheduler will, rather than a second copy of the shapes.

        Raises where a cost was not measured at its parameter: a `Table` and an
        `Interp` hold measurements, which are read and never extrapolated."""
        if not uses:
            return {}
        from ..._mlir.dialects.allo import ResourceUseAttr

        spent = ResourceUseAttr.evaluate_all(_res_use_attr(uses), list(params))
        if spent is None:
            raise ValueError(
                f"{_unmeasured(uses, params)}, so it does not price "
                f"{tuple(params)}; measure it there, or price the realization "
                "at a parameter it covers"
            )
        return dict(spent)

    def comb_delay(self, kind: CombKind | str, width: int) -> float:
        """The chaining delay (ns) of a native operator kind at ``width`` bits,
        including the register floor the measurement saw.

        Evaluated by the compiler's own ``CostAttr::evaluate``, as :meth:`price`
        is, so a reader outside the compiler reads the same curve the scheduler
        does. Returns 0.0 where the device declares no row, and raises at a
        width the row was not measured at.
        """
        name = kind.value if isinstance(kind, CombKind) else kind
        cost = self.comb.get(name)
        if cost is None:
            return 0.0
        delay = _cost_attr(cost).evaluate(int(width))
        if delay is None:
            raise ValueError(
                f"the {self.name} {name!r} delay is a {_measured_over(cost)} "
                f"and was not measured at {width} bits"
            )
        return delay

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

    # Every argument is one independent fact the device declares about the row,
    # and they are keyword-only, so grouping them would only add a name to spell.
    # pylint: disable-next=too-many-arguments
    def add_storage(
        self,
        name: str,
        *,
        read_latency: int,
        write_latency: int,
        read_delay_ns: float = 0.0,
        write_delay_ns: float = 0.0,
        is_scatter: bool = False,
        max_reads: int | None = None,
        max_writes: int | None = None,
        max_ports: int | None = None,
        ram_style: str | None = None,
        can_init: bool = True,
        uses: dict[Resource, Cost | Sequence] | None = None,
    ) -> Storage:
        """Declare a storage realization and return the handle ``bind_storage``
        and :meth:`set_default_storage` refer to.

        Redeclaring a name REPLACES the row: retuning one primitive of a copied
        device is the normal way to build a variant, and the default, being a
        name, keeps pointing at whatever is declared under it.

        ``is_scatter`` marks the row that is not a memory at all: one cell per
        element, which is what a completely partitioned array becomes. A device
        marks at most one, and one that marks none cannot hold a complete
        partition.

        ``max_reads`` / ``max_writes`` are how many ports of each direction the
        structure has, omitted where it has no limit. An array needing more is
        built as a register file. A scatter row declares neither, having no
        addressed port to count.

        ``max_ports`` is how many it has altogether, each serving a read or a
        write in a cycle. Declare it wherever the two directions draw on one
        pool, as a block RAM's two ports do; omit it where they are independent
        structures, as a LUT RAM's single write port and its replicated reads
        are.

        ``ram_style`` is the vendor attribute that pins an array to this
        structure, stamped on the emitted declaration. Omit it and the
        synthesizer chooses instead.

        ``can_init`` is whether the structure comes up holding contents. Clear
        it for one that powers up undefined, as an UltraRAM does; an array
        declared with compile-time contents is then refused there.

        Storage carries two parameters, ``(depth, width)``, so a ``uses`` term
        is a pair of costs, or the single :func:`Tiled` that reads them
        together.
        """
        if read_latency < 0 or write_latency < 0:
            raise ValueError(f"storage {name!r}: latency must be non-negative")
        if read_delay_ns < 0 or write_delay_ns < 0:
            raise ValueError(f"storage {name!r}: delay must be non-negative")
        limits = (
            ("max_reads", max_reads),
            ("max_writes", max_writes),
            ("max_ports", max_ports),
        )
        for role, limit in limits:
            if limit is not None and limit < 1:
                raise ValueError(f"storage {name!r}: {role} must be at least one port")
            if limit is not None and is_scatter:
                raise ValueError(
                    f"storage {name!r} holds one cell per element, so it has no "
                    f"{role} to declare"
                )
        # A pool below one of the directions it serves would make that
        # direction's own limit unreachable, so the row would be describing two
        # different structures.
        for role, limit in limits[:2]:
            if max_ports is not None and limit is not None and limit > max_ports:
                raise ValueError(
                    f"storage {name!r}: {role}={limit} exceeds max_ports={max_ports}, "
                    "but an access of either direction takes one port of the pool"
                )
        other = next(
            (s for s in self.storage.values() if s.is_scatter and s.name != name), None
        )
        if is_scatter and other is not None:
            raise ValueError(
                f"device {self.name!r} already scatters into {other.name!r}; "
                "a device has at most one storage an array can be scattered into"
            )
        s = Storage(
            name=name,
            read_latency=int(read_latency),
            write_latency=int(write_latency),
            read_delay_ns=float(read_delay_ns),
            write_delay_ns=float(write_delay_ns),
            is_scatter=bool(is_scatter),
            max_reads=None if max_reads is None else int(max_reads),
            max_writes=None if max_writes is None else int(max_writes),
            max_ports=None if max_ports is None else int(max_ports),
            ram_style=ram_style,
            can_init=bool(can_init),
            uses=self._spend(f"storage {name!r}", "depth, width", uses),
        )
        self.storage[name] = s
        return s

    def set_storage_uses(
        self, name: str, uses: dict[Resource, Cost | Sequence]
    ) -> Device:
        """What one storage realization spends, over ``(depth, width)``. Apart
        from :meth:`add_storage` so that a device's timing and its area can be
        declared apart, the way a combinational kind's are."""
        s = self.storage.get(name)
        if s is None:
            raise ValueError(f"{name!r} is not a storage of device {self.name!r}")
        self.storage[name] = replace(
            s, uses=self._spend(f"storage {name!r}", "depth, width", uses)
        )
        return self

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
        delay_ns: Cost | float,
        uses: dict[Resource, Cost | Sequence] | None = None,
    ) -> Device:
        """Set the combinational chaining delay of a native operator kind, and
        optionally what one instance of it spends. A comb kind carries one
        parameter, its operand width, and both the delay and each cost are
        functions of it. A bare number is the constant function."""
        if not isinstance(kind, CombKind):
            raise TypeError(f"kind must be a CombKind, got {kind!r}")
        if not isinstance(delay_ns, Cost):
            if delay_ns < 0:
                raise ValueError(f"comb delay for {kind.value!r} must be non-negative")
            delay_ns = Const(float(delay_ns))
        self.comb[kind.value] = delay_ns
        if uses:
            self.comb_uses[kind.value] = self._spend(
                f"comb kind {kind.value!r}", "width", uses
            )
        return self

    def set_operator_uses(
        self, operator: OperatorIP, uses: dict[Resource, Cost | Sequence]
    ) -> Device:
        """What one instance of an operator IP spends. Its parameter is the
        operand width, as a native operator kind's is, even though the IP's
        signature already fixes that width: the arity follows the realization's
        kind so that one rule covers every row."""
        if operator not in self.operators:
            raise ValueError(
                f"{operator.symbol!r} is not an operator of device {self.name!r}"
            )
        self.operator_uses[operator.symbol] = self._spend(
            f"operator {operator.symbol!r}", "width", uses
        )
        return self

    def set_mux_uses(self, uses: dict[Resource, Sequence]) -> Device:
        """What one select over ``k`` sources of ``width`` bits spends."""
        self.mux_uses = self._spend("a multiplexer", "fan-in, width", uses)
        return self

    def set_chain_uses(self, uses: dict[Resource, Sequence]) -> Device:
        """What one ``depth``-stage, ``width``-bit value delay chain spends."""
        self.chain_uses = self._spend("a delay chain", "depth, width", uses)
        return self

    def set_rom_uses(self, uses: dict[Resource, Sequence]) -> Device:
        """What one ``depth`` x ``width`` constant table spends.

        A read-only table is not held in a storage realization at all: it is
        emitted as a constant array read by an index, so the part builds it out
        of logic. That is why it is a whole-device row like the multiplexer and
        the delay chain rather than a ``dcp.storage`` an array binds to."""
        self.rom_uses = self._spend("a constant table", "depth, width", uses)
        return self

    def set_register_floor(self, delay_ns: float) -> Device:
        """The register-to-register floor (ns): a source flip-flop's clock-to-out
        plus the routing every path pays, measured with nothing between the
        registers.

        Every combinational delay this device declares includes it. A cycle pays
        it once however many operators chain within it, so the scheduler charges
        a comb row its whole delay where a chain ends and the delay less this
        floor where a successor extends the chain. It defaults to zero.
        """
        if delay_ns < 0:
            raise ValueError("the register floor must be non-negative")
        self.reg_delay_ns = float(delay_ns)
        return self

    def set_default_frequency(self, freq_mhz: float) -> Device:
        if freq_mhz <= 0:
            raise ValueError("default frequency must be positive")
        self.default_freq_mhz = float(freq_mhz)
        return self

    def add_operator(self, operator: OperatorIP) -> Device:
        """Declare a core this device offers."""
        if not isinstance(operator, OperatorIP):
            raise TypeError(f"expected an operator IP, got {type(operator).__name__}")
        symbol = operator.symbol
        if any(o.symbol == symbol for o in self.operators):
            raise ValueError(
                f"device {self.name!r} already declares an operator {symbol!r}; two "
                "`dcp.operator`s under one symbol is a symbol table error, and a "
                "core differing in kind, signature or latency is named apart on "
                "its own (see OperatorIP.symbol)"
            )
        self.operators.append(operator)
        return self

    def add_operators(self, *ips: OperatorIP) -> Device:
        for operator in ips:
            self.add_operator(operator)
        return self

    def remove_operator(self, operator: OperatorIP | str) -> Device:
        """Drop a core, named by handle or by symbol, and what it spends with
        it. Several cores of one kind and signature are candidates the library
        chooses between, so withdrawing one is how a device offers fewer.
        Raises where the device does not declare it."""
        symbol = operator.symbol if isinstance(operator, OperatorIP) else operator
        found = next((o for o in self.operators if o.symbol == symbol), None)
        if found is None:
            raise ValueError(f"{symbol!r} is not an operator of device {self.name!r}")
        self.operators.remove(found)
        self.operator_uses.pop(symbol, None)
        return self

    def validate(self) -> Device:
        """Check the device is complete enough to compile and to price against,
        and return it. Call it where a device is built, so a part missing a row
        fails there rather than inside a compile.
        """
        if not self.resources:
            raise ValueError(f"device {self.name!r} declares no resources")
        scatter = [s.name for s in self.storage.values() if s.is_scatter]
        if len(scatter) != 1:
            raise ValueError(
                f"device {self.name!r} marks {len(scatter)} scatter storages; it "
                "needs exactly one, since that is what a completely partitioned "
                "array and an array that failed RAM inference both become"
            )
        if self.default_storage is None:
            raise ValueError(
                f"device {self.name!r} sets no default storage, so an array with "
                "no `bind_storage` has nothing to resolve to"
            )
        if self.stream_timing is None:
            raise ValueError(f"device {self.name!r} declares no stream timing")
        # The estimator prices every mux, delay chain and constant table through
        # the one whole-device row, so an absent row reads as free rather than
        # as unmodelled.
        for what, spent in (
            ("mux", self.mux_uses),
            ("chain", self.chain_uses),
            ("constant table", self.rom_uses),
        ):
            if not spent:
                raise ValueError(
                    f"device {self.name!r} declares no {what} cost; every {what} "
                    "in the design would then price at nothing"
                )
        return self

    def copy(self) -> Device:
        """An independent copy, so extending it does not mutate this device. The
        timing and IP objects are shared, never mutated."""
        d = Device(self.name, part=self.part, fabric=self.fabric, grade=self.grade)
        d.comb = dict(self.comb)
        d.resources = dict(self.resources)
        d.comb_uses = dict(self.comb_uses)
        d.operator_uses = dict(self.operator_uses)
        d.mux_uses = self.mux_uses
        d.chain_uses = self.chain_uses
        d.rom_uses = self.rom_uses
        d.storage = dict(self.storage)
        d.default_storage = self.default_storage
        d.stream_timing = self.stream_timing
        d.operators = list(self.operators)
        d.default_freq_mhz = self.default_freq_mhz
        d.reg_delay_ns = self.reg_delay_ns
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


def operator_descs(operators: Sequence[OperatorIP]) -> list[OpDesc]:
    """The device operators as behavioral :class:`OpDesc` descriptors, the cosim
    source of truth for each extern IP's kind, latency and dtypes. ``name`` is
    the operator's symbol, the extern module the emitter instantiates and what
    the model joins on."""
    out = []
    for op in operators:
        kind = (
            op.optype.value if isinstance(op.optype, OperatorType) else str(op.optype)
        )
        rets = op.parse_return_annotation()
        out.append(
            OpDesc(
                name=op.symbol,
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
def inject_operators(module, device: Device):
    """Inject each device operator as a module-level ``dcp.operator`` symbol the
    scheduler and reifier match concrete ``arith.*``/``math.*`` ops onto. The
    ``sym_name`` is the operator's :attr:`~allo.lang.ip.OperatorIP.symbol` and
    the stem of the RTL module name the emitter instantiates. One declaration
    can cover several distinct pieces of hardware, so the emitter appends
    whatever else distinguishes them (a float compare's predicate:
    ``cmp_f32_f32_u1_l1`` -> ``cmp_f32_f32_u1_l1_ogt``).

    The resources an IP spends are the device's, but this op is not in the
    device's symbol table, so its references reach through the device symbol
    (``@u55c::@lut``) and resolve from where they are written."""
    if not device.operators:
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
        for op in device.operators:
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
            stall = StallContractAttr.get(t.style or "ce", ctx)
            DCPathOperatorOp(
                sym_name=op.symbol,
                kind=kind,
                signature=TypeAttr.get(sig),
                latency=t.latency,
                in_delay=FloatAttr.get(f32ty, t.in_delay_ns),
                out_delay=FloatAttr.get(f32ty, t.out_delay_ns),
                pipelined=t.pipelined,
                stall=stall,
                uses=_uses_attr(device.operator_uses.get(op.symbol), device.name),
                ip=insert,
            )


def _uses_attr(spent, scope: str = ""):
    """``uses`` as a ``#allo.res_use`` array, or None when nothing is declared:
    an undeclared cost spends nothing, it is not a zero."""
    return _res_use_array(spent, scope) if spent else None


def inject_device(module, device: Device):
    """Inject the device technology tables as a module-level ``dcp.device`` op:
    the per-kind combinational chaining delays and the storage model, which
    override the built-in library defaults. Target frequency is not injected: it
    is a per-run scheduling parameter, not technology data."""
    from ..._mlir.ir import (
        InsertionPoint,
        Location,
        FloatAttr,
        IntegerAttr,
        F32Type,
        IntegerType,
    )
    from ..._mlir.dialects.allo import (
        OpKindAttr,
        DCPathChainOp,
        DCPathCombOp,
        DCPathDeviceOp,
        DCPathMuxOp,
        DCPathResourceOp,
        DCPathStorageOp,
        DCPathStreamTimingOp,
    )

    with module.context, Location.unknown():
        f32ty = F32Type.get()
        i64 = IntegerType.get_signless(64)

        def _port_limit(ty, n):
            return None if n is None else IntegerAttr.get(ty, n)

        def _timing(t) -> dict:
            return {
                "rd_latency": IntegerAttr.get(i64, t.read_latency),
                "rd_delay": FloatAttr.get(f32ty, t.read_delay_ns),
                "wr_latency": IntegerAttr.get(i64, t.write_latency),
                "wr_delay": FloatAttr.get(f32ty, t.write_delay_ns),
            }

        dev = DCPathDeviceOp(
            sym_name=device.name,
            reg_delay=FloatAttr.get(f32ty, device.reg_delay_ns),
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
                DCPathCombOp(
                    kind=OpKindAttr.get(kind),
                    delay=delay._attr(),
                    uses=_uses_attr(device.comb_uses.get(kind)),
                )
            for s in device.storage.values():
                DCPathStorageOp(
                    sym_name=s.name,
                    is_default=s.name == device.default_storage,
                    is_scatter=s.is_scatter,
                    max_reads=_port_limit(i64, s.max_reads),
                    max_writes=_port_limit(i64, s.max_writes),
                    max_ports=_port_limit(i64, s.max_ports),
                    ram_style=s.ram_style,
                    no_init=not s.can_init,
                    uses=_uses_attr(s.uses),
                    **_timing(s),
                )
            if device.mux_uses:
                DCPathMuxOp(uses=_uses_attr(device.mux_uses))
            if device.chain_uses:
                DCPathChainOp(uses=_uses_attr(device.chain_uses))
            if device.stream_timing is not None:
                DCPathStreamTimingOp(**_timing(device.stream_timing))


__all__ = [
    "Device",
    "Resource",
    "Storage",
    "StreamTiming",
    "CombKind",
    "Cost",
    "Const",
    "Linear",
    "Quadratic",
    "Step",
    "Table",
    "Interp",
    "Piecewise",
    "Tiled",
    "operator_descs",
    "inject_device",
    "inject_operators",
]
