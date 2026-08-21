# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the microarchitecture stage decided, as data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class RegRole(str, Enum):
    """Why a register exists. The emitter knows this where it builds the
    register; a reader of the emitted Verilog can only guess it from the name,
    and a counted delay it cannot attribute at all."""

    VALUE = "value"  # a value delay chain: a datum carried across cycles
    PULSE = "pulse"  # an activation chain: a region's issue delayed to a stage
    COUNTED = "counted"  # the counter a deep pulse delay is built as instead
    SURVIVOR = "survivor"  # a region result, or a loop-carried latch
    COUNTER = "counter"  # an iteration counter, or one of its address strides
    CONTROL = "control"  # run / phase / pending / done, and the rest
    STORAGE = "storage"  # one element of an array scattered into registers


@dataclass(frozen=True)
class RegClass:
    """``count`` runs of ``depth`` registers in series, ``width`` bits each.

    The run, not the register, is the cost unit: past the synthesizer's
    shift-register extraction threshold a run stops costing flip-flops per
    stage, so a model handed only a register total cannot price it. A
    synchronous reset blocks that extraction and pays fabric per bit; a clock
    enable is free."""

    role: RegRole
    width: int
    depth: int
    count: int
    reset: bool = True
    enable: bool = False

    @property
    def bits(self) -> int:
        return self.width * self.depth * self.count

    @classmethod
    def from_json(cls, d: dict) -> RegClass:
        return cls(
            RegRole(d["role"]),
            d["width"],
            d["depth"],
            d["count"],
            d["reset"],
            d["enable"],
        )


@dataclass(frozen=True)
class Unit:
    """One functional-unit instance. ``bound_ops > 1`` is exactly a sharing
    decision: the trivial binding leaves every operation its own unit."""

    identity: str  # the sharing equivalence class
    width: int  # result width in bits
    latency: int
    bound_ops: int
    comb: bool  # native combinational, against an IP instance
    pipelined: bool
    impl: str | None = None  # the device operator symbol; None for a native unit
    module: str | None = None  # the extern RTL module; None for a native unit
    #: a standalone apply unit's cone, the operator counts of its map; zero
    #: everywhere else
    adders: int = 0
    multipliers: int = 0
    dividers: int = 0

    @classmethod
    def from_json(cls, d: dict) -> Unit:
        return cls(
            identity=d["identity"],
            width=d["width"],
            latency=d["latency"],
            bound_ops=d["bound_ops"],
            comb=d["comb"],
            pipelined=d["pipelined"],
            impl=d.get("impl"),
            module=d.get("module"),
            adders=d.get("adders", 0),
            multipliers=d.get("multipliers", 0),
            dividers=d.get("dividers", 0),
        )


@dataclass(frozen=True)
class Chain:
    """One value delay chain: a datum carried across cycle boundaries.

    ``range_bits`` is the width of the value range a model-level interval walk
    proved for the carried datum, ``None`` where it could not."""

    region: int  # owning region's order
    width: int  # built carrier bits
    carried: str  # the held type spelled ("index", "i32", "f32")
    depth: int  # chain length in cycles
    ii: int  # owning region's interval; folds registers at > 1
    taps: int  # distinct consumer read depths
    source: str  # driving cell class, or a unit's mnemonic / IP symbol
    range_bits: int | None = None

    @classmethod
    def from_json(cls, d: dict) -> Chain:
        return cls(
            region=d["region"],
            width=d["width"],
            carried=d["carried"],
            depth=d["depth"],
            ii=d["ii"],
            taps=d["taps"],
            source=d["source"],
            range_bits=d.get("range_bits"),
        )


@dataclass(frozen=True)
class MuxClass:
    """``count`` multiplexers, each ``fanin`` sources wide at ``width`` bits.
    Aggregated rather than enumerated: a mux costs a function of exactly those
    two numbers, and nothing downstream needs to know which ports it feeds."""

    fanin: int
    width: int
    count: int

    @classmethod
    def from_json(cls, d: dict) -> MuxClass:
        return cls(d["fanin"], d["width"], d["count"])


@dataclass(frozen=True)
class MuxCone:
    """``count`` select cones around storage: shared-port address selects,
    commit sinks and bank/scatter crossbars. Disjoint from :class:`MuxClass`,
    which holds the allocation's own muxes."""

    role: str  # "address" / "commit" / "crossbar"
    fanin: int
    width: int
    count: int

    @classmethod
    def from_json(cls, d: dict) -> MuxCone:
        return cls(d["role"], d["fanin"], d["width"], d["count"])


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class MemoryCost:
    """What the cost model needs of one array and no reader does: the ports it
    was bound with, and who drives them.

    ``call_reads`` / ``call_writes`` count the ports a child drives: several
    ports of one child are that child's own boundary, several children are
    concurrent writers, and only the second is a banking problem."""

    call_reads: int
    call_writes: int
    #: ports one bank is built with, per direction: the distinct ports the
    #: binding assigned. Accesses share one only where the model proved they
    #: never issue in the same cycle.
    read_ports: int
    write_ports: int
    #: ports it is built with altogether, which is not their sum: a port of a
    #: pooled storage may carry both a read and a write that never issue
    #: together, and then one address bus carries both.
    ports: int
    #: instances of the storage row each bank is held in, as the compiler
    #: decided them once the ports were bound. The row's price is multiplied by
    #: it.
    instances: int = 1
    #: instances the schedule reserved against, which ``instances`` above may
    #: exceed when the binding replicates further for read bandwidth.
    copies_budget: int = 1
    #: ports one instance of the row provides, 0 for no limit, which
    #: ``instances`` above is the multiplier of. The device row narrowed by the
    #: topology the array asked for.
    row_reads: int = 0
    row_writes: int = 0
    #: a lower bound on what one cycle asks of one bank, per direction; the
    #: ports above are what was built to serve it. Zero for a ROM or a scattered
    #: array, neither addressed.
    read_concurrency: int = 0
    write_concurrency: int = 0
    #: module interface groups this array contributes: one per bound boundary
    #: port, one per group a child masters on it, or one per element of a
    #: scattered argument.
    boundary_ports: int = 0

    @classmethod
    def from_json(cls, d: dict) -> MemoryCost:
        return cls(
            call_reads=d["call_reads"],
            call_writes=d["call_writes"],
            read_ports=d["read_ports"],
            write_ports=d["write_ports"],
            ports=d["ports"],
            instances=d.get("instances", 1),
            copies_budget=d.get("copies_budget", 1),
            row_reads=d.get("row_reads", 0),
            row_writes=d.get("row_writes", 0),
            read_concurrency=d.get("read_concurrency", 0),
            write_concurrency=d.get("write_concurrency", 0),
            boundary_ports=d.get("boundary_ports", 0),
        )


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Memory:
    """One array, and the storage decision taken for it."""

    owner: str  # the name its ports are spelled from
    shape: tuple[int, ...]
    width: int  # element bits
    banks: int
    layout: str  # "none" / "cyclic" / "block" / "skew" / "mixed" / "complete"
    storage: str  # the resolved device storage realization
    depth_words: int  # elements per bank
    read_latency: int
    write_latency: int
    reads: int
    writes: int
    cost: MemoryCost
    external: bool
    scattered: bool
    writes_independent: bool
    rom: bool
    skewed: bool
    #: what the module built to hold it: ``"boundary"`` (the cells are the
    #: caller's), ``"rom"``, ``"scatter"`` or ``"ram"``.
    realization: str
    #: whether the partition BOUGHT the bandwidth it costs: every access reaches
    #: one bank. An access the analysis could not fix takes a port on every
    #: bank, so a partition resolving none of them is N memories at the
    #: bandwidth of one. True for an unpartitioned array, which resolves
    #: nothing because it has nothing to resolve.
    partition_resolved: bool

    @property
    def bits(self) -> int:
        """Stored bits across every bank."""
        return self.depth_words * self.width * self.banks

    @classmethod
    def from_json(cls, d: dict) -> Memory:
        return cls(
            owner=d["owner"],
            shape=tuple(d["shape"]),
            width=d["width"],
            banks=d["banks"],
            layout=d["layout"],
            storage=d["storage"],
            depth_words=d["depth_words"],
            read_latency=d["read_latency"],
            write_latency=d["write_latency"],
            reads=d["reads"],
            writes=d["writes"],
            cost=MemoryCost.from_json(d["cost"]),
            external=d["external"],
            scattered=d["scattered"],
            writes_independent=d["writes_independent"],
            rom=d["rom"],
            skewed=d["skewed"],
            realization=d["realization"],
            partition_resolved=d["partition_resolved"],
        )


@dataclass(frozen=True)
class Stream:
    """One FIFO channel."""

    owner: str
    width: int
    depth: int
    crosses_call: bool  # an end of it is a child port, not a local access
    internal: bool  # created in this body: its ``seq.fifo`` lives here

    @classmethod
    def from_json(cls, d: dict) -> Stream:
        return cls(d["owner"], d["width"], d["depth"], d["crosses_call"], d["internal"])


@dataclass(frozen=True)
class Call:
    """Sub-kernel invocations of one callee."""

    callee: str
    count: int
    spawns: int  # of those, `await` spawns rather than scheduled calls
    #: How those calls are released, counted: on a predecessor's ``done``, on
    #: the container's own start, or at a scheduled offset.
    handshake: int = 0
    broadcast: int = 0
    timed: int = 0
    latency: int | None = None  # the child's declared span, when static

    @classmethod
    def from_json(cls, d: dict) -> Call:
        return cls(
            d["callee"],
            d["count"],
            d["spawns"],
            d.get("handshake", 0),
            d.get("broadcast", 0),
            d.get("timed", 0),
            d.get("latency"),
        )


@dataclass(frozen=True)
class StrideCost:
    """One address stride register beside the counter: its width and which
    update cells it builds (a step adder, a carry adder with its select, a
    wrap compare with its fix adder and select). ``is_counter`` marks the
    stride that is the counter itself and so builds no register."""

    width: int
    step: bool
    carry: bool
    wrap: bool
    is_counter: bool = False

    @classmethod
    def from_json(cls, d: dict) -> StrideCost:
        return cls(
            d["width"], d["step"], d["carry"], d["wrap"], d.get("is_counter", False)
        )


@dataclass(frozen=True)
class RegionCost:
    """What the cost model needs of one region and no reader does.

    ``mux_bits`` is 2:1-equivalent: a k:1 mux costs about (k-1) 2:1 muxes per
    bit, which is the shape a device prices."""

    mux_inputs: int
    mux_bits: int
    counter_width: int  # the iteration counter this region builds
    phase_width: int  # the [0, II) phase counter of a pipelined leaf at II>1
    addr_strides: int  # address registers riding beside it
    strides: tuple[StrideCost, ...] = ()

    @classmethod
    def from_json(cls, d: dict) -> RegionCost:
        return cls(
            mux_inputs=d["mux_inputs"],
            mux_bits=d["mux_bits"],
            counter_width=d["counter_width"],
            phase_width=d.get("phase_width", 0),
            addr_strides=d["addr_strides"],
            strides=tuple(StrideCost.from_json(s) for s in d.get("strides", ())),
        )


@dataclass(frozen=True)
class RegionUarch:
    """One region's allocation. ``order`` is the join key to the schedule
    result's :class:`RegionSchedule`: both are program order within the func."""

    order: int
    shape: str  # leaf / container / guard / callnode
    kind: str  # "cyclic" or "acyclic"
    compute_ops: int  # operations bound to a unit in this region
    cost: RegionCost
    units: list[Unit] = field(default_factory=list)
    muxes: list[MuxClass] = field(default_factory=list)
    interval: int | None = None  # initiation interval; cyclic regions only

    @property
    def shared_units(self) -> list[Unit]:
        """Units carrying more than one operation, which is what a sharing
        binding bought and the trivial binding never has."""
        return [u for u in self.units if u.bound_ops > 1]

    @classmethod
    def from_json(cls, d: dict) -> RegionUarch:
        return cls(
            order=d["order"],
            shape=d["shape"],
            kind=d["kind"],
            compute_ops=d["compute_ops"],
            cost=RegionCost.from_json(d["cost"]),
            units=[Unit.from_json(u) for u in d["units"]],
            muxes=[MuxClass.from_json(m) for m in d["muxes"]],
            interval=d.get("interval"),
        )


@dataclass(frozen=True)
class TimingStep:
    """One step of a combinational path: what the signal passes through, and
    what the path spends there in ns. A step is one model cell, so it can be a
    lump the model prices without decomposing (an address cone, a select)."""

    what: str
    delay: float  # ns

    @classmethod
    def from_json(cls, d: dict) -> TimingStep:
        return cls(d["what"], d["ns"])


@dataclass(frozen=True)
class TimingPath:
    """One combinational path, start point first: a register or port launches
    it, each step adds its delay, and it is captured at ``endpoint``. ``total``
    is the sum of the steps by construction."""

    total: float  # ns
    slack: float  # period - total; negative means it misses the clock
    endpoint: str
    where: str  # source anchor of the endpoint, empty when it has none
    steps: tuple[TimingStep, ...]

    def describe(self, indent: str = "") -> str:
        """The path as a table: each step's own delay, the running total, and
        what the signal passes through."""
        head = (
            f"{indent}{self.total:.2f} ns, {self.slack:+.2f} ns slack, reaching "
            f"{self.endpoint}" + (f" at {self.where}" if self.where else "")
        )
        lines, run = [head], 0.0
        for s in self.steps:
            run += s.delay
            lines.append(f"{indent}  {s.delay:6.2f} {run:8.2f}  {s.what}")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, d: dict) -> TimingPath:
        return cls(
            total=d["total_ns"],
            slack=d["slack_ns"],
            endpoint=d["endpoint"],
            where=d.get("where", ""),
            steps=tuple(TimingStep.from_json(s) for s in d["steps"]),
        )


@dataclass(frozen=True)
class FuncUarch:
    """One emitted module."""

    func: str  # the MLIR symbol; joins to `FuncSchedule.name`
    module: str  # the emitted RTL module name; joins to `Interfaces`
    top: bool
    read_ports: int
    write_ports: int
    regions: list[RegionUarch] = field(default_factory=list)
    #: module-wide: a register run belongs to the value it carries, not to a
    #: region, and the ledger counts it where it is BUILT.
    regs: list[RegClass] = field(default_factory=list)
    #: every value delay chain the model holds, one row each. The ledger's
    #: value-role classes in ``regs`` also count chains built outside the
    #: model (read-data alignment, stall holds), so these sum to less.
    chains: list[Chain] = field(default_factory=list)
    #: module-wide, like ``regs``: the select cones built around storage.
    mux_cones: list[MuxCone] = field(default_factory=list)
    mems: list[Memory] = field(default_factory=list)
    streams: list[Stream] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    #: this module's worst combinational paths, longest first. Structures with
    #: no delay model are absent from them, so they are estimates and no
    #: substitute for place and route.
    critical_paths: tuple[TimingPath, ...] = ()

    @property
    def critical_ns(self) -> float:
        """The longest path's total in ns, zero where none was published."""
        return self.critical_paths[0].total if self.critical_paths else 0.0

    @property
    def reg_bits(self) -> int:
        """Flip-flops in this module. A COUNT, not an estimate: every register
        is built at one place in the emitter and charged there."""
        return sum(c.bits for c in self.regs)

    def reg_bits_by_role(self) -> dict[RegRole, int]:
        out: dict[RegRole, int] = {}
        for c in self.regs:
            out[c.role] = out.get(c.role, 0) + c.bits
        return out

    def region(self, order: int) -> RegionUarch:
        return next(r for r in self.regions if r.order == order)

    @classmethod
    def from_json(cls, d: dict) -> FuncUarch:
        return cls(
            func=d["func"],
            module=d["module"],
            top=d["top"],
            read_ports=d["read_ports"],
            write_ports=d["write_ports"],
            regions=[RegionUarch.from_json(r) for r in d["regions"]],
            regs=[RegClass.from_json(c) for c in d["regs"]],
            chains=[Chain.from_json(c) for c in d.get("chains", [])],
            mux_cones=[MuxCone.from_json(m) for m in d["mux_cones"]],
            mems=[Memory.from_json(m) for m in d["mems"]],
            streams=[Stream.from_json(s) for s in d["streams"]],
            calls=[Call.from_json(c) for c in d["calls"]],
            critical_paths=tuple(TimingPath.from_json(p) for p in d["critical_paths"]),
        )


@dataclass(frozen=True)
class MicroarchReport:
    """One emission: every module it built, in emit order (callees first)."""

    binding: str  # the sharing policy this emission ran under
    cycle_time: float  # ns, the period the schedule was cut to
    funcs: list[FuncUarch] = field(default_factory=list)

    @property
    def reg_bits(self) -> int:
        """Flip-flops across the design."""
        return sum(f.reg_bits for f in self.funcs)

    def func(self, suffix: str) -> FuncUarch:
        """The module whose MLIR symbol ends with ``suffix`` (kernels compose by
        calling sub-kernels, so results carry ``top.sub`` funcs)."""
        return next(f for f in self.funcs if f.func.endswith(suffix))

    @property
    def top(self) -> FuncUarch:
        return next(f for f in self.funcs if f.top)

    def mem(self, owner: str) -> Memory:
        """The array named ``owner``, wherever in the design it was built."""
        return next(m for f in self.funcs for m in f.mems if m.owner == owner)

    @classmethod
    def from_json(cls, text: str | dict) -> MicroarchReport:
        d = json.loads(text) if isinstance(text, str) else text
        return cls(
            binding=d["binding"],
            cycle_time=d["cycle_time"],
            funcs=[FuncUarch.from_json(f) for f in d["funcs"]],
        )
