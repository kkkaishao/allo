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
        self.operators: list[IP] = []  # built-in and user `@ip` operators
        self.default_freq_mhz: float = 100.0

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

    def set_comb_delay(self, kind: CombKind, delay_ns: float) -> Device:
        """Set the combinational chaining delay (ns) of a native operator kind."""
        if not isinstance(kind, CombKind):
            raise TypeError(f"kind must be a CombKind, got {kind!r}")
        if delay_ns < 0:
            raise ValueError(f"comb delay for {kind.value!r} must be non-negative")
        self.comb[kind.value] = float(delay_ns)
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
    ``sym_name`` IS the RTL module name the emitter instantiates."""
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
        InsertionPoint,
        Location,
        DictAttr,
        FloatAttr,
        IntegerAttr,
        F32Type,
        IntegerType,
    )
    from ..._mlir.dialects.allo import DCPathDeviceOp

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

        comb = DictAttr.get(
            {k: FloatAttr.get(f32ty, v) for k, v in device.comb.items()}
        )
        memory = DictAttr.get(
            {
                t.kind.value: _timing(t)
                for t in device.memory.values()
                if t.kind is not MemoryKind.FIFO
            }
        )
        kwargs = {"comb": comb, "memory": memory}
        fifo = device.memory.get(MemoryKind.FIFO)
        if fifo is not None:
            kwargs["fifo"] = _timing(fifo)
        if device.default_memory is not None:
            kwargs["default_memory"] = device.default_memory.kind.value
        DCPathDeviceOp(ip=InsertionPoint.at_block_begin(module.body), **kwargs)


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
