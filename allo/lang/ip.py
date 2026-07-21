import ast
from enum import Enum
from dataclasses import dataclass
from typing import TypeVar, ParamSpec, overload, Literal
from collections.abc import Callable, Sequence

from .kernel import Kernel
from .core import TypeBase
from .._mlir.ir import Module

P = ParamSpec("P")
R = TypeVar("R")


class OperatorType(Enum):
    """The abstract operator kinds an IP can characterize (mirrors ``OpKind`` in
    ``OperatorLibrary.h``). The value is the abstract ``kind`` string the injected
    ``dcp.operator`` carries; ``classify`` maps concrete ``arith.*`` ops onto these
    (int vs float split by the signature dtype). An *advanced* operator
    (``math.sqrt``/``sin``/``cos``/...) has no member here — pass its MLIR op name
    as a bare string for ``optype`` instead."""

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
    INT_CAST = "icast"  # sext / zext / trunc / index_cast
    INT_FLOAT_CAST = "ifcast"  # si/ui-to-fp, fp-to-si/ui
    FLOAT_CAST = "fcast"  # extf / truncf


@dataclass
class Timing:
    # number of cycles to complete the operation
    latency: int
    # time in nanoseconds to receive the input data
    in_delay_ns: float
    # time in nanoseconds to send the output data
    out_delay_ns: float
    # whether the operation can be pipelined
    pipelined: bool
    # pipelining style: free running, elastic, or clock enable
    style: Literal["free", "elastic", "ce"] | None = None


def verify_timing(timing: Timing):
    if timing.latency < 0:
        raise ValueError("Latency must be non-negative.")
    if timing.in_delay_ns < 0:
        raise ValueError("Input delay must be non-negative.")
    if timing.out_delay_ns < 0:
        raise ValueError("Output delay must be non-negative.")
    if timing.pipelined:
        if timing.style is None:
            raise ValueError("Pipelined operations must specify a style.")
        if timing.style not in ("free", "elastic", "ce"):
            raise ValueError("Pipeling style must be one of 'free', 'elastic', 'ce'")
    else:
        if timing.style is not None:
            raise ValueError("Non-pipelined operations cannot specify a style.")


class IP(Kernel[P, R]):
    def __init__(
        self,
        fn: Callable[P, R],
        name: str | None = None,
        latency: int = 1,
        in_delay_ns: float = 0.0,
        out_delay_ns: float = 0.0,
        pipelined: bool = False,
        style: Literal["free", "elastic", "ce"] | None = None,
        optype: OperatorType | str | None = None,
    ):
        super().__init__(fn, mapping=())
        if self.is_async:
            raise TypeError("External IPs cannot be asynchronous.")
        self.func_name = name or fn.__name__
        self.timing = Timing(
            latency=latency,
            in_delay_ns=in_delay_ns,
            out_delay_ns=out_delay_ns,
            pipelined=pipelined,
            style=style,
        )
        self.optype = optype
        verify_timing(self.timing)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    def compile(self):
        raise RuntimeError(
            "External IPs cannot be compiled standalone. They must be used within a kernel."
        )

    def schedule(self):
        raise NotImplementedError(
            "External IPs cannot be scheduled standalone. They must be used within a kernel."
        )

    @property
    def module(self):
        raise NotImplementedError(
            "External IPs do not have a module. They must be used within a kernel."
        )


@overload
def ip(
    fn: Callable[P, R],
    *,
    name: str | None = None,
    latency: int = 1,
    in_delay_ns: float = 0.0,
    out_delay_ns: float = 0.0,
    pipelined: bool = False,
    style: Literal["free", "elastic", "ce"] | None = None,
    optype: OperatorType | str | None = None,
) -> IP[P, R]: ...


@overload
def ip(
    *,
    name: str | None = None,
    latency: int = 1,
    in_delay_ns: float = 0.0,
    out_delay_ns: float = 0.0,
    pipelined: bool = False,
    style: Literal["free", "elastic", "ce"] | None = None,
    optype: OperatorType | str | None = None,
) -> Callable[[Callable[P, R]], IP[P, R]]: ...


def ip(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    latency: int = 1,
    in_delay_ns: float = 0.0,
    out_delay_ns: float = 0.0,
    pipelined: bool = False,
    style: Literal["free", "elastic", "ce"] | None = None,
    optype: OperatorType | str | None = None,
) -> IP[P, R] | Callable[[Callable[P, R]], IP[P, R]]:
    if fn is not None:
        assert callable(fn), "The first argument must be a callable function."
        return IP(
            fn=fn,
            name=name,
            latency=latency,
            in_delay_ns=in_delay_ns,
            out_delay_ns=out_delay_ns,
            pipelined=pipelined,
            style=style,
            optype=optype,
        )
    else:

        def decorator(fn: Callable[P, R]) -> IP[P, R]:
            return IP(
                fn=fn,
                name=name,
                latency=latency,
                in_delay_ns=in_delay_ns,
                out_delay_ns=out_delay_ns,
                pipelined=pipelined,
                style=style,
                optype=optype,
            )

        return decorator
