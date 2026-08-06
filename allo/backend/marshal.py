# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a kernel value crosses the host boundary, stated once for every backend.

The backends differ in what they can carry, and used to differ only in which
helper they happened to call. Here each target declares an :class:`Abi` and asks
:func:`host_type` the same question, so a difference is a row in a table rather
than a divergence between copies."""

from __future__ import annotations

import ctypes

from dataclasses import dataclass
from enum import Enum

import ml_dtypes
import numpy as np

from ..lang.core import APInt, BufferType, DType, IndexType


class Widening(Enum):
    """What a target does with an integer whose width the host cannot name,
    which is anything but 1/8/16/32/64."""

    IR = "ir"  # `generate-apint-wrapper` widened the kernel boundary itself
    HOST = "host"  # the design keeps the exact width, so the host closes the gap
    NONE = "none"  # such a width cannot cross at all


@dataclass(frozen=True)
class Abi:
    """What one target carries across the host boundary. ``name`` is how the
    target calls itself in a diagnostic.

    ``narrow_floats`` and ``wide_scalars`` are separate questions: a boundary
    made of C types can carry binary16 as its raw 16 bits without being able to
    carry a 128-bit integer at all, while one made of ports carries any width
    there is."""

    name: str
    index_width: int  # `index` lowers to i64 under LLVM, to kIndexWidth in RTL
    widening: Widening
    narrow_floats: bool  # binary16 / bfloat16
    wide_scalars: bool  # an integer past every numpy container, as a Python int


LLVM_ABI = Abi("CPU", 64, Widening.IR, narrow_floats=True, wide_scalars=False)
# No narrow floats: the C simulation calls the kernel's own `half` signature, and
# a `half` is not passed the way the raw 16 bits behind it would be.
HLS_CSIM_ABI = Abi(
    "Vitis csim", 32, Widening.IR, narrow_floats=False, wide_scalars=False
)
HLS_HW_ABI = Abi(
    "Vitis emulation/hardware",
    32,
    Widening.NONE,
    narrow_floats=False,
    wide_scalars=False,
)
# The RTL boundary is ports rather than C types, so any bit layout crosses it:
# a narrow float is its own 16 bits, and an integer past every numpy container
# is a Python int, which has no width to run out of.
RTL_ABI = Abi("RTL cosim", 32, Widening.HOST, narrow_floats=True, wide_scalars=True)


# --- one kernel dtype at one boundary ---------------------------------------

_STD_WIDTHS = (8, 16, 32, 64)
_INT_NP = {
    (1, False): np.bool_,
    (8, True): np.int8,
    (16, True): np.int16,
    (32, True): np.int32,
    (64, True): np.int64,
    (8, False): np.uint8,
    (16, False): np.uint16,
    (32, False): np.uint32,
    (64, False): np.uint64,
}
_FLOAT_NP = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
    "float64": np.float64,
}
_NARROW_FLOATS = {"float16", "bfloat16"}
_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
#: The C scalar a generated host reads a value into, by its numpy container.
_NP_C = {
    np.bool_: "uint8_t",
    np.int8: "int8_t",
    np.int16: "int16_t",
    np.int32: "int32_t",
    np.int64: "int64_t",
    np.uint8: "uint8_t",
    np.uint16: "uint16_t",
    np.uint32: "uint32_t",
    np.uint64: "uint64_t",
    np.float32: "float",
    np.float64: "double",
}


@dataclass(frozen=True)
class HostType:
    """One kernel dtype as it crosses one boundary: the numpy dtype the host
    holds it in, and how many of that container's bits the value occupies.

    The two differ exactly where a design keeps a width the host cannot name,
    and closing that gap is what :func:`to_bits` and :func:`from_bits` are for.
    Where the IR widened the boundary instead, they are equal and both are
    no-ops.

    ``np_dtype`` is ``None`` for a value past every numpy container, which
    crosses as a Python int. That is a scalar-only form: numpy is what an array
    is made of, so an array still needs a container."""

    np_dtype: type | None
    value_bits: int
    signed: bool

    @property
    def container_bits(self) -> int:
        """Bits the container carries. A Python int is exactly as wide as the
        value, so it never pads."""
        if self.np_dtype is None:
            return self.value_bits
        return int(np.dtype(self.np_dtype).itemsize) * 8

    @property
    def padded(self) -> bool:
        """Whether the container carries more bits than the value defines."""
        return self.value_bits < self.container_bits

    @property
    def mask(self) -> int:
        return (1 << self.value_bits) - 1

    @property
    def narrow_float(self) -> bool:
        """Whether this is a 16-bit float, which C has no scalar for."""
        return self.np_dtype in (np.float16, ml_dtypes.bfloat16)

    @property
    def ctype(self) -> type:
        """The ctypes scalar for a value crossing by value. A narrow float has
        none, so it crosses as raw ``c_int16`` and the caller reinterprets."""
        if self.narrow_float:
            return ctypes.c_int16
        return np.ctypeslib.as_ctypes_type(np.dtype(self.np_dtype))

    @property
    def c_scalar(self) -> str:
        """The C type this crosses as in a generated host."""
        return _NP_C[self.np_dtype]


def host_type(dtype: DType, abi: Abi) -> HostType:
    """How ``dtype`` crosses ``abi``'s boundary, or the error saying it cannot."""
    if isinstance(dtype, IndexType):
        return HostType(_INT_NP[(abi.index_width, True)], abi.index_width, True)
    if isinstance(dtype, APInt):
        width, signed = dtype.primitive_width, dtype.signed
        if width in (1, *_STD_WIDTHS):
            np_dtype = _INT_NP.get((width, signed))
            if np_dtype is None:
                raise TypeError(f"{abi.name}: dtype {dtype.name} cannot cross")
            return HostType(np_dtype, width, signed)
        if width > 64:
            if not abi.wide_scalars:
                raise TypeError(
                    f"{abi.name}: integer width {width} exceeds the 64 bits the "
                    "host boundary carries"
                )
            return HostType(None, width, signed)
        if abi.widening is Widening.NONE:
            raise TypeError(
                f"{abi.name}: non-standard integer width {width} is unsupported "
                "at the host boundary; use a standard width (8/16/32/64)"
            )
        std = next(s for s in _STD_WIDTHS if width <= s)
        # Under IR widening the kernel boundary IS the wider type, so nothing is
        # left for the host; under host widening the design kept `width`.
        carried = std if abi.widening is Widening.IR else width
        return HostType(_INT_NP[(std, signed)], carried, signed)
    if dtype.name in _FLOAT_NP:
        if dtype.name in _NARROW_FLOATS and not abi.narrow_floats:
            raise TypeError(
                f"{abi.name}: dtype {dtype.name} is unsupported at the host boundary"
            )
        return HostType(_FLOAT_NP[dtype.name], dtype.primitive_width, False)
    raise TypeError(
        f"{abi.name}: dtype {dtype.name} is unsupported at the host boundary"
    )


def element_type(dtype: DType, abi: Abi) -> HostType:
    """``dtype`` where it has to be an ARRAY element. An array is made of numpy,
    so unlike a scalar it cannot fall back to a Python int."""
    host = host_type(dtype, abi)
    if host.np_dtype is None:
        raise TypeError(
            f"{abi.name}: {dtype.name} has no numpy container, so it can cross "
            "only as a scalar, not as an array element"
        )
    return host


# --- numpy arrays across the boundary ---------------------------------------


def as_array(arg, buffer_type: BufferType, abi: Abi) -> np.ndarray:
    """``arg`` as a C-contiguous array of the element type ``abi`` carries.
    Returns the caller's own object when it already is one, which is what lets
    :func:`writeback` tell whether there is a copy to write back."""
    if not isinstance(arg, np.ndarray):
        raise TypeError(f"{abi.name} buffer arguments must be numpy arrays")
    if tuple(arg.shape) != tuple(buffer_type.shape):
        raise ValueError(
            f"Expected buffer shape {tuple(buffer_type.shape)}, got {arg.shape}"
        )
    np_dtype = element_type(buffer_type.dtype, abi).np_dtype
    array = arg if arg.dtype == np_dtype else arg.astype(np_dtype)
    return array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)


def to_ctype_scalar(value, host: HostType):
    """One value as a 1-element ctypes array, for a boundary that takes scalars
    by value. A narrow float has no ctype, so it crosses as the raw 16 bits its
    numpy scalar holds."""
    if host.narrow_float:
        value = host.np_dtype(value).view(np.int16)
    return (host.ctype * 1)(value)


def from_ctype_scalar(raw, host: HostType):
    """Inverse of :func:`to_ctype_scalar`: what crossed as raw bits has to be
    read back through them, or it comes out as the integer they spell."""
    if host.narrow_float:
        return np.int16(raw).view(host.np_dtype)
    return raw


def writeback(pairs) -> None:
    """Copy each coerced array back into the caller's own, skipping the ones the
    boundary could use in place."""
    for original, array in pairs:
        if isinstance(original, np.ndarray) and original is not array:
            original[...] = array.astype(original.dtype, copy=False)


# --- raw bit patterns, for a boundary that is ports rather than memory ------


def to_bits(array: np.ndarray, host: HostType) -> np.ndarray:
    """``array`` as a flat unsigned bit pattern at the boundary's own width. A
    padded value is masked down to its own bits, since that is all a port that
    wide can accept."""
    arr = np.ascontiguousarray(array, dtype=host.np_dtype).reshape(-1)
    bits = arr.view(_UINT[host.container_bits])
    if host.padded:
        return bits & _UINT[host.container_bits](host.mask)
    return bits


def from_bits(bits: np.ndarray, host: HostType, shape) -> np.ndarray:
    """Inverse of :func:`to_bits`. A padded SIGNED value is sign-extended here:
    the design's ports are only as wide as the type, so nothing below the host
    has done it."""
    container = _UINT[host.container_bits]
    u = bits.astype(container, copy=False)
    if host.padded and host.signed:
        sign = container(1 << (host.value_bits - 1))
        pad = container(((1 << host.container_bits) - 1) ^ host.mask)
        u = np.where(u & sign != 0, u | pad, u)
    return u.view(host.np_dtype).reshape(shape)


def scalar_to_bits(value, host: HostType) -> int:
    """One value's bit pattern at the boundary's own width."""
    if host.np_dtype is None:  # a Python int already is its own bit pattern
        return int(value) & host.mask
    return int(to_bits(np.array(value, host.np_dtype), host)[0])


def scalar_from_bits(bits: int, host: HostType):
    """Inverse of :func:`scalar_to_bits`: a numpy scalar, or a Python int where
    the value is past every numpy container."""
    value = bits & host.mask
    if host.np_dtype is not None:
        word = np.array([value], _UINT[host.container_bits])
        return from_bits(word, host, ())[()]
    if host.signed and value >> (host.value_bits - 1):
        value -= 1 << host.value_bits
    return value
