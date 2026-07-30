# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind numpy kernel arguments to the emitted module's ports"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interface import FIFO, Memory, ModuleInterface, RegisterFile
from ....lang.core import BufferType, DType, StreamType, widen_apint_to_std
from ...utils import dtype_to_numpy_dtype

_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


def bank_elements(shape, axes: tuple[Memory.Axis, ...], bank: int) -> np.ndarray:
    """The flat element indices bank ``bank`` holds, in in-bank offset order.

    The host-side mirror of ``allo::BankLayout``: a cyclic axis of factor ``F``
    puts element ``i`` in bank ``i % F`` at local ``i // F``; a block axis in
    bank ``i // extent`` at ``i % extent``, ``extent = ceil(dim / F)``; a skew
    axis in bank ``(sum of all subscripts) % F``, keeping ``i_d // F`` on its
    distribution dimension ``d``. Axes compose in mixed radix in the order the
    emitter applied them, so the inverse walks them in reverse. ``-1`` marks a
    padding word, a bank slot with no element behind it.
    """
    bank_shape = list(shape)
    peeled = []  # (dim, factor, kind, extent), in the order the emitter applied
    for a in axes:
        extent = -(-bank_shape[a.dim] // a.factor)
        peeled.append((a.dim, a.factor, a.kind, extent))
        bank_shape[a.dim] = extent
    # `bank` in mixed radix over the axis factors, most significant first.
    digits, rest = [], bank
    for _, factor, _, _ in reversed(peeled):
        digits.append(rest % factor)
        rest //= factor
    digits.reverse()
    # Rebuild each original coordinate from this bank's local grid, undoing the
    # axes in reverse.
    coord = list(np.indices(bank_shape))
    for (dim, factor, kind, extent), digit in zip(reversed(peeled), reversed(digits)):
        if kind == "block":
            coord[dim] = coord[dim] + digit * extent
        elif kind == "cyclic":
            coord[dim] = coord[dim] * factor + digit
        else:
            # Skew: `i_d` is the one subscript in [q*F, q*F+F) whose total sum
            # lands on this bank, so the residue is the digit less the others.
            others = sum(coord[k] for k in range(len(shape)) if k != dim)
            coord[dim] = coord[dim] * factor + (digit - others) % factor
    flat, stride, valid = 0, 1, True
    for k in reversed(range(len(shape))):
        flat = flat + coord[k] * stride
        valid = valid & (coord[k] < shape[k])
        stride *= shape[k]
    return np.where(valid, flat, -1).reshape(-1)


@dataclass
class Mem:
    """One backing array behind an external kernel argument (one *bank* of it
    when the argument is partitioned), with the manifest's :class:`Memory`
    interfaces that read from / write to it. A GROUP rather than one interface,
    so ``arg``/``bank`` are its own identity. ``elements`` is this bank's flat
    index per in-bank offset, where the host's layout meets the RTL's address
    arithmetic."""

    arg: int
    np_dtype: type
    width: int  # host word width (the numpy itemsize), not the port's bit width
    size: int  # elements in this bank (== the flattened argument when unbanked)
    bank: int = 0  # which bank of the argument (0 when unbanked)
    elements: np.ndarray | None = None  # flat index per offset (None = unbanked)
    readers: list[Memory] = field(default_factory=list)
    writers: list[Memory] = field(default_factory=list)

    @property
    def writeback(self) -> bool:
        return bool(self.writers)

    def slice_in(self, array: np.ndarray, width: int) -> np.ndarray:
        """This bank's flat uint bit pattern of ``array`` (its own elements for a
        partitioned argument, the whole array otherwise). A padding slot reads 0.
        """
        bits = bit_pattern(array, self.np_dtype, width)
        if self.elements is None:
            return bits
        return np.where(self.elements >= 0, bits[np.maximum(self.elements, 0)], 0)

    def scatter_out(self, array: np.ndarray, values: np.ndarray) -> None:
        """Write this bank's ``values`` back into ``array`` at its own elements,
        skipping padding slots."""
        if self.elements is None:
            array[...] = values.reshape(array.shape)
            return
        live = self.elements >= 0
        array.reshape(-1)[self.elements[live]] = values[live]


@dataclass
class RegFile:
    """A completely-partitioned argument, held at the boundary as one port per
    element rather than as an addressed memory.

    Simpler than :class:`Mem` rather than a variation on it: no address to serve
    and no latency to honor, so the read side is a held assignment and the write
    side commits on the edge its ``we`` is high."""

    port: RegisterFile
    np_dtype: type
    width: int  # host word width (the numpy itemsize)


def plan_regfiles(interface: ModuleInterface, arg_types) -> list[RegFile]:
    """One :class:`RegFile` per completely-partitioned argument."""
    out = []
    for rf in interface.registers:
        np_dt, width, size = _elem(arg_types[rf.arg])
        assert size == len(rf.elements), (
            "the manifest declares a port per element, so the count must equal "
            "the flattened argument"
        )
        out.append(RegFile(rf, np_dt, width))
    return out


def _elem(arg_type) -> tuple[type, int, int]:
    """(numpy dtype, element bit width, flattened size) for a buffer argument."""
    assert isinstance(arg_type, BufferType), "memory port on a non-buffer argument"
    np_dt = dtype_to_numpy_dtype(arg_type.dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    size = int(np.prod(arg_type.shape))
    return np_dt, width, size


def plan_mems(interface: ModuleInterface, arg_types) -> list[Mem]:
    """Group the interface's read/write ports into one :class:`Mem` per
    (argument, bank): a partitioned argument yields one array per bank."""
    mems: dict[tuple[int, int], Mem] = {}

    def entry(port: Memory) -> Mem:
        key = (port.arg, port.bank)
        if key not in mems:
            np_dt, width, total = _elem(arg_types[port.arg])
            elements = None
            if port.factor > 1:
                # One slot per in-bank offset, padding included, which is
                # exactly the RTL bank's address space.
                elements = bank_elements(port.shape, port.axes, port.bank)
                total = int(elements.size)
            mems[key] = Mem(
                port.arg, np_dt, width, total, bank=port.bank, elements=elements
            )
        return mems[key]

    for acc in interface.reads:
        for r in acc:
            entry(r).readers.append(r)
    for acc in interface.writes:
        for w in acc:
            entry(w).writers.append(w)
    return list(mems.values())


@dataclass
class StreamCh:
    """One FIFO channel bound to a kernel stream argument: an input the host
    feeds token-by-token, or an output it drains."""

    port: FIFO
    np_dtype: type
    width: int  # host word width (the numpy itemsize), not the payload's


def _stream_elem(arg_type) -> tuple[type, int]:
    """(numpy dtype, payload bit width) for a stream argument."""
    assert isinstance(arg_type, StreamType), "stream port on a non-stream argument"
    np_dt = dtype_to_numpy_dtype(arg_type.base_type)
    width = int(np.dtype(np_dt).itemsize) * 8
    return np_dt, width


def plan_streams(interface: ModuleInterface, arg_types) -> list[StreamCh]:
    """One :class:`StreamCh` per stream port, in interface order."""
    out = []
    for s in interface.streams:
        np_dt, width = _stream_elem(arg_types[s.arg])
        out.append(StreamCh(s, np_dt, width))
    return out


def bit_pattern(array: np.ndarray, np_dtype: type, width: int) -> np.ndarray:
    """Reinterpret ``array`` (coerced to ``np_dtype``) as a flat uint bit pattern."""
    arr = np.ascontiguousarray(array, dtype=np_dtype).reshape(-1)
    return arr.view(_UINT[width])


def from_bits(bits: np.ndarray, np_dtype: type, width: int, shape) -> np.ndarray:
    """Inverse of :func:`bit_pattern`: uint bits -> ``np_dtype`` array of ``shape``."""
    return bits.astype(_UINT[width], copy=False).view(np_dtype).reshape(shape)


def scalar_bits(value, arg_type) -> int:
    """The integer bit pattern of a scalar argument at its port width."""
    dtype = (
        widen_apint_to_std(arg_type) if not isinstance(arg_type, DType) else arg_type
    )
    np_dt = dtype_to_numpy_dtype(dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    return int(np.array(value, np_dt).view(_UINT[width]))


def from_scalar_bits(bits: int, res_type):
    """Inverse of :func:`scalar_bits`: a result port's integer bit pattern to the
    numpy scalar of the kernel's return type."""
    dtype = (
        widen_apint_to_std(res_type) if not isinstance(res_type, DType) else res_type
    )
    np_dt = dtype_to_numpy_dtype(dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    return np.array(bits & ((1 << width) - 1), _UINT[width]).view(np_dt)[()]
