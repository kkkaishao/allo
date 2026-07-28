# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Port model: bind numpy kernel arguments to the emitted module's ports.

The emitter's port manifest names the kernel argument behind each scalar / read /
write / stream port. Argument index is the *sharing key*: read and write ports on
the same argument are backed by one array (an in-place read-modify-write), and
several read ports on one argument share its array. Given the concrete numpy
arguments, this module plans the backing memories the testbench services.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....lang.core import BufferType, DType, StreamType, widen_apint_to_std
from ...vitis.csim import _numpy_dtype_for_dtype

_UINT = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}


# The port-interface manifest arrives as JSON, one object per emitted module, keyed
# by the RTL module name. Every port name in it is a concrete field name
# (`out_wr0_addr`, `s_st_data`, ...), so the harness never re-derives one. The shape
# of a module's interface:
#   module / symbol: the emitted RTL module name and the MLIR symbol it came from
#   control: {clk, rst, start, done}, the fixed control ABI
#   scalars: [{arg, width, name}]
#   streams: [{arg, input, depth, width, base, data, valid, ready}]
#   reads / writes: [[{arg, bank, factor, width, latency, base, addr, data, [we],
#                      [shape], [axes: [{dim, factor, kind}]]}]]
#     `shape`/`axes` appear only for a partitioned argument: they are the
#     emitter's element-space bank decomposition (``allo::BankLayout``), which
#     the host replays to shard the argument across its bank interfaces.
#   results: [{width, name}]
#   operators: [{module, impl, predicate, ports: [{name, width, role, input}]}]


def bank_elements(shape, axes, bank: int) -> np.ndarray:
    """The flat element indices bank ``bank`` holds, in in-bank offset order.

    The host-side mirror of the emitter's element-space bank decomposition
    (``allo::BankLayout``): a cyclic axis of factor ``F`` puts element ``i`` in
    bank ``i % F`` at local ``i // F``; a block axis puts it in bank
    ``i // extent`` at ``i % extent``, ``extent = ceil(dim / F)``; a skew axis
    puts it in bank ``(sum of all subscripts) % F``, keeping ``i_d // F`` on its
    distribution dimension ``d``. Axes compose in mixed radix, in the order the
    emitter applied them, so the inverse walks them in reverse. ``-1`` marks a
    padding word, i.e. a bank slot with no element behind it, left over when a
    factor does not divide its dimension.
    """
    bank_shape = list(shape)
    peeled = []  # (dim, factor, kind, extent), in the order the emitter applied
    for a in axes:
        dim, factor = a["dim"], a["factor"]
        extent = -(-bank_shape[dim] // factor)
        peeled.append((dim, factor, a["kind"], extent))
        bank_shape[dim] = extent
    # `bank` in mixed radix over the axis factors, most significant first.
    digits, rest = [], bank
    for _, factor, _, _ in reversed(peeled):
        digits.append(rest % factor)
        rest //= factor
    digits.reverse()
    # Rebuild each original coordinate from this bank's local coordinate grid,
    # undoing the axes in reverse (a later axis split what an earlier one left).
    coord = list(np.indices(bank_shape))
    for (dim, factor, kind, extent), digit in zip(reversed(peeled), reversed(digits)):
        if kind == "block":
            coord[dim] = coord[dim] + digit * extent
        elif kind == "cyclic":
            coord[dim] = coord[dim] * factor + digit
        else:
            # Skew: `i_d` is the one subscript in [q*F, q*F+F) whose total sum
            # lands on this bank, and the others are already whole, so the
            # residue is what the digit less their sum leaves.
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
    when the argument is partitioned), with the ports that read from / write to
    it. ``readers`` are ``{addr, data, latency}`` port-name dicts, ``writers``
    are ``{addr, data, we, latency}`` (the concrete field names the emitter
    chose). ``latency`` is the device access latency the schedule was solved
    against, and the driver must honor it (see ``_serve_mem``).
    ``writeback`` marks an argument the kernel writes. ``elements`` is this
    bank's flat index per in-bank offset (see :func:`bank_elements`), the one
    place the host's layout meets the RTL's address arithmetic."""

    arg: int
    np_dtype: type
    width: int  # element bit width
    size: int  # elements in this bank (== the flattened argument when unbanked)
    bank: int = 0  # which bank of the argument (0 when unbanked)
    factor: int = 1  # the argument's total bank count (1 when unbanked)
    elements: np.ndarray | None = None  # flat index per offset (None = unbanked)
    readers: list[dict] = field(default_factory=list)
    writers: list[dict] = field(default_factory=list)

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


def _elem(arg_type) -> tuple[type, int, int]:
    """(numpy dtype, element bit width, flattened size) for a buffer argument."""
    assert isinstance(arg_type, BufferType), "memory port on a non-buffer argument"
    np_dt = _numpy_dtype_for_dtype(arg_type.dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    size = int(np.prod(arg_type.shape))
    return np_dt, width, size


def plan_mems(interface: dict, arg_types) -> list[Mem]:
    """Group the interface's read/write ports into one :class:`Mem` per
    (argument, bank) -- a partitioned argument yields one backing array per
    physical bank."""
    mems: dict[tuple[int, int], Mem] = {}

    def entry(port: dict) -> Mem:
        arg, bank, factor = port["arg"], port["bank"], port["factor"]
        key = (arg, bank)
        if key not in mems:
            np_dt, width, total = _elem(arg_types[arg])
            elements = None
            if factor > 1:
                # The emitter published this argument's bank decomposition; the
                # bank's depth is one slot per in-bank offset, padding included,
                # which is exactly the RTL bank's address space.
                elements = bank_elements(port["shape"], port["axes"], bank)
                total = int(elements.size)
            mems[key] = Mem(
                arg, np_dt, width, total, bank=bank, factor=factor, elements=elements
            )
        return mems[key]

    for acc in interface["reads"]:
        for r in acc:
            entry(r).readers.append(
                {"addr": r["addr"], "data": r["data"], "latency": r["latency"]}
            )
    for acc in interface["writes"]:
        for w in acc:
            entry(w).writers.append(
                {
                    "addr": w["addr"],
                    "data": w["data"],
                    "we": w["we"],
                    "latency": w["latency"],
                }
            )
    return list(mems.values())


@dataclass
class StreamCh:
    """One FIFO channel bound to a kernel stream argument: an input the host
    feeds token-by-token, or an output it drains. ``data``/``valid``/``ready``
    are the concrete handshake port names; ``base`` keys its output buffer."""

    arg: int
    base: str
    is_input: bool
    np_dtype: type
    width: int  # payload bit width
    data: str = ""
    valid: str = ""
    ready: str = ""


def _stream_elem(arg_type) -> tuple[type, int]:
    """(numpy dtype, payload bit width) for a stream argument."""
    assert isinstance(arg_type, StreamType), "stream port on a non-stream argument"
    np_dt = _numpy_dtype_for_dtype(arg_type.base_type)
    width = int(np.dtype(np_dt).itemsize) * 8
    return np_dt, width


def plan_streams(interface: dict, arg_types) -> list[StreamCh]:
    """One :class:`StreamCh` per stream port, in interface order."""
    out = []
    for s in interface["streams"]:
        np_dt, width = _stream_elem(arg_types[s["arg"]])
        out.append(
            StreamCh(
                s["arg"],
                s["base"],
                bool(s["input"]),
                np_dt,
                width,
                s["data"],
                s["valid"],
                s["ready"],
            )
        )
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
    np_dt = _numpy_dtype_for_dtype(dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    return int(np.array(value, np_dt).view(_UINT[width]))


def from_scalar_bits(bits: int, res_type):
    """Inverse of :func:`scalar_bits`: a result port's integer bit pattern ->
    the numpy scalar of the kernel's return type (a float is reinterpreted from
    its bit pattern, a signed int from its two's-complement bits)."""
    dtype = (
        widen_apint_to_std(res_type) if not isinstance(res_type, DType) else res_type
    )
    np_dt = _numpy_dtype_for_dtype(dtype)
    width = int(np.dtype(np_dt).itemsize) * 8
    return np.array(bits & ((1 << width) - 1), _UINT[width]).view(np_dt)[()]
