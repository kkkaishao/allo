# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-facing operator timing library for the RTL scheduler.

A library describes, for one device at one target frequency, the timing every
operator presents to the scheduler: ``latency`` (cycles), ``delay`` (ns, for
chaining/clock closure), and ``pipelined``. It is authored by someone who reads
FPGA timing reports -- no knowledge of the compiler IR is required. Operators are
named by an abstract ``(kind, dtype, width)`` signature; the C++ backend maps
concrete IR ops onto it. Rare ops we do not predefine (e.g. ``math.rsqrt``) are
characterized through :meth:`OperatorLibrary.advanced` by their MLIR name.

This mirrors ``mlir/.../OperatorLibrary.{h,cpp}``; it serializes to the YAML that
the ``allo-schedule`` pass consumes via its ``operator-library`` option.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# The abstract operator kinds (mirror OpKind in OperatorLibrary.h). The three
# cast kinds are split because int-resize, int<->float conversion, and
# float-resize have distinct hardware timing.
OP_KINDS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "neg",
        "cmp",
        "and",
        "or",
        "xor",
        "shl",
        "shr",
        "select",
        "int_cast",
        "int_float_cast",
        "float_cast",
        "mem_read",
        "mem_write",
        "stream_read",
        "stream_write",
    }
)

# The abstract datatypes (mirror OpDType). ``int`` is an umbrella that also
# matches unsigned ops; ``uint`` matches only ops the IR marks unsigned.
OP_DTYPES = frozenset({"int", "uint", "half", "bfloat16", "float", "double", "any"})

_OPLIB_DIR = Path(__file__).parent / "oplib"


def _row(
    *,
    op=None,
    mlir_op=None,
    dtype=None,
    width=None,
    width_ge=None,
    width_le=None,
    latency=0,
    delay_ns=None,
    in_delay_ns=None,
    out_delay_ns=None,
    pipelined=None,
    unit=None,
):
    """Build a YAML row dict, omitting unset optionals so the C++ parser's
    ``mapOptional`` sees only present keys."""
    row = {}
    if op is not None:
        row["op"] = op
    if mlir_op is not None:
        row["mlir_op"] = mlir_op
    if dtype is not None:
        row["dtype"] = dtype
    if width is not None:
        row["width"] = int(width)
    if width_ge is not None:
        row["width_ge"] = int(width_ge)
    if width_le is not None:
        row["width_le"] = int(width_le)
    row["latency"] = int(latency)
    if delay_ns is not None:
        row["delay_ns"] = float(delay_ns)
    if in_delay_ns is not None:
        row["in_delay_ns"] = float(in_delay_ns)
    if out_delay_ns is not None:
        row["out_delay_ns"] = float(out_delay_ns)
    if pipelined is not None:
        row["pipelined"] = bool(pipelined)
    if unit is not None:
        if int(latency) <= 0:
            raise ValueError(
                f"unit {unit!r} requires latency > 0 "
                "(a combinational unit is not allocation-limited)"
            )
        row["unit"] = unit
    return row


def _check_kind(kind):
    if kind not in OP_KINDS:
        raise ValueError(
            f"unknown operator kind {kind!r}; expected one of {sorted(OP_KINDS)}"
        )


def _check_dtype(dtype):
    if dtype is not None and dtype not in OP_DTYPES:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(OP_DTYPES)}"
        )


class OperatorLibrary:
    """A device operator timing library. Build it fluently, load it from YAML,
    or pick a shipped built-in; pass it to :func:`allo.backend.rtl.schedule`."""

    def __init__(self, *, device=None, frequency_mhz=None, cycle_time_ns=None):
        self.device = device
        self.frequency_mhz = frequency_mhz
        self.cycle_time_ns = cycle_time_ns
        self._units = {}
        self._operators = []
        self._advanced = []
        self._default = {"latency": 0, "delay_ns": 0.0}

    # -- building -----------------------------------------------------------

    def unit(self, name, count):
        """Declare an allocation pool of ``count`` units, referenced by an
        operator row's ``unit=name``. Ops sharing a pool bound ResII. Returns
        ``self``."""
        self._units[name] = int(count)
        return self

    def op(
        self,
        kind,
        *,
        dtype=None,
        width=None,
        width_ge=None,
        width_le=None,
        latency=0,
        delay_ns=None,
        in_delay_ns=None,
        out_delay_ns=None,
        pipelined=None,
        unit=None,
    ):
        """Characterize an abstract operator ``kind`` (optionally gated by
        ``dtype`` and a width predicate). ``unit`` joins an allocation pool.
        Returns ``self`` for chaining."""
        _check_kind(kind)
        _check_dtype(dtype)
        self._operators.append(
            _row(
                op=kind,
                dtype=dtype,
                width=width,
                width_ge=width_ge,
                width_le=width_le,
                latency=latency,
                delay_ns=delay_ns,
                in_delay_ns=in_delay_ns,
                out_delay_ns=out_delay_ns,
                pipelined=pipelined,
                unit=unit,
            )
        )
        return self

    def advanced(
        self,
        mlir_op,
        *,
        dtype=None,
        width=None,
        width_ge=None,
        width_le=None,
        latency=0,
        delay_ns=None,
        in_delay_ns=None,
        out_delay_ns=None,
        pipelined=None,
        unit=None,
    ):
        """Escape hatch: characterize a raw MLIR op (e.g. ``"math.rsqrt"``) that
        has no abstract kind. Matched before abstract rows; the name is validated
        against registered ops by the pass. Returns ``self``."""
        _check_dtype(dtype)
        self._advanced.append(
            _row(
                mlir_op=mlir_op,
                dtype=dtype,
                width=width,
                width_ge=width_ge,
                width_le=width_le,
                latency=latency,
                delay_ns=delay_ns,
                in_delay_ns=in_delay_ns,
                out_delay_ns=out_delay_ns,
                pipelined=pipelined,
                unit=unit,
            )
        )
        return self

    def default(
        self,
        *,
        latency=0,
        delay_ns=None,
        in_delay_ns=None,
        out_delay_ns=None,
        pipelined=None,
    ):
        """Set the catch-all row applied to any op no other row matches."""
        self._default = _row(
            latency=latency,
            delay_ns=delay_ns,
            in_delay_ns=in_delay_ns,
            out_delay_ns=out_delay_ns,
            pipelined=pipelined,
        )
        return self

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data):
        """Build from a plain dict (the parsed YAML). Validates kinds/dtypes."""
        data = data or {}
        lib = cls(
            device=data.get("device"),
            frequency_mhz=data.get("frequency_mhz"),
            cycle_time_ns=data.get("cycle_time_ns"),
        )
        lib._units = dict(data.get("units", {}))
        for r in data.get("operators", []):
            _check_kind(r.get("op"))
            _check_dtype(r.get("dtype"))
            lib._operators.append(dict(r))
        for r in data.get("advanced_operators", []):
            _check_dtype(r.get("dtype"))
            lib._advanced.append(dict(r))
        if "default" in data:
            lib._default = dict(data["default"])
        return lib

    @classmethod
    def from_yaml(cls, path):
        """Load an editable library from a YAML file."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def builtin(cls, name):
        """Load a shipped built-in library by name (see ``oplib/*.yaml``)."""
        path = _OPLIB_DIR / f"{name}.yaml"
        if not path.exists():
            avail = sorted(p.stem for p in _OPLIB_DIR.glob("*.yaml"))
            raise ValueError(
                f"unknown built-in operator library {name!r}; available: {avail}"
            )
        return cls.from_yaml(path)

    # -- serialization ------------------------------------------------------

    def cycle_time(self):
        """Resolve the target cycle time (ns): ``cycle_time_ns`` if set, else
        ``1000 / frequency_mhz``, else ``None``."""
        if self.cycle_time_ns is not None:
            return float(self.cycle_time_ns)
        if self.frequency_mhz:
            return 1000.0 / float(self.frequency_mhz)
        return None

    def to_dict(self):
        self._validate_units()
        d = {}
        if self.device is not None:
            d["device"] = self.device
        if self.frequency_mhz is not None:
            d["frequency_mhz"] = float(self.frequency_mhz)
        if self.cycle_time_ns is not None:
            d["cycle_time_ns"] = float(self.cycle_time_ns)
        if self._units:
            d["units"] = dict(self._units)
        if self._operators:
            d["operators"] = self._operators
        if self._advanced:
            d["advanced_operators"] = self._advanced
        d["default"] = self._default
        return d

    def _validate_units(self):
        for r in self._operators + self._advanced:
            u = r.get("unit")
            if u is not None and u not in self._units:
                raise ValueError(
                    f"operator row references undeclared unit {u!r}; "
                    f"declare it with .unit({u!r}, count)"
                )

    def to_yaml(self, path=None):
        """Serialize to YAML. Writes to ``path`` and returns it, or returns the
        YAML string when ``path`` is ``None``."""
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is None:
            return text
        Path(path).write_text(text)
        return path

    def __repr__(self):
        return (
            f"OperatorLibrary(device={self.device!r}, "
            f"operators={len(self._operators)}, "
            f"advanced={len(self._advanced)})"
        )
