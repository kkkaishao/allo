# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the RTL tests.

The named latencies (``FADD``, ``FMUL``, ...) are read from the shipped built-in
device, the one the RTL backend uses by default, so the II assertions read as the
recurrence arithmetic they check while tracking the device's real numbers.
"""

from __future__ import annotations

import re

from allo.backend.rtl import RTL, ScheduleResult, MemoryKind, builtin_device


# Operator latencies keyed by (kind, arg bit width), read off the built-in
# operator IPs (each an `@ip(optype=...)`).
def _key(op):
    dt = op.parse_argument_annotations()[0]
    return (op.optype.value, int(dt.primitive_width))


_LAT = {_key(o): o.timing.latency for o in builtin_device.operators}

FADD = FSUB = _LAT[("add", 32)]  # floating-point add/sub latency (cycles)
FMUL = _LAT[("mul", 32)]  # floating-point multiply latency
FDIV = _LAT[("div", 32)]  # floating-point divide latency
IMUL = 0  # integer multiply is combinational (latency 0)
MEM = builtin_device.memory[MemoryKind.LUTRAM].read_latency  # default read/write
MEM_URAM = builtin_device.memory[MemoryKind.URAM].read_latency

# Combinational delay in ns by op kind, the table the chaining scheduler cuts
# against, and the default clock it cuts to. A test that picks a clock to make a
# chain fit or not fit derives the period from these rather than restating the
# device's numbers.
COMB = builtin_device.comb
PERIOD_NS = 1000.0 / builtin_device.default_freq_mhz

# A memory-carried accumulate (`M[x] += ...`) closes a distance-1 recurrence
# read -> add -> write, so its II is the sum; a scalar-carried accumulate keeps
# the partial in a register, so its II is just the add latency.
MEM_REDUCE_II = MEM + FADD + MEM


def _to_rtl(kernel, **kw) -> RTL:
    """Export ``kernel`` to the RTL backend."""
    return kernel.schedule().export("rtl", **kw)


def _sched(kernel, **kw) -> ScheduleResult:
    """Schedule ``kernel`` through the RTL backend."""
    return _to_rtl(kernel, **kw).schedule()


def _latency(kernel, **kw):
    """Whole-kernel latency (cycles) of ``kernel`` scheduled on its own; ``None``
    when a trip count is not statically known."""
    return _sched(kernel, **kw).func(kernel.__name__).latency


def _iis(regions):
    """Sorted IIs of ``regions``; a dynamic-trip sequential wrapper (``ii`` is
    ``None``) is skipped."""
    return sorted(r.ii for r in regions if r.ii is not None)


# --- structural reading of the emitted RTL -----------------------------------

_DEF = re.compile(r"^%([\w.$-]+) = (.+)$")
_COMPREG = re.compile(r'^seq\.compreg (?:name "([^"]*)" )?%([\w.$-]+),')
_MUX = re.compile(r"^comb\.mux (?:bin )?%([\w.$-]+), %([\w.$-]+), %([\w.$-]+)")
_HINT = re.compile(r'sv\.namehint = "([^"]+)"')
_OPERAND = re.compile(r"%([\w.$-]+)")


class Mod:
    """The ops of one ``hw.module``, indexed for structural assertions.

    Text-level rather than a real parse: the tests that use it are locks on the
    *shape* of a small, named piece of the emitted hardware (a stall shell, a
    controller), so what they need is the def of each SSA value, its namehint,
    and the register list rather than an IR data structure.
    """

    def __init__(self, mlir, name):
        body, seen = [], False
        for line in mlir.splitlines():
            s = line.strip()
            if s.startswith(f"hw.module @{name}("):
                seen = True
                continue
            if seen:
                if s == "}":
                    break
                body.append(s)
        assert seen, f"no hw.module @{name} in the emitted module"
        # The module body verbatim, for what the per-op index cannot hold: a
        # multi-result op (an `hw.instance`) defines no single value and so has
        # no entry in `defs`.
        self.text = "\n".join(body)
        self.defs, self.hint, self.regs = {}, {}, []
        for s in body:
            m = _DEF.match(s)
            if not m:
                continue
            res, rhs = m.group(1), m.group(2)
            self.defs[res] = rhs
            h = _HINT.search(rhs)
            if h:
                self.hint[res] = h.group(1)
            r = _COMPREG.match(rhs)
            if r:
                self.regs.append((r.group(1) or res, res, r.group(2)))

    def hinted(self, name):
        """The single SSA value labelled ``sv.namehint = name``."""
        hits = [v for v, h in self.hint.items() if h == name]
        assert len(hits) == 1, f"expected one {name!r}, got {hits}"
        return hits[0]

    def hints_like(self, pattern):
        return sorted({h for h in self.hint.values() if re.search(pattern, h)})

    def signal(self, name):
        """The SSA value carrying ``name``.

        A named register prints AS its name (CIRCT takes the SSA name from
        ``seq.compreg``'s ``name`` attribute); named combinational logic carries
        an ``sv.namehint`` instead. Callers of a control signal should not have
        to know which of the two the emitter happened to build.
        """
        return name if name in self.defs else self.hinted(name)

    def regions_with(self, suffix):
        """The region ids for which an ``r<N>_<suffix>`` signal exists."""
        pat = re.compile(rf"^r(\d+)_{suffix}$")
        names = set(self.defs) | set(self.hint.values())
        return sorted(int(m.group(1)) for m in map(pat.match, names) if m)

    def operands(self, v):
        return _OPERAND.findall(self.defs.get(v, ""))

    def mux(self, v):
        """``(sel, t, f)`` of ``v`` when it is a 2:1 mux, else ``None``."""
        m = _MUX.match(self.defs.get(v, ""))
        return m.groups() if m else None

    def enable_of(self, reg, inp):
        """The enable selecting ``reg``'s next value, or None if unconditional.

        An enabled cell is ``reg = compreg(mux(en, in, reg))``, so the register
        holds itself on the false arm. Both ``enabledReg`` (a chain stage) and
        ``stallHold`` (a held address) emit exactly this pair; which node is
        called the output is the only difference, and it does not matter here.
        """
        m = self.mux(inp)
        return m[0] if m and m[2] == reg else None

    def cone(self, root, limit=64):
        """The SSA values reachable from ``root`` through comb logic.

        Leaves are module ports, instance results and registers, anything with
        no combinational def in this module. They are IN the result, since
        "does this signal reach `start`" is exactly the sort of question a
        control-structure lock asks.
        """
        seen, work = set(), [root]
        while work and len(seen) < limit:
            v = work.pop()
            if v in seen:
                continue
            seen.add(v)
            rhs = self.defs.get(v, "")
            if not rhs or rhs.startswith("seq."):  # a register ends the cone
                continue
            work += _OPERAND.findall(rhs)
        return seen

    def reg_named(self, label):
        hits = [(r, i) for lb, r, i in self.regs if lb == label]
        assert len(hits) == 1, f"expected one {label!r} register, got {hits}"
        return hits[0]


def _one_region(m):
    """The single done-driven region of `m` (the one that emits an `r<N>_fire`)."""
    ids = m.regions_with("fire")
    assert len(ids) == 1, f"expected one done-driven region, got {ids}"
    return ids[0]


def _hold_done(m, region):
    """The set-pulse of region `region`'s done latch.

    `holdDone` is `done = compreg(mux(start, false, mux(set, true, done)))`:
    cleared by the region start so a retriggered region re-edges, set by the
    completion pulse. Returns `set`, having checked the shape.
    """
    reg, inp = m.reg_named(f"r{region}_done")
    clear = m.mux(inp)
    assert clear and clear[1].startswith("false"), f"r{region}_done not cleared: {inp}"
    hold = m.mux(clear[2])
    assert (
        hold and hold[1].startswith("true") and hold[2] == reg
    ), f"r{region}_done is not a hold latch: {clear[2]}"
    return hold[0]
