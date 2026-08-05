# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predicted FPGA area of an emitted design: count the structures, price them
against the device.

This is P7: the scoreboard S3 has to be argued from. `report.py` says what a
schedule COSTS in the units the compiler already counts (cycles, flip-flops,
instances); this says what it costs in the units a device is actually spent in,
so a change to the allocation objective can be checked against something other
than its own currency.

What is left here is the half only a reader of the emitted IR can do: find the
register chains, recognize the one-hot multiplexer cones, count each array's
write ports, and decide which structure the synthesizer will build. What each
structure COSTS is the device's own declaration
(`allo/backend/rtl/area_tables.py`), evaluated through the compiler's one cost
evaluator. Until this split there were two measured models running in parallel,
and the two had already drifted.

The tables are MEASURED, not estimated: Vivado 2023.2, `xcu55c-fsvh2892-2L-e`,
out-of-context synthesis of one DUT per (kind, width) and one Xilinx
Floating-Point core per device operator at its declared latency, primitives
counted off the netlist. `drafts/p6-area/` holds the harness and the raw CSVs.

Three properties of the model that a reader has to know before quoting a number:

  - **The one-hot multiplexer is priced as a STRUCTURE, not as its operations.**
    `EmitContext::oneHotSelect` emits `or(and(v, replicate(sel)))`, and a LUT6
    absorbs three (data, select) pairs, so synthesis fuses the whole cone.
    Pricing the `and`s and the `or` separately over-counts it about fivefold,
    which is exactly the term an allocation objective is most sensitive to.
  - **Accuracy, against real synthesis of four bed kernels** (`validate.py` in
    `drafts/p6-area/`): DSP is EXACT, LUT lands between 0.86x and 1.54x, and
    flip-flops between 1.01x and 1.20x. It walks the IR and so cannot see LUT
    fusion or constant folding, which is why it reads high; SRL extraction it
    reads low, since the synthesizer splits chains this treats as tapped. Use it
    for COMPARING two schedules, which is what it is for, not as a utilization
    estimate.
  - **Memory is priced by its WRITE PORT COUNT, and the cliff is enormous.** One
    writer infers a block RAM and costs no fabric at all; two writers sharing an
    always block infer nothing, so the array becomes a register file with a data
    multiplexer in front of every word. Measured at 512x32: one BRAM18 against
    33,245 LUTs and 16,416 flip-flops. The TEMPLATE decides and not the port
    count, so two writers each in their own block are a true dual port and free
    again, which the emitter marks with `allo.mem.independent_writes`. Block RAM
    is still reported apart from the fabric totals, since no scheduling decision
    trades one for the other.

The one result here that contradicts the compiler: **a value delay chain deeper
than three does not cost flip-flops.** Vivado extracts it into SRLs, so its cost
is about `w` SRL sites plus `2w` flip-flops and is nearly INDEPENDENT of depth,
against the `depth * width` flip-flops the scheduling objective's register term
charges. Measured on both a DUT sweep and a whole kernel; the device declares it
as `dcp.chain`, and two terms of the measurement do not survive that
declaration, both as under-counts (see `set_chain_uses` in `area_tables.py`).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from allo.backend.rtl.device import CombKind, builtin_device

# The `comb` op each device operator kind prices. The device characterizes "an
# integer add", not `addi` against `subi`, so several ops share a row; a cast is
# WIRING and reaches no cell, which is why `comb.extract` and friends are in
# `_FREE` rather than here.
COMB_KIND = {
    "comb.and": CombKind.AND, "comb.or": CombKind.OR, "comb.xor": CombKind.XOR,
    "comb.mux": CombKind.SELECT,
    "comb.add": CombKind.ADD, "comb.sub": CombKind.SUB,
    "comb.icmp": CombKind.CMP,
    "comb.shl": CombKind.SHL, "comb.shru": CombKind.SHR,
    "comb.shrs": CombKind.SHR,
    "comb.mul": CombKind.MUL,
    "comb.divu": CombKind.DIV, "comb.divs": CombKind.DIV,
    "comb.modu": CombKind.REM, "comb.mods": CombKind.REM,
}

def _register_file(device):
    """What an array that failed RAM inference falls back to: every word gets a
    data multiplexer and a write decode, which is what a complete partition
    builds too. The device's `is_scatter` row, since the compiler names no
    storage of its own and neither does this scoreboard."""
    row = next((s for s in device.storage.values() if s.is_scatter), None)
    if row is None:
        raise ValueError(
            f"device {device.name!r} marks no scatter storage, so there is "
            "nothing to price an array that failed RAM inference against"
        )
    return row.uses


# --- reading the emitted design ---------------------------------------------

_ASSIGN = re.compile(r"^\s*(%[\w.$]+)\s*=\s*(\S+)\s*(.*)$")
_STMT = re.compile(r"^\s*(\S+)\s+(.*)$")
_WIDTH = re.compile(r"\bi(\d+)\b")
_INSTANCE = re.compile(r'hw\.instance\s+"[^"]*"\s+@([\w$.]+)')
_HLMEM = re.compile(r"seq\.hlmem\s+@\S+.*<([\dx]+)x?i(\d+)>")
#: The emitter's promise that no two write ports touch one word in a cycle, so
#: the lowering gives each its own always block and a true dual port infers.
_INDEPENDENT = "allo.mem.independent_writes"
_OPERANDS = re.compile(r"%[\w.$]+")

# Everything with no area of its own: declarations, constants, wiring, and the
# module scaffolding.
_FREE = {
    "hw.constant", "hw.module", "hw.module.extern", "hw.output", "hw.instance",
    "seq.to_clock", "seq.from_clock", "seq.hlmem", "seq.read", "seq.write",
    "sv.namehint", "comb.extract", "comb.concat", "comb.replicate",
    "builtin.unrealized_conversion_cast",
}


@dataclass
class _Op:
    name: str
    operands: list[str]
    width: int
    line: str
    result: str | None = None


def _parse(mlir: str) -> tuple[dict[str, _Op], list[_Op]]:
    """SSA name -> defining op, and every op in order.

    Names are qualified by the module they are in, because an SSA name is
    module-scoped: two instantiations of one sub-kernel each declare `%temp`,
    and reading them as one array charged `merge_sort` a second writer it does
    not have, and let a register chain link across the module boundary."""
    defs: dict[str, _Op] = {}
    ops: list[_Op] = []
    module = 0
    for line in mlir.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s in ("}", "{"):
            continue
        m = _ASSIGN.match(s)
        if m:
            res, opname, rest = m.group(1), m.group(2), m.group(3)
        else:
            m = _STMT.match(s)
            if not m:
                continue
            res, opname, rest = None, m.group(1), m.group(2)
        if not re.match(r"^(comb|seq|hw|sv)\.", opname):
            continue
        if opname.startswith("hw.module"):
            module += 1
        scope = f"{module}:"
        widths = _WIDTH.findall(rest)
        # The trailing type is the one that prices the op: an `icmp` returns i1
        # but costs its OPERAND width, and that is what the last type spells.
        width = int(widths[-1]) if widths else 0
        op = _Op(opname, [scope + v for v in _OPERANDS.findall(rest)], width, s,
                 scope + res if res else None)
        ops.append(op)
        if res:
            defs[op.result] = op
    return defs, ops


def _chains(defs: dict[str, _Op], ops: list[_Op]) -> list[tuple[int, int, list]]:
    """Maximal `seq.compreg` chains, as (depth, width, [ops]).

    A stage extends its predecessor only when that predecessor feeds NOTHING
    else: a tapped stage has to keep a real flip-flop, which is also what stops
    the synthesizer extracting an SRL across the tap."""
    uses: dict[str, int] = {}
    for op in ops:
        for v in op.operands:
            uses[v] = uses.get(v, 0) + 1

    regs = [op for op in ops if op.name == "seq.compreg"]
    # A compreg's first operand is its data input; the rest are clock and reset.
    prev: dict[int, _Op] = {}
    for op in regs:
        if not op.operands:
            continue
        src = defs.get(op.operands[0])
        if (src is not None and src.name == "seq.compreg"
                and src.width == op.width and uses.get(src.result, 0) == 1):
            prev[id(op)] = src

    tails = {id(o) for o in regs} - {id(s) for s in prev.values()}
    out = []
    for op in regs:
        if id(op) not in tails:
            continue
        run = [op]
        cur = op
        while id(cur) in prev:
            cur = prev[id(cur)]
            run.append(cur)
        out.append((len(run), op.width, run))
    return out


def _onehot_muxes(defs: dict[str, _Op], ops: list[_Op]) -> dict[int, int]:
    """`comb.or` ops that are a one-hot select, as id(op) -> source count.

    The shape `oneHotSelect` builds: every operand is a `comb.and` against a
    `comb.replicate` of a 1-bit select, or against the select directly when the
    port is one bit wide."""
    found = {}
    for op in ops:
        if op.name != "comb.or" or len(op.operands) < 2:
            continue
        replicated = 0
        for v in op.operands:
            d = defs.get(v)
            if d is None or d.name != "comb.and" or len(d.operands) != 2:
                break
            if any(defs.get(a, _Op("", [], 0, "")).name == "comb.replicate"
                   for a in d.operands):
                replicated += 1
        else:
            # Every operand is a 2-input `and`. At width 1 there is no
            # `replicate` to find, so accept it; wider, require the mask.
            if replicated == len(op.operands) or op.width == 1:
                found[id(op)] = len(op.operands)
    return found


def _operator_costs(device) -> dict[str, tuple]:
    """Each priced operator's declared cost and the operand width it is a
    function of. The width is fixed by the IP's signature, which is why the
    declaration is a constant, but it is still the parameter its kind carries."""
    out = {}
    for op in device.operators:
        uses = device.operator_uses.get(op.func_name)
        if uses:
            widths = [a.primitive_width for a in op.parse_argument_annotations()]
            out[op.func_name] = (uses, max(widths))
    return out


def score(mlir: str, device=builtin_device) -> dict:
    """Predicted area of the emitted design `mlir` (the `hw` dialect module),
    priced against `device`. Resource names are the device's own, so the totals
    below hold whatever a part calls its primitives."""
    defs, ops = _parse(mlir)
    muxes = _onehot_muxes(defs, ops)
    consumed = set()
    for op in ops:
        if id(op) in muxes:
            for v in op.operands:
                consumed.add(id(defs[v]))  # the `and`s belong to the mux

    price = device.price
    ip_costs = _operator_costs(device)
    register_file = _register_file(device)

    # Registers are priced per CHAIN, not per stage: past the extraction cliff
    # the chain is an SRL and the flip-flop count stops tracking depth.
    chains = _chains(defs, ops)
    regs: Counter = Counter()
    chain_stages = 0
    deep_stages = 0
    for depth, width, run in chains:
        spent = price(device.chain_uses, (depth, width))
        regs.update(spent)
        chain_stages += depth
        # "Deep" is whatever the device charges SLICEM for: the extraction
        # threshold is the part's, not this reader's.
        if spent.get("slicem_lut"):
            deep_stages += depth
        for o in run:
            consumed.add(id(o))

    # Write ports per array, off the memory each `seq.write` names as its first
    # operand: one writer is a RAM, two share it only if independent.
    writers: dict[str, int] = {}
    for op in ops:
        if op.name == "seq.write":
            writers[op.operands[0]] = writers.get(op.operands[0], 0) + 1

    total = Counter(regs)
    datapath: Counter = Counter()
    ip: Counter = Counter()
    mux_total: Counter = Counter()
    mem_fabric: Counter = Counter()
    n_mux = 0
    mux_sources = 0
    instances: dict[str, int] = {}
    mem_bits = 0
    regfile_arrays = 0
    unmodelled: dict[str, int] = {}

    for op in ops:
        if id(op) in consumed:
            continue
        if id(op) in muxes:
            k = muxes[id(op)]
            a = price(device.mux_uses, (k, op.width))
            mux_total.update(a)
            total.update(a)
            n_mux += 1
            mux_sources += k
            continue
        if op.name == "seq.compreg":
            continue  # priced above, as part of its chain
        if op.name == "hw.instance":
            m = _INSTANCE.search(op.line)
            assert m, f"an instance names a module: {op.line}"
            mod = m.group(1)
            instances[mod] = instances.get(mod, 0) + 1
            # The module name is an operator stem plus whatever else
            # distinguishes the hardware (a compare's predicate).
            cost = ip_costs.get(mod) or next(
                (c for stem, c in ip_costs.items() if mod.startswith(stem + "_")),
                None,
            )
            if cost is None:
                unmodelled[mod] = unmodelled.get(mod, 0) + 1
                continue
            a = price(cost[0], (cost[1],))
            ip.update(a)
            total.update(a)
            continue
        if op.name == "seq.hlmem":
            m = _HLMEM.search(op.line)
            if m:
                shape = [int(x) for x in m.group(1).split("x") if x]
                depth = 1
                for d in shape:
                    depth *= d
                width = int(m.group(2))
                n = writers.get(op.result, 0)
                # A single writer is the RAM template the synthesizer
                # recognizes, so it costs no fabric; extra READ ports only
                # replicate the RAM. Two writers still infer a TRUE dual port
                # when each is described in its own always block; sharing a
                # block, or asking for a third port, infers nothing and the
                # array falls back to a register file.
                assert depth >= 1 and width >= 1 and n >= 0
                if n <= 1 or (n == 2 and _INDEPENDENT in op.line):
                    mem_bits += depth * width
                else:
                    a = price(register_file, (depth, width))
                    mem_fabric.update(a)
                    total.update(a)
                    regfile_arrays += 1
            continue
        if op.name in _FREE:
            continue
        kind = COMB_KIND.get(op.name)
        if kind is None:
            unmodelled[op.name] = unmodelled.get(op.name, 0) + 1
            continue
        a = price(device.comb_uses.get(kind.value, ()), (op.width,))
        datapath.update(a)
        total.update(a)

    return {
        # `lut` is every LUT site the design occupies, SRLs included, since an
        # SRL is a LUT that happens to hold state. The device counts the two
        # apart because only a SLICEM LUT can be one.
        "lut": total["lut"] + total["slicem_lut"],
        "logic_only_lut": total["lut"],
        "srl": total["slicem_lut"], "ff": total["ff"], "dsp": total["dsp"],
        "carry8": total["carry8"],
        "ip_lut": ip["lut"], "ip_ff": ip["ff"], "ip_dsp": ip["dsp"],
        "mux_lut": mux_total["lut"],
        "logic_lut": datapath["lut"], "logic_carry8": datapath["carry8"],
        "reg_ff": regs["ff"], "reg_lut": regs["lut"] + regs["slicem_lut"],
        "mem_lut": mem_fabric["lut"], "mem_ff": mem_fabric["ff"],
        # Arrays that failed RAM inference and became a register file, which is
        # not the same as arrays with several writers: two independent ports
        # still infer a true dual port.
        "regfile_arrays": regfile_arrays,
        # What the scheduling objective charges for the same registers today.
        "reg_ff_modelled": sum(d * w for d, w, _ in chains),
        "chain_stages": chain_stages, "deep_chain_stages": deep_stages,
        "muxes": n_mux, "mux_sources": mux_sources,
        "instances": sum(instances.values()),
        "mem_bits": mem_bits,
        "unmodelled": unmodelled,
    }
