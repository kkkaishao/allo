# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predicted FPGA area of an emitted design, from measured device tables.

This is P7: the scoreboard S3 has to be argued from. `report.py` says what a
schedule COSTS in the units the compiler already counts (cycles, flip-flops,
instances); this says what it costs in the units a device is actually spent in,
so a change to the allocation objective can be checked against something other
than its own currency.

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
charges. Measured on both a DUT sweep and a whole kernel; see `chain_area`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

# --- measured device area (P6, xcu55c) --------------------------------------


@dataclass(frozen=True)
class Area:
    """Physical resources. Kept as a VECTOR: a scalar cannot rank an f32
    divider (766 LUT, no DSP) against an f64 multiplier (205 LUT, 7 DSP), which
    is the mistake `AllocatableUnit::cost` makes today."""

    lut: int = 0
    ff: int = 0
    dsp: int = 0
    carry8: int = 0
    #: Shift-register LUTs. Their own field because they occupy LUT sites but
    #: only in SLICEM, so they are neither free nor interchangeable with logic.
    srl: int = 0

    def __add__(self, o: "Area") -> "Area":
        return Area(self.lut + o.lut, self.ff + o.ff, self.dsp + o.dsp,
                    self.carry8 + o.carry8, self.srl + o.srl)

    def __mul__(self, n: int) -> "Area":
        return Area(self.lut * n, self.ff * n, self.dsp * n,
                    self.carry8 * n, self.srl * n)


ZERO = Area()

# Comb operators, LUTs as a function of operand width. Each is the measured
# shape, not a fitted curve: `and`/`or`/`xor`/`mux` are exactly w, an adder adds
# a carry chain, a shift is a barrel (about w*ceil(log4 w)), and a divider is
# quadratic. A cast is WIRING and costs nothing at all.
def _logic(w: int) -> Area:
    return Area(lut=w)


def _addsub(w: int) -> Area:
    return Area(lut=w, carry8=math.ceil(w / 8))


def _cmp(w: int) -> Area:
    return Area(lut=w, carry8=math.ceil(w / 16))


def _shift(w: int) -> Area:
    return Area(lut=w * max(1, math.ceil(math.log(w, 4))))


def _mul(w: int) -> Area:
    # Measured 1/3/10 DSP48E2 at w=16/32/64, with a little glue; below 18 bits
    # one DSP holds it, above that the partial products multiply up.
    dsp = {8: 0, 16: 1, 32: 3, 64: 10}.get(w, max(1, math.ceil((w / 18) ** 2)))
    return Area(lut=15 if w >= 32 else 39, dsp=dsp, carry8=math.ceil(w / 16))


def _div(w: int) -> Area:
    # Measured 75/286/1086 LUTs at w=8/16/32: about 1.06*w^2.
    return Area(lut=round(1.06 * w * w), carry8=5 * w)


COMB_AREA = {
    "comb.and": _logic, "comb.or": _logic, "comb.xor": _logic,
    "comb.mux": _logic,
    "comb.add": _addsub, "comb.sub": _addsub,
    "comb.icmp": _cmp,
    "comb.shl": _shift, "comb.shru": _shift, "comb.shrs": _shift,
    "comb.mul": _mul,
    "comb.divu": _div, "comb.divs": _div, "comb.modu": _div, "comb.mods": _div,
    # Pure wiring: a rename of bits, which synthesis charges nothing for.
    "comb.extract": lambda w: ZERO, "comb.concat": lambda w: ZERO,
    "comb.replicate": lambda w: ZERO,
}

# The device operator IPs, each the Xilinx Floating-Point core at the latency
# `device.py` declares. `sym_name` is the STEM of the module name, so a
# compare arrives with its predicate appended (`fcmp_l1_ogt`).
IP_AREA = {
    "fadd_l7": Area(247, 315, 2, 10),
    "fsub_l7": Area(247, 315, 2, 10),
    "fmul_l4": Area(115, 173, 2, 9),
    "fdiv_l12": Area(766, 1381, 0, 111),
    "fcmp_l1": Area(64, 12, 0, 7),
    "dadd_l14": Area(710, 872, 3, 30),
    "dsub_l14": Area(710, 872, 3, 30),
    "dmul_l9": Area(205, 542, 7, 16),
    "ddiv_l24": Area(3185, 6035, 0, 399),
    "dcmp_l1": Area(118, 12, 0, 12),
    "i2f_l3": Area(165, 228, 0, 11),
    "f2i_l3": Area(183, 232, 0, 6),
    "fcvt_l2": Area(50, 99, 0, 1),
    # bf16 has no measured core; priced from its f32 sibling by width.
    "bfadd_l4": Area(124, 158, 1, 5),
    "bfsub_l4": Area(124, 158, 1, 5),
    "bfmul_l2": Area(58, 87, 1, 5),
    "bf2f_l2": Area(25, 50, 0, 1),
}


def mux_lut_per_bit(k: int) -> int:
    """LUTs per bit of a `k`-source one-hot AND-OR select.

    A LUT6 absorbs three (data, select) pairs and about 2.5 more per further
    level, so this is LINEAR in k rather than logarithmic. Fits all ten
    measured points (k = 2..40) exactly. `muxLevels(k) = ceil(log2 k)` prices
    the DELAY of the same structure and says nothing about its area."""
    assert k >= 1, "a select over nothing is not a value"
    if k == 1:
        return 0  # one source is a wire
    return 1 if k <= 3 else 1 + math.ceil((k - 3) / 2.5)


def mux_area(k: int, width: int) -> Area:
    return Area(lut=mux_lut_per_bit(k) * width)


#: Below this depth a chain stays in flip-flops; at or above it Vivado extracts
#: an SRL, even though the emitter resets every stage. Measured exactly here.
SRL_MIN_DEPTH = 4
#: A one-bit chain is left in flip-flops whatever its depth. Measured at
#: w = 1, 8 and 32, so the threshold itself is only bracketed.
SRL_MIN_WIDTH = 8


def chain_area(depth: int, width: int) -> Area:
    """A `depth`-stage, `width`-bit value delay chain.

    Deep chains are SRLs, so the cost is about `width` SRL sites plus `2*width`
    flip-flops and barely moves with depth: at w=32 the measured flip-flop count
    is 67 at depth 4 and 127 at depth 64, against the `depth*width` (128 and
    2048) that `RegisterTerm` charges. This is the single largest disagreement
    between the objective's area model and the part."""
    assert depth >= 1 and width >= 1
    if depth < SRL_MIN_DEPTH or width < SRL_MIN_WIDTH:
        return Area(ff=depth * width)
    # An SRL32E holds 32 stages, plus one LUT per bit of addressing and output
    # multiplexing, and the head and tail stages stay in flip-flops.
    return Area(lut=width, srl=width * math.ceil(depth / 32),
                ff=2 * width + depth - 1)


#: Fabric LUTs per bit of a memory that failed RAM inference. Every word needs a
#: data multiplexer and a write decode, so it scales with the whole array.
#: Measured 1.6x to 3.3x of `depth*width` over 64..512 deep and 8..32 wide.
MULTIWRITE_LUT_PER_BIT = 2.0


def memory_area(depth: int, width: int, writers: int,
                independent: bool) -> tuple[Area, int]:
    """Fabric cost of one array, and the bits that went to block RAM instead.

    A single writer is the RAM template the synthesizer recognizes, so it costs
    no fabric; extra READ ports only replicate the RAM. Two writers still infer
    a TRUE dual port when each is described in its own always block, which is
    what `independent` reports; sharing a block, or asking for a third port,
    infers nothing and the array falls back to registers."""
    assert depth >= 1 and width >= 1 and writers >= 0
    bits = depth * width
    if writers <= 1 or (independent and writers == 2):
        return ZERO, bits
    return Area(lut=round(MULTIWRITE_LUT_PER_BIT * bits), ff=bits), 0


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


def score(mlir: str) -> dict:
    """Predicted area of the emitted design `mlir` (the `hw` dialect module)."""
    defs, ops = _parse(mlir)
    muxes = _onehot_muxes(defs, ops)
    consumed = set()
    for op in ops:
        if id(op) in muxes:
            for v in op.operands:
                consumed.add(id(defs[v]))  # the `and`s belong to the mux

    # Registers are priced per CHAIN, not per stage: past depth 3 the chain is
    # an SRL and the flip-flop count stops tracking depth.
    chains = _chains(defs, ops)
    reg_area = ZERO
    chain_stages = 0
    deep_stages = 0
    for depth, width, run in chains:
        reg_area += chain_area(depth, width)
        chain_stages += depth
        if depth >= SRL_MIN_DEPTH and width >= SRL_MIN_WIDTH:
            deep_stages += depth
        for o in run:
            consumed.add(id(o))

    # Write ports per array, off the memory each `seq.write` names as its first
    # operand: one writer is a RAM, two share it only if independent.
    writers: dict[str, int] = {}
    for op in ops:
        if op.name == "seq.write":
            writers[op.operands[0]] = writers.get(op.operands[0], 0) + 1

    total = reg_area
    datapath = ZERO
    ip = ZERO
    mux_area_total = ZERO
    mem_fabric = ZERO
    regs = reg_area
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
            a = mux_area(k, op.width)
            mux_area_total += a
            total += a
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
            a = _ip_area(mod)
            if a is None:
                unmodelled[mod] = unmodelled.get(mod, 0) + 1
                continue
            ip += a
            total += a
            continue
        if op.name == "seq.hlmem":
            m = _HLMEM.search(op.line)
            if m:
                shape = [int(x) for x in m.group(1).split("x") if x]
                depth = 1
                for d in shape:
                    depth *= d
                a, ram = memory_area(depth, int(m.group(2)),
                                     writers.get(op.result, 0),
                                     _INDEPENDENT in op.line)
                mem_fabric += a
                total += a
                mem_bits += ram
                regfile_arrays += ram == 0
            continue
        if op.name in _FREE:
            continue
        fn = COMB_AREA.get(op.name)
        if fn is None:
            unmodelled[op.name] = unmodelled.get(op.name, 0) + 1
            continue
        a = fn(op.width)
        datapath += a
        total += a

    return {
        # `lut` is every LUT site the design occupies, SRLs included, since an
        # SRL is a LUT that happens to hold state.
        "lut": total.lut + total.srl, "logic_only_lut": total.lut,
        "srl": total.srl, "ff": total.ff, "dsp": total.dsp,
        "carry8": total.carry8,
        "ip_lut": ip.lut, "ip_ff": ip.ff, "ip_dsp": ip.dsp,
        "mux_lut": mux_area_total.lut,
        "logic_lut": datapath.lut, "logic_carry8": datapath.carry8,
        "reg_ff": regs.ff, "reg_lut": regs.lut + regs.srl,
        "mem_lut": mem_fabric.lut, "mem_ff": mem_fabric.ff,
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


def _ip_area(module: str):
    """The measured area of extern `module`, whose name is an operator stem
    plus whatever else distinguishes the hardware (a compare's predicate)."""
    if module in IP_AREA:
        return IP_AREA[module]
    for stem, a in IP_AREA.items():
        if module.startswith(stem + "_"):
            return a
    return None
