# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""QoR over the benchmark bed: one command, one table, two schedulers.

    python -m benchmark.report                    # both schedulers, compile stage
    python -m benchmark.report --per-region       # the region-level dump
    python -m benchmark.report --compare base.json

This is the measurement half of the bed. `spec.py` says what a benchmark is and
`verify.py` answers whether a variant is CORRECT; this answers what it COSTS,
which is the question a scheduling-model change has to be argued from.

What it reports, and why each number rather than a neighbouring one:

    latency      the kernel's published span, and whether it is EXACT. Only an
                 exact one may be compared against hardware, so a variant whose
                 latency is a bound or whose kernel is indeterminate is carried
                 but never summed into a headline.
    ii           per region, what the solver decided.
    length       the schedule DEPTH: the cycle by which every op has completed.
    drain        the TERMINAL cycle, the last issue pulse to the deepest output
                 committing. Reported beside `length` rather than instead of it
                 because a span composes off `drain` and the two differ by
                 whatever slack the solver left above the last commit, which is
                 a scheduling decision worth seeing.
    reg bits     flip-flops in the emitted Verilog, split into the pulse chains
                 (`r<region>_v<k>`), the value delay chains (`<value>_d<k>`) and
                 everything else. Bits, not registers: a narrower counter is
                 fewer bits on the same declaration line, so counting lines or
                 declarations hides the one axis a schedule can move. Per
                 VARIANT and not per region, because only a pulse chain's name
                 carries the region it belongs to; a value chain's does not, and
                 a split that is right for one role and guessed for the other
                 would read as a per-region number without being one.
    solve ms     wall time of each region's solve, from the scheduler itself.
                 Per REGION, because a whole-compile figure cannot tell a model
                 change that cost 2 ms everywhere from one that cost 4 s once.

Each (benchmark, variant, scheduler) runs in its own subprocess, for the reason
the cosim probe does: a scheduler that does not terminate, an assert that fires
and a solver that runs away all have to be survivable, and only a process
boundary survives all three.

NOT a correctness suite. It stops at `compile`, so nothing here says a variant
computes the right answer; that is `verify.py`'s job and the two should not be
conflated again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark import area

REPO = Path(__file__).resolve().parents[1]
MARK = "@@QOR@@"

# --- register accounting -----------------------------------------------------

# CIRCT's ExportVerilog emits one `reg` declaration per `seq.compreg`, with an
# explicit range for anything wider than a bit and a trailing dimension for a
# memory. Matching the declaration rather than the `always_ff` block is what
# gives the WIDTH, which is the number that matters.
_REG_DECL = re.compile(
    r"^[ \t]*reg[ \t]+(?:\[[ \t]*(\d+)[ \t]*:[ \t]*(\d+)[ \t]*\][ \t]+)?"
    r"(\S+?)[ \t]*(\[[^\]]*\])?[ \t]*;",
    re.M,
)
_ALWAYS_FF = re.compile(r"^[ \t]*always_ff[ \t]*@\(posedge", re.M)
# A region's activation-pulse chain, and a value's delay chain. Both may carry a
# uniquifying suffix, which CIRCT appends when two names collide.
_PULSE_REG = re.compile(r"^r\d+_v\d+(_\d+)?$")
_DELAY_REG = re.compile(r"_d\d+(_\d+)?$")


def count_registers(verilog: str) -> dict:
    """Flip-flops in `verilog`, split by the role their name states.

    `parse_ok` is the parser holding itself to the design: every `reg`
    declaration is driven by one `always_ff`, so a count that disagrees means
    this regex missed a declaration form and the numbers below are not to be
    trusted rather than quietly low."""
    total = delay = pulse = mem = 0
    n_regs = n_mem = 0
    for hi, lo, name, dims in _REG_DECL.findall(verilog):
        width = int(hi) - int(lo) + 1 if hi else 1
        if dims:  # `reg [31:0] cell[0:63]`: storage, not a schedule register
            mem += width
            n_mem += 1
            continue
        total += width
        n_regs += 1
        if _PULSE_REG.match(name):
            pulse += width
        elif _DELAY_REG.search(name):
            delay += width
    return {
        "reg_bits": total,
        "reg_count": n_regs,
        "pulse_bits": pulse,
        "delay_bits": delay,
        "mem_bits": mem,
        "mem_count": n_mem,
        "parse_ok": n_regs + n_mem == len(_ALWAYS_FF.findall(verilog)),
    }


# --- one run -----------------------------------------------------------------

# The scheduler's own report that a region's II exceeds the bound its LP
# justifies, which is the one place the compiler admits it may have lost. It is
# the HEURISTIC's warning, so it fires under `scheduler="exact"` too, which runs
# the heuristic as its warm start; the count is the same in both columns by
# construction and describes the problem rather than the solver.
_II_GAP = re.compile(r"Scheduled at II=(\d+) against a lower bound of II=(\d+)")
# An exact solve that found a schedule but not the best one. Unlike a spent
# budget with nothing in hand, this SHIPS: the region takes a legal placement
# that no bound relates to the heuristic's, so it is the one way the exact path
# can come back worse than the default one.
_BUDGET = re.compile(r"ran out of budget")
# Per cyclic region, how many of its loop-carried memory edges hold a distance
# the polyhedral test PROVED versus one it assumed, and how many `memref`
# accesses were raised into the test's reach on the way. Both need
# ALLO_LOG_LEVEL=info and are absent otherwise; they price what a dependence
# hint, or a better raise, could still recover.
_CARRIED = re.compile(
    r"Carried memory dependences: (\d+) total, (\d+) non-affine, "
    r"(\d+) unknown-distance"
)
_RAISED = re.compile(r"Raised (\d+) loop\(s\) and (\d+) further memref access")
# Per region, what the allocation cost: compute ops, the units they were bound
# to, and the interconnect sharing grew.
_ALLOC = re.compile(
    r"Allocation: (\d+) compute ops on (\d+) units \((\d+) IP\), "
    r"(\d+) muxes, (\d+) mux inputs, (\d+) 2:1 mux bits"
)
# Per array, the write ports it was given and how many REGIONS they came from.
# A second write port defeats RAM inference, so writers spread over regions are
# paying for a concurrency that region ordering rules out.
_MEMPORTS = re.compile(
    r"Memory: (\d+) write ports \((\d+) from calls\) on (\d+)x(\d+) bits "
    r"over (\d+) regions, needs (\d+) write (\d+) total, (\w+)"
)
# Per region, the operator types whose timing row covers several operator
# identities. A binder folds only within one identity, so a limit keyed on the
# type over-approximates the hardware it can build.
_SPLIT = re.compile(
    r"Operator classes: \d+ of \d+ operator types cover several operator "
    r"identities:(.*)"
)
_SPLIT_ONE = re.compile(r"(\S+) (\d+) ops / (\d+) classes")


def _load(key):
    sys.path.insert(0, str(REPO))
    from benchmark.spec import find

    return find(key)


def measure_one(
    key: str,
    variant: str,
    scheduler: str,
    stage: str,
    freq: float | None = None,
    budget: float | None = None,
    binding: str = "trivial",
) -> dict:
    """Schedule (and by default compile) one variant, returning its metrics.

    ``freq`` overrides the device's default clock (MHz), i.e. the period the
    chaining half of every problem is cut against. ``budget`` overrides what one
    exact solve may spend, in deterministic time units. ``binding`` is the
    operator-sharing policy, i.e. how many physical units the schedule is
    realized on."""
    bench = _load(key)
    out: dict = {
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
        "freq_mhz": freq,
        "budget": budget,
        "binding": binding,
        "stage": "build",
        "status": "error",
    }
    t0 = time.time()
    if variant in bench.skip:
        out.update(status="skip", stage="skip", note=bench.skip[variant])
        return out

    try:
        parts = bench.build()
        sched = bench.schedules[variant](parts)

        out["stage"] = "schedule"
        opts = {"scheduler": scheduler, "binding": binding}
        if freq is not None:
            opts["freq_mhz"] = freq
        if budget is not None:
            opts["budget"] = budget
        rtl = sched.export("rtl", **opts)
        t1 = time.time()
        res = rtl.schedule()
        out["schedule_s"] = round(time.time() - t1, 2)

        fn = res.func(rtl.top)
        out["latency"] = fn.latency
        out["latency_exact"] = fn.latency_is_exact
        out["determinacy"] = fn.determinacy
        # Every func, not just the top: a sub-kernel's regions are as much of
        # the hardware, and a schedule change may land entirely in one.
        out["regions"] = [
            {
                "func": f.name,
                "order": r.order,
                "nesting": r.depth,
                "kind": str(r.kind.value),
                "container": r.container,
                "ii": r.ii,
                "length": r.length,
                "drain": r.drain,
                "trip": r.trip,
                "latency": r.latency,
                "ops": len(r.ops),
            }
            for f in res.funcs
            for r in f.regions
        ]
        out["solves"] = [
            {
                "func": s.func,
                "where": s.where,
                "kind": s.kind,
                "ops": s.ops,
                "limited_ops": s.limited_ops,
                "allocated_ops": s.allocated_ops,
                "allocated_units": s.allocated_units,
                "ii": s.ii,
                "ms": round(s.ms, 2),
            }
            for s in res.solves
        ]
        out["solve_ms"] = round(sum(s.ms for s in res.solves), 1)
        out["solve_ms_max"] = round(max((s.ms for s in res.solves), default=0.0), 1)
        out["ops_max"] = max((s.ops for s in res.solves), default=0)

        if stage != "schedule":
            out["stage"] = "compile"
            t1 = time.time()
            rtl.compile()
            out["compile_s"] = round(time.time() - t1, 2)
            # Before `rtl.verilog`, which lowers `seq` to `sv` IN PLACE: the
            # scorer wants `seq.compreg` and the `comb` cones, not their SV
            # expansion.
            out["area"] = area.score(rtl.mlir)
            verilog = rtl.verilog
            out["verilog_lines"] = verilog.count("\n")
            # The RTL itself, so a re-run can be checked for BYTE identity
            # rather than for metrics that agree. Determinism is a property of
            # the emitted hardware, and two schedules can differ while every
            # figure below matches.
            out["verilog_sha"] = hashlib.sha256(verilog.encode()).hexdigest()[:16]
            out.update(count_registers(verilog))
            # The allocation as the emitted hardware states it: one
            # `hw.instance` per IP operator, so a fold drops one. A
            # combinational unit is an expression with no instance to count, so
            # only its muxes show up here.
            hw = rtl.mlir
            out["hw_instances"] = hw.count("hw.instance")
            out["comb_muxes"] = hw.count("comb.mux")

        out["status"] = "pass"
    except BaseException as e:  # a fired assert is a result, not a crash
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        out["seconds"] = round(time.time() - t0, 1)
    return out


def _run_child(
    item,
    stage: str,
    timeout: int,
    freq: float | None,
    budget: float | None,
    binding: str,
) -> dict:
    key, variant, scheduler = item
    env = dict(os.environ)
    env["XILINX_VITIS"] = "/nonexistent"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("ALLO_LOG_LEVEL", "warn")
    cmd = [
        sys.executable,
        "-m",
        "benchmark.report",
        "--one",
        f"{key}::{variant}::{scheduler}",
        "--stage",
        stage,
        "--binding",
        binding,
    ]
    if freq is not None:
        cmd += ["--freq", str(freq)]
    if budget is not None:
        cmd += ["--budget", str(budget)]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(REPO)
        )
    except subprocess.TimeoutExpired:
        return {
            "key": key,
            "variant": variant,
            "scheduler": scheduler,
            "status": "timeout",
            "stage": "?",
            "seconds": round(time.time() - t0, 1),
        }
    text = p.stdout + p.stderr
    for line in p.stdout.splitlines():
        if line.startswith(MARK):
            d = json.loads(line[len(MARK) :])
            # The II-vs-bound warnings, which no field of the schedule result
            # carries: the bound is settled inside the simplex and reported only
            # as a diagnostic.
            d["ii_gaps"] = [
                {"ii": int(a), "bound": int(b)} for a, b in _II_GAP.findall(text)
            ]
            d["budget_exhausted"] = len(_BUDGET.findall(text))
            d["carried_deps"] = [
                [int(x) for x in m] for m in _CARRIED.findall(text)
            ]
            d["raised"] = [
                sum(int(m[i]) for m in _RAISED.findall(text)) for i in (0, 1)
            ]
            # Both need ALLO_LOG_LEVEL=info and are absent at the default level.
            d["alloc"] = [
                dict(
                    zip(("ops", "units", "ip", "muxes", "mux_inputs", "mux_bits"),
                        (int(x) for x in m))
                )
                for m in _ALLOC.findall(text)
            ]
            if d["alloc"]:
                for f in ("ops", "units", "muxes", "mux_bits"):
                    d[f"alloc_{f}"] = sum(a[f] for a in d["alloc"])
            d["mem_ports"] = [
                {"writes": int(w), "from_calls": int(c), "depth": int(dp),
                 "width": int(wd), "regions": int(rg), "needs": int(nd),
                 "needs_total": int(nt), "external": ext == "external"}
                for w, c, dp, wd, rg, nd, nt, ext in _MEMPORTS.findall(text)
            ]
            d["class_splits"] = [
                {"type": t, "ops": int(n), "classes": int(c)}
                for tail in _SPLIT.findall(text)
                for t, n, c in _SPLIT_ONE.findall(tail)
            ]
            d["warnings"] = [l.strip()[:300] for l in text.splitlines() if "WARN" in l][
                :20
            ]
            return d
    return {
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
        "status": "crash",
        "stage": "?",
        "seconds": round(time.time() - t0, 1),
        "error": text[-3000:],
    }


# --- tables ------------------------------------------------------------------


def _fmt(v, width, prec=None):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{prec or 1}f}".rjust(width)
    return str(v).rjust(width)


def _key_of(r) -> str:
    return f"{r['key']}/{r['variant']}"


# Per-scheduler columns. `gaps` counts the regions whose achieved II exceeded
# the bound the LP justifies, the one figure the compiler publishes about its
# own possible loss; `bdgt` counts the exact solves that shipped an unproven
# placement, which is the one way an exact run can be WORSE than a heuristic one.
_COLS = [
    ("latency", "latency", 10),
    ("reg_bits", "regFF", 7),
    ("delay_bits", "dlyFF", 7),
    ("gaps", "gaps", 5),
    ("budget_exhausted", "bdgt", 5),
    ("solve_ms", "solve_ms", 9),
]
_GROUP = sum(w for _, _, w in _COLS) + len(_COLS)


def variant_table(results: list[dict], schedulers: list[str]) -> str:
    """One row per variant, one column group per scheduler."""
    by = {}
    for r in results:
        by.setdefault(_key_of(r), {})[r["scheduler"]] = r

    top = f"{'':<34}" + "".join(f"  {('[' + s + ']').center(_GROUP)}" for s in schedulers)
    head = f"{'benchmark/variant':<34}" + "".join(
        "  " + "".join(" " + label.rjust(w) for _, label, w in _COLS)
        for _ in schedulers
    )
    lines = [top, head, "-" * len(head)]
    for name in sorted(by):
        row = f"{name:<34}"
        for s in schedulers:
            r = by[name].get(s)
            row += "  "
            if r is None or r["status"] != "pass":
                row += ((r or {}).get("status", "-")).center(_GROUP)
                continue
            for field, _, w in _COLS:
                v = len(r.get("ii_gaps", [])) if field == "gaps" else r.get(field)
                # A latency that is not exact is parenthesized: it is an upper
                # bound, so it may not be differenced against another run's.
                if field == "latency" and v is not None and not r.get("latency_exact"):
                    row += " " + f"({v})".rjust(w)
                else:
                    row += " " + _fmt(v, w)
        lines.append(row)
    return "\n".join(lines)


def region_table(results: list[dict]) -> str:
    """One row per region, for the runs that reached a schedule."""
    head = (
        f"{'benchmark/variant':<30} {'sched':<5} {'func':<18} {'#':>3} {'kind':<8}"
        f" {'ii':>5} {'len':>6} {'drain':>6} {'trip':>7} {'lat':>9} {'ops':>5}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        for g in r.get("regions", []):
            lines.append(
                f"{_key_of(r):<30} {r['scheduler'][:5]:<5} {g['func'][:18]:<18}"
                f" {g['order']:>3} {g['kind']:<8}"
                f" {_fmt(g['ii'], 5)} {_fmt(g['length'], 6)} {_fmt(g['drain'], 6)}"
                f" {_fmt(g['trip'], 7)} {_fmt(g['latency'], 9)} {g['ops']:>5}"
            )
    return "\n".join(lines)


def solve_table(results: list[dict], top: int) -> str:
    """The slowest solves, which is what a compile-time regression shows up in."""
    rows = [
        (s["ms"], r["scheduler"], _key_of(r), s)
        for r in results
        for s in r.get("solves", [])
    ]
    rows.sort(reverse=True, key=lambda t: t[0])
    head = (
        f"{'ms':>9} {'sched':<5} {'benchmark/variant':<30} {'kind':<8}"
        f" {'ops':>5} {'lim':>5} {'ii':>5}  where"
    )
    lines = [head, "-" * len(head)]
    for ms, sched, name, s in rows[:top]:
        lines.append(
            f"{ms:>9.1f} {sched[:5]:<5} {name:<30} {s['kind']:<8}"
            f" {s['ops']:>5} {s['limited_ops']:>5} {_fmt(s['ii'], 5)}  {s['where']}"
        )
    return "\n".join(lines)


def alloc_table(results: list[dict]) -> str:
    """Per variant, how many physical units the schedule was realized on and
    what interconnect that took.

    `ops/unit` is the sharing ratio, 1.00 under the trivial binding. `muxFF` is
    the 2:1-mux bit count that sharing cost. Needs ALLO_LOG_LEVEL=info; empty
    otherwise."""
    rows = [r for r in results if r.get("alloc")]
    if not rows:
        return "no allocation data (re-run with ALLO_LOG_LEVEL=info)"
    head = (
        f"{'benchmark/variant':<34} {'sched':<6} {'regions':>7} {'ops':>7}"
        f" {'units':>7} {'ops/unit':>9} {'muxes':>7} {'muxFF':>9} {'splits':>7}"
    )
    lines = [head, "-" * len(head)]
    tot = dict.fromkeys(("ops", "units", "muxes", "mux_bits"), 0)
    for r in sorted(rows, key=lambda r: -r.get("alloc_mux_bits", 0)):
        o, u = r["alloc_ops"], r["alloc_units"]
        for f in tot:
            tot[f] += r[f"alloc_{f}"]
        lines.append(
            f"{_key_of(r):<34} {r['scheduler'][:6]:<6} {len(r['alloc']):>7}"
            f" {o:>7} {u:>7} {o / u:>9.2f} {r['alloc_muxes']:>7}"
            f" {r['alloc_mux_bits']:>9} {len(r.get('class_splits', [])):>7}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<34} {'':<6} {'':>7} {tot['ops']:>7} {tot['units']:>7}"
        f" {tot['ops'] / max(tot['units'], 1):>9.2f} {tot['muxes']:>7}"
        f" {tot['mux_bits']:>9}"
    )
    return "\n".join(lines)


def area_table(results: list[dict]) -> str:
    """Per variant, predicted area from the measured device tables.

    `ipLUT`/`muxLUT`/`lgcLUT` split the LUT total by what spends it, which is
    the split an allocation objective trades along: a fold removes IP and grows
    mux. `SRL` is the delay chains, which is where the register term's cost
    ACTUALLY lands; `regFF` beside `modFF` is what they cost against what the
    objective charges for them. Memory is carried apart and never summed in."""
    rows = [r for r in results if r.get("area")]
    if not rows:
        return "no area data (needs --stage compile)"
    head = (
        f"{'benchmark/variant':<34} {'sched':<6} {'LUT':>8} {'ipLUT':>8}"
        f" {'muxLUT':>8} {'lgcLUT':>8} {'memLUT':>8} {'SRL':>6} {'DSP':>5}"
        f" {'regFF':>8} {'modFF':>8} {'ramKb':>7}"
    )
    lines = [head, "-" * len(head)]
    tot = dict.fromkeys(
        ("lut", "ip_lut", "mux_lut", "logic_lut", "mem_lut", "srl", "dsp",
         "reg_ff", "reg_ff_modelled", "mem_bits"), 0)
    multiwrite = 0
    unmodelled: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: -r["area"]["lut"]):
        a = r["area"]
        for f in tot:
            tot[f] += a[f]
        for k, n in a.get("unmodelled", {}).items():
            unmodelled[k] = unmodelled.get(k, 0) + n
        multiwrite += a["multiwrite_arrays"]
        lines.append(
            f"{_key_of(r):<34} {r['scheduler'][:6]:<6} {a['lut']:>8}"
            f" {a['ip_lut']:>8} {a['mux_lut']:>8} {a['logic_lut']:>8}"
            f" {a['mem_lut']:>8} {a['srl']:>6} {a['dsp']:>5} {a['reg_ff']:>8}"
            f" {a['reg_ff_modelled']:>8} {a['mem_bits'] / 1024:>7.1f}"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<34} {'':<6} {tot['lut']:>8} {tot['ip_lut']:>8}"
        f" {tot['mux_lut']:>8} {tot['logic_lut']:>8} {tot['mem_lut']:>8}"
        f" {tot['srl']:>6} {tot['dsp']:>5} {tot['reg_ff']:>8}"
        f" {tot['reg_ff_modelled']:>8} {tot['mem_bits'] / 1024:>7.1f}"
    )
    over = tot["reg_ff_modelled"] / max(tot["reg_ff"], 1)
    lines.append("")
    lines.append(
        f"the objective charges {tot['reg_ff_modelled']} flip-flops for chains "
        f"that cost {tot['reg_ff']} FF + {tot['srl']} SRL: {over:.1f}x over"
    )
    if multiwrite:
        lines.append(
            f"{multiwrite} arrays have more than one writer, so they infer no "
            f"RAM and cost {tot['mem_lut']} LUTs of register file"
        )
    if unmodelled:
        lines.append(f"UNMODELLED (scored as zero): {unmodelled}")
    return "\n".join(lines)


def split_table(results: list[dict]) -> str:
    """Where one timing row covers several operator identities, aggregated over
    the bed. A binder folds only within one identity, so a count keyed on the
    operator type spans several physical operators."""
    agg: dict[str, list[int]] = {}
    for r in results:
        for s in r.get("class_splits", []):
            row = agg.setdefault(s["type"], [0, 0, 0])
            row[0] += 1  # regions where this type splits
            row[1] += s["ops"]
            row[2] = max(row[2], s["classes"])
    if not agg:
        return "no operator type splits (or ALLO_LOG_LEVEL is not info)"
    head = f"{'operator type':<28} {'regions':>8} {'ops':>8} {'max classes':>12}"
    lines = [head, "-" * len(head)]
    for t, (n, ops, mx) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{t:<28} {n:>8} {ops:>8} {mx:>12}")
    return "\n".join(lines)


def compare_table(base: list[dict], new: list[dict]) -> str:
    """What moved between two runs.

    Only exact latencies are differenced: a bound may move because the
    assumption behind it moved, which is not a schedule getting better."""
    fields = [("latency", "latency"), ("reg_bits", "regFF"), ("solve_ms", "solve_ms")]
    index = lambda rs: {(_key_of(r), r["scheduler"]): r for r in rs}
    b, n = index(base), index(new)
    head = f"{'benchmark/variant':<30} {'sched':<6}"
    for _, label in fields:
        head += f" {label + ' base':>13} {label + ' new':>13} {'delta':>9}"
    lines = [head, "-" * len(head)]
    moved = 0
    for k in sorted(set(b) & set(n)):
        rb, rn = b[k], n[k]
        if rb["status"] != "pass" or rn["status"] != "pass":
            continue
        cells, changed = "", False
        for field, _ in fields:
            vb, vn = rb.get(field), rn.get(field)
            if field == "latency" and not (
                rb.get("latency_exact") and rn.get("latency_exact")
            ):
                vb = vn = None
            if vb is None or vn is None:
                cells += f" {'-':>13} {'-':>13} {'-':>9}"
                continue
            d = vn - vb
            # Solve time is wall time, so a small move is noise; a schedule
            # figure is exact and any move is real.
            if d and (field != "solve_ms" or abs(d) > 0.2 * max(vb, 1)):
                changed = True
            cells += f" {_fmt(vb, 13)} {_fmt(vn, 13)} {_fmt(d, 9)}"
        if changed:
            moved += 1
            lines.append(f"{k[0]:<30} {k[1][:6]:<6}{cells}")
    only = (set(b) ^ set(n)) or None
    lines.append("")
    lines.append(f"{moved} of {len(set(b) & set(n))} runs moved")
    if only:
        lines.append(f"present in only one run: {sorted(x[0] for x in only)}")
    return "\n".join(lines)


# --- driver ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--one", help=argparse.SUPPRESS)  # the child entry point
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("-k", "--filter", default="", help="substring of suite/name")
    ap.add_argument(
        "--stage",
        default="compile",
        choices=("schedule", "compile"),
        help="`schedule` is fast and has no register counts",
    )
    ap.add_argument(
        "--scheduler",
        default="heuristic,exact",
        help="comma-separated; `exact` is dropped on a build without OR-Tools",
    )
    ap.add_argument(
        "--freq",
        type=float,
        help="target clock (MHz), overriding the device default. The period is "
        "what chaining is cut against, so this is the axis a chaining change "
        "is swept over",
    )
    ap.add_argument(
        "--budget",
        type=float,
        help="what ONE exact solve may spend, in the solver's "
        "deterministic time units (default 10). The axis a budget policy is "
        "swept over; it does nothing to the heuristic",
    )
    ap.add_argument(
        "--binding",
        default="trivial",
        help="operator-sharing policy: 'trivial' (the default, one unit per "
        "op), 'greedy-share' or 'planned', which builds the allocation the "
        "exact scheduler decided",
    )
    ap.add_argument("--timeout", type=int, default=900, help="wall seconds per run")
    ap.add_argument("-o", "--out", default="qor.json")
    ap.add_argument("--per-region", action="store_true")
    ap.add_argument("--solves", type=int, default=0, metavar="N", help="slowest N")
    ap.add_argument(
        "--alloc",
        action="store_true",
        help="the per-variant allocation (units, muxes) and the operator-type "
        "splits. Both need ALLO_LOG_LEVEL=info",
    )
    ap.add_argument(
        "--area",
        action="store_true",
        help="predicted LUT/DSP/FF from the measured device tables, split by "
        "what spends them. The scoreboard an allocation change is argued from",
    )
    ap.add_argument("--compare", metavar="BASE.json", help="diff against a saved run")
    args = ap.parse_args()

    if args.one:
        key, variant, scheduler = args.one.split("::")
        print(
            MARK
            + json.dumps(
                measure_one(
                    key,
                    variant,
                    scheduler,
                    args.stage,
                    args.freq,
                    args.budget,
                    args.binding,
                )
            )
        )
        return

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        new = json.loads(Path(args.out).read_text())
        print(compare_table(base, new))
        return

    sys.path.insert(0, str(REPO))
    from allo.backend.rtl import has_exact_scheduler
    from benchmark.spec import discover

    schedulers = [s for s in args.scheduler.split(",") if s]
    if not has_exact_scheduler() and any(s.startswith("exact") for s in schedulers):
        print("this build has no OR-Tools, so the exact modes are dropped", file=sys.stderr)
        schedulers = [s for s in schedulers if not s.startswith("exact")]
    if not schedulers:
        raise SystemExit("no scheduler to run")

    work = [
        (b.key, v, s)
        for b in discover()
        for v in b.schedules
        for s in schedulers
        if args.filter in b.key
    ]
    clock = f", freq={args.freq}MHz" if args.freq else ""
    pool_size = f", budget={args.budget}" if args.budget else ""
    print(
        f"{len(work)} runs, {args.jobs} jobs, stage={args.stage}"
        f", binding={args.binding}{clock}{pool_size}",
        flush=True,
    )

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [
            pool.submit(
                _run_child,
                w,
                args.stage,
                args.timeout,
                args.freq,
                args.budget,
                args.binding,
            )
            for w in work
        ]
        for f in futs:
            r = f.result()
            results.append(r)
            done += 1
            tag = {"pass": "ok", "skip": "--"}.get(r["status"], r["status"].upper())
            print(
                f"[{done:3d}/{len(work)}] {tag:>8}  {_key_of(r)}"
                f" [{r['scheduler']}]  {r.get('seconds', 0)}s",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}\n")

    ok = [r for r in results if r["status"] == "pass"]
    print(variant_table(results, schedulers))
    if args.per_region:
        print("\n" + region_table(ok))
    if args.solves:
        print("\n" + solve_table(ok, args.solves))
    if args.alloc:
        print("\n" + alloc_table(ok))
        print("\n" + split_table(ok))
    if args.area:
        print("\n" + area_table(ok))

    tally = {}
    for r in results:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    bad = [r for r in ok if r.get("parse_ok") is False]
    if bad:
        print(
            f"WARNING: the register parser missed a declaration in {len(bad)} run(s);"
            " their register counts are not to be trusted"
        )


if __name__ == "__main__":
    main()
