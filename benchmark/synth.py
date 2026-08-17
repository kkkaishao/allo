# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual area over the benchmark bed: scaffold, synthesize, compare.

    python -m benchmark.synth -k atax/none            # OOC synthesis, one case
    python -m benchmark.synth --impl                  # place+route, timing too
    python -m benchmark.synth --skip-synth            # scaffolds only

The bed's third leg: `report.py` predicts what a variant costs, `verify.py`
checks what it computes, this synthesizes what it emits, with the real
operator cores, and reports what Vivado actually built. Each case scaffolds
through `RTL.scaffold_project`, so the RTL, the operator-core wrappers and the
core-generation script are exactly the shipped flow's, never a bed-local
re-derivation.

Three disciplines, each paid for:

  - One Vivado process per design. Vivado segfaults on some emitted RTL and a
    segfault is not something `catch` survives, so a bad design must not cost
    the queue behind it; separate processes also keep two variants of one
    kernel, which share a top module name, out of one in-memory project.
  - A design whose extern operators have no realization is synthesized but
    marked: a black box synthesizes to nothing, so its actual area
    under-counts and the row says so rather than passing as a measurement.
  - A stale utilization report is deleted before its design runs: a fresh
    prediction silently paired with an actual measured off older hardware is
    worse than a hole in the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARK = "@@SYNTH@@"


# --- the child: emit and scaffold one design --------------------------------


def emit_one(
    key: str, variant: str, scheduler: str, binding: str, freq: float | None, work: Path
) -> dict:
    """Compile one (benchmark, variant, scheduler) and scaffold it under
    ``work``, returning the row the synthesis phase consumes."""
    sys.path.insert(0, str(REPO))
    from benchmark.report import area_of
    from benchmark.spec import find

    tag = f"{key}/{variant}/{scheduler}".replace("/", "_")
    out: dict = {
        "tag": tag,
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
        "binding": binding,
        "status": "error",
    }
    bench = find(key)
    if variant in bench.skip:
        out.update(status="skip", note=bench.skip[variant])
        return out
    try:
        parts = bench.build()
        sched = bench.schedules[variant](parts)
        opts = {"freq_mhz": freq} if freq is not None else {}
        rtl = sched.export("rtl", **opts).set_scheduler_opt(scheduler=scheduler)
        if binding == "trivial":
            rtl.use_trivial_binding()
        res = rtl.schedule()
        rtl.compile()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rtl.scaffold_project(str(work / f"{tag}.prj"))
        q = rtl.estimation
        out.update(
            status="pass",
            top=rtl.top,
            part=rtl.device.part,
            clk=rtl.interfaces.of_symbol(rtl.top).control.clk,
            cycle_ns=res.cycle_ns or 1000.0 / rtl.freq_mhz,
            predicted={**area_of(q), "mem_bits": q.mem_bits},
            blackboxes=[str(w.message) for w in caught],
        )
    except BaseException as e:  # a fired assert is a row, not a crash
        out["error"] = f"{type(e).__name__}: {e}"[:2000]
    return out


def _run_child(
    item, binding: str, freq: float | None, work: Path, timeout: int
) -> dict:
    key, variant, scheduler = item
    env = dict(os.environ)
    env["XILINX_VITIS"] = "/nonexistent"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("ALLO_LOG_LEVEL", "warn")
    cmd = [
        sys.executable,
        "-m",
        "benchmark.synth",
        "--one",
        f"{key}::{variant}::{scheduler}",
        "--binding",
        binding,
        "--work",
        str(work),
    ]
    if freq is not None:
        cmd += ["--freq", str(freq)]
    row = {
        "tag": f"{key}/{variant}/{scheduler}".replace("/", "_"),
        "key": key,
        "variant": variant,
        "scheduler": scheduler,
    }
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(REPO),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {**row, "status": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK) :])
    return {**row, "status": "crash", "error": (p.stdout + p.stderr)[-3000:]}


# --- the synthesis phase -----------------------------------------------------


def vivado_command(explicit: str | None) -> str:
    """The shell prefix that reaches a `vivado` binary: an explicit path, one
    already on PATH, or the newest install under /tools/Xilinx/Vivado."""
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            settings = p / "settings64.sh"
            if not settings.exists():
                raise SystemExit(f"--vivado: no {settings}")
            return f"source {settings} && vivado"
        if not p.exists():
            raise SystemExit(f"--vivado: no such binary {p}")
        return str(p)
    if shutil.which("vivado"):
        return "vivado"
    if xv := os.environ.get("XILINX_VIVADO"):
        settings = Path(xv) / "settings64.sh"
        if not settings.exists():
            raise SystemExit(f"XILINX_VIVADO: no {settings}")
        return f"source {settings} && vivado"
    installs = sorted(Path("/tools/Xilinx/Vivado").glob("*/settings64.sh"))
    if installs:
        return f"source {installs[-1]} && vivado"
    raise SystemExit(
        "no vivado: put one on PATH, set XILINX_VIVADO to its install "
        "directory, or pass --vivado"
    )


def design_tcl(d: dict, work: Path, impl: bool) -> Path:
    """One design's whole run: its own project (via the scaffold's
    `gen_ip.tcl` when it has cores, so the part and the generated IP come from
    the shipped script), the split RTL off `filelist.f`, OOC synthesis, and
    under ``--impl`` a clock constraint plus place and route."""
    prj = work / f"{d['tag']}.prj"
    gen_ip = prj / "gen_ip.tcl"
    if gen_ip.exists():
        project = [f"source {gen_ip}"]
    else:
        project = [
            f"create_project -in_memory -part {d['part']}",
            "set_property target_language Verilog [current_project]",
        ]
    reads = [
        f"  read_verilog -sv -quiet {prj / name}"
        for name in (prj / "filelist.f").read_text().split()
    ]
    if (prj / "shims.v").exists():
        reads.append(f"  read_verilog -sv -quiet {prj / 'shims.v'}")
    steps = [
        f"  synth_design -top {d['top']} -part {d['part']}"
        " -mode out_of_context -flatten_hierarchy none",
    ]
    if impl:
        steps += [
            f"  create_clock -period {d['cycle_ns']} -name bed_clk"
            f" [get_ports {d['clk']}]",
            "  opt_design",
            "  place_design",
            "  route_design",
            f"  report_timing_summary -file {work / d['tag']}_timing.rpt",
        ]
    steps.append(f"  report_utilization -file {work / d['tag']}_util.rpt")
    lines = [
        *project,
        "if {[catch {",
        *reads,
        *steps,
        f'  puts "### DONE {d["tag"]}"',
        '} err]} { puts "### FAIL ' + d["tag"] + ': $err" }',
    ]
    p = work / f"synth_{d['tag']}.tcl"
    p.write_text("\n".join(lines) + "\n")
    return p


def run_vivado(vivado: str, tcl: Path, log: Path, timeout: int) -> None:
    """One Vivado process; a hung one is killed so it cannot pin its slot, and
    the missing report marks the design's row."""
    with log.open("w") as sink:
        try:
            subprocess.run(
                f"{vivado} -mode batch -nojournal -nolog -source {tcl}",
                shell=True,
                executable="/bin/bash",
                cwd=tcl.parent,
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            sink.write(f"\n### TIMEOUT after {timeout}s\n")


#: `report_utilization` row -> the key it lands under. "LUT as Memory" is left
#: out on purpose: it is the sum of the two rows below it, and counting it too
#: would double-charge.
_UTIL_ROWS = {
    "LUT as Logic": "lut_logic",
    "LUT as Distributed RAM": "lut_mem",
    "LUT as Shift Register": "srl",
    "CLB Registers": "ff",
    "Block RAM Tile": "bram",
    "DSPs": "dsp",
    "CARRY8": "carry8",
}


def read_utilization(work: Path, tag: str) -> dict | None:
    """One design's cell counts off its `report_utilization`. A Block RAM Tile
    is reported in halves (an 18Kb RAMB is 0.5), so it is kept a float. The
    first occurrence of a row wins: a post-route report repeats the names in
    later per-region tables whose leading column is not a count."""
    p = work / f"{tag}_util.rpt"
    if not p.exists():
        return None
    out = {v: 0.0 for v in _UTIL_ROWS.values()}
    seen: set[str] = set()
    for line in p.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and (key := _UTIL_ROWS.get(cells[1])):
            if key not in seen:
                seen.add(key)
                out[key] = float(cells[2])
    # A LUT the design spends is a LUT whichever role it is in.
    out["lut"] = out["lut_logic"] + out["lut_mem"] + out["srl"]
    return {k: int(v) if v == int(v) else v for k, v in out.items()}


def read_wns(work: Path, tag: str) -> float | None:
    """The design's worst negative slack off `report_timing_summary`: the
    first value row under the ``WNS(ns)`` header."""
    p = work / f"{tag}_timing.rpt"
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" not in line:
            continue
        for row in lines[i + 1 :]:
            tok = row.split()
            if tok and not set(row) <= set("- |"):
                try:
                    return float(tok[0])
                except ValueError:
                    break
        break
    return None


# --- the table and the CSV ---------------------------------------------------

_PRED = ("lut", "ff", "dsp", "srl", "mem_bits")
_ACT = ("lut", "lut_logic", "lut_mem", "srl", "ff", "dsp", "carry8", "bram")


def write_csv(work: Path, rows: list[dict], impl: bool) -> None:
    with (work / "synth.csv").open("w") as f:
        w = csv.writer(f)
        head = ["tag", "status"] + [f"pred_{k}" for k in _PRED] + list(_ACT)
        if impl:
            head += ["wns_ns", "fmax_mhz"]
        w.writerow(head)
        for r in rows:
            line = [r["tag"], r["status"]]
            line += [r.get("predicted", {}).get(k, "") for k in _PRED]
            a = r.get("actual") or {}
            line += [a.get(k, "") for k in _ACT]
            if impl:
                line += [r.get("wns_ns", ""), r.get("fmax_mhz", "")]
            w.writerow(line)


def print_table(rows: list[dict], impl: bool) -> None:
    head = (
        f"{'design':<38} {'LUT p/a':>15} {'ratio':>6} {'FF p/a':>15}"
        f" {'ratio':>6} {'DSP p/a':>9} {'SRL p/a':>11}"
    )
    if impl:
        head += f" {'WNS':>7} {'fmax':>6}"
    print()
    print(head)
    print("-" * len(head))
    for r in rows:
        if r["status"] != "pass":
            note = r.get("note") or r.get("error", "")
            print(
                f"{r['tag']:<38} [{r['status']}] {note.splitlines()[0][:80]}"
                if note
                else f"{r['tag']:<38} [{r['status']}]"
            )
            continue
        p, a = r["predicted"], r.get("actual")
        note = "  [BLACK BOXES]" if r.get("blackboxes") else ""
        if a is None:
            print(f"{r['tag']:<38} {p['lut']:>7}/{'--':<7}{note}")
            continue
        # `srl` in the estimate is every state-holding LUT site, so its actual
        # is the shift registers plus the distributed RAM.
        astate = a["srl"] + a["lut_mem"]
        line = (
            f"{r['tag']:<38} {p['lut']:>7}/{a['lut']:<7}"
            f" {p['lut'] / max(a['lut'], 1):>6.2f}"
            f" {p['ff']:>7}/{a['ff']:<7} {p['ff'] / max(a['ff'], 1):>6.2f}"
            f" {p['dsp']:>4}/{a['dsp']:<4} {p['srl']:>5}/{astate:<5}"
        )
        if impl:
            wns = r.get("wns_ns")
            fmax = r.get("fmax_mhz")
            line += f" {wns:>7.3f}" if wns is not None else f" {'--':>7}"
            line += f" {fmax:>6.1f}" if fmax is not None else f" {'--':>6}"
        print(line + note)


# --- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--one", help=argparse.SUPPRESS)  # the child entry point
    ap.add_argument(
        "-k", "--filter", default="", help="substring of suite/name/variant"
    )
    ap.add_argument(
        "--scheduler",
        default="heuristic",
        help="comma-separated; each case is emitted under each",
    )
    ap.add_argument(
        "--binding",
        default="trivial",
        help="'trivial' or 'auto', as the bed scan takes it",
    )
    ap.add_argument(
        "--freq", type=float, help="target clock (MHz), overriding the device default"
    )
    ap.add_argument(
        "--impl",
        action="store_true",
        help="place and route after synthesis, and report timing",
    )
    ap.add_argument(
        "--skip-synth", action="store_true", help="scaffold and predict only"
    )
    ap.add_argument("-j", "--jobs", type=int, default=8, help="emit children in flight")
    ap.add_argument(
        "--synth-jobs",
        type=int,
        default=4,
        help="Vivado sessions in flight (about 8 GB peak each)",
    )
    ap.add_argument(
        "--timeout", type=int, default=900, help="wall seconds per emit child"
    )
    ap.add_argument(
        "--synth-timeout",
        type=int,
        default=7200,
        help="wall seconds per Vivado session",
    )
    ap.add_argument("--vivado", help="vivado binary or install directory")
    ap.add_argument("--work", default=str(REPO / "benchmark" / "synth_work"))
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if args.one:
        key, variant, scheduler = args.one.split("::")
        row = emit_one(key, variant, scheduler, args.binding, args.freq, work)
        print(MARK + json.dumps(row), flush=True)
        return

    vivado = None if args.skip_synth else vivado_command(args.vivado)

    sys.path.insert(0, str(REPO))
    from benchmark.spec import discover

    items = [
        (b.key, v, s)
        for b in discover()
        for v in b.schedules
        for s in args.scheduler.split(",")
        if args.filter in f"{b.key}/{v}"
    ]
    print(
        f"{len(items)} designs, binding={args.binding}"
        + (f", freq={args.freq}MHz" if args.freq else ""),
        flush=True,
    )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [
            pool.submit(_run_child, it, args.binding, args.freq, work, args.timeout)
            for it in items
        ]
        for i, f in enumerate(futs, 1):
            r = f.result()
            rows.append(r)
            print(f"[{i}/{len(items)}] emit {r['tag']}: {r['status']}", flush=True)
            for miss in r.get("blackboxes", []):
                print(
                    f"  !! {r['tag']}: {miss} -- actual area will "
                    "under-count; do not believe this row",
                    flush=True,
                )

    designs = [r for r in rows if r["status"] == "pass"]
    if not args.skip_synth and designs:

        def synth(d):
            (work / f"{d['tag']}_util.rpt").unlink(missing_ok=True)
            (work / f"{d['tag']}_timing.rpt").unlink(missing_ok=True)
            t0 = time.time()
            run_vivado(
                vivado,
                design_tcl(d, work, args.impl),
                work / f"synth_{d['tag']}.out",
                args.synth_timeout,
            )
            return d["tag"], round(time.time() - t0, 1)

        with ThreadPoolExecutor(max_workers=args.synth_jobs) as pool:
            for i, f in enumerate([pool.submit(synth, d) for d in designs], 1):
                tag, seconds = f.result()
                print(f"[{i}/{len(designs)}] synth {tag}: {seconds}s", flush=True)

    for d in designs:
        d["actual"] = read_utilization(work, d["tag"])
        if args.impl and (wns := read_wns(work, d["tag"])) is not None:
            d["wns_ns"] = wns
            d["fmax_mhz"] = round(1000.0 / (d["cycle_ns"] - wns), 1)

    write_csv(work, rows, args.impl)
    print_table(rows, args.impl)
    print(f"\nrows: {work / 'synth.csv'}")


if __name__ == "__main__":
    main()
