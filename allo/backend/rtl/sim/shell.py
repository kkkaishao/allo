# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the DUT and run it through ``cocotb_tools.runner``.

``cosim`` emits the DUT (Verilog + extern-IP behavioral models + DPI), marshals
the numpy arguments to files, runs the generic testbench (``cocotb_tb``) on the
named simulator (verilator / icarus / ...), and reads the written arrays + cycle
count back.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ip_models
from . import ports as _ports

_TB_MODULE = "allo.backend.rtl.sim.cocotb_tb"


def available(simulator: str = "verilator") -> bool:
    # Verilog is emitted in-process (CIRCT is linked into the package), so only
    # the simulator itself is an external dependency.
    return shutil.which(simulator) is not None


@dataclass
class CosimResult:
    cycles: int
    latency_ns: float
    waveform: Path | None = None
    # The kernel's scalar return value sampled at `done`: the bare value for a
    # single scalar result, a tuple for several, or None when the kernel returns
    # nothing / only array (out-param) results. Array results are written back in
    # place, so they do not appear here.
    result: object = None


def _write_sources(
    hw_ir: str, verilog: str, top: str, workdir: Path
) -> tuple[list[Path], list[str]]:
    """Write the DUT Verilog (+ extern-IP behavioral models) and DPI C. Returns
    (verilog_sources, build_args) for the runner. The extern-IP models and DPI are
    derived from the hw IR, which names the operator instances."""
    dut = workdir / f"{top}.sv"
    dut.write_text(verilog + "\n" + ip_models.sv_models(hw_ir))
    build_args: list[str] = []
    dpi = ip_models.dpi_c(hw_ir)
    if dpi:
        cpp = workdir / "dpi.cpp"
        cpp.write_text(dpi)
        build_args.append(str(cpp))
    return [dut], build_args


def _build_config(
    interface,
    top,
    mems,
    streams,
    arg_types,
    args,
    *,
    clock_ps,
    timeout,
    workdir,
    stall_prob=0.0,
) -> dict:
    """Serialize each backing array to ``.npy`` and build the testbench config."""
    mem_cfgs = []
    for m in mems:
        tag = f"{m.arg}_b{m.bank}"  # one backing array per (argument, bank)
        file_in = None
        if m.readers or not m.writeback:  # read/RMW args are preloaded from the arg
            bits = m.slice_in(args[m.arg], m.width)
            file_in = workdir / f"in_arg{tag}.npy"
            np.save(file_in, bits.astype(np.uint64))
        file_out = str(workdir / f"out_arg{tag}.npy") if m.writeback else None
        mem_cfgs.append(
            {
                "file_in": str(file_in) if file_in else None,
                "file_out": file_out,
                "size": m.size,
                "width": m.width,
                "readers": m.readers,
                "writers": m.writers,
            }
        )
    scalars = [
        {
            "name": sc["name"],
            "value": _ports.scalar_bits(args[sc["arg"]], arg_types[sc["arg"]]),
        }
        for sc in interface["scalars"]
    ]
    # Streams: an input's token sequence is serialized to `.npy` (the feeder
    # drives them through the valid/ready handshake); an output records where to
    # write the drained tokens and how many to expect (its pre-allocated buffer's
    # length). Each config carries the concrete handshake port names.
    stream_cfgs = []
    for s in streams:
        cfg = {
            "base": s.base,
            "data": s.data,
            "valid": s.valid,
            "ready": s.ready,
            "input": s.is_input,
        }
        if s.is_input:
            bits = _ports.bit_pattern(np.asarray(args[s.arg]), s.np_dtype, s.width)
            file_in = workdir / f"in_stream{s.arg}.npy"
            np.save(file_in, bits.astype(np.uint64))
            cfg["file_in"] = str(file_in)
        else:
            cfg["count"] = int(np.asarray(args[s.arg]).reshape(-1).shape[0])
            cfg["file_out"] = str(workdir / f"out_stream{s.arg}.npy")
        stream_cfgs.append(cfg)
    return {
        "top": top,
        "clock_ps": clock_ps,
        "timeout": timeout,
        "reset_cycles": 3,
        "settle_cycles": 2,
        "mems": mem_cfgs,
        "scalars": scalars,
        "streams": stream_cfgs,
        "stream_gap": stall_prob,
        "result_ports": [r["name"] for r in interface["results"]],
        "results_out": str(workdir / "results.json"),
        "cycles_out": str(workdir / "cycles.txt"),
    }


def cosim(
    hw_ir: str,
    verilog: str,
    interface: dict,
    top: str,
    arg_types,
    args,
    *,
    result_types=(),
    simulator: str = "verilator",
    freq_mhz: float = 300.0,
    timeout: int = 40000,
    workdir: str | os.PathLike | None = None,
    waves: bool = False,
    stall_prob: float = 0.0,
) -> CosimResult:
    """Drive the emitted module (``verilog``, named ``top``, with ``hw_ir`` for the
    extern-IP models) under cocotb + ``simulator`` with the numpy ``args``, bound
    to ports by the ``interface`` port manifest. Writes each output argument back
    in place; returns the cycle count."""
    from cocotb_tools.runner import get_runner

    assert len(args) == len(
        arg_types
    ), f"cosim expected {len(arg_types)} kernel arguments, got {len(args)}"
    mems = _ports.plan_mems(interface, arg_types)
    streams = _ports.plan_streams(interface, arg_types)

    tmp = workdir is None
    wd = Path(tempfile.mkdtemp(prefix="allo_cosim_")) if tmp else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    try:
        verilog_sources, build_args = _write_sources(hw_ir, verilog, top, wd)
        if simulator == "verilator":
            # The extern-IP behavioral models are width-approximate -- a fixed
            # 64-bit DPI backs a possibly-wider operator (e.g. a chained widened
            # multiply is i96) -- so verilator's width-mismatch warnings are
            # benign here. Keep them non-fatal; the golden comparison still
            # catches any real value corruption.
            build_args = ["-Wno-fatal", *build_args]
        # Clock period as an even integer ps (cocotb splits it into two half
        # periods); it only affects sim time, not the reported cycle count.
        clock_ps = round(1.0e6 / freq_mhz)
        clock_ps += clock_ps & 1
        cfg = _build_config(
            interface,
            top,
            mems,
            streams,
            arg_types,
            args,
            clock_ps=clock_ps,
            timeout=timeout,
            workdir=wd,
            stall_prob=stall_prob,
        )
        cfg_path = wd / "cosim.json"
        cfg_path.write_text(json.dumps(cfg))

        runner = get_runner(simulator)
        runner.build(
            sources=verilog_sources,
            build_args=build_args,
            hdl_toplevel=top,
            build_dir=str(wd / "sim_build"),
            always=True,
            waves=waves,
        )
        from cocotb_tools.runner import get_results

        xml = runner.test(
            hdl_toplevel=top,
            test_module=_TB_MODULE,
            test_dir=str(wd),
            extra_env={"ALLO_COSIM_CFG": str(cfg_path)},
            waves=waves,
        )
        _, failed = get_results(xml)
        assert failed == 0, f"cosim testbench failed (see {wd}/sim.log)"

        cycles = int((wd / "cycles.txt").read_text().strip())
        for m in mems:
            if m.writeback:
                bits = np.load(wd / f"out_arg{m.arg}_b{m.bank}.npy").astype(
                    _ports._UINT[m.width]
                )
                vals = _ports.from_bits(bits, m.np_dtype, m.width, (m.size,))
                m.scatter_out(args[m.arg], vals)
        # Drained output-stream tokens, written into the caller's buffer in place.
        for s in streams:
            if not s.is_input:
                buf = np.asarray(args[s.arg])
                bits = np.load(wd / f"out_stream{s.arg}.npy").astype(
                    _ports._UINT[s.width]
                )
                buf[...] = _ports.from_bits(bits, s.np_dtype, s.width, buf.shape)
        # Decode each sampled scalar-result port by its return type (the manifest
        # order matches the scalar `result_types`); surface a bare value / tuple.
        raw = json.loads((wd / "results.json").read_text())
        assert len(raw) == len(
            result_types
        ), f"cosim sampled {len(raw)} result ports, expected {len(result_types)}"
        vals = [_ports.from_scalar_bits(b, t) for b, t in zip(raw, result_types)]
        result = vals[0] if len(vals) == 1 else (tuple(vals) if vals else None)
        wave = next(iter(wd.glob("sim_build/*.fst")), None) if waves else None
        return CosimResult(cycles, cycles * 1000.0 / freq_mhz, wave, result)
    finally:
        if tmp:
            shutil.rmtree(wd, ignore_errors=True)
