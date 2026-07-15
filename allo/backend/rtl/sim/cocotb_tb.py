# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic, config-driven cocotb testbench.

This module runs inside the simulator's embedded Python (cocotb imports it by
name). It reads a JSON config (path in ``$ALLO_COSIM_CFG``) describing the module's
ports and the numpy-backed memories, drives clk/rst/start, services every memory
port, waits for ``done``, and writes the output arrays + cycle count back to files
for the host to read.

Each backing memory is modeled as a 1-cycle synchronous (registered) RAM: read
data presented in cycle k+1 reflects the address sampled at the edge ending cycle
k, and a write commits at that edge. Reads latch *before* writes apply at the same
edge, so a read-modify-write on one array sees the pre-write value (SV NBA
semantics).
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge


def _i(sig) -> int:
    """Integer value of a signal, or 0 when unresolvable (X during reset)."""
    v = sig.value
    return int(v) if v.is_resolvable else 0


async def _serve_mem(hdl, mem, readers, writers, size):
    """Service one backing array as a 1-cycle synchronous RAM, matching an
    ``always_ff @(posedge clk)`` model: the read address / write enable+addr+data
    are sampled *before* the edge (in ReadOnly, so they are the settled cycle
    values, like an NBA's RHS), then the read data is presented and the write
    committed at the edge."""
    rd = [(getattr(hdl, r["addr"]), getattr(hdl, r["data"])) for r in readers]
    wr = [
        (getattr(hdl, w["we"]), getattr(hdl, w["addr"]), getattr(hdl, w["data"]))
        for w in writers
    ]
    clamp = lambda a: a if 0 <= a < size else 0
    while True:
        await ReadOnly()  # end of cycle: settled (pre-edge) values
        r_addr = [clamp(_i(addr)) for addr, _ in rd]
        w = [(_i(we), clamp(_i(addr)), _i(dat)) for we, addr, dat in wr]
        await RisingEdge(hdl.clk)  # commit at the edge (NBA-like)
        for k, (_, data) in enumerate(rd):
            data.value = int(mem[r_addr[k]])
        for we, addr, dat in w:
            if we:
                mem[addr] = dat


async def _feed_stream(hdl, s, tokens, gap=0.0):
    """Source a FIFO stream: drive data + valid, and advance to the next token
    only on a cycle the DUT's ready is high at the edge (valid/ready handshake).
    With ``gap`` > 0, randomly withholds valid to starve the DUT (a latency-
    insensitive process must stall, not lose/duplicate a token). Holds valid low
    once the sequence is exhausted."""
    data = getattr(hdl, s["data"])
    valid = getattr(hdl, s["valid"])
    ready = getattr(hdl, s["ready"])
    i = 0
    while i < len(tokens):
        if gap and random.random() < gap:  # starve: offer nothing this cycle
            valid.value = 0
            await RisingEdge(hdl.clk)
            continue
        data.value = int(tokens[i])
        valid.value = 1
        await ReadOnly()  # settled: is the DUT ready this cycle?
        fired = _i(ready) == 1
        await RisingEdge(hdl.clk)
        if fired:
            i += 1
    valid.value = 0
    data.value = 0


async def _drain_stream(hdl, s, out, count, gap=0.0):
    """Sink a FIFO stream: capture data on every cycle the DUT drives valid while
    ready is held, until ``count`` tokens are collected. With ``gap`` > 0,
    randomly deasserts ready to back-pressure the DUT (which must freeze, not
    drop a token)."""
    data = getattr(hdl, s["data"])
    valid = getattr(hdl, s["valid"])
    ready = getattr(hdl, s["ready"])
    while len(out) < count:
        stall = bool(gap) and random.random() < gap
        ready.value = 0 if stall else 1
        await ReadOnly()
        if not stall and _i(valid) == 1:
            out.append(_i(data))
        await RisingEdge(hdl.clk)
    ready.value = 0


@cocotb.test()
async def cosim(hdl):
    with open(os.environ["ALLO_COSIM_CFG"]) as f:
        cfg = json.load(f)

    cocotb.start_soon(Clock(hdl.clk, cfg["clock_ps"], unit="ps").start())
    for s in cfg["scalars"]:
        getattr(hdl, s["name"]).value = s["value"]

    # Streams: quiesce the handshake lines during reset (feeders/drainers take
    # over after) and prepare an output-capture list per drained stream.
    stream_out: dict[str, list] = {}
    for s in cfg["streams"]:
        if s["input"]:
            getattr(hdl, s["valid"]).value = 0
        else:
            getattr(hdl, s["ready"]).value = 0
            stream_out[s["base"]] = []

    # Load each backing array (preloaded input / RMW seed, or zeros for a pure output).
    arrays = []
    for m in cfg["mems"]:
        if m["file_in"]:
            arr = np.load(m["file_in"]).reshape(-1).astype(np.uint64)
        else:
            arr = np.zeros(m["size"], dtype=np.uint64)
        arrays.append(arr)

    hdl.rst.value = 1
    hdl.start.value = 0
    for _ in range(cfg["reset_cycles"]):
        await RisingEdge(hdl.clk)
    hdl.rst.value = 0
    await RisingEdge(hdl.clk)

    for m, arr in zip(cfg["mems"], arrays):
        cocotb.start_soon(_serve_mem(hdl, arr, m["readers"], m["writers"], m["size"]))
    # Feed each input stream its token sequence; drain each output stream. A
    # non-zero `stream_gap` randomly starves inputs / back-pressures outputs to
    # exercise the latency-insensitive stall shell (the result must be identical
    # -- KPN determinism).
    gap = cfg.get("stream_gap", 0.0)
    for s in cfg["streams"]:
        if s["input"]:
            toks = np.load(s["file_in"]).reshape(-1).astype(np.uint64)
            cocotb.start_soon(_feed_stream(hdl, s, toks, gap))
        else:
            cocotb.start_soon(
                _drain_stream(hdl, s, stream_out[s["base"]], s["count"], gap)
            )

    hdl.start.value = 1
    await RisingEdge(hdl.clk)
    hdl.start.value = 0

    cycles = 0
    timeout = cfg["timeout"]
    while _i(hdl.done) != 1 and cycles < timeout:
        await RisingEdge(hdl.clk)
        cycles += 1
    for _ in range(cfg["settle_cycles"]):
        await RisingEdge(hdl.clk)

    for m, arr in zip(cfg["mems"], arrays):
        if m["file_out"]:
            np.save(m["file_out"], arr[: m["size"]].astype(np.uint64))
    for s in cfg["streams"]:
        if not s["input"]:
            np.save(s["file_out"], np.array(stream_out[s["base"]], dtype=np.uint64))
    # Scalar results: sample each output port now that `done` has settled and the
    # port holds its final value; the host decodes the bit pattern.
    results = [_i(getattr(hdl, n)) for n in cfg["result_ports"]]
    with open(cfg["results_out"], "w") as f:
        json.dump(results, f)
    with open(cfg["cycles_out"], "w") as f:
        f.write(str(cycles))
    assert cycles < timeout, f"cosim timed out after {timeout} cycles"
