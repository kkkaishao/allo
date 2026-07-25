# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic, config-driven cocotb testbench.

This module runs inside the simulator's embedded Python (cocotb imports it by
name). It reads a JSON config (path in ``$ALLO_COSIM_CFG``) describing the module's
ports and the numpy-backed memories, drives clk/rst/start, services every memory
port, waits for ``done``, and writes the output arrays + cycle count back to files
for the host to read.

Each backing memory is modeled as a synchronous (registered) RAM at the access
latency its interface manifest declares, the device number the schedule was
solved against, not a fixed 1. At the base latency of 1, read data presented in
cycle k+1 reflects the address sampled at the edge ending cycle k, and a write
commits at that edge; a latency of L defers that by L-1 further edges. Reads
latch *before* writes apply at the same edge, so a read-modify-write on one array
sees the pre-write value (SV NBA semantics).
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


async def _serve_mem(hdl, clk, mem, readers, writers, size):
    """Service one backing array as a synchronous RAM at each port's DECLARED
    access latency, matching an ``always_ff @(posedge clk)`` model: the read
    address / write enable+addr+data are sampled *before* the edge (in ReadOnly,
    so they are the settled cycle values, like an NBA's RHS), then the read data
    is presented and the write committed at the edge.

    Each port's ``latency`` comes from the interface manifest, which carries the
    device memory model's number, the one the scheduler solved against. It is
    the driver's half of a contract the emitted RTL does not enforce: the module
    binds its read-data input with no delay elements, so it simply *expects* the
    datum ``latency`` cycles after the address. Serving every port in 1 cycle
    regardless (as this did) makes a URAM-bound argument, scheduled at 2, read a
    cycle early, and cosim would pass while the hardware was wrong.

    A latency of L presents/commits L-1 edges later than the base 1-cycle model,
    via a per-port pipeline of in-flight values.
    """
    rd = [
        (getattr(hdl, r["addr"]), getattr(hdl, r["data"]), int(r["latency"]))
        for r in readers
    ]
    wr = [
        (
            getattr(hdl, w["we"]),
            getattr(hdl, w["addr"]),
            getattr(hdl, w["data"]),
            int(w["latency"]),
        )
        for w in writers
    ]
    assert all(lat >= 1 for *_, lat in rd) and all(
        lat >= 1 for *_, lat in wr
    ), "a boundary port needs a >= 1 cycle access latency to be edge-triggered"
    # In-flight values for a port whose latency exceeds the 1-cycle base: each
    # holds the L-1 results/commits not yet due.
    rd_pipe = [[0] * (lat - 1) for *_, lat in rd]
    wr_pipe = [[None] * (lat - 1) for *_, lat in wr]
    clamp = lambda a: a if 0 <= a < size else 0
    while True:
        await ReadOnly()  # end of cycle: settled (pre-edge) values
        r_addr = [clamp(_i(addr)) for addr, _, _ in rd]
        w = [(_i(we), clamp(_i(addr)), _i(dat), lat) for we, addr, dat, lat in wr]
        await RisingEdge(clk)  # commit at the edge (NBA-like)
        # Reads resolve against pre-write memory (read-during-write returns the
        # old datum), so they are presented before the writes commit below.
        for k, (_, data, lat) in enumerate(rd):
            v = int(mem[r_addr[k]])
            if lat == 1:
                data.value = v
            else:  # due now: the value fetched lat-1 edges ago
                data.value = rd_pipe[k].pop(0)
                rd_pipe[k].append(v)
        for k, (we, addr, dat, lat) in enumerate(w):
            due = (we, addr, dat)
            if lat > 1:  # defer the commit by lat-1 edges
                due = wr_pipe[k].pop(0)
                wr_pipe[k].append((we, addr, dat))
            if due and due[0]:  # a pipe slot is None until the first commit
                mem[due[1]] = due[2]


async def _feed_stream(hdl, clk, s, tokens, gap=0.0):
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
            await RisingEdge(clk)
            continue
        data.value = int(tokens[i])
        valid.value = 1
        await ReadOnly()  # settled: is the DUT ready this cycle?
        fired = _i(ready) == 1
        await RisingEdge(clk)
        if fired:
            i += 1
    valid.value = 0
    data.value = 0


async def _drain_stream(hdl, clk, s, out, count, gap=0.0):
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
        await RisingEdge(clk)
    ready.value = 0


@cocotb.test()
async def cosim(hdl):
    with open(os.environ["ALLO_COSIM_CFG"]) as f:
        cfg = json.load(f)

    # The control ports come from the manifest like every other port, so this
    # harness holds no hardware name of its own.
    ctl = cfg["control"]
    clk = getattr(hdl, ctl["clk"])
    rst = getattr(hdl, ctl["rst"])
    start = getattr(hdl, ctl["start"])
    done = getattr(hdl, ctl["done"])

    cocotb.start_soon(Clock(clk, cfg["clock_ps"], unit="ps").start())
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

    rst.value = 1
    start.value = 0
    for _ in range(cfg["reset_cycles"]):
        await RisingEdge(clk)
    rst.value = 0
    await RisingEdge(clk)

    for m, arr in zip(cfg["mems"], arrays):
        cocotb.start_soon(
            _serve_mem(hdl, clk, arr, m["readers"], m["writers"], m["size"])
        )
    # Feed each input stream its token sequence; drain each output stream. A
    # non-zero `stream_gap` randomly starves inputs / back-pressures outputs to
    # exercise the latency-insensitive stall shell (the result must be identical
    # -- KPN determinism).
    gap = cfg.get("stream_gap", 0.0)
    for s in cfg["streams"]:
        if s["input"]:
            toks = np.load(s["file_in"]).reshape(-1).astype(np.uint64)
            cocotb.start_soon(_feed_stream(hdl, clk, s, toks, gap))
        else:
            cocotb.start_soon(
                _drain_stream(hdl, clk, s, stream_out[s["base"]], s["count"], gap)
            )

    start.value = 1
    await RisingEdge(clk)
    start.value = 0

    cycles = 0
    timeout = cfg["timeout"]
    while _i(done) != 1 and cycles < timeout:
        await RisingEdge(clk)
        cycles += 1
    for _ in range(cfg["settle_cycles"]):
        await RisingEdge(clk)

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
