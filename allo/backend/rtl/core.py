# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RTL backend handle.

``schedule.export("rtl", ...)`` returns an :class:`RTL`, which compiles the
kernel to hw/Verilog and exposes:

* ``schedule()`` -- the SDC scheduling result, for inspecting II / latency.
* ``csim(*args)`` -- functional golden, run on the CPU/LLVM-JIT path (untimed).
* ``cosim(*args)`` -- drive the emitted RTL under cocotb + a simulator, write the
  outputs back in place, and return the cycle count. No auto-compare: keep a
  golden copy from ``csim`` and assert against it.

One handle is one hardware configuration: the scheduling options are fixed at
export, and ``schedule()``, ``compile()`` and ``cosim()`` all describe the same
scheduled module. The kernel module is snapshotted at export -- modifying the
original kernel afterwards does not affect an existing handle.
"""

from __future__ import annotations

import json

from typing import Any, ParamSpec, TypeVar

from ..base import Backend, run_pipeline
from ..cpu import CPU
from ..._mlir.ir import Module
from ..._mlir._mlir_libs._allo import ir_ext
from ..._mlir.dialects.allo import emit_verilog, emit_datapath_to_hw
from .operator_library import OperatorLibrary
from .schedule import run_schedule, ScheduleResult
from .sim import shell
from ...lang.core import ShapedType
from ...lang.kernel import Kernel

P = ParamSpec("P")
R = TypeVar("R")

# Clock used for cosim when neither `freq_mhz` nor the library declares a target.
# It only scales simulated time, not the reported cycle count.
_DEFAULT_FREQ_MHZ = 300.0

# DCP normalization before emit: dcp-resolve-banking splits a statically-banked
# partitioned array into per-bank memrefs (reusing the scheduler's staticBank),
# then dcp-flatten-memref linearizes each per-bank address map, which the
# datapath emitter cannot lower on its own.
_NORMALIZE_PIPELINE = (
    "builtin.module(func.func(dcp-resolve-banking,dcp-flatten-memref))"
)


class RTL(Backend[P, R]):
    name = "rtl"

    def __init__(
        self,
        kernel: Kernel[P, R],
        *,
        device: str | None = None,
        library: OperatorLibrary | str | None = None,
        freq_mhz: float | None = None,
        simulator: str = "verilator",
        binding: str = "trivial",
        accumulators: int = 0,
        float_reassoc: bool = True,
        unroll_under_pipeline: bool = True,
        perfectize: bool = False,
    ):
        """Build an RTL handle for one hardware configuration.

        Args:
            device: selects a shipped operator library by name.
            library: an :class:`OperatorLibrary` or a path to a YAML library;
                defaults to the built-in one (``oplib/builtin.yaml``).
            freq_mhz: target frequency, driving both the SDC cycle time and the
                cosim clock. Defaults to the library's declared frequency.
            simulator: the engine cocotb drives for ``cosim``.
            binding: operator-sharing policy.
            accumulators: rotate float reductions across this many accumulators,
                dropping their II to ``ceil(latency / accumulators)`` (0 = off;
                at least the reduction operator's latency gives II=1).
            float_reassoc: rebalance float reduction chains into logarithmic
                trees. Not bit-exact, so pass ``False`` when exact floating-point
                semantics are required.
            unroll_under_pipeline: fully unroll the loops nested inside a
                pipelined loop, so the nest pipelines at one II (Vitis ``#pragma
                HLS pipeline`` semantics). ``False`` keeps them rolled and lets
                the scheduler pipeline the imperfect nest by overlap instead.
            perfectize: sink an imperfect nest's prologue/epilogue into the inner
                loop under a guard, fusing it into one pipeline. A QoR
                alternative -- the scheduler handles imperfect nests without it.
        """
        super().__init__(kernel)
        if library is None:
            library = OperatorLibrary.builtin(device or "builtin")
        self.library = library
        # One target frequency drives both the SDC cycle time and the cosim
        # clock; the library's declared target is the default. A library given
        # as a path declares its frequency to the scheduler directly.
        if freq_mhz is not None:
            self._cycle_time = 1000.0 / freq_mhz
        elif isinstance(library, OperatorLibrary):
            self._cycle_time = library.cycle_time()
        else:
            self._cycle_time = None
        self.freq_mhz = (
            1000.0 / self._cycle_time if self._cycle_time else _DEFAULT_FREQ_MHZ
        )
        self.simulator = simulator
        self.binding = binding
        self._sched_opts = {
            "accumulators": accumulators,
            "float_reassoc": float_reassoc,
            "unroll_under_pipeline": unroll_under_pipeline,
            "perfectize": perfectize,
        }
        self.arg_types = kernel.parse_argument_annotations()
        self.res_types = kernel.parse_return_annotation()
        # The three stage artifacts, each built once on first use: `self.module`
        # is the pristine snapshot, `_dcp_ir` the scheduled DCP module, `_hw_ir`
        # the emitted hw/comb/seq module.
        self._dcp_ir: Module | None = None
        self._schedule_result: ScheduleResult | None = None
        self._hw_ir: Module | None = None
        self._verilog: str | None = None
        self._cpu: CPU[P, R] | None = None
        # {module name -> port-interface manifest}, authored by the C++ emitter.
        self._interfaces: dict[str, Any] | None = None

    @property
    def top(self) -> str:
        """The DUT module name"""
        return self.kernel.func_name

    # -- scheduling -------------------------------------------------------

    def schedule(self) -> ScheduleResult:
        """Schedule the kernel and return the result: per-func regions with their
        II, latency and per-op start times. Computed once and reused by
        ``compile()``, so it always describes the RTL that ``cosim`` runs."""
        if self._schedule_result is None:
            # Schedule a copy: the driver reifies the schedule into the module in
            # place, and `self.module` stays the pristine snapshot.
            self._dcp_ir = ir_ext.clone_module(self.module)
            self._schedule_result = run_schedule(
                self.top,
                self._dcp_ir,
                library=self.library,
                cycle_time=self._cycle_time,
                **self._sched_opts,
            )
        return self._schedule_result

    @property
    def dcp(self) -> str:
        """The scheduled DCP MLIR module"""
        self.schedule()
        return str(self._dcp_ir)

    # -- emission ---------------------------------------------------------

    def compile(self) -> Module:
        """Compile the kernel to hw/comb/seq MLIR"""
        if self._hw_ir is None:
            # An array return has no meaning at a hardware port. This constrains
            # emission only -- such a kernel still schedules.
            if any(isinstance(t, ShapedType) for t in self.res_types):
                raise TypeError(
                    "RTL does not support returning arrays; use an out-parameter "
                    "instead"
                )
            self.schedule()
            # Normalize and emit on a copy, so `dcp` keeps reading the scheduled
            # module rather than this pipeline's lowered remains.
            work = ir_ext.clone_module(self._dcp_ir)
            run_pipeline(work, _NORMALIZE_PIPELINE)
            # The datapath emitter is a direct C++ call, not a pass, so its
            # `emitError` diagnostics do not flow through the PassManager ->
            # MLIRError path; capture them here so a failed emission raises the
            # diagnostic instead of returning None.
            diagnostics: list[str] = []
            handler = work.context.attach_diagnostic_handler(
                lambda d: bool(diagnostics.append(d.message)) or True
            )
            try:
                manifests = emit_datapath_to_hw(work, self.binding, self.top)
            finally:
                handler.detach()
            if manifests is None:
                raise RuntimeError(
                    "An error occurred during code generation process:\n"
                    + "\n".join(diagnostics)
                )
            self._interfaces = json.loads(manifests)
            self._hw_ir = work
        return self._hw_ir

    @property
    def mlir(self) -> str:
        """The emitted hw/comb/seq MLIR module"""
        return str(self.compile())

    @property
    def verilog(self) -> str:
        """The emitted (System)Verilog via CIRCT"""
        if self._verilog is None:
            verilog = emit_verilog(self.compile())
            assert verilog is not None, "RTL Verilog emission failed"
            self._verilog = verilog
        return self._verilog

    @property
    def interfaces(self) -> dict[str, Any]:
        """{module name -> port-interface manifest} for the emitted modules"""
        self.compile()
        return self._interfaces

    # -- verbs ------------------------------------------------------------

    def csim(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Functional golden: run the kernel on the CPU/LLVM-JIT path (in place)."""
        if self._cpu is None:
            self._cpu = CPU(self.kernel)
        return self._cpu(*args, **kwargs)

    def cosim(
        self,
        *args: Any,
        simulator: str | None = None,
        timeout: int = 40000,
        waves: bool = False,
        stall_prob: float = 0.0,
    ) -> shell.CosimResult:
        """Drive the emitted RTL under cocotb; write outputs back in place and
        return the cycle count. Does not compare -- keep a ``csim`` golden.

        An array output is written through an explicit out-parameter argument, so
        pass a pre-allocated buffer for each such argument; it is written in
        place. A scalar (``-> i32``) result stays an output port, sampled at
        ``done``.

        A stream (``Stream[...]``) argument is driven token-by-token over its
        FIFO handshake: pass a 1-D array of tokens for each input stream and a
        pre-allocated buffer for each output stream (drained in place).
        ``stall_prob`` (0..1) randomly starves inputs / back-pressures outputs to
        exercise the latency-insensitive shell -- the result must be unchanged.
        """
        self.compile()  # fills self._interfaces
        return shell.cosim(
            self.mlir,
            self.verilog,
            self.interfaces[self.top],
            self.top,
            self.arg_types,
            list(args),
            result_types=self.res_types,
            simulator=simulator or self.simulator,
            freq_mhz=self.freq_mhz,
            timeout=timeout,
            waves=waves,
            stall_prob=stall_prob,
        )

    def run(self, mode: str, *args: Any, **kwargs: Any) -> Any:
        if mode == "csim":
            return self.csim(*args, **kwargs)
        if mode == "cosim":
            return self.cosim(*args, **kwargs)
        raise NotImplementedError(f"RTL mode '{mode}' is not implemented")

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the kernel with CPU functional simulation"""
        return self.csim(*args, **kwargs)

    def scaffold_project(self, project: str | None = None, *, exist_ok: bool = True):
        raise NotImplementedError("RTL project scaffolding is not implemented")
