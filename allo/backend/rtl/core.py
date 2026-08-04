# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RTL backend"""

from __future__ import annotations

import warnings

from typing import Any, ParamSpec, TypeVar

from ..base import Backend, run_pipeline
from ..cpu import CPU
from ..._mlir.ir import Module
from ..._mlir._mlir_libs._allo import ir_ext
from ..._mlir.dialects.allo import emit_verilog, emit_datapath_to_hw
from .device import (
    builtin_device,
    Device,
    inject_operators,
    inject_device,
    operator_descs,
)
from .interface import Interfaces
from .schedule import run_schedule, ScheduleResult
from .sim import shell
from ...lang.core import ShapedType
from ...lang.kernel import Kernel

P = ParamSpec("P")
R = TypeVar("R")

# The one DCP normalization before emit: it materializes the per-bank memrefs of
# a partitioned array. Addresses stay in element space until the emitter.
_NORMALIZE_PIPELINE = "builtin.module(dcp-resolve-banking)"


# pylint: disable-next=too-many-instance-attributes
class RTL(Backend[P, R]):
    name = "rtl"

    # the backend knobs are all keyword-only
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        kernel: Kernel[P, R],
        *,
        device: Device | None = None,
        freq_mhz: float | None = None,
        simulator: str = "verilator",
        binding: str = "trivial",
        accumulators: int = 0,
        float_reassoc: bool = True,
        unroll_under_pipeline: bool = True,
        perfectize: bool = False,
        scalarize_threshold: int = 16,
        scheduler: str = "heuristic",
        budget: float | None = None,
    ):
        """Build an RTL handle for one hardware configuration.

        Args:
            device: the hardware platform: storage primitives, native chaining
                delays, operator IPs and a default clock.
            freq_mhz: target frequency, overriding the device default. Drives
                both the SDC cycle time and the cosim clock.
            simulator: the engine cocotb drives for ``cosim``.
            binding: operator-sharing policy. ``"trivial"`` gives every
                operation its own unit; ``"greedy-share"`` folds every
                compatible pair the clock allows; ``"planned"`` builds the
                allocation the scheduler decided, which only an exact scheduler
                makes, so under the heuristic it is the trivial binding.
            accumulators: rotate float reductions across this many accumulators,
                dropping their II to ``ceil(latency / accumulators)`` (0 = off).
            float_reassoc: rebalance float reduction chains into logarithmic
                trees. Not bit-exact.
            unroll_under_pipeline: fully unroll the loops nested inside a
                pipelined loop, so the nest pipelines at one II (Vitis ``#pragma
                HLS pipeline`` semantics). ``False`` keeps them rolled and the
                directive is then not honored.
            perfectize: sink an imperfect nest's prologue/epilogue into the inner
                loop under a guard, fusing it into one pipeline. A QoR
                alternative; the scheduler handles imperfect nests without it.
            scalarize_threshold: keep arrays of at most this many elements in
                registers rather than a memory (0 = off).
            scheduler: the solver that settles the resource half of each
                scheduling problem. ``"heuristic"`` is the SDC simplex plus
                greedy placement, ``"exact"`` CP-SAT, and ``"exact-chaining"``
                CP-SAT deciding the chain breaks too (both need OR-Tools).
            budget: what one exact solve may spend, in the solver's
                deterministic time units; None takes the default. Only the
                largest regions ever reach it.
        """
        super().__init__(kernel)
        self._device = device if device is not None else builtin_device
        self.freq_mhz = (
            freq_mhz if freq_mhz is not None else self._device.default_freq_mhz
        )
        self._cycle_time = 1000.0 / self.freq_mhz
        self.simulator = simulator
        self.binding = binding
        self._sched_opts = {
            "accumulators": accumulators,
            "float_reassoc": float_reassoc,
            "unroll_under_pipeline": unroll_under_pipeline,
            "perfectize": perfectize,
            "scalarize_threshold": scalarize_threshold,
            "scheduler": scheduler,
            "budget": budget,
            # An allocation is only worth deciding where the emitter builds it:
            # the trivial binding keeps one unit per operation.
            "allocate": binding != "trivial",
        }
        self.arg_types = kernel.parse_argument_annotations()
        self.res_types = kernel.parse_return_annotation()
        # The stage artifacts, each built once on first use. `self.module` stays
        # the pristine snapshot.
        self._dcp_ir: Module | None = None
        self._schedule_result: ScheduleResult | None = None
        self._hw_ir: Module | None = None
        self._verilog: str | None = None
        self._cpu: CPU[P, R] | None = None
        self._interfaces: Interfaces | None = None

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
            # The schedule is reified in place, so it runs on a copy. Operator
            # and device timing is injected into that copy only, keeping the CPU
            # functional path clear of it.
            self._dcp_ir = ir_ext.clone_module(self.module)
            inject_operators(self._dcp_ir, self._device)
            inject_device(self._dcp_ir, self._device)
            self._schedule_result = run_schedule(
                self.top,
                self._dcp_ir,
                cycle_time=self._cycle_time,
                **self._sched_opts,
            )
        return self._schedule_result

    @property
    def dcp_module(self) -> Module:
        """The scheduled DCP module object."""
        self.schedule()
        assert self._dcp_ir is not None  # set by schedule()
        return self._dcp_ir

    @property
    def dcp(self) -> str:
        """The textual scheduled DCP MLIR module.
        NOTE: the textual form is not stable"""
        return str(self.dcp_module)

    # -- emission ---------------------------------------------------------

    def compile(self) -> Module:
        """Compile the kernel to hw/comb/seq MLIR"""
        if self._hw_ir is None:
            # An array return has no meaning at a hardware port. Emission only:
            # such a kernel still schedules.
            if any(isinstance(t, ShapedType) for t in self.res_types):
                raise TypeError(
                    "RTL does not support returning arrays; use an out-parameter "
                    "instead"
                )
            self.schedule()
            # Emit on a copy, so `dcp` keeps reading the scheduled module.
            work = ir_ext.clone_module(self._dcp_ir)
            run_pipeline(work, _NORMALIZE_PIPELINE)
            # The emitter is a direct call rather than a pass, so its diagnostics
            # do not reach the PassManager -> MLIRError path. Capture them here.
            diagnostics: list[str] = []
            handler = work.context.attach_diagnostic_handler(
                lambda d: bool(diagnostics.append(d.message)) or True
            )
            try:
                manifests = emit_datapath_to_hw(
                    work, self.binding, self.top, self._cycle_time
                )
            finally:
                handler.detach()
            if manifests is None:
                raise RuntimeError(
                    "An error occurred during code generation process:\n"
                    + "\n".join(diagnostics)
                )
            self._interfaces = Interfaces.from_json(manifests)
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
    def interfaces(self) -> Interfaces:
        """The emitted modules' port interfaces, keyed by RTL module name"""
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
        return the cycle count. Does not compare: keep a ``csim`` golden.

        An array output crosses as an out-parameter, so pass a pre-allocated
        buffer for each; a scalar result stays an output port sampled at
        ``done``. A ``Stream[...]`` argument is driven token-by-token over its
        FIFO handshake: a 1-D array of tokens for an input, a pre-allocated
        buffer for an output. ``stall_prob`` (0..1) randomly starves inputs and
        back-pressures outputs; the result must be unchanged.
        """
        self.compile()  # fills self._interfaces
        result = shell.cosim(
            self.verilog,
            self.interfaces,
            self.top,
            self.arg_types,
            list(args),
            result_types=self.res_types,
            operators=operator_descs(self._device.operators),
            simulator=simulator or self.simulator,
            freq_mhz=self.freq_mhz,
            timeout=timeout,
            waves=waves,
            stall_prob=stall_prob,
        )
        if stall_prob == 0:
            self._check_latency(result.cycles)
        return result

    def _check_latency(self, cycles: int) -> None:
        """Hold the latency model to the hardware: a kernel whose span is an
        exact static contract must run for exactly that many cycles. The only
        check in the compiler that compares a model against a measurement rather
        than against another model.
        """
        fn = self.schedule().func(self.top)
        # A bounded, indeterminate or concurrent kernel publishes a figure that
        # is deliberately not tight, so only an exact contract is held to a
        # measured count.
        if not fn.latency_is_exact:
            return
        modelled = fn.latency
        assert modelled is not None  # implied by latency_is_exact
        if modelled == cycles:
            return
        msg = (
            f"DEV-ONLY: latency model disagrees with the hardware for '{fn.name}': "
            f"declared latency = {modelled}, measured {cycles} cycles "
            f"(delta {cycles - modelled:+d}), which may indicate a bug in "
            "the compiler or the RTL."
        )
        warnings.warn(msg, stacklevel=2)

    # pylint: disable-next=arguments-differ
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
