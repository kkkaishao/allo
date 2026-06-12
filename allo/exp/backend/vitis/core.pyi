# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the Vitis HLS backend."""
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, Literal

P = ParamSpec("P")
R = TypeVar("R")

AxiOffset = Literal["off", "direct", "slave"]
AxisRegisterMode = Literal["forward", "reverse", "both", "off"]
AxiliteStorageImpl = Literal["auto", "bram", "uram"]
VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]

class VitisSynthReport:
    @property
    def xml_path(self) -> Path: ...
    def __getattr__(self, name: str) -> Any: ...

class Vitis(Generic[P, R]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @property
    def hls_code(self) -> str: ...
    def run(self, mode: VitisMode, *args: Any, exist_ok: bool = ...) -> Any: ...
    def synth(self, *, exist_ok: bool = ...) -> VitisSynthReport: ...
    def precheck(
        self, mode: VitisMode, project: str | None = ..., *, exist_ok: bool = ...
    ) -> Path: ...
    def scaffold_project(
        self, project: str | None = ..., *, exist_ok: bool = ...
    ) -> Path: ...
    def set_axi(
        self,
        index: int,
        *,
        bundle: str | None = None,
        depth: int | None = None,
        offset: AxiOffset | None = None,
        channel: str | None = None,
        latency: int | None = None,
        num_read_outstanding: int | None = None,
        num_write_outstanding: int | None = None,
        max_read_burst_length: int | None = None,
        max_write_burst_length: int | None = None,
        max_widen_bitwidth: int | None = None,
        alignment_byte_size: int | None = None,
        name: str | None = None,
        **kwargs: str | int | bool | None,
    ) -> None: ...
    def set_axis(
        self,
        index: int,
        *,
        register: bool | None = None,
        register_mode: AxisRegisterMode | None = None,
        depth: int | None = None,
        name: str | None = None,
        bundle: str | None = None,
        **kwargs: str | int | bool | None,
    ) -> None: ...
    def set_axilite(
        self,
        index: int,
        *,
        bundle: str | None = None,
        register: bool | None = None,
        clock: str | None = None,
        name: str | None = None,
        offset: str | None = None,
        storage_impl: AxiliteStorageImpl | None = None,
        **kwargs: str | int | bool | None,
    ) -> None: ...

def is_vitis_available() -> bool: ...
