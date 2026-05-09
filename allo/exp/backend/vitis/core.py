from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from ..base import Backend, text_hash, write_json_if_changed, write_text_if_changed
from ..utils import make_project_path
from .csim import (
    CSIM_MAKEFILE,
    PythonNativeCSimulator,
    _CSIM_UNSET,
    _generate_csim_makefile,
    _normalize_csim_make_vars,
)
from .report import VitisSynthReport
from .utils import (
    VitisTool,
    _INTERFACE_MODES,
    _add_extern_c_to_top,
    _apply_interface_pragmas,
    _detect_vitis_tool,
    _generate_kernel_header,
    _generate_run_tcl,
    _is_stream_type,
    _log_synth_failure,
    _log_synth_note,
    _normalize_interface_options,
    _resolve_settings64,
    _set_if_provided,
    _source_settings_env,
    _synth_log_path,
)
from ..._C import passes
from ..._C.ir import UnitAttr
from ..._C.passes import emit_vivado_hls
from ...lang.core import BufferType, ShapedType, TypeBase
from ...lang.kernel import Kernel
from ...logging import run_command, stage, terminate_on_error

VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]
FlowTarget = Literal["vitis", "vivado"]
AxiOffset = Literal["off", "direct", "slave"]
AxisRegisterMode = Literal["forward", "reverse", "both", "off"]
AxiliteStorageImpl = Literal["auto", "bram", "uram"]

DEFAULT_DEVICE = "u280"
DEFAULT_FREQ_MHZ = 300.0
DEFAULT_VITIS_SETTINGS = Path("/opt/xilinx/2025.2/Vitis/settings64.sh")
HLS_PREPARE_PIPELINE = (
    "builtin.module(func.func(convert-linalg-to-affine-loops),canonicalize,cse)"
)
VITIS_COMPILE_CACHE_VERSION = 1
VITIS_CSIM_CACHE_VERSION = 1
CSIM_CACHE_DIR_KEY_LENGTH = 24


@dataclass(frozen=True)
class CompiledArtifacts:
    kernel_cpp: str
    kernel_h: str
    top: str


@dataclass(frozen=True)
class InterfacePragma:
    mode: str
    options: Mapping[str, str | int | bool | None]


PART_NUMBERS = {
    # Embedded and Zynq.
    "ultra96v2": "xczu3eg-sbva484-1-i",
    "pynqz2": "xc7z020clg400-1",
    "zedboard": "xc7z020clg484-1",
    "zcu102": "xczu9eg-ffvb1156-2-e",
    "zcu104": "xczu7ev-ffvc1156-2-e",
    "zcu106": "xczu7ev-ffvc1156-2-e",
    "zcu111": "xczu28dr-ffvg1517-2MP-e-S",
    # Versal and Alveo.
    "vck190": "xcvc1902-vsva2197-2MP-e-S",
    "vhk158": "xcvh1582-vsva3697-2MP-e-S-es1",
    "u200": "xcu200-fsgd2104-2-e",
    "u250": "xcu250-figd2104-2L-e",
    "u280": "xcu280-fsvh2892-2L-e",
}


class Vitis(Backend):
    name = "vitis"

    @terminate_on_error
    def __init__(
        self,
        kernel: Kernel | None = None,
        vitis_home: str | None = None,
        project_path: str | None = None,
        *,
        settings64: str | os.PathLike[str] | None = None,
        device: str | None = DEFAULT_DEVICE,
        part: str | None = None,
        freq_mhz: float = DEFAULT_FREQ_MHZ,
        flow: FlowTarget = "vitis",
    ):
        super().__init__(kernel)
        self._settings64 = _resolve_settings64(
            settings64, vitis_home, DEFAULT_VITIS_SETTINGS
        )
        self._project_path = Path(project_path) if project_path else None
        self._vitis_home = Path(vitis_home) if vitis_home else None
        self._device = ""
        self._part = ""
        self._freq_mhz = DEFAULT_FREQ_MHZ
        self._flow: FlowTarget = "vitis"
        self._csim_make_vars: dict[str, str] = {}
        self._csim_tb_path: Path | None = None
        self._interface_pragmas: dict[int, dict[str, InterfacePragma]] = {}
        self._scaffolded_project_path: Path | None = None
        self.artifacts: CompiledArtifacts | None = None
        self.csimulator: PythonNativeCSimulator | None = None
        self.tool: VitisTool | None = None

        if part is not None:
            self.part = part
        else:
            self.device = device or DEFAULT_DEVICE
        self.freq_mhz = freq_mhz
        self.flow = flow

    @property
    def settings64(self) -> Path:
        return self._settings64

    @property
    def part(self) -> str:
        return self._part

    @part.setter
    def part(self, part: str) -> None:
        if not part:
            raise ValueError("Part number must be non-empty")
        self._part = part
        self._device = ""

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        part = PART_NUMBERS.get(device, "")
        if not part:
            raise ValueError(
                f"Unknown device {device}. Please set part number manually."
            )
        self._device = device
        self._part = part

    @property
    def freq_mhz(self) -> float:
        return self._freq_mhz

    @freq_mhz.setter
    def freq_mhz(self, freq: float) -> None:
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        self._freq_mhz = float(freq)

    @property
    def flow(self) -> FlowTarget:
        return self._flow

    @flow.setter
    def flow(self, flow: FlowTarget) -> None:
        if flow not in ("vitis", "vivado"):
            raise ValueError("Flow must be either 'vitis' or 'vivado'")
        self._flow = flow

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    def call_kernel(self, kernel: Kernel, *args, **kwargs) -> Any:
        backend = Vitis(
            kernel,
            os.fspath(self._vitis_home) if self._vitis_home is not None else None,
            settings64=self._settings64,
        )
        backend._csim_make_vars = self._csim_make_vars.copy()
        backend._csim_tb_path = self._csim_tb_path
        backend.tool = self.tool
        try:
            return backend.csim(*args, **kwargs)
        finally:
            self.tool = backend.tool

    def _is_scaffolded_project_path(self, project_path: Path) -> bool:
        return (
            self._scaffolded_project_path is not None
            and project_path.resolve() == self._scaffolded_project_path.resolve()
        )

    def set_csim_override(
        self,
        *,
        vitis_root: str = _CSIM_UNSET,
        cxx: str = _CSIM_UNSET,
        gcc_toolchain: str = _CSIM_UNSET,
        vitis_host_lib: str = _CSIM_UNSET,
        mathhls_lib: str = _CSIM_UNSET,
        fpo_lib: str = _CSIM_UNSET,
        kernel_cpp: str = _CSIM_UNSET,
        kernel_h: str = _CSIM_UNSET,
        out: str = _CSIM_UNSET,
        hls_includes: str = _CSIM_UNSET,
        hls_defines: str = _CSIM_UNSET,
        hls_cxxflags: str = _CSIM_UNSET,
        hls_ldflags: str = _CSIM_UNSET,
        extra_cxxflags: str = _CSIM_UNSET,
        extra_ldflags: str = _CSIM_UNSET,
        **kwargs: str,
    ):
        updates: dict[str, str] = {}
        for key, value in {
            "vitis_root": vitis_root,
            "cxx": cxx,
            "gcc_toolchain": gcc_toolchain,
            "vitis_host_lib": vitis_host_lib,
            "mathhls_lib": mathhls_lib,
            "fpo_lib": fpo_lib,
            "kernel_cpp": kernel_cpp,
            "kernel_h": kernel_h,
            "out": out,
            "hls_includes": hls_includes,
            "hls_defines": hls_defines,
            "hls_cxxflags": hls_cxxflags,
            "hls_ldflags": hls_ldflags,
            "extra_cxxflags": extra_cxxflags,
            "extra_ldflags": extra_ldflags,
        }.items():
            if value is not _CSIM_UNSET:
                updates[key] = value
        updates.update(kwargs)

        for key, value in _normalize_csim_make_vars(updates).items():
            if value is None:
                self._csim_make_vars.pop(key, None)
            else:
                self._csim_make_vars[key] = value
        self.csimulator = None

    def set_csim_tb(self, tb_path: str | os.PathLike[str] | None):
        self._csim_tb_path = Path(tb_path) if tb_path is not None else None
        self.csimulator = None

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
    ):
        arg_type = self._validate_interface_index(index, allow_return=False)
        if not isinstance(arg_type, BufferType):
            raise ValueError(
                "Vitis m_axi interface can only be set on buffer arguments"
            )

        options: dict[str, Any] = {}
        for key, value in {
            "bundle": bundle,
            "depth": depth,
            "offset": offset,
            "channel": channel,
            "latency": latency,
            "num_read_outstanding": num_read_outstanding,
            "num_write_outstanding": num_write_outstanding,
            "max_read_burst_length": max_read_burst_length,
            "max_write_burst_length": max_write_burst_length,
            "max_widen_bitwidth": max_widen_bitwidth,
            "alignment_byte_size": alignment_byte_size,
            "name": name,
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
        self._set_interface_pragma(index, "m_axi", options)

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
    ):
        raise NotImplementedError(
            "Vitis axis interface is not available until frontend Stream type "
            "support is enabled."
        )
        arg_type = self._validate_interface_index(index, allow_return=False)
        if not _is_stream_type(arg_type):
            raise ValueError("Vitis axis interface can only be set on stream arguments")

        options: dict[str, Any] = {}
        for key, value in {
            "register": register,
            "register_mode": register_mode,
            "depth": depth,
            "name": name,
            "bundle": bundle,
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
        self._set_interface_pragma(index, "axis", options)

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
    ):
        self._validate_interface_index(index, allow_return=True)
        if index == -1 and register is True:
            raise ValueError(
                "Vitis s_axilite return port does not support the register option"
            )

        options: dict[str, Any] = {}
        for key, value in {
            "bundle": bundle,
            "register": register,
            "clock": clock,
            "name": name,
            "offset": offset,
            "storage_impl": storage_impl,
        }.items():
            _set_if_provided(options, key, value)
        options.update(kwargs)
        self._set_interface_pragma(index, "s_axilite", options)

    @terminate_on_error
    def run(
        self,
        mode: VitisMode,
        *args,
        overwrite: bool = False,
        **kwargs,
    ) -> Any:
        if kwargs:
            raise TypeError(
                "Vitis backend only accepts positional kernel arguments and "
                "the overwrite keyword"
            )
        if mode == "csim":
            return self.csim(*args, overwrite=overwrite)
        if args:
            raise TypeError("Vitis csyn does not accept runtime arguments")
        if mode == "csyn":
            return self.synth(overwrite=overwrite)
        raise NotImplementedError(f"Vitis mode '{mode}' is not implemented yet")

    @terminate_on_error
    def csim(self, *args, overwrite: bool = False) -> Any:
        if self._csim_tb_path is not None:
            raise NotImplementedError(
                "External tb.cpp csim mode is not implemented yet"
            )
        project_path, cache_key = self._materialize_csim_cache(overwrite=overwrite)
        if overwrite:
            self.csimulator = None
            self._process_cache_pop("vitis.csim", cache_key)
        simulator = self._get_csimulator(project_path, cache_key)
        return simulator.run(*args, overwrite=overwrite)

    @terminate_on_error
    def synth(self, *, overwrite: bool = False) -> VitisSynthReport:
        """Generate an HLS csyn project and invoke Vitis HLS."""
        project_path = self._materialize_project(overwrite=overwrite)
        self._invoke_csyn(project_path)
        artifacts = self._ensure_compiled()
        rpt = VitisSynthReport(project_path=project_path, top=artifacts.top)
        rpt.render()
        return rpt

    @terminate_on_error
    def scaffold_project(
        self, project: str | None = None, *, overwrite: bool = False
    ) -> Path:
        return self._materialize_project(project, overwrite=overwrite)

    def _materialize_project(
        self, project: str | None = None, *, overwrite: bool = False
    ) -> Path:
        project_path = self._resolve_project_path(project, overwrite=overwrite)

        artifacts = self._ensure_compiled()
        with stage(f"Generating Vitis HLS Project to: {project_path.resolve()}"):
            write_text_if_changed(project_path / "kernel.cpp", artifacts.kernel_cpp)
            write_text_if_changed(project_path / "kernel.h", artifacts.kernel_h)
            write_text_if_changed(
                project_path / "run.tcl",
                _generate_run_tcl(artifacts.top, self.part, self.freq_mhz, self.flow),
            )

        self._project_path = project_path
        self._scaffolded_project_path = project_path
        return project_path

    def _materialize_csim_cache(self, *, overwrite: bool = False) -> tuple[Path, str]:
        artifacts = self._ensure_compiled()
        self._get_tool()
        vitis_root = self._get_vitis_root()
        makefile = _generate_csim_makefile(vitis_root)
        payload = self._csim_cache_payload(artifacts, makefile, vitis_root)
        cache_key = self._cache_key(payload)
        project_path = self._cache_dir(
            "vitis",
            "csim",
            cache_key[:CSIM_CACHE_DIR_KEY_LENGTH],
        )
        cache_files = (
            project_path / "kernel.cpp",
            project_path / "kernel.h",
            project_path / CSIM_MAKEFILE,
            project_path / "cache.json",
        )

        if not overwrite and all(path.exists() for path in cache_files):
            return project_path, cache_key

        with stage(f"Generating Vitis C Simulation Cache to: {project_path}"):
            project_path.mkdir(parents=True, exist_ok=True)
            write_text_if_changed(project_path / "kernel.cpp", artifacts.kernel_cpp)
            write_text_if_changed(project_path / "kernel.h", artifacts.kernel_h)
            write_text_if_changed(project_path / CSIM_MAKEFILE, makefile)
            write_json_if_changed(
                project_path / "cache.json",
                {
                    "key": cache_key,
                    "payload": payload,
                    "overwrite": overwrite,
                },
            )

        return project_path, cache_key

    def _resolve_project_path(
        self, project: str | None = None, *, overwrite: bool = False
    ) -> Path:
        if project is None and self._project_path is None:
            return make_project_path(
                None, f"allo-vitis-prj-{self.kernel.func_name}", overwrite
            )

        project_path = Path(project) if project is not None else self._project_path
        assert project_path is not None
        overwrite = overwrite or self._is_scaffolded_project_path(project_path)
        if project_path.exists() and any(project_path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Project path {project_path} already exists and is not empty. "
                "Use overwrite=True to overwrite."
            )
        project_path.mkdir(parents=True, exist_ok=True)
        return project_path

    def _ensure_compiled(self) -> CompiledArtifacts:
        if self.artifacts is None:
            self.artifacts = self.compile()
        return self.artifacts

    def _release_working_module(self) -> None:
        self.module = None
        self._module_owner = None

    def _invalidate_compiled_artifacts(self) -> None:
        self.artifacts = None
        self.csimulator = None

    @terminate_on_error
    def _validate_interface_index(
        self, index: int, *, allow_return: bool
    ) -> TypeBase | None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Vitis interface index must be an integer")

        arg_types = self.kernel.parse_argument_annotations()
        if index == -1:
            if not allow_return:
                raise ValueError("This Vitis interface cannot be set on return value")
            ret_types = self.kernel.parse_return_annotation()
            if len(ret_types) > 1:
                raise ValueError(
                    "Vitis backend supports at most one return value for "
                    "interface configuration"
                )
            if ret_types and isinstance(ret_types[0], ShapedType):
                raise ValueError(
                    "Vitis backend does not support shaped return interfaces. "
                    "Pass output buffers as explicit arguments instead."
                )
            return None

        if index < 0 or index >= len(arg_types):
            raise IndexError(
                f"Vitis interface index {index} is out of range for "
                f"{len(arg_types)} input arguments"
            )
        return arg_types[index]

    @terminate_on_error
    def _set_interface_pragma(
        self,
        index: int,
        mode: str,
        options: Mapping[str, Any],
    ) -> None:
        if mode not in _INTERFACE_MODES:
            raise ValueError(f"Unsupported Vitis HLS interface mode '{mode}'")
        self._interface_pragmas.setdefault(index, {})[mode] = InterfacePragma(
            mode=mode,
            options=_normalize_interface_options(mode, options),
        )
        self._invalidate_compiled_artifacts()

    def _validate_interface_pragmas(self) -> None:
        for index, pragmas in self._interface_pragmas.items():
            arg_type = self._validate_interface_index(
                index, allow_return="s_axilite" in pragmas
            )
            if "m_axi" in pragmas and not isinstance(arg_type, BufferType):
                raise ValueError(
                    "Vitis m_axi interface can only be set on buffer arguments"
                )
            if "axis" in pragmas and (
                arg_type is None or not _is_stream_type(arg_type)
            ):
                raise ValueError(
                    "Vitis axis interface can only be set on stream arguments"
                )

    def _get_tool(self) -> VitisTool:
        if self.tool is None:
            self.tool = _detect_vitis_tool(self._settings64)
            if self.tool is None:
                raise RuntimeError(
                    "Vitis HLS tool not found. Source "
                    f"{self._settings64} or pass settings64=... to the Vitis backend."
                )
        return self.tool

    def _get_vitis_env(self) -> dict[str, str]:
        if self.tool is not None:
            return dict(self.tool.env)
        with stage("Load Vitis environment"):
            sourced_env = _source_settings_env(self._settings64)
            if sourced_env is not None:
                return sourced_env
        return dict(os.environ)

    def _get_vitis_root(self) -> Path:
        if self._vitis_home is not None:
            return self._vitis_home
        if self._settings64.name == "settings64.sh":
            return self._settings64.parent
        if self.tool is not None and self.tool.executable.parent.name == "bin":
            return self.tool.executable.parent.parent
        return DEFAULT_VITIS_SETTINGS.parent

    def _tool_cache_payload(self) -> dict[str, str]:
        tool = self._get_tool()
        return {
            "name": tool.name,
            "executable": os.fspath(tool.executable),
            "version": tool.version,
            "settings64": os.fspath(self._settings64),
        }

    def _interface_cache_payload(self) -> dict[int, dict[str, Mapping[str, Any]]]:
        return {
            index: {
                mode: pragma.options
                for mode, pragma in sorted(pragmas.items(), key=lambda item: item[0])
            }
            for index, pragmas in sorted(self._interface_pragmas.items())
        }

    def _compile_cache_key(self) -> str:
        return self._cache_key(
            {
                "backend": self.name,
                "phase": "hls-codegen",
                "version": VITIS_COMPILE_CACHE_VERSION,
                "hls_prepare_pipeline": HLS_PREPARE_PIPELINE,
                "interface_pragmas": self._interface_cache_payload(),
            }
        )

    def _csim_cache_payload(
        self, artifacts: CompiledArtifacts, makefile: str, vitis_root: Path
    ) -> dict[str, Any]:
        return {
            "backend": self.name,
            "phase": "python-native-csim",
            "version": VITIS_CSIM_CACHE_VERSION,
            "top": artifacts.top,
            "kernel_cpp_sha256": text_hash(artifacts.kernel_cpp),
            "kernel_h_sha256": text_hash(artifacts.kernel_h),
            "makefile_sha256": text_hash(makefile),
            "arg_types": [str(arg) for arg in self.kernel.parse_argument_annotations()],
            "res_types": [str(res) for res in self.kernel.parse_return_annotation()],
            "make_vars": self._csim_make_vars,
            "vitis_root": os.fspath(vitis_root),
            "tool": self._tool_cache_payload(),
        }

    def _get_csimulator(
        self, project_path: Path, cache_key: str
    ) -> PythonNativeCSimulator:
        if self.csimulator is not None and self.csimulator.project_path == project_path:
            return self.csimulator
        cached = self._process_cache_get("vitis.csim", cache_key)
        if cached is not None:
            self.csimulator = cached
            return self.csimulator
        self.csimulator = PythonNativeCSimulator(
            top=self.kernel.func_name,
            project_path=project_path,
            vitis_root=self._get_vitis_root(),
            env=self._get_vitis_env(),
            arg_types=self.kernel.parse_argument_annotations(),
            res_types=self.kernel.parse_return_annotation(),
            make_vars=self._csim_make_vars,
        )
        self._process_cache_set("vitis.csim", cache_key, self.csimulator)
        return self.csimulator

    def _invoke_csyn(self, project_path: Path) -> None:
        tool = self._get_tool()
        log_path = _synth_log_path(project_path)
        if tool.name == "vitis-run":
            cmd = [
                os.fspath(tool.executable),
                "--mode",
                "hls",
                "--tcl",
                "--work_dir",
                ".",
                "run.tcl",
            ]
        else:
            cmd = [os.fspath(tool.executable), "-f", "run.tcl"]

        with stage(
            "Running Vitis HLS synthesis",
            on_error=lambda error: _log_synth_failure(log_path, error),
            on_exit=lambda: _log_synth_note(log_path),
        ):
            run_command(cmd, cwd=project_path, env=dict(tool.env))

    def _validate_top_abi(self) -> None:
        ret_types = self.kernel.parse_return_annotation()
        if len(ret_types) > 1:
            raise ValueError(
                "Vitis backend only supports void or a single scalar return from "
                "the top kernel. Pass output buffers as explicit arguments instead."
            )
        if ret_types and isinstance(ret_types[0], ShapedType):
            raise ValueError(
                "Vitis backend does not support returning shaped values from the "
                "top kernel. Pass the output buffer as an explicit argument instead."
            )

    @terminate_on_error
    def compile(self) -> CompiledArtifacts:
        if self.kernel.func_name == "kernel":
            raise ValueError(
                "'kernel' is a reserved name for Vitis HLS. Please rename your kernel function."
            )
        self._validate_top_abi()
        self._validate_interface_pragmas()

        cache_key = self._compile_cache_key()
        cached = self._process_cache_get("vitis.compile", cache_key)
        if cached is not None:
            self.artifacts = cached
            return cached

        with stage("Compiling Vitis HLS Kernels"):
            module = self._get_working_module()
            top_fn = module.lookup_func(self.kernel.func_name)
            if top_fn is None:
                raise RuntimeError(
                    f"Kernel function {self.kernel.func_name} not found in the module"
                )
            top_fn.set_attr("top", UnitAttr.get(module.get_context()))

            passes.run(HLS_PREPARE_PIPELINE, module.get_operation())
            hls_code = emit_vivado_hls(module)
            if hls_code is None:
                raise RuntimeError("Failed to emit Vitis HLS code")
            hls_code = _add_extern_c_to_top(hls_code, self.kernel.func_name)
            hls_code = _apply_interface_pragmas(
                hls_code, self.kernel.func_name, self._interface_pragmas
            )
            top_fn = None
            module = None
            self._release_working_module()

            artifacts = CompiledArtifacts(
                kernel_cpp=hls_code,
                kernel_h=_generate_kernel_header(hls_code, self.kernel.func_name),
                top=self.kernel.func_name,
            )
            self.artifacts = artifacts
            self._process_cache_set("vitis.compile", cache_key, artifacts)
            return artifacts
